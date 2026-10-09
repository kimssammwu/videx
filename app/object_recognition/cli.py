"""Command-line runner for local files, JPEG sequences, and videos."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .api import (
    DEFAULT_OPENAI_VISION_MODEL,
    HttpVisionClient,
    MockVisionClient,
    VisionApiError,
    VisionClient,
)
from .models import RecognitionConfig, RecognitionState
from .recognizer import ObjectRecognizer
from .sources import packet_from_file, packets_from_files, packets_from_video
from .usb_source import UsbCameraSource


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VIDEX 물체 인식 Python 검증 프로그램")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", help="시간순 JPEG 이미지 목록")
    source.add_argument("--video", help="테스트 동영상 경로")
    source.add_argument("--usb-port", help="ESP32-S3 native USB CDC 포트 (예: COM5)")
    source.add_argument(
        "--manual-pair",
        nargs=2,
        metavar=("FRONT", "BACK"),
        help="앞면/뒷면 JPEG를 수동 확정하여 전체 파이프라인 검증",
    )
    parser.add_argument(
        "--camera-id",
        default="local",
        help="입력 카메라 ID; USB에서는 0 또는 1 (기본값: 0)",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--usb-max-fps", type=float, default=10.0)
    parser.add_argument(
        "--usb-retries",
        type=int,
        default=1,
        help="USB 자동 촬영 timeout 재시도 횟수 (기본값: 1)",
    )
    parser.add_argument("--mock-product", default="테스트 제품 1L", help="Mock 모드 제품명")
    parser.add_argument(
        "--live",
        action="store_true",
        help="VIDEX_VISION_API_KEY를 사용해 실제 OpenAI API 호출",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_VISION_MODEL,
        help=f"--live에서 사용할 OpenAI 모델 (기본값: {DEFAULT_OPENAI_VISION_MODEL})",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stable-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _build_vision_client(args: argparse.Namespace) -> VisionClient:
    if not args.live:
        return MockVisionClient(args.mock_product)
    client = HttpVisionClient(model=args.model)
    client.validate_configuration()
    return client


def _usb_camera_id(args: argparse.Namespace) -> int:
    value = "0" if args.camera_id == "local" else str(args.camera_id)
    if value not in ("0", "1"):
        raise ValueError("USB --camera-id는 0 또는 1이어야 합니다.")
    return int(value)


def _poll_usb_command() -> str | None:
    """Read one Windows console key without blocking the serial loop."""

    if sys.platform != "win32" or not sys.stdin.isatty():
        return None
    import msvcrt

    if not msvcrt.kbhit():
        return None
    return msvcrt.getwch().lower()


def _run_usb(args: argparse.Namespace, recognizer: ObjectRecognizer) -> None:
    camera_id = _usb_camera_id(args)
    if args.usb_retries < 0:
        raise ValueError("--usb-retries는 0 이상이어야 합니다.")
    retries = 0
    manual_capture_pending = False
    print("USB 조작: C=현재 프레임 수동 확정, R=오류 후 재시도, Q=종료")
    with UsbCameraSource(
        args.usb_port,
        camera_id,
        baudrate=args.baudrate,
        max_fps=args.usb_max_fps,
    ) as source:
        while recognizer.state is not RecognitionState.COMPLETED:
            command = _poll_usb_command()
            if command == "q":
                raise KeyboardInterrupt
            if recognizer.state is RecognitionState.ERROR:
                if recognizer.merged_path is not None:
                    break
                if retries < args.usb_retries:
                    retries += 1
                    recognizer.retry_recognition()
                    logging.warning(
                        "USB 촬영 timeout: 자동 재시도 %d/%d",
                        retries,
                        args.usb_retries,
                    )
                    continue
                if command == "r":
                    recognizer.retry_recognition()
                    logging.info("사용자 요청으로 촬영을 재시도합니다.")
                    continue
                if sys.platform == "win32" and sys.stdin.isatty():
                    time.sleep(0.05)
                    continue
                break
            if command == "c":
                manual_capture_pending = True
                logging.info("다음 유효 프레임을 수동 촬영 확정합니다.")
            elif command == "r":
                logging.info("현재 상태에서는 재시도가 필요하지 않습니다.")

            packets = source.poll()
            if not packets:
                recognizer.tick()
                continue

            for packet in packets:
                metrics = recognizer.process_frame(
                    packet,
                    manual_capture=manual_capture_pending,
                )
                if manual_capture_pending:
                    logging.info(
                        "수동 확정 결과 selected=%s reason=%s",
                        metrics.selected,
                        metrics.reason,
                    )
                    manual_capture_pending = False
                if recognizer.state in (RecognitionState.COMPLETED, RecognitionState.ERROR):
                    break
        logging.info("USB receive stats=%s", source.stats)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        vision_client = _build_vision_client(args)
    except VisionApiError as exc:
        logging.error("실제 API를 시작할 수 없습니다: %s", exc)
        return 2
    config = RecognitionConfig(
        output_dir=args.output_dir,
        stable_duration=args.stable_seconds,
        phase_timeout=args.timeout,
    )
    recognizer = ObjectRecognizer(config, vision_client)
    try:
        recognition_camera_id = str(_usb_camera_id(args)) if args.usb_port else args.camera_id
    except ValueError as exc:
        logging.error("입력 설정 오류: %s", exc)
        return 2
    recognizer.start_recognition(recognition_camera_id)
    try:
        if args.usb_port:
            _run_usb(args, recognizer)
        elif args.manual_pair:
            front, back = args.manual_pair
            recognizer.process_frame(
                packet_from_file(front, args.camera_id, "manual-front"), manual_capture=True
            )
            recognizer.process_frame(
                packet_from_file(back, args.camera_id, "manual-back"), manual_capture=True
            )
        else:
            packets = (
                packets_from_files(args.images, args.camera_id, args.fps)
                if args.images
                else packets_from_video(args.video, args.camera_id)
            )
            for packet in packets:
                recognizer.process_frame(packet)
                if recognizer.state in (RecognitionState.COMPLETED, RecognitionState.ERROR):
                    break
            if recognizer.state not in (RecognitionState.COMPLETED, RecognitionState.ERROR):
                # A finite file/video ending before capture completion is an input
                # failure; live transports should instead keep calling tick().
                recognizer.tick(recognizer.phase_started_at + args.timeout + 0.001)
    except KeyboardInterrupt:
        recognizer.stop_recognition()
        logging.info("사용자 요청으로 종료했습니다. USB 포트를 정리했습니다.")
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("입력 처리 실패: %s", exc)
        return 2
    state = recognizer.get_recognition_state()
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if recognizer.state is RecognitionState.COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())

"""Command-line runner for local files, JPEG sequences, and videos."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

from .api import (
    DEFAULT_OPENAI_VISION_MODEL,
    HttpVisionClient,
    MockVisionClient,
    VisionApiError,
    VisionClient,
)
from .models import RecognitionConfig, RecognitionState
from .preview import LEFT_CAMERA_ID, LeftCameraViewer, RecognitionPreviewWorker
from .recognizer import ObjectRecognizer
from .sources import packet_from_file, packets_from_files, packets_from_video
from .usb_source import DEFAULT_RESOLUTION_MODE, UsbCameraSource


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VIDEX 물체 인식 Python 검증 프로그램")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", help="시간순 JPEG 이미지 목록")
    source.add_argument("--video", help="테스트 동영상 경로")
    source.add_argument("--usb-port", help="ESP32-S3 native USB CDC 포트 (예: COM3)")
    source.add_argument(
        "--manual-pair",
        nargs=2,
        metavar=("FRONT", "BACK"),
        help="앞면/뒷면 JPEG를 수동 확정하여 전체 파이프라인 검증",
    )
    parser.add_argument(
        "--camera-id",
        default=str(LEFT_CAMERA_ID),
        help="입력 카메라 ID; USB 상품 인식은 LEFT인 0만 지원 (기본값: 0)",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--usb-max-fps", type=float, default=10.0)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="LEFT 실시간 영상과 촬영 결과를 한 화면에 표시",
    )
    parser.add_argument(
        "--dual-camera",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.040,
        help="작은 움직임 허용 임계값 (초기 실험값: 0.040)",
    )
    parser.add_argument(
        "--motion-reset-threshold",
        type=float,
        default=0.100,
        help="즉시 안정화를 초기화할 큰 움직임 임계값 (기본값: 0.100)",
    )
    parser.add_argument(
        "--stable-ratio",
        type=float,
        default=0.80,
        help="시간 기반 안정 구간 최소 비율 (기본값: 0.80)",
    )
    parser.add_argument(
        "--stability-exit-ratio",
        type=float,
        default=0.60,
        help="안정 상태 해제 비율(히스테리시스, 기본값: 0.60)",
    )
    parser.add_argument(
        "--stability-window",
        type=float,
        default=1.0,
        help="움직임 이력 시간 창(초, 기본값: 1.0)",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=float,
        default=0.35,
        help="안정 이력을 초기화할 최대 프레임 간격(초, 기본값: 0.35)",
    )
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
    value = str(args.camera_id)
    if value != str(LEFT_CAMERA_ID):
        raise ValueError("USB 상품 인식은 LEFT 카메라인 --camera-id 0만 지원합니다.")
    return LEFT_CAMERA_ID


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
        resolution_mode=DEFAULT_RESOLUTION_MODE,
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


def _run_preview_usb(args: argparse.Namespace, recognizer: ObjectRecognizer) -> dict[str, object]:
    """Keep a LEFT-only preview responsive while recognition runs in a worker."""

    if not args.usb_port:
        raise ValueError("--preview는 --usb-port와 함께 사용해야 합니다.")
    _usb_camera_id(args)
    viewer = LeftCameraViewer(recognizer.config.roi_fraction)
    worker = RecognitionPreviewWorker(recognizer)
    latest_packet = None
    recognition_resolution: tuple[int, int] | None = None
    print(
        "LEFT 640x480 고정 미리보기 조작: C=현재 프레임 수동 촬영, "
        "R=처음부터 재시도, Q/ESC=종료"
    )
    try:
        viewer.open()
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI를 열 수 없습니다. opencv-python 패키지와 데스크톱 환경을 확인하세요."
        ) from exc
    try:
        try:
            worker.start()
            with UsbCameraSource(
                args.usb_port,
                LEFT_CAMERA_ID,
                baudrate=args.baudrate,
                max_fps=args.usb_max_fps,
                resolution_mode=DEFAULT_RESOLUTION_MODE,
            ) as source:
                viewer.update_usb_status(
                    connected=True,
                    resolution_status=source.resolution_request_status,
                    resolution_support_status=source.selected_resolution_status,
                    requested_resolution_mode=source.resolution_mode,
                )
                last_resolution_observation = source.selected_resolution_status
                while True:
                    packets = source.poll()
                    frame_info = source.last_frame_info_by_camera.get(LEFT_CAMERA_ID)
                    if frame_info is not None:
                        viewer.update_usb_status(
                            connected=True,
                            camera_id=frame_info.camera_id,
                            frame_id=frame_info.frame_id,
                            width=frame_info.width,
                            height=frame_info.height,
                            received_at=frame_info.received_at,
                            paused=frame_info.paused,
                            resolution_status=source.resolution_request_status,
                            resolution_support_status=source.selected_resolution_status,
                            requested_resolution_mode=source.resolution_mode,
                        )
                    if source.selected_resolution_status != last_resolution_observation:
                        if source.selected_resolution_status == "selected_camera_paused":
                            viewer.set_resolution_notice(
                                "LEFT paused unexpectedly; check Proxy/camera firmware"
                            )
                        elif source.selected_resolution_status == "actual_mismatch":
                            actual = source.selected_actual_resolution
                            viewer.set_resolution_notice(
                                f"LEFT actual {actual[0]}x{actual[1]} does not match requested mode; firmware ignored/unsupported"
                                if actual is not None
                                else "LEFT resolution does not match the request"
                            )
                        elif source.selected_resolution_status == "actual_matches_request":
                            actual = source.selected_actual_resolution
                            if actual is not None:
                                viewer.set_resolution_notice(
                                    f"LEFT actual {actual[0]}x{actual[1]} matches requested mode"
                                )
                        last_resolution_observation = source.selected_resolution_status
                    for packet in packets:
                        viewer.update_frame(packet)
                        packet_resolution = (packet.data.shape[1], packet.data.shape[0])
                        if (
                            recognition_resolution is not None
                            and packet_resolution != recognition_resolution
                        ):
                            if not worker.reset():
                                latest_packet = None
                                viewer.set_resolution_notice(
                                    "LEFT resolution changed; waiting to reset capture safely"
                                )
                                continue
                            latest_packet = None
                            viewer.set_resolution_notice(
                                f"LEFT actual resolution changed to {packet_resolution[0]}x{packet_resolution[1]}; "
                                "capture restarted from FRONT"
                            )
                            logging.info(
                                "LEFT 실제 해상도가 %dx%d로 변경되어 안정화 및 촬영 세션을 초기화했습니다.",
                                packet_resolution[0],
                                packet_resolution[1],
                            )
                        recognition_resolution = packet_resolution
                        latest_packet = packet
                        worker.submit(packet)
                    canvas = viewer.render(
                        recognizer,
                        worker.latest_metrics,
                        worker_busy=worker.busy,
                        worker_error=worker.latest_error,
                    )
                    gui_command = viewer.show(canvas)
                    command = gui_command or _poll_usb_command()
                    if command == "q":
                        break
                    if command == "c":
                        if latest_packet is None:
                            logging.warning("수동 촬영할 LEFT 프레임이 아직 없습니다.")
                        elif recognizer.state in (
                            RecognitionState.COMPLETED,
                            RecognitionState.ERROR,
                        ):
                            logging.warning("현재 상태에서는 수동 촬영할 수 없습니다. R로 다시 시작하세요.")
                        else:
                            worker.submit(latest_packet, manual_capture=True)
                            logging.info("LEFT 현재 프레임을 수동 촬영 요청했습니다.")
                    elif command == "r":
                        if not worker.reset():
                            logging.warning("프레임 또는 API 처리 중에는 재시도할 수 없습니다.")
                        else:
                            logging.info("앞면 촬영부터 다시 시작합니다.")
                logging.info("USB receive stats=%s", source.stats)
        except cv2.error as exc:
            raise RuntimeError("OpenCV 미리보기 창을 갱신할 수 없습니다.") from exc
    finally:
        worker.stop()
        viewer.close()
    return recognizer.get_recognition_state()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if (args.preview or args.dual_camera) and not args.usb_port:
            raise ValueError("--preview는 --usb-port와 함께 사용해야 합니다.")
        if args.dual_camera:
            logging.warning("--dual-camera는 호환용 별칭입니다. LEFT 전용 --preview로 실행합니다.")
        vision_client = _build_vision_client(args)
    except (ValueError, VisionApiError) as exc:
        logging.error("실행을 시작할 수 없습니다: %s", exc)
        return 2
    try:
        config = RecognitionConfig(
            output_dir=args.output_dir,
            stable_duration=args.stable_seconds,
            motion_threshold=args.motion_threshold,
            motion_reset_threshold=args.motion_reset_threshold,
            stable_ratio=args.stable_ratio,
            stability_exit_ratio=args.stability_exit_ratio,
            stability_window=args.stability_window,
            max_frame_gap=args.max_frame_gap,
            phase_timeout=args.timeout,
        )
    except ValueError as exc:
        logging.error("안정성 설정 오류: %s", exc)
        return 2
    recognizer = ObjectRecognizer(config, vision_client)
    try:
        recognition_camera_id = str(_usb_camera_id(args)) if args.usb_port else args.camera_id
    except ValueError as exc:
        logging.error("입력 설정 오류: %s", exc)
        return 2
    recognizer.start_recognition(recognition_camera_id)
    preview_state: dict[str, object] | None = None
    try:
        if args.preview or args.dual_camera:
            preview_state = _run_preview_usb(args, recognizer)
        elif args.usb_port:
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
        logging.info("사용자 요청으로 종료했습니다. USB 포트를 정리했습니다.")
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("입력 처리 실패: %s", exc)
        return 2
    state = preview_state or recognizer.get_recognition_state()
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if recognizer.state is RecognitionState.COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())

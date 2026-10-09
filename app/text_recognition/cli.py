"""Subcommand CLI for Preview, capture, and one-shot text recognition."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .camera import FileImageSource, UsbLeftCamera, UsbPreviewCamera
from .config import (
    CAMERA_PORT_ENV,
    DEFAULT_CAPTURE_DIR,
    DEFAULT_VISION_ENDPOINT,
    DEFAULT_VISION_MODEL,
    VISION_API_KEY_ENV,
    VISION_ENDPOINT_ENV,
    VISION_MODEL_ENV,
)
from .errors import TextRecognitionError
from .preview import CaptureStore, TextRecognitionPreview
from .service import TextRecognitionService
from .vision import HttpVisionClient, MockVisionClient, VisionClient


COMMANDS = {"preview", "capture", "recognize", "recognize-image"}


def _add_camera_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--usb-port", help=f"LEFT CAM2 USB 포트 (또는 {CAMERA_PORT_ENV})"
    )
    parser.add_argument("--baudrate", type=int, default=115200)


def _add_api_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vision-timeout", type=float, default=30.0)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get(VISION_ENDPOINT_ENV, DEFAULT_VISION_ENDPOINT),
    )
    parser.add_argument(
        "--model", default=os.environ.get(VISION_MODEL_ENV, DEFAULT_VISION_MODEL)
    )
    parser.add_argument("--api-key-env", default=VISION_API_KEY_ENV)
    parser.add_argument(
        "--mock-text",
        help="실제 API 대신 지정한 글씨를 반환하는 네트워크 없는 검증 모드",
    )


def _add_capture_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
        help=f"촬영 JPEG 저장 폴더 (기본값: {DEFAULT_CAPTURE_DIR})",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VIDEX 범용 글씨 인식")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser(
        "preview", help="LEFT 실시간 화면; Space=인식, S=저장, Q=종료"
    )
    _add_camera_options(preview)
    _add_api_options(preview)
    _add_capture_options(preview)
    preview.add_argument("--usb-max-fps", type=float, default=10.0)
    _add_verbose(preview)

    capture = subparsers.add_parser(
        "capture", help="LEFT 이미지 한 장을 저장하고 API는 호출하지 않음"
    )
    _add_camera_options(capture)
    _add_capture_options(capture)
    capture.add_argument("--camera-timeout", type=float, default=10.0)
    _add_verbose(capture)

    recognize = subparsers.add_parser(
        "recognize", help="LEFT 이미지 한 장을 Vision API로 인식"
    )
    _add_camera_options(recognize)
    _add_api_options(recognize)
    recognize.add_argument("--camera-timeout", type=float, default=10.0)
    _add_verbose(recognize)

    recognize_image = subparsers.add_parser(
        "recognize-image", help="저장된 JPEG를 카메라 없이 Vision API로 인식"
    )
    recognize_image.add_argument("--image", required=True, type=Path)
    _add_api_options(recognize_image)
    _add_verbose(recognize_image)
    return parser


def _normalized_argv(argv: list[str] | None) -> list[str]:
    """Keep the original no-subcommand recognize syntax backward compatible."""

    values = list(sys.argv[1:] if argv is None else argv)
    if values == ["--help"] or (values and values[0] in COMMANDS):
        return values
    command = "recognize-image" if "--image" in values else "recognize"
    return [command, *values]


def _resolve_port(args: argparse.Namespace) -> str:
    port = args.usb_port or os.environ.get(CAMERA_PORT_ENV)
    if not port or not port.strip():
        raise ValueError(f"--usb-port 또는 환경변수 {CAMERA_PORT_ENV}가 필요합니다.")
    return port.strip()


def _build_vision_client(
    args: argparse.Namespace, *, validate_configuration: bool
) -> VisionClient:
    if args.mock_text is not None:
        return MockVisionClient(args.mock_text)
    client = HttpVisionClient(
        endpoint=args.endpoint,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout=args.vision_timeout,
    )
    if validate_configuration:
        client.validate_configuration()
    return client


def _run_preview(args: argparse.Namespace) -> None:
    # Preview itself works without a key. A missing/invalid key is reported only
    # after Space, and the camera window remains available for another attempt.
    preview = TextRecognitionPreview(
        UsbPreviewCamera(
            _resolve_port(args),
            baudrate=args.baudrate,
            max_fps=args.usb_max_fps,
        ),
        _build_vision_client(args, validate_configuration=False),
        CaptureStore(args.output_dir, jpeg_quality=args.jpeg_quality),
    )
    preview.run()


def _run_capture(args: argparse.Namespace) -> None:
    camera = UsbLeftCamera(
        _resolve_port(args),
        baudrate=args.baudrate,
        capture_timeout=args.camera_timeout,
    )
    capture = CaptureStore(
        args.output_dir, jpeg_quality=args.jpeg_quality
    ).save_jpeg(camera.capture())
    print(
        json.dumps(
            {
                "path": str(capture.path),
                "width": capture.width,
                "height": capture.height,
            },
            ensure_ascii=False,
        )
    )


def _run_recognize(args: argparse.Namespace) -> None:
    camera = UsbLeftCamera(
        _resolve_port(args),
        baudrate=args.baudrate,
        capture_timeout=args.camera_timeout,
    )
    result = TextRecognitionService(
        camera,
        _build_vision_client(args, validate_configuration=True),
    ).recognize_text()
    print(json.dumps(result, ensure_ascii=False))


def _run_recognize_image(args: argparse.Namespace) -> None:
    result = TextRecognitionService(
        FileImageSource(args.image),
        _build_vision_client(args, validate_configuration=True),
    ).recognize_text()
    print(json.dumps(result, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(_normalized_argv(argv))
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    try:
        if args.command == "preview":
            _run_preview(args)
        elif args.command == "capture":
            _run_capture(args)
        elif args.command == "recognize":
            _run_recognize(args)
        elif args.command == "recognize-image":
            _run_recognize_image(args)
        else:  # pragma: no cover - argparse constrains the command.
            raise ValueError(f"지원하지 않는 명령입니다: {args.command}")
    except KeyboardInterrupt:
        logging.info("사용자 요청으로 종료했습니다.")
        return 130
    except (TextRecognitionError, OSError, ValueError, RuntimeError) as exc:
        logging.error("글씨 인식 실패: %s", exc)
        print(json.dumps({"error": "text_recognition_failed"}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

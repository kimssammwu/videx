"""One-shot orchestration and the public Mode 1 entry point."""

from __future__ import annotations

import threading

from .camera import ImageSource, UsbLeftCamera
from .config import TextRecognitionConfig
from .errors import RecognitionBusyError, VisionResponseError
from .vision import HttpVisionClient, VisionClient


class TextRecognitionService:
    """Capture once, call Vision once, and return exactly one text field."""

    def __init__(self, camera: ImageSource, vision_client: VisionClient) -> None:
        self.camera = camera
        self.vision_client = vision_client
        self._request_lock = threading.Lock()

    def recognize_text(self) -> dict[str, str]:
        if not self._request_lock.acquire(blocking=False):
            raise RecognitionBusyError("이전 글씨 인식 요청이 아직 처리 중입니다.")
        try:
            jpeg_bytes = self.camera.capture()
            result = self.vision_client.recognize(jpeg_bytes)
            if not isinstance(result, dict) or set(result) != {"text"}:
                raise VisionResponseError("최종 응답에는 text 필드 하나만 있어야 합니다.")
            if not isinstance(result["text"], str):
                raise VisionResponseError("최종 text 값은 문자열이어야 합니다.")
            return {"text": result["text"]}
        finally:
            self._request_lock.release()


def build_default_service(
    config: TextRecognitionConfig | None = None,
) -> TextRecognitionService:
    settings = config or TextRecognitionConfig.from_env()
    camera = UsbLeftCamera(
        settings.require_left_camera_port(),
        baudrate=settings.baudrate,
        capture_timeout=settings.camera_timeout,
        read_timeout=settings.camera_read_timeout,
    )
    vision = HttpVisionClient(
        endpoint=settings.vision_endpoint,
        model=settings.vision_model,
        api_key_env=settings.vision_api_key_env,
        timeout=settings.vision_timeout,
        max_output_tokens=settings.max_output_tokens,
    )
    vision.validate_configuration()
    return TextRecognitionService(camera, vision)


_default_service: TextRecognitionService | None = None
_default_service_lock = threading.Lock()


def _get_default_service() -> TextRecognitionService:
    global _default_service
    with _default_service_lock:
        if _default_service is None:
            _default_service = build_default_service()
        return _default_service


def recognize_text() -> dict[str, str]:
    """Mode 1 entry point: capture one LEFT image and recognize its text."""

    return _get_default_service().recognize_text()

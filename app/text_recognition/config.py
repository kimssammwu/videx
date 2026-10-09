"""Environment-backed settings for one-shot LEFT-camera text recognition."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError


LEFT_CAMERA_ID = 0
CAMERA_RESOLUTION_MODE = 1
CAMERA_PORT_ENV = "VIDEX_LEFT_CAMERA_PORT"
VISION_API_KEY_ENV = "VIDEX_VISION_API_KEY"
VISION_ENDPOINT_ENV = "VIDEX_VISION_ENDPOINT"
VISION_MODEL_ENV = "VIDEX_VISION_MODEL"
CAMERA_TIMEOUT_ENV = "VIDEX_CAMERA_TIMEOUT"
VISION_TIMEOUT_ENV = "VIDEX_VISION_TIMEOUT"

DEFAULT_VISION_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_VISION_MODEL = "gpt-4.1-mini"
DEFAULT_CAPTURE_DIR = Path(__file__).resolve().parent / "captures"


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"환경변수 {name}는 숫자여야 합니다.") from exc
    if value <= 0:
        raise ConfigurationError(f"환경변수 {name}는 0보다 커야 합니다.")
    return value


@dataclass(frozen=True)
class TextRecognitionConfig:
    """All settings needed by the camera and Vision API adapters."""

    left_camera_port: str | None = None
    baudrate: int = 115200
    camera_timeout: float = 10.0
    camera_read_timeout: float = 0.1
    vision_endpoint: str = DEFAULT_VISION_ENDPOINT
    vision_model: str = DEFAULT_VISION_MODEL
    vision_api_key_env: str = VISION_API_KEY_ENV
    vision_timeout: float = 30.0
    max_output_tokens: int = 8192

    def __post_init__(self) -> None:
        if self.baudrate <= 0:
            raise ConfigurationError("baudrate는 0보다 커야 합니다.")
        if self.camera_timeout <= 0 or self.camera_read_timeout <= 0:
            raise ConfigurationError("카메라 timeout은 0보다 커야 합니다.")
        if self.vision_timeout <= 0 or self.max_output_tokens <= 0:
            raise ConfigurationError("Vision API timeout과 출력 토큰 수는 0보다 커야 합니다.")
        if not self.vision_endpoint.lower().startswith("https://"):
            raise ConfigurationError("Vision API endpoint는 HTTPS를 사용해야 합니다.")
        if not self.vision_model.strip():
            raise ConfigurationError("Vision API model이 비어 있습니다.")
        if not self.vision_api_key_env.strip():
            raise ConfigurationError("Vision API 키 환경변수 이름이 비어 있습니다.")

    @classmethod
    def from_env(cls) -> "TextRecognitionConfig":
        port = os.environ.get(CAMERA_PORT_ENV)
        return cls(
            left_camera_port=port.strip() if port and port.strip() else None,
            camera_timeout=_positive_float(CAMERA_TIMEOUT_ENV, 10.0),
            vision_endpoint=os.environ.get(
                VISION_ENDPOINT_ENV, DEFAULT_VISION_ENDPOINT
            ).strip(),
            vision_model=os.environ.get(VISION_MODEL_ENV, DEFAULT_VISION_MODEL).strip(),
            vision_timeout=_positive_float(VISION_TIMEOUT_ENV, 30.0),
        )

    def require_left_camera_port(self) -> str:
        if not self.left_camera_port:
            raise ConfigurationError(
                f"LEFT 카메라 포트가 없습니다. 환경변수 {CAMERA_PORT_ENV}를 설정하세요."
            )
        return self.left_camera_port

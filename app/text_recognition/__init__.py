"""Public API for VIDEX Mode 1 text recognition."""

from .errors import (
    CameraConnectionError,
    CameraError,
    CameraTimeoutError,
    ConfigurationError,
    InvalidImageError,
    RecognitionBusyError,
    TextRecognitionError,
    VisionApiError,
    VisionAuthenticationError,
    VisionResponseError,
    VisionTimeoutError,
)
from .service import TextRecognitionService, recognize_text

__all__ = [
    "CameraConnectionError",
    "CameraError",
    "CameraTimeoutError",
    "ConfigurationError",
    "InvalidImageError",
    "RecognitionBusyError",
    "TextRecognitionError",
    "TextRecognitionService",
    "VisionApiError",
    "VisionAuthenticationError",
    "VisionResponseError",
    "VisionTimeoutError",
    "recognize_text",
]

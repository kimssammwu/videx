"""VIDEX product front/back capture and recognition pipeline."""

from .api import (
    PRODUCT_PROMPT,
    CachingVisionClient,
    HttpVisionClient,
    MockVisionClient,
    VisionApiError,
    parse_product_name,
)
from .models import FramePacket, RecognitionConfig, RecognitionState
from .recognizer import ObjectRecognizer

__all__ = [
    "PRODUCT_PROMPT",
    "CachingVisionClient",
    "FramePacket",
    "HttpVisionClient",
    "MockVisionClient",
    "ObjectRecognizer",
    "RecognitionConfig",
    "RecognitionState",
    "VisionApiError",
    "parse_product_name",
]

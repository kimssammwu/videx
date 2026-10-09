"""Exceptions exposed by the isolated text-recognition feature."""


class TextRecognitionError(RuntimeError):
    """Base class for expected text-recognition failures."""


class ConfigurationError(TextRecognitionError):
    """Required runtime configuration is missing or invalid."""


class CameraError(TextRecognitionError):
    """The LEFT camera could not provide a valid image."""


class CameraConnectionError(CameraError):
    """The camera transport could not be opened or was disconnected."""


class CameraTimeoutError(CameraError):
    """No LEFT-camera JPEG arrived before the configured deadline."""


class InvalidImageError(CameraError):
    """The camera or file supplied invalid JPEG data."""


class VisionApiError(TextRecognitionError):
    """The remote Vision API request failed."""


class VisionAuthenticationError(VisionApiError):
    """Vision API credentials are missing or were rejected."""


class VisionTimeoutError(VisionApiError):
    """The Vision API did not answer before the configured deadline."""


class VisionResponseError(VisionApiError):
    """The Vision API response did not match the required JSON schema."""


class RecognitionBusyError(TextRecognitionError):
    """Another request is already using this recognition service."""

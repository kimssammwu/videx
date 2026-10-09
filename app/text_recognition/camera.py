"""One-shot image sources for the existing VIDEX CAM2 USB transport."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    from object_recognition.image_ops import decode_image
    from object_recognition.usb_source import (
        MAX_READ_SIZE,
        Cam2FrameParser,
        UsbCameraSource,
        resolution_command,
    )
except ModuleNotFoundError as exc:
    # Support both documented ``cd app; import text_recognition`` usage and
    # namespace-package ``import app.text_recognition`` usage from the repo root.
    if exc.name != "object_recognition":
        raise
    from app.object_recognition.image_ops import decode_image
    from app.object_recognition.usb_source import (
        MAX_READ_SIZE,
        Cam2FrameParser,
        UsbCameraSource,
        resolution_command,
    )

from .config import CAMERA_RESOLUTION_MODE, LEFT_CAMERA_ID
from .errors import (
    CameraConnectionError,
    CameraTimeoutError,
    InvalidImageError,
)

LOGGER = logging.getLogger("videx.text_recognition.camera")
LOGGER.addHandler(logging.NullHandler())


class ImageSource(Protocol):
    def capture(self) -> bytes:
        """Return exactly one validated JPEG image."""


@dataclass(frozen=True)
class PreviewFrame:
    """One decoded LEFT frame and the transport metadata shown in Preview."""

    image: np.ndarray
    frame_id: str | int | None
    received_at: float | None


class PreviewSource(Protocol):
    def open(self) -> "PreviewSource":
        """Open the streaming camera transport."""

    def poll(self) -> list[PreviewFrame]:
        """Return newly received valid LEFT frames without blocking indefinitely."""

    def close(self) -> None:
        """Release the camera transport."""


def validate_jpeg(jpeg_bytes: bytes) -> bytes:
    """Validate JPEG markers and pixels while preserving the original bytes."""

    if not isinstance(jpeg_bytes, bytes):
        raise InvalidImageError("JPEG 데이터가 bytes 형식이 아닙니다.")
    if not jpeg_bytes.startswith(b"\xff\xd8") or not jpeg_bytes.endswith(b"\xff\xd9"):
        raise InvalidImageError("JPEG 시작 또는 종료 마커가 올바르지 않습니다.")
    try:
        decode_image(jpeg_bytes)
    except ValueError as exc:
        raise InvalidImageError("JPEG 이미지를 디코딩할 수 없습니다.") from exc
    return jpeg_bytes


class FileImageSource:
    """Read one JPEG once; useful for offline integration checks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def capture(self) -> bytes:
        try:
            data = self.path.read_bytes()
        except OSError as exc:
            raise CameraConnectionError(f"이미지 파일을 읽을 수 없습니다: {self.path}") from exc
        return validate_jpeg(data)


class UsbLeftCamera:
    """Capture the first valid LEFT JPEG from one CAM2 USB connection.

    The parser and mode command are imported from the existing object-recognition
    implementation.  The raw JPEG from the CAM2 frame is returned without
    decoding/re-encoding, and RIGHT frames are discarded before pixel decoding.
    """

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        capture_timeout: float = 10.0,
        read_timeout: float = 0.1,
        serial_factory: Callable[..., Any] | None = None,
        serial_exceptions: tuple[type[BaseException], ...] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not port or not port.strip():
            raise ValueError("LEFT 카메라 USB 포트가 비어 있습니다.")
        if baudrate <= 0 or capture_timeout <= 0 or read_timeout <= 0:
            raise ValueError("baudrate와 timeout은 0보다 커야 합니다.")
        self.port = port
        self.baudrate = baudrate
        self.capture_timeout = capture_timeout
        self.read_timeout = read_timeout
        self.serial_factory = serial_factory
        self.serial_exceptions = serial_exceptions or (OSError,)
        self.clock = clock

    def _factory(self) -> Callable[..., Any]:
        if self.serial_factory is not None:
            return self.serial_factory
        try:
            import serial
        except ImportError as exc:
            raise CameraConnectionError(
                "pyserial이 설치되지 않았습니다. app/requirements.txt를 설치하세요."
            ) from exc
        self.serial_exceptions = (serial.SerialException, OSError)
        return serial.Serial

    def _open(self) -> Any:
        factory = self._factory()
        try:
            return factory(
                self.port,
                baudrate=self.baudrate,
                timeout=self.read_timeout,
                write_timeout=0.5,
            )
        except self.serial_exceptions as exc:
            raise CameraConnectionError(
                f"LEFT 카메라 USB 포트 {self.port}를 열 수 없습니다."
            ) from exc

    def capture(self) -> bytes:
        connection = self._open()
        parser = Cam2FrameParser(
            incomplete_timeout=min(self.capture_timeout, 6.0)
        )
        deadline = self.clock() + self.capture_timeout
        try:
            self._request_mode_one(connection)
            while self.clock() < deadline:
                try:
                    waiting = int(getattr(connection, "in_waiting", 0))
                    data = connection.read(min(max(waiting, 1), MAX_READ_SIZE))
                except self.serial_exceptions as exc:
                    raise CameraConnectionError("LEFT 카메라 USB 연결이 끊겼습니다.") from exc
                now = self.clock()
                for frame in parser.feed(data, now):
                    if frame.camera_id != LEFT_CAMERA_ID or frame.paused:
                        continue
                    jpeg = validate_jpeg(frame.jpeg)
                    image = decode_image(jpeg)
                    if image.shape[:2] != (frame.height, frame.width):
                        raise InvalidImageError(
                            "CAM2 헤더와 JPEG의 이미지 크기가 일치하지 않습니다."
                        )
                    LOGGER.info(
                        "Captured one LEFT frame id=%d size=%dx%d bytes=%d",
                        frame.frame_id,
                        frame.width,
                        frame.height,
                        len(jpeg),
                    )
                    return jpeg
            raise CameraTimeoutError(
                f"{self.capture_timeout:g}초 안에 LEFT 카메라 이미지를 받지 못했습니다."
            )
        finally:
            try:
                connection.close()
            except (OSError, *self.serial_exceptions):
                LOGGER.warning("LEFT 카메라 USB 포트를 닫는 중 오류가 발생했습니다.")

    def _request_mode_one(self, connection: Any) -> None:
        """Mirror the existing best-effort resolution request for legacy firmware."""

        try:
            written = connection.write(resolution_command(CAMERA_RESOLUTION_MODE))
            if written != 1:
                raise OSError("해상도 명령을 완전히 쓰지 못했습니다.")
        except (AttributeError, TimeoutError, OSError, *self.serial_exceptions) as exc:
            # Existing VIDEX behavior keeps receiving when old firmware has no
            # writable control endpoint.
            LOGGER.warning(
                "CAM2 Mode 1 요청에 실패하여 기존 스트림을 계속 사용합니다: %s", exc
            )


class UsbPreviewCamera:
    """Persistent LEFT-only CAM2 stream for the interactive Preview window."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        max_fps: float = 10.0,
        source_factory: Callable[..., Any] = UsbCameraSource,
    ) -> None:
        if not port or not port.strip():
            raise ValueError("LEFT 카메라 USB 포트가 비어 있습니다.")
        if baudrate <= 0 or max_fps <= 0:
            raise ValueError("baudrate와 max_fps는 0보다 커야 합니다.")
        self.port = port
        self.baudrate = baudrate
        self.max_fps = max_fps
        self.source_factory = source_factory
        self._source: Any | None = None

    def open(self) -> "UsbPreviewCamera":
        if self._source is not None:
            return self
        source = self.source_factory(
            self.port,
            LEFT_CAMERA_ID,
            baudrate=self.baudrate,
            max_fps=self.max_fps,
            resolution_mode=CAMERA_RESOLUTION_MODE,
        )
        try:
            source.open()
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                source.close()
            except (OSError, RuntimeError):
                pass
            raise CameraConnectionError(
                f"LEFT 카메라 Preview 포트 {self.port}를 열 수 없습니다."
            ) from exc
        self._source = source
        return self

    def poll(self) -> list[PreviewFrame]:
        if self._source is None:
            raise CameraConnectionError("LEFT 카메라 Preview가 열려 있지 않습니다.")
        try:
            packets = self._source.poll()
        except (OSError, RuntimeError, ValueError) as exc:
            raise CameraConnectionError("LEFT 카메라 Preview 수신에 실패했습니다.") from exc
        frames: list[PreviewFrame] = []
        for packet in packets:
            if str(packet.camera_id) != str(LEFT_CAMERA_ID):
                continue
            frames.append(
                PreviewFrame(
                    image=decode_image(packet.data).copy(),
                    frame_id=packet.frame_id,
                    received_at=packet.timestamp,
                )
            )
        return frames

    def close(self) -> None:
        source, self._source = self._source, None
        if source is None:
            return
        try:
            source.close()
        except (OSError, RuntimeError):
            LOGGER.warning("LEFT 카메라 Preview 포트를 닫는 중 오류가 발생했습니다.")

    def __enter__(self) -> "UsbPreviewCamera":
        return self.open()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

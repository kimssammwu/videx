"""CAM2 USB CDC input adapter compatible with proxy/viewer.py."""

from __future__ import annotations

import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .image_ops import decode_image
from .models import FramePacket

LOGGER = logging.getLogger("videx.object_recognition.usb")
LOGGER.addHandler(logging.NullHandler())

CAM2_MAGIC = b"CAM2"
CAM2_HEADER = struct.Struct("<IB3xIIHHQ")
MAX_JPEG_SIZE = 1024 * 1024
INCOMPLETE_FRAME_TIMEOUT = 6.0
MAX_READ_SIZE = 65536


class UsbCameraError(RuntimeError):
    """A safe, user-facing USB connection or dependency error."""


@dataclass(frozen=True)
class Cam2Frame:
    camera_id: int
    frame_id: int
    width: int
    height: int
    timestamp_us: int
    jpeg: bytes
    received_at: float


@dataclass
class UsbReceiveStats:
    bytes_received: int = 0
    frames_parsed: int = 0
    frames_selected: int = 0
    invalid_frames: int = 0
    other_camera_frames: int = 0
    rate_limited_frames: int = 0
    duplicate_frames: int = 0


class Cam2FrameParser:
    """Incrementally split CAM2 frames regardless of USB read boundaries.

    The framing and validation rules intentionally match proxy/viewer.py.
    """

    def __init__(self, incomplete_timeout: float = INCOMPLETE_FRAME_TIMEOUT) -> None:
        self.buffer = bytearray()
        self.started: float | None = None
        self.incomplete_timeout = incomplete_timeout

    def feed(self, data: bytes, now: float | None = None) -> list[Cam2Frame]:
        now = time.monotonic() if now is None else now
        self.buffer.extend(data)
        frames: list[Cam2Frame] = []
        while True:
            offset = self.buffer.find(CAM2_MAGIC)
            if offset < 0:
                # Keep a possible partial CAM2 magic prefix across USB reads.
                self.buffer[:] = self.buffer[-3:]
                self.started = None
                break
            if offset:
                del self.buffer[:offset]
                self.started = None
            if self.started is None:
                self.started = now
            if now - self.started >= self.incomplete_timeout:
                del self.buffer[0]
                self.started = None
                continue
            if len(self.buffer) < CAM2_HEADER.size:
                break
            _, camera_id, frame_id, size, width, height, timestamp_us = CAM2_HEADER.unpack_from(
                self.buffer
            )
            if not (
                camera_id in (0, 1)
                and self.buffer[5:8] == b"\x00\x00\x00"
                and 4 <= size <= MAX_JPEG_SIZE
                and 0 < width <= 2000
                and 0 < height <= 2000
            ):
                del self.buffer[0]
                self.started = None
                continue
            end = CAM2_HEADER.size + size
            if len(self.buffer) < end:
                break
            jpeg = bytes(self.buffer[CAM2_HEADER.size:end])
            if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
                del self.buffer[0]
                self.started = None
                continue
            frames.append(
                Cam2Frame(camera_id, frame_id, width, height, timestamp_us, jpeg, now)
            )
            del self.buffer[:end]
            self.started = None
        return frames


def cam2_frame_to_packet(frame: Cam2Frame) -> FramePacket:
    """Validate JPEG pixels and use PC receipt time for stability timing."""

    image = decode_image(frame.jpeg)
    if image.shape[:2] != (frame.height, frame.width):
        raise ValueError(
            f"CAM2 크기 불일치: header={frame.width}x{frame.height}, "
            f"jpeg={image.shape[1]}x{image.shape[0]}"
        )
    return FramePacket(
        data=image,
        camera_id=str(frame.camera_id),
        frame_id=frame.frame_id,
        timestamp=frame.received_at,
    )


class UsbCameraSource:
    """Poll one selected CAM2 camera from a USB CDC serial connection."""

    def __init__(
        self,
        port: str,
        camera_id: int,
        *,
        baudrate: int = 115200,
        read_timeout: float = 0.1,
        max_fps: float = 10.0,
        serial_factory: Callable[..., Any] | None = None,
        serial_exceptions: tuple[type[BaseException], ...] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if camera_id not in (0, 1):
            raise ValueError("camera_id must be 0 or 1")
        if baudrate <= 0 or read_timeout <= 0 or max_fps <= 0:
            raise ValueError("baudrate, read_timeout, and max_fps must be positive")
        self.port = port
        self.camera_id = camera_id
        self.baudrate = baudrate
        self.read_timeout = read_timeout
        self.max_fps = max_fps
        self.serial_factory = serial_factory
        self.serial_exceptions = serial_exceptions or (OSError,)
        self.clock = clock
        self.parser = Cam2FrameParser()
        self.stats = UsbReceiveStats()
        self.connection: Any | None = None
        self.last_packet_at: float | None = None
        self._last_emitted_at: float | None = None
        self._last_frame_id: int | None = None

    def open(self) -> "UsbCameraSource":
        factory = self.serial_factory
        if factory is None:
            try:
                import serial
            except ImportError as exc:
                raise UsbCameraError(
                    "pyserial이 설치되지 않았습니다. requirements.txt를 설치하세요."
                ) from exc
            factory = serial.Serial
            self.serial_exceptions = (serial.SerialException, OSError)
        try:
            self.connection = factory(
                self.port,
                baudrate=self.baudrate,
                timeout=self.read_timeout,
            )
        except self.serial_exceptions as exc:
            raise UsbCameraError(f"USB 포트 {self.port}를 열 수 없습니다: {exc}") from None
        LOGGER.info(
            "USB connected port=%s camera=%d baudrate=%d max_fps=%.1f",
            self.port,
            self.camera_id,
            self.baudrate,
            self.max_fps,
        )
        return self

    def close(self) -> None:
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                connection.close()
            except (OSError, *self.serial_exceptions):
                LOGGER.warning("USB 포트를 닫는 중 오류가 발생했습니다: %s", self.port)
            else:
                LOGGER.info("USB closed port=%s", self.port)

    def __enter__(self) -> "UsbCameraSource":
        return self.open()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def poll(self) -> list[FramePacket]:
        if self.connection is None:
            raise UsbCameraError("USB 포트가 열려 있지 않습니다.")
        try:
            waiting = int(getattr(self.connection, "in_waiting", 0))
            data = self.connection.read(min(max(waiting, 1), MAX_READ_SIZE))
        except self.serial_exceptions as exc:
            raise UsbCameraError(f"USB 연결이 끊겼습니다: {exc}") from None
        now = self.clock()
        self.stats.bytes_received += len(data)
        frames = self.parser.feed(data, now)
        self.stats.frames_parsed += len(frames)
        packets: list[FramePacket] = []
        minimum_interval = 1.0 / self.max_fps
        for frame in frames:
            if frame.camera_id != self.camera_id:
                self.stats.other_camera_frames += 1
                continue
            if frame.frame_id == self._last_frame_id:
                self.stats.duplicate_frames += 1
                continue
            if (
                self._last_emitted_at is not None
                and frame.received_at - self._last_emitted_at < minimum_interval
            ):
                self.stats.rate_limited_frames += 1
                continue
            try:
                packet = cam2_frame_to_packet(frame)
            except ValueError as exc:
                self.stats.invalid_frames += 1
                LOGGER.warning(
                    "Invalid CAM2 JPEG camera=%d frame=%d: %s",
                    frame.camera_id,
                    frame.frame_id,
                    exc,
                )
                continue
            self._last_frame_id = frame.frame_id
            self._last_emitted_at = frame.received_at
            self.last_packet_at = frame.received_at
            self.stats.frames_selected += 1
            packets.append(packet)
            LOGGER.info(
                "USB frame camera=%d frame=%d size=%dx%d timestamp_us=%d bytes=%d",
                frame.camera_id,
                frame.frame_id,
                frame.width,
                frame.height,
                frame.timestamp_us,
                len(frame.jpeg),
            )
        return packets


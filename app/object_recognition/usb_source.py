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
CAM2_HEADER = struct.Struct("<IBBBBIIHHQ")
BUTTON_PRESSED = 1 << 0
BUTTON_VALID = 1 << 1
BUTTON_CLASSIFIED = 1 << 2
STREAM_PAUSED = 1 << 3
BUTTON_RESULTS = {0: None, 1: "SHORT", 2: "LONG"}
MAX_JPEG_SIZE = 1024 * 1024
MAX_FRAME_DIMENSION = 4096
INCOMPLETE_FRAME_TIMEOUT = 6.0
MAX_READ_SIZE = 65536
RESOLUTION_MODES = {0: (320, 240), 1: (640, 480), 2: (1280, 1024)}
DEFAULT_RESOLUTION_MODE = 1
RESOLUTION_MISMATCH_WARNING_FRAMES = 3


def resolution_command(mode: int) -> bytes:
    """Return the one-byte resolution command used by proxy/viewer.py."""

    if mode not in RESOLUTION_MODES:
        raise ValueError("resolution mode must be 0, 1, or 2")
    return bytes([mode])


def resolution_label(mode: int) -> str:
    """Describe a resolution mode using the current Viewer semantics."""

    if mode not in RESOLUTION_MODES:
        raise ValueError("resolution mode must be 0, 1, or 2")
    width, height = RESOLUTION_MODES[mode]
    scope = "both cameras" if mode == 0 else "LEFT only in current Viewer"
    return f"mode {mode} {width}x{height} ({scope})"


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
    flags: int = 0
    button_result_code: int = 0
    button_pressed: bool | None = None
    button_classified: bool = False
    button_result: str | None = None
    paused: bool = False


@dataclass(frozen=True)
class Cam2FrameInfo:
    """Small transport snapshot that does not retain the JPEG payload."""

    camera_id: int
    frame_id: int
    width: int
    height: int
    timestamp_us: int
    received_at: float
    flags: int
    button_pressed: bool | None
    button_classified: bool
    button_result: str | None
    paused: bool


@dataclass
class UsbReceiveStats:
    bytes_received: int = 0
    frames_parsed: int = 0
    frames_selected: int = 0
    invalid_frames: int = 0
    other_camera_frames: int = 0
    rate_limited_frames: int = 0
    duplicate_frames: int = 0
    paused_messages: int = 0


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
            (
                _,
                camera_id,
                flags,
                button_result_code,
                reserved,
                frame_id,
                size,
                width,
                height,
                timestamp_us,
            ) = CAM2_HEADER.unpack_from(self.buffer)
            paused = (
                camera_id == 1
                and flags == STREAM_PAUSED
                and button_result_code == 0
                and reserved == 0
                and size == 0
                and width == 0
                and height == 0
            )
            valid_image = (
                camera_id in (0, 1)
                and reserved == 0
                and (
                    (flags == 0 and button_result_code == 0)
                    or (camera_id == 0 and flags in (2, 3) and button_result_code == 0)
                    or (
                        camera_id == 0
                        and flags == BUTTON_CLASSIFIED
                        and button_result_code == 0
                    )
                    or (
                        camera_id == 0
                        and flags in (6, 7)
                        and button_result_code in BUTTON_RESULTS
                    )
                )
                and 4 <= size <= MAX_JPEG_SIZE
                and 0 < width <= MAX_FRAME_DIMENSION
                and 0 < height <= MAX_FRAME_DIMENSION
            )
            if not (paused or valid_image):
                del self.buffer[0]
                self.started = None
                continue
            end = CAM2_HEADER.size + size
            if len(self.buffer) < end:
                break
            jpeg = bytes(self.buffer[CAM2_HEADER.size:end])
            if not paused and not (
                jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
            ):
                del self.buffer[0]
                self.started = None
                continue
            button_pressed = (
                bool(flags & BUTTON_PRESSED) if flags & BUTTON_VALID else None
            )
            frames.append(
                Cam2Frame(
                    camera_id,
                    frame_id,
                    width,
                    height,
                    timestamp_us,
                    jpeg,
                    now,
                    flags=flags,
                    button_result_code=button_result_code,
                    button_pressed=button_pressed,
                    button_classified=bool(flags & BUTTON_CLASSIFIED),
                    button_result=BUTTON_RESULTS[button_result_code],
                    paused=paused,
                )
            )
            del self.buffer[:end]
            self.started = None
        return frames


def cam2_frame_to_packet(frame: Cam2Frame) -> FramePacket:
    """Validate JPEG pixels and use PC receipt time for stability timing."""

    if frame.paused:
        raise ValueError("STREAM_PAUSED 메시지에는 JPEG 이미지가 없습니다.")
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
    """Poll one or both CAM2 cameras from one USB CDC serial connection."""

    def __init__(
        self,
        port: str,
        camera_id: int | None,
        *,
        baudrate: int = 115200,
        read_timeout: float = 0.1,
        max_fps: float = 10.0,
        serial_factory: Callable[..., Any] | None = None,
        serial_exceptions: tuple[type[BaseException], ...] | None = None,
        clock: Callable[[], float] = time.monotonic,
        resolution_mode: int | None = DEFAULT_RESOLUTION_MODE,
    ) -> None:
        if camera_id not in (0, 1, None):
            raise ValueError("camera_id must be 0, 1, or None for both cameras")
        if baudrate <= 0 or read_timeout <= 0 or max_fps <= 0:
            raise ValueError("baudrate, read_timeout, and max_fps must be positive")
        if resolution_mode is not None and resolution_mode not in RESOLUTION_MODES:
            raise ValueError("resolution_mode must be 0, 1, 2, or None")
        self.port = port
        self.camera_id = camera_id
        self.baudrate = baudrate
        self.read_timeout = read_timeout
        self.max_fps = max_fps
        self.serial_factory = serial_factory
        self.serial_exceptions = serial_exceptions or (OSError,)
        self.clock = clock
        self.resolution_mode = resolution_mode
        self.parser = Cam2FrameParser()
        self.stats = UsbReceiveStats()
        self.connection: Any | None = None
        self.last_packet_at: float | None = None
        self.last_packet_at_by_camera: dict[int, float] = {}
        self.last_frame_info_by_camera: dict[int, Cam2FrameInfo] = {}
        self.resolution_request_status = "not_requested"
        self.resolution_request_error: str | None = None
        self.selected_actual_resolution: tuple[int, int] | None = None
        self.selected_resolution_status = "waiting_for_camera"
        self._last_emitted_at: dict[int, float] = {}
        self._last_frame_id: dict[int, int] = {}
        self._paused_by_camera: dict[int, bool] = {}
        self._resolution_mismatch_frames = 0

    def open(self) -> "UsbCameraSource":
        # A reopened CDC stream starts at a new byte boundary and must request
        # the fixed recognition mode again.
        self.parser = Cam2FrameParser()
        self.resolution_request_status = "not_requested"
        self.resolution_request_error = None
        self.selected_actual_resolution = None
        self.selected_resolution_status = "waiting_for_camera"
        self._last_emitted_at.clear()
        self._last_frame_id.clear()
        self._paused_by_camera.clear()
        self._resolution_mismatch_frames = 0
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
                write_timeout=0.5,
            )
        except self.serial_exceptions as exc:
            raise UsbCameraError(f"USB 포트 {self.port}를 열 수 없습니다: {exc}") from None
        LOGGER.info(
            "USB connected port=%s camera=%s baudrate=%d max_fps=%.1f",
            self.port,
            "both" if self.camera_id is None else self.camera_id,
            self.baudrate,
            self.max_fps,
        )
        self._request_resolution()
        return self

    def _request_resolution(self) -> None:
        """Request the viewer-compatible mode without requiring firmware ACK support."""

        if self.resolution_mode is None:
            return
        self.request_resolution(self.resolution_mode, force=True)

    def request_resolution(self, mode: int, *, force: bool = False) -> bool:
        """Send a live resolution command over the already-open COM connection.

        The protocol has no acknowledgement. The separate selected-camera
        observation status is updated only after a frame from that camera.
        """

        resolution_command(mode)  # Validate before mutating request state.
        if self.connection is None:
            raise UsbCameraError("USB 포트가 열려 있지 않습니다.")
        if not force and mode == self.resolution_mode:
            LOGGER.info("USB resolution request unchanged: %s", resolution_label(mode))
            return False
        self.resolution_mode = mode
        self.resolution_request_status = "sending"
        self.resolution_request_error = None
        self.selected_resolution_status = "waiting_for_camera"
        self._resolution_mismatch_frames = 0
        try:
            written = self.connection.write(resolution_command(mode))
            if written != 1:
                raise OSError(f"incomplete write ({written!r}/1 byte)")
        except (AttributeError, TimeoutError, OSError, *self.serial_exceptions) as exc:
            self.resolution_request_status = "failed"
            self.resolution_request_error = str(exc)
            LOGGER.warning(
                "USB resolution mode %d request failed; continuing for legacy firmware: %s",
                mode,
                exc,
            )
            return False
        self.resolution_request_status = "sent_unconfirmed"
        self.resolution_request_error = None
        width, height = RESOLUTION_MODES[mode]
        LOGGER.info(
            "Requested USB resolution mode=%d expected=%dx%d; firmware has no ACK, support unconfirmed",
            mode,
            width,
            height,
        )
        return True

    def _observe_selected_resolution(self, frame: Cam2Frame) -> None:
        if self.camera_id is None or frame.camera_id != self.camera_id:
            return
        if frame.paused:
            self.selected_resolution_status = "selected_camera_paused"
            self._resolution_mismatch_frames = 0
            return
        self.selected_actual_resolution = (frame.width, frame.height)
        expected = RESOLUTION_MODES.get(self.resolution_mode)
        if expected == self.selected_actual_resolution:
            if self.selected_resolution_status == "actual_mismatch":
                LOGGER.info(
                    "Requested resolution is now active camera=%d actual=%dx%d",
                    frame.camera_id,
                    frame.width,
                    frame.height,
                )
            self.selected_resolution_status = "actual_matches_request"
            self._resolution_mismatch_frames = 0
            return
        self.selected_resolution_status = "actual_mismatch"
        self._resolution_mismatch_frames += 1
        if self._resolution_mismatch_frames == RESOLUTION_MISMATCH_WARNING_FRAMES:
            expected_text = (
                f"{expected[0]}x{expected[1]}" if expected is not None else "unknown"
            )
            LOGGER.warning(
                "Requested resolution was not applied camera=%d mode=%s expected=%s actual=%dx%d",
                frame.camera_id,
                self.resolution_mode,
                expected_text,
                frame.width,
                frame.height,
            )

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
            self._observe_selected_resolution(frame)
            info = Cam2FrameInfo(
                camera_id=frame.camera_id,
                frame_id=frame.frame_id,
                width=frame.width,
                height=frame.height,
                timestamp_us=frame.timestamp_us,
                received_at=frame.received_at,
                flags=frame.flags,
                button_pressed=frame.button_pressed,
                button_classified=frame.button_classified,
                button_result=frame.button_result,
                paused=frame.paused,
            )
            self.last_frame_info_by_camera[frame.camera_id] = info
            if self.camera_id is not None and frame.camera_id != self.camera_id:
                self.stats.other_camera_frames += 1
                continue
            if frame.paused:
                self.stats.paused_messages += 1
                if not self._paused_by_camera.get(frame.camera_id, False):
                    LOGGER.warning(
                        "CAM2 camera=%d STREAM_PAUSED; RIGHT recognition requires resolution mode 0",
                        frame.camera_id,
                    )
                self._paused_by_camera[frame.camera_id] = True
                continue
            if self._paused_by_camera.get(frame.camera_id, False):
                LOGGER.info("CAM2 camera=%d stream resumed", frame.camera_id)
            self._paused_by_camera[frame.camera_id] = False
            if frame.frame_id == self._last_frame_id.get(frame.camera_id):
                self.stats.duplicate_frames += 1
                continue
            if (
                frame.camera_id in self._last_emitted_at
                and frame.received_at - self._last_emitted_at[frame.camera_id] < minimum_interval
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
            self._last_frame_id[frame.camera_id] = frame.frame_id
            self._last_emitted_at[frame.camera_id] = frame.received_at
            self.last_packet_at = frame.received_at
            self.last_packet_at_by_camera[frame.camera_id] = frame.received_at
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

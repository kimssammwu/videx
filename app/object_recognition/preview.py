"""LEFT-camera recognition worker and integrated OpenCV preview."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .image_ops import decode_image
from .models import FrameMetrics, FramePacket, RecognitionState
from .recognizer import ObjectRecognizer
from .usb_source import DEFAULT_RESOLUTION_MODE, RESOLUTION_MODES

LOGGER = logging.getLogger("videx.object_recognition.preview")
LOGGER.addHandler(logging.NullHandler())

LEFT_CAMERA_ID = 0


class RecognitionPreviewWorker:
    """Process only the latest LEFT frame without blocking the GUI thread."""

    def __init__(self, recognizer: ObjectRecognizer) -> None:
        self.recognizer = recognizer
        self.latest_metrics: FrameMetrics | None = None
        self.latest_error: str | None = None
        self.last_processed_frame_id: str | int | None = None
        self._condition = threading.Condition()
        self._pending: tuple[FramePacket, bool] | None = None
        self._stop = False
        self._process_lock = threading.Lock()
        self._processing = False
        self._thread = threading.Thread(
            target=self._run,
            name="videx-left-recognition",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def submit(self, packet: FramePacket, *, manual_capture: bool = False) -> None:
        if packet.camera_id != str(LEFT_CAMERA_ID):
            raise ValueError("LEFT(camera_id=0) 프레임만 인식 작업에 전달할 수 있습니다.")
        with self._condition:
            if self._pending is not None and self._pending[1] and not manual_capture:
                return
            self._pending = (packet, manual_capture)
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._pending is None and not self._stop:
                    self._condition.wait(timeout=0.05)
                if self._stop:
                    return
                task = self._pending
                self._pending = None
            if task is None:
                if self._process_lock.acquire(blocking=False):
                    try:
                        self.recognizer.tick()
                    finally:
                        self._process_lock.release()
                continue
            packet, manual_capture = task
            if self.recognizer.state in (
                RecognitionState.IDLE,
                RecognitionState.COMPLETED,
                RecognitionState.ERROR,
            ):
                continue
            with self._process_lock:
                self._processing = True
                try:
                    self.latest_metrics = self.recognizer.process_frame(
                        packet,
                        manual_capture=manual_capture,
                    )
                    self.last_processed_frame_id = packet.frame_id
                    self.latest_error = None
                except (OSError, ValueError, RuntimeError) as exc:
                    self.latest_error = str(exc)
                    LOGGER.error("LEFT 프레임 처리 실패: %s", exc)
                finally:
                    self._processing = False

    def reset(self) -> bool:
        """Restart from the front unless an API/frame operation is in progress."""

        if not self._process_lock.acquire(blocking=False):
            return False
        try:
            with self._condition:
                self._pending = None
            self.recognizer.reset_recognition()
            self.recognizer.start_recognition(str(LEFT_CAMERA_ID))
            self.latest_metrics = None
            self.latest_error = None
            self.last_processed_frame_id = None
            return True
        finally:
            self._process_lock.release()

    @property
    def busy(self) -> bool:
        return self._processing

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class LeftCameraViewer:
    """Show LEFT live video plus front/back/merged capture results."""

    WINDOW_NAME = "VIDEX LEFT Camera Recognition"
    WIDTH = 1280
    HEIGHT = 720
    LIVE_WIDTH = 800
    HEADER_HEIGHT = 136
    STALE_AFTER = 2.0

    def __init__(self, roi_fraction: float = 1.0) -> None:
        self.roi_fraction = roi_fraction
        self.latest_image: np.ndarray | None = None
        self.latest_received_at: float | None = None
        self.latest_frame_id: str | int | None = None
        self.usb_connected = False
        self.transport_camera_id: int | None = None
        self.transport_frame_id: int | None = None
        self.transport_width: int | None = None
        self.transport_height: int | None = None
        self.transport_received_at: float | None = None
        self.stream_paused = False
        self.resolution_request_status = "not_requested"
        self.resolution_support_status = "waiting_for_camera"
        self.requested_resolution_mode = DEFAULT_RESOLUTION_MODE
        self.resolution_notice: str | None = None
        self._opened = False
        self._image_cache: dict[str, np.ndarray] = {}
        self._banner_cache: tuple[str, np.ndarray] | None = None

    def update_frame(self, packet: FramePacket) -> None:
        if packet.camera_id != str(LEFT_CAMERA_ID):
            return
        self.latest_image = decode_image(packet.data).copy()
        self.latest_received_at = packet.timestamp
        self.latest_frame_id = packet.frame_id

    def update_usb_status(
        self,
        *,
        connected: bool,
        camera_id: int | None = None,
        frame_id: int | None = None,
        width: int | None = None,
        height: int | None = None,
        received_at: float | None = None,
        paused: bool | None = None,
        resolution_status: str | None = None,
        resolution_support_status: str | None = None,
        requested_resolution_mode: int | None = None,
    ) -> None:
        """Update transport-only state, including messages with no JPEG payload."""

        self.usb_connected = connected
        if camera_id is not None:
            self.transport_camera_id = camera_id
            self.transport_frame_id = frame_id
            self.transport_received_at = received_at
            if width and height:
                self.transport_width = width
                self.transport_height = height
        if paused is not None:
            self.stream_paused = paused
        if resolution_status is not None:
            self.resolution_request_status = resolution_status
        if resolution_support_status is not None:
            self.resolution_support_status = resolution_support_status
        if requested_resolution_mode is not None:
            self.requested_resolution_mode = requested_resolution_mode

    def set_resolution_notice(self, message: str | None) -> None:
        self.resolution_notice = message

    def render(
        self,
        recognizer: ObjectRecognizer,
        metrics: FrameMetrics | None,
        *,
        worker_busy: bool = False,
        worker_error: str | None = None,
        now: float | None = None,
    ) -> np.ndarray:
        current = time.monotonic() if now is None else now
        canvas = np.full((self.HEIGHT, self.WIDTH, 3), 22, dtype=np.uint8)
        state = recognizer.get_recognition_state()
        stale = (
            self.latest_received_at is None
            or current - self.latest_received_at > self.STALE_AFTER
        )
        self._draw_header(canvas, state, metrics, worker_busy, stale, worker_error)
        self._draw_live(canvas, stale)
        self._draw_resolution_notice(canvas)
        self._draw_capture(canvas, "FRONT", state.get("front_path"), 16)
        self._draw_capture(canvas, "BACK", state.get("back_path"), 203)
        self._draw_capture(canvas, "MERGED", state.get("merged_path"), 390)
        result = str(state.get("result") or "Waiting for recognition")
        self._draw_result_banner(canvas, result)
        cv2.putText(
            canvas,
            "C manual capture | R restart | Q/ESC quit",
            (self.LIVE_WIDTH + 14, 704),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (0, 205, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _draw_header(
        self,
        canvas: np.ndarray,
        state: dict[str, object],
        metrics: FrameMetrics | None,
        worker_busy: bool,
        stale: bool,
        worker_error: str | None,
    ) -> None:
        usb = "CONNECTED" if self.usb_connected else "DISCONNECTED"
        camera = self.transport_camera_id
        frame_id = self.transport_frame_id
        actual_dimensions = (
            f"{self.transport_width}x{self.transport_height}"
            if self.transport_width and self.transport_height
            else "size -"
        )
        cv2.putText(
            canvas,
            f"USB {usb} | CAMERA LEFT ({camera if camera is not None else 0}) | "
            f"frame {frame_id} | state {state['state']}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if metrics is None:
            details = "motion - | sharpness - | brightness - | stable no | reason waiting_frame"
            ratio = 0.0
            progress = 0.0
        else:
            motion = "-" if metrics.motion is None else f"{metrics.motion:.4f}"
            ratio = metrics.stability_ratio or 0.0
            progress = metrics.stability_progress
            details = (
                f"motion {motion} | sharpness {metrics.sharpness:.1f} | "
                f"brightness {metrics.brightness:.0f} | stable {metrics.stable} | "
                f"ratio {ratio:.0%} | reason {metrics.reason}"
            )
        details += " | analysis FULL FRAME" if self.roi_fraction >= 0.999 else " | analysis ROI"
        if stale:
            details += " | STALE"
        if worker_busy:
            details += " | PROCESSING"
        if worker_error:
            details += " | ERROR"
        if self.stream_paused:
            details += " | STREAM_PAUSED"
        cv2.putText(
            canvas,
            details,
            (15, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 220, 255) if not stale else (0, 80, 255),
            1,
            cv2.LINE_AA,
        )
        requested_width, requested_height = RESOLUTION_MODES[self.requested_resolution_mode]
        actual = (
            f"PAUSED (last {actual_dimensions})"
            if self.stream_paused
            else actual_dimensions
        )
        cv2.putText(
            canvas,
            f"stability {progress:.0%} / required ratio 80% | "
            f"Resolution: {requested_width}x{requested_height} "
            f"(Mode {self.requested_resolution_mode}) | Actual: {actual}",
            (15, 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"resolution command {self.resolution_request_status} | "
            f"LEFT status {self.resolution_support_status}",
            (15, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (15, 116), (785, 128), (70, 70, 70), 1)
        cv2.rectangle(
            canvas,
            (16, 117),
            (16 + round(768 * min(1.0, progress)), 127),
            (0, 190, 0) if ratio >= 0.80 else (0, 170, 255),
            -1,
        )

    def _draw_live(self, canvas: np.ndarray, stale: bool) -> None:
        top = self.HEADER_HEIGHT
        area_height = self.HEIGHT - top
        if self.latest_image is None:
            cv2.putText(
                canvas,
                (
                    f"LEFT STREAM PAUSED / REQUESTED MODE {self.requested_resolution_mode}"
                    if self.stream_paused
                    else "WAITING FOR LEFT CAM2 JPEG"
                ),
                (225, top + area_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )
            return
        image = self.latest_image
        scale = min(self.LIVE_WIDTH / image.shape[1], area_height / image.shape[0])
        width = max(1, round(image.shape[1] * scale))
        height = max(1, round(image.shape[0] * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(image, (width, height), interpolation=interpolation)
        left = (self.LIVE_WIDTH - width) // 2
        image_top = top + (area_height - height) // 2
        canvas[image_top : image_top + height, left : left + width] = resized
        if self.roi_fraction < 0.999:
            roi_width = round(width * self.roi_fraction)
            roi_height = round(height * self.roi_fraction)
            roi_left = left + (width - roi_width) // 2
            roi_top = image_top + (height - roi_height) // 2
            cv2.rectangle(
                canvas,
                (roi_left, roi_top),
                (roi_left + roi_width, roi_top + roi_height),
                (0, 255, 255),
                2,
            )
        if self.stream_paused:
            cv2.putText(
                canvas,
                f"LEFT STREAM PAUSED / REQUESTED MODE {self.requested_resolution_mode}",
                (left + 18, image_top + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if stale:
            cv2.putText(
                canvas,
                "LEFT CAMERA STALE",
                (left + 18, image_top + (68 if self.stream_paused else 35)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_resolution_notice(self, canvas: np.ndarray) -> None:
        if not self.resolution_notice:
            return
        cv2.rectangle(canvas, (8, self.HEIGHT - 35), (self.LIVE_WIDTH - 8, self.HEIGHT - 7), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            self.resolution_notice[:105],
            (16, self.HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )

    def _load_image(self, value: object) -> np.ndarray | None:
        if not value:
            return None
        path = str(value)
        if path not in self._image_cache:
            try:
                self._image_cache[path] = decode_image(Path(path).read_bytes())
            except (OSError, ValueError):
                return None
        return self._image_cache[path]

    def _draw_capture(self, canvas: np.ndarray, label: str, value: object, top: int) -> None:
        left = self.LIVE_WIDTH + 12
        width = self.WIDTH - left - 12
        height = 172
        cv2.rectangle(canvas, (left, top), (left + width, top + height), (70, 70, 70), 1)
        cv2.putText(
            canvas,
            label,
            (left + 8, top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        image = self._load_image(value)
        if image is None:
            cv2.putText(
                canvas,
                "not captured",
                (left + 155, top + 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )
            return
        available_width = width - 16
        available_height = height - 32
        scale = min(available_width / image.shape[1], available_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        x = left + (width - resized.shape[1]) // 2
        y = top + 27 + (available_height - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized

    def _draw_result_banner(self, canvas: np.ndarray, result: str) -> None:
        top = 574
        left = self.LIVE_WIDTH + 12
        width = self.WIDTH - left - 12
        height = 105
        if self._banner_cache is None or self._banner_cache[0] != result:
            banner = np.full((height, width, 3), 34, dtype=np.uint8)
            try:
                from PIL import Image, ImageDraw, ImageFont

                rgb = cv2.cvtColor(banner, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(pil_image)
                font_path = Path("C:/Windows/Fonts/malgun.ttf")
                font = ImageFont.truetype(str(font_path), 24) if font_path.is_file() else ImageFont.load_default()
                draw.text((12, 13), "인식 결과", font=font, fill=(220, 220, 220))
                draw.text((12, 52), result, font=font, fill=(80, 255, 80))
                banner = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
            except (ImportError, OSError):
                cv2.putText(banner, "RESULT", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(banner, result, (12, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 1, cv2.LINE_AA)
            self._banner_cache = (result, banner)
        canvas[top : top + height, left : left + width] = self._banner_cache[1]

    def open(self) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.WIDTH, self.HEIGHT)
        self._opened = True

    def show(self, canvas: np.ndarray) -> str | None:
        if not self._opened:
            self.open()
        cv2.imshow(self.WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "q"
        if key in (ord("q"), ord("Q"), 27):
            return "q"
        if key in (ord("c"), ord("C")):
            return "c"
        if key in (ord("r"), ord("R")):
            return "r"
        return None

    def close(self) -> None:
        if self._opened:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except cv2.error:
                pass
            self._opened = False

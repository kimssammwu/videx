"""Interactive LEFT-camera Preview, capture storage, and async recognition."""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

try:
    from object_recognition.image_ops import decode_image, encode_jpeg
except ModuleNotFoundError as exc:
    if exc.name != "object_recognition":
        raise
    from app.object_recognition.image_ops import decode_image, encode_jpeg

from .camera import PreviewFrame, PreviewSource, validate_jpeg
from .config import DEFAULT_CAPTURE_DIR
from .errors import TextRecognitionError
from .vision import VisionClient

LOGGER = logging.getLogger("videx.text_recognition.preview")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class SavedCapture:
    path: Path
    jpeg_bytes: bytes
    width: int
    height: int


class CaptureStore:
    """Save unique JPEG captures and report their exact dimensions."""

    def __init__(
        self,
        directory: str | Path = DEFAULT_CAPTURE_DIR,
        *,
        jpeg_quality: int = 95,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality는 1 이상 100 이하여야 합니다.")
        self.directory = Path(directory)
        self.jpeg_quality = jpeg_quality
        self.now = now

    def save_image(self, image: np.ndarray) -> SavedCapture:
        """Encode once, then save and return the exact bytes for API use."""

        return self.save_jpeg(encode_jpeg(image, self.jpeg_quality))

    def save_jpeg(self, jpeg_bytes: bytes) -> SavedCapture:
        jpeg = validate_jpeg(jpeg_bytes)
        image = decode_image(jpeg)
        height, width = image.shape[:2]
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = self.now().strftime("%Y%m%d_%H%M%S_%f")
        for sequence in range(10_000):
            suffix = "" if sequence == 0 else f"_{sequence:04d}"
            path = self.directory / f"text_{stamp}{suffix}.jpg"
            try:
                with path.open("xb") as output:
                    output.write(jpeg)
            except FileExistsError:
                continue
            return SavedCapture(path, jpeg, width, height)
        raise OSError("고유한 촬영 이미지 파일명을 만들 수 없습니다.")


class PreviewUI(Protocol):
    def open(self) -> None:
        """Create the Preview UI resources."""

    def show(self, frame: PreviewFrame | None, status: str) -> str | None:
        """Render one frame and return capture/save/quit when a key is pressed."""

    def close(self) -> None:
        """Release Preview UI resources."""


class OpenCvPreviewUI:
    """Small OpenCV window with Space/S/Q keyboard controls."""

    WINDOW_NAME = "VIDEX Text Recognition - LEFT Camera"
    HEADER_HEIGHT = 88

    def __init__(self) -> None:
        self._opened = False

    def open(self) -> None:
        try:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, 960, 720)
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV Preview 창을 열 수 없습니다. 데스크톱 환경을 확인하세요."
            ) from exc
        self._opened = True

    def _canvas(self, frame: PreviewFrame | None, status: str) -> np.ndarray:
        if frame is None:
            image = np.full((480, 640, 3), 24, dtype=np.uint8)
            frame_text = "Waiting for LEFT camera frame"
        else:
            image = decode_image(frame.image).copy()
            height, width = image.shape[:2]
            frame_text = f"LEFT frame={frame.frame_id}  {width}x{height}"
        canvas = cv2.copyMakeBorder(
            image,
            self.HEADER_HEIGHT,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(24, 24, 24),
        )
        cv2.putText(
            canvas,
            "SPACE: capture + recognize   S: save only   Q/ESC: quit",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            frame_text,
            (16, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (100, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            status[:100],
            (16, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (100, 255, 130),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def show(self, frame: PreviewFrame | None, status: str) -> str | None:
        if not self._opened:
            self.open()
        try:
            cv2.imshow(self.WINDOW_NAME, self._canvas(frame, status))
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
        except cv2.error as exc:
            raise RuntimeError("OpenCV Preview 화면을 갱신할 수 없습니다.") from exc
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key == 32:
            return "capture"
        if key in (ord("s"), ord("S")):
            return "save"
        return None

    def close(self) -> None:
        if not self._opened:
            return
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except cv2.error:
            pass
        self._opened = False


@dataclass(frozen=True)
class RecognitionOutcome:
    capture: SavedCapture
    result: dict[str, str] | None = None
    error: Exception | None = None


class PreviewRecognitionWorker:
    """Run at most one billable Vision request without blocking the GUI."""

    def __init__(self, vision_client: VisionClient) -> None:
        self.vision_client = vision_client
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._outcome: RecognitionOutcome | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(
                self._outcome is not None
                or (self._thread is not None and self._thread.is_alive())
            )

    def submit(self, capture: SavedCapture) -> bool:
        with self._lock:
            if self._outcome is not None or (
                self._thread is not None and self._thread.is_alive()
            ):
                return False
            self._thread = threading.Thread(
                target=self._recognize,
                args=(capture,),
                name="videx-text-recognition",
                daemon=True,
            )
            self._thread.start()
            return True

    def _recognize(self, capture: SavedCapture) -> None:
        try:
            result = self.vision_client.recognize(capture.jpeg_bytes)
            outcome = RecognitionOutcome(capture, result={"text": result["text"]})
        except Exception as exc:  # Delivered to the GUI thread without closing Preview.
            outcome = RecognitionOutcome(capture, error=exc)
        with self._lock:
            self._outcome = outcome

    def poll(self) -> RecognitionOutcome | None:
        with self._lock:
            outcome, self._outcome = self._outcome, None
            if outcome is not None and self._thread is not None:
                self._thread.join(timeout=0)
                self._thread = None
            return outcome

    def close(self) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


def _write_result(result: dict[str, str]) -> None:
    print(json.dumps(result, ensure_ascii=False), flush=True)


def _write_event(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class TextRecognitionPreview:
    """Coordinate streaming, keyboard commands, capture storage, and Vision."""

    def __init__(
        self,
        source: PreviewSource,
        vision_client: VisionClient,
        capture_store: CaptureStore,
        *,
        ui: PreviewUI | None = None,
        result_writer: Callable[[dict[str, str]], None] = _write_result,
        event_writer: Callable[[str], None] = _write_event,
    ) -> None:
        self.source = source
        self.capture_store = capture_store
        self.ui = ui or OpenCvPreviewUI()
        self.worker = PreviewRecognitionWorker(vision_client)
        self.result_writer = result_writer
        self.event_writer = event_writer
        self.latest_frame: PreviewFrame | None = None
        self.status = "Waiting for LEFT camera"

    def _consume_outcome(self) -> None:
        outcome = self.worker.poll()
        if outcome is None:
            return
        if outcome.error is not None:
            self.status = "Recognition failed - ready to retry"
            self.event_writer(f"글씨 인식 실패: {outcome.error}")
            return
        if outcome.result is None:
            self.status = "Recognition failed - ready to retry"
            self.event_writer("글씨 인식 실패: 결과가 없습니다.")
            return
        self.status = "Recognition complete - ready for next capture"
        self.result_writer(outcome.result)

    def _save_latest(self) -> SavedCapture:
        if self.latest_frame is None:
            raise ValueError("저장할 LEFT Preview 프레임이 아직 없습니다.")
        capture = self.capture_store.save_image(self.latest_frame.image)
        self.event_writer(
            f"촬영 이미지 저장: {capture.path} ({capture.width}x{capture.height}, "
            f"{len(capture.jpeg_bytes)} bytes)"
        )
        return capture

    def _handle_command(self, command: str | None) -> bool:
        if command == "quit":
            return False
        if command == "save":
            try:
                capture = self._save_latest()
                self.status = f"Saved {capture.path.name}"
            except (TextRecognitionError, OSError, ValueError) as exc:
                self.status = "Save failed"
                self.event_writer(f"이미지 저장 실패: {exc}")
            return True
        if command == "capture":
            if self.worker.busy:
                self.status = "Recognition already in progress"
                self.event_writer("이전 글씨 인식 요청이 처리 중이므로 촬영을 건너뜁니다.")
                return True
            try:
                capture = self._save_latest()
                if not self.worker.submit(capture):
                    self.status = "Recognition already in progress"
                    self.event_writer("이전 글씨 인식 요청이 처리 중입니다.")
                else:
                    self.status = f"Recognizing {capture.path.name}"
            except (TextRecognitionError, OSError, ValueError) as exc:
                self.status = "Capture failed"
                self.event_writer(f"촬영 실패: {exc}")
            return True
        return True

    def run(self) -> None:
        """Run until Q/Esc/window-close while always releasing camera and GUI."""

        source_opened = False
        ui_opened = False
        try:
            self.source.open()
            source_opened = True
            self.ui.open()
            ui_opened = True
            while True:
                frames = self.source.poll()
                if frames:
                    self.latest_frame = frames[-1]
                    if not self.worker.busy:
                        self.status = "Ready - SPACE to recognize, S to save"
                self._consume_outcome()
                command = self.ui.show(self.latest_frame, self.status)
                if not self._handle_command(command):
                    break
        finally:
            self.worker.close()
            if ui_opened:
                self.ui.close()
            if source_opened:
                self.source.close()

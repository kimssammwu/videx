"""Dual-camera quality tracking, selection, and integrated OpenCV preview."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .image_ops import central_roi, decode_image, normalized_difference
from .models import FramePacket, RecognitionState
from .recognizer import ObjectRecognizer

LOGGER = logging.getLogger("videx.object_recognition.dual")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class DualCameraConfig:
    """Thresholds for independent per-camera analysis and cross-camera choice."""

    roi_fraction: float = 1.0
    analysis_width: int = 160
    analysis_height: int = 120
    stable_duration: float = 1.0
    motion_threshold: float = 0.025
    rotation_threshold: float = 0.10
    strong_rotation_threshold: float = 0.14
    min_sharpness: float = 25.0
    min_brightness: float = 22.0
    max_brightness: float = 238.0
    min_presence: float = 0.04
    stale_after: float = 2.0
    compare_window: float = 0.40
    switch_quality_margin: float = 1.20

    def __post_init__(self) -> None:
        if not 0 < self.roi_fraction <= 1:
            raise ValueError("roi_fraction must be in (0, 1]")
        if self.analysis_width < 16 or self.analysis_height < 16:
            raise ValueError("analysis size must be at least 16x16")
        if self.stable_duration < 0 or self.stale_after <= 0 or self.compare_window < 0:
            raise ValueError("time thresholds are invalid")
        if not 0 < self.motion_threshold <= 1:
            raise ValueError("motion_threshold must be in (0, 1]")
        if not 0 <= self.rotation_threshold <= self.strong_rotation_threshold <= 1:
            raise ValueError("rotation thresholds are invalid")
        if self.switch_quality_margin < 1:
            raise ValueError("switch_quality_margin must be at least 1")


@dataclass(frozen=True)
class QualityMetrics:
    camera_id: int
    frame_id: str | int | None
    received_at: float
    motion: float | None
    sharpness: float
    brightness: float
    presence: float
    stability: float
    quality: float
    stable: bool
    rotated: bool
    rotation_difference: float | None
    status: str


@dataclass(frozen=True)
class FrameCandidate:
    camera_id: int
    frame_id: str | int | None
    received_at: float
    image: np.ndarray
    metrics: QualityMetrics

    @property
    def quality(self) -> float:
        return self.metrics.quality


def normalized_analysis_gray(image: np.ndarray, config: DualCameraConfig) -> np.ndarray:
    """Return a fixed-size ROI so resolution does not bias camera comparison."""

    roi = central_roi(image, config.roi_fraction)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    interpolation = (
        cv2.INTER_AREA
        if gray.shape[1] > config.analysis_width or gray.shape[0] > config.analysis_height
        else cv2.INTER_CUBIC
    )
    return cv2.resize(
        gray,
        (config.analysis_width, config.analysis_height),
        interpolation=interpolation,
    )


def _presence_score(gray: np.ndarray) -> float:
    """Estimate whether structured content occupies the center (not object semantics)."""

    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.nonzero(edges)
    if xs.size < 8:
        return 0.0
    height, width = gray.shape
    box_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    coverage = box_area / float(width * height)
    edge_density = xs.size / float(width * height)
    center_x = (float(xs.min()) + float(xs.max())) / 2.0
    center_y = (float(ys.min()) + float(ys.max())) / 2.0
    offset = math.hypot(
        (center_x - width / 2.0) / max(width / 2.0, 1.0),
        (center_y - height / 2.0) / max(height / 2.0, 1.0),
    )
    centered = max(0.0, 1.0 - offset)
    return float(
        centered
        * min(1.0, coverage / 0.25)
        * min(1.0, edge_density / 0.035)
    )


def analyze_quality(
    image: np.ndarray,
    config: DualCameraConfig,
    *,
    motion: float | None = None,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """Return fixed gray, sharpness, brightness, presence, stability, quality."""

    gray = normalized_analysis_gray(image, config)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    presence = _presence_score(gray)
    stability = 0.0 if motion is None else max(0.0, 1.0 - motion / config.motion_threshold)
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    quality = sharpness * (0.65 + 0.20 * exposure + 0.10 * presence + 0.05 * stability)
    return gray, sharpness, brightness, presence, stability, quality


class CameraQualityTracker:
    """Keep motion, stabilization, and rotation history for exactly one camera."""

    def __init__(self, camera_id: int, config: DualCameraConfig) -> None:
        self.camera_id = camera_id
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.baseline_gray: np.ndarray | None = None
        self.stable_since: float | None = None
        self.rotated = False
        self.max_rotation_difference = 0.0
        self.latest_image: np.ndarray | None = None
        self.latest_received_at: float | None = None
        self.latest_frame_id: str | int | None = None
        self.metrics: QualityMetrics | None = None
        self.best_candidate: FrameCandidate | None = None

    def begin_back(self) -> None:
        """Snapshot this camera's own view as its rotation baseline."""

        self.baseline_gray = None if self.previous_gray is None else self.previous_gray.copy()
        self.previous_gray = None
        self.stable_since = None
        self.rotated = False
        self.max_rotation_difference = 0.0
        self.best_candidate = None

    def update(self, packet: FramePacket, phase: str) -> QualityMetrics:
        if str(packet.camera_id) != str(self.camera_id):
            raise ValueError("packet camera does not match tracker")
        now = time.monotonic() if packet.timestamp is None else packet.timestamp
        image = decode_image(packet.data)
        gray = normalized_analysis_gray(image, self.config)
        motion = (
            normalized_difference(self.previous_gray, gray)
            if self.previous_gray is not None
            else None
        )
        _, sharpness, brightness, presence, stability, quality = analyze_quality(
            image, self.config, motion=motion
        )
        self.latest_image = image.copy()
        self.latest_received_at = now
        self.latest_frame_id = packet.frame_id

        rotation_difference: float | None = None
        if phase == "back" and self.baseline_gray is not None:
            rotation_difference = normalized_difference(self.baseline_gray, gray)
            self.max_rotation_difference = max(self.max_rotation_difference, rotation_difference)
            if not self.rotated and rotation_difference >= self.config.rotation_threshold:
                self.rotated = True
                self.stable_since = None
                self.best_candidate = None

        stable = motion is not None and motion <= self.config.motion_threshold
        if phase == "back" and not self.rotated:
            stable = False
        quality_ok = (
            sharpness >= self.config.min_sharpness
            and self.config.min_brightness <= brightness <= self.config.max_brightness
            and presence >= self.config.min_presence
        )

        if not stable:
            self.stable_since = None
            self.best_candidate = None
            status = "WAIT_ROTATION" if phase == "back" and not self.rotated else "MOVING"
        else:
            if self.stable_since is None:
                self.stable_since = now
            stable_for = max(0.0, now - self.stable_since)
            if not quality_ok:
                status = "LOW_QUALITY"
            elif stable_for + 1e-9 < self.config.stable_duration:
                status = "STABILIZING"
            else:
                status = "CANDIDATE"

        if phase == "back" and self.baseline_gray is None:
            status = "NO_BASELINE"
            stable = False

        metrics = QualityMetrics(
            self.camera_id,
            packet.frame_id,
            now,
            motion,
            sharpness,
            brightness,
            presence,
            stability,
            quality,
            stable,
            self.rotated,
            rotation_difference,
            status,
        )
        self.metrics = metrics
        self.previous_gray = gray
        LOGGER.info(
            "camera=%d frame=%s phase=%s sharpness=%.2f brightness=%.1f "
            "motion=%s presence=%.3f stable=%s rotated=%s status=%s",
            self.camera_id,
            packet.frame_id,
            phase,
            sharpness,
            brightness,
            "-" if motion is None else f"{motion:.4f}",
            presence,
            stable,
            self.rotated,
            status,
        )

        if status == "CANDIDATE":
            candidate = FrameCandidate(
                self.camera_id,
                packet.frame_id,
                now,
                image.copy(),
                metrics,
            )
            if (
                self.best_candidate is None
                or now - self.best_candidate.received_at > self.config.stale_after
                or candidate.quality >= self.best_candidate.quality
            ):
                self.best_candidate = candidate
        return metrics

    def fresh_candidate(self, now: float) -> FrameCandidate | None:
        candidate = self.best_candidate
        if candidate is None or now - candidate.received_at > self.config.stale_after:
            return None
        return candidate

    def manual_candidate(self, now: float) -> FrameCandidate | None:
        if (
            self.latest_image is None
            or self.latest_received_at is None
            or now - self.latest_received_at > self.config.stale_after
            or self.metrics is None
        ):
            return None
        return FrameCandidate(
            self.camera_id,
            self.latest_frame_id,
            self.latest_received_at,
            self.latest_image.copy(),
            self.metrics,
        )


def select_front_candidate(
    candidates: list[FrameCandidate], now: float, stale_after: float
) -> FrameCandidate | None:
    fresh = [item for item in candidates if now - item.received_at <= stale_after]
    return max(fresh, key=lambda item: item.quality, default=None)


def select_back_candidate(
    candidates: list[FrameCandidate],
    *,
    front_camera_id: int,
    now: float,
    config: DualCameraConfig,
) -> tuple[FrameCandidate | None, bool]:
    """Prefer the front camera unless a stronger, well-supported alternative exists."""

    fresh = {
        item.camera_id: item
        for item in candidates
        if now - item.received_at <= config.stale_after
    }
    preferred = fresh.get(front_camera_id)
    others = [item for camera_id, item in fresh.items() if camera_id != front_camera_id]
    alternate = max(others, key=lambda item: item.quality, default=None)
    alternate_is_supported = bool(
        alternate
        and alternate.metrics.rotation_difference is not None
        and alternate.metrics.rotation_difference >= config.strong_rotation_threshold
    )
    if preferred is not None:
        if (
            alternate_is_supported
            and alternate is not None
            and alternate.quality >= preferred.quality * config.switch_quality_margin
        ):
            return alternate, False
        return preferred, False
    if alternate_is_supported:
        return alternate, False
    return None, alternate is not None


class DualCameraCoordinator:
    """Coordinate two independent trackers with the existing recognizer pipeline."""

    def __init__(
        self,
        recognizer: ObjectRecognizer,
        config: DualCameraConfig | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.config = config or DualCameraConfig()
        self.trackers = {
            0: CameraQualityTracker(0, self.config),
            1: CameraQualityTracker(1, self.config),
        }
        self.phase = "front"
        self.front_camera_id: int | None = None
        self.back_camera_id: int | None = None
        self.manual_confirmation_required = False
        self._first_candidate_at: float | None = None
        self._worker: threading.Thread | None = None
        self._worker_result: tuple[FrameCandidate, object] | None = None
        self._worker_error: BaseException | None = None
        self._worker_lock = threading.Lock()

    def process_packet(self, packet: FramePacket) -> QualityMetrics:
        camera_id = int(packet.camera_id)
        if camera_id not in self.trackers:
            raise ValueError(f"지원하지 않는 카메라 ID입니다: {packet.camera_id}")
        tracking_phase = self.phase if self.phase in ("front", "back") else "display"
        metrics = self.trackers[camera_id].update(packet, tracking_phase)
        if self.phase in ("front", "back"):
            self._consider_automatic(metrics.received_at)
        return metrics

    def _fresh_candidates(self, now: float) -> list[FrameCandidate]:
        return [
            candidate
            for tracker in self.trackers.values()
            if (candidate := tracker.fresh_candidate(now)) is not None
        ]

    def _consider_automatic(self, now: float) -> None:
        candidates = self._fresh_candidates(now)
        if not candidates:
            self._first_candidate_at = None
            return
        if self._first_candidate_at is None:
            self._first_candidate_at = now
        both_ready = len({item.camera_id for item in candidates}) == 2
        if not both_ready and now - self._first_candidate_at < self.config.compare_window:
            return
        if self.phase == "front":
            selected = select_front_candidate(candidates, now, self.config.stale_after)
            if selected is not None:
                self._capture_front(selected)
            return
        if self.front_camera_id is None:
            return
        selected, needs_manual = select_back_candidate(
            candidates,
            front_camera_id=self.front_camera_id,
            now=now,
            config=self.config,
        )
        self.manual_confirmation_required = needs_manual
        if selected is not None:
            self._start_back_capture(selected)

    @staticmethod
    def _recognizer_packet(candidate: FrameCandidate) -> FramePacket:
        # Physical cameras intentionally share one logical recognizer session.
        return FramePacket(
            candidate.image,
            "dual",
            f"camera-{candidate.camera_id}:{candidate.frame_id}",
            candidate.received_at,
        )

    def _capture_front(self, candidate: FrameCandidate) -> None:
        metrics = self.recognizer.process_frame(
            self._recognizer_packet(candidate), manual_capture=True
        )
        if not metrics.selected:
            return
        self.front_camera_id = candidate.camera_id
        self.phase = "back"
        self._first_candidate_at = None
        self.manual_confirmation_required = False
        for tracker in self.trackers.values():
            tracker.begin_back()
        LOGGER.info("dual selected front camera=%d quality=%.2f", candidate.camera_id, candidate.quality)

    def _start_back_capture(self, candidate: FrameCandidate) -> None:
        if self._worker is not None:
            return
        self.phase = "recognizing"
        self.manual_confirmation_required = False

        def work() -> None:
            try:
                result = self.recognizer.process_frame(
                    self._recognizer_packet(candidate), manual_capture=True
                )
                with self._worker_lock:
                    self._worker_result = (candidate, result)
            except Exception as exc:  # surfaced safely by poll_background
                with self._worker_lock:
                    self._worker_error = exc

        self._worker = threading.Thread(target=work, name="videx-vision", daemon=True)
        self._worker.start()

    def poll_background(self) -> None:
        worker = self._worker
        if worker is None or worker.is_alive():
            return
        with self._worker_lock:
            error = self._worker_error
            result = self._worker_result
            self._worker = None
            self._worker_error = None
            self._worker_result = None
        if error is not None:
            self.phase = "error"
            LOGGER.error("백그라운드 인식 실패: %s", error)
            return
        if result is None:
            self.phase = "error"
            return
        candidate, metrics = result
        if getattr(metrics, "selected", False):
            self.back_camera_id = candidate.camera_id
            self.phase = (
                "completed"
                if self.recognizer.state is RecognitionState.COMPLETED
                else "error"
            )
            LOGGER.info(
                "dual selected back camera=%d quality=%.2f", candidate.camera_id, candidate.quality
            )
        else:
            self.phase = "back"
            self.manual_confirmation_required = True

    def tick(self, now: float | None = None) -> None:
        self.poll_background()
        if self.phase not in ("front", "back"):
            return
        current = time.monotonic() if now is None else now
        self._consider_automatic(current)
        if self.recognizer.tick():
            self.phase = "error"

    def manual_capture(self, camera_id: int | None = None, now: float | None = None) -> bool:
        if self.phase not in ("front", "back"):
            return False
        current = time.monotonic() if now is None else now
        candidates = [
            candidate
            for tracker in self.trackers.values()
            if (candidate := tracker.manual_candidate(current)) is not None
        ]
        if camera_id is not None:
            candidates = [item for item in candidates if item.camera_id == camera_id]
        if not candidates:
            return False
        if self.phase == "back" and camera_id is None and self.front_camera_id is not None:
            same_camera = [item for item in candidates if item.camera_id == self.front_camera_id]
            selected = max(same_camera or candidates, key=lambda item: item.quality)
        else:
            selected = max(candidates, key=lambda item: item.quality)
        if self.phase == "front":
            self._capture_front(selected)
        else:
            self._start_back_capture(selected)
        return True

    def reset(self) -> None:
        self.poll_background()
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("API 호출이 끝날 때까지 재시도할 수 없습니다.")
        self.recognizer.reset_recognition()
        self.recognizer.start_recognition("dual")
        for tracker in self.trackers.values():
            tracker.reset()
        self.phase = "front"
        self.front_camera_id = None
        self.back_camera_id = None
        self.manual_confirmation_required = False
        self._first_candidate_at = None

    @property
    def worker_active(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def state_dict(self) -> dict[str, object]:
        state = self.recognizer.get_recognition_state()
        state.update(
            {
                "dual_camera_phase": self.phase,
                "front_camera_id": self.front_camera_id,
                "back_camera_id": self.back_camera_id,
            }
        )
        return state


class DualCameraViewer:
    """Render both feeds using the same left/right panel concept as proxy/viewer.py."""

    WINDOW_NAME = "VIDEX Dual Camera Recognition"
    PANEL_WIDTH = 640
    HEADER_HEIGHT = 82
    IMAGE_HEIGHT = 480
    FOOTER_HEIGHT = 76

    def __init__(self, roi_fraction: float = 1.0) -> None:
        self.roi_fraction = roi_fraction
        self._opened = False

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.HEADER_HEIGHT + self.IMAGE_HEIGHT + self.FOOTER_HEIGHT, self.PANEL_WIDTH * 2

    def render(self, coordinator: DualCameraCoordinator, now: float | None = None) -> np.ndarray:
        current = time.monotonic() if now is None else now
        height, width = self.canvas_size
        canvas = np.full((height, width, 3), 22, dtype=np.uint8)
        for camera_id, label in ((0, "LEFT"), (1, "RIGHT")):
            self._render_panel(
                canvas,
                camera_id,
                label,
                coordinator.trackers[camera_id],
                coordinator,
                current,
            )
        cv2.line(canvas, (self.PANEL_WIDTH, 0), (self.PANEL_WIDTH, height), (90, 90, 90), 1)
        state = coordinator.recognizer.get_recognition_state()
        result = state.get("result") or "-"
        footer_y = self.HEADER_HEIGHT + self.IMAGE_HEIGHT + 27
        summary = (
            f"PHASE {coordinator.phase.upper()} | FRONT CAM {coordinator.front_camera_id} | "
            f"BACK CAM {coordinator.back_camera_id} | RESULT {result}"
        )
        cv2.putText(canvas, summary, (18, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 240, 240), 1, cv2.LINE_AA)
        hint = "C best manual | 0/1 camera manual | R retry | Q or ESC quit"
        if coordinator.manual_confirmation_required:
            hint = "CAMERA SWITCH NEEDS CONFIRMATION - press 0 or 1 | " + hint
        cv2.putText(canvas, hint, (18, footer_y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 205, 255), 1, cv2.LINE_AA)
        return canvas

    def _render_panel(
        self,
        canvas: np.ndarray,
        camera_id: int,
        label: str,
        tracker: CameraQualityTracker,
        coordinator: DualCameraCoordinator,
        now: float,
    ) -> None:
        x0 = camera_id * self.PANEL_WIDTH
        stale = (
            tracker.latest_received_at is None
            or now - tracker.latest_received_at > coordinator.config.stale_after
        )
        status = "NO SIGNAL" if tracker.metrics is None else tracker.metrics.status
        if stale and tracker.metrics is not None:
            status = "STALE"
        color = (0, 210, 0) if tracker.metrics and tracker.metrics.stable and not stale else (0, 170, 255)
        cv2.putText(canvas, f"{label} / CAMERA {camera_id}", (x0 + 14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if tracker.metrics is None:
            details = "sharp - | bright - | motion - | stable no"
        else:
            motion = "-" if tracker.metrics.motion is None else f"{tracker.metrics.motion:.3f}"
            details = (
                f"sharp {tracker.metrics.sharpness:.1f} | bright {tracker.metrics.brightness:.0f} | "
                f"motion {motion} | present {tracker.metrics.presence:.2f}"
            )
        cv2.putText(canvas, details, (x0 + 14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (205, 205, 205), 1, cv2.LINE_AA)
        cv2.putText(canvas, status, (x0 + 14, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.49, color, 1, cv2.LINE_AA)

        if tracker.latest_image is None:
            cv2.putText(canvas, "WAITING FOR CAM2 JPEG", (x0 + 172, self.HEADER_HEIGHT + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (130, 130, 130), 1, cv2.LINE_AA)
            return
        image = tracker.latest_image
        scale = min(self.PANEL_WIDTH / image.shape[1], self.IMAGE_HEIGHT / image.shape[0])
        display_width = max(1, round(image.shape[1] * scale))
        display_height = max(1, round(image.shape[0] * scale))
        resized = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        left = x0 + (self.PANEL_WIDTH - display_width) // 2
        top = self.HEADER_HEIGHT + (self.IMAGE_HEIGHT - display_height) // 2
        canvas[top : top + display_height, left : left + display_width] = resized

        if self.roi_fraction < 0.999:
            roi_width = round(display_width * self.roi_fraction)
            roi_height = round(display_height * self.roi_fraction)
            roi_left = left + (display_width - roi_width) // 2
            roi_top = top + (display_height - roi_height) // 2
            cv2.rectangle(
                canvas,
                (roi_left, roi_top),
                (roi_left + roi_width, roi_top + roi_height),
                (0, 255, 255),
                2,
            )
        selected = []
        if coordinator.front_camera_id == camera_id:
            selected.append("FRONT")
        if coordinator.back_camera_id == camera_id:
            selected.append("BACK")
        if selected:
            cv2.rectangle(canvas, (x0 + 2, 2), (x0 + self.PANEL_WIDTH - 3, self.HEADER_HEIGHT + self.IMAGE_HEIGHT - 2), (0, 255, 0), 3)
            cv2.putText(canvas, "SELECTED " + "+".join(selected), (x0 + 430, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    def open(self) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.PANEL_WIDTH * 2, self.canvas_size[0])
        self._opened = True

    def show(self, canvas: np.ndarray) -> str | None:
        if not self._opened:
            self.open()
        cv2.imshow(self.WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "q"
        if key in (ord("q"), 27):
            return "q"
        if key in (ord("c"), ord("r"), ord("0"), ord("1")):
            return chr(key)
        return None

    def close(self) -> None:
        if self._opened:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except cv2.error:
                pass
            self._opened = False

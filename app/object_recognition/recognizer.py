"""Time-based front/back automatic capture state machine."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .api import CachingVisionClient, MockVisionClient, VisionApiError, VisionClient
from .image_ops import (
    analysis_gray,
    brightness_score,
    decode_image,
    encode_jpeg,
    merge_side_by_side,
    normalized_difference,
    save_jpeg,
    sharpness_score,
)
from .models import FrameMetrics, FramePacket, RecognitionConfig, RecognitionState

LOGGER = logging.getLogger("videx.object_recognition")
LOGGER.addHandler(logging.NullHandler())


@dataclass
class _Candidate:
    image: np.ndarray
    sharpness: float
    brightness: float
    frame_id: str | int | None


class ObjectRecognizer:
    """Select two stable sharp frames, merge them, and recognize one product name."""

    def __init__(
        self,
        config: RecognitionConfig | None = None,
        vision_client: VisionClient | None = None,
        announce: Callable[[str], None] = print,
    ) -> None:
        self.config = config or RecognitionConfig()
        default_output = Path(__file__).resolve().parents[1] / "test_output"
        self.output_dir = Path(self.config.output_dir or default_output)
        self.vision_client = CachingVisionClient(vision_client or MockVisionClient())
        self.announce = announce
        self.state = RecognitionState.IDLE
        self.state_history: list[RecognitionState] = [self.state]
        self._clear_session()

    def _clear_session(self) -> None:
        self.camera_id: str | None = None
        self.previous_gray: np.ndarray | None = None
        self.front_gray: np.ndarray | None = None
        self.last_check_at: float | None = None
        self.stable_since: float | None = None
        self.phase_started_at: float | None = None
        self.candidates: list[_Candidate] = []
        self.last_valid_candidate: _Candidate | None = None
        self.front_path: Path | None = None
        self.back_path: Path | None = None
        self.merged_path: Path | None = None
        self.front_sharpness: float | None = None
        self.back_sharpness: float | None = None
        self.result: str | None = None
        self.error: str | None = None
        self._session_tag = ""

    def _set_state(self, state: RecognitionState, now: float | None = None) -> None:
        self.state = state
        self.state_history.append(state)
        if now is not None:
            # Capture timestamps may be wall-clock, stream-relative, or monotonic.
            # Timeouts deliberately use the local monotonic receipt clock instead.
            self.phase_started_at = time.monotonic()
        LOGGER.info("state=%s", state.value)

    def start_recognition(self, camera_id: str | None = None) -> dict[str, object]:
        if self.state not in (RecognitionState.IDLE, RecognitionState.COMPLETED, RecognitionState.ERROR):
            raise RuntimeError("이미 인식 작업이 진행 중입니다.")
        self._clear_session()
        self.camera_id = camera_id
        now = time.monotonic()
        self._session_tag = f"{int(time.time() * 1000)}"
        self._set_state(RecognitionState.WAIT_FRONT, now)
        self.announce("제품 앞면을 보여주세요.")
        return self.get_recognition_state()

    def stop_recognition(self) -> dict[str, object]:
        self._clear_session()
        self._set_state(RecognitionState.IDLE)
        return self.get_recognition_state()

    def reset_recognition(self) -> dict[str, object]:
        return self.stop_recognition()

    def retry_recognition(self) -> dict[str, object]:
        """Explicit retry after a capture/API error; never loops automatically."""

        if self.state is not RecognitionState.ERROR:
            raise RuntimeError("ERROR 상태에서만 재시도할 수 있습니다.")
        self.error = None
        self.result = None
        self.previous_gray = None
        self.stable_since = None
        self.candidates.clear()
        now = time.monotonic()
        if self.front_path is None:
            self._set_state(RecognitionState.WAIT_FRONT, now)
            self.announce("촬영에 실패했습니다. 다시 보여주세요.")
            self.announce("제품 앞면을 보여주세요.")
        elif self.back_path is None:
            self._set_state(RecognitionState.WAIT_ROTATION, now)
            self.announce("촬영에 실패했습니다. 다시 보여주세요.")
            self.announce("앞면 촬영이 완료되었습니다. 제품을 뒤집어 주세요.")
        else:
            self._finish_pipeline(now)
        return self.get_recognition_state()

    def get_recognition_state(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "camera_id": self.camera_id,
            "front_path": str(self.front_path) if self.front_path else None,
            "back_path": str(self.back_path) if self.back_path else None,
            "merged_path": str(self.merged_path) if self.merged_path else None,
            "front_sharpness": self.front_sharpness,
            "back_sharpness": self.back_sharpness,
            "result": self.result,
            "error": self.error,
        }

    def tick(self, timestamp: float | None = None) -> bool:
        """Check a no-frame timeout; call this while an input stream is idle."""

        if self.state in (RecognitionState.IDLE, RecognitionState.COMPLETED, RecognitionState.ERROR):
            return False
        now = time.monotonic() if timestamp is None else timestamp
        if self.phase_started_at is not None and now - self.phase_started_at > self.config.phase_timeout:
            self._fail("제한 시간 안에 적절한 이미지를 확보하지 못했습니다.")
            return True
        return False

    def process_frame(
        self,
        frame: FramePacket | bytes | np.ndarray,
        *,
        camera_id: str = "default",
        frame_id: str | int | None = None,
        timestamp: float | None = None,
        manual_capture: bool = False,
    ) -> FrameMetrics:
        if self.state in (RecognitionState.IDLE, RecognitionState.COMPLETED, RecognitionState.ERROR):
            raise RuntimeError(f"{self.state.value} 상태에서는 프레임을 처리할 수 없습니다.")
        if isinstance(frame, FramePacket):
            packet = frame
        else:
            packet = FramePacket(frame, camera_id, frame_id, timestamp)
        now = time.monotonic() if packet.timestamp is None else packet.timestamp
        if self.tick():
            return self._metrics(packet.frame_id, 0.0, 0.0, None, None, False, False, "timeout")
        if self.camera_id is None:
            self.camera_id = packet.camera_id
        if packet.camera_id != self.camera_id:
            return self._metrics(packet.frame_id, 0.0, 0.0, None, None, False, False, "camera_mismatch")
        image = decode_image(packet.data)
        gray = analysis_gray(image, self.config.roi_fraction, self.config.analysis_width)
        sharpness = sharpness_score(image, self.config.roi_fraction)
        brightness = brightness_score(image, self.config.roi_fraction)
        quality_ok = (
            sharpness >= self.config.min_sharpness
            and self.config.min_brightness <= brightness <= self.config.max_brightness
        )
        side_difference = (
            normalized_difference(self.front_gray, gray) if self.front_gray is not None else None
        )

        if self.state is RecognitionState.FRONT_CAPTURED:
            return self._metrics(
                packet.frame_id,
                sharpness,
                brightness,
                None,
                side_difference,
                False,
                False,
                "waiting_guidance_completion",
            )

        if manual_capture:
            selected, reason = self._manual_capture(
                image, gray, sharpness, brightness, packet.frame_id, now, quality_ok, side_difference
            )
            return self._metrics(
                packet.frame_id, sharpness, brightness, None, side_difference, False, selected, reason
            )

        if self.state is RecognitionState.WAIT_ROTATION:
            if side_difference is not None and side_difference >= self.config.rotation_threshold:
                self.previous_gray = gray
                self.last_check_at = now
                self.stable_since = None
                self.candidates.clear()
                self._set_state(RecognitionState.WAIT_BACK, now)
                reason = "rotation_detected"
            else:
                reason = "waiting_rotation"
            return self._metrics(
                packet.frame_id, sharpness, brightness, None, side_difference, False, False, reason
            )

        if self.last_check_at is not None and now - self.last_check_at < self.config.frame_check_interval:
            return self._metrics(
                packet.frame_id,
                sharpness,
                brightness,
                None,
                side_difference,
                self.stable_since is not None,
                False,
                "check_interval",
            )

        motion = normalized_difference(self.previous_gray, gray) if self.previous_gray is not None else None
        self.previous_gray = gray
        self.last_check_at = now
        stable = motion is not None and motion <= self.config.motion_threshold
        if not stable:
            self.stable_since = None
            self.candidates.clear()
            reason = "moving" if motion is not None else "first_frame"
        else:
            if self.stable_since is None:
                self.stable_since = now
                self.candidates.clear()
            if quality_ok and (
                self.state is not RecognitionState.WAIT_BACK
                or side_difference is None
                or side_difference >= self.config.duplicate_threshold
            ):
                candidate = _Candidate(image.copy(), sharpness, brightness, packet.frame_id)
                self.last_valid_candidate = candidate
                self.candidates.append(candidate)
                self.candidates.sort(key=lambda item: item.sharpness, reverse=True)
                del self.candidates[self.config.max_candidates :]
                reason = "candidate"
            elif not quality_ok:
                reason = "quality_rejected"
            else:
                reason = "duplicate_rejected"

            if now - self.stable_since >= self.config.stable_duration and self.candidates:
                best = self.candidates[0]
                if self.state is RecognitionState.WAIT_FRONT:
                    self._select_front(best, now)
                elif self.state is RecognitionState.WAIT_BACK:
                    self._select_back(best, now)
                return self._metrics(
                    packet.frame_id,
                    sharpness,
                    brightness,
                    motion,
                    side_difference,
                    True,
                    True,
                    "selected",
                )

        return self._metrics(
            packet.frame_id, sharpness, brightness, motion, side_difference, stable, False, reason
        )

    def _manual_capture(
        self,
        image: np.ndarray,
        gray: np.ndarray,
        sharpness: float,
        brightness: float,
        frame_id: str | int | None,
        now: float,
        quality_ok: bool,
        side_difference: float | None,
    ) -> tuple[bool, str]:
        # Manual confirmation is the fallback when automatic quality selection is
        # impossible. Duplicate-side protection remains active below.
        candidate = _Candidate(image.copy(), sharpness, brightness, frame_id)
        if self.state is RecognitionState.WAIT_FRONT:
            self._select_front(candidate, now)
            return True, "manual_front_selected"
        if self.state in (RecognitionState.WAIT_ROTATION, RecognitionState.WAIT_BACK):
            if side_difference is None or side_difference < self.config.duplicate_threshold:
                return False, "duplicate_rejected"
            if self.state is RecognitionState.WAIT_ROTATION:
                self._set_state(RecognitionState.WAIT_BACK, now)
            self._select_back(candidate, now)
            return True, "manual_back_selected"
        return False, "manual_capture_not_available"

    def _select_front(self, candidate: _Candidate, now: float) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.front_path = save_jpeg(
            self.output_dir / f"{self._session_tag}_front.jpg",
            candidate.image,
            self.config.jpeg_quality,
        )
        self.front_sharpness = candidate.sharpness
        self.front_gray = analysis_gray(
            candidate.image, self.config.roi_fraction, self.config.analysis_width
        )
        self._set_state(RecognitionState.FRONT_CAPTURED, now)
        LOGGER.info("selected=front sharpness=%.2f path=%s", candidate.sharpness, self.front_path)
        self.announce("앞면 촬영이 완료되었습니다. 제품을 뒤집어 주세요.")
        # Console output is synchronous by default. Set the config flag to False
        # when an async caller must acknowledge a TTS completion event itself.
        if self.config.auto_acknowledge_guidance:
            self.acknowledge_rotation_guidance(now)

    def acknowledge_rotation_guidance(self, timestamp: float | None = None) -> None:
        """Signal that the flip instruction finished and rotation checks may start."""

        if self.state is not RecognitionState.FRONT_CAPTURED:
            raise RuntimeError("앞면 촬영 안내 완료를 기다리는 상태가 아닙니다.")
        now = time.monotonic() if timestamp is None else timestamp
        self.previous_gray = None
        self.stable_since = None
        self.candidates.clear()
        self._set_state(RecognitionState.WAIT_ROTATION, now)

    def _select_back(self, candidate: _Candidate, now: float) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.back_path = save_jpeg(
            self.output_dir / f"{self._session_tag}_back.jpg",
            candidate.image,
            self.config.jpeg_quality,
        )
        self.back_sharpness = candidate.sharpness
        self._set_state(RecognitionState.BACK_CAPTURED, now)
        LOGGER.info("selected=back sharpness=%.2f path=%s", candidate.sharpness, self.back_path)
        self.announce("뒷면 촬영이 완료되었습니다.")
        self._finish_pipeline(now)

    def _finish_pipeline(self, now: float) -> None:
        try:
            if self.front_path is None or self.back_path is None:
                raise ValueError("앞면 또는 뒷면 이미지가 없습니다.")
            self._set_state(RecognitionState.MERGING, now)
            front = decode_image(self.front_path.read_bytes())
            back = decode_image(self.back_path.read_bytes())
            merged = merge_side_by_side(
                front,
                back,
                self.config.merge_max_height,
                self.config.merge_max_width,
            )
            self.merged_path = save_jpeg(
                self.output_dir / f"{self._session_tag}_merged.jpg",
                merged,
                self.config.jpeg_quality,
            )
            merged_jpeg = encode_jpeg(merged, self.config.jpeg_quality)
            self._set_state(RecognitionState.RECOGNIZING, now)
            self.announce("제품을 인식하고 있습니다.")
            product_name = self.vision_client.recognize(merged_jpeg)
            if product_name == "인식 실패":
                raise VisionApiError("제품을 식별할 수 없습니다.")
            self.result = product_name
            self._set_state(RecognitionState.COMPLETED, now)
            self.announce(f"인식된 제품은 {product_name}입니다.")
            LOGGER.info("result=%s merged_path=%s", product_name, self.merged_path)
        except (OSError, ValueError, VisionApiError) as exc:
            self._fail(str(exc))

    def _fail(self, message: str) -> None:
        self.error = message
        self.result = "인식 실패"
        self._set_state(RecognitionState.ERROR)
        if "식별" in message:
            self.announce("제품을 인식하지 못했습니다.")
        else:
            self.announce("촬영에 실패했습니다. 다시 보여주세요.")
        LOGGER.error("error=%s", message)

    def _metrics(
        self,
        frame_id: str | int | None,
        sharpness: float,
        brightness: float,
        motion: float | None,
        side_difference: float | None,
        stable: bool,
        selected: bool,
        reason: str,
    ) -> FrameMetrics:
        metrics = FrameMetrics(
            frame_id,
            self.state,
            motion,
            sharpness,
            brightness,
            side_difference,
            stable,
            selected,
            reason,
        )
        LOGGER.info(
            "frame=%s state=%s motion=%s sharpness=%.2f brightness=%.2f "
            "side_difference=%s stable=%s selected=%s reason=%s",
            frame_id,
            self.state.value,
            "-" if motion is None else f"{motion:.4f}",
            sharpness,
            brightness,
            "-" if side_difference is None else f"{side_difference:.4f}",
            stable,
            selected,
            reason,
        )
        return metrics

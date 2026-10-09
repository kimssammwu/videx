from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from object_recognition.api import MockVisionClient
from object_recognition.dual_camera import (
    CameraQualityTracker,
    DualCameraConfig,
    DualCameraCoordinator,
    DualCameraViewer,
    FrameCandidate,
    QualityMetrics,
    analyze_quality,
    select_back_candidate,
    select_front_candidate,
)
from object_recognition.models import FramePacket, RecognitionConfig, RecognitionState
from object_recognition.recognizer import ObjectRecognizer


def product_image(*, inverted: bool = False, blur: int = 0, width: int = 320) -> np.ndarray:
    height = width * 3 // 4
    background = 210 if not inverted else 42
    foreground = 25 if not inverted else 235
    image = np.full((height, width, 3), background, dtype=np.uint8)
    cv2.rectangle(
        image,
        (width * 3 // 10, height // 8),
        (width * 7 // 10, height * 7 // 8),
        (foreground,) * 3,
        max(2, width // 50),
    )
    for y in range(height // 4, height * 4 // 5, max(5, height // 10)):
        cv2.line(
            image,
            (width * 7 // 20, y),
            (width * 13 // 20, y),
            (foreground,) * 3,
            max(1, width // 100),
        )
    if blur:
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    return image


def metrics(
    camera_id: int,
    *,
    received_at: float,
    quality: float,
    rotation: float = 0.2,
) -> QualityMetrics:
    return QualityMetrics(
        camera_id=camera_id,
        frame_id=camera_id,
        received_at=received_at,
        motion=0.0,
        sharpness=quality,
        brightness=128.0,
        presence=1.0,
        stability=1.0,
        quality=quality,
        stable=True,
        rotated=True,
        rotation_difference=rotation,
        status="CANDIDATE",
    )


def candidate(
    camera_id: int,
    *,
    received_at: float,
    quality: float,
    rotation: float = 0.2,
) -> FrameCandidate:
    return FrameCandidate(
        camera_id,
        camera_id,
        received_at,
        product_image(inverted=bool(camera_id)),
        metrics(camera_id, received_at=received_at, quality=quality, rotation=rotation),
    )


class DualCameraQualityTests(unittest.TestCase):
    def test_sharpness_is_normalized_to_same_analysis_size(self) -> None:
        config = DualCameraConfig()
        source = product_image(width=640)
        low_resolution = cv2.resize(source, (320, 240), interpolation=cv2.INTER_AREA)
        high_score = analyze_quality(source, config)[1]
        low_score = analyze_quality(low_resolution, config)[1]
        ratio = max(high_score, low_score) / min(high_score, low_score)
        self.assertLess(ratio, 1.35)

    def test_better_fresh_front_candidate_is_selected_and_stale_is_excluded(self) -> None:
        sharp = candidate(1, received_at=10.0, quality=300.0)
        soft = candidate(0, received_at=10.0, quality=100.0)
        self.assertEqual(select_front_candidate([soft, sharp], 10.1, 2.0).camera_id, 1)
        self.assertEqual(select_front_candidate([soft, sharp], 12.1, 2.0), None)

    def test_back_prefers_same_camera_without_strong_switch_evidence(self) -> None:
        config = DualCameraConfig(switch_quality_margin=1.2, strong_rotation_threshold=0.14)
        preferred = candidate(0, received_at=10.0, quality=100.0)
        weak_other = candidate(1, received_at=10.0, quality=300.0, rotation=0.11)
        selected, needs_manual = select_back_candidate(
            [preferred, weak_other], front_camera_id=0, now=10.1, config=config
        )
        self.assertEqual(selected.camera_id, 0)
        self.assertFalse(needs_manual)

        selected, needs_manual = select_back_candidate(
            [weak_other], front_camera_id=0, now=10.1, config=config
        )
        self.assertIsNone(selected)
        self.assertTrue(needs_manual)

        strong_other = candidate(1, received_at=10.0, quality=300.0, rotation=0.20)
        selected, needs_manual = select_back_candidate(
            [preferred, strong_other], front_camera_id=0, now=10.1, config=config
        )
        self.assertEqual(selected.camera_id, 1)
        self.assertFalse(needs_manual)

    def test_each_camera_compares_rotation_only_with_its_own_baseline(self) -> None:
        config = DualCameraConfig(stable_duration=0, min_sharpness=1, min_presence=0)
        left = CameraQualityTracker(0, config)
        right = CameraQualityTracker(1, config)
        left_image = product_image(inverted=False)
        right_image = product_image(inverted=True)
        left.update(FramePacket(left_image, "0", 1, 1.0), "front")
        right.update(FramePacket(right_image, "1", 1, 1.0), "front")
        left.begin_back()
        right.begin_back()

        right_metrics = right.update(FramePacket(right_image, "1", 2, 1.1), "back")
        self.assertAlmostEqual(right_metrics.rotation_difference or 0.0, 0.0, places=5)
        self.assertFalse(right_metrics.rotated)
        self.assertEqual(right_metrics.status, "WAIT_ROTATION")

    def test_trackers_have_independent_stability_state(self) -> None:
        config = DualCameraConfig(stable_duration=0.2, min_sharpness=1, min_presence=0)
        left = CameraQualityTracker(0, config)
        right = CameraQualityTracker(1, config)
        still = product_image()
        moving_a = product_image(inverted=True)
        moving_b = np.roll(moving_a, 50, axis=1)
        for index, timestamp in enumerate((1.0, 1.1, 1.3)):
            left.update(FramePacket(still, "0", index, timestamp), "front")
        right.update(FramePacket(moving_a, "1", 1, 1.0), "front")
        right.update(FramePacket(moving_b, "1", 2, 1.3), "front")
        self.assertIsNotNone(left.fresh_candidate(1.3))
        self.assertIsNone(right.fresh_candidate(1.3))


class DualCameraIntegrationTests(unittest.TestCase):
    def _coordinator(
        self, directory: str, client: MockVisionClient | object
    ) -> DualCameraCoordinator:
        recognizer = ObjectRecognizer(
            RecognitionConfig(
                output_dir=Path(directory),
                min_sharpness=1,
                stable_duration=0,
                phase_timeout=5,
            ),
            client,  # type: ignore[arg-type]
            lambda _message: None,
        )
        recognizer.start_recognition("dual")
        return DualCameraCoordinator(
            recognizer,
            DualCameraConfig(
                stable_duration=0,
                compare_window=10,
                min_sharpness=1,
                min_presence=0,
            ),
        )

    def test_mock_pipeline_merges_once_and_records_physical_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mock = MockVisionClient("듀얼 카메라 테스트 제품")
            coordinator = self._coordinator(directory, mock)
            now = time.monotonic()
            coordinator.process_packet(FramePacket(product_image(), "0", 1, now))
            self.assertTrue(coordinator.manual_capture(0, now))
            coordinator.process_packet(
                FramePacket(product_image(inverted=True), "1", 2, now + 0.1)
            )
            self.assertTrue(coordinator.manual_capture(1, now + 0.1))
            assert coordinator._worker is not None
            coordinator._worker.join(timeout=2)
            coordinator.poll_background()

            state = coordinator.state_dict()
            self.assertEqual(state["state"], RecognitionState.COMPLETED.value)
            self.assertEqual(state["result"], "듀얼 카메라 테스트 제품")
            self.assertEqual(state["front_camera_id"], 0)
            self.assertEqual(state["back_camera_id"], 1)
            self.assertEqual(mock.calls, 1)
            self.assertTrue(Path(str(state["front_path"])).is_file())
            self.assertTrue(Path(str(state["back_path"])).is_file())
            self.assertTrue(Path(str(state["merged_path"])).is_file())

    def test_two_camera_candidates_are_automatically_compared_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mock = MockVisionClient("자동 비교 제품")
            recognizer = ObjectRecognizer(
                RecognitionConfig(
                    output_dir=Path(directory),
                    min_sharpness=1,
                    stable_duration=0.1,
                    phase_timeout=5,
                ),
                mock,
                lambda _message: None,
            )
            recognizer.start_recognition("dual")
            coordinator = DualCameraCoordinator(
                recognizer,
                DualCameraConfig(
                    stable_duration=0.1,
                    compare_window=0.5,
                    min_sharpness=1,
                    min_presence=0,
                ),
            )
            base = time.monotonic()
            soft_front = product_image(blur=9)
            sharp_front = product_image()
            for frame_id, offset in enumerate((0.0, 0.05, 0.16)):
                coordinator.process_packet(
                    FramePacket(soft_front, "0", frame_id, base + offset)
                )
                coordinator.process_packet(
                    FramePacket(sharp_front, "1", frame_id, base + offset)
                )
            self.assertEqual(coordinator.phase, "back")
            self.assertEqual(coordinator.front_camera_id, 1)

            soft_back = product_image(inverted=True, blur=9)
            sharp_back = product_image(inverted=True)
            for frame_id, offset in enumerate((0.30, 0.36, 0.48), start=10):
                coordinator.process_packet(
                    FramePacket(soft_back, "0", frame_id, base + offset)
                )
                coordinator.process_packet(
                    FramePacket(sharp_back, "1", frame_id, base + offset)
                )
            self.assertEqual(coordinator.phase, "recognizing")
            assert coordinator._worker is not None
            coordinator._worker.join(timeout=2)
            coordinator.poll_background()
            self.assertEqual(coordinator.phase, "completed")
            self.assertEqual(coordinator.back_camera_id, 1)
            self.assertEqual(mock.calls, 1)

    def test_recognition_runs_in_background(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class SlowMock:
            calls = 0

            def recognize(self, _jpeg: bytes) -> str:
                self.calls += 1
                started.set()
                release.wait(timeout=2)
                return "느린 Mock 제품"

        with tempfile.TemporaryDirectory() as directory:
            slow = SlowMock()
            coordinator = self._coordinator(directory, slow)
            now = time.monotonic()
            coordinator.process_packet(FramePacket(product_image(), "0", 1, now))
            coordinator.manual_capture(0, now)
            coordinator.process_packet(
                FramePacket(product_image(inverted=True), "0", 2, now + 0.1)
            )
            before = time.monotonic()
            coordinator.manual_capture(0, now + 0.1)
            elapsed = time.monotonic() - before
            self.assertLess(elapsed, 0.2)
            self.assertTrue(started.wait(timeout=1))
            self.assertTrue(coordinator.worker_active)
            canvas = DualCameraViewer().render(coordinator, now + 0.2)
            self.assertEqual(canvas.shape, (638, 1280, 3))
            release.set()
            assert coordinator._worker is not None
            coordinator._worker.join(timeout=2)
            coordinator.poll_background()
            self.assertEqual(coordinator.phase, "completed")
            self.assertEqual(slow.calls, 1)


if __name__ == "__main__":
    unittest.main()

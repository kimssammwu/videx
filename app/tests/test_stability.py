from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from object_recognition.api import MockVisionClient
from object_recognition.models import FramePacket, RecognitionConfig, RecognitionState
from object_recognition.recognizer import ObjectRecognizer
from object_recognition.stability import TimeWeightedStability


def product_image() -> np.ndarray:
    image = np.full((240, 320, 3), 127, dtype=np.uint8)
    cv2.rectangle(image, (70, 30), (250, 210), (245, 245, 245), 5)
    for y in range(55, 200, 18):
        cv2.line(image, (90, y), (230, y), (15, 15, 15), 3)
    return image


class StabilityWindowTests(unittest.TestCase):
    def test_default_recognition_roi_is_full_frame(self) -> None:
        self.assertEqual(RecognitionConfig().roi_fraction, 1.0)

    def tracker(self) -> TimeWeightedStability:
        return TimeWeightedStability(
            motion_threshold=0.040,
            reset_threshold=0.100,
            window_seconds=1.0,
            required_seconds=1.0,
            enter_ratio=0.80,
            exit_ratio=0.60,
            max_frame_gap=0.35,
        )

    def test_real_camera_log_values_reach_stability_by_elapsed_time(self) -> None:
        tracker = self.tracker()
        snapshot = None
        start = 0.0
        for motion in (0.0194, 0.0304, 0.0226, 0.0286):
            snapshot = tracker.add(start, start + 0.25, motion)
            start += 0.25
        assert snapshot is not None
        self.assertTrue(snapshot.ready)
        self.assertAlmostEqual(snapshot.stable_ratio, 1.0)
        self.assertAlmostEqual(snapshot.covered_seconds, 1.0)

    def test_one_small_spike_is_tolerated_but_not_used_as_capture_frame(self) -> None:
        tracker = self.tracker()
        snapshot = None
        spike = None
        start = 0.0
        for motion in (0.01, 0.01, 0.05, 0.01, 0.01):
            snapshot = tracker.add(start, start + 0.2, motion)
            if motion == 0.05:
                spike = snapshot
            start += 0.2
        assert snapshot is not None and spike is not None
        self.assertFalse(spike.current_stable)
        self.assertTrue(snapshot.ready)
        self.assertAlmostEqual(snapshot.stable_ratio, 0.80)
        self.assertTrue(snapshot.current_stable)

    def test_large_motion_and_long_frame_gap_reset_elapsed_stability(self) -> None:
        tracker = self.tracker()
        tracker.add(0.0, 0.25, 0.01)
        tracker.add(0.25, 0.50, 0.01)
        large = tracker.add(0.50, 0.75, 0.12)
        self.assertTrue(large.large_motion)
        self.assertEqual(large.covered_seconds, 0.0)
        after = tracker.add(0.75, 1.0, 0.01)
        self.assertFalse(after.ready)
        gap = tracker.add(1.0, 1.5, 0.01)
        self.assertTrue(gap.frame_gap)
        self.assertFalse(gap.stable)

    def test_recognizer_automatically_captures_with_logged_motion_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recognizer = ObjectRecognizer(
                RecognitionConfig(
                    output_dir=Path(directory),
                    frame_check_interval=0,
                    stable_duration=1.0,
                    motion_threshold=0.040,
                    motion_reset_threshold=0.100,
                    min_sharpness=1,
                    phase_timeout=5,
                ),
                MockVisionClient(),
                lambda _message: None,
            )
            recognizer.start_recognition("1")
            image = product_image()
            packets = [FramePacket(image, "1", index, index * 0.25) for index in range(5)]
            motions = iter((0.0194, 0.0304, 0.0226, 0.0286))
            with patch(
                "object_recognition.recognizer.normalized_difference",
                side_effect=lambda _first, _second: next(motions),
            ):
                last = None
                for packet in packets:
                    last = recognizer.process_frame(packet)
            assert last is not None
            self.assertTrue(last.selected)
            self.assertEqual(recognizer.state, RecognitionState.WAIT_ROTATION)
            self.assertIsNotNone(recognizer.front_path)

    def test_large_motion_prevents_capture_until_a_new_stable_second(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recognizer = ObjectRecognizer(
                RecognitionConfig(
                    output_dir=Path(directory),
                    frame_check_interval=0,
                    stable_duration=1.0,
                    motion_threshold=0.040,
                    motion_reset_threshold=0.100,
                    min_sharpness=1,
                    phase_timeout=5,
                ),
                MockVisionClient(),
                lambda _message: None,
            )
            recognizer.start_recognition("1")
            image = product_image()
            motions = iter((0.01, 0.01, 0.12, 0.01, 0.01, 0.01))
            with patch(
                "object_recognition.recognizer.normalized_difference",
                side_effect=lambda _first, _second: next(motions),
            ):
                for index in range(7):
                    recognizer.process_frame(
                        FramePacket(image, "1", index, index * 0.25)
                    )
            self.assertEqual(recognizer.state, RecognitionState.WAIT_FRONT)
            self.assertIsNone(recognizer.front_path)

    def test_small_motion_spike_keeps_candidates_and_still_auto_captures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recognizer = ObjectRecognizer(
                RecognitionConfig(
                    output_dir=Path(directory),
                    frame_check_interval=0,
                    stable_duration=1.0,
                    motion_threshold=0.040,
                    motion_reset_threshold=0.100,
                    min_sharpness=1,
                    phase_timeout=5,
                ),
                MockVisionClient(),
                lambda _message: None,
            )
            recognizer.start_recognition("1")
            image = product_image()
            motions = iter((0.01, 0.01, 0.05, 0.01, 0.01))
            results = []
            with patch(
                "object_recognition.recognizer.normalized_difference",
                side_effect=lambda _first, _second: next(motions),
            ):
                for index in range(6):
                    results.append(
                        recognizer.process_frame(
                            FramePacket(image, "1", index, index * 0.2)
                        )
                    )
            self.assertEqual(results[3].reason, "noise_tolerated")
            self.assertTrue(results[-1].selected)
            self.assertEqual(recognizer.state, RecognitionState.WAIT_ROTATION)


if __name__ == "__main__":
    unittest.main()

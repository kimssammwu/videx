from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from object_recognition.api import MockVisionClient
from object_recognition.image_ops import encode_jpeg
from object_recognition.models import FramePacket, RecognitionConfig, RecognitionState
from object_recognition.recognizer import ObjectRecognizer


def product_image(inverted: bool = False) -> np.ndarray:
    image = np.full((240, 320, 3), 210 if not inverted else 45, dtype=np.uint8)
    color = (25, 25, 25) if not inverted else (235, 235, 235)
    cv2.rectangle(image, (75, 35), (245, 205), color, 5)
    for y in range(60, 195, 20):
        cv2.line(image, (90, y), (230, y), color, 3)
    cv2.putText(image, "FRONT" if not inverted else "BACK", (95, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def left_edge_product(inverted: bool = False) -> np.ndarray:
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    fill = 30 if inverted else 225
    ink = 240 if inverted else 15
    cv2.rectangle(image, (2, 3), (59, 236), (fill, fill, fill), -1)
    for y in range(15, 230, 18):
        cv2.line(image, (7, y), (54, y), (ink, ink, ink), 3)
    return image


class RecognizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.messages: list[str] = []
        self.mock = MockVisionClient("VIDEX 테스트 음료 500mL")
        self.config = RecognitionConfig(
            output_dir=Path(self.temp_dir.name),
            frame_check_interval=0,
            stable_duration=0.3,
            motion_threshold=0.03,
            rotation_threshold=0.25,
            duplicate_threshold=0.10,
            min_sharpness=10,
            phase_timeout=5,
        )
        self.recognizer = ObjectRecognizer(self.config, self.mock, self.messages.append)

    def packet(self, image: np.ndarray, frame_id: int, timestamp: float) -> FramePacket:
        return FramePacket(encode_jpeg(image), "cam-1", frame_id, timestamp)

    def test_automatic_end_to_end_pipeline(self) -> None:
        front = product_image(False)
        back = product_image(True)
        self.recognizer.start_recognition("cam-1")
        for index, timestamp in enumerate((0.0, 0.1, 0.2, 0.4, 0.5)):
            self.recognizer.process_frame(self.packet(front, index, timestamp))
        self.assertEqual(self.recognizer.state, RecognitionState.WAIT_ROTATION)
        rotation = self.recognizer.process_frame(self.packet(back, 10, 0.6))
        self.assertEqual(rotation.reason, "rotation_detected")
        for index, timestamp in enumerate((0.7, 0.8, 0.9, 1.1, 1.2), 11):
            self.recognizer.process_frame(self.packet(back, index, timestamp))
            if self.recognizer.state is RecognitionState.COMPLETED:
                break
        state = self.recognizer.get_recognition_state()
        self.assertEqual(state["state"], "COMPLETED")
        self.assertEqual(state["result"], "VIDEX 테스트 음료 500mL")
        self.assertEqual(self.mock.calls, 1)
        self.assertTrue(Path(state["front_path"]).is_file())
        self.assertTrue(Path(state["back_path"]).is_file())
        self.assertTrue(Path(state["merged_path"]).is_file())
        self.assertIn(RecognitionState.MERGING, self.recognizer.state_history)
        self.assertIn(RecognitionState.RECOGNIZING, self.recognizer.state_history)

    def test_off_center_product_automatically_captures_with_full_frame_roi(self) -> None:
        self.assertEqual(self.config.roi_fraction, 1.0)
        self.recognizer.config.rotation_threshold = 0.10
        front = left_edge_product(False)
        back = left_edge_product(True)
        self.recognizer.start_recognition("cam-1")
        for index, timestamp in enumerate((0.0, 0.1, 0.2, 0.4, 0.5)):
            self.recognizer.process_frame(self.packet(front, index, timestamp))
        self.assertEqual(self.recognizer.state, RecognitionState.WAIT_ROTATION)
        rotation = self.recognizer.process_frame(self.packet(back, 10, 0.6))
        self.assertEqual(rotation.reason, "rotation_detected")
        for index, timestamp in enumerate((0.7, 0.8, 0.9, 1.1, 1.2), 11):
            self.recognizer.process_frame(self.packet(back, index, timestamp))
            if self.recognizer.state is RecognitionState.COMPLETED:
                break
        self.assertEqual(self.recognizer.state, RecognitionState.COMPLETED)
        self.assertEqual(self.mock.calls, 1)
        self.assertTrue(self.recognizer.front_path and self.recognizer.front_path.is_file())
        self.assertTrue(self.recognizer.back_path and self.recognizer.back_path.is_file())
        self.assertTrue(self.recognizer.merged_path and self.recognizer.merged_path.is_file())

    def test_duplicate_manual_back_is_rejected(self) -> None:
        front = product_image(False)
        self.recognizer.start_recognition("cam-1")
        self.recognizer.process_frame(self.packet(front, 1, 0), manual_capture=True)
        result = self.recognizer.process_frame(self.packet(front, 2, 0.1), manual_capture=True)
        self.assertFalse(result.selected)
        self.assertEqual(result.reason, "duplicate_rejected")
        self.assertEqual(self.recognizer.state, RecognitionState.WAIT_ROTATION)

    def test_manual_capture_overrides_quality_threshold(self) -> None:
        dark = np.zeros((120, 160, 3), dtype=np.uint8)
        back = np.full((120, 160, 3), 180, dtype=np.uint8)
        self.recognizer.start_recognition("cam-1")
        first = self.recognizer.process_frame(self.packet(dark, 1, 0), manual_capture=True)
        second = self.recognizer.process_frame(self.packet(back, 2, 0.1), manual_capture=True)
        self.assertTrue(first.selected)
        self.assertTrue(second.selected)
        self.assertEqual(self.recognizer.state, RecognitionState.COMPLETED)

    def test_async_guidance_must_be_acknowledged_before_rotation(self) -> None:
        config = RecognitionConfig(
            output_dir=Path(self.temp_dir.name),
            min_sharpness=10,
            auto_acknowledge_guidance=False,
        )
        recognizer = ObjectRecognizer(config, self.mock, self.messages.append)
        recognizer.start_recognition("cam-1")
        recognizer.process_frame(self.packet(product_image(False), 1, 0), manual_capture=True)
        self.assertEqual(recognizer.state, RecognitionState.FRONT_CAPTURED)
        waiting = recognizer.process_frame(self.packet(product_image(True), 2, 0.1))
        self.assertEqual(waiting.reason, "waiting_guidance_completion")
        recognizer.acknowledge_rotation_guidance(0.2)
        self.assertEqual(recognizer.state, RecognitionState.WAIT_ROTATION)

    def test_blurry_frames_are_not_automatic_candidates(self) -> None:
        self.recognizer.start_recognition("cam-1")
        blurred = np.full((240, 320, 3), 128, dtype=np.uint8)
        reasons = []
        for index, timestamp in enumerate((0.0, 0.1, 0.2, 0.5)):
            result = self.recognizer.process_frame(self.packet(blurred, index, timestamp))
            reasons.append(result.reason)
        self.assertIn("quality_rejected", reasons)
        self.assertEqual(self.recognizer.state, RecognitionState.WAIT_FRONT)

    def test_camera_ids_do_not_mix(self) -> None:
        self.recognizer.start_recognition("cam-1")
        result = self.recognizer.process_frame(
            FramePacket(encode_jpeg(product_image()), "cam-2", 1, 0)
        )
        self.assertEqual(result.reason, "camera_mismatch")

    def test_timeout_can_be_retried(self) -> None:
        self.recognizer.start_recognition("cam-1")
        self.assertTrue(self.recognizer.tick(self.recognizer.phase_started_at + 6))
        self.assertEqual(self.recognizer.state, RecognitionState.ERROR)
        self.recognizer.retry_recognition()
        self.assertEqual(self.recognizer.state, RecognitionState.WAIT_FRONT)

    def test_stop_clears_in_memory_work(self) -> None:
        self.recognizer.start_recognition("cam-1")
        self.recognizer.process_frame(self.packet(product_image(), 1, 0))
        state = self.recognizer.stop_recognition()
        self.assertEqual(state["state"], "IDLE")
        self.assertIsNone(state["camera_id"])


if __name__ == "__main__":
    unittest.main()

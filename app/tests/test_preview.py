from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from object_recognition.models import FramePacket, RecognitionConfig, RecognitionState
from object_recognition.preview import LeftCameraViewer, RecognitionPreviewWorker
from object_recognition.recognizer import ObjectRecognizer


def product_image(inverted: bool = False) -> np.ndarray:
    image = np.full((240, 320, 3), 210 if not inverted else 45, dtype=np.uint8)
    color = (20, 20, 20) if not inverted else (240, 240, 240)
    cv2.rectangle(image, (70, 30), (250, 210), color, 5)
    cv2.putText(
        image,
        "FRONT" if not inverted else "BACK",
        (90, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    return image


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class PreviewTests(unittest.TestCase):
    def test_preview_reports_left_usb_requested_and_actual_resolution(self) -> None:
        recognizer = ObjectRecognizer(announce=lambda _message: None)
        recognizer.start_recognition("0")
        viewer = LeftCameraViewer()
        viewer.update_usb_status(
            connected=True,
            camera_id=0,
            frame_id=77,
            width=640,
            height=480,
            received_at=10.0,
            paused=False,
            resolution_status="sent_unconfirmed",
            resolution_support_status="actual_matches_request",
            requested_resolution_mode=1,
        )
        with patch("cv2.putText", wraps=cv2.putText) as put_text:
            canvas = viewer.render(recognizer, None, now=10.1)
        rendered_text = "\n".join(str(call.args[1]) for call in put_text.call_args_list)
        self.assertIn("USB CONNECTED", rendered_text)
        self.assertIn("CAMERA LEFT (0)", rendered_text)
        self.assertIn("frame 77", rendered_text)
        self.assertIn("Resolution: 640x480 (Mode 1)", rendered_text)
        self.assertIn("Actual: 640x480", rendered_text)
        self.assertIn("LEFT status actual_matches_request", rendered_text)
        self.assertEqual(canvas.shape, (720, 1280, 3))

    def test_full_frame_preview_does_not_draw_a_central_roi_box(self) -> None:
        recognizer = ObjectRecognizer(announce=lambda _message: None)
        recognizer.start_recognition("0")
        viewer = LeftCameraViewer()
        now = time.monotonic()
        viewer.update_frame(FramePacket(np.full((240, 320, 3), 90, dtype=np.uint8), "0", 1, now))
        canvas = viewer.render(recognizer, None, now=now)
        live_area = canvas[viewer.HEADER_HEIGHT :, : viewer.LIVE_WIDTH]
        pure_yellow = np.all(live_area == np.array([0, 255, 255], dtype=np.uint8), axis=2)
        self.assertFalse(bool(np.any(pure_yellow)))

    def test_worker_rejects_right_and_api_does_not_block_rendering(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class SlowMock:
            calls = 0

            def recognize(self, _jpeg: bytes) -> str:
                self.calls += 1
                started.set()
                release.wait(timeout=2)
                return "LEFT 테스트 제품"

        with tempfile.TemporaryDirectory() as directory:
            slow = SlowMock()
            recognizer = ObjectRecognizer(
                RecognitionConfig(output_dir=Path(directory), min_sharpness=1),
                slow,
                lambda _message: None,
            )
            recognizer.start_recognition("0")
            worker = RecognitionPreviewWorker(recognizer)
            worker.start()
            self.addCleanup(worker.stop)
            now = time.monotonic()
            with self.assertRaisesRegex(ValueError, "LEFT"):
                worker.submit(FramePacket(product_image(), "1", 1, now))

            front = FramePacket(product_image(), "0", 2, now)
            worker.submit(front, manual_capture=True)
            self.assertTrue(
                wait_until(lambda: recognizer.state is RecognitionState.WAIT_ROTATION)
            )
            back = FramePacket(product_image(True), "0", 3, now + 0.1)
            worker.submit(back, manual_capture=True)
            self.assertTrue(started.wait(timeout=1))
            self.assertTrue(worker.busy)

            viewer = LeftCameraViewer()
            viewer.update_frame(back)
            before = time.monotonic()
            canvas = viewer.render(
                recognizer,
                worker.latest_metrics,
                worker_busy=worker.busy,
                now=now + 0.2,
            )
            self.assertLess(time.monotonic() - before, 0.2)
            self.assertEqual(canvas.shape, (720, 1280, 3))

            release.set()
            self.assertTrue(
                wait_until(lambda: recognizer.state is RecognitionState.COMPLETED)
            )
            self.assertEqual(slow.calls, 1)
            state = recognizer.get_recognition_state()
            self.assertTrue(Path(str(state["front_path"])).is_file())
            self.assertTrue(Path(str(state["back_path"])).is_file())
            self.assertTrue(Path(str(state["merged_path"])).is_file())

    def test_worker_reset_restarts_front_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recognizer = ObjectRecognizer(
                RecognitionConfig(output_dir=Path(directory), min_sharpness=1),
                announce=lambda _message: None,
            )
            recognizer.start_recognition("0")
            worker = RecognitionPreviewWorker(recognizer)
            recognizer.process_frame(
                FramePacket(product_image(), "0", 1, time.monotonic()),
                manual_capture=True,
            )
            self.assertIsNotNone(recognizer.front_path)
            self.assertTrue(worker.reset())
            self.assertEqual(recognizer.state, RecognitionState.WAIT_FRONT)
            self.assertEqual(recognizer.camera_id, "0")
            self.assertIsNone(recognizer.front_path)
            self.assertIsNone(recognizer.previous_gray)
            self.assertEqual(recognizer.candidates, [])

    def test_auto_capture_works_at_fixed_left_640x480(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recognizer = ObjectRecognizer(
                RecognitionConfig(
                    output_dir=Path(directory),
                    min_sharpness=1,
                    frame_check_interval=0,
                    stable_duration=0.3,
                    stability_window=0.3,
                    max_frame_gap=0.2,
                ),
                announce=lambda _message: None,
            )
            recognizer.start_recognition("0")
            base = time.monotonic()
            high_resolution = cv2.resize(
                product_image(), (640, 480), interpolation=cv2.INTER_CUBIC
            )
            for frame_id, offset in enumerate((0.0, 0.1, 0.2, 0.3, 0.4), start=1):
                recognizer.process_frame(
                    FramePacket(high_resolution, "0", frame_id, base + 1.0 + offset)
                )
                if recognizer.front_path is not None:
                    break
            self.assertIsNotNone(recognizer.front_path)
            captured = cv2.imread(str(recognizer.front_path))
            self.assertEqual(captured.shape[:2], (480, 640))

    def test_viewer_disables_resolution_keys_and_keeps_c_r_q_controls(self) -> None:
        viewer = LeftCameraViewer()
        viewer._opened = True
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        for key, expected in (
            (ord("0"), None),
            (ord("1"), None),
            (ord("2"), None),
            (ord("c"), "c"),
            (ord("r"), "r"),
            (ord("q"), "q"),
            (27, "q"),
        ):
            with self.subTest(key=key), patch("cv2.imshow"), patch(
                "cv2.waitKey", return_value=key
            ), patch("cv2.getWindowProperty", return_value=1.0):
                self.assertEqual(viewer.show(canvas), expected)
        viewer._opened = False


if __name__ == "__main__":
    unittest.main()

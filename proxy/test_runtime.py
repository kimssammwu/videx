"""Integration tests use actual pipelines and fake camera/API boundaries."""
from collections import deque
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

# Works both from the repository root and unittest discover -s proxy.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.object_recognition.api import MockVisionClient as MockObject
from app.object_recognition.models import RecognitionConfig
from app.text_recognition.vision import MockVisionClient as MockText
from proxy.runtime import CameraSource, Runtime, WalkingPipeline, demo_image
from proxy.viewer import ButtonEvents, Frame, HEADER, receive_frames


class FakeSource:
    port = None

    def __init__(self):
        self.latest = {}
        self.events = []
        self.epoch = 1
        self.counter = 0
        self.closed = False

    def set_mode(self, mode):
        self.mode = mode
        self.latest.clear()

    def snapshot(self):
        events, self.events = self.events, []
        return self.latest.copy(), {"epoch": self.epoch, "status": "test"}, events

    def feed(self, image=None, age=0, camera=0):
        self.counter += 1
        self.latest[camera] = (demo_image(0) if image is None else image,
                               self.counter, time.monotonic() - age, None, False, None, False)

    def close(self):
        self.closed = True


class Speech:
    def __init__(self):
        self.messages = []

    def say(self, text):
        self.messages.append(text)

    def reset(self):
        pass

    def close(self):
        pass


class BlockingText(MockText):
    def __init__(self):
        super().__init__("old response")
        self.started = threading.Event()
        self.release = threading.Event()

    def recognize(self, image):
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError("test timed out")
        return super().recognize(image)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = FakeSource()
        self.text = MockText("두 번째 촬영도 가능합니다.")
        self.objects = MockObject("테스트 제품")
        self.speech = Speech()
        self.runtime = Runtime(self.source, self.text, self.objects,
                               output_dir=Path(self.tmp.name), speech=self.speech, mode=1,
                               object_config=RecognitionConfig(stable_duration=100, frame_check_interval=0))
        self.addCleanup(self.runtime.close)

    def wait_for(self, predicate, seconds=3):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.runtime.tick()
            if predicate():
                return
            time.sleep(.005)
        self.fail(f"runtime timeout: {self.runtime.status()}")

    def capture(self, image=None):
        self.source.feed(image)
        self.runtime.tick()
        self.assertTrue(self.runtime.action())

    def test_text_repeated_capture_keeps_mode_and_saves_exact_api_image(self):
        for count in (1, 2):
            self.capture()
            self.wait_for(lambda: not self.runtime.vision.busy)
            self.assertEqual(self.runtime.mode, 1)
            self.assertEqual(self.runtime.state, "COMPLETED")
            self.assertEqual(self.text.calls, count)
            path = Path(self.runtime.history[-1]["path"])
            self.assertEqual(path.read_bytes(), self.text.last_image)

    def test_slow_api_does_not_block_mode_switch_or_walking_and_old_reply_is_discarded(self):
        blocking = BlockingText()
        self.addCleanup(blocking.release.set)
        self.runtime.text_client = blocking
        self.capture()
        self.assertTrue(blocking.started.wait(1))
        self.assertFalse(self.runtime.action())
        started = time.monotonic()
        self.runtime.select_mode(0)
        self.assertLess(time.monotonic() - started, .2)
        self.source.feed()
        self.wait_for(lambda: self.runtime.walk_result is not None)
        blocking.release.set()
        self.wait_for(lambda: not self.runtime.vision.busy)
        self.assertEqual(self.runtime.mode, 0)
        self.assertIsNone(self.runtime.result)
        self.assertNotIn("old response", self.speech.messages)

    def test_switch_away_and_back_also_discards_old_request(self):
        blocking = BlockingText()
        self.addCleanup(blocking.release.set)
        self.runtime.text_client = blocking
        self.capture()
        self.assertTrue(blocking.started.wait(1))
        self.runtime.select_mode(0)
        self.runtime.select_mode(1)
        blocking.release.set()
        self.wait_for(lambda: not self.runtime.vision.busy)
        self.assertEqual(self.runtime.state, "READY")
        self.assertIsNone(self.runtime.result)

    def test_api_error_can_retry_in_same_mode(self):
        class Failing:
            def recognize(self, jpeg):
                raise TimeoutError("network unavailable")
        self.runtime.text_client = Failing()
        self.capture()
        self.wait_for(lambda: not self.runtime.vision.busy)
        self.assertEqual(self.runtime.state, "ERROR")
        self.assertEqual(self.runtime.mode, 1)
        self.runtime.text_client = self.text
        self.capture()
        self.wait_for(lambda: not self.runtime.vision.busy)
        self.assertEqual(self.runtime.state, "COMPLETED")
        self.assertIsNone(self.runtime.error)

    def test_object_front_back_merge_and_next_product_in_same_mode(self):
        self.runtime.select_mode(2)
        for expected in (1, 2):
            self.capture(demo_image(0, 0))
            self.wait_for(lambda: not self.runtime.vision.busy)
            self.assertEqual(self.runtime.state, "WAIT_ROTATION")
            # Same frame as the front cannot be falsely accepted as the back.
            self.assertTrue(self.runtime.action())
            self.wait_for(lambda: not self.runtime.vision.busy)
            self.assertEqual(self.runtime.state, "WAIT_ROTATION")
            self.source.feed(demo_image(0, 1))
            self.runtime.snapshot = self.source.latest.copy()
            self.assertTrue(self.runtime.action())
            self.wait_for(lambda: not self.runtime.vision.busy)
            self.assertEqual(self.runtime.mode, 2)
            self.assertEqual(self.runtime.state, "COMPLETED")
            self.assertEqual(self.objects.calls, expected)
            self.assertTrue(Path(self.runtime.history[-1]["merged_path"]).is_file())
        self.assertEqual(len(list(Path(self.tmp.name).glob("object_*/*_merged.jpg"))), 2)

    def test_object_auto_capture_keeps_running_between_front_and_back(self):
        self.runtime.object_config = replace(self.runtime.object_config, stable_duration=.05)
        self.runtime.select_mode(2)
        self.capture(demo_image(0, 0))
        self.wait_for(lambda: not self.runtime.vision.busy)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and self.runtime.state != "COMPLETED":
            self.source.feed(demo_image(0, 1))
            self.runtime.tick()
            time.sleep(.03)
        self.assertEqual(self.runtime.state, "COMPLETED")
        self.assertEqual(self.objects.calls, 1)
        self.assertEqual(self.runtime.mode, 2)

    def test_stale_frame_or_wrong_usb_resolution_never_reaches_api(self):
        self.source.feed(age=5)
        self.runtime.tick()
        self.assertFalse(self.runtime.action())
        self.source.port = "COM_TEST"
        self.source.feed(np.zeros((240, 320, 3), np.uint8))
        self.runtime.tick()
        self.assertFalse(self.runtime.action())
        self.assertEqual(self.text.calls, 0)

    def test_right_camera_alone_never_triggers_recognition(self):
        self.source.feed(camera=1)
        self.runtime.tick()
        self.assertFalse(self.runtime.action())

    def test_mode_switch_cannot_capture_a_cached_previous_mode_frame(self):
        self.source.feed()
        self.runtime.tick()
        self.assertIsNotNone(self.runtime.frame)
        self.runtime.select_mode(2)
        self.assertFalse(self.runtime.action())
        self.assertIsNone(self.runtime.recognizer)

    def test_reconnect_preserves_mode_and_resets_partial_object_session(self):
        self.runtime.select_mode(2)
        self.capture()
        self.wait_for(lambda: not self.runtime.vision.busy)
        self.source.epoch += 1
        self.runtime.tick()
        self.assertEqual(self.runtime.mode, 2)
        self.assertIsNone(self.runtime.recognizer)
        self.assertEqual(self.runtime.state, "READY")
        self.assertEqual(self.source.mode, 2)

    def test_stale_walking_decision_is_unknown_and_does_not_reprocess_duplicates(self):
        self.runtime.select_mode(0)
        self.source.feed()
        self.wait_for(lambda: self.runtime.walk_result is not None)
        processed = self.runtime.pipeline.flood.processed
        for _ in range(5):
            self.runtime.tick()
        self.assertEqual(self.runtime.pipeline.flood.processed, processed)
        self.runtime.tick(time.monotonic() + 3)
        self.assertEqual(self.runtime.state, "UNKNOWN")
        self.assertIsNone(self.runtime.overlay)

    def test_button_short_captures_and_long_changes_persistent_mode(self):
        self.source.feed()
        self.source.events = [("SHORT", time.monotonic())]
        self.wait_for(lambda: self.runtime.state == "COMPLETED")
        self.assertEqual(self.text.calls, 1)
        self.source.events = [("LONG", time.monotonic())]
        self.runtime.tick()
        self.assertEqual(self.runtime.mode, 2)
        self.source.events = [("LONG", time.monotonic() - 3)]
        self.runtime.tick()
        self.assertEqual(self.runtime.mode, 2)

    def test_shutdown_does_not_wait_for_remote_api(self):
        blocking = BlockingText()
        self.addCleanup(blocking.release.set)
        self.runtime.text_client = blocking
        self.capture()
        self.assertTrue(blocking.started.wait(1))
        started = time.monotonic()
        self.runtime.close()
        self.assertLess(time.monotonic() - started, .7)
        blocking.release.set()
        self.assertTrue(self.source.closed)


class TransportTests(unittest.TestCase):
    def test_classified_button_dedup_and_next_press(self):
        detector = ButtonEvents()
        frame = Frame(0, 1, 320, 240, 0, False, True, "SHORT", b"jpeg")
        self.assertIsNone(detector.observe(frame, 0))  # Reconnect baseline is not a press.
        self.assertIsNone(detector.observe(replace(frame, frame_id=2), .1))
        detector.observe(replace(frame, button_pressed=True, button_result=None), .2)
        self.assertEqual(detector.observe(frame, .3), "SHORT")
        self.assertIsNone(detector.observe(replace(frame, camera_id=1), .4))

    def test_legacy_button_release_classifies_press_duration(self):
        detector = ButtonEvents()
        frame = Frame(0, 1, 320, 240, 0, True, False, None, b"jpeg")
        detector.observe(replace(frame, button_pressed=False), -.1)
        self.assertIsNone(detector.observe(frame, 0))
        self.assertEqual(detector.observe(replace(frame, button_pressed=False), 1.2), "LONG")
        self.assertIsNone(detector.observe(replace(frame, button_pressed=False), 1.3))

    def test_single_serial_owner_retains_button_event_when_newer_frames_overwrite_it(self):
        stop = threading.Event()
        writes = []
        image = np.full((240, 320, 3), 150, np.uint8)
        jpeg = cv2.imencode(".jpg", image)[1].tobytes()
        def packet(index, result, pressed=False):
            return HEADER.pack(int.from_bytes(b"CAM2", "little"), 0, 7 if pressed else 6, result, 0,
                               index, len(jpeg), 320, 240, index * 1000) + jpeg
        stream = packet(1, 0) + packet(2, 0, True) + packet(3, 1) + packet(4, 1)
        class Connection:
            in_waiting = len(stream)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def write(self, data):
                writes.append(data)
                return len(data)
            def read(self, count):
                stop.set()
                return stream
        class Serial:
            SerialException = OSError
            def Serial(self, *args, **kwargs):
                return Connection()
        latest = {}
        control = {"resolution": 1, "button_events": deque(maxlen=32)}
        receive_frames("COM_TEST", 115200, stop, latest, threading.Lock(), Serial(), cv2, np, control)
        self.assertEqual(writes, [b"\x01"])
        self.assertEqual(latest[0][1], 4)
        self.assertEqual(control["button_events"][0][0], "SHORT")
        self.assertEqual(control["epoch"], 1)

    def test_walking_decision_matches_flood_fill_without_other_models(self):
        from proxy.flood_fill import Detector
        pipeline, reference = WalkingPipeline(), Detector()
        for index in range(5):
            image = demo_image(index)
            expected = reference.process(image, float(index), frame_id=index)
            result = pipeline.process(image, index, float(index))
            self.assertEqual(result["state"], expected.state)
            self.assertEqual(result["reason"], expected.reason)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import threading
import unittest

from text_recognition.errors import RecognitionBusyError, VisionResponseError
from text_recognition.service import TextRecognitionService
from text_recognition.vision import MockVisionClient


class CountingCamera:
    def __init__(self, image: bytes = b"jpeg") -> None:
        self.image = image
        self.calls = 0

    def capture(self) -> bytes:
        self.calls += 1
        return self.image


class BlockingCamera(CountingCamera):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def capture(self) -> bytes:
        self.calls += 1
        self.entered.set()
        if not self.release.wait(timeout=2):
            raise AssertionError("test did not release blocking camera")
        return self.image


class InvalidVisionClient:
    def recognize(self, _jpeg_bytes: bytes) -> dict[str, object]:
        return {"text": "글씨", "extra": True}


class ServiceTests(unittest.TestCase):
    def test_one_request_captures_and_calls_vision_once(self) -> None:
        camera = CountingCamera()
        vision = MockVisionClient("시설 점검 안내\n시설 이용이 제한됩니다.")
        service = TextRecognitionService(camera, vision)

        result = service.recognize_text()

        self.assertEqual(result, {"text": "시설 점검 안내\n시설 이용이 제한됩니다."})
        self.assertEqual(camera.calls, 1)
        self.assertEqual(vision.calls, 1)
        self.assertEqual(vision.last_image, b"jpeg")

    def test_empty_text_is_a_successful_result(self) -> None:
        service = TextRecognitionService(CountingCamera(), MockVisionClient(""))
        self.assertEqual(service.recognize_text(), {"text": ""})

    def test_overlapping_request_is_rejected_without_second_capture(self) -> None:
        camera = BlockingCamera()
        service = TextRecognitionService(camera, MockVisionClient("간판"))
        results: list[dict[str, str]] = []
        worker = threading.Thread(target=lambda: results.append(service.recognize_text()))
        worker.start()
        self.assertTrue(camera.entered.wait(timeout=1))
        try:
            with self.assertRaises(RecognitionBusyError):
                service.recognize_text()
        finally:
            camera.release.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(camera.calls, 1)
        self.assertEqual(results, [{"text": "간판"}])

    def test_service_rejects_unexpected_vision_fields(self) -> None:
        service = TextRecognitionService(CountingCamera(), InvalidVisionClient())
        with self.assertRaises(VisionResponseError):
            service.recognize_text()


if __name__ == "__main__":
    unittest.main()

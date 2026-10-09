from __future__ import annotations

import unittest

import cv2
import numpy as np

from object_recognition.models import FramePacket
from object_recognition.usb_source import CAM2_HEADER, CAM2_MAGIC, resolution_command
from text_recognition.camera import UsbLeftCamera, UsbPreviewCamera, validate_jpeg
from text_recognition.errors import CameraTimeoutError, InvalidImageError


def jpeg_bytes(value: int) -> bytes:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise AssertionError("test JPEG encoding failed")
    return encoded.tobytes()


def cam2_frame(jpeg: bytes, camera_id: int, frame_id: int) -> bytes:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    return CAM2_HEADER.pack(
        int.from_bytes(CAM2_MAGIC, "little"),
        camera_id,
        0,
        0,
        0,
        frame_id,
        len(jpeg),
        width,
        height,
        frame_id * 1000,
    ) + jpeg


class FakeSerial:
    def __init__(self, data: bytes) -> None:
        self.data = bytearray(data)
        self.writes: list[bytes] = []
        self.read_calls = 0
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self.data)

    def read(self, size: int) -> bytes:
        self.read_calls += 1
        chunk = bytes(self.data[:size])
        del self.data[:size]
        return chunk

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class CameraTests(unittest.TestCase):
    def test_capture_uses_left_only_and_returns_original_jpeg_once(self) -> None:
        right = jpeg_bytes(40)
        left = jpeg_bytes(210)
        fake = FakeSerial(cam2_frame(right, 1, 1) + cam2_frame(left, 0, 2))
        camera = UsbLeftCamera(
            "COM_TEST",
            serial_factory=lambda *_args, **_kwargs: fake,
            capture_timeout=1,
        )

        result = camera.capture()

        self.assertEqual(result, left)
        self.assertNotEqual(result, right)
        self.assertEqual(fake.writes, [resolution_command(1)])
        self.assertEqual(fake.read_calls, 1)
        self.assertTrue(fake.closed)

    def test_camera_timeout_is_not_an_empty_text_result(self) -> None:
        fake = FakeSerial(b"")
        camera = UsbLeftCamera(
            "COM_TEST",
            serial_factory=lambda *_args, **_kwargs: fake,
            capture_timeout=0.001,
        )
        with self.assertRaises(CameraTimeoutError):
            camera.capture()
        self.assertTrue(fake.closed)

    def test_invalid_jpeg_is_rejected(self) -> None:
        with self.assertRaises(InvalidImageError):
            validate_jpeg(b"not-a-jpeg")
        with self.assertRaises(InvalidImageError):
            validate_jpeg(b"\xff\xd8broken\xff\xd9")

    def test_preview_stream_configures_and_emits_left_only(self) -> None:
        class FakeSource:
            def __init__(self) -> None:
                self.opened = False
                self.closed = False

            def open(self) -> None:
                self.opened = True

            def poll(self) -> list[FramePacket]:
                image = np.full((24, 32, 3), 150, dtype=np.uint8)
                return [
                    FramePacket(image, "1", 10, 1.0),
                    FramePacket(image, "0", 11, 1.1),
                ]

            def close(self) -> None:
                self.closed = True

        created: dict[str, object] = {}
        source = FakeSource()

        def factory(port: str, camera_id: int, **kwargs: object) -> FakeSource:
            created.update(port=port, camera_id=camera_id, **kwargs)
            return source

        camera = UsbPreviewCamera("COM_TEST", max_fps=8, source_factory=factory)
        camera.open()
        frames = camera.poll()
        camera.close()

        self.assertEqual(created["port"], "COM_TEST")
        self.assertEqual(created["camera_id"], 0)
        self.assertEqual(created["resolution_mode"], 1)
        self.assertEqual(created["max_fps"], 8)
        self.assertTrue(source.opened)
        self.assertTrue(source.closed)
        self.assertEqual([frame.frame_id for frame in frames], [11])


if __name__ == "__main__":
    unittest.main()

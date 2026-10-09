from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from object_recognition.api import MockVisionClient
from object_recognition.image_ops import encode_jpeg
from object_recognition.models import RecognitionConfig, RecognitionState
from object_recognition.recognizer import ObjectRecognizer
from object_recognition.usb_source import (
    CAM2_HEADER,
    CAM2_MAGIC,
    Cam2Frame,
    Cam2FrameParser,
    UsbCameraSource,
    UsbCameraError,
    cam2_frame_to_packet,
)


def product_image(inverted: bool = False) -> np.ndarray:
    image = np.full((120, 160, 3), 210 if not inverted else 45, dtype=np.uint8)
    color = (25, 25, 25) if not inverted else (235, 235, 235)
    cv2.rectangle(image, (35, 15), (125, 105), color, 4)
    for y in range(30, 100, 15):
        cv2.line(image, (45, y), (115, y), color, 2)
    return image


def cam2_bytes(
    image: np.ndarray,
    camera_id: int,
    frame_id: int,
    timestamp_us: int,
) -> bytes:
    jpeg = encode_jpeg(image)
    height, width = image.shape[:2]
    header = CAM2_HEADER.pack(
        int.from_bytes(CAM2_MAGIC, "little"),
        camera_id,
        frame_id,
        len(jpeg),
        width,
        height,
        timestamp_us,
    )
    return header + jpeg


class FakeSerial:
    def __init__(self, data: bytes) -> None:
        self.data = bytearray(data)
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self.data)

    def read(self, size: int) -> bytes:
        chunk = bytes(self.data[:size])
        del self.data[:size]
        return chunk

    def close(self) -> None:
        self.closed = True


class UsbSourceTests(unittest.TestCase):
    def test_parser_handles_noise_split_reads_and_multiple_frames(self) -> None:
        first = cam2_bytes(product_image(False), 0, 10, 111_000)
        second = cam2_bytes(product_image(True), 1, 11, 222_000)
        parser = Cam2FrameParser()
        frames = []
        stream = b"noise" + first + second
        for offset in range(0, len(stream), 17):
            frames.extend(parser.feed(stream[offset : offset + 17], now=1.0 + offset / 1000))
        self.assertEqual([(frame.camera_id, frame.frame_id) for frame in frames], [(0, 10), (1, 11)])
        self.assertEqual(frames[0].timestamp_us, 111_000)

    def test_incomplete_frame_expires_and_parser_resynchronizes(self) -> None:
        parser = Cam2FrameParser(incomplete_timeout=1.0)
        raw = cam2_bytes(product_image(), 0, 1, 10)
        self.assertEqual(parser.feed(raw[:20], now=0.0), [])
        self.assertEqual(parser.feed(b"", now=1.1), [])
        frames = parser.feed(raw, now=1.2)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame_id, 1)

    def test_packet_uses_pc_monotonic_time_not_esp_timestamp(self) -> None:
        image = product_image()
        jpeg = encode_jpeg(image)
        frame = Cam2Frame(1, 42, 160, 120, 99_000_000, jpeg, 12.5)
        packet = cam2_frame_to_packet(frame)
        self.assertEqual(packet.camera_id, "1")
        self.assertEqual(packet.frame_id, 42)
        self.assertEqual(packet.timestamp, 12.5)
        self.assertNotEqual(packet.timestamp, frame.timestamp_us / 1_000_000)

    def test_invalid_jpeg_or_dimensions_are_rejected(self) -> None:
        invalid = Cam2Frame(0, 1, 10, 10, 0, b"\xff\xd8broken\xff\xd9", 1.0)
        with self.assertRaises(ValueError):
            cam2_frame_to_packet(invalid)
        image = product_image()
        mismatch = Cam2Frame(0, 2, 999, 999, 0, encode_jpeg(image), 2.0)
        with self.assertRaisesRegex(ValueError, "크기 불일치"):
            cam2_frame_to_packet(mismatch)

    def test_source_filters_camera_and_closes_connection(self) -> None:
        stream = cam2_bytes(product_image(), 0, 1, 10) + cam2_bytes(
            product_image(True), 1, 2, 20
        )
        fake = FakeSerial(stream)

        def factory(_port: str, **_kwargs: object) -> FakeSerial:
            return fake

        source = UsbCameraSource(
            "COM_TEST",
            1,
            serial_factory=factory,
            clock=lambda: 5.0,
        )
        with source:
            packets = source.poll()
        self.assertTrue(fake.closed)
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].camera_id, "1")
        self.assertEqual(packets[0].frame_id, 2)
        self.assertEqual(source.stats.other_camera_frames, 1)

    def test_serial_disconnect_becomes_safe_error_and_closes(self) -> None:
        class DisconnectedSerial(FakeSerial):
            def read(self, size: int) -> bytes:
                raise OSError("device removed")

        fake = DisconnectedSerial(b"")

        def factory(_port: str, **_kwargs: object) -> FakeSerial:
            return fake

        source = UsbCameraSource(
            "COM_TEST",
            0,
            serial_factory=factory,
            serial_exceptions=(OSError,),
        )
        with self.assertRaisesRegex(UsbCameraError, "USB 연결이 끊겼습니다"):
            with source:
                source.poll()
        self.assertTrue(fake.closed)

    def test_cam2_to_recognizer_mock_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mock = MockVisionClient("CAM2 테스트 제품 500mL")
            recognizer = ObjectRecognizer(
                RecognitionConfig(
                    output_dir=Path(directory),
                    frame_check_interval=0,
                    stable_duration=0.3,
                    motion_threshold=0.03,
                    rotation_threshold=0.25,
                    duplicate_threshold=0.10,
                    min_sharpness=10,
                    phase_timeout=5,
                ),
                mock,
                lambda _message: None,
            )
            parser = Cam2FrameParser()
            recognizer.start_recognition("0")
            front = product_image(False)
            back = product_image(True)
            sequence = [
                *(front for _ in range(5)),
                back,
                *(back for _ in range(5)),
            ]
            timestamps = (0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2)
            for frame_id, (image, received_at) in enumerate(zip(sequence, timestamps)):
                raw = cam2_bytes(image, 0, frame_id, frame_id * 100_000)
                frames = parser.feed(raw, now=received_at)
                self.assertEqual(len(frames), 1)
                recognizer.process_frame(cam2_frame_to_packet(frames[0]))
                if recognizer.state is RecognitionState.COMPLETED:
                    break
            state = recognizer.get_recognition_state()
            self.assertEqual(state["state"], "COMPLETED")
            self.assertEqual(state["result"], "CAM2 테스트 제품 500mL")
            self.assertEqual(mock.calls, 1)
            self.assertTrue(Path(state["merged_path"]).is_file())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from object_recognition.api import MockVisionClient
from object_recognition.image_ops import encode_jpeg
from object_recognition.models import RecognitionConfig, RecognitionState
from object_recognition.recognizer import ObjectRecognizer
from object_recognition.usb_source import (
    BUTTON_CLASSIFIED,
    BUTTON_PRESSED,
    BUTTON_VALID,
    CAM2_HEADER,
    CAM2_MAGIC,
    STREAM_PAUSED,
    Cam2Frame,
    Cam2FrameParser,
    UsbCameraSource,
    UsbCameraError,
    cam2_frame_to_packet,
    resolution_command,
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
    *,
    flags: int = 0,
    button_result: int = 0,
    reserved: int = 0,
) -> bytes:
    jpeg = encode_jpeg(image)
    height, width = image.shape[:2]
    header = CAM2_HEADER.pack(
        int.from_bytes(CAM2_MAGIC, "little"),
        camera_id,
        flags,
        button_result,
        reserved,
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
        self.writes: list[bytes] = []

    @property
    def in_waiting(self) -> int:
        return len(self.data)

    def read(self, size: int) -> bytes:
        chunk = bytes(self.data[:size])
        del self.data[:size]
        return chunk

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class UsbSourceTests(unittest.TestCase):
    def test_old_reserved_header_remains_compatible(self) -> None:
        image = product_image()
        jpeg = encode_jpeg(image)
        old_header = struct.Struct("<IB3xIIHHQ")
        raw = old_header.pack(
            int.from_bytes(CAM2_MAGIC, "little"),
            1,
            7,
            len(jpeg),
            image.shape[1],
            image.shape[0],
            123_000,
        ) + jpeg
        frames = Cam2FrameParser().feed(raw, now=2.0)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].camera_id, 1)
        self.assertEqual(frames[0].flags, 0)
        self.assertIsNone(frames[0].button_pressed)

    def test_new_header_preserves_left_button_metadata(self) -> None:
        raw = cam2_bytes(
            product_image(),
            0,
            8,
            456_000,
            flags=BUTTON_PRESSED | BUTTON_VALID | BUTTON_CLASSIFIED,
            button_result=1,
        )
        frame = Cam2FrameParser().feed(raw, now=3.0)[0]
        self.assertEqual(frame.flags, 7)
        self.assertTrue(frame.button_pressed)
        self.assertTrue(frame.button_classified)
        self.assertEqual(frame.button_result, "SHORT")

    def test_app_parser_matches_latest_proxy_viewer(self) -> None:
        viewer_path = Path(__file__).resolve().parents[2] / "proxy" / "viewer.py"
        spec = importlib.util.spec_from_file_location("videx_proxy_viewer_test", viewer_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: module}):
            spec.loader.exec_module(module)

        raw = cam2_bytes(
            product_image(),
            0,
            9,
            789_000,
            flags=BUTTON_VALID | BUTTON_CLASSIFIED,
            button_result=2,
        )
        app_frame = Cam2FrameParser().feed(raw, now=4.0)[0]
        viewer_frame = module.FrameParser().feed(raw, now=4.0)[0]
        self.assertEqual(
            (
                app_frame.camera_id,
                app_frame.frame_id,
                app_frame.width,
                app_frame.height,
                app_frame.timestamp_us,
                app_frame.button_pressed,
                app_frame.button_classified,
                app_frame.button_result,
                app_frame.jpeg,
                app_frame.paused,
            ),
            (
                viewer_frame.camera_id,
                viewer_frame.frame_id,
                viewer_frame.width,
                viewer_frame.height,
                viewer_frame.timestamp_us,
                viewer_frame.button_pressed,
                viewer_frame.button_classified,
                viewer_frame.button_result,
                viewer_frame.jpeg,
                viewer_frame.paused,
            ),
        )

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

    def test_invalid_header_and_damaged_jpeg_recover_to_next_frame(self) -> None:
        image = product_image()
        jpeg = encode_jpeg(image)
        invalid_header = CAM2_HEADER.pack(
            int.from_bytes(CAM2_MAGIC, "little"),
            1,
            0,
            0,
            1,
            1,
            len(jpeg),
            image.shape[1],
            image.shape[0],
            1,
        ) + jpeg
        damaged = CAM2_HEADER.pack(
            int.from_bytes(CAM2_MAGIC, "little"), 1, 0, 0, 0, 2, 4, 1, 1, 2
        ) + b"nope"
        valid = cam2_bytes(image, 1, 3, 3)
        frames = Cam2FrameParser().feed(invalid_header + damaged + valid, now=1.0)
        self.assertEqual([frame.frame_id for frame in frames], [3])

    def test_stream_paused_is_metadata_not_a_jpeg(self) -> None:
        raw = CAM2_HEADER.pack(
            int.from_bytes(CAM2_MAGIC, "little"),
            1,
            STREAM_PAUSED,
            0,
            0,
            55,
            0,
            0,
            0,
            999,
        )
        frame = Cam2FrameParser().feed(raw, now=5.0)[0]
        self.assertTrue(frame.paused)
        self.assertEqual(frame.jpeg, b"")
        with self.assertRaisesRegex(ValueError, "STREAM_PAUSED"):
            cam2_frame_to_packet(frame)

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
            0,
            serial_factory=factory,
            clock=lambda: 5.0,
        )
        with source:
            packets = source.poll()
        self.assertTrue(fake.closed)
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].camera_id, "0")
        self.assertEqual(packets[0].frame_id, 1)
        self.assertEqual(source.stats.other_camera_frames, 1)
        self.assertEqual(fake.writes, [resolution_command(1)])
        self.assertEqual(source.resolution_request_status, "sent_unconfirmed")

    def test_right_frame_is_filtered_before_jpeg_decode(self) -> None:
        stream = cam2_bytes(product_image(), 0, 1, 10) + cam2_bytes(
            product_image(True), 1, 2, 20
        )
        fake = FakeSerial(stream)

        def factory(_port: str, **_kwargs: object) -> FakeSerial:
            return fake

        from object_recognition import usb_source

        original = usb_source.cam2_frame_to_packet
        with patch(
            "object_recognition.usb_source.cam2_frame_to_packet",
            wraps=original,
        ) as convert:
            with UsbCameraSource(
                "COM_TEST",
                0,
                serial_factory=factory,
                clock=lambda: 5.0,
            ) as source:
                packets = source.poll()
        self.assertEqual([packet.camera_id for packet in packets], ["0"])
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(convert.call_args.args[0].camera_id, 0)
        self.assertEqual(source.stats.other_camera_frames, 1)

    def test_right_pause_is_excluded_from_left_recognition(self) -> None:
        paused = CAM2_HEADER.pack(
            int.from_bytes(CAM2_MAGIC, "little"),
            1,
            STREAM_PAUSED,
            0,
            0,
            7,
            0,
            0,
            0,
            700,
        )
        left = cam2_bytes(product_image(), 0, 8, 800)
        fake = FakeSerial(paused + left)
        from object_recognition import usb_source

        with patch(
            "object_recognition.usb_source.cam2_frame_to_packet",
            wraps=usb_source.cam2_frame_to_packet,
        ) as convert:
            with UsbCameraSource(
                "COM_TEST",
                0,
                serial_factory=lambda *_args, **_kwargs: fake,
                clock=lambda: 6.0,
            ) as source:
                packets = source.poll()
        self.assertEqual([packet.camera_id for packet in packets], ["0"])
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(convert.call_args.args[0].camera_id, 0)
        self.assertTrue(source.last_frame_info_by_camera[1].paused)
        self.assertEqual(source.stats.other_camera_frames, 1)

    def test_default_mode1_command_and_left_actual_resolution(self) -> None:
        image = cv2.resize(product_image(), (640, 480), interpolation=cv2.INTER_CUBIC)
        fake = FakeSerial(cam2_bytes(image, 0, 1, 1000))
        with UsbCameraSource(
            "COM_TEST",
            0,
            serial_factory=lambda *_args, **_kwargs: fake,
            clock=lambda: 8.0,
        ) as source:
            while fake.data:
                source.poll()
            self.assertEqual(source.resolution_mode, 1)
            self.assertEqual(source.selected_actual_resolution, (640, 480))
            self.assertEqual(source.selected_resolution_status, "actual_matches_request")
        self.assertEqual(fake.writes, [resolution_command(1)])

    def test_reopening_usb_requests_mode1_again(self) -> None:
        connections = [FakeSerial(b""), FakeSerial(b"")]

        def factory(_port: str, **_kwargs: object) -> FakeSerial:
            return connections.pop(0)

        first, second = connections
        source = UsbCameraSource("COM_TEST", 0, serial_factory=factory)
        source.open()
        source.close()
        source.open()
        source.close()
        self.assertEqual(first.writes, [resolution_command(1)])
        self.assertEqual(second.writes, [resolution_command(1)])

    def test_resolution_mismatch_logs_warning_after_repeated_frames(self) -> None:
        stream = b"".join(
            cam2_bytes(product_image(), 0, frame_id, frame_id * 1000)
            for frame_id in range(1, 4)
        )
        fake = FakeSerial(stream)
        with UsbCameraSource(
            "COM_TEST",
            0,
            serial_factory=lambda *_args, **_kwargs: fake,
            clock=lambda: 9.0,
        ) as source:
            with self.assertLogs("videx.object_recognition.usb", level="WARNING") as logs:
                source.poll()
        self.assertEqual(source.selected_actual_resolution, (160, 120))
        self.assertEqual(source.selected_resolution_status, "actual_mismatch")
        self.assertIn("Requested resolution was not applied", "\n".join(logs.output))

    def test_resolution_write_failure_keeps_legacy_stream_available(self) -> None:
        class LegacySerial(FakeSerial):
            def write(self, data: bytes) -> int:
                raise OSError("control endpoint unsupported")

        fake = LegacySerial(cam2_bytes(product_image(), 0, 1, 10))
        with self.assertLogs("videx.object_recognition.usb", level="WARNING"):
            with UsbCameraSource(
                "COM_TEST",
                0,
                serial_factory=lambda *_args, **_kwargs: fake,
                serial_exceptions=(OSError,),
                clock=lambda: 7.0,
            ) as source:
                packets = source.poll()
        self.assertEqual(len(packets), 1)
        self.assertEqual(source.resolution_request_status, "failed")

    def test_source_emits_both_cameras_from_one_serial_connection(self) -> None:
        stream = cam2_bytes(product_image(), 0, 1, 10) + cam2_bytes(
            product_image(True), 1, 2, 20
        )
        fake = FakeSerial(stream)
        factory_calls = 0

        def factory(_port: str, **_kwargs: object) -> FakeSerial:
            nonlocal factory_calls
            factory_calls += 1
            return fake

        source = UsbCameraSource(
            "COM_TEST",
            None,
            serial_factory=factory,
            clock=lambda: 5.0,
        )
        with source:
            packets = source.poll()
        self.assertEqual(factory_calls, 1)
        self.assertEqual([packet.camera_id for packet in packets], ["0", "1"])
        self.assertEqual(source.last_packet_at_by_camera, {0: 5.0, 1: 5.0})
        self.assertEqual(source.stats.other_camera_frames, 0)

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

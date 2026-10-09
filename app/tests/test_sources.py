from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from object_recognition.image_ops import decode_image, encode_jpeg
from object_recognition.sources import (
    packet_from_file,
    packets_from_jpeg_bytes,
    packets_from_video,
)


class SourceTests(unittest.TestCase):
    def test_file_and_jpeg_byte_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.jpg"
            path.write_bytes(encode_jpeg(np.full((60, 80, 3), 127, dtype=np.uint8)))
            packet = packet_from_file(path, "file-cam", 7, 1.25)
            self.assertEqual(packet.camera_id, "file-cam")
            self.assertEqual(packet.frame_id, 7)
            self.assertEqual(decode_image(packet.data).shape[:2], (60, 80))
            packets = list(packets_from_jpeg_bytes([packet.data, packet.data], "byte-cam", 5))
            self.assertEqual(len(packets), 2)
            self.assertAlmostEqual(packets[1].timestamp - packets[0].timestamp, 0.2, places=4)

    def test_video_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
            )
            if not writer.isOpened():
                self.skipTest("현재 OpenCV 빌드에 MJPG VideoWriter가 없습니다.")
            for value in (40, 100, 180):
                writer.write(np.full((64, 96, 3), value, dtype=np.uint8))
            writer.release()
            packets = list(packets_from_video(path, "video-cam"))
            self.assertEqual(len(packets), 3)
            self.assertTrue(all(packet.camera_id == "video-cam" for packet in packets))
            self.assertEqual(decode_image(packets[0].data).shape[:2], (64, 96))


if __name__ == "__main__":
    unittest.main()

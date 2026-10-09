import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from proxy.viewer import Frame, FrameParser, HEADER
from app.stereo.capture import freeze_pair
from app.stereo.dataset import CapturedFrame, load_pair, save_pair
from .support import workspace


def jpeg_image(width=320, height=240, value=130):
    image = np.full((height, width, 3), value, np.uint8)
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return data.tobytes()


def wire_frame(camera_id=0, width=320, height=240, frame_id=7, timestamp=123456):
    return Frame(camera_id, frame_id, width, height, timestamp, jpeg_image(width, height))


class CaptureTests(unittest.TestCase):
    def test_actual_viewer_parser_fragmentation_and_capture_metadata(self):
        left, right = wire_frame(0), wire_frame(1, frame_id=44, timestamp=999999)
        packet = b"log noise" + b"".join(
            HEADER.pack(int.from_bytes(b"CAM2", "little"), f.camera_id, f.frame_id,
                        len(f.jpeg), f.width, f.height, f.timestamp_us) + f.jpeg
            for f in (left, right))
        parser, parsed = FrameParser(), []
        for offset in range(0, len(packet), 17):
            parsed.extend(parser.feed(packet[offset:offset + 17], now=0))
        self.assertEqual(parsed, [left, right])
        captured = [CapturedFrame.from_wire(f, 1700000000 + i, 1000000000 + i * 1000000)
                    for i, f in enumerate(parsed)]
        with workspace() as tmp:
            pair = save_pair(tmp, *captured)
            metadata, paths, images = load_pair(pair / "pair.json")
            self.assertEqual(paths[0].read_bytes(), left.jpeg)
            self.assertEqual(paths[1].read_bytes(), right.jpeg)
            self.assertEqual(metadata["left"]["frame_id"], 7)
            self.assertEqual(metadata["right"]["frame_id"], 44)
            self.assertEqual(metadata["left"]["timestamp_us"], 123456)
            self.assertEqual(metadata["right"]["timestamp_us"], 999999)
            self.assertEqual(metadata["pc_receive_delta_ms"], 1)
            self.assertIsNone(metadata["synchronized"])
            self.assertFalse(metadata["physical_camera_mapping_verified"])
            self.assertIn("+00:00", metadata["left"]["pc_received_at_utc"])
            self.assertEqual(images[0].shape, (240, 320, 3))

    def test_incomplete_and_bad_frames_recover_without_false_pair(self):
        frame = wire_frame()
        packet = HEADER.pack(int.from_bytes(b"CAM2", "little"), 0, 7,
                             len(frame.jpeg), 320, 240, 123456) + frame.jpeg
        parser = FrameParser()
        self.assertEqual(parser.feed(packet[:35], now=0), [])
        self.assertEqual(parser.feed(b"", now=7), [])
        self.assertEqual(parser.feed(packet, now=8), [frame])
        bad = HEADER.pack(int.from_bytes(b"CAM2", "little"), 2, 1, 4, 320, 240, 0) + b"abcd"
        self.assertEqual(FrameParser().feed(bad + packet, now=0), [frame])

    def test_missing_mismatch_stale_and_bad_decode_are_rejected(self):
        left = CapturedFrame.from_wire(wire_frame(0), 0, 1_000_000_000)
        right = CapturedFrame.from_wire(wire_frame(1), 0, 1_000_000_000)
        with workspace() as tmp:
            with self.assertRaisesRegex(ValueError, "Both"):
                save_pair(tmp, left, None)
            wrong_size = CapturedFrame.from_wire(wire_frame(1, width=640), 0, 1_000_000_000)
            with self.assertRaisesRegex(ValueError, "resolutions"):
                save_pair(tmp, left, wrong_size)
        with self.assertRaisesRegex(ValueError, "stale"):
            freeze_pair({0: left, 1: right}, 2, now_ns=4_000_000_001)
        self.assertEqual(freeze_pair({0: left, 1: right}, 2, now_ns=2_000_000_000), (left, right))
        with self.assertRaises(ValueError):
            CapturedFrame.from_wire(Frame(0, 1, 320, 240, 0, b"\xff\xd8bad\xff\xd9"))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            CapturedFrame.from_wire(Frame(0, 1, 640, 240, 0, left.jpeg))

    def test_offline_import_keeps_raw_bytes_and_unknown_source_metadata(self):
        with workspace() as tmp:
            source = Path(tmp) / "input.jpg"
            source.write_bytes(jpeg_image())
            left, right = CapturedFrame.from_file(source, 0), CapturedFrame.from_file(source, 1)
            first, second = save_pair(tmp, left, right), save_pair(tmp, left, right)
            self.assertNotEqual(first, second)
            metadata, paths, _ = load_pair(first / "pair.json")
            self.assertEqual(paths[0].read_bytes(), source.read_bytes())
            self.assertIsNone(metadata["left"]["frame_id"])
            self.assertIsNone(metadata["left"]["timestamp_us"])
            self.assertEqual(metadata["left"]["source"], "jpeg_file_import")

    def test_incomplete_pair_is_never_published_on_write_failure(self):
        frames = [CapturedFrame.from_wire(wire_frame(i)) for i in (0, 1)]
        with workspace() as tmp:
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_pair(tmp, *frames)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_manifest_cannot_escape_pair_directory(self):
        frames = [CapturedFrame.from_wire(wire_frame(i)) for i in (0, 1)]
        with workspace() as tmp:
            pair = save_pair(tmp, *frames)
            path = pair / "pair.json"
            metadata = json.loads(path.read_text())
            metadata["left"]["saved_path"] = "../external.jpg"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "within"):
                load_pair(path)


if __name__ == "__main__":
    unittest.main()

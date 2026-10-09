"""Synthetic geometry, temporal decisions, and original CAM2 receiver regression."""
import struct
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

try:
    from . import flood_fill as mvp
    from . import flood_fill_mvp as viewer
    from . import outline
except ImportError:
    import flood_fill as mvp
    import flood_fill_mvp as viewer
    import outline


class FloodTests(unittest.TestCase):
    def setUp(self):
        self.clear = viewer.synthetic(0)
        self.blocked = viewer.synthetic(40)
        self.detector = mvp.Detector()

    def update(self, image, frame_id, at=None):
        at = frame_id * .1 if at is None else at
        return self.detector.update(image, frame_id, at, at)

    def test_reuses_original_functions(self):
        self.assertIs(mvp.extract_outline, outline.extract_outline)
        self.assertIs(viewer.receive_frames, outline.receive_frames)

    def test_clear_scene_fills_roi_only(self):
        r = mvp.analyze(self.clear, mvp.Config())
        self.assertEqual(r['total'], 0.)
        self.assertFalse(r['filled'][~r['roi']].any())
        self.assertTrue(np.array_equal(r['filled'] | r['unfilled'], r['roi']))
        self.assertFalse((r['filled'] & r['unfilled']).any())
        self.assertIsNone(r['invalid'])

    def test_central_object_requires_three_new_frames(self):
        for frame_id in (1, 2):
            self.update(self.blocked, frame_id)
            self.assertEqual(self.detector.state, 'MONITORING')
        self.update(self.blocked, 3)
        self.assertEqual(self.detector.state, 'WARNING')
        self.assertGreater(self.detector.result['total'], .18)
        self.assertGreater(self.detector.result['central'], .30)

    def test_duplicate_id_does_not_confirm_or_replace_measurement(self):
        self.update(self.blocked, 1)
        r = self.detector.result
        for _ in range(20):
            self.assertFalse(self.update(self.clear, 1, .2))
        self.assertEqual(self.detector.hits, 1)
        self.assertEqual(self.detector.processed, 1)
        self.assertIs(self.detector.result, r)

    def test_single_flash_does_not_warn(self):
        for i in range(1, 10):
            self.update(self.blocked if i % 2 else self.clear, i)
            self.assertEqual(self.detector.state, 'MONITORING')

    def test_three_clear_frames_release_warning(self):
        for i in range(1, 4):
            self.update(self.blocked, i)
        for i in (4, 5):
            self.update(self.clear, i)
            self.assertEqual(self.detector.state, 'WARNING')
        self.update(self.clear, 6)
        self.assertEqual(self.detector.state, 'MONITORING')

    def test_missing_frames_unknown_and_reconfirmation(self):
        for i in range(1, 4):
            self.update(self.blocked, i)
        self.detector.tick(3.)
        self.assertEqual(self.detector.state, 'UNKNOWN')
        self.update(self.blocked, 4, 3.1)
        self.assertEqual(self.detector.state, 'MONITORING')
        self.assertEqual(self.detector.hits, 1)

    def test_duplicate_stream_still_times_out(self):
        self.update(self.blocked, 1)
        self.update(self.blocked, 1, 5.)
        self.assertEqual(self.detector.state, 'UNKNOWN')
        self.assertEqual(self.detector.last_received, .1)

    def test_frame_counter_restart_resets_streak(self):
        self.update(self.blocked, 10, .1)
        self.update(self.blocked, 11, .2)
        self.update(self.blocked, 0, .3)
        self.assertEqual(self.detector.hits, 1)
        self.assertEqual(self.detector.state, 'MONITORING')

    def test_stale_arrival_not_counted(self):
        self.assertFalse(self.detector.update(self.blocked, 1, 0., 3.))
        self.assertEqual(self.detector.processed, 0)
        self.assertEqual(self.detector.state, 'UNKNOWN')

    def test_dark_and_overexposed_unknown(self):
        for i, value in enumerate((0, 255)):
            self.update(np.full_like(self.clear, value), i)
            self.assertEqual(self.detector.state, 'UNKNOWN')

    def test_object_outside_roi_does_not_warn(self):
        image = self.clear.copy()
        cv2.rectangle(image, (0, 20), (60, 100), (0, 0, 0), -1)
        r = mvp.analyze(image, mvp.Config())
        self.assertEqual(r['total'], 0.)

    def test_side_object_does_not_meet_center_threshold(self):
        image = self.clear.copy()
        cv2.rectangle(image, (20, 220), (75, 280), (0, 0, 0), -1)
        for i in range(1, 5):
            self.update(image, i)
        self.assertEqual(self.detector.result['central'], 0.)
        self.assertEqual(self.detector.state, 'MONITORING')

    def test_small_gap_closed_but_large_gap_leaks(self):
        edges = np.zeros(self.clear.shape[:2], np.uint8)
        cv2.rectangle(edges, (80, 165), (160, 260), 255, 1)
        tiny = edges.copy()
        tiny[165, 119:122] = 0
        wide = edges.copy()
        wide[165, 105:136] = 0
        with patch.object(mvp, 'extract_outline', return_value=tiny):
            sealed = mvp.analyze(self.clear, mvp.Config())
        with patch.object(mvp, 'extract_outline', return_value=wide):
            leaky = mvp.analyze(self.clear, mvp.Config())
        self.assertGreater(sealed['total'], .18)
        self.assertLess(leaky['total'], .08)

    def test_seed_blocked_unknown(self):
        edges = np.zeros(self.clear.shape[:2], np.uint8)
        edges[285:311, 105:135] = 255
        with patch.object(mvp, 'extract_outline', return_value=edges):
            self.update(self.clear, 1)
        self.assertEqual(self.detector.state, 'UNKNOWN')
        self.assertIn('seed blocked', self.detector.reason)

    def test_seed_moves_locally_off_edge(self):
        edges = np.zeros(self.clear.shape[:2], np.uint8)
        edges[300, 120] = 255
        with patch.object(mvp, 'extract_outline', return_value=edges):
            r = mvp.analyze(self.clear, mvp.Config())
        self.assertIsNotNone(r['seed'])
        self.assertFalse(r['barriers'][r['seed'][1], r['seed'][0]])
        self.assertIsNone(r['invalid'])

    def test_render_valid_and_unknown(self):
        self.update(self.blocked, 1)
        self.assertEqual(viewer.render(self.detector, .1, 'TEST').shape, (740, 1280, 3))
        self.detector.tick(5.)
        self.assertEqual(viewer.render(self.detector, 5., 'TEST').shape, (740, 1280, 3))

    def test_original_cam2_receiver_partial_reads_and_rotation(self):
        # A real JPEG in fragmented CAM2 packets traverses the original receiver.
        raw = cv2.rotate(self.blocked, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ok, jpeg = cv2.imencode('.jpg', raw)
        self.assertTrue(ok)
        header = outline.HEADER.pack(struct.unpack('<I', b'CAM2')[0], 0, 42,
                                     len(jpeg), raw.shape[1], raw.shape[0], 123456)
        packet = header + jpeg.tobytes()
        stop = threading.Event()
        latest = {}
        class Port:
            def __init__(self):
                self.data = bytearray(packet)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            @property
            def in_waiting(self):
                return min(17, len(self.data))
            def read(self, size):
                data = bytes(self.data[:size])
                del self.data[:size]
                if not self.data:
                    stop.set()
                return data
        class Serial:
            SerialException = OSError
            @staticmethod
            def Serial(*args, **kwargs):
                return Port()
        outline.receive_frames('mock', 115200, stop, latest, threading.Lock(), Serial, cv2, np)
        decoded, frame_id, received = latest[0]
        corrected = cv2.rotate(decoded, cv2.ROTATE_90_CLOCKWISE)
        self.assertEqual(corrected.shape, self.blocked.shape)
        self.assertEqual(frame_id, 42)
        self.assertGreater(mvp.analyze(corrected, mvp.Config())['total'], .18)


if __name__ == '__main__':
    unittest.main()

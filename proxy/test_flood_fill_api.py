"""Public API and captured pre-refactor behavior, without USB hardware."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

if __package__:
    from .flood_fill import Config, DetectionResult, Detector
    from .flood_fill_mvp import synthetic
else:
    from flood_fill import Config, DetectionResult, Detector
    from flood_fill_mvp import synthetic


def regression_sequence():
    """Fixed stimuli; expected decisions were captured from the original MVP."""
    clear, blocked = synthetic(0), synthetic(40)
    outside = clear.copy()
    cv2.rectangle(outside, (0, 20), (60, 100), (0, 0, 0), -1)
    yield 'clear', clear, 0, 0., 0.
    yield 'outside_roi', outside, 1, .1, .1
    yield 'candidate_1', blocked, 2, .2, .2
    yield 'duplicate', clear, 2, .25, .25
    yield 'candidate_2', blocked, 3, .3, .3
    yield 'warning', blocked, 4, .4, .4
    yield 'clear_1_hold', clear, 5, .5, .5
    yield 'clear_2_hold', clear, 6, .6, .6
    yield 'clear_3_release', clear, 7, .7, .7
    for i in range(8, 11):
        yield f'warning_again_{i}', blocked, i, i * .1, i * .1
    yield 'duplicate_timeout', blocked, 10, 4., 4.
    yield 'recovery_1', blocked, 11, 4.1, 4.1
    yield 'recovery_2', blocked, 12, 4.2, 4.2
    yield 'recovery_3', blocked, 13, 4.3, 4.3
    yield 'counter_restart', blocked, 0, 4.4, 4.4
    yield 'larger_frame', cv2.resize(blocked, (480, 640)), 1, 4.5, 4.5
    yield 'landscape_frame', cv2.resize(blocked, (320, 240)), 2, 4.6, 4.6
    yield 'dark', np.zeros_like(clear), 3, 4.7, 4.7
    yield 'bright', np.full_like(clear, 255), 4, 4.8, 4.8
    yield 'too_small', np.full((16, 16, 3), 145, np.uint8), 5, 4.9, 4.9
    yield 'stale_arrival', clear, 6, 5., 8.


def decision_record(detector, updated):
    r = detector.result
    return dict(state=detector.state, reason=detector.reason,
                updated=updated, frame_id=detector.last_id,
                hits=detector.hits, clears=detector.clears, processed=detector.processed,
                total=r['total'] if r else None, central=r['central'] if r else None,
                seed=list(r['seed']) if r and r['seed'] else None,
                shape=list(r['roi'].shape) if r else None,
                mask_counts={key: int(np.count_nonzero(r[key])) for key in
                             ('roi', 'center', 'barriers', 'filled', 'unfilled')} if r else None)


class ApiTests(unittest.TestCase):
    def test_captured_pre_refactor_sequence(self):
        baseline = json.loads((Path(__file__).parent / 'fixtures' / 'flood_fill_baseline.json').read_text())
        detector = Detector()
        for name, image, frame_id, received, now in regression_sequence():
            with self.subTest(name=name):
                updated = detector.update(image, frame_id, received, now)
                actual = decision_record(detector, updated)
                expected = baseline['decisions'][name]
                self.assertEqual(actual, expected)

    def test_process_frame_only_and_return_fields(self):
        detector = Detector()
        with patch('time.monotonic', side_effect=[1., 1.1, 1.2]):
            results = [detector.process(synthetic(40)) for _ in range(3)]
        result = results[-1]
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.state, 'WARNING')
        self.assertTrue(result.valid)
        self.assertTrue(result.updated)
        self.assertGreater(result.unfilled_ratio, .18)
        self.assertGreater(result.central_ratio, .30)
        self.assertEqual(result.analysis['unfilled'].dtype, np.bool_)
        self.assertEqual(results[0].state, 'MONITORING')

    def test_timestamp_only_duplicate_and_timeout(self):
        detector = Detector()
        first = detector.process(synthetic(40), 10.)
        duplicate = detector.process(synthetic(0), 10.)
        self.assertFalse(duplicate.updated)
        self.assertEqual(duplicate.consecutive_hits, 1)
        self.assertIs(duplicate.analysis, first.analysis)
        stale = detector.poll(13.)
        self.assertEqual(stale.state, 'UNKNOWN')
        self.assertFalse(stale.valid)
        self.assertFalse(stale.updated)
        recovered = detector.process(synthetic(40), 13.1)
        self.assertEqual(recovered.consecutive_hits, 1)

    def test_explicit_frame_ids_take_precedence_over_timestamp(self):
        detector = Detector()
        for i in range(3):
            result = detector.process(synthetic(40), 10., frame_id=i)
        self.assertEqual(result.state, 'WARNING')
        self.assertFalse(detector.process(synthetic(40), 10.5, frame_id=2).updated)

    def test_stale_queued_frame_and_initial_poll(self):
        detector = Detector()
        empty = detector.poll(0.)
        self.assertIsNone(empty.analysis)
        self.assertIsNone(empty.unfilled_ratio)
        result = detector.process(synthetic(40), 1., now=4.)
        self.assertFalse(result.updated)
        self.assertEqual(result.state, 'UNKNOWN')

    def test_changed_dimensions_rebuild_masks_keep_temporal_history(self):
        detector = Detector()
        for i, size in enumerate(((240, 320), (480, 640), (320, 240))):
            result = detector.process(cv2.resize(synthetic(40), size), i * .1)
            self.assertEqual(result.analysis['roi'].shape, (size[1], size[0]))
        # Original behavior: size changes do not reset the confirmation streak.
        self.assertEqual(result.state, 'WARNING')
        self.assertEqual(result.consecutive_hits, 3)

    def test_inputs_are_not_modified(self):
        frame = synthetic(40)
        before = frame.copy()
        Detector().process(frame)
        np.testing.assert_array_equal(frame, before)

    def test_invalid_frame_rejected_without_consuming_id(self):
        detector = Detector()
        for frame in (None, np.zeros((20, 20), np.uint8),
                      np.zeros((20, 20, 3), np.float32), np.zeros((0, 20, 3), np.uint8)):
            with self.subTest(frame_type=type(frame)):
                with self.assertRaises((TypeError, ValueError)):
                    detector.process(frame, 1., frame_id=7)
                self.assertEqual(detector.processed, 0)
                self.assertIsNone(detector.last_id)

    def test_invalid_settings_and_times(self):
        for kwargs in ({'kernel': 4}, {'confirm': 0}, {'total_threshold': float('nan')},
                       {'stale': float('inf')}, {'center_bounds': (.8, 0., .2, 1.)},
                       {'roi_vertices': ((0., 0.),)}, {'low': 60, 'high': 50}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        with self.assertRaises(ValueError):
            Detector().process(synthetic(0), float('nan'))

    def test_core_has_no_viewer_import(self):
        # GUI functions and serial must never be called by the public API.
        with patch.object(cv2, 'imshow', side_effect=AssertionError('GUI called')):
            result = Detector().process(synthetic(0), 1.)
        self.assertEqual(result.state, 'MONITORING')


if __name__ == '__main__':
    unittest.main()

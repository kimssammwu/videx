"""Tests for signed depth and reference-only updates, without camera hardware."""

from copy import deepcopy
import unittest

import numpy as np

from app.stereo.calibration import Observation
from app.stereo.validate import cross_validation_splits, evaluate_points, q_points, reference_copy


class ValidationTests(unittest.TestCase):
    def test_q_signed_disparity_and_zero_depth_singularity(self):
        q = np.array([[1, 0, 0, -120], [0, 1, 0, -160],
                      [0, 0, 0, 300], [0, 0, -1/95, 0]], float)
        xyz, valid = q_points(q, [[120, 160]]*3, [-30, 30, 0])
        np.testing.assert_allclose(xyz[:2, 2], [950, -950])
        self.assertEqual(valid.tolist(), [True, True, False])
        self.assertTrue(np.isnan(xyz[2]).all())

    def test_cheirality_agrees_with_actual_projection_for_positive_tx(self):
        k = np.array([[300, 0, 120], [0, 300, 160], [0, 0, 1]], float)
        p1 = np.column_stack((k, np.zeros(3)))
        p2 = np.column_stack((k, k @ np.array([95, 0, 0])))
        q = np.array([[1, 0, 0, -120], [0, 1, 0, -160],
                      [0, 0, 0, 300], [0, 0, -1/95, 0]], float)
        xyz = np.array([[0, 0, 950], [50, -30, 600]], float)
        xh = np.column_stack((xyz, np.ones(2)))
        images = [xh @ p.T for p in (p1, p2)]
        points = [(image[:, :2]/image[:, 2:3]).reshape(-1, 1, 2) for image in images]
        result = {"length_unit": "mm", "matrices": {"K_left": k, "K_right": k,
                  "D_left": np.zeros(5), "D_right": np.zeros(5), "R": np.eye(3),
                  "T": [[95], [0], [0]], "R1": np.eye(3), "R2": np.eye(3),
                  "P1": p1, "P2": p2, "Q": q}}
        metric, arrays = evaluate_points(result, Observation("synthetic", *points, (240, 320)))
        self.assertEqual(metric["both_cameras_positive_depth_count"], 2)
        self.assertTrue(np.all(arrays["disparity"] < 0))
        np.testing.assert_allclose(arrays["q_depth_mm"], xyz[:, 2], atol=1e-6)

    def test_baseline_reference_update_preserves_every_matrix_and_original(self):
        result = {"quality_settings": {"expected_baseline_cm": 9, "baseline_relative_tolerance": 0.2},
                  "length_unit": "mm", "matrices": {"T": [[97.23], [0], [0]], "Q": [[1, 2], [3, 4]]},
                  "report": {"warnings": ["some unrelated warning"], "numeric_checks_passed": False}}
        previous = deepcopy(result)
        updated = reference_copy(result, 95)
        self.assertEqual(previous, result)
        self.assertEqual(updated["matrices"], result["matrices"])
        self.assertEqual(updated["quality_settings"]["expected_baseline_cm"], 9.5)
        self.assertAlmostEqual(updated["report"]["baseline_relative_difference"], 2.23/95)
        self.assertFalse(updated["report"]["numeric_checks_passed"])
        self.assertEqual(updated["report"]["warnings"], ["some unrelated warning"])

    def test_cv_has_no_train_test_overlap_and_keeps_original_minimum(self):
        splits = cross_validation_splits(14)
        self.assertEqual(len(splits), 18)
        for scheme, train, test in splits:
            self.assertGreaterEqual(len(train), 10)
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(train) | set(test), set(range(14)))
        for scheme in ("leave_one_out", "chronological_blocks"):
            tests = [i for name, train, test in splits if name == scheme for i in test]
            self.assertEqual(sorted(tests), list(range(14)))


if __name__ == "__main__":
    unittest.main()

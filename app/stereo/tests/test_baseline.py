import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.stereo.baseline import StereoResidual, angles_from_translation, direction_translation
from app.stereo.calibration import (BoardSettings, Observation, QualitySettings, corner_consistency,
                                   detect_corners, fit_calibration, validate_result)
from app.stereo.compare_baseline import evaluate_geometry, scale_diagnostic
from .support import BOARD, CAMERA, SIZE, STEREO_R, poses, render_checkerboard


def observations95():
    rng = np.random.default_rng(42)
    translation = np.array([-94.9, 2., 1.])
    translation *= 95 / np.linalg.norm(translation)
    observations = []
    for i, (r, t) in enumerate(poses()):
        rr = cv2.Rodrigues(STEREO_R @ cv2.Rodrigues(r)[0])[0]
        tr = STEREO_R @ t + translation.reshape(3, 1)
        left = cv2.projectPoints(BOARD.object_points(), r, t, CAMERA, None)[0]
        right = cv2.projectPoints(BOARD.object_points(), rr, tr, CAMERA, None)[0]
        left += rng.normal(0, .03, left.shape).astype(np.float32)
        right += rng.normal(0, .03, right.shape).astype(np.float32)
        observations.append(Observation(str(i), left, right, SIZE))
    return observations, translation


class FixedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations, cls.translation = observations95()
        cls.quality = QualitySettings(expected_baseline_cm=9.5)
        cls.result = fit_calibration(cls.observations, BOARD, cls.quality, baseline_mm=95)

    def test_exact_length_free_rotation_and_known_geometry(self):
        m, report = self.result["matrices"], self.result["report"]
        self.assertAlmostEqual(np.linalg.norm(m["T"]), 95, places=10)
        # Independently fitted intrinsics also vary under pixel noise; tolerance
        # is 0.6% of the known length while length itself must remain exact.
        self.assertLess(np.linalg.norm(np.asarray(m["T"]).ravel() - self.translation), .5)
        error = cv2.Rodrigues(np.asarray(m["R"]) @ STEREO_R.T)[0]
        self.assertLess(np.linalg.norm(error), .002)
        self.assertGreater(np.linalg.norm(cv2.Rodrigues(np.asarray(m["R"]))[0]), .004)
        self.assertLess(report["stereo_rms_px"], .1)
        self.assertLess(report["vertical_rms_px"], .1)
        self.assertTrue(report["optimizer"]["converged"])
        self.assertTrue(report["numeric_checks_passed"], report["warnings"])
        self.assertAlmostEqual(report["stereo_rms_px"]**2,
                               (report["stereo_left_reprojection_rms_px"]**2
                                + report["stereo_right_reprojection_rms_px"]**2) / 2, places=10)

    def test_mm_constraint_converts_to_board_unit(self):
        result = fit_calibration(self.observations, BoardSettings(9, 6, 2.4, "cm"), self.quality, baseline_mm=95)
        self.assertAlmostEqual(np.linalg.norm(result["matrices"]["T"]), 9.5, places=10)
        self.assertAlmostEqual(result["report"]["baseline_cm"], 9.5, places=10)
        self.assertLess(result["report"]["vertical_rms_px"], .1)

    def test_constraint_alone_does_not_make_bad_correspondence_pass(self):
        bad = [Observation(o.pair_id, o.left, o.right[::-1].copy() if i == 0 else o.right, SIZE)
               for i, o in enumerate(self.observations)]
        result = fit_calibration(bad, BOARD, self.quality, baseline_mm=95)
        self.assertAlmostEqual(np.linalg.norm(result["matrices"]["T"]), 95, places=10)
        self.assertFalse(result["report"]["numeric_checks_passed"])
        self.assertGreater(result["report"]["stereo_rms_px"], 1.5)

    def test_scaling_translation_preserves_image_alignment(self):
        scaled = scale_diagnostic(self.result, 60)
        self.assertEqual(scaled["matrices"]["R"], self.result["matrices"]["R"])
        for key in ("R1", "R2"):
            np.testing.assert_allclose(scaled["matrices"][key], self.result["matrices"][key], atol=1e-12)
        original = evaluate_geometry(self.result, self.observations[:3], BOARD)
        diagnostic = evaluate_geometry(scaled, self.observations[:3], BOARD)
        self.assertAlmostEqual(original["vertical_rms_px"], diagnostic["vertical_rms_px"], places=6)
        self.assertGreater(diagnostic["stereo_rms_px"], original["stereo_rms_px"] + 1)
        self.assertFalse(scaled["report"]["numeric_checks_passed"])

    def test_invalid_lengths_are_rejected(self):
        for length in (0, -95, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fit_calibration(self.observations, BOARD, baseline_mm=length)
        broken = copy.deepcopy(self.result)
        broken["matrices"]["T"] = (np.asarray(broken["matrices"]["T"]) * 2).tolist()
        with self.assertRaisesRegex(ValueError, "declared baseline constraint"):
            validate_result(broken)

    def test_analytic_jacobian_matches_finite_differences(self):
        o = self.observations[0]
        r, t = next(poses(1))
        objective = StereoResidual(BOARD.object_points(), [o.left], [o.right], CAMERA, None, CAMERA, None, 95)
        parameters = np.r_[cv2.Rodrigues(STEREO_R)[0].ravel(), angles_from_translation(self.translation), r, t.ravel()]
        analytic = objective.evaluate(parameters, True)
        numerical = np.empty_like(analytic)
        for i in range(len(parameters)):
            step = 1e-5 if i >= 8 else 1e-6
            delta = np.zeros_like(parameters)
            delta[i] = step
            numerical[:, i] = (objective.evaluate(parameters + delta) - objective.evaluate(parameters - delta)) / (2 * step)
        np.testing.assert_allclose(analytic, numerical, atol=1e-4, rtol=1e-5)
        for angles in ([0, 0], [3, .4], [-2, -.8]):
            self.assertAlmostEqual(np.linalg.norm(direction_translation(angles, 95)[0]), 95, places=10)


class ConsistentCornerTests(unittest.TestCase):
    def test_index_reversal_is_explicit_and_non_mutating(self):
        observations, _ = observations95()
        o = observations[0]
        reversed_points = o.right[::-1].copy()
        original = reversed_points.copy()
        unchanged, audit = corner_consistency(o.left, reversed_points, BOARD)
        self.assertFalse(audit["consistent_parallel_directions"])
        np.testing.assert_array_equal(unchanged, original)
        corrected, audit = corner_consistency(o.left, reversed_points, BOARD, align=True)
        self.assertTrue(audit["right_order_reversed"])
        self.assertTrue(audit["consistent_parallel_directions"])
        np.testing.assert_array_equal(corrected, o.right)
        np.testing.assert_array_equal(reversed_points, original)

    def test_fallback_window_stays_below_neighbor_spacing(self):
        r, t = next(poses(1))
        image = cv2.resize(render_checkerboard(r, t), (320, 240))
        original_subpix = cv2.cornerSubPix
        windows = []

        def subpix(gray, corners, window, *args):
            windows.append(window)
            grid = corners.reshape(BOARD.rows, BOARD.columns, 2)
            pitch = min(np.linalg.norm(np.diff(grid, axis=0), axis=2).min(),
                        np.linalg.norm(np.diff(grid, axis=1), axis=2).min())
            self.assertLess(2 * window[0], pitch)
            return original_subpix(gray, corners, window, *args)

        with patch("cv2.findChessboardCornersSB", return_value=(False, None)), patch("cv2.cornerSubPix", side_effect=subpix):
            points = detect_corners(image, BOARD, "consistent")
        self.assertIsNotNone(points)
        self.assertEqual(len(windows), 1)
        _, audit = corner_consistency(points, points, BOARD)
        self.assertLess(audit["homography_rms_px"][0], .5)


if __name__ == "__main__":
    unittest.main()

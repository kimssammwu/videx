import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.stereo.calibration import detect_corners
from app.stereo.diagnose import cosine, grid_metrics, inspect_detection, numbered_corners
from .support import BOARD, poses, render_checkerboard


class DiagnosticTests(unittest.TestCase):
    def test_grid_and_180_direction_conflict_are_separate_checks(self):
        points = (BOARD.object_points()[:, :2] / BOARD.square_size * 10 + [20, 30]).astype(np.float32).reshape(-1, 1, 2)
        normal = grid_metrics(points, BOARD)
        reverse = grid_metrics(points[::-1].copy(), BOARD)
        self.assertLess(normal["homography_rms_px"], 1e-4)
        self.assertLess(reverse["homography_rms_px"], 1e-4)
        self.assertAlmostEqual(cosine(normal["row_axis"], reverse["row_axis"]), -1)
        self.assertAlmostEqual(cosine(normal["column_axis"], reverse["column_axis"]), -1)

    def test_fallback_inspection_reproduces_production_without_mutating_image(self):
        rotation, translation = next(poses())
        image = cv2.resize(render_checkerboard(rotation, translation), (320, 240), interpolation=cv2.INTER_AREA)
        before = image.copy()
        with patch("cv2.findChessboardCornersSB", return_value=(False, None)):
            current, info, alternatives = inspect_detection(image, BOARD)
            expected = detect_corners(image, BOARD)
        self.assertIsNotNone(current)
        np.testing.assert_allclose(current, expected)
        np.testing.assert_array_equal(image, before)
        self.assertEqual(info["detector"], "classic_fallback")
        self.assertIn("classic_initial", alternatives)
        self.assertIn("classic_window_5", alternatives)

    def test_numbered_corner_debug_and_failed_detection_preserve_pixels(self):
        image = np.full((240, 320, 3), 160, np.uint8)
        corners = (BOARD.object_points()[:, :2] / BOARD.square_size * 10 + [20, 30]).astype(np.float32).reshape(-1, 1, 2)
        before = image.copy()
        output = numbered_corners(image, corners, BOARD, "debug")
        self.assertEqual(output.shape, (762, 960, 3))
        self.assertEqual(numbered_corners(image, None, BOARD, "missing").shape, output.shape)
        np.testing.assert_array_equal(image, before)


if __name__ == "__main__":
    unittest.main()

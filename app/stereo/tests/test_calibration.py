import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import cv2
import numpy as np

from app.stereo.calibration import (
    BoardSettings, QualitySettings, calibrate_dataset, detect_corners,
    fit_calibration, load_result, save_result,
)
from app.stereo.dataset import CapturedFrame, save_pair
from app.stereo.rectify import epipolar_preview, rectify_pair
from .support import (
    BOARD, SIZE, poses, projected_observations, render_checkerboard,
    right_pose, workspace, write_jpeg,
)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations = projected_observations()
        cls.result = fit_calibration(cls.observations, BOARD)

    def test_known_geometry_reprojection_and_rectification(self):
        report = self.result["report"]
        self.assertAlmostEqual(report["baseline_cm"], 9, delta=0.15)
        self.assertLess(report["left_reprojection_rms_px"], 0.15)
        self.assertLess(report["right_reprojection_rms_px"], 0.15)
        self.assertLess(report["stereo_rms_px"], 0.2)
        self.assertLess(report["vertical_rms_px"], 0.2)
        self.assertTrue(report["numeric_checks_passed"], report["warnings"])
        self.assertEqual(report["validation_status"], "physical_measurement_unverified")
        self.assertEqual(report["used_pairs"], 12)

    def test_length_unit_is_preserved(self):
        result = fit_calibration(self.observations, BoardSettings(9, 6, 2.4, "cm"))
        self.assertEqual(result["length_unit"], "cm")
        self.assertAlmostEqual(result["report"]["baseline_cm"], 9, delta=0.15)
        self.assertAlmostEqual(np.linalg.norm(result["matrices"]["T"]), 9, delta=0.15)

    def test_low_count_and_bad_numeric_quality_require_review(self):
        low_count = fit_calibration(self.observations[:4], BOARD)
        self.assertFalse(low_count["report"]["numeric_checks_passed"])
        self.assertTrue(any("Only 4" in warning for warning in low_count["report"]["warnings"]))
        noisy = fit_calibration(projected_observations(noise=3), BOARD)
        self.assertFalse(noisy["report"]["numeric_checks_passed"])
        self.assertTrue(any("RMS" in warning for warning in noisy["report"]["warnings"]))

    def test_too_few_and_invalid_settings_are_errors(self):
        with self.assertRaisesRegex(ValueError, "At least 3"):
            fit_calibration(self.observations[:2], BOARD)
        with self.assertRaises(ValueError):
            BoardSettings(9, 6, 0)
        with self.assertRaises(ValueError):
            BoardSettings(9, 6, float("nan"))
        with self.assertRaises(ValueError):
            BoardSettings(9.5, 6, 24)
        with self.assertRaises(ValueError):
            QualitySettings(min_pairs=2)
        with self.assertRaises(ValueError):
            QualitySettings(max_vertical_rms_px=float("inf"))
        with workspace() as tmp:
            with self.assertRaisesRegex(ValueError, "No pair_"):
                calibrate_dataset(tmp, BOARD)

    def test_parameter_roundtrip_no_overwrite_and_input_size_validation(self):
        with workspace() as tmp:
            path = Path(tmp) / "calibration.json"
            save_result(path, self.result)
            restored = load_result(path)
            self.assertEqual(restored, self.result)
            with self.assertRaises(FileExistsError):
                save_result(path, self.result)
            image = np.full((SIZE[1], SIZE[0], 3), 120, np.uint8)
            left, right = rectify_pair(image, image, restored)
            self.assertEqual(left.shape, image.shape)
            self.assertEqual(right.shape, image.shape)
            self.assertEqual(epipolar_preview(left, right).shape, (720, 1920, 3))
            with self.assertRaisesRegex(ValueError, "resolution"):
                rectify_pair(image[:100], image, restored)
            broken = copy.deepcopy(self.result)
            broken["matrices"]["T"] = [[float("nan")], [0], [0]]
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "matrix"):
                load_result(path)
            broken = copy.deepcopy(self.result)
            broken["matrices"]["D_left"] = [[0, 0], [0, 0]]
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "distortion"):
                load_result(path)


class RenderedDatasetTests(unittest.TestCase):
    def test_real_jpeg_detection_calibration_exclusions_and_cli(self):
        with workspace() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            pair_paths = []
            for rotation, translation in poses():
                left_file, right_file = root / "left.jpg", root / "right.jpg"
                write_jpeg(left_file, render_checkerboard(rotation, translation))
                rr, tr = right_pose(rotation, translation)
                write_jpeg(right_file, render_checkerboard(rr, tr))
                pair_paths.append(save_pair(dataset, CapturedFrame.from_file(left_file, 0),
                                            CapturedFrame.from_file(right_file, 1)))
            # A valid JPEG without a board, a missing JPEG, a duplicate, and mismatched resolution.
            blank = np.full((720, 960, 3), 150, np.uint8)
            write_jpeg(root / "blank.jpg", blank)
            save_pair(dataset, CapturedFrame.from_file(root / "blank.jpg", 0),
                      CapturedFrame.from_file(root / "blank.jpg", 1))
            missing = save_pair(dataset, CapturedFrame.from_file(left_file, 0),
                                CapturedFrame.from_file(right_file, 1))
            (missing / "right.jpg").unlink()
            save_pair(dataset, CapturedFrame.from_file(left_file, 0), CapturedFrame.from_file(right_file, 1))
            write_jpeg(root / "small.jpg", np.full((240, 320, 3), 150, np.uint8))
            save_pair(dataset, CapturedFrame.from_file(root / "small.jpg", 0),
                      CapturedFrame.from_file(root / "small.jpg", 1))
            result = calibrate_dataset(dataset, BOARD)
            report = result["report"]
            self.assertEqual(report["total_pairs"], 16)
            self.assertEqual(report["used_pairs"], 12, report["excluded"])
            self.assertEqual(report["excluded_pairs"], 4)
            self.assertEqual(report["detection_failed_pairs"], 1)
            self.assertAlmostEqual(report["baseline_cm"], 9, delta=0.3)
            self.assertLess(report["vertical_rms_px"], 0.5)
            path = root / "calibration.json"
            save_result(path, result)
            rectification = subprocess.run([
                sys.executable, "-B", "-m", "app.stereo.rectify", "--parameters", str(path),
                "--left", str(pair_paths[0] / "left.jpg"), "--right", str(pair_paths[0] / "right.jpg"),
                "--output", str(root / "preview")], capture_output=True, text=True)
            self.assertEqual(rectification.returncode, 0, rectification.stderr)
            self.assertTrue((root / "preview" / "rectified.png").is_file())
            imported = subprocess.run([
                sys.executable, "-B", "-m", "app.stereo.capture", "import",
                "--left", str(left_file), "--right", str(right_file),
                "--output", str(root / "imported"), "--save-without-preview"], capture_output=True, text=True)
            self.assertEqual(imported.returncode, 0, imported.stderr)
            cli = subprocess.run([
                sys.executable, "-B", "-m", "app.stereo.calibration", "--dataset", str(dataset),
                "--columns", "9", "--rows", "6", "--square-size", "24", "--unit", "mm",
                "--min-pairs", "20", "--output", str(root / "cli.json")], capture_output=True, text=True)
            self.assertEqual(cli.returncode, 2, cli.stderr)
            self.assertIn("REVIEW REQUIRED", cli.stdout)
            self.assertIn("UNVERIFIED", cli.stdout)
            self.assertTrue((root / "cli.json").is_file())
            print(f"\nSynthetic JPEG integration: used={report['used_pairs']}, excluded={report['excluded_pairs']}, "
                  f"baseline={report['baseline_cm']:.4f}cm, vertical RMS={report['vertical_rms_px']:.4f}px")

    def test_detector_finds_inner_corner_grid_and_rejects_blank(self):
        rotation, translation = next(poses(1))
        corners = detect_corners(render_checkerboard(rotation, translation), BOARD)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (54, 1, 2))
        self.assertIsNone(detect_corners(np.zeros((240, 320, 3), np.uint8), BOARD))


if __name__ == "__main__":
    unittest.main()

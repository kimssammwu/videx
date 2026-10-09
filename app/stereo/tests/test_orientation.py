import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.stereo.calibration import calibrate_dataset, fit_calibration, load_result, save_result
from app.stereo.capture import main as capture_main, preview_canvas, rotation_choices_preview
from app.stereo.dataset import CapturedFrame, load_pair, save_pair
from app.stereo.orientation import (
    OrientationMismatchError, OrientationSettings, add_orientation_arguments,
    load_orientation, orientation_from_args, prepare_pair, result_orientation,
    rotate_image, save_orientation,
)
from app.stereo.rectify import load_rectification_images, prepare_calibration_inputs, rectify_pair
from .support import BOARD, SIZE, poses, projected_observations, render_checkerboard, right_pose, workspace, write_jpeg


class RotationTests(unittest.TestCase):
    def test_each_rotation_has_correct_direction_pixels_and_dimensions(self):
        raw = np.array([[1, 2, 3], [4, 5, 6]], np.uint8)
        original = raw.copy()
        expected = {
            "0": [[1, 2, 3], [4, 5, 6]],
            "cw90": [[4, 1], [5, 2], [6, 3]],
            "ccw90": [[3, 6], [2, 5], [1, 4]],
            "180": [[6, 5, 4], [3, 2, 1]],
        }
        for rotation, values in expected.items():
            with self.subTest(rotation=rotation):
                rotated = rotate_image(raw, rotation)
                np.testing.assert_array_equal(rotated, values)
                self.assertFalse(np.shares_memory(raw, rotated))
        np.testing.assert_array_equal(raw, original)
        with self.assertRaises(ValueError):
            rotate_image(raw, "45")

    def test_independent_rotation_allows_matching_transposed_raw_dimensions(self):
        left = np.arange(6, dtype=np.uint8).reshape(2, 3)
        right = np.array([[3, 0], [4, 1], [5, 2]], np.uint8)
        a, b = prepare_pair(left, right, OrientationSettings("0", "ccw90"), (3, 2))
        np.testing.assert_array_equal(a, b)
        with self.assertRaisesRegex(ValueError, "after rotation"):
            prepare_pair(left, left, OrientationSettings("0", "cw90"))
        with self.assertRaisesRegex(ValueError, "calibration"):
            prepare_pair(left, right, OrientationSettings("0", "ccw90"), (2, 3))

    def test_profile_roundtrip_cli_conflicts_and_cycle(self):
        parser = argparse.ArgumentParser()
        add_orientation_arguments(parser)
        settings = OrientationSettings("cw90", "ccw90")
        with workspace() as tmp:
            path = Path(tmp) / "orientation.json"
            save_orientation(path, settings)
            self.assertEqual(load_orientation(path), settings)
            with self.assertRaises(FileExistsError):
                save_orientation(path, settings)
            self.assertEqual(orientation_from_args(parser.parse_args(["--orientation", str(path)])), settings)
            with self.assertRaises(OrientationMismatchError):
                orientation_from_args(parser.parse_args(["--orientation", str(path), "--right-rotation", "0"]))
            self.assertIsNone(orientation_from_args(parser.parse_args([])))
        cycled = OrientationSettings()
        for _ in range(4):
            cycled = cycled.cycle(0)
        self.assertEqual(cycled, OrientationSettings())
        self.assertEqual(OrientationSettings().cycle(1), OrientationSettings("0", "cw90"))

    def test_capture_raw_bytes_and_processed_dimensions(self):
        with workspace() as tmp:
            root = Path(tmp)
            image = np.full((240, 320, 3), 180, np.uint8)
            write_jpeg(root / "left.jpg", image)
            write_jpeg(root / "right.jpg", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
            left = CapturedFrame.from_file(root / "left.jpg", 0)
            right = CapturedFrame.from_file(root / "right.jpg", 1)
            orientation = OrientationSettings("0", "ccw90")
            pair = save_pair(root / "data", left, right, orientation)
            metadata, paths, raw = load_pair(pair / "pair.json")
            self.assertEqual(paths[0].read_bytes(), left.jpeg)
            self.assertEqual(paths[1].read_bytes(), right.jpeg)
            self.assertEqual(metadata["orientation"], orientation.to_dict())
            self.assertEqual((metadata["right"]["width"], metadata["right"]["height"]), (240, 320))
            self.assertEqual((metadata["right"]["processed_width"], metadata["right"]["processed_height"]), (320, 240))
            self.assertEqual(raw[0].shape[:2], (240, 320))
            self.assertEqual(raw[1].shape[:2], (320, 240))
            self.assertEqual(preview_canvas({0: left, 1: right}, "test", orientation).shape, (570, 1280, 3))
            self.assertEqual(rotation_choices_preview(*raw).shape, (600, 1280, 3))
            metadata["right"]["processed_width"] = 999
            (pair / "pair.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Processed dimensions"):
                load_pair(pair / "pair.json")

    def test_interactive_preview_selects_cameras_independently_without_editing_jpeg(self):
        with workspace() as tmp:
            root = Path(tmp)
            image = np.full((240, 320, 3), 160, np.uint8)
            write_jpeg(root / "left.jpg", image)
            write_jpeg(root / "right.jpg", image)
            before = [(root / name).read_bytes() for name in ("left.jpg", "right.jpg")]
            profile = root / "chosen.json"
            # LEFT: cw90; RIGHT: ccw90. Both processed images remain 240x320.
            with patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.destroyAllWindows"), \
                    patch("cv2.getWindowProperty", return_value=1), \
                    patch("cv2.waitKey", side_effect=[ord("a"), ord("d"), ord("d"), ord("d"), ord("s")]):
                code = capture_main(["preview", "--left", str(root / "left.jpg"),
                                     "--right", str(root / "right.jpg"), "--output", str(profile)])
            self.assertEqual(code, 0)
            self.assertEqual(load_orientation(profile), OrientationSettings("cw90", "ccw90"))
            self.assertEqual(before, [(root / name).read_bytes() for name in ("left.jpg", "right.jpg")])

    def test_headless_comparison_exports_png_without_selecting_a_profile(self):
        with workspace() as tmp:
            root = Path(tmp)
            write_jpeg(root / "left.jpg", np.full((240, 320, 3), 140, np.uint8))
            write_jpeg(root / "right.jpg", np.full((240, 320, 3), 150, np.uint8))
            command = [sys.executable, "-B", "-m", "app.stereo.capture", "preview",
                       "--left", str(root / "left.jpg"), "--right", str(root / "right.jpg"),
                       "--comparison-output", str(root / "choices.png"), "--no-gui"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(cv2.imread(str(root / "choices.png")).shape, (600, 1280, 3))


class RotationCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orientation = OrientationSettings("cw90", "ccw90")
        cls.result = fit_calibration(projected_observations(), BOARD, orientation=cls.orientation)

    def test_rectification_loads_saved_settings_rotates_once_and_rejects_same_size_conflict(self):
        image = np.arange(SIZE[0] * SIZE[1] * 3, dtype=np.uint8).reshape(SIZE[1], SIZE[0], 3)
        raw_l, raw_r = rotate_image(image, "ccw90"), rotate_image(image, "cw90")
        with workspace() as tmp:
            path = Path(tmp) / "calibration.json"
            save_result(path, self.result)
            restored = load_result(path)
            self.assertEqual(result_orientation(restored), self.orientation)
            prepared = prepare_calibration_inputs(raw_l, raw_r, restored)
            for actual in prepared:
                np.testing.assert_array_equal(actual, image)
            rectified = rectify_pair(raw_l, raw_r, restored, self.orientation)
            self.assertEqual(rectified[0].shape, image.shape)
            # Opposite quarter turns have identical dimensions; check settings, not just shape.
            with self.assertRaises(OrientationMismatchError):
                rectify_pair(raw_l, raw_r, restored, OrientationSettings("ccw90", "cw90"))
            with self.assertRaisesRegex(ValueError, "resolution"):
                rectify_pair(image, image, restored)
            missing = copy.deepcopy(restored)
            del missing["orientation"]
            path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError, "missing orientation"):
                load_result(path)

    def test_legacy_result_and_pair_are_readable_with_zero_rotation(self):
        old = copy.deepcopy(self.result)
        old["schema_version"] = 1
        old.pop("orientation")
        old.pop("input_space")
        with workspace() as tmp:
            root = Path(tmp)
            path = root / "old.json"
            save_result(path, old)
            self.assertEqual(result_orientation(load_result(path)), OrientationSettings())
            write_jpeg(root / "input.jpg", np.full((240, 320, 3), 120, np.uint8))
            pair = save_pair(root / "dataset", CapturedFrame.from_file(root / "input.jpg", 0),
                             CapturedFrame.from_file(root / "input.jpg", 1))
            metadata = json.loads((pair / "pair.json").read_text())
            metadata["schema_version"] = 1
            metadata.pop("orientation")
            for side in ("left", "right"):
                metadata[side].pop("processed_width")
                metadata[side].pop("processed_height")
            (pair / "pair.json").write_text(json.dumps(metadata))
            self.assertEqual(load_pair(pair / "pair.json")[2][0].shape, (240, 320, 3))

    def test_rotated_jpeg_dataset_calibrates_before_detection_and_conflicts_fail(self):
        with workspace() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            pairs = []
            for rotation, translation in poses():
                rr, tr = right_pose(rotation, translation)
                # Sensor JPEG orientations differ. The configured rotations restore the same upright space.
                write_jpeg(root / "left.jpg", rotate_image(render_checkerboard(rotation, translation), "ccw90"))
                write_jpeg(root / "right.jpg", rotate_image(render_checkerboard(rr, tr), "cw90"))
                pairs.append(save_pair(dataset, CapturedFrame.from_file(root / "left.jpg", 0),
                                       CapturedFrame.from_file(root / "right.jpg", 1), self.orientation))
            result = calibrate_dataset(dataset, BOARD)  # Infer the recorded settings.
            self.assertEqual(result["orientation"], self.orientation.to_dict())
            self.assertEqual(result["image_size"], list(SIZE))
            self.assertEqual(result["report"]["used_pairs"], 12)
            self.assertAlmostEqual(result["report"]["baseline_cm"], 9, delta=0.3)
            self.assertLess(result["report"]["vertical_rms_px"], 0.5)
            profile = root / "orientation.json"
            save_orientation(profile, self.orientation)
            parameters = root / "parameters.json"
            save_result(parameters, result)
            conflicting_result = copy.deepcopy(result)
            conflicting_result["orientation"] = OrientationSettings("ccw90", "cw90").to_dict()
            with self.assertRaises(OrientationMismatchError):
                load_rectification_images(pairs[0] / "left.jpg", pairs[0] / "right.jpg", conflicting_result)
            with self.assertRaisesRegex(ValueError, "different stored pair"):
                load_rectification_images(pairs[0] / "left.jpg", pairs[1] / "right.jpg", result)
            command = [sys.executable, "-B", "-m", "app.stereo.rectify", "--parameters", str(parameters),
                       "--left", str(pairs[0] / "left.jpg"), "--right", str(pairs[0] / "right.jpg"),
                       "--orientation", str(profile), "--output", str(root / "rectified")]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(cv2.imread(str(root / "rectified" / "rectified.png")).shape, (720, 1920, 3))
            rejected = subprocess.run(command[:-1] + [str(root / "rejected"), "--left-rotation", "ccw90"],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Rotation settings mismatch", rejected.stderr)
            self.assertFalse((root / "rejected").exists())
            with self.assertRaises(OrientationMismatchError):
                calibrate_dataset(dataset, BOARD, orientation=OrientationSettings())
            legacy = root / "legacy"
            legacy.mkdir()
            for pair in pairs:
                copy_dir = legacy / pair.name
                copy_dir.mkdir()
                metadata = json.loads((pair / "pair.json").read_text())
                metadata["schema_version"] = 1
                metadata.pop("orientation")
                for side in ("left", "right"):
                    metadata[side].pop("processed_width")
                    metadata[side].pop("processed_height")
                    (copy_dir / f"{side}.jpg").write_bytes((pair / f"{side}.jpg").read_bytes())
                (copy_dir / "pair.json").write_text(json.dumps(metadata))
            legacy_result = calibrate_dataset(legacy, BOARD, orientation=self.orientation)
            self.assertEqual(legacy_result["orientation"], self.orientation.to_dict())
            self.assertAlmostEqual(legacy_result["report"]["baseline_cm"], 9, delta=0.3)
            save_pair(dataset, CapturedFrame.from_file(root / "left.jpg", 0),
                      CapturedFrame.from_file(root / "right.jpg", 1), OrientationSettings("ccw90", "cw90"))
            with self.assertRaises(OrientationMismatchError):
                calibrate_dataset(dataset, BOARD)
            print(f"\nRotated JPEG integration: baseline={result['report']['baseline_cm']:.4f}cm, "
                  f"vertical RMS={result['report']['vertical_rms_px']:.4f}px")


if __name__ == "__main__":
    unittest.main()

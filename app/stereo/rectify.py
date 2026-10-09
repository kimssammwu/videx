"""Load calibration JSON and visualize remapped stereo images with epipolar lines."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from .calibration import load_result, validate_result
from .dataset import decode_jpeg, load_pair
from .orientation import (
    add_orientation_arguments, orientation_from_args, pair_orientation, prepare_pair,
    require_matching_orientation, result_orientation,
)


def prepare_calibration_inputs(left, right, result, orientation=None):
    """Future stereo consumers can reuse this RAW -> rotated-pixel contract."""
    validate_result(result)
    expected = result_orientation(result)
    if orientation is not None:
        require_matching_orientation(orientation, expected)
    return prepare_pair(left, right, expected, expected_size=result["image_size"])


def rectify_pair(left, right, result, orientation=None):
    """Accept decoded RAW images only; apply saved rotation before rectification."""
    left, right = prepare_calibration_inputs(left, right, result, orientation)
    width, height = result["image_size"]
    matrices = {key: np.asarray(value, np.float64) for key, value in result["matrices"].items()}
    corrected = []
    for image, side, rotation, projection in ((left, "left", "R1", "P1"),
                                               (right, "right", "R2", "P2")):
        map_x, map_y = cv2.initUndistortRectifyMap(
            matrices["K_" + side], matrices["D_" + side], matrices[rotation],
            matrices[projection], (width, height), cv2.CV_32FC1)
        corrected.append(cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT))
    return tuple(corrected)


def epipolar_preview(left, right, spacing=40):
    canvas = np.hstack((left, right)).copy()
    for y in range(spacing, canvas.shape[0], spacing):
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (0, 255, 0), 1)
    return canvas


def load_rectification_images(left_path, right_path, result):
    """Check a companion pair.json when present, including same-size conflicts."""
    paths = [Path(left_path).resolve(), Path(right_path).resolve()]
    manifests = {path.parent / "pair.json" for path in paths if (path.parent / "pair.json").exists()}
    if len(manifests) > 1:
        raise ValueError("LEFT/RIGHT refer to different stored pair directories")
    if manifests:
        try:
            metadata, stored_paths, images = load_pair(manifests.pop())
        except (KeyError, TypeError) as error:
            raise ValueError(f"Malformed pair metadata: {error}") from error
        if paths != stored_paths:
            raise ValueError("JPEG paths disagree with LEFT/RIGHT paths in pair.json")
        declared = pair_orientation(metadata)
        if declared is not None:
            require_matching_orientation(declared, result_orientation(result))
        return images
    return [decode_jpeg(path.read_bytes()) for path in paths]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--show", action="store_true", help="Also open a GUI preview")
    add_orientation_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = load_result(args.parameters)
        left, right = load_rectification_images(args.left, args.right, result)
        orientation = orientation_from_args(args, default=result_orientation(result))
        corrected = rectify_pair(left, right, result, orientation)
        prepared = prepare_calibration_inputs(left, right, result, orientation)
        before, after = epipolar_preview(*prepared), epipolar_preview(*corrected)
        args.output.mkdir(parents=True, exist_ok=False)
        for name, image in (("left_rectified.png", corrected[0]), ("right_rectified.png", corrected[1]),
                            ("before.png", before), ("rectified.png", after)):
            if not cv2.imwrite(str(args.output / name), image):
                raise OSError(f"Could not write {name}")
        print(f"Rectification previews written: {args.output}")
        print("Quality: physical measurement UNVERIFIED; inspect matching corners along green lines.")
        for warning in result["report"]["warnings"]:
            print("CALIBRATION WARNING:", warning)
        if args.show:
            cv2.imshow("Before rectification", before)
            cv2.imshow("After rectification | physical quality UNVERIFIED", after)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0
    except (ValueError, OSError, cv2.error) as error:
        print(f"Rectification not completed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

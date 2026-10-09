"""Shared raw-pixel rotation contract for capture, calibration and rectification."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import cv2
import numpy as np


ROTATIONS = ("0", "cw90", "180", "ccw90")
ROTATE_CODES = {"cw90": cv2.ROTATE_90_CLOCKWISE, "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
                "180": cv2.ROTATE_180}


class OrientationMismatchError(ValueError):
    """A declared preprocessing setting differs from the calibration contract."""


@dataclass(frozen=True)
class OrientationSettings:
    left_rotation: str = "0"
    right_rotation: str = "0"

    def __post_init__(self):
        if self.left_rotation not in ROTATIONS or self.right_rotation not in ROTATIONS:
            raise ValueError("Rotation must be 0, cw90, ccw90 or 180")

    def to_dict(self):
        return {"schema_version": 1, "left_rotation": self.left_rotation,
                "right_rotation": self.right_rotation}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("Invalid orientation schema_version")
        if "left_rotation" not in value or "right_rotation" not in value:
            raise ValueError("Both camera rotations must be declared")
        return cls(value["left_rotation"], value["right_rotation"])

    def cycle(self, camera_id):
        if camera_id not in (0, 1):
            raise ValueError("camera_id must be 0 or 1")
        field = "left_rotation" if camera_id == 0 else "right_rotation"
        return replace(self, **{field: ROTATIONS[(ROTATIONS.index(getattr(self, field)) + 1) % 4]})


def rotate_image(image, rotation):
    """Rotate decoded raw pixels once, without JPEG encoding or interpolation."""
    if rotation not in ROTATIONS:
        raise ValueError("Rotation must be 0, cw90, ccw90 or 180")
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or min(image.shape[:2]) <= 0:
        raise ValueError("Expected a nonempty image array")
    return image.copy() if rotation == "0" else cv2.rotate(image, ROTATE_CODES[rotation])


def prepare_pair(left, right, orientation=None, expected_size=None):
    """Accept RAW arrays; rotate once, validate, then return processed arrays."""
    orientation = OrientationSettings() if orientation is None else orientation
    left = rotate_image(left, orientation.left_rotation)
    right = rotate_image(right, orientation.right_rotation)
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"LEFT/RIGHT resolutions differ after rotation: "
                         f"LEFT={left.shape[1]}x{left.shape[0]}, RIGHT={right.shape[1]}x{right.shape[0]}")
    if expected_size is not None and (left.shape[1], left.shape[0]) != tuple(expected_size):
        raise ValueError("Input resolution after rotation differs from calibration; do not silently resize")
    return left, right


def require_matching_orientation(actual, expected):
    if actual != expected:
        raise OrientationMismatchError(f"Rotation settings mismatch: requested={actual.to_dict()}, "
                                       f"expected={expected.to_dict()}; recalibrate with the new settings")
    return expected


def load_orientation(path):
    return OrientationSettings.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_orientation(path, orientation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(orientation.to_dict(), indent=2) + "\n"
    output = path.open("x", encoding="utf-8")
    try:
        with output:
            output.write(content)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def add_orientation_arguments(parser):
    parser.add_argument("--orientation", type=Path, help="Shared orientation JSON profile")
    parser.add_argument("--left-rotation", choices=ROTATIONS, default=None)
    parser.add_argument("--right-rotation", choices=ROTATIONS, default=None)


def orientation_from_args(args, default=None):
    """None means unspecified; explicit flags must agree with a supplied profile."""
    profile = load_orientation(args.orientation) if args.orientation is not None else None
    base = profile or default or OrientationSettings()
    settings = OrientationSettings(
        base.left_rotation if args.left_rotation is None else args.left_rotation,
        base.right_rotation if args.right_rotation is None else args.right_rotation)
    if profile is not None:
        require_matching_orientation(settings, profile)
    if profile is None and args.left_rotation is None and args.right_rotation is None and default is None:
        return None
    return settings


def result_orientation(result):
    if result.get("schema_version") == 1:
        settings = OrientationSettings.from_dict(result["orientation"]) if "orientation" in result else OrientationSettings()
        require_matching_orientation(settings, OrientationSettings())
        return settings
    if "orientation" not in result:
        raise ValueError("Calibration result is missing orientation settings")
    return OrientationSettings.from_dict(result["orientation"])


def pair_orientation(metadata):
    if metadata.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported pair schema_version")
    if metadata.get("schema_version") == 1:
        return None  # Old capture format did not record an explicit choice.
    if "orientation" not in metadata:
        raise ValueError("Pair metadata is missing orientation settings")
    return OrientationSettings.from_dict(metadata["orientation"])

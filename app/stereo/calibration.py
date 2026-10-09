"""Offline pinhole stereo calibration, numeric quality checks and JSON parameters."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .dataset import load_pair
from .orientation import (
    OrientationMismatchError, OrientationSettings, add_orientation_arguments, orientation_from_args,
    pair_orientation, prepare_pair, require_matching_orientation, result_orientation,
)


UNIT_METERS = {"mm": 0.001, "cm": 0.01, "m": 1.0}
MATRIX_SHAPES = {
    "K_left": (3, 3), "K_right": (3, 3), "R": (3, 3), "T": (3, 1),
    "E": (3, 3), "F": (3, 3), "R1": (3, 3), "R2": (3, 3),
    "P1": (3, 4), "P2": (3, 4), "Q": (4, 4),
}


@dataclass(frozen=True)
class BoardSettings:
    columns: int
    rows: int
    square_size: float
    unit: str = "mm"

    def __post_init__(self):
        if type(self.columns) is not int or type(self.rows) is not int or self.columns < 3 or self.rows < 3:
            raise ValueError("Use at least 3 inner corners in each direction")
        if not np.isfinite(self.square_size) or self.square_size <= 0:
            raise ValueError("square_size must be finite and positive")
        if self.unit not in UNIT_METERS:
            raise ValueError("unit must be mm, cm or m")

    def object_points(self):
        points = np.zeros((self.columns * self.rows, 3), np.float32)
        points[:, :2] = np.mgrid[0:self.columns, 0:self.rows].T.reshape(-1, 2)
        points *= self.square_size
        return points


@dataclass(frozen=True)
class QualitySettings:
    min_pairs: int = 10
    max_reprojection_px: float = 1.0
    max_stereo_rms_px: float = 1.5
    max_vertical_rms_px: float = 1.0
    expected_baseline_cm: float = 9.0
    baseline_relative_tolerance: float = 0.20

    def __post_init__(self):
        if type(self.min_pairs) is not int or self.min_pairs < 3:
            raise ValueError("min_pairs must be at least 3")
        for key, value in asdict(self).items():
            if key != "min_pairs" and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")


@dataclass
class Observation:
    pair_id: str
    left: np.ndarray
    right: np.ndarray
    image_size: tuple


def detect_corners(image, board, corner_policy="legacy"):
    if corner_policy not in ("legacy", "consistent"):
        raise ValueError("corner_policy must be legacy or consistent")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (board.columns, board.rows)
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            window = 11
            if corner_policy == "consistent":
                grid = corners.reshape(board.rows, board.columns, 2)
                spacing = np.r_[np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                                np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()]
                window = max(1, min(5, int(np.percentile(spacing, 10) * 0.35)))
            corners = cv2.cornerSubPix(gray, corners, (window, window), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
    if not found or corners is None or len(corners) != board.columns * board.rows:
        return None
    return corners.astype(np.float32).reshape(-1, 1, 2)


def corner_consistency(left, right, board, align=False):
    """Audit lattice geometry; optional 180-degree alignment for a parallel rig.

    This uses the declared, upright image space, not the known baseline. It cannot
    establish physical board identity or resolve all unmarked-board ambiguities.
    """
    def directions(points):
        grid = np.asarray(points).reshape(board.rows, board.columns, 2)
        return [np.median(np.diff(grid, axis=axis).reshape(-1, 2), axis=0) for axis in (1, 0)]

    ld, rd = directions(left), directions(right)
    cosine = [float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
              for a, b in zip(ld, rd)]
    reverse = align and all(c < -0.5 for c in cosine)
    if reverse:
        right = np.asarray(right)[::-1].copy()
    homography = []
    for points in (left, right):
        h, _ = cv2.findHomography(board.object_points()[:, :2], points, 0)
        if h is None:
            homography.append(None)
            continue
        projected = cv2.perspectiveTransform(board.object_points()[:, :2].reshape(-1, 1, 2), h)
        delta = projected.reshape(-1, 2) - np.asarray(points).reshape(-1, 2)
        homography.append(float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))))
    info = {"right_order_reversed": bool(reverse), "direction_cosines_before": cosine,
            "direction_cosines_after": [-c for c in cosine] if reverse else cosine,
            "homography_rms_px": homography,
            "consistent_parallel_directions": all(c > 0.5 for c in ([-c for c in cosine] if reverse else cosine))}
    return right, info


def rectification_coverage(matrices, size):
    masks = []
    for side, r, p in (("left", "R1", "P1"), ("right", "R2", "P2")):
        mx, my = cv2.initUndistortRectifyMap(matrices[f"K_{side}"], matrices[f"D_{side}"],
                                            matrices[r], matrices[p], size, cv2.CV_32FC1)
        masks.append(np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (mx < size[0] - 1)
                     & (my >= 0) & (my < size[1] - 1))
    return {"left_fraction": float(np.mean(masks[0])), "right_fraction": float(np.mean(masks[1])),
            "common_fraction": float(np.mean(masks[0] & masks[1])),
            "definition": "rectified output pixels with in-bounds bilinear source support; alpha=0",
            "rectified_focal_px": float(matrices["P1"][0, 0])}


def reprojection_errors(objects, corners, rotations, translations, camera, distortion):
    errors = []
    for obj, actual, rotation, translation in zip(objects, corners, rotations, translations):
        projected, _ = cv2.projectPoints(obj, rotation, translation, camera, distortion)
        difference = actual.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1)))))
    return errors


def fit_calibration(observations, board, quality=None, orientation=None, *, baseline_mm=None):
    """Fit already-prepared corner observations in the declared rotated image space."""
    orientation = OrientationSettings() if orientation is None else orientation
    quality = QualitySettings() if quality is None else quality
    if baseline_mm is not None and (not np.isfinite(baseline_mm) or baseline_mm <= 0):
        raise ValueError("baseline_mm must be finite and positive")
    if len(observations) < 3:
        raise ValueError(f"At least 3 valid, distinct poses are required; got {len(observations)}")
    size = tuple(observations[0].image_size)
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("Invalid image size")
    count = board.columns * board.rows
    for observation in observations:
        if tuple(observation.image_size) != size:
            raise ValueError("All calibration pairs must use the same resolution")
        for points in (observation.left, observation.right):
            if np.asarray(points).shape not in ((count, 1, 2), (count, 2)) or not np.all(np.isfinite(points)):
                raise ValueError("Invalid corner correspondences")
    objects = [board.object_points() for _ in observations]
    left = [np.asarray(o.left, np.float32).reshape(-1, 1, 2) for o in observations]
    right = [np.asarray(o.right, np.float32).reshape(-1, 1, 2) for o in observations]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    rms_l, kl, dl, rl, tl = cv2.calibrateCamera(objects, left, size, None, None, criteria=criteria)
    rms_r, kr, dr, rr, tr = cv2.calibrateCamera(objects, right, size, None, None, criteria=criteria)
    stereo_rms, kl, dl, kr, dr, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        objects, left, right, kl, dl, kr, dr, size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    optimizer, stereo_errors = None, None
    if baseline_mm is not None:
        from .baseline import fit_fixed_baseline
        length = baseline_mm * UNIT_METERS["mm"] / UNIT_METERS[board.unit]
        rotation, translation, stereo_rms, stereo_errors, optimizer = fit_fixed_baseline(
            objects[0], left, right, kl, dl, kr, dr, rotation, translation, rl, tl, rr, tr, length)
        tx, ty, tz = translation.ravel()
        skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
        essential = skew @ rotation
        fundamental = np.linalg.inv(kr).T @ essential @ np.linalg.inv(kl)
    r1, r2, p1, p2, q, roi_l, roi_r = cv2.stereoRectify(
        kl, dl, kr, dr, size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    errors_l = reprojection_errors(objects, left, rl, tl, kl, dl)
    errors_r = reprojection_errors(objects, right, rr, tr, kr, dr)
    vertical, per_pair = [], []
    for index, observation in enumerate(observations):
        l_rect = cv2.undistortPoints(left[index], kl, dl, R=r1, P=p1).reshape(-1, 2)
        r_rect = cv2.undistortPoints(right[index], kr, dr, R=r2, P=p2).reshape(-1, 2)
        delta = np.abs(l_rect[:, 1] - r_rect[:, 1])
        vertical.extend(delta.tolist())
        per_pair.append({"pair_id": observation.pair_id,
                         "left_reprojection_rms_px": errors_l[index],
                         "right_reprojection_rms_px": errors_r[index],
                         "vertical_rms_px": float(np.sqrt(np.mean(delta ** 2))),
                         "vertical_max_px": float(np.max(delta))})
        if stereo_errors is not None:
            per_pair[-1].update(stereo_left_reprojection_rms_px=float(stereo_errors[index, 0]),
                                stereo_right_reprojection_rms_px=float(stereo_errors[index, 1]))
    vertical = np.asarray(vertical)
    baseline = float(np.linalg.norm(translation))
    baseline_cm = baseline * UNIT_METERS[board.unit] * 100
    relative_difference = abs(baseline_cm - quality.expected_baseline_cm) / quality.expected_baseline_cm
    matrices = {"K_left": kl, "D_left": dl, "K_right": kr, "D_right": dr,
                "R": rotation, "T": translation, "E": essential, "F": fundamental,
                "R1": r1, "R2": r2, "P1": p1, "P2": p2, "Q": q}
    if not all(np.all(np.isfinite(value)) for value in matrices.values()):
        raise ValueError("Calibration produced non-finite parameters")
    metrics = {
        "left_reprojection_rms_px": float(rms_l), "right_reprojection_rms_px": float(rms_r),
        "stereo_rms_px": float(stereo_rms), "baseline": baseline, "baseline_unit": board.unit,
        "baseline_cm": baseline_cm, "expected_baseline_cm": quality.expected_baseline_cm,
        "baseline_relative_difference": relative_difference,
        "vertical_mean_abs_px": float(np.mean(vertical)),
        "vertical_rms_px": float(np.sqrt(np.mean(vertical ** 2))),
        "vertical_p95_px": float(np.percentile(vertical, 95)),
        "vertical_max_px": float(np.max(vertical)),
    }
    if not all(np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
        raise ValueError("Calibration produced non-finite quality metrics")
    warnings = []
    if len(observations) < quality.min_pairs:
        warnings.append(f"Only {len(observations)} valid pairs; recommend at least {quality.min_pairs}")
    for side in ("left", "right"):
        if metrics[f"{side}_reprojection_rms_px"] > quality.max_reprojection_px:
            warnings.append(f"{side.upper()} reprojection RMS exceeds {quality.max_reprojection_px}px")
    if max(errors_l + errors_r) > 2 * quality.max_reprojection_px:
        warnings.append("Individual pair reprojection error is high; inspect per_pair metrics")
    if stereo_rms > quality.max_stereo_rms_px:
        warnings.append(f"Stereo RMS exceeds {quality.max_stereo_rms_px}px")
    if metrics["vertical_rms_px"] > quality.max_vertical_rms_px:
        warnings.append(f"Rectified vertical RMS exceeds {quality.max_vertical_rms_px}px")
    if metrics["vertical_p95_px"] > 2 * quality.max_vertical_rms_px:
        warnings.append("Rectified vertical p95 is high; inspect individual pairs")
    if relative_difference > quality.baseline_relative_tolerance:
        warnings.append("Estimated baseline differs substantially from the approximate hardware baseline")
    if abs(float(translation[0, 0])) < abs(float(translation[1, 0])):
        warnings.append("Vertical stereo geometry detected; verify camera orientation before horizontal SGBM")
    coverage = rectification_coverage(matrices, size)
    if min(roi_l[2:]) <= 0 or min(roi_r[2:]) <= 0 or coverage["common_fraction"] < 0.5:
        warnings.append("Rectification has empty ROI or limited common valid image coverage")
    if optimizer is not None:
        if not optimizer["converged"]:
            warnings.append("Fixed baseline optimizer did not converge")
        if optimizer["minimum_corner_depth"] <= 0:
            warnings.append("Fitted board corners are behind a camera")
    # Good in-sample errors alone cannot establish calibration observability.
    normals = [cv2.Rodrigues(r)[0][:, 2] for r in rl]
    angles = [np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))
              for i, a in enumerate(normals) for b in normals[i + 1:]]
    tilt_span = float(max(angles, default=0))
    if tilt_span < 10:
        warnings.append("Board tilt diversity is low (<10 degrees); collect more varied poses")
    if board.columns == board.rows:
        warnings.append("Square inner-corner grid has additional ordering ambiguity; prefer a rectangular board")
    if kl[0, 0] <= 0 or kl[1, 1] <= 0 or kr[0, 0] <= 0 or kr[1, 1] <= 0 or baseline <= 0:
        raise ValueError("Invalid focal length or baseline")
    result = {
        "schema_version": 2, "opencv_version": cv2.__version__,
        "orientation": orientation.to_dict(), "input_space": "rotated_raw_pixels",
        "image_size": list(size), "board": asdict(board), "length_unit": board.unit,
        "quality_settings": asdict(quality),
        "matrices": {key: value.tolist() for key, value in matrices.items()},
        "roi_left": list(roi_l), "roi_right": list(roi_r),
        "calibration_method": "free_baseline" if baseline_mm is None else "fixed_baseline",
        "baseline_constraint_mm": baseline_mm,
        "report": {**metrics, "used_pairs": len(observations), "board_tilt_span_degrees": tilt_span,
                   "per_pair": per_pair, "warnings": warnings, "valid_image_coverage": coverage,
                   "numeric_checks_passed": not warnings,
                   "validation_status": "physical_measurement_unverified",
                   "quality_evaluation": "in_sample; not an independent validation set",
                   "synchronization_verified": False, "physical_camera_mapping_verified": False},
    }
    if optimizer is not None:
        result["report"]["optimizer"] = optimizer
        result["report"].update(stereo_left_reprojection_rms_px=float(np.sqrt(np.mean(stereo_errors[:, 0]**2))),
                                stereo_right_reprojection_rms_px=float(np.sqrt(np.mean(stereo_errors[:, 1]**2))))
    return result


def dataset_orientation(pair_dirs, requested=None):
    declared = None
    for directory in pair_dirs:
        try:
            metadata = json.loads((directory / "pair.json").read_text(encoding="utf-8"))
            settings = pair_orientation(metadata) if isinstance(metadata, dict) else None
        except (OSError, ValueError, KeyError, TypeError):
            continue  # Invalid manifests are counted as exclusions by load_pair below.
        if settings is not None:
            if declared is not None:
                require_matching_orientation(settings, declared)
            declared = settings
    if requested is not None and declared is not None:
        require_matching_orientation(requested, declared)
    return requested or declared or OrientationSettings()


def collect_observations(dataset, board, orientation=None, *, corner_policy="legacy"):
    if corner_policy not in ("legacy", "consistent"):
        raise ValueError("corner_policy must be legacy or consistent")
    dataset = Path(dataset).resolve()
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    pair_dirs = sorted(path for path in dataset.glob("pair_*") if path.is_dir())
    if not pair_dirs:
        raise ValueError("No pair_* directories; capture/import JPEG pairs first")
    orientation = dataset_orientation(pair_dirs, orientation)
    observations, excluded, seen, consistency = [], [], set(), []
    image_size = None
    detection_failed = 0
    for directory in pair_dirs:
        try:
            metadata, paths, images = load_pair(directory / "pair.json")
            declared = pair_orientation(metadata)
            if declared is not None:
                require_matching_orientation(declared, orientation)
            images = prepare_pair(*images, orientation)
            size = (images[0].shape[1], images[0].shape[0])
            digest = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
            if digest in seen:
                raise ValueError("Duplicate JPEG pair")
            if image_size is not None and size != image_size:
                raise ValueError("Resolution differs from accepted calibration pairs")
            left, right = (detect_corners(image, board, corner_policy) for image in images)
            if left is None or right is None:
                detection_failed += 1
                missing = [side for side, corners in (("LEFT", left), ("RIGHT", right)) if corners is None]
                raise ValueError("Checkerboard not detected: " + ", ".join(missing))
            image_size = size
            seen.add(digest)
            right, audit = corner_consistency(left, right, board, align=corner_policy == "consistent")
            consistency.append({"pair_id": metadata["pair_id"], **audit})
            observations.append(Observation(metadata["pair_id"], left, right, size))
        except OrientationMismatchError:
            raise  # Never hide rotation conflicts by quietly excluding images.
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
            excluded.append({"pair_directory": str(directory), "reason": str(error)})
    if len(observations) < 3:
        raise ValueError(f"Insufficient valid pairs: total={len(pair_dirs)}, used={len(observations)}, "
                         f"excluded={len(excluded)}, detection_failed={detection_failed}. "
                         f"Reasons: {json.dumps(excluded, ensure_ascii=False)}")
    metadata = dict(total_pairs=len(pair_dirs), excluded_pairs=len(excluded),
                    detection_failed_pairs=detection_failed, excluded=excluded,
                    corner_policy=corner_policy, corner_consistency=consistency,
                    corner_ordering_assumption="parallel rig, same board face in upright images; physical identity unverified")
    return observations, orientation, metadata


def calibrate_dataset(dataset, board, quality=None, orientation=None, *, baseline_mm=None, corner_policy="legacy"):
    observations, orientation, metadata = collect_observations(dataset, board, orientation, corner_policy=corner_policy)
    result = fit_calibration(observations, board, quality, orientation, baseline_mm=baseline_mm)
    result["dataset"] = str(Path(dataset).resolve())
    result["report"].update(metadata)
    if any(not a["consistent_parallel_directions"] for a in metadata["corner_consistency"]):
        result["report"]["warnings"].append("Corner index directions disagree; inspect checkerboard correspondence")
        result["report"]["numeric_checks_passed"] = False
    if any(any(e is None or e > 1 for e in a["homography_rms_px"]) for a in metadata["corner_consistency"]):
        result["report"]["warnings"].append("Corner lattice homography error is high; inspect detection (distortion can also contribute)")
        result["report"]["numeric_checks_passed"] = False
    return result


def validate_result(result):
    if not isinstance(result, dict):
        raise ValueError("Calibration parameters must be an object")
    if result.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported calibration schema_version")
    result_orientation(result)
    if result["schema_version"] == 2 and result.get("input_space") != "rotated_raw_pixels":
        raise ValueError("Calibration result has an invalid input coordinate space")
    size = result["image_size"]
    if len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size):
        raise ValueError("Invalid calibration image_size")
    if result["length_unit"] not in UNIT_METERS:
        raise ValueError("Invalid length unit")
    if not isinstance(result.get("report"), dict) or not isinstance(result["report"].get("warnings"), list):
        raise ValueError("Missing calibration quality report")
    for key, shape in MATRIX_SHAPES.items():
        value = np.asarray(result["matrices"][key], dtype=np.float64)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid calibration matrix: {key}")
    for key in ("D_left", "D_right"):
        value = np.asarray(result["matrices"][key], dtype=np.float64)
        vector_shape = value.ndim == 1 or (value.ndim == 2 and 1 in value.shape)
        if not vector_shape or value.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid distortion coefficients: {key}")
    for key in ("K_left", "K_right"):
        camera = np.asarray(result["matrices"][key])
        if camera[0, 0] <= 0 or camera[1, 1] <= 0:
            raise ValueError("Invalid focal length")
    if np.linalg.norm(result["matrices"]["T"]) <= 0:
        raise ValueError("Invalid zero baseline")
    fixed = result.get("baseline_constraint_mm")
    if fixed is not None:
        if not isinstance(fixed, (int, float)) or not np.isfinite(fixed) or fixed <= 0:
            raise ValueError("Invalid baseline constraint")
        length_mm = np.linalg.norm(result["matrices"]["T"]) * UNIT_METERS[result["length_unit"]] / UNIT_METERS["mm"]
        if not np.isclose(length_mm, fixed, rtol=1e-9, atol=1e-7):
            raise ValueError("Translation length disagrees with declared baseline constraint")
    return result


def save_result(path, result):
    validate_result(result)
    path = Path(path)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing calibration parameters from overwrite.
    output = path.open("x", encoding="utf-8")
    try:
        with output:
            output.write(content)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def load_result(path):
    try:
        return validate_result(json.loads(Path(path).read_text(encoding="utf-8")))
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed calibration parameters: {error}") from error


def print_report(result):
    report = result["report"]
    orientation = result_orientation(result)
    print(f"Raw pixel rotations: LEFT={orientation.left_rotation}, RIGHT={orientation.right_rotation}; "
          f"processed image_size={result['image_size']}")
    print(f"Pairs: used={report['used_pairs']}, excluded={report.get('excluded_pairs', 0)}, "
          f"detection_failed={report.get('detection_failed_pairs', 0)}, total={report.get('total_pairs', report['used_pairs'])}")
    print(f"LEFT reprojection RMS: {report['left_reprojection_rms_px']:.4f}px")
    print(f"RIGHT reprojection RMS: {report['right_reprojection_rms_px']:.4f}px")
    print(f"Stereo RMS: {report['stereo_rms_px']:.4f}px")
    print(f"Calibration method: {result.get('calibration_method', 'free_baseline')}; "
          f"fixed length (mm): {result.get('baseline_constraint_mm')}; corner policy: {report.get('corner_policy', 'legacy')}")
    print(f"Baseline: {report['baseline']:.4f}{report['baseline_unit']} = {report['baseline_cm']:.4f}cm; "
          f"reference ~{report['expected_baseline_cm']:.2f}cm; difference={report['baseline_relative_difference']:.1%}")
    print(f"Rectified vertical error: mean={report['vertical_mean_abs_px']:.4f}px, "
          f"RMS={report['vertical_rms_px']:.4f}px, p95={report['vertical_p95_px']:.4f}px, max={report['vertical_max_px']:.4f}px")
    print("Stereo R:", np.asarray(result["matrices"]["R"]))
    print(f"Stereo T ({result['length_unit']}):", np.asarray(result["matrices"]["T"]).ravel())
    if "valid_image_coverage" in report:
        print(f"Valid ROI: LEFT={result['roi_left']}, RIGHT={result['roi_right']}; "
              f"common remap coverage={report['valid_image_coverage']['common_fraction']:.1%}")
    for warning in report["warnings"]:
        print("WARNING:", warning)
    for excluded in report.get("excluded", []):
        print("EXCLUDED:", excluded["pair_directory"], "-", excluded["reason"])
    print("Numeric checks:", "PASS (in-sample only)" if report["numeric_checks_passed"] else "REVIEW REQUIRED")
    print("Physical measurement quality, actual L/R mapping and exposure synchronization: UNVERIFIED")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--columns", type=int, required=True, help="Inner corner columns, not squares")
    parser.add_argument("--rows", type=int, required=True, help="Inner corner rows, not squares")
    parser.add_argument("--square-size", type=float, required=True, help="Measured square edge length")
    parser.add_argument("--unit", choices=UNIT_METERS, default="mm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--max-reprojection-px", type=float, default=1.0)
    parser.add_argument("--max-stereo-rms-px", type=float, default=1.5)
    parser.add_argument("--max-vertical-rms-px", type=float, default=1.0)
    parser.add_argument("--expected-baseline-cm", type=float, default=9.0)
    parser.add_argument("--baseline-relative-tolerance", type=float, default=0.20)
    parser.add_argument("--fixed-baseline-mm", type=float, default=None,
                        help="Optimize R and T direction with exactly this measured length; default is free baseline")
    parser.add_argument("--corner-policy", choices=("legacy", "consistent"), default="legacy",
                        help="Opt-in spacing-aware fallback and parallel-rig 180-degree index alignment")
    add_orientation_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new calibration filename")
        board = BoardSettings(args.columns, args.rows, args.square_size, args.unit)
        quality = QualitySettings(args.min_pairs, args.max_reprojection_px, args.max_stereo_rms_px,
                                  args.max_vertical_rms_px, args.expected_baseline_cm,
                                  args.baseline_relative_tolerance)
        result = calibrate_dataset(args.dataset, board, quality, orientation_from_args(args),
                                   baseline_mm=args.fixed_baseline_mm, corner_policy=args.corner_policy)
        save_result(args.output, result)
        print_report(result)
        print(f"Parameters written: {args.output}")
        return 0 if result["report"]["numeric_checks_passed"] else 2
    except (ValueError, OSError, cv2.error) as error:
        print(f"Calibration not completed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

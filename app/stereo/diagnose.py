"""Read-only source audit: numbered corners, detector comparison and filtered refit.

No corner permutation, image mirroring, baseline constraint or threshold relaxation.
The output must be a NEW directory. Source JPEGs/metadata/results are hashed and preserved.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .calibration import BoardSettings, QualitySettings, calibrate_dataset, load_result, save_result
from .dataset import load_pair
from .orientation import pair_orientation, prepare_pair, require_matching_orientation, result_orientation


CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)


def grid_metrics(corners, board):
    if corners is None:
        return None
    points = np.asarray(corners, np.float32).reshape(-1, 2)
    grid = points.reshape(board.rows, board.columns, 2)
    dx, dy = np.diff(grid, axis=1), np.diff(grid, axis=0)
    lengths = np.concatenate((np.linalg.norm(dx, axis=2).ravel(), np.linalg.norm(dy, axis=2).ravel()))
    homography, _ = cv2.findHomography(board.object_points()[:, :2], points, 0)
    if homography is None:
        raise ValueError("Could not fit a checkerboard grid homography")
    prediction = cv2.perspectiveTransform(board.object_points()[:, :2].reshape(-1, 1, 2), homography)
    error = points - prediction.reshape(-1, 2)
    # A homography is a diagnostic approximation; lens distortion can also contribute.
    return {
        "homography_rms_px": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
        "homography_max_px": float(np.max(np.linalg.norm(error, axis=1))),
        "neighbor_pitch_median_px": float(np.median(lengths)),
        "neighbor_pitch_min_px": float(np.min(lengths)),
        "row_axis": np.mean(dx, axis=(0, 1)).tolist(),
        "column_axis": np.mean(dy, axis=(0, 1)).tolist(),
        "index_0": points[0].tolist(), "index_last": points[-1].tolist(),
    }


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else None


def inspect_detection(image, board):
    """Reproduce detect_corners exactly; alternate refinements are evidence only."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (board.columns, board.rows)
    found, sb = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        corners = sb.astype(np.float32).reshape(-1, 1, 2)
        return corners, {"detector": "SB", "grid": grid_metrics(corners, board)}, {}
    found, initial = cv2.findChessboardCorners(
        gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None, {"detector": "not_detected", "grid": None}, {}
    initial = initial.astype(np.float32).reshape(-1, 1, 2)
    alternatives = {"classic_initial": initial.copy()}
    for half_window in (2, 3, 5, 11):
        alternatives[f"classic_window_{half_window}"] = cv2.cornerSubPix(
            gray, initial.copy(), (half_window, half_window), (-1, -1), CRITERIA)
    current = alternatives["classic_window_11"]
    comparison = {}
    for name, corners in alternatives.items():
        comparison[name] = grid_metrics(corners, board)
        comparison[name]["mean_shift_from_initial_px"] = float(
            np.mean(np.linalg.norm(corners.reshape(-1, 2) - initial.reshape(-1, 2), axis=1)))
    return current, {"detector": "classic_fallback", "grid": grid_metrics(current, board),
                     "refinement_comparison": comparison}, alternatives


def numbered_corners(image, corners, board, title, scale=3):
    """Enlarge only the debug PNG. Calibration always uses unscaled coordinates."""
    enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = cv2.copyMakeBorder(enlarged, 42, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(canvas, title[:90], (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if corners is None:
        cv2.putText(canvas, "NOT DETECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return canvas
    points = np.round(corners.reshape(-1, 2) * scale).astype(np.int32)
    points[:, 1] += 42
    grid = points.reshape(board.rows, board.columns, 2)
    for row in grid:
        cv2.polylines(canvas, [np.ascontiguousarray(row).reshape(-1, 1, 2)], False, (100, 150, 100), 1)
    for column in grid.transpose(1, 0, 2):
        cv2.polylines(canvas, [np.ascontiguousarray(column).reshape(-1, 1, 2)], False, (100, 100, 150), 1)
    for index, (x, y) in enumerate(points):
        color = (0, 0, 255) if index == 0 else ((0, 255, 255) if index == len(points) - 1 else (0, 255, 0))
        cv2.circle(canvas, (int(x), int(y)), 3 if index in (0, len(points) - 1) else 2, color, -1)
        position = (int(x) + 3, int(y) - 3)
        cv2.putText(canvas, str(index), position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(index), position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


def write_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def summary(result):
    report = result["report"]
    rotation = np.asarray(result["matrices"]["R"], np.float64)
    return {key: report[key] for key in (
        "used_pairs", "left_reprojection_rms_px", "right_reprojection_rms_px",
        "stereo_rms_px", "baseline_cm", "vertical_rms_px", "warnings", "numeric_checks_passed"
    )} | {"R": rotation.tolist(), "T": result["matrices"]["T"],
         "relative_rotation_degrees": float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation)[0])))}


def file_hashes(paths):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_diagnosis(dataset, reference_path, output):
    dataset, reference_path, output = Path(dataset).resolve(), Path(reference_path).resolve(), Path(output).resolve()
    reference = load_result(reference_path)
    board = BoardSettings(**reference["board"])
    quality = QualitySettings(**reference["quality_settings"])
    orientation = result_orientation(reference)
    source_paths = [path for path in dataset.rglob("*") if path.is_file()] + [reference_path]
    hashes = file_hashes(source_paths)
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError("Diagnostics must be outside the source dataset")
    output.mkdir(parents=True, exist_ok=False)
    debug = output / "corners"
    debug.mkdir()
    filtered = output / "dataset_filtered"
    filtered.mkdir()
    reproduced = calibrate_dataset(dataset, board, quality, orientation)
    save_result(output / "reproduced_original.json", reproduced)
    errors = {item["pair_id"]: item for item in reproduced["report"]["per_pair"]}
    records, selected, rejected = [], [], []
    for directory in sorted(dataset.glob("pair_*")):
        metadata, paths, raw = load_pair(directory / "pair.json")
        declared = pair_orientation(metadata)
        if declared is not None:
            require_matching_orientation(declared, orientation)
        images = prepare_pair(*raw, orientation)
        detections = [inspect_detection(image, board) for image in images]
        record = {"pair_id": metadata["pair_id"], "source_directory": str(directory),
                  "pc_receive_delta_ms": metadata["pc_receive_delta_ms"],
                  "left": detections[0][1], "right": detections[1][1],
                  "original_fit_errors": errors.get(metadata["pair_id"]),
                  "corner_coordinates": {side: None if detection[0] is None else detection[0].reshape(-1, 2).tolist()
                                         for side, detection in zip(("left", "right"), detections)}}
        numbered = []
        for side, image, detection in zip(("left", "right"), images, detections):
            corners, info, alternatives = detection
            numbered.append(numbered_corners(image, corners, board, f"{side.upper()} | {info['detector']} | {metadata['pair_id'][:22]}"))
            write_png(debug / f"{directory.name}_{side}.png", numbered[-1])
            if alternatives:
                panels = [numbered_corners(image, alternatives[name], board, name)
                          for name in ("classic_initial", "classic_window_11", "classic_window_5")]
                write_png(debug / f"{directory.name}_{side}_refinement.png", np.hstack(panels))
        write_png(debug / f"{directory.name}_pair.png", np.hstack(numbered))
        if all(detection[0] is not None for detection in detections):
            a, b = record["left"]["grid"], record["right"]["grid"]
            record["row_axis_cosine"] = cosine(a["row_axis"], b["row_axis"])
            record["column_axis_cosine"] = cosine(a["column_axis"], b["column_axis"])
            record["possible_180_index_conflict"] = record["row_axis_cosine"] < -0.9 and record["column_axis_cosine"] < -0.9
        fit_errors = record["original_fit_errors"]
        # Use the EXISTING monochrome quality threshold, not desired baseline or filtered-fit scores.
        retain = fit_errors is not None and max(fit_errors["left_reprojection_rms_px"],
                                                fit_errors["right_reprojection_rms_px"]) <= quality.max_reprojection_px
        record["retained"] = retain
        record["selection_reason"] = (
            "both per-camera RMS <= original threshold" if retain else
            ("checkerboard detection failed" if fit_errors is None else "per-camera RMS exceeds original threshold"))
        (selected if retain else rejected).append(metadata["pair_id"])
        if retain:
            destination = filtered / directory.name
            destination.mkdir()
            for source in [directory / "pair.json", *paths]:
                relative = source.relative_to(directory)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        records.append(record)
    refitted = calibrate_dataset(filtered, board, quality, orientation)
    save_result(output / "calibration_filtered.json", refitted)
    report = {
        "source_dataset": str(dataset), "reference_result": str(reference_path),
        "board": reference["board"], "orientation": orientation.to_dict(),
        "selection_rule": f"Original per-camera reprojection RMS <= {quality.max_reprojection_px}px; no reordering",
        "selected_pair_ids": selected, "rejected_pair_ids": rejected,
        "reference": summary(reference), "reproduced_original": summary(reproduced), "filtered": summary(refitted),
        "pairs": records, "source_sha256": hashes,
        "source_preserved": file_hashes(source_paths) == hashes,
        "physical_mapping_and_mirroring": "unverified; positive T_x is compatible with swapped labels or common mirror",
        "validation_status": "diagnostic_in_sample_only; physical_measurement_unverified",
    }
    if not report["source_preserved"]:
        raise RuntimeError("Source hashes changed during diagnosis")
    (output / "diagnosis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    columns = ["pair_id", "left_detector", "right_detector", "left_rms_px", "right_rms_px",
               "vertical_rms_px", "left_grid_rms_px", "right_grid_rms_px", "possible_180_index_conflict", "retained"]
    with (output / "pair_errors.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            error = record["original_fit_errors"] or {}
            writer.writerow({"pair_id": record["pair_id"], "left_detector": record["left"]["detector"],
                             "right_detector": record["right"]["detector"],
                             "left_rms_px": error.get("left_reprojection_rms_px"),
                             "right_rms_px": error.get("right_reprojection_rms_px"),
                             "vertical_rms_px": error.get("vertical_rms_px"),
                             "left_grid_rms_px": (record["left"]["grid"] or {}).get("homography_rms_px"),
                             "right_grid_rms_px": (record["right"]["grid"] or {}).get("homography_rms_px"),
                             "possible_180_index_conflict": record.get("possible_180_index_conflict"),
                             "retained": record["retained"]})
    for name in ("reference", "reproduced_original", "filtered"):
        print(name, json.dumps(report[name], ensure_ascii=False))
    print(f"Diagnostic outputs: {output}; source hashes preserved={report['source_preserved']}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the source dataset")
    args = parser.parse_args(argv)
    try:
        run_diagnosis(args.dataset, args.reference, args.output)
        return 0
    except (OSError, ValueError, KeyError, cv2.error) as error:
        print(f"Diagnosis not completed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

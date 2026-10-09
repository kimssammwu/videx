"""Preserving audit of a stereo result: signed depth, unused views and cross-validation.

No image/corner reversal, camera swap, baseline constraint or model selection.
Only a NEW copy's validation reference changes; all original matrices stay intact.
"""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np

from .calibration import BoardSettings, Observation, QualitySettings, fit_calibration, load_result, save_result, UNIT_METERS
from .dataset import load_pair
from .diagnose import cosine, file_hashes, grid_metrics, numbered_corners, write_png
from .orientation import pair_orientation, prepare_pair, require_matching_orientation, result_orientation


SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


def stats(values):
    values = np.asarray(values, np.float64).ravel()
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0, "nonfinite_count": int(len(values))}
    return {"count": int(len(finite)), "nonfinite_count": int(len(values) - len(finite)),
            "min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean()),
            "median": float(np.median(finite)), "rms": float(np.sqrt(np.mean(finite**2))),
            "abs_p95": float(np.percentile(np.abs(finite), 95))}


def q_points(q, left_xy, disparity):
    """Q input uses x_LEFT - x_RIGHT, in pixels; output is in the board's unit."""
    left_xy = np.asarray(left_xy, np.float64).reshape(-1, 2)
    disparity = np.asarray(disparity, np.float64).reshape(-1)
    if len(left_xy) != len(disparity):
        raise ValueError("One disparity is required per left point")
    vectors = np.column_stack((left_xy, disparity, np.ones(len(disparity))))
    homogeneous = vectors @ np.asarray(q, np.float64).T
    xyz = np.full((len(disparity), 3), np.nan)
    valid = np.all(np.isfinite(homogeneous), axis=1) & (np.abs(homogeneous[:, 3]) > 1e-12)
    xyz[valid] = homogeneous[valid, :3] / homogeneous[valid, 3:4]
    return xyz, valid


def reference_copy(result, baseline_mm):
    if not np.isfinite(baseline_mm) or baseline_mm <= 0:
        raise ValueError("Measured reference baseline must be positive")
    copy = deepcopy(result)
    copy["quality_settings"]["expected_baseline_cm"] = baseline_mm / 10
    report = copy["report"]
    baseline = np.linalg.norm(result["matrices"]["T"]) * UNIT_METERS[result["length_unit"]] * 1000
    difference = abs(baseline - baseline_mm) / baseline_mm
    report["expected_baseline_cm"] = baseline_mm / 10
    report["baseline_relative_difference"] = float(difference)
    baseline_warning = "Estimated baseline differs substantially from the approximate hardware baseline"
    report["warnings"] = [w for w in report["warnings"] if w != baseline_warning]
    if difference > copy["quality_settings"]["baseline_relative_tolerance"]:
        report["warnings"].append(baseline_warning)
    report["numeric_checks_passed"] = not report["warnings"]
    report["validation_status"] = "physical_measurement_unverified"
    copy["validation_reference_update"] = {"expected_baseline_mm": baseline_mm,
                                           "matrices_unchanged": copy["matrices"] == result["matrices"],
                                           "meaning": "reference-only; not a constrained calibration or distance certification"}
    return copy


def evaluate_points(result, observation):
    m = {k: np.asarray(v, np.float64) for k, v in result["matrices"].items()}
    rect = [cv2.undistortPoints(points, m["K_" + side], m["D_" + side],
                                R=m["R" + str(i + 1)], P=m["P" + str(i + 1)]).reshape(-1, 2)
            for i, (side, points) in enumerate((("left", observation.left), ("right", observation.right)))]
    disparity = rect[0][:, 0] - rect[1][:, 0]
    vertical = rect[0][:, 1] - rect[1][:, 1]
    xyz_q, valid_q = q_points(m["Q"], rect[0], disparity)
    # Independent triangulation in the original camera coordinates, using undistorted rays.
    normal = [cv2.undistortPoints(points, m["K_" + side], m["D_" + side]).reshape(-1, 2)
              for side, points in (("left", observation.left), ("right", observation.right))]
    h = cv2.triangulatePoints(np.column_stack((np.eye(3), np.zeros(3))),
                             np.column_stack((m["R"], m["T"])), normal[0].T, normal[1].T).T
    valid_t = np.all(np.isfinite(h), axis=1) & (np.abs(h[:, 3]) > 1e-12)
    xyz_l = np.full((len(h), 3), np.nan)
    xyz_l[valid_t] = h[valid_t, :3] / h[valid_t, 3:4]
    xyz_r = xyz_l @ m["R"].T + m["T"].reshape(1, 3)
    xyz_rect = xyz_l @ m["R1"].T
    scale = UNIT_METERS[result["length_unit"]] * 1000
    metrics = {"pair_id": observation.pair_id, "vertical_px": stats(vertical), "disparity_px": stats(disparity),
               "q_depth_mm": stats(xyz_q[:, 2] * scale),
               "left_depth_mm": stats(xyz_l[:, 2] * scale), "right_depth_mm": stats(xyz_r[:, 2] * scale),
               "q_positive_depth_count": int(np.sum(valid_q & (xyz_q[:, 2] > 0))),
               "both_cameras_positive_depth_count": int(np.sum(valid_t & (xyz_l[:, 2] > 0) & (xyz_r[:, 2] > 0))),
               "corner_count": len(h),
               "q_vs_triangulated_rect_depth_delta_mm": stats((xyz_q[:, 2] - xyz_rect[:, 2]) * scale)}
    return metrics, {"left_rect": rect[0], "right_rect": rect[1], "disparity": disparity,
                     "vertical": vertical, "q_depth_mm": xyz_q[:, 2] * scale,
                     "left_depth_mm": xyz_l[:, 2] * scale, "right_depth_mm": xyz_r[:, 2] * scale}


def cross_validation_splits(count):
    if count < 11:
        raise ValueError("At least 11 views are needed to keep >=10 training pairs in each split")
    indices = np.arange(count)
    splits = [("leave_one_out", [i]) for i in range(count)]
    # Contiguous capture-time groups give a stronger test than neighboring frames shuffled at random.
    splits += [("chronological_blocks", group.tolist()) for group in np.array_split(indices, 4)]
    return [(scheme, sorted(set(range(count)) - set(test)), test) for scheme, test in splits]


def radial_audit(result, observations):
    """Check radial extrapolation on a nominal K-normalized sensor field (not measured FOV)."""
    m = {k: np.asarray(v, float) for k, v in result["matrices"].items()}
    w, h = result["image_size"]
    output = {}
    for side in ("left", "right"):
        k, d = m["K_" + side], m["D_" + side].ravel()
        if len(d) != 5:
            output[side] = {"evaluated": False, "reason": "only five coefficient model audited"}
            continue
        corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], float)
        nominal_r = np.linalg.norm((corners - k[:2, 2]) / [k[0, 0], k[1, 1]], axis=1).max()
        observed = np.concatenate([getattr(o, side).reshape(-1, 2) for o in observations])
        rays = cv2.undistortPoints(observed.reshape(-1, 1, 2), k, d).reshape(-1, 2)
        observed_r = np.linalg.norm(rays, axis=1).max()
        def minimum_derivative(radius):
            r = np.linspace(0, radius, 1000)
            return float(np.min(1 + 3*d[0]*r**2 + 5*d[1]*r**4 + 7*d[4]*r**6))
        output[side] = {"evaluated": True, "nominal_sensor_radius": float(nominal_r),
                        "observed_undistorted_radius_max": float(observed_r),
                        "radial_derivative_min_nominal_sensor": minimum_derivative(nominal_r),
                        "radial_derivative_min_observed": minimum_derivative(observed_r),
                        "interpretation": "negative derivative warns about extrapolation; not an independent lens measurement"}
    return output


def run_validation(dataset, result_path, selection_path, output, baseline_mm=95.0, reviewed_holdouts=None):
    dataset, result_path, selection_path, output = [Path(p).resolve() for p in
                                                   (dataset, result_path, selection_path, output)]
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError("Validation output must be outside original data")
    result = load_result(result_path)
    if result.get("baseline_constraint_mm") is not None:
        raise ValueError("This audit requires the unconstrained calibration")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    quality = QualitySettings(**reference_copy(result, baseline_mm)["quality_settings"])
    board, orientation = BoardSettings(**result["board"]), result_orientation(result)
    protected = list(dataset.rglob("*")) + list(result_path.parent.rglob("*"))
    old_result = Path(selection.get("reference_result", ""))
    if old_result.is_file():
        protected.append(old_result)
    source_modules = [Path(__file__).with_name(name) for name in
                      ("validate.py", "calibration.py", "diagnose.py", "orientation.py", "dataset.py")]
    protected.extend(source_modules)
    protected = sorted(set(p for p in protected if p.is_file()))
    if any(output == p or output.is_relative_to(p.parent) for p in (result_path, selection_path)):
        raise ValueError("Use an output directory separate from existing result/selection directories")
    hashes = file_hashes(protected)
    output.mkdir(parents=True, exist_ok=False)
    debug = output / "holdout_corners"
    debug.mkdir()
    write_json = lambda name, value: (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                                                          allow_nan=False), encoding="utf-8")
    write_json("input_hashes.json", hashes)
    events = output / "experiment.jsonl"
    def log(event, **payload):
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                                     "event": event, **payload}, ensure_ascii=False, allow_nan=False) + "\n")
        print(event, json.dumps(payload, ensure_ascii=False), flush=True)
    log("start", python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
        expected_baseline_mm=baseline_mm, source_result=str(result_path),
        fixed_baseline=False, corner_reversal=False, camera_swap=False)
    reference95 = reference_copy(result, baseline_mm)
    save_result(output / "calibration_reference95.json", reference95)
    source_records = {r["pair_id"]: r for r in selection["pairs"]}
    threshold = result["quality_settings"]["max_reprojection_px"]
    reconstructed_selection = sorted(r["pair_id"] for r in selection["pairs"]
                                     if r["original_fit_errors"] is not None and
                                     max(r["original_fit_errors"]["left_reprojection_rms_px"],
                                         r["original_fit_errors"]["right_reprojection_rms_px"]) <= threshold)
    selected = sorted(selection["selected_pair_ids"])
    fitted = sorted(p["pair_id"] for p in result["report"]["per_pair"])
    if reconstructed_selection != selected or fitted != selected:
        raise ValueError("Selection provenance disagrees with the result's actual fitted views")
    reviewed = {}
    if reviewed_holdouts:
        review_path = Path(reviewed_holdouts).resolve()
        review_content = json.loads(review_path.read_text(encoding="utf-8"))
        reviewed = {r["pair_id"]: r for r in review_content["pairs"]}
        write_json("visual_review_manifest.json", review_content)
    train, unused, failed, corners_rows, all_metrics = [], [], [], [], []
    for directory in sorted(dataset.glob("pair_*")):
        metadata, paths, raw = load_pair(directory / "pair.json")
        declared = pair_orientation(metadata)
        if declared is not None:
            require_matching_orientation(declared, orientation)
        images = prepare_pair(*raw, orientation, result["image_size"])
        pair_id = metadata["pair_id"]
        if pair_id in selected:
            coordinates = source_records[pair_id]["corner_coordinates"]
            pts = [np.asarray(coordinates[side], np.float32).reshape(-1, 1, 2) for side in ("left", "right")]
            # Fixed, previously fitted coordinates avoid replacing the input to the model under audit.
            for image, point in zip(images, pts):
                found, same = cv2.findChessboardCornersSB(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                                        (board.columns, board.rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                if not found or not np.allclose(same.reshape(-1, 2), point.reshape(-1, 2), atol=1e-4, rtol=0):
                    raise ValueError("Stored fitted corners do not reproduce with the original SB detector")
            obs = Observation(pair_id, *pts, tuple(result["image_size"]))
            train.append(obs)
            role = "training_in_sample"
        else:
            pts = []
            for image in images:
                found, corners = cv2.findChessboardCornersSB(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                                            (board.columns, board.rows), flags=SB_FLAGS)
                pts.append(corners if found else None)
            panels = [numbered_corners(image, points, board, side + " SB accurate " + pair_id[-8:], scale=2)
                      for image, points, side in zip(images, pts, ("LEFT", "RIGHT"))]
            write_png(debug / (pair_id + ".png"), np.hstack(panels))
            audit = {"pair_id": pair_id, "found": [p is not None for p in pts], "input_to_fit": False,
                     "camera_ids": [metadata[side]["camera_id"] for side in ("left", "right")],
                     "physical_camera_mapping_verified": metadata.get("physical_camera_mapping_verified"),
                     "grid": [grid_metrics(p, board) for p in pts]}
            if any(p is None for p in pts):
                audit["evaluation_status"] = "not_evaluable_detection_failure"
                failed.append(audit)
                continue
            axes = [cosine(audit["grid"][0][key], audit["grid"][1][key])
                    for key in ("row_axis", "column_axis")]
            digest = hashlib.sha256(np.concatenate(pts).astype(np.float32).tobytes()).hexdigest()
            source_digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            review = reviewed.get(pair_id)
            audit.update(direction_cosines=axes, corner_sha256=digest, source_jpeg_sha256=source_digests,
                         visual_review_verified=bool(review and review["corner_sha256"] == digest and
                                                     review["source_jpeg_sha256"] == source_digests))
            # Do not choose an ordering based on epipolar score, expected depth or baseline.
            usable = (audit["visual_review_verified"] and all(a > 0.5 for a in axes) and
                      all(g["homography_rms_px"] <= 1.0 for g in audit["grid"]))
            audit["evaluation_status"] = "reviewed_same_index" if usable else "candidate_requires_review"
            unused.append(audit)
            if not usable:
                continue
            obs = Observation(pair_id, *pts, tuple(result["image_size"]))
            role = "unused_retrospective_holdout"
        metrics, arrays = evaluate_points(result, obs)
        metrics["role"] = role
        all_metrics.append(metrics)
        if role == "unused_retrospective_holdout":
            unused[-1]["metrics"] = metrics
        for index in range(len(arrays["disparity"])):
            corners_rows.append({"pair_id": pair_id, "role": role, "corner_index": index,
                                 **{k: float(arrays[k][index]) for k in
                                    ("disparity", "vertical", "q_depth_mm", "left_depth_mm", "right_depth_mm")}})
    def group_summary(role):
        rows = [r for r in corners_rows if r["role"] == role]
        return {"pair_count": len({r["pair_id"] for r in rows}), "corner_count": len(rows),
                **{k: stats([r[k] for r in rows]) for k in ("vertical", "disparity", "q_depth_mm")},
                "both_camera_positive_depth_count": sum(r["left_depth_mm"] > 0 and r["right_depth_mm"] > 0 for r in rows),
                "q_positive_depth_count": sum(r["q_depth_mm"] > 0 for r in rows)}
    reference_m = {k: np.asarray(v, float) for k, v in result["matrices"].items()}
    sample_d = np.array([-50., -10., 0., 10., 50.])
    q_xyz, q_valid = q_points(reference_m["Q"], np.tile(reference_m["P1"][:2, 2], (len(sample_d), 1)), sample_d)
    sign_probe = [{"disparity_px": float(d), "depth_mm": float(z*UNIT_METERS[result["length_unit"]]*1000)
                   if valid else None, "finite": bool(valid)} for d, z, valid in zip(sample_d, q_xyz[:, 2], q_valid)]
    sign = {"convention": "d = x_LEFT - x_RIGHT after saved rectification",
            "T_mm": (reference_m["T"].ravel()*UNIT_METERS[result["length_unit"]]*1000).tolist(),
            "camera2_center_in_camera1_mm": (-reference_m["R"].T @ reference_m["T"] * UNIT_METERS[result["length_unit"]]*1000).ravel().tolist(),
            "rectified_Tx_mm": float(reference_m["P2"][0, 3]/reference_m["P2"][0, 0]*UNIT_METERS[result["length_unit"]]*1000),
            "Q_3_2": float(reference_m["Q"][3, 2]), "Q": result["matrices"]["Q"], "probe": sign_probe,
            "physical_mapping_verified": False,
            "interpretation": "positive Tx places camera2 at negative X in camera1; physical labels/mirroring need hardware evidence"}
    log("fixed_result_evaluated", training=group_summary("training_in_sample"),
        unused=group_summary("unused_retrospective_holdout"), unevaluable=len(failed))
    cv_dir = output / "cross_validation"
    cv_dir.mkdir()
    folds = []
    for index, (scheme, training, test) in enumerate(cross_validation_splits(len(train))):
        fit = fit_calibration([train[i] for i in training], board, quality, orientation, baseline_mm=None)
        save_result(cv_dir / f"{index:02}_{scheme}.json", fit)
        evaluation = [evaluate_points(fit, train[i]) for i in test]
        vertical = np.concatenate([a["vertical"] for _, a in evaluation])
        normalized = vertical * reference_m["P1"][0, 0] / fit["matrices"]["P1"][0][0]
        fold = {"scheme": scheme, "train_pair_ids": [train[i].pair_id for i in training],
                "test_pair_ids": [train[i].pair_id for i in test], "heldout": [m for m, _ in evaluation],
                "vertical_px": stats(vertical), "vertical_at_reference_focal_px": stats(normalized),
                "baseline_mm": float(fit["report"]["baseline_cm"]*10),
                "rectified_focal_px": float(fit["matrices"]["P1"][0][0]),
                "K_focal_px": [fit["matrices"][key][axis][axis] for key in ("K_left", "K_right") for axis in (0, 1)],
                "radial_audit": radial_audit(fit, [train[i] for i in training])}
        folds.append(fold)
        log("cross_validation_fold", fold=index, scheme=scheme, baseline_mm=fold["baseline_mm"],
            heldout_vertical_rms_px=fold["vertical_px"]["rms"],
            heldout_reference_focal_rms_px=fold["vertical_at_reference_focal_px"]["rms"])
    cv_summary = {}
    for scheme in ("leave_one_out", "chronological_blocks"):
        group = [f for f in folds if f["scheme"] == scheme]
        count = sum(f["vertical_px"]["count"] for f in group)
        cv_summary[scheme] = {
            "heldout_vertical_rms_px": float(np.sqrt(sum(f["vertical_px"]["rms"]**2*f["vertical_px"]["count"] for f in group)/count)),
            "heldout_vertical_at_reference_focal_rms_px": float(np.sqrt(sum(f["vertical_at_reference_focal_px"]["rms"]**2*f["vertical_px"]["count"] for f in group)/count)),
            "baseline_mm": stats([f["baseline_mm"] for f in group]),
            "baseline_std_mm": float(np.std([f["baseline_mm"] for f in group])),
            "max_fold_vertical_rms_px": max(f["vertical_px"]["rms"] for f in group),
            "folds_negative_radial_derivative_nominal_sensor": sum(any(v.get("radial_derivative_min_nominal_sensor", 1) < 0 for v in f["radial_audit"].values()) for f in group)}
    if file_hashes(protected) != hashes:
        raise RuntimeError("An input changed while validation ran; do not trust this experiment")
    validation = {"source_result": str(result_path), "source_dataset": str(dataset),
                  "expected_baseline_mm": baseline_mm, "baseline_mm": float(reference95["report"]["baseline_cm"]*10),
                  "baseline_relative_difference": reference95["report"]["baseline_relative_difference"],
                  "matrices_unchanged": reference95["matrices"] == result["matrices"],
                  "source_preservation": {"checked_files": len(hashes), "all_hashes_match": True},
                  "selection_rule": selection["selection_rule"], "selection_rule_reproduced": True,
                  "selection": [{"pair_id": r["pair_id"], "retained": r["retained"],
                                 "reason": r["selection_reason"], "original_fit_errors": r["original_fit_errors"]}
                                for r in selection["pairs"]],
                  "training": group_summary("training_in_sample"),
                  "unused_holdout": group_summary("unused_retrospective_holdout"),
                  "per_pair": all_metrics, "unused_detection_audit": unused, "unevaluable": failed,
                  "sign_and_cheirality": sign, "cross_validation": cv_summary, "folds": folds,
                  "radial_audit": radial_audit(result, train),
                  "limitations": ["Retrospective selection from the same session; not a prospective external validation",
                                  "Leave-one-out and blocks condition on the previously selected 14 pairs; selection leakage remains",
                                  "Measured target distances and physical camera/mirror evidence are unavailable",
                                  "Positive depth establishes sign consistency, not metric depth accuracy",
                                  "No baseline constraint, corner permutation, image mirror, or camera swapping"],
                  "decision": "experimental_sparse_geometry_supported; metric_distance_use_not_yet_validated"}
    write_json("validation.json", validation)
    with (output / "corner_geometry.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(corners_rows[0]))
        writer.writeheader()
        writer.writerows(corners_rows)
    log("complete", cross_validation=cv_summary, matrices_unchanged=True, all_source_hashes_match=True,
        decision=validation["decision"])
    return validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside original data/results")
    parser.add_argument("--expected-baseline-mm", type=float, default=95.0)
    parser.add_argument("--reviewed-holdouts", type=Path, help="Visual review manifest with JPEG and corner hashes")
    args = parser.parse_args(argv)
    try:
        run_validation(args.dataset, args.result, args.selection, args.output,
                       args.expected_baseline_mm, args.reviewed_holdouts)
        return 0  # Audit completion; this is not a distance-accuracy PASS.
    except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as error:
        print(f"Validation incomplete: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

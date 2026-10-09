"""Reproduce free/fixed/scaled stereo comparisons without modifying inputs."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .baseline import StereoResidual, angles_from_translation
from .calibration import (BoardSettings, QualitySettings, collect_observations, corner_consistency,
                          fit_calibration, load_result, rectification_coverage, save_result)
from .dataset import load_pair
from .orientation import prepare_pair, result_orientation
from .rectify import epipolar_preview, rectify_pair


def evaluate_geometry(result, observations, board):
    """Fit each board pose with fixed stereo geometry; use both cameras' residuals.

    Stereo RMS uses sqrt(sum squared 2D pixel errors / (2*N*corners)),
    the OpenCV convention. Monocular RMS in result.report has a different pose fit.
    """
    from scipy.optimize import least_squares
    m = {key: np.asarray(value, np.float64) for key, value in result["matrices"].items()}
    rvec = cv2.Rodrigues(m["R"])[0].ravel()
    angle = angles_from_translation(m["T"])
    length = np.linalg.norm(m["T"])
    errors, vertical, per_pair = [], [], []
    for o in observations:
        _, r, t = cv2.solvePnP(board.object_points(), o.left, m["K_left"], m["D_left"])
        _, rr, tr = cv2.solvePnP(board.object_points(), o.right, m["K_right"], m["D_right"])
        inverse_r = cv2.Rodrigues(m["R"].T @ cv2.Rodrigues(rr)[0])[0]
        inverse_t = m["R"].T @ (tr - m["T"])
        objective = StereoResidual(board.object_points(), [o.left], [o.right],
                                   m["K_left"], m["D_left"], m["K_right"], m["D_right"], length)
        fixed = np.r_[rvec, angle]
        seeds = [np.r_[r.ravel(), t.ravel()], np.r_[inverse_r.ravel(), inverse_t.ravel()]]
        fits = [least_squares(lambda x: objective.evaluate(np.r_[fixed, x]), seed,
                             jac=lambda x: objective.evaluate(np.r_[fixed, x], True)[:, 5:],
                             x_scale="jac", max_nfev=200, ftol=1e-10, xtol=1e-10, gtol=1e-8) for seed in seeds]
        fit = min(fits, key=lambda f: f.cost)
        residual = fit.fun.reshape(2, -1, 2)
        side_errors = np.sqrt(np.mean(np.sum(residual**2, axis=2), axis=1))
        errors.append(side_errors)
        l = cv2.undistortPoints(o.left, m["K_left"], m["D_left"], R=m["R1"], P=m["P1"])
        r = cv2.undistortPoints(o.right, m["K_right"], m["D_right"], R=m["R2"], P=m["P2"])
        delta = (l[:, 0, 1] - r[:, 0, 1]).astype(np.float64)
        vertical.extend(delta)
        _, audit = corner_consistency(o.left, o.right, board)
        points_l = (cv2.Rodrigues(fit.x[:3])[0] @ np.asarray(board.object_points(), np.float64).T).T + fit.x[3:]
        points_r = (m["R"] @ points_l.T).T + m["T"].ravel()
        per_pair.append({"pair_id": o.pair_id, "left_joint_reprojection_rms_px": float(side_errors[0]),
                         "right_joint_reprojection_rms_px": float(side_errors[1]),
                         "vertical_rms_px": float(np.sqrt(np.mean(delta**2))),
                         "board_pose_optimizer_converged": bool(fit.success),
                         "minimum_corner_depth": float(min(points_l[:, 2].min(), points_r[:, 2].min())),
                         "corner_consistency": audit})
    errors, vertical = np.asarray(errors), np.asarray(vertical)
    coverage = rectification_coverage(m, tuple(result["image_size"]))
    report = result["report"]
    joint_fit_valid = all(p["minimum_corner_depth"] > 0 and p["board_pose_optimizer_converged"] for p in per_pair)
    horizontal = abs(m["T"][0, 0]) > abs(m["T"][1, 0])
    geometry_passed = bool(horizontal and np.sqrt(np.mean(errors**2)) <= 1.5
                           and np.sqrt(np.mean(vertical**2)) <= 1
                           and all(min(result[key][2:]) > 0 for key in ("roi_left", "roi_right"))
                           and coverage["common_fraction"] >= 0.5
                           and all(p["corner_consistency"]["consistent_parallel_directions"]
                                   and all(e is not None and e <= 1 for e in p["corner_consistency"]["homography_rms_px"])
                                   and p["minimum_corner_depth"] > 0
                                   and p["board_pose_optimizer_converged"] for p in per_pair))
    return {"pairs": len(observations), "baseline_mm": float(length * {"mm": 1, "cm": 10, "m": 1000}[board.unit]),
            "left_monocular_reprojection_rms_px": report["left_reprojection_rms_px"],
            "right_monocular_reprojection_rms_px": report["right_reprojection_rms_px"],
            "left_joint_reprojection_rms_px": float(np.sqrt(np.mean(errors[:, 0]**2))),
            "right_joint_reprojection_rms_px": float(np.sqrt(np.mean(errors[:, 1]**2))),
            "stereo_rms_px": float(np.sqrt(np.mean(errors**2))),
            "solver_reported_stereo_rms_px": report["stereo_rms_px"],
            "vertical_rms_px": float(np.sqrt(np.mean(vertical**2))),
            "vertical_p95_px": float(np.percentile(abs(vertical), 95)),
            "vertical_max_px": float(np.max(abs(vertical))),
            "relative_rotation_degrees": float(np.degrees(np.linalg.norm(rvec))),
            "translation": m["T"].ravel().tolist(), "horizontal_stereo": bool(horizontal),
            "roi_left": result["roi_left"], "roi_right": result["roi_right"],
            "valid_image_coverage": coverage, "geometry_numeric_checks_passed": geometry_passed,
            "joint_fit_physically_valid": joint_fit_valid,
            "physical_camera_mapping_verified": False, "physical_measurement_verified": False,
            "per_pair": per_pair}


def scale_diagnostic(result, baseline_mm):
    """Change only baseline scale. R/T direction and image alignment stay fixed."""
    result = copy.deepcopy(result)
    m = {k: np.asarray(v, np.float64) for k, v in result["matrices"].items()}
    length = baseline_mm / {"mm": 1, "cm": 10, "m": 1000}[result["length_unit"]]
    m["T"] *= length / np.linalg.norm(m["T"])
    x, y, z = m["T"].ravel()
    m["E"] = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ m["R"]
    m["F"] = np.linalg.inv(m["K_right"]).T @ m["E"] @ np.linalg.inv(m["K_left"])
    r1, r2, p1, p2, q, roi_l, roi_r = cv2.stereoRectify(
        m["K_left"], m["D_left"], m["K_right"], m["D_right"], tuple(result["image_size"]),
        m["R"], m["T"], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m.update(R1=r1, R2=r2, P1=p1, P2=p2, Q=q)
    result["matrices"] = {k: v.tolist() for k, v in m.items()}
    result.update(roi_left=list(roi_l), roi_right=list(roi_r), calibration_method="scaled_T_diagnostic",
                  baseline_constraint_mm=baseline_mm)
    result["report"].update(baseline=length, baseline_cm=baseline_mm / 10,
                            baseline_relative_difference=abs(baseline_mm / 10 - result["report"]["expected_baseline_cm"])
                            / result["report"]["expected_baseline_cm"], numeric_checks_passed=False,
                            valid_image_coverage=rectification_coverage(m, tuple(result["image_size"])))
    result["report"]["warnings"].append("DIAGNOSTIC ONLY: T rescaling is not recalibration; no geometry was optimized")
    result["report"].pop("optimizer", None)
    for pair in result["report"]["per_pair"]:
        pair.pop("stereo_left_reprojection_rms_px", None)
        pair.pop("stereo_right_reprojection_rms_px", None)
    # Inherited stereo RMS is stale after changing T: evaluate_geometry fills it before saving.
    return result


def input_hashes(dataset, previous):
    paths = sorted(p for p in Path(dataset).rglob("*") if p.is_file()) + [Path(previous)]
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_comparison(previous, output, baseline_mm=95.0, dataset=None, holdout=False):
    if not np.isfinite(baseline_mm) or baseline_mm <= 0:
        raise ValueError("baseline_mm must be finite and positive")
    old = load_result(previous)
    board, orientation = BoardSettings(**old["board"]), result_orientation(old)
    dataset = Path(dataset or old["dataset"]).resolve()
    output = Path(output).resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("Comparison output must be outside the original dataset")
    hashes = input_hashes(dataset, previous)
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    quality = QualitySettings(**{**old["quality_settings"], "expected_baseline_cm": baseline_mm / 10})
    legacy, _, meta_l = collect_observations(dataset, board, orientation)
    consistent, _, meta_c = collect_observations(dataset, board, orientation, corner_policy="consistent")
    free = fit_calibration(legacy, board, quality, orientation)
    # Verify that this is the same legacy failure, rather than silently comparing a new subset.
    if [o.pair_id for o in legacy] != [p["pair_id"] for p in old["report"]["per_pair"]]:
        raise ValueError("Legacy pair IDs do not reproduce the supplied previous calibration")
    tag = f"{baseline_mm:g}".replace(".", "p")
    methods = [("original_free", copy.deepcopy(old), legacy, meta_l),
               (f"original_scaled{tag}_diagnostic", scale_diagnostic(old, baseline_mm), legacy, meta_l),
               (f"original_fixed{tag}", fit_calibration(legacy, board, quality, orientation, baseline_mm=baseline_mm), legacy, meta_l)]
    improved_free = fit_calibration(consistent, board, quality, orientation)
    methods.extend([("consistent_free", improved_free, consistent, meta_c),
                    (f"consistent_scaled{tag}_diagnostic", scale_diagnostic(improved_free, baseline_mm), consistent, meta_c),
                    (f"consistent_fixed{tag}", fit_calibration(consistent, board, quality, orientation, baseline_mm=baseline_mm), consistent, meta_c)])
    comparison = {"baseline_constraint_mm": baseline_mm, "previous_calibration": str(Path(previous).resolve()),
                  "dataset": str(dataset), "alpha": 0, "intrinsics_policy": "independent mono, fixed in stereo",
                  "rms_definition": "sqrt(sum of squared 2D errors / (2 * pairs * corners)); joint board pose",
                  "baseline_is_not_success_criterion": True, "quality_evaluation": "in_sample",
                  "legacy_observations": meta_l, "consistent_observations": meta_c,
                  "original_report": old["report"],
                  "legacy_free_reproduction": {k: free["report"][k] for k in ("baseline", "stereo_rms_px", "vertical_rms_px")},
                  "methods": {}}
    for name, result, observations, metadata in methods:
        result["dataset"] = str(dataset)
        result["report"].update(metadata)
        evaluation = evaluate_geometry(result, observations, board)
        evaluation["reevaluated_stereo_rms_px"] = evaluation["stereo_rms_px"]
        result["report"].update(stereo_left_reprojection_rms_px=evaluation["left_joint_reprojection_rms_px"],
                                stereo_right_reprojection_rms_px=evaluation["right_joint_reprojection_rms_px"])
        if "diagnostic" in name:
            result["report"]["stereo_rms_px"] = evaluation["stereo_rms_px"]
            evaluation["solver_reported_stereo_rms_px"] = None
            evaluation["successful_recalibration"] = False
        else:
            # Preserve the stereo solver's objective and the exact original stored RMS.
            evaluation["stereo_rms_px"] = result["report"]["stereo_rms_px"]
            evaluation["successful_recalibration"] = bool(evaluation["geometry_numeric_checks_passed"]
                                                          and result["report"]["numeric_checks_passed"])
        comparison["methods"][name] = evaluation
        save_result(output / f"{name}.json", result)
        preview = output / name
        preview.mkdir()
        # A representative pose and two originally problematic observations.
        selected = [observations[0]] + [o for o in observations if "034109" in o.pair_id or "034214" in o.pair_id]
        for o in selected:
            _, _, raw = load_pair(dataset / f"pair_{o.pair_id}" / "pair.json")
            left, right = rectify_pair(*raw, result)
            cv2.imwrite(str(preview / f"{o.pair_id}_rectified.png"), epipolar_preview(left, right))
    # Save indexed overlays for every improved observation, including all reversals.
    corner_dir = output / "indexed_corners"
    corner_dir.mkdir()
    for o in consistent:
        _, _, raw = load_pair(dataset / f"pair_{o.pair_id}" / "pair.json")
        prepared = prepare_pair(*raw, orientation)
        images = []
        for image, points in zip(prepared, (o.left, o.right)):
            canvas = cv2.resize(image, (480, 640))
            cv2.drawChessboardCorners(canvas, (board.columns, board.rows), points * 2, True)
            for i in (0, board.columns - 1, board.columns * (board.rows - 1), board.columns * board.rows - 1):
                cv2.putText(canvas, str(i), tuple((points[i, 0] * 2).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            images.append(canvas)
        cv2.imwrite(str(corner_dir / f"{o.pair_id}.png"), np.hstack(images))
    if holdout:
        training = [o for i, o in enumerate(consistent) if i % 3 != 0]
        validation = [o for i, o in enumerate(consistent) if i % 3 == 0]
        comparison["held_out_validation"] = {"training_ids": [o.pair_id for o in training],
                                             "validation_ids": [o.pair_id for o in validation], "methods": {}}
        for name, baseline in (("free", None), (f"fixed{tag}", baseline_mm)):
            fit = fit_calibration(training, board, quality, orientation, baseline_mm=baseline)
            comparison["held_out_validation"]["methods"][name] = evaluate_geometry(fit, validation, board)
            save_result(output / f"holdout_{name}.json", fit)
    comparison["inputs_unchanged"] = input_hashes(dataset, previous) == hashes
    if not comparison["inputs_unchanged"]:
        raise ValueError("Inputs changed during comparison")
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Baseline comparison", "", "Baseline length alone is not evidence of calibration success.", "",
             "| Method | Pairs | Baseline mm | Mono L/R px | Joint L/R px | Stereo px | Vertical px | ROI L/R | Common valid % | Geometry checks |",
             "|---|---:|---:|---|---|---:|---:|---|---:|---|"]
    for name, e in comparison["methods"].items():
        joint_text = (f"{e['left_joint_reprojection_rms_px']:.4f}/{e['right_joint_reprojection_rms_px']:.4f}"
                      if e["joint_fit_physically_valid"] else "INVALID (behind camera / nonconvergence)")
        stereo_text = f"{e['stereo_rms_px']:.4f}" if e["joint_fit_physically_valid"] else "INVALID"
        lines.append(f"| {name} | {e['pairs']} | {e['baseline_mm']:.4f} | {e['left_monocular_reprojection_rms_px']:.4f}/{e['right_monocular_reprojection_rms_px']:.4f} | "
                     f"{joint_text} | {stereo_text} | {e['vertical_rms_px']:.4f} | "
                     f"{e['roi_left']}/{e['roi_right']} | {e['valid_image_coverage']['common_fraction']*100:.2f} | {e['geometry_numeric_checks_passed']} |")
    lines.extend(["", "The mono fits are independent of the stereo baseline; joint errors use a shared board pose.",
                  "All fits use fixed independently calibrated intrinsics. No high-error observation is removed in these six methods.",
                  "Corner policy `consistent` changes fallback subpixel windows and explicitly reverses 180-degree right indices for a parallel rig.",
                  "Scaled T methods are diagnostics, never successful recalibrations. Their Stereo RMS is recomputed with fixed geometry.",
                  "Stereo values for fitted methods preserve the solver objective; comparison.json also records per-view reevaluation (local minima may differ for invalid geometry).",
                  "Valid fractions describe remap source support, not retained source field of view; check rectified focal length and previews too.",
                  "Physical L/R mapping, board identity and synchronization remain unverified. An unmarked board has unresolved physical ambiguities.",
                  f"Inputs unchanged (SHA-256): {comparison['inputs_unchanged']}."])
    if holdout:
        lines.extend(["", "## Held-out image pairs (12 train / 7 validation for session_02)", ""])
        for name, e in comparison["held_out_validation"]["methods"].items():
            lines.append(f"- {name}: stereo {e['stereo_rms_px']:.4f}px; vertical {e['vertical_rms_px']:.4f}px; baseline {e['baseline_mm']:.4f}mm.")
        lines.append("The split separates images from parameter fitting, but remains within the same capture session.")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-mm", type=float, default=95)
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_comparison(args.previous, args.output, args.baseline_mm, args.dataset, args.holdout)
        for name, e in result["methods"].items():
            stereo_text = f"{e['stereo_rms_px']:.4f}px" if e["joint_fit_physically_valid"] else "INVALID (behind camera / nonconvergence)"
            print(f"{name}: baseline={e['baseline_mm']:.4f}mm stereo={stereo_text} "
                  f"vertical={e['vertical_rms_px']:.4f}px geometry_checks={e['geometry_numeric_checks_passed']}")
        print(f"Comparison saved: {args.output}; input hashes unchanged={result['inputs_unchanged']}")
        return 0
    except (ValueError, OSError, cv2.error) as error:
        print(f"Comparison not completed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

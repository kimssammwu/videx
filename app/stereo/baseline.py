"""Fixed-length stereo bundle adjustment with independently fitted intrinsics."""

import cv2
import numpy as np


def direction_translation(angles, length):
    """Two angular degrees of freedom; length is exact at every iteration."""
    azimuth, elevation = angles
    ca, sa, ce, se = np.cos(azimuth), np.sin(azimuth), np.cos(elevation), np.sin(elevation)
    translation = length * np.array([ce * ca, ce * sa, se])
    derivative = length * np.array([[-ce * sa, -se * ca], [ce * ca, -se * sa], [0, ce]])
    return translation.reshape(3, 1), derivative


def angles_from_translation(translation):
    x, y, z = np.asarray(translation).ravel()
    return [np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))]


class StereoResidual:
    """Pixel residuals and analytic Jacobian for R, direction(T), and board poses."""

    def __init__(self, objects, left, right, kl, dl, kr, dr, length):
        self.objects = np.asarray(objects, np.float64)
        self.left = np.asarray(left, np.float64).reshape(len(left), -1, 2)
        self.right = np.asarray(right, np.float64).reshape(len(right), -1, 2)
        self.kl, self.dl, self.kr, self.dr, self.length = kl, dl, kr, dr, length

    def evaluate(self, parameters, jacobian=False):
        n, count = self.left.shape[:2]
        residual = np.empty((n, 2, count * 2))
        jac = np.zeros((n * 4 * count, len(parameters))) if jacobian else None
        translation, dt = direction_translation(parameters[3:5], self.length)
        for i in range(n):
            start = 5 + 6 * i
            rotation = parameters[start:start + 3].reshape(3, 1)
            position = parameters[start + 3:start + 6].reshape(3, 1)
            composed = cv2.composeRT(rotation, position, parameters[:3].reshape(3, 1), translation)
            pl, jl = cv2.projectPoints(self.objects, rotation, position, self.kl, self.dl)
            pr, jr = cv2.projectPoints(self.objects, composed[0], composed[1], self.kr, self.dr)
            residual[i, 0] = (pl.reshape(-1, 2) - self.left[i]).ravel()
            residual[i, 1] = (pr.reshape(-1, 2) - self.right[i]).ravel()
            if jacobian:
                a, b = i * 4 * count, (i * 4 + 2) * count
                jac[a:b, start:start + 6] = jl[:, :6]
                jac[b:b + 2 * count, :3] = jr[:, :3] @ composed[4] + jr[:, 3:6] @ composed[8]
                jac[b:b + 2 * count, 3:5] = (jr[:, :3] @ composed[5] + jr[:, 3:6] @ composed[9]) @ dt
                jac[b:b + 2 * count, start:start + 3] = jr[:, :3] @ composed[2] + jr[:, 3:6] @ composed[6]
                jac[b:b + 2 * count, start + 3:start + 6] = jr[:, :3] @ composed[3] + jr[:, 3:6] @ composed[7]
        return jac if jacobian else residual.ravel()


def fit_fixed_baseline(objects, left, right, kl, dl, kr, dr, rotation, translation,
                       rotations_l, translations_l, rotations_r, translations_r, length):
    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise ValueError("Fixed baseline requires scipy; install stereo/requirements.txt") from error
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Fixed baseline must be finite and positive")
    objective = StereoResidual(objects, left, right, kl, dl, kr, dr, length)
    # A failed unconstrained fit can be far from the physical basin. A second seed
    # comes from per-view relative poses, without forcing the fitted R to identity.
    relative = [cv2.Rodrigues(rr)[0] @ cv2.Rodrigues(rl)[0].T
                for rl, rr in zip(rotations_l, rotations_r)]
    median_r = np.median([cv2.Rodrigues(r)[0].ravel() for r in relative], axis=0)
    median_t = np.median([np.asarray(tr).ravel() - r @ np.asarray(tl).ravel()
                          for r, tl, tr in zip(relative, translations_l, translations_r)], axis=0)
    poses = np.concatenate([np.r_[r.ravel(), t.ravel()] for r, t in zip(rotations_l, translations_l)])
    seeds = [("free_stereo", cv2.Rodrigues(rotation)[0].ravel(), translation),
             ("median_relative_pose", median_r, median_t)]
    fits, attempts = [], []
    for name, r, t in seeds:
        x0 = np.r_[r, angles_from_translation(t), poses]
        fit = least_squares(objective.evaluate, x0, jac=lambda x: objective.evaluate(x, True),
                            method="trf", x_scale="jac", max_nfev=300,
                            ftol=1e-10, xtol=1e-10, gtol=1e-8)
        attempts.append({"seed": name, "initial_stereo_rms_px": float(np.sqrt(np.mean(objective.evaluate(x0)**2) * 2)),
                         "stereo_rms_px": float(np.sqrt(np.mean(fit.fun**2) * 2)),
                         "converged": bool(fit.success), "evaluations": int(fit.nfev),
                         "optimality": float(fit.optimality), "message": str(fit.message)})
        fits.append(fit)
    best = min(range(len(fits)), key=lambda i: fits[i].cost)
    fit = fits[best]
    r = cv2.Rodrigues(fit.x[:3])[0]
    t = direction_translation(fit.x[3:5], length)[0]
    fitted_poses = fit.x[5:].reshape(-1, 6)
    residuals = fit.fun.reshape(len(left), 2, -1, 2)
    errors = np.sqrt(np.mean(np.sum(residuals**2, axis=3), axis=2))
    depths = []
    for pose in fitted_poses:
        points_l = (cv2.Rodrigues(pose[:3])[0] @ objective.objects.T).T + pose[3:]
        points_r = (r @ points_l.T).T + t.ravel()
        depths.extend([float(points_l[:, 2].min()), float(points_r[:, 2].min())])
    info = {"method": "fixed_baseline_bundle_adjustment", "intrinsics": "fixed_independent_monocular",
            "optimized": ["relative_rotation_3dof", "translation_direction_2dof", "board_pose_6dof_per_pair"],
            "selected_seed": attempts[best]["seed"], "converged": bool(fit.success),
            "minimum_corner_depth": min(depths), "attempts": attempts,
            "baseline_length_error": float(abs(np.linalg.norm(t) - length))}
    return r, t, float(np.sqrt(np.mean(fit.fun**2) * 2)), errors, info

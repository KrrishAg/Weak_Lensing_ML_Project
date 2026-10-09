from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def regression_metrics(y_true: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    sigma = np.clip(np.asarray(sigma), 1e-8, None)
    err = mean - y_true
    out = {
        "mae_omega_m": float(np.mean(np.abs(err[:, 0]))),
        "mae_s8": float(np.mean(np.abs(err[:, 1]))),
        "rmse_omega_m": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_s8": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "coverage68_omega_m": float(np.mean(np.abs(err[:, 0]) <= sigma[:, 0])),
        "coverage68_s8": float(np.mean(np.abs(err[:, 1]) <= sigma[:, 1])),
        "mean_pred_sigma_omega_m": float(np.mean(sigma[:, 0])),
        "mean_pred_sigma_s8": float(np.mean(sigma[:, 1])),
    }
    out["coverage68_mean"] = (out["coverage68_omega_m"] + out["coverage68_s8"]) / 2
    return out


def nuisance_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    names = ("tagn", "f0", "delta_z")
    out = {}
    for j, name in enumerate(names):
        err = pred[:, j] - y_true[:, j]
        out[f"mae_{name}"] = float(np.mean(np.abs(err)))
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(err**2)))
    return out


def fit_sigma_scale(
    y_true: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    target_coverage: float = 0.68,
) -> np.ndarray:
    """
    Fit a multiplicative sigma calibration factor on a calibration/validation set.

    For each target, choose q such that target_coverage of |error| <= q*sigma.
    The factor is then applied only to the independent test split.
    """

    sigma = np.clip(np.asarray(sigma), 1e-8, None)
    ratio = np.abs(np.asarray(y_true) - np.asarray(mean)) / sigma
    return np.quantile(ratio, target_coverage, axis=0).astype(np.float32)


def coverage_at_levels(
    y_true: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    levels=(0.50, 0.68, 0.90, 0.95),
) -> dict:
    """Gaussian central-interval coverage at several nominal probability levels."""

    # Standard-normal z values for common two-sided central intervals.
    z_lookup = {0.50: 0.67448975, 0.68: 0.99445788, 0.90: 1.64485363, 0.95: 1.95996398}
    err = np.abs(np.asarray(y_true) - np.asarray(mean))
    sigma = np.clip(np.asarray(sigma), 1e-8, None)
    out = {}
    for level in levels:
        z = z_lookup.get(float(level))
        if z is None:
            raise ValueError(f"Unsupported level {level}; add its normal quantile to z_lookup")
        cov = np.mean(err <= z * sigma, axis=0)
        out[f"coverage_{int(round(level * 100))}_omega_m"] = float(cov[0])
        out[f"coverage_{int(round(level * 100))}_s8"] = float(cov[1])
    return out


def phase1_score(y_true, mean, sigma, scale_factor=1000.0):
    """Organizer Phase-1 score, useful as a reference metric for this offline project."""

    sigma = np.clip(sigma, 1e-6, None)
    sq = (y_true - mean) ** 2
    score = -np.sum(sq / sigma**2 + np.log(sigma**2) + scale_factor * sq, axis=1)
    score = float(np.mean(score))
    return max(score, -1e6)


def low_fpr_tpr_score(labels, scores, fpr_min=0.001, fpr_max=0.05, n_grid=100):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    grid = np.geomspace(fpr_min, fpr_max, n_grid)
    interp = np.interp(grid, fpr, tpr, left=0.0, right=tpr[-1])
    return float(interp.mean())


def ood_metrics(ind_scores, ood_scores):
    y = np.r_[np.zeros(len(ind_scores)), np.ones(len(ood_scores))]
    s = np.r_[ind_scores, ood_scores]
    return {
        "auroc": float(roc_auc_score(y, s)),
        "low_fpr_tpr": low_fpr_tpr_score(y, s),
    }

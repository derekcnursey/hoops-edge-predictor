"""Simple post-hoc cover-probability calibrators for fixed model scores."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def clip_probabilities(probabilities: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Keep probabilities inside open interval bounds for scoring stability."""
    probs = np.asarray(probabilities, dtype=float)
    return np.clip(probs, eps, 1.0 - eps)


def normal_cdf_from_z(z_score: np.ndarray) -> np.ndarray:
    """Convert a z-score into a Normal-CDF probability."""
    z = np.asarray(z_score, dtype=float)
    erf_vec = np.vectorize(erf)
    return clip_probabilities(0.5 * (1.0 + erf_vec(z / sqrt(2.0))))


def fit_logistic_calibrator(z_score: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    """Fit a 1D logistic calibrator, falling back to a constant if needed."""
    z = np.asarray(z_score, dtype=float).reshape(-1, 1)
    y = np.asarray(outcome, dtype=int)
    if len(np.unique(y)) < 2:
        return {"method": "constant", "prob": float(np.mean(y))}

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(z, y)
    return {
        "method": "logistic",
        "intercept": float(model.intercept_[0]),
        "coef": float(model.coef_[0, 0]),
    }


def predict_logistic_calibrator(
    z_score: np.ndarray,
    *,
    intercept: float,
    coef: float,
) -> np.ndarray:
    """Apply a fitted logistic calibrator."""
    z = np.asarray(z_score, dtype=float)
    logits = float(intercept) + float(coef) * z
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return clip_probabilities(probabilities)


def fit_isotonic_calibrator(z_score: np.ndarray, outcome: np.ndarray) -> dict[str, object]:
    """Fit a monotonic isotonic calibrator, falling back to a constant if needed."""
    z = np.asarray(z_score, dtype=float)
    y = np.asarray(outcome, dtype=int)
    if len(np.unique(y)) < 2:
        return {"method": "constant", "prob": float(np.mean(y))}

    order = np.argsort(z)
    model = IsotonicRegression(
        y_min=1e-6,
        y_max=1.0 - 1e-6,
        out_of_bounds="clip",
        increasing=True,
    )
    model.fit(z[order], y[order])
    return {
        "method": "isotonic",
        "x_thresholds": [float(x) for x in model.X_thresholds_],
        "y_thresholds": [float(y_val) for y_val in model.y_thresholds_],
    }


def predict_isotonic_calibrator(
    z_score: np.ndarray,
    *,
    x_thresholds: list[float],
    y_thresholds: list[float],
) -> np.ndarray:
    """Apply a fitted isotonic calibrator via piecewise-linear interpolation."""
    z = np.asarray(z_score, dtype=float)
    x = np.asarray(x_thresholds, dtype=float)
    y = np.asarray(y_thresholds, dtype=float)
    probabilities = np.interp(z, x, y, left=y[0], right=y[-1])
    return clip_probabilities(probabilities)


def apply_probability_calibration(
    z_score: np.ndarray,
    calibration: dict[str, object],
) -> np.ndarray:
    """Apply a fitted or trivial cover-probability calibration."""
    method = str(calibration["method"])
    if method == "normal":
        return normal_cdf_from_z(z_score)
    if method == "constant":
        return clip_probabilities(np.full(len(np.asarray(z_score)), float(calibration["prob"])))
    if method == "logistic":
        return predict_logistic_calibrator(
            z_score,
            intercept=float(calibration["intercept"]),
            coef=float(calibration["coef"]),
        )
    if method == "isotonic":
        return predict_isotonic_calibrator(
            z_score,
            x_thresholds=list(calibration["x_thresholds"]),
            y_thresholds=list(calibration["y_thresholds"]),
        )
    raise ValueError(f"Unsupported calibration method: {method}")

"""Simple post-hoc sigma transformations for inference and offline study."""

from __future__ import annotations

import numpy as np


def apply_sigma_transform(
    sigma: np.ndarray,
    mode: str | None = None,
    *,
    cap_max: float | None = None,
    scale: float | None = None,
    affine_a: float | None = None,
    affine_b: float | None = None,
    shrink_alpha: float | None = None,
    shrink_target: float | None = None,
    sigma_min: float = 0.5,
    sigma_max: float = 30.0,
) -> np.ndarray:
    """Apply a simple monotonic sigma transform and clamp to valid bounds."""
    out = np.asarray(sigma, dtype=float).copy()
    out = np.clip(out, sigma_min, sigma_max)

    if mode in (None, "", "raw"):
        return out
    if mode == "cap":
        if cap_max is None:
            raise ValueError("cap mode requires cap_max")
        out = np.minimum(out, float(cap_max))
    elif mode == "scale":
        if scale is None:
            raise ValueError("scale mode requires scale")
        out = float(scale) * out
    elif mode == "affine":
        if affine_a is None or affine_b is None:
            raise ValueError("affine mode requires affine_a and affine_b")
        out = float(affine_a) + float(affine_b) * out
    elif mode == "shrink":
        if shrink_alpha is None or shrink_target is None:
            raise ValueError("shrink mode requires shrink_alpha and shrink_target")
        alpha = float(shrink_alpha)
        target = float(shrink_target)
        out = alpha * out + (1.0 - alpha) * target
    else:
        raise ValueError(f"Unsupported sigma calibration mode: {mode}")

    return np.clip(out, sigma_min, sigma_max)

"""Utilities for model-implied moneyline probabilities and fair odds."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

MlSigmaMode = Literal["raw", "cap14", "cap17", "const14"]
MlOddsMode = Literal["cap14_mu_sigma", "meta_small_v1"]

META_SMALL_V1 = {
    "intercept": 0.020175630994879585,
    "coefficients": {
        "mu": 0.15059080225978677,
        "sigma_cap14": -0.008192640820978753,
        "z14": 0.08477442454897068,
        "post_dec15": 0.0740213341463982,
        "abs_mu": 0.004037300255673246,
    },
}

NEUTRAL_BETA_BLEND_V1 = {
    "alpha": 0.6,
    "intercept": -0.3232338741070838,
    "coefficients": {
        "log_p": 1.0790706225223776,
        "log1m_p": -1.6575437915942683,
    },
}


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF using erf."""
    if np.isscalar(x):
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
    x_arr = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x_arr / math.sqrt(2.0)))


def stabilize_sigma_for_ml(
    sigma: np.ndarray | float,
    mode: MlSigmaMode = "cap14",
) -> np.ndarray | float:
    """Apply the chosen sigma stabilization for ML odds."""
    sigma_arr = np.asarray(sigma, dtype=float)
    if mode == "raw":
        out = sigma_arr
    elif mode == "cap14":
        out = np.minimum(sigma_arr, 14.0)
    elif mode == "cap17":
        out = np.minimum(sigma_arr, 17.0)
    elif mode == "const14":
        out = np.full_like(sigma_arr, 14.0, dtype=float)
    else:
        raise ValueError(f"Unsupported ML sigma mode: {mode}")
    out = np.maximum(out, 0.5)
    if np.isscalar(sigma):
        return float(out)
    return out


def mu_sigma_home_win_prob(
    mu_home: np.ndarray | float,
    sigma: np.ndarray | float,
    *,
    sigma_mode: MlSigmaMode = "cap14",
) -> np.ndarray | float:
    """Convert margin mean/sigma to home win probability."""
    sigma_used = stabilize_sigma_for_ml(sigma, mode=sigma_mode)
    z = np.asarray(mu_home, dtype=float) / np.asarray(sigma_used, dtype=float)
    p = normal_cdf(z)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if np.isscalar(mu_home) and np.isscalar(sigma):
        return float(p)
    return p


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic transform."""
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr, dtype=float)
    pos = x_arr >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def _neutral_site_beta_blend(
    prob: np.ndarray | float,
    neutral_site: np.ndarray | float | bool | None,
) -> np.ndarray | float:
    """Blend the current site probability toward a neutral-only beta calibrator."""
    if neutral_site is None:
        return prob

    prob_arr = np.asarray(prob, dtype=float)
    neutral_arr = np.asarray(neutral_site)
    if neutral_arr.shape == ():
        neutral_mask = np.full_like(prob_arr, bool(neutral_arr), dtype=bool)
    else:
        neutral_mask = neutral_arr.astype(bool)
    if not np.any(neutral_mask):
        return float(prob_arr) if np.isscalar(prob) else prob_arr

    safe_prob = np.clip(prob_arr, 1e-6, 1.0 - 1e-6)
    beta_score = (
        NEUTRAL_BETA_BLEND_V1["intercept"]
        + NEUTRAL_BETA_BLEND_V1["coefficients"]["log_p"] * np.log(safe_prob)
        + NEUTRAL_BETA_BLEND_V1["coefficients"]["log1m_p"] * np.log1p(-safe_prob)
    )
    beta_prob = np.clip(logistic(beta_score), 1e-6, 1.0 - 1e-6)
    blended = (
        (1.0 - NEUTRAL_BETA_BLEND_V1["alpha"]) * safe_prob
        + NEUTRAL_BETA_BLEND_V1["alpha"] * beta_prob
    )
    out = np.where(neutral_mask, blended, safe_prob)
    if np.isscalar(prob):
        return float(out)
    return out


def site_home_win_prob_from_mu_sigma(
    mu_home: np.ndarray | float,
    sigma: np.ndarray | float,
    *,
    start_month: int,
    start_day: int,
    neutral_site: np.ndarray | float | bool | None = None,
    odds_mode: MlOddsMode = "meta_small_v1",
) -> np.ndarray | float:
    """Match the site-facing ML probability path.

    The current site starts from the cap14 mu+sigma baseline and optionally
    applies the small logistic correction layer (``meta_small_v1``).
    """
    baseline = mu_sigma_home_win_prob(mu_home, sigma, sigma_mode="cap14")
    if odds_mode == "cap14_mu_sigma":
        return baseline
    if odds_mode != "meta_small_v1":
        raise ValueError(f"Unsupported ML odds mode: {odds_mode}")

    sigma_cap14 = stabilize_sigma_for_ml(sigma, mode="cap14")
    mu_arr = np.asarray(mu_home, dtype=float)
    sigma_arr = np.asarray(sigma_cap14, dtype=float)
    z14 = mu_arr / sigma_arr
    post_dec15 = 1.0 if (start_month > 12 or start_month < 11 or (start_month == 12 and start_day >= 15) or start_month in (1, 2, 3)) else 0.0
    score = (
        META_SMALL_V1["intercept"]
        + META_SMALL_V1["coefficients"]["mu"] * mu_arr
        + META_SMALL_V1["coefficients"]["sigma_cap14"] * sigma_arr
        + META_SMALL_V1["coefficients"]["z14"] * z14
        + META_SMALL_V1["coefficients"]["post_dec15"] * post_dec15
        + META_SMALL_V1["coefficients"]["abs_mu"] * np.abs(mu_arr)
    )
    prob = np.clip(logistic(score), 1e-6, 1 - 1e-6)
    prob = _neutral_site_beta_blend(prob, neutral_site)
    if np.isscalar(mu_home) and np.isscalar(sigma):
        return float(prob)
    return prob


def fair_american_odds(prob: float | np.ndarray) -> float | np.ndarray:
    """Convert probability to fair American odds."""
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1 - 1e-9)
    out = np.full_like(p, np.nan, dtype=float)
    fav = p >= 0.5
    dog = ~fav
    out[fav] = -100.0 * (p[fav] / (1.0 - p[fav]))
    out[dog] = 100.0 * ((1.0 - p[dog]) / p[dog])
    if np.isscalar(prob):
        return float(out)
    return out

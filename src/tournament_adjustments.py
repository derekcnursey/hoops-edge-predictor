"""Tournament-only post-processing and blend helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .efficiency_blend import gold_weight_for_start_dates


def _series_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def market_blended_display_margin(
    raw_margin: float,
    market_margin: float | None,
    market_weight: float | None = None,
) -> float:
    """Blend a raw model margin toward market when a market margin is available."""
    if market_margin is None or pd.isna(market_margin):
        return float(raw_margin)
    weight = float(
        np.clip(
            config.NCAA_TOURNAMENT_MARKET_WEIGHT if market_weight is None else market_weight,
            0.0,
            1.0,
        )
    )
    return float((1.0 - weight) * raw_margin + weight * float(market_margin))


def is_ncaa_tournament(frame: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for NCAA Tournament rows."""
    if frame.empty:
        return pd.Series(dtype=bool)

    tournament = _series_or_default(frame, "tournament", pd.NA).astype("string")
    if "startDate" in frame.columns:
        dates = pd.to_datetime(frame["startDate"], errors="coerce", utc=True)
        months = dates.dt.tz_convert("America/New_York").dt.month
    else:
        months = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    neutral = _series_or_default(
        frame,
        "neutralSite" if "neutralSite" in frame.columns else "neutral_site",
        False,
    ).fillna(False).astype(bool)
    game_type = _series_or_default(frame, "gameType", "").astype("string")
    conference_game = _series_or_default(frame, "conferenceGame", False).fillna(False).astype(bool)

    explicit = tournament.eq("NCAA")
    heuristic = (
        tournament.isna()
        & game_type.eq("TRNMNT")
        & neutral
        & ~conference_game
        & months.isin([3, 4])
    )
    return explicit.fillna(False) | heuristic.fillna(False)


def tournament_primary_weights(frame: pd.DataFrame) -> np.ndarray:
    """Return primary-model blend weights after optional NCAA override."""
    if frame.empty:
        return np.array([], dtype=np.float32)

    weights = gold_weight_for_start_dates(
        frame["startDate"],
        start_day=config.PRODUCTION_MU_BLEND_START_DAY,
        end_day=config.PRODUCTION_MU_BLEND_END_DAY,
    ).astype(np.float32, copy=True)

    if not config.NCAA_TOURNAMENT_TORVIK_OVERRIDE_ENABLED:
        return weights

    mask = is_ncaa_tournament(frame).to_numpy(dtype=bool)
    if mask.any():
        override = np.float32(np.clip(config.NCAA_TOURNAMENT_PRIMARY_WEIGHT, 0.0, 1.0))
        weights[mask] = np.minimum(weights[mask], override)
    return weights


def needs_secondary_mu_features(frame: pd.DataFrame) -> bool:
    """Whether secondary Torvik-side features are needed for the current slate."""
    if frame.empty:
        return False
    weights = tournament_primary_weights(frame)
    return bool(np.nanmin(weights) < 1.0)


def add_tournament_market_display_columns(preds: pd.DataFrame) -> pd.DataFrame:
    """Attach display-only NCAA Tournament market-blend columns."""
    out = preds.copy()
    if out.empty:
        return out

    if "predicted_spread" not in out.columns:
        return out

    out["display_predicted_spread"] = out["predicted_spread"]
    out["display_model_spread"] = -out["display_predicted_spread"]
    if "book_spread" in out.columns:
        out["display_edge_home_points"] = out["display_predicted_spread"] + pd.to_numeric(
            out["book_spread"], errors="coerce"
        )
        out["display_spread_diff"] = out["display_model_spread"] - pd.to_numeric(
            out["book_spread"], errors="coerce"
        )

    if (
        not config.NCAA_TOURNAMENT_MARKET_BLEND_ENABLED
        or "book_spread" not in out.columns
    ):
        return out

    mask = is_ncaa_tournament(out).to_numpy(dtype=bool)
    book = pd.to_numeric(out["book_spread"], errors="coerce")
    market_margin = -book
    market_weight = float(np.clip(config.NCAA_TOURNAMENT_MARKET_WEIGHT, 0.0, 1.0))
    valid = mask & market_margin.notna().to_numpy(dtype=bool)
    if valid.any():
        out.loc[valid, "display_predicted_spread"] = (
            pd.to_numeric(out.loc[valid, "predicted_spread"], errors="coerce").combine(
                market_margin.loc[valid],
                lambda raw_margin, market_margin_value: market_blended_display_margin(
                    float(raw_margin),
                    float(market_margin_value),
                    market_weight,
                ),
            )
        )
        out.loc[valid, "display_model_spread"] = -out.loc[valid, "display_predicted_spread"]
        out.loc[valid, "display_edge_home_points"] = (
            out.loc[valid, "display_predicted_spread"] + book.loc[valid]
        )
        out.loc[valid, "display_spread_diff"] = (
            out.loc[valid, "display_model_spread"] - book.loc[valid]
        )

    return out

"""Seasonal blending between Torvik and internal efficiency-backed mu models."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _season_anchor_day(d: date) -> date:
    season_start_year = d.year if d.month >= 11 else d.year - 1
    return date(season_start_year, 11, 1)


def season_day_from_date(value: pd.Timestamp | str | date) -> int:
    if isinstance(value, str):
        ts = pd.Timestamp(value)
    elif isinstance(value, date) and not isinstance(value, pd.Timestamp):
        ts = pd.Timestamp(value)
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    d = ts.date()
    return (d - _season_anchor_day(d)).days


def gold_weight_for_season_day(season_day: int, start_day: int = 0, end_day: int = 75) -> float:
    if season_day <= start_day:
        return 0.0
    if season_day >= end_day:
        return 1.0
    return float((season_day - start_day) / (end_day - start_day))


def gold_weight_for_start_dates(start_dates: pd.Series, start_day: int = 0, end_day: int = 75) -> np.ndarray:
    ts = pd.to_datetime(start_dates, errors="coerce", utc=True)
    local = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    days = local.map(lambda x: season_day_from_date(x) if pd.notna(x) else np.nan).to_numpy(dtype=float)
    weights = np.clip((days - start_day) / (end_day - start_day), 0.0, 1.0)
    weights[np.isnan(days)] = 1.0
    return weights.astype(np.float32)


def blend_enabled() -> bool:
    return True

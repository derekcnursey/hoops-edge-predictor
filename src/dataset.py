"""PyTorch Dataset for hoops-edge-predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config


class HoopsDataset(Dataset):
    """PyTorch Dataset that yields (features, spread, home_win) tuples.

    Features are the N-element vector in FEATURE_ORDER (currently 53 features).
    """

    def __init__(
        self,
        features: np.ndarray,
        spread: np.ndarray | None = None,
        home_win: np.ndarray | None = None,
    ):
        """
        Args:
            features: (N, D) float array of pre-scaled features.
            spread: (N,) float array of home spread (home_pts - away_pts).
            home_win: (N,) float array of 0/1 labels.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.spread = (
            torch.tensor(spread, dtype=torch.float32) if spread is not None else None
        )
        self.home_win = (
            torch.tensor(home_win, dtype=torch.float32) if home_win is not None else None
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        if self.spread is not None and self.home_win is not None:
            return x, self.spread[idx], self.home_win[idx]
        return x


def load_season_features(
    season: int,
    no_garbage: bool = False,
    adj_suffix: str | None = None,
) -> pd.DataFrame:
    """Load pre-built features from local parquet file.

    Args:
        season: Season year.
        no_garbage: Use no-garbage-time variant.
        adj_suffix: Optional adjustment suffix (e.g. "adj_a0.85_p10").
    """
    suffix = "_no_garbage" if no_garbage else ""
    if adj_suffix:
        suffix += f"_{adj_suffix}"
    path = config.FEATURES_DIR / f"season_{season}{suffix}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    return pd.read_parquet(path)


def load_multi_season_features(
    seasons: list[int],
    no_garbage: bool = False,
    adj_suffix: str | None = None,
    min_month_day: str | None = None,
) -> pd.DataFrame:
    """Load and concatenate features for multiple seasons.

    Args:
        seasons: List of season years to load.
        no_garbage: Use no-garbage-time variant.
        adj_suffix: Optional adjustment suffix (e.g. "adj_a0.85_p10").
        min_month_day: If set (e.g. "12-20"), exclude games before this date
            within each season. For season S, the cutoff is (S-1)-MM-DD.
            This filters out early-season noise from training data.
    """
    dfs = []
    for s in seasons:
        try:
            dfs.append(load_season_features(s, no_garbage=no_garbage,
                                            adj_suffix=adj_suffix))
        except FileNotFoundError:
            print(f"Warning: No features for season {s}, skipping.")
    if not dfs:
        raise FileNotFoundError(f"No feature files found for seasons {seasons}")
    df = pd.concat(dfs, ignore_index=True)

    if min_month_day is not None:
        before = len(df)
        df = _filter_by_min_date(df, min_month_day)
        print(f"  Date filter ({min_month_day}): {before} → {len(df)} rows "
              f"({before - len(df)} dropped)")

    return df


def _filter_by_min_date(df: pd.DataFrame, min_month_day: str) -> pd.DataFrame:
    """Filter out games before a per-season cutoff date.

    Season S spans fall (S-1) through spring S:
      - Fall: (S-1)-Aug through (S-1)-Dec
      - Spring: S-Jan through S-Jul

    If cutoff month is 8-12 (fall), cutoff = (S-1)-MM-DD.
    If cutoff month is 1-7 (spring), cutoff = S-MM-DD.

    E.g. for season 2025 with min_month_day="12-20": cutoff = 2024-12-20.
    For season 2025 with min_month_day="01-15": cutoff = 2025-01-15.
    """
    dates = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    game_dates = dates.dt.tz_localize(None).dt.normalize()

    month, day = (int(x) for x in min_month_day.split("-"))

    # Determine season year for each game
    game_years = game_dates.dt.year
    game_months = game_dates.dt.month
    # Season convention: games Aug-Dec belong to season=year+1, Jan-Jul to season=year
    season_year = game_years.where(game_months <= 7, game_years + 1)

    # Cutoff year depends on whether cutoff is in fall or spring of the season
    if month >= 8:
        cutoff_year = season_year - 1  # fall portion
    else:
        cutoff_year = season_year  # spring portion

    cutoffs = pd.to_datetime(
        cutoff_year.astype(int).astype(str) + f"-{month:02d}-{day:02d}",
        errors="coerce",
    )
    mask = game_dates >= cutoffs
    return df[mask].reset_index(drop=True)

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

    Features are the 37-element vector in FEATURE_ORDER.
    """

    def __init__(
        self,
        features: np.ndarray,
        spread: np.ndarray | None = None,
        home_win: np.ndarray | None = None,
    ):
        """
        Args:
            features: (N, 37) float array of pre-scaled features.
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


def load_season_features(season: int, no_garbage: bool = False) -> pd.DataFrame:
    """Load pre-built features from local parquet file."""
    suffix = "_no_garbage" if no_garbage else ""
    path = config.FEATURES_DIR / f"season_{season}{suffix}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    return pd.read_parquet(path)


def load_multi_season_features(seasons: list[int], no_garbage: bool = False) -> pd.DataFrame:
    """Load and concatenate features for multiple seasons."""
    dfs = []
    for s in seasons:
        try:
            dfs.append(load_season_features(s, no_garbage=no_garbage))
        except FileNotFoundError:
            print(f"Warning: No features for season {s}, skipping.")
    if not dfs:
        raise FileNotFoundError(f"No feature files found for seasons {seasons}")
    return pd.concat(dfs, ignore_index=True)

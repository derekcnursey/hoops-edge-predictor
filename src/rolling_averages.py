"""Compute 15-game exponentially-decayed rolling averages per team.

Uses pandas ewm(span=15) so that more recent games get higher weight,
matching the original BartTorvik pipeline's "decreasing" weighting scheme.

IMPORTANT: For each game, the rolling average is computed using stats
from games BEFORE that date (to avoid data leakage).
"""

from __future__ import annotations

import pandas as pd

from .config import EWM_SPAN
from .four_factors import FOUR_FACTOR_COLS


def compute_rolling_averages(four_factors: pd.DataFrame) -> pd.DataFrame:
    """Compute exponentially-weighted rolling averages of four-factor stats per team.

    Args:
        four_factors: DataFrame from four_factors.compute_game_four_factors()
            with columns: gameid, teamid, startdate, ishometeam, + 13 four-factor cols.

    Returns:
        DataFrame with gameid, teamid, startdate, ishometeam, and 13 rolling_* columns.
        Each rolling value is the EWM average of that stat using all games
        PRIOR to the current game's date (shifted by 1 to prevent leakage).
    """
    df = four_factors.copy()

    # Parse dates for sorting
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.sort_values(["teamid", "_date", "gameid"]).reset_index(drop=True)

    rolling_cols = [f"rolling_{c}" for c in FOUR_FACTOR_COLS]

    # Compute per-team rolling averages
    results = []
    for _tid, group in df.groupby("teamid"):
        g = group.copy()
        for stat, rcol in zip(FOUR_FACTOR_COLS, rolling_cols):
            # shift(1) ensures we only use games BEFORE the current one
            g[rcol] = (
                g[stat]
                .ewm(span=EWM_SPAN, min_periods=1)
                .mean()
                .shift(1)
            )
        results.append(g)

    out = pd.concat(results, ignore_index=True)
    keep = ["gameid", "teamid", "startdate", "ishometeam"] + rolling_cols
    return out[keep].copy()


# Mapping from model feature names to rolling column names.
# The model expects features like "away_eff_fg_pct" which maps to "rolling_eff_fg_pct"
# for the away team. This mapping is used by features.py to assemble the final vector.

# For the AWAY team (indices 11-23 in feature_order):
AWAY_ROLLING_MAP = {
    "away_eff_fg_pct": "rolling_eff_fg_pct",
    "away_ft_pct": "rolling_ft_pct",
    "away_ft_rate": "rolling_ft_rate",
    "away_3pt_rate": "rolling_three_pt_rate",
    "away_3p_pct": "rolling_three_p_pct",
    "away_off_rebound_pct": "rolling_off_rebound_pct",
    "away_def_rebound_pct": "rolling_def_rebound_pct",
    "away_def_eff_fg_pct": "rolling_def_eff_fg_pct",
    "away_def_ft_rate": "rolling_def_ft_rate",
    "away_def_3pt_rate": "rolling_def_3pt_rate",
    "away_def_3p_pct": "rolling_def_3p_pct",
    "away_def_off_rebound_pct": "rolling_def_off_rebound_pct",
    "away_def_def_rebound_pct": "rolling_def_def_rebound_pct",
}

# For the HOME team (indices 24-36 in feature_order):
HOME_ROLLING_MAP = {
    "home_eff_fg_pct": "rolling_eff_fg_pct",
    "home_ft_pct": "rolling_ft_pct",
    "home_ft_rate": "rolling_ft_rate",
    "home_3pt_rate": "rolling_three_pt_rate",
    "home_3p_pct": "rolling_three_p_pct",
    "home_off_rebound_pct": "rolling_off_rebound_pct",
    "home_def_rebound_pct": "rolling_def_rebound_pct",
    "home_def_eff_fg_pct": "rolling_def_eff_fg_pct",
    "home_opp_ft_rate": "rolling_def_ft_rate",  # NOTE: "opp_ft_rate" = defensive ft_rate
    "home_def_3pt_rate": "rolling_def_3pt_rate",
    "home_def_3p_pct": "rolling_def_3p_pct",
    "home_def_off_rebound_pct": "rolling_def_off_rebound_pct",
    "home_def_def_rebound_pct": "rolling_def_def_rebound_pct",
}

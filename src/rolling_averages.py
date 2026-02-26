"""Compute 15-game exponentially-decayed rolling averages per team.

Uses pandas ewm(span=15) so that more recent games get higher weight,
matching the original BartTorvik pipeline's "decreasing" weighting scheme.

IMPORTANT: For each game, the rolling average is computed using stats
from games BEFORE that date (to avoid data leakage).
"""

from __future__ import annotations

import numpy as np
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


def compute_form_delta(four_factors: pd.DataFrame) -> pd.DataFrame:
    """Compute form delta: difference between short-term and long-term EWM averages.

    For each team, computes EWM(span=5) - EWM(span=15) of the 13 four-factor stats,
    then averages the 13 deltas into a single summary 'form_delta' per team-game.

    Anti-leakage: Both EWMs use .shift(1) so current game is excluded.

    Args:
        four_factors: DataFrame from compute_game_four_factors() with columns:
            gameid, teamid, startdate, ishometeam, + 13 four-factor cols.

    Returns:
        DataFrame with gameid, teamid, form_delta.
    """
    df = four_factors.copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.sort_values(["teamid", "_date", "gameid"]).reset_index(drop=True)

    results = []
    for _tid, group in df.groupby("teamid"):
        g = group.copy()
        deltas = []
        for stat in FOUR_FACTOR_COLS:
            short = g[stat].ewm(span=5, min_periods=1).mean().shift(1)
            long = g[stat].ewm(span=EWM_SPAN, min_periods=1).mean().shift(1)
            deltas.append(short - long)
        # Average the 13 deltas into a single summary
        g["form_delta"] = np.mean(deltas, axis=0)
        results.append(g[["gameid", "teamid", "form_delta"]])

    return pd.concat(results, ignore_index=True)


def compute_rolling_turnovers(box: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling turnover rate averages from boxscore data.

    Uses team_tov_ratio (offensive) and opp_tov_ratio (defensive) from
    fct_pbp_game_teams_flat, applying the same EWM(span=15) + shift(1) pattern.

    Anti-leakage: .shift(1) excludes the current game.

    Args:
        box: DataFrame from fct_pbp_game_teams_flat with columns:
            gameid, teamid, startdate, team_tov_ratio, opp_tov_ratio.

    Returns:
        DataFrame with gameid, teamid, rolling_tov_rate, rolling_def_tov_rate.
    """
    needed = ["gameid", "teamid", "startdate", "team_tov_ratio", "opp_tov_ratio"]
    missing = [c for c in needed if c not in box.columns]
    if missing:
        return pd.DataFrame(columns=["gameid", "teamid", "rolling_tov_rate", "rolling_def_tov_rate"])

    df = box[needed].copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.sort_values(["teamid", "_date", "gameid"]).reset_index(drop=True)

    results = []
    for _tid, group in df.groupby("teamid"):
        g = group.copy()
        g["rolling_tov_rate"] = (
            g["team_tov_ratio"]
            .ewm(span=EWM_SPAN, min_periods=1)
            .mean()
            .shift(1)
        )
        g["rolling_def_tov_rate"] = (
            g["opp_tov_ratio"]
            .ewm(span=EWM_SPAN, min_periods=1)
            .mean()
            .shift(1)
        )
        results.append(g[["gameid", "teamid", "rolling_tov_rate", "rolling_def_tov_rate"]])

    return pd.concat(results, ignore_index=True)


# Turnover rate feature mappings (parallel to AWAY/HOME_ROLLING_MAP)
AWAY_TOV_MAP = {
    "away_tov_rate": "rolling_tov_rate",
    "away_def_tov_rate": "rolling_def_tov_rate",
}
HOME_TOV_MAP = {
    "home_tov_rate": "rolling_tov_rate",
    "home_def_tov_rate": "rolling_def_tov_rate",
}

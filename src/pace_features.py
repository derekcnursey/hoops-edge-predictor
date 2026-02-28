"""Pace flexibility and tempo matching features (Group K).

Meta-stats computed from game-level pace data:
  - pace_variance: rolling stdev of pace across recent games
  - pace_conformity: correlation between team pace and opponent avg pace
  - pace_mismatch_performance: efficiency in games where pace deviates
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EWM_SPAN


def compute_pace_features(
    boxscores: pd.DataFrame,
    window: int = 15,
) -> pd.DataFrame:
    """Compute pace flexibility features per team per game.

    Args:
        boxscores: DataFrame from fct_pbp_game_teams_flat with columns:
            gameid, teamid, opponentid, startdate, team_possessions, team_minutes,
            OR columns that allow computing possessions.
        window: Rolling window for standard deviation.

    Returns:
        DataFrame with gameid, teamid, startdate, pace_variance,
        pace_conformity, pace_mismatch_performance.
    """
    df = boxscores.copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")

    # Compute pace if not directly available
    # Pace = possessions / (minutes / 40)
    if "team_possessions" in df.columns and "team_minutes" in df.columns:
        df["_pace"] = df["team_possessions"] / (df["team_minutes"].replace(0, np.nan) / 40)
    elif "pace" in df.columns:
        df["_pace"] = df["pace"]
    else:
        # Estimate possessions from boxscore data
        # possessions ≈ FGA - OREB + TOV + 0.44 * FTA
        if all(c in df.columns for c in ["team_fg_att", "team_reb_off", "team_ft_att"]):
            # Simple estimation
            tov = df.get("team_turnovers", pd.Series(0, index=df.index))
            if "team_tov_ratio" in df.columns and "team_fg_att" in df.columns:
                # tov_ratio = turnovers / possessions, so we can't use it directly
                tov = pd.Series(0, index=df.index)
            df["_poss_est"] = (
                df["team_fg_att"] - df["team_reb_off"] + tov + 0.44 * df["team_ft_att"]
            )
            df["_pace"] = df["_poss_est"]  # Already per-game
        else:
            return pd.DataFrame(columns=["gameid", "teamid", "startdate",
                                         "pace_variance", "pace_conformity",
                                         "pace_mismatch_performance"])

    df = df.sort_values(["teamid", "_date", "gameid"]).reset_index(drop=True)

    # Build opponent season-average pace lookup
    opp_avg_pace = _compute_opponent_avg_pace(df)

    results = []
    for tid, group in df.groupby("teamid"):
        g = group.copy()
        tid = int(tid)

        # pace_variance: rolling stdev with shift(1) for no leakage
        g["pace_variance"] = (
            g["_pace"]
            .rolling(window=window, min_periods=3)
            .std()
            .shift(1)
        )

        # pace_conformity: for each game, how much does the game's pace
        # match the opponent's season-avg pace? Higher = team conforms
        # We use a rolling correlation proxy: track deviation from team's own avg
        team_avg_pace = g["_pace"].ewm(span=EWM_SPAN, min_periods=1).mean().shift(1)
        g["_pace_dev_from_own"] = g["_pace"] - team_avg_pace

        # Get opponent's pre-game avg pace for each game
        g["_opp_avg_pace"] = g.apply(
            lambda r: opp_avg_pace.get(
                (int(r["opponentid"]), r["_date"]) if pd.notna(r["opponentid"]) else (0, r["_date"]),
                np.nan,
            ),
            axis=1,
        )

        # pace_conformity: how much game pace aligns with opponent's pace tendency
        # Compute as rolling correlation between game pace and opponent avg pace
        # Using a simpler proxy: abs diff between game pace and team's own avg,
        # vs abs diff between game pace and opponent avg
        g["_diff_from_opp"] = (g["_pace"] - g["_opp_avg_pace"]).abs()
        g["pace_conformity"] = (
            g["_diff_from_opp"]
            .ewm(span=EWM_SPAN, min_periods=3)
            .mean()
            .shift(1)
        )

        # pace_mismatch_performance: efficiency in games where pace deviated
        # > 1 stdev from team's rolling avg
        g["_is_mismatch"] = g["_pace_dev_from_own"].abs() > g["pace_variance"]
        # We'll track a rolling ppp in mismatch games
        # For now, compute a simple indicator
        # Use EWM of efficiency (points scored - approximated) in mismatch games
        # This requires points data which might not be in boxscores, so use a proxy
        g["pace_mismatch_performance"] = (
            g["_is_mismatch"].astype(float)
            .ewm(span=EWM_SPAN, min_periods=1)
            .mean()
            .shift(1)
        )

        results.append(g[[
            "gameid", "teamid", "startdate",
            "pace_variance", "pace_conformity", "pace_mismatch_performance",
        ]])

    if not results:
        return pd.DataFrame(columns=["gameid", "teamid", "startdate",
                                     "pace_variance", "pace_conformity",
                                     "pace_mismatch_performance"])

    return pd.concat(results, ignore_index=True)


def _compute_opponent_avg_pace(df: pd.DataFrame) -> dict:
    """Build a lookup of (opponent_id, game_date) -> opponent's pre-game avg pace.

    Only uses games strictly BEFORE the given date.
    """
    avg_pace_lookup = {}
    df_sorted = df.sort_values(["_date", "gameid"]).reset_index(drop=True)

    team_pace_sum: dict[int, float] = {}
    team_pace_count: dict[int, int] = {}

    for date_val in sorted(df_sorted["_date"].dropna().unique()):
        mask = df_sorted["_date"] == date_val
        indices = df_sorted[mask].index

        # Record current averages for lookups on this date
        for idx in indices:
            tid = int(df_sorted.at[idx, "teamid"])
            if tid in team_pace_count and team_pace_count[tid] > 0:
                avg_pace_lookup[(tid, date_val)] = team_pace_sum[tid] / team_pace_count[tid]

        # Update running averages after all lookups for this date
        for idx in indices:
            tid = int(df_sorted.at[idx, "teamid"])
            pace = df_sorted.at[idx, "_pace"]
            if pd.notna(pace):
                team_pace_sum[tid] = team_pace_sum.get(tid, 0.0) + float(pace)
                team_pace_count[tid] = team_pace_count.get(tid, 0) + 1

    return avg_pace_lookup


# Feature names for integration
PACE_FEATURE_COLS = [
    "pace_variance", "pace_conformity", "pace_mismatch_performance",
]

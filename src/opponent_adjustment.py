"""Universal opponent adjustment for per-game per-team stats.

Adjusts raw stats by opponent quality using the additive formula:

    adj_stat = raw_stat - opponent_season_avg_of_defensive_stat + league_avg

For offensive stats, adjusts against the opponent's defensive version.
For defensive stats, adjusts against the opponent's offensive version.

Anti-leakage: all season-to-date averages use only games PRIOR to each
game date. The first few games of the season will have minimal adjustment
(falls back to league average for unknown opponents).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# Mapping of offensive stats → their defensive counterparts (and vice versa).
# Format: {stat_name: counterpart_stat_name}
# Offensive stats are adjusted against opponent's defensive version.
# Defensive stats are adjusted against opponent's offensive version.
DEFAULT_STAT_PAIRS: dict[str, str] = {
    # Four factors: offense → defense
    "eff_fg_pct": "def_eff_fg_pct",
    "ft_rate": "def_ft_rate",
    "three_pt_rate": "def_3pt_rate",
    "three_p_pct": "def_3p_pct",
    "off_rebound_pct": "def_off_rebound_pct",
    "def_rebound_pct": "off_rebound_pct",  # defensive rebound vs opp offensive rebound
    # Four factors: defense → offense
    "def_eff_fg_pct": "eff_fg_pct",
    "def_ft_rate": "ft_rate",
    "def_3pt_rate": "three_pt_rate",
    "def_3p_pct": "three_p_pct",
    "def_off_rebound_pct": "off_rebound_pct",
    "def_def_rebound_pct": "def_rebound_pct",
}

# Stats that should NOT be opponent-adjusted (opponent-independent)
NO_ADJUST_STATS = {"ft_pct"}


def opponent_adjust(
    df: pd.DataFrame,
    stat_cols: list[str],
    stat_pairs: Optional[dict[str, str]] = None,
    no_adjust: Optional[set[str]] = None,
    team_col: str = "teamid",
    opponent_col: str = "opponentid",
    date_col: str = "startdate",
    game_col: str = "gameid",
) -> pd.DataFrame:
    """Opponent-adjust per-game per-team stats using the additive formula.

    For each stat in stat_cols:
        adj_stat = raw_stat - opp_season_avg(counterpart_stat) + league_avg(counterpart_stat)

    Where opp_season_avg and league_avg only use games BEFORE the current date.

    Args:
        df: DataFrame with one row per team per game, containing raw stat values.
            Must have columns: team_col, opponent_col, date_col, game_col, + stat_cols.
        stat_cols: List of stat column names to adjust.
        stat_pairs: Dict mapping each stat to its counterpart for opponent adjustment.
            If a stat is not in this dict and not in no_adjust, it is left unchanged.
            Defaults to DEFAULT_STAT_PAIRS.
        no_adjust: Set of stat names that should not be adjusted.
            Defaults to NO_ADJUST_STATS.
        team_col: Column name for team ID.
        opponent_col: Column name for opponent team ID.
        date_col: Column name for game date.
        game_col: Column name for game ID.

    Returns:
        DataFrame with same schema as input but with adjusted stat values.
        Stats in no_adjust or without a defined counterpart are unchanged.
    """
    if stat_pairs is None:
        stat_pairs = DEFAULT_STAT_PAIRS
    if no_adjust is None:
        no_adjust = NO_ADJUST_STATS

    result = df.copy()
    result["_date"] = pd.to_datetime(result[date_col], errors="coerce")
    result = result.sort_values(["_date", game_col, team_col]).reset_index(drop=True)

    n = len(result)
    if n == 0:
        return result.drop(columns=["_date"], errors="ignore")

    # All stat columns we need to track (both the stats being adjusted and their counterparts)
    all_tracked = set(stat_cols)
    for s in stat_cols:
        if s not in no_adjust and s in stat_pairs:
            all_tracked.add(stat_pairs[s])

    # Pre-allocate output arrays
    adjusted = {}
    for col in stat_cols:
        adjusted[col] = result[col].values.copy().astype(np.float64)

    # Running per-team sums and counts (for computing season-to-date averages)
    team_sum: dict[int, dict[str, float]] = {}
    team_count: dict[int, int] = {}

    # Running league-wide sums and count
    league_sum: dict[str, float] = {s: 0.0 for s in all_tracked}
    league_count = 0

    # Process date by date to ensure causal ordering
    dates = result["_date"].values
    unique_dates = sorted(result["_date"].dropna().unique())

    for date_val in unique_dates:
        mask = dates == date_val
        indices = np.where(mask)[0]

        # Compute league averages from all games BEFORE this date
        league_avg = {}
        for s in all_tracked:
            if league_count > 0:
                league_avg[s] = league_sum[s] / league_count
            else:
                league_avg[s] = None

        # Adjust each game on this date
        for i in indices:
            opp_id = result.iat[i, result.columns.get_loc(opponent_col)]
            if pd.isna(opp_id):
                continue
            opp_id = int(opp_id)

            opp_n = team_count.get(opp_id, 0)
            opp_sums = team_sum.get(opp_id, {})

            for stat in stat_cols:
                if stat in no_adjust:
                    continue

                counterpart = stat_pairs.get(stat)
                if counterpart is None:
                    continue

                raw_val = result.iat[i, result.columns.get_loc(stat)]
                if pd.isna(raw_val):
                    continue

                lg_avg = league_avg.get(counterpart)
                if lg_avg is None:
                    continue

                # Opponent's season-to-date average of the counterpart stat
                if opp_n > 0 and counterpart in opp_sums:
                    opp_avg = opp_sums[counterpart] / opp_n
                else:
                    opp_avg = lg_avg  # Fall back to league avg for unknown opponents

                # Additive adjustment
                adjusted[stat][i] = raw_val - opp_avg + lg_avg

        # After adjusting all games on this date, update running averages
        # using RAW values (not adjusted) to prevent feedback loops
        for i in indices:
            tid = result.iat[i, result.columns.get_loc(team_col)]
            if pd.isna(tid):
                continue
            tid = int(tid)

            if tid not in team_sum:
                team_sum[tid] = {s: 0.0 for s in all_tracked}
                team_count[tid] = 0

            has_valid = False
            for s in all_tracked:
                if s in result.columns:
                    val = result.iat[i, result.columns.get_loc(s)]
                    if pd.notna(val):
                        team_sum[tid][s] += float(val)
                        has_valid = True

            if has_valid:
                team_count[tid] += 1
                for s in all_tracked:
                    if s in result.columns:
                        val = result.iat[i, result.columns.get_loc(s)]
                        if pd.notna(val):
                            league_sum[s] += float(val)
                league_count += 1

    # Write adjusted values back
    for stat in stat_cols:
        result[stat] = adjusted[stat]

    result = result.drop(columns=["_date"], errors="ignore")
    return result


def build_stat_pairs(
    offensive_stats: list[str],
    defensive_stats: list[str],
) -> dict[str, str]:
    """Build a stat_pairs mapping from parallel lists of offensive/defensive stats.

    Each offensive stat is paired with the corresponding defensive stat (by index),
    and vice versa.

    Args:
        offensive_stats: List of offensive stat names.
        defensive_stats: List of defensive stat names (same length).

    Returns:
        Dict mapping each stat to its counterpart.
    """
    if len(offensive_stats) != len(defensive_stats):
        raise ValueError(
            f"offensive_stats ({len(offensive_stats)}) and defensive_stats "
            f"({len(defensive_stats)}) must have the same length"
        )
    pairs = {}
    for off, deff in zip(offensive_stats, defensive_stats):
        pairs[off] = deff
        pairs[deff] = off
    return pairs

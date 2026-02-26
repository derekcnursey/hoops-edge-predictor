"""Opponent-adjust per-game four-factor stats before rolling averages.

Adjusts each game's raw four-factor values by the opponent's season-to-date
averages, normalizing for opponent quality. This is analogous to how the
iterative ratings engine adjusts efficiency:

    adj_game_oe = game_oe * (league_avg / opp_adj_de) ^ alpha

For four-factors:
    adj_stat = raw_stat * (league_avg_counterpart / opp_avg_counterpart) ^ alpha

Uses Bayesian shrinkage for early-season stability when opponent sample
sizes are small.

Anti-leakage: all averages use only games PRIOR to the current date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .four_factors import FOUR_FACTOR_COLS

# Maps each stat to (opponent_counterpart_stat, invert_ratio).
#
# Standard (invert=False): factor = (league_avg / opp_avg) ^ alpha
#   Higher counterpart value = easier task → deflate raw stat.
#
# Inverted (invert=True): factor = (opp_avg / league_avg) ^ alpha
#   Higher counterpart value = harder task → boost raw stat.
#   Used when stat direction and counterpart direction are misaligned.
STAT_ADJUSTMENTS: dict[str, tuple[str, bool] | None] = {
    # OFFENSIVE stats → adjusted by opponent's defensive counterpart
    "eff_fg_pct": ("def_eff_fg_pct", False),
    "ft_pct": None,  # free throw accuracy is opponent-independent
    "ft_rate": ("def_ft_rate", False),
    "three_pt_rate": ("def_3pt_rate", False),
    "three_p_pct": ("def_3p_pct", False),
    "off_rebound_pct": ("def_off_rebound_pct", False),
    # Team DREB (higher=better) vs opp OREB (higher=harder) → inverted
    "def_rebound_pct": ("off_rebound_pct", True),
    # DEFENSIVE stats → adjusted by opponent's offensive counterpart
    "def_eff_fg_pct": ("eff_fg_pct", False),
    "def_ft_rate": ("ft_rate", False),
    "def_3pt_rate": ("three_pt_rate", False),
    "def_3p_pct": ("three_p_pct", False),
    "def_off_rebound_pct": ("off_rebound_pct", False),
    "def_def_rebound_pct": ("def_rebound_pct", False),
}


def adjust_four_factors(
    four_factors: pd.DataFrame,
    prior_weight: float = 5.0,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Opponent-adjust per-game four-factor stats.

    For each game, adjusts the 13 four-factor stats using the opponent's
    season-to-date averages of the counterpart stat, with Bayesian shrinkage
    toward the league average for early-season stability.

    Args:
        four_factors: DataFrame from compute_game_four_factors() with columns:
            gameid, teamid, opponentid, startdate, ishometeam, + 13 stats.
        prior_weight: Bayesian prior weight for shrinkage (higher = more
            conservative early season). Default 5.
        alpha: SOS exponent on the adjustment factor. 1.0 = full adjustment,
            <1.0 dampens the effect. Default 1.0.

    Returns:
        DataFrame with same schema as input but with adjusted stat values.
        ft_pct is unchanged (no defensive counterpart).
    """
    df = four_factors.copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.sort_values(["_date", "gameid", "teamid"]).reset_index(drop=True)

    n = len(df)
    if n == 0:
        return df

    # Pre-allocate output arrays
    adjusted = {}
    for stat in FOUR_FACTOR_COLS:
        adjusted[stat] = df[stat].values.copy().astype(np.float64)

    # Running per-team sums and counts
    team_sum: dict[int, dict[str, float]] = {}
    team_count: dict[int, int] = {}

    # Running league-wide sums and count
    league_sum: dict[str, float] = {s: 0.0 for s in FOUR_FACTOR_COLS}
    league_count = 0

    # Process date by date to ensure causal ordering
    dates = df["_date"].values
    unique_dates = sorted(df["_date"].dropna().unique())

    for date_val in unique_dates:
        mask = dates == date_val
        indices = np.where(mask)[0]

        # Compute league averages from all games BEFORE this date
        league_avg = {}
        for stat in FOUR_FACTOR_COLS:
            if league_count > 0:
                league_avg[stat] = league_sum[stat] / league_count
            else:
                league_avg[stat] = None  # no prior data yet

        # Adjust each game on this date
        for i in indices:
            oppid = int(df.iat[i, df.columns.get_loc("opponentid")])

            opp_n = team_count.get(oppid, 0)
            opp_sums = team_sum.get(oppid, {})

            for stat in FOUR_FACTOR_COLS:
                raw_val = df.iat[i, df.columns.get_loc(stat)]
                if pd.isna(raw_val):
                    continue

                pair = STAT_ADJUSTMENTS.get(stat)
                if pair is None:
                    # No adjustment (e.g., ft_pct)
                    continue

                counterpart, invert = pair
                lg_avg = league_avg.get(counterpart)

                if lg_avg is None or lg_avg == 0:
                    # No league data yet — no adjustment possible
                    continue

                # Opponent's season-to-date average of counterpart stat
                if opp_n > 0:
                    opp_raw_avg = opp_sums.get(counterpart, 0.0) / opp_n
                else:
                    opp_raw_avg = lg_avg

                # Bayesian shrinkage toward league average
                opp_avg = (
                    opp_n * opp_raw_avg + prior_weight * lg_avg
                ) / (opp_n + prior_weight)

                # Division-by-zero guard
                if abs(opp_avg) < 0.01:
                    continue  # factor = 1.0, no change

                # Compute adjustment factor
                if invert:
                    factor = (opp_avg / lg_avg) ** alpha
                else:
                    factor = (lg_avg / opp_avg) ** alpha

                adjusted[stat][i] = raw_val * factor

        # After adjusting all games on this date, update running averages
        for i in indices:
            tid = int(df.iat[i, df.columns.get_loc("teamid")])

            if tid not in team_sum:
                team_sum[tid] = {s: 0.0 for s in FOUR_FACTOR_COLS}
                team_count[tid] = 0

            has_valid = False
            for stat in FOUR_FACTOR_COLS:
                val = df.iat[i, df.columns.get_loc(stat)]
                if pd.notna(val):
                    # Use RAW values for running averages (not adjusted)
                    # This prevents adjustment drift / feedback loops
                    team_sum[tid][stat] += float(val)
                    has_valid = True

            if has_valid:
                team_count[tid] += 1

                for stat in FOUR_FACTOR_COLS:
                    val = df.iat[i, df.columns.get_loc(stat)]
                    if pd.notna(val):
                        league_sum[stat] += float(val)
                league_count += 1

    # Write adjusted values back
    result = df.copy()
    for stat in FOUR_FACTOR_COLS:
        result[stat] = adjusted[stat]

    # Drop temp column
    result = result.drop(columns=["_date"], errors="ignore")

    return result

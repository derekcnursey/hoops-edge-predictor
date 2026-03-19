"""Iterative opponent-adjustment solver for per-game four-factor stats.

Applies the same Gauss-Seidel iterative approach used for adjusted efficiencies
(in the ETL repo's build_pbp_team_daily_rollup_adj.py) to all 13 four-factor stats.

Key difference from the efficiency solver: four-factor stat pairs may be on
different scales (e.g. DREB% ≈ 0.68 vs OREB% ≈ 0.32), so residuals must be
centered around league averages to avoid scale bias. Additionally, some pairs
are "inverted" — higher counterpart means harder task, not easier.

Algorithm per date snapshot:
  1. Collect all games played before date D
  2. Estimate per-stat HCA from home/away splits
  3. Iterate (centered residuals):
     For each team, for each stat:
       centered_raw = raw_game_stat - league_avg_stat
       centered_opp = opp_adj_counterpart - league_avg_counterpart
       residual = centered_raw - direction * centered_opp - hca_correction
       team_adj_stat = league_avg + mean(residuals)
     Apply Bayesian shrinkage toward league average
  4. Output per-game adjusted values:
       Standard:  adj = raw + (league_avg_counterpart - opp_adj_counterpart)
       Inverted:  adj = raw + (opp_adj_counterpart - league_avg_counterpart)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .adjusted_four_factors import STAT_ADJUSTMENTS
from .four_factors import FOUR_FACTOR_COLS

# Stats that have a counterpart (i.e., should be adjusted)
_ADJUSTABLE_STATS = [s for s, v in STAT_ADJUSTMENTS.items() if v is not None]

# Build off/def pairing: stat -> (counterpart_stat, invert)
_STAT_PAIRING = {
    s: v for s, v in STAT_ADJUSTMENTS.items() if v is not None
}


def solve_four_factors(
    four_factors: pd.DataFrame,
    n_iterations: int = 25,
    prior_weight: float = 5.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Day-by-day iterative solver for opponent-adjusted four-factor stats.

    For each date in the season, runs the iterative solver on all games
    played before that date, then adjusts each game's raw stats using
    the converged opponent ratings.

    Args:
        four_factors: DataFrame from compute_game_four_factors() with columns:
            gameid, teamid, opponentid, startdate, ishometeam, + 13 stats.
        n_iterations: Number of Gauss-Seidel iterations per snapshot.
        prior_weight: Bayesian shrinkage weight (higher = more conservative).
        verbose: Print convergence diagnostics.

    Returns:
        DataFrame with same schema as input but with opponent-adjusted stat values.
        ft_pct is unchanged (no defensive counterpart).
    """
    df = four_factors.copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.sort_values(["_date", "gameid", "teamid"]).reset_index(drop=True)

    n = len(df)
    if n == 0:
        return df

    # Pre-extract arrays for fast access
    teamids = df["teamid"].values.astype(int)
    oppids = df["opponentid"].values.astype(int)
    is_home = df["ishometeam"].values
    dates = df["_date"].values
    unique_dates = sorted(df["_date"].dropna().unique())

    stat_names = FOUR_FACTOR_COLS
    raw_vals = {}
    for s in stat_names:
        raw_vals[s] = df[s].values.astype(np.float64).copy()

    # Output: adjusted per-game values
    adj_vals = {s: raw_vals[s].copy() for s in stat_names}

    # Accumulate games as we walk through dates
    prior_games: list[dict] = []

    for date_idx, date_val in enumerate(unique_dates):
        mask = dates == date_val
        today_indices = np.where(mask)[0]

        if len(prior_games) == 0:
            # No prior games — can't adjust, leave raw values
            for i in today_indices:
                game_data = _extract_game(i, teamids, oppids, is_home, raw_vals, stat_names)
                prior_games.append(game_data)
            continue

        # Run iterative solver on prior_games
        team_adj, league_avgs = _iterate_ff_ratings(
            prior_games, stat_names, n_iterations, prior_weight,
            verbose and date_idx < 3,
        )

        # Adjust today's games using converged ratings
        for i in today_indices:
            opp = int(oppids[i])
            for s in _ADJUSTABLE_STATS:
                raw_v = raw_vals[s][i]
                if np.isnan(raw_v):
                    continue

                counterpart, invert = _STAT_PAIRING[s]
                lg_avg_c = league_avgs.get(counterpart)
                if lg_avg_c is None:
                    continue

                opp_adj_c = team_adj.get((opp, counterpart), lg_avg_c)
                opp_gap = opp_adj_c - lg_avg_c

                # Standard: high counterpart = easier → deflate (subtract gap)
                # Inverted: high counterpart = harder → boost (add gap)
                if invert:
                    adj_vals[s][i] = raw_v + opp_gap
                else:
                    adj_vals[s][i] = raw_v - opp_gap

        # Add today's games to prior_games
        for i in today_indices:
            game_data = _extract_game(i, teamids, oppids, is_home, raw_vals, stat_names)
            prior_games.append(game_data)

    # Write adjusted values back
    result = df.copy()
    for s in stat_names:
        result[s] = adj_vals[s]
    result = result.drop(columns=["_date"], errors="ignore")
    return result


def _extract_game(i, teamids, oppids, is_home, raw_vals, stat_names):
    """Extract a single game row into a dict for the solver."""
    game_data = {
        "idx": i,
        "teamid": int(teamids[i]),
        "oppid": int(oppids[i]),
        "is_home": bool(is_home[i]) if not pd.isna(is_home[i]) else None,
    }
    for s in stat_names:
        v = raw_vals[s][i]
        game_data[s] = float(v) if not np.isnan(v) else np.nan
    return game_data


def _iterate_ff_ratings(
    games: list[dict],
    stat_names: list[str],
    n_iterations: int,
    prior_weight: float,
    verbose: bool,
) -> tuple[dict[tuple[int, str], float], dict[str, float]]:
    """Run iterative Gauss-Seidel solver for all 13 four-factor stats.

    Uses centered residuals to handle stat pairs on different scales:
      residual = (raw - lg_avg_stat) - direction * (opp_adj_counterpart - lg_avg_counterpart) - hca
    where direction = +1 (standard) or -1 (inverted).

    Returns:
        (team_adj, league_avgs) where:
          team_adj: dict mapping (teamid, stat_name) -> adjusted rating
          league_avgs: dict mapping stat_name -> league average
    """
    # Compute league averages
    league_sums: dict[str, float] = {s: 0.0 for s in stat_names}
    league_counts: dict[str, int] = {s: 0 for s in stat_names}
    for g in games:
        for s in stat_names:
            v = g[s]
            if not np.isnan(v):
                league_sums[s] += v
                league_counts[s] += 1

    league_avgs: dict[str, float] = {}
    for s in stat_names:
        if league_counts[s] > 0:
            league_avgs[s] = league_sums[s] / league_counts[s]
        else:
            league_avgs[s] = 0.0

    # Estimate HCA per stat
    hca: dict[str, float] = {}
    for s in _ADJUSTABLE_STATS:
        home_vals = [g[s] for g in games if g["is_home"] is True and not np.isnan(g[s])]
        away_vals = [g[s] for g in games if g["is_home"] is False and not np.isnan(g[s])]
        if home_vals and away_vals:
            hca[s] = (sum(home_vals) / len(home_vals) - sum(away_vals) / len(away_vals)) / 2.0
        else:
            hca[s] = 0.0

    # Collect unique teams and build per-team game lists
    team_games: dict[int, list[dict]] = {}
    for g in games:
        tid = g["teamid"]
        if tid not in team_games:
            team_games[tid] = []
        team_games[tid].append(g)

    teams = sorted(team_games.keys())

    # Initialize all team ratings to league average
    team_adj: dict[tuple[int, str], float] = {}
    for t in teams:
        for s in stat_names:
            team_adj[(t, s)] = league_avgs[s]

    # Iterate
    for iteration in range(n_iterations):
        max_delta = 0.0

        for t in teams:
            t_games = team_games[t]
            if not t_games:
                continue

            n_games = len(t_games)

            for s in _ADJUSTABLE_STATS:
                counterpart, invert = _STAT_PAIRING[s]
                lg_avg_s = league_avgs[s]
                lg_avg_c = league_avgs.get(counterpart, 0.0)
                residual_sum = 0.0
                residual_n = 0

                for g in t_games:
                    raw_v = g[s]
                    if np.isnan(raw_v):
                        continue

                    opp = g["oppid"]
                    opp_adj_c = team_adj.get((opp, counterpart), lg_avg_c)

                    # HCA correction
                    hca_val = hca.get(s, 0.0)
                    if g["is_home"] is True:
                        hca_corr = hca_val
                    elif g["is_home"] is False:
                        hca_corr = -hca_val
                    else:
                        hca_corr = 0.0

                    # Centered residual (handles different scales)
                    centered_raw = raw_v - lg_avg_s
                    centered_opp = opp_adj_c - lg_avg_c

                    # Standard: subtract centered_opp (high opp = easy = less credit)
                    # Inverted: add centered_opp (high opp = hard = more credit)
                    if invert:
                        residual = centered_raw + centered_opp - hca_corr
                    else:
                        residual = centered_raw - centered_opp - hca_corr

                    residual_sum += residual
                    residual_n += 1

                if residual_n > 0:
                    raw_rating = lg_avg_s + residual_sum / residual_n
                    # Bayesian shrinkage toward league average
                    shrunk = (
                        residual_n * raw_rating + prior_weight * lg_avg_s
                    ) / (residual_n + prior_weight)

                    old = team_adj[(t, s)]
                    team_adj[(t, s)] = shrunk
                    max_delta = max(max_delta, abs(shrunk - old))

        if verbose:
            print(f"  iter {iteration+1:>2}: max_delta = {max_delta:.6f}")

        if max_delta < 1e-6:
            if verbose:
                print(f"  Converged at iteration {iteration+1}")
            break

    return team_adj, league_avgs

"""Kill-shot / momentum-run detection and scoring-drought analysis.

Computes per-game per-team metrics capturing a team's ability to go on
scoring runs (offensive) and its vulnerability to opponent runs (defensive).

Selected metrics (based on EDA of 577 games from the 2025 season):
  - max_run_10poss: max scoring differential in any 10-possession window
  - def_max_run_10poss: max run ALLOWED in any 10-possession window
  - run_frequency: fraction of 15-poss windows with >= 8-pt differential
  - def_run_frequency: fraction of 15-poss windows where opponent had >= 8-pt diff
  - avg_run_magnitude: average size of consecutive scoring runs

Stability analysis (ICC, split-half, temporal):
  - def_run_frequency is the most stable metric (ICC=0.520, split-half r=0.387,
    early-to-late season r=0.374). Defensive run vulnerability is scheme-driven
    and persists across games.
  - Offensive metrics are noisier game-to-game but still show meaningful
    between-team variance (ICC 0.29--0.39).
  - All metrics benefit from rolling-average smoothing across prior games
    before use as predictive features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Exported column names ────────────────────────────────────────────

KILL_SHOT_COLS: list[str] = [
    "max_run_10poss",
    "def_max_run_10poss",
    "run_frequency",
    "def_run_frequency",
    "avg_run_magnitude",
]


# ── Public API ───────────────────────────────────────────────────────

def compute_kill_shot_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game per-team kill-shot / momentum-run metrics.

    Args:
        pbp: Enriched PBP DataFrame with columns:
            gameId, teamId, offense_team_id, defense_team_id,
            possession_id, scoringPlay, scoreValue, secondsRemaining,
            period, garbage_time, homeScore, awayScore.
            Optional: opponentId, gameStartDate, isHomeTeam.

    Returns:
        DataFrame with columns:
            gameid, teamid, opponentid, startdate,
            max_run_10poss, def_max_run_10poss,
            run_frequency, def_run_frequency,
            avg_run_magnitude.
        One row per team per game. Garbage-time plays are excluded.
    """
    # Filter garbage time
    df = pbp[~pbp["garbage_time"]].copy()
    if df.empty:
        return _empty_result()

    df["scoreValue"] = pd.to_numeric(df["scoreValue"], errors="coerce").fillna(0)

    # Build possession-level scoring timeline
    poss_data = _build_possession_timeline(df)
    if poss_data.empty:
        return _empty_result()

    # Extract metadata (opponentId, startdate) per game-team
    meta = _extract_metadata(df)

    # Compute metrics per game-team
    results: list[dict] = []
    for game_id, game_poss in poss_data.groupby("gameId"):
        teams = game_poss["offense_team"].dropna().unique()
        for team_id in teams:
            row = _compute_game_team_metrics(game_poss, int(team_id))
            row["gameid"] = int(game_id)
            row["teamid"] = int(team_id)

            # Attach metadata
            m = meta.get((int(game_id), int(team_id)), {})
            row["opponentid"] = m.get("opponentid")
            row["startdate"] = m.get("startdate")

            results.append(row)

    if not results:
        return _empty_result()

    out = pd.DataFrame(results)
    # Ensure column order
    id_cols = ["gameid", "teamid", "opponentid", "startdate"]
    return out[id_cols + KILL_SHOT_COLS].copy()


# ── Internal helpers ─────────────────────────────────────────────────

def _empty_result() -> pd.DataFrame:
    """Return an empty DataFrame with the expected schema."""
    cols = ["gameid", "teamid", "opponentid", "startdate"] + KILL_SHOT_COLS
    return pd.DataFrame(columns=cols)


def _build_possession_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-level data to one row per possession.

    Returns DataFrame with columns:
        gameId, possession_id, offense_team, scored, pts, period.
    """
    # Pre-compute scoring points to avoid lambda in agg
    scoring_mask = df["scoringPlay"] == True  # noqa: E712
    df = df.copy()
    df["_score_pts"] = np.where(scoring_mask, df["scoreValue"], 0.0)

    poss = (
        df.groupby(["gameId", "possession_id"])
        .agg(
            offense_team=("offense_team_id", "first"),
            scored=("scoringPlay", "any"),
            pts=("_score_pts", "sum"),
            period=("period", "first"),
        )
        .reset_index()
    )
    return poss.sort_values(["gameId", "possession_id"]).reset_index(drop=True)


def _extract_metadata(df: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """Build a lookup of (gameId, teamId) -> {opponentid, startdate}."""
    meta: dict[tuple[int, int], dict] = {}
    has_opp = "opponentId" in df.columns
    has_date = "gameStartDate" in df.columns

    for (gid, tid), grp in df.groupby(["gameId", "teamId"]):
        entry: dict = {}
        if has_opp:
            opp_vals = grp["opponentId"].dropna()
            entry["opponentid"] = int(opp_vals.iloc[0]) if len(opp_vals) > 0 else None
        else:
            entry["opponentid"] = None
        if has_date:
            entry["startdate"] = grp["gameStartDate"].iloc[0]
        else:
            entry["startdate"] = None
        meta[(int(gid), int(tid))] = entry

    return meta


def _compute_game_team_metrics(game_poss: pd.DataFrame, team_id: int) -> dict:
    """Compute all kill-shot metrics for one team in one game.

    Uses numpy arrays for performance.
    """
    offense_arr = game_poss["offense_team"].values
    pts_arr = game_poss["pts"].values.astype(np.float64)
    n = len(game_poss)

    pts_for = np.where(offense_arr == team_id, pts_arr, 0.0)
    pts_against = np.where(offense_arr != team_id, pts_arr, 0.0)

    row: dict = {}

    # ── 1. Sliding window max run (10 possessions) ──────────────────
    row["max_run_10poss"], row["def_max_run_10poss"] = _sliding_window_max(
        pts_for, pts_against, n, window=10
    )

    # ── 2. Run frequency (15-poss window, 8-pt threshold) ───────────
    row["run_frequency"], row["def_run_frequency"] = _run_frequency(
        pts_for, pts_against, n, window=15, threshold=8
    )

    # ── 3. Average run magnitude (streak-based) ─────────────────────
    row["avg_run_magnitude"] = _avg_streak_magnitude(
        offense_arr, pts_for, pts_against, team_id
    )

    return row


def _sliding_window_max(
    pts_for: np.ndarray,
    pts_against: np.ndarray,
    n: int,
    window: int,
) -> tuple[float, float]:
    """Max scoring differential in any *window*-possession window.

    Returns (max_offensive_run, max_defensive_run_allowed).
    """
    if n < window:
        return 0.0, 0.0

    cum_for = np.cumsum(pts_for)
    cum_against = np.cumsum(pts_against)

    # Vectorised window sums using cumulative sums
    end_for = cum_for[window - 1 :]
    start_for = np.concatenate([[0.0], cum_for[: n - window]])
    window_for = end_for - start_for

    end_against = cum_against[window - 1 :]
    start_against = np.concatenate([[0.0], cum_against[: n - window]])
    window_against = end_against - start_against

    diff = window_for - window_against  # positive = team outscored opponent

    max_run = float(np.max(diff)) if len(diff) > 0 else 0.0
    max_allowed = float(np.max(-diff)) if len(diff) > 0 else 0.0

    return max(max_run, 0.0), max(max_allowed, 0.0)


def _run_frequency(
    pts_for: np.ndarray,
    pts_against: np.ndarray,
    n: int,
    window: int,
    threshold: int,
) -> tuple[float, float]:
    """Fraction of *window*-possession windows with >= *threshold*-pt differential.

    Returns (offensive_frequency, defensive_frequency).
    """
    if n < window:
        return 0.0, 0.0

    cum_for = np.cumsum(pts_for)
    cum_against = np.cumsum(pts_against)

    end_for = cum_for[window - 1 :]
    start_for = np.concatenate([[0.0], cum_for[: n - window]])
    window_for = end_for - start_for

    end_against = cum_against[window - 1 :]
    start_against = np.concatenate([[0.0], cum_against[: n - window]])
    window_against = end_against - start_against

    diff = window_for - window_against
    total_windows = len(diff)

    off_freq = float(np.sum(diff >= threshold)) / total_windows
    def_freq = float(np.sum(diff <= -threshold)) / total_windows

    return off_freq, def_freq


def _avg_streak_magnitude(
    offense_arr: np.ndarray,
    pts_for: np.ndarray,
    pts_against: np.ndarray,
    team_id: int,
) -> float:
    """Average magnitude of consecutive scoring runs.

    A run is a sequence of consecutive team scoring events (where the team
    has possession and scores) not interrupted by opponent scoring. Empty
    possessions (no one scores) do NOT break a run -- only an opponent
    scoring possession ends the current run.

    Args:
        offense_arr: per-possession offense_team_id array.
        pts_for: per-possession points scored by team_id (0 on opponent poss).
        pts_against: per-possession points scored by opponent (0 on team poss).
        team_id: the team to compute for.

    Returns:
        Mean streak magnitude (points). Returns 0.0 if no streaks found.
    """
    n = len(offense_arr)
    team_scored = (offense_arr == team_id) & (pts_for > 0)
    opp_scored = (offense_arr != team_id) & (pts_against > 0)

    runs: list[float] = []
    current_run = 0.0

    for i in range(n):
        if team_scored[i]:
            current_run += pts_for[i]
        elif opp_scored[i]:
            # Opponent scored -- end the current run
            if current_run > 0:
                runs.append(current_run)
            current_run = 0.0
        # else: empty possession (no scoring by either side) -- does not break run

    # Capture trailing run
    if current_run > 0:
        runs.append(current_run)

    return float(np.mean(runs)) if runs else 0.0

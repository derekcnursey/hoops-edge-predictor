"""Assemble the 37-feature vector for each game.

Combines:
  - Group 1 (11 features): efficiency metrics from gold/team_adjusted_efficiencies + fct_games
  - Group 2 (26 features): rolling four-factor averages from fct_pbp_game_teams_flat
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import pandas as pd

from . import config, s3_reader
from .four_factors import compute_game_four_factors
from .rolling_averages import (
    AWAY_ROLLING_MAP,
    HOME_ROLLING_MAP,
    compute_rolling_averages,
)


def load_games(season: int) -> pd.DataFrame:
    """Load fct_games for a season, return DataFrame with key columns."""
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame()
    df = tbl.to_pandas()
    # Normalize column names — API may use different casing
    rename = {}
    for target, candidates in [
        ("gameId", ["gameId"]),
        ("homeTeamId", ["homeTeamId"]),
        ("awayTeamId", ["awayTeamId"]),
        ("homeTeam", ["homeTeam"]),
        ("awayTeam", ["awayTeam"]),
        ("homeScore", ["homeScore", "homePoints"]),
        ("awayScore", ["awayScore", "awayPoints"]),
        ("neutralSite", ["neutralSite", "neutralsite"]),
        ("startDate", ["startDate", "startTime", "date"]),
    ]:
        for cand in candidates:
            if cand in df.columns:
                rename[cand] = target
                break
    df = df.rename(columns=rename)
    if "gameId" in df.columns:
        df = df.drop_duplicates(subset=["gameId"], keep="last")
    return df


def load_efficiency_ratings(season: int, no_garbage: bool = True) -> pd.DataFrame:
    """Load team_adjusted_efficiencies from the gold layer for a season.

    Args:
        season: Season year.
        no_garbage: If True, read from team_adjusted_efficiencies_no_garbage.

    Returns a DataFrame with columns: teamId, rating_date, adj_oe, adj_de,
    adj_tempo, barthag, sorted by (teamId, rating_date) for as-of lookups.
    """
    table_name = "team_adjusted_efficiencies_no_garbage" if no_garbage else "team_adjusted_efficiencies"
    tbl = s3_reader.read_gold_table(table_name, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame()
    df = tbl.to_pandas()
    # Ensure expected columns exist
    needed = ["teamId", "rating_date", "adj_oe", "adj_de", "adj_tempo", "barthag"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Gold table team_adjusted_efficiencies missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
    df["rating_date"] = pd.to_datetime(df["rating_date"], errors="coerce")
    df = df.sort_values(["teamId", "rating_date"]).reset_index(drop=True)
    return df


def load_boxscores(season: int) -> pd.DataFrame:
    """Load fct_pbp_game_teams_flat for a season."""
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAME_TEAMS, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame()
    return tbl.to_pandas()


def load_lines(season: int) -> pd.DataFrame:
    """Load fct_lines for a season."""
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_LINES, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame()
    return tbl.to_pandas()


def _compute_barthag(adj_oe: float, adj_de: float) -> Optional[float]:
    """BARTHAG = adj_oe^11.5 / (adj_oe^11.5 + adj_de^11.5)."""
    if adj_oe is None or adj_de is None:
        return None
    exp = config.BARTHAG_EXPONENT
    oe_pow = adj_oe ** exp
    de_pow = adj_de ** exp
    denom = oe_pow + de_pow
    if denom == 0:
        return None
    return oe_pow / denom


def _build_efficiency_lookup(
    ratings: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Build a per-team lookup table from the gold efficiency ratings.

    Returns:
        Dict mapping teamId -> DataFrame sorted by rating_date with columns
        adj_oe, adj_de, adj_tempo, barthag.
    """
    lookup: dict[int, pd.DataFrame] = {}
    if ratings.empty:
        return lookup
    for tid, group in ratings.groupby("teamId"):
        lookup[int(tid)] = group[["rating_date", "adj_oe", "adj_de", "adj_tempo", "barthag"]].copy()
    return lookup


def _get_asof_rating(
    team_lookup: dict[int, pd.DataFrame],
    team_id: int,
    game_date: pd.Timestamp,
) -> dict:
    """Look up a team's efficiency ratings as of the day before game_date.

    Uses the most recent rating_date that is strictly before game_date.
    """
    team_df = team_lookup.get(team_id)
    if team_df is None or team_df.empty:
        return {}
    # Normalize game_date to tz-naive date-only for comparison
    if hasattr(game_date, 'tz') and game_date.tz is not None:
        game_date = game_date.tz_localize(None)
    cutoff = game_date.normalize() - timedelta(days=1)
    eligible = team_df[team_df["rating_date"] <= cutoff]
    if eligible.empty:
        return {}
    # Last row is most recent (already sorted by rating_date)
    row = eligible.iloc[-1]
    return {
        "adj_oe": row["adj_oe"],
        "adj_de": row["adj_de"],
        "adj_tempo": row["adj_tempo"],
        "barthag": row["barthag"],
    }


def build_features(
    season: int,
    game_date: Optional[str] = None,
    no_garbage: bool = True,
) -> pd.DataFrame:
    """Build the full 37-feature matrix for games in a season.

    Args:
        season: Season year (e.g. 2026).
        game_date: If provided, only build features for games on this date.
        no_garbage: If True (default), use no-garbage-time efficiency ratings.

    Returns:
        DataFrame with columns: gameId, homeTeamId, awayTeamId, startDate,
        homeScore, awayScore, + the 37 feature columns in FEATURE_ORDER.
    """
    # Load raw data
    games = load_games(season)
    if games.empty:
        return pd.DataFrame()

    eff_ratings = load_efficiency_ratings(season, no_garbage=no_garbage)
    boxscores = load_boxscores(season)

    if game_date is not None and "startDate" in games.columns:
        games["_date_str"] = (
            pd.to_datetime(games["startDate"], errors="coerce", utc=True)
            .dt.tz_convert("America/New_York")
            .dt.strftime("%Y-%m-%d")
        )
        games = games[games["_date_str"] == game_date].copy()
        games = games.drop(columns=["_date_str"])
        if games.empty:
            return pd.DataFrame()

    # ── Group 2: Rolling four-factor averages ──────────────────────
    rolling_df = pd.DataFrame()
    if not boxscores.empty:
        ff = compute_game_four_factors(boxscores)
        rolling_df = compute_rolling_averages(ff)

    # Build rolling lookups: (gameid, teamid) -> {rolling_col: value}
    rolling_lookup: dict[tuple[int, int], dict[str, float]] = {}
    if not rolling_df.empty:
        for _, row in rolling_df.iterrows():
            key = (int(row["gameid"]), int(row["teamid"]))
            rolling_lookup[key] = row.to_dict()

    # Build date-aware efficiency lookup: teamId -> DataFrame of dated ratings
    eff_lookup = _build_efficiency_lookup(eff_ratings)

    # Parse game dates once
    games["_game_dt"] = pd.to_datetime(games["startDate"], errors="coerce")

    # ── Assemble features per game ─────────────────────────────────
    records = []
    for _, game in games.iterrows():
        gid = int(game["gameId"])
        home_tid = int(game["homeTeamId"])
        away_tid = int(game["awayTeamId"])
        game_dt = game["_game_dt"]

        neutral = bool(game.get("neutralSite", False))

        # Look up efficiency ratings as of the day before the game
        if pd.isna(game_dt):
            home_eff = {}
            away_eff = {}
        else:
            home_eff = _get_asof_rating(eff_lookup, home_tid, game_dt)
            away_eff = _get_asof_rating(eff_lookup, away_tid, game_dt)

        # Group 1: Efficiency features
        # Map: adj_oe → adj_oe, adj_de → adj_de, adj_tempo → adj_pace, barthag → BARTHAG
        feat = {
            "neutral_site": int(neutral),
            "away_team_adj_oe": away_eff.get("adj_oe"),
            "away_team_BARTHAG": away_eff.get("barthag"),
            "away_team_adj_de": away_eff.get("adj_de"),
            "away_team_adj_pace": away_eff.get("adj_tempo"),
            "home_team_adj_oe": home_eff.get("adj_oe"),
            "home_team_adj_de": home_eff.get("adj_de"),
            "home_team_adj_pace": home_eff.get("adj_tempo"),
            "home_team_BARTHAG": home_eff.get("barthag"),
            "home_team_home": int(not neutral),
            "away_team_home": 0,  # always False
        }

        # Group 2: Rolling four-factor averages (away team)
        away_rolling = rolling_lookup.get((gid, away_tid), {})
        for feat_name, rolling_col in AWAY_ROLLING_MAP.items():
            feat[feat_name] = away_rolling.get(rolling_col)

        # Group 2: Rolling four-factor averages (home team)
        home_rolling = rolling_lookup.get((gid, home_tid), {})
        for feat_name, rolling_col in HOME_ROLLING_MAP.items():
            feat[feat_name] = home_rolling.get(rolling_col)

        # Metadata (not part of the 37 features)
        feat["gameId"] = gid
        feat["homeTeamId"] = home_tid
        feat["awayTeamId"] = away_tid
        feat["homeTeam"] = game.get("homeTeam")
        feat["awayTeam"] = game.get("awayTeam")
        feat["startDate"] = game.get("startDate")
        feat["homeScore"] = game.get("homeScore")
        feat["awayScore"] = game.get("awayScore")

        records.append(feat)

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # Verify we have exactly the 37 features
    missing = [f for f in config.FEATURE_ORDER if f not in result.columns]
    if missing:
        for col in missing:
            result[col] = None

    return result


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract just the 37 feature columns in the correct order."""
    return df[config.FEATURE_ORDER].copy()


def get_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Extract training targets: spread_home and home_win."""
    out = pd.DataFrame()
    out["spread_home"] = df["homeScore"].astype(float) - df["awayScore"].astype(float)
    out["home_win"] = (out["spread_home"] > 0).astype(float)
    return out

"""Assemble the feature vector for each game.

Combines:
  - Group 1 (11 features): efficiency metrics from gold/team_adjusted_efficiencies + fct_games
  - Group 2 (26 features): rolling four-factor averages from fct_pbp_game_teams_flat
  - Extra feature groups (optional): rest_days, sos, conf_strength, form_delta, tov_rate, margin_std
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from . import config, s3_reader
from .adjusted_four_factors import adjust_four_factors
from .four_factors import compute_game_four_factors
from .rolling_averages import (
    AWAY_ROLLING_MAP,
    AWAY_TOV_MAP,
    HOME_ROLLING_MAP,
    HOME_TOV_MAP,
    compute_form_delta,
    compute_rolling_averages,
    compute_rolling_turnovers,
)

# All supported extra feature group names
EXTRA_FEATURE_GROUPS = {
    "rest_days",       # 3 features: home_rest_days, away_rest_days, rest_advantage
    "sos",             # 4 features: home_sos_oe, home_sos_de, away_sos_oe, away_sos_de
    "conf_strength",   # 2 features: home_conf_strength, away_conf_strength
    "form_delta",      # 2 features: home_form_delta, away_form_delta
    "tov_rate",        # 4 features: home_tov_rate, home_def_tov_rate, away_tov_rate, away_def_tov_rate
    "margin_std",      # 2 features: home_margin_std, away_margin_std
}


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
    include_sos: bool = False,
) -> dict[int, pd.DataFrame]:
    """Build a per-team lookup table from the gold efficiency ratings.

    Returns:
        Dict mapping teamId -> DataFrame sorted by rating_date with columns
        adj_oe, adj_de, adj_tempo, barthag (+ optionally sos_oe, sos_de).
    """
    lookup: dict[int, pd.DataFrame] = {}
    if ratings.empty:
        return lookup
    keep_cols = ["rating_date", "adj_oe", "adj_de", "adj_tempo", "barthag"]
    if include_sos:
        for col in ["sos_oe", "sos_de"]:
            if col in ratings.columns:
                keep_cols.append(col)
    for tid, group in ratings.groupby("teamId"):
        available = [c for c in keep_cols if c in group.columns]
        lookup[int(tid)] = group[available].copy()
    return lookup


def _get_asof_rating(
    team_lookup: dict[int, pd.DataFrame],
    team_id: int,
    game_date: pd.Timestamp,
    include_sos: bool = False,
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
    result = {
        "adj_oe": row["adj_oe"],
        "adj_de": row["adj_de"],
        "adj_tempo": row["adj_tempo"],
        "barthag": row["barthag"],
    }
    if include_sos:
        if "sos_oe" in row.index:
            result["sos_oe"] = row["sos_oe"]
        if "sos_de" in row.index:
            result["sos_de"] = row["sos_de"]
    return result


# ── Extra feature helper functions ───────────────────────────────


def _get_asof_rolling(
    team_lookup: dict[int, pd.DataFrame],
    team_id: int,
    game_date: pd.Timestamp,
    value_cols: list[str],
) -> dict[str, float]:
    """Look up a team's most recent rolling stats before game_date.

    Falls back to the team's latest entry before the game date, using the same
    pattern as _get_asof_rating() for efficiency ratings.
    """
    team_df = team_lookup.get(team_id)
    if team_df is None or team_df.empty:
        return {}
    # Normalize both sides to tz-naive for comparison
    if hasattr(game_date, 'tz') and game_date.tz is not None:
        game_date = game_date.tz_localize(None)
    cutoff = game_date.normalize()
    dates = team_df["_date"]
    if hasattr(dates.dtype, 'tz') and dates.dtype.tz is not None:
        dates = dates.dt.tz_localize(None)
    eligible = team_df[dates < cutoff]
    if eligible.empty:
        # Fall back to the very first entry if no prior data
        eligible = team_df
    row = eligible.iloc[-1]
    return {col: row[col] for col in value_cols if col in row.index and pd.notna(row[col])}


def _compute_rest_days(games: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Compute days since previous game for each team in each game.

    Anti-leakage: Rest days are computed from schedule dates, not results.

    Args:
        games: DataFrame with gameId, homeTeamId, awayTeamId, startDate.

    Returns:
        Dict mapping (gameId, teamId) -> rest_days (float).
        First game of season -> NaN -> filled with 5.0 (typical preseason rest).
    """
    dates = pd.to_datetime(games["startDate"], errors="coerce")

    # Expand to per-team view
    rows = []
    for _, g in games.iterrows():
        dt = dates[g.name]
        rows.append({"gameId": int(g["gameId"]), "teamId": int(g["homeTeamId"]), "date": dt})
        rows.append({"gameId": int(g["gameId"]), "teamId": int(g["awayTeamId"]), "date": dt})
    team_games = pd.DataFrame(rows)
    team_games = team_games.sort_values(["teamId", "date", "gameId"]).reset_index(drop=True)

    # Compute days since previous game per team
    team_games["prev_date"] = team_games.groupby("teamId")["date"].shift(1)
    team_games["rest_days"] = (team_games["date"] - team_games["prev_date"]).dt.total_seconds() / 86400
    team_games["rest_days"] = team_games["rest_days"].fillna(5.0)  # first game of season
    # Cap at 30 to avoid outliers from long breaks
    team_games["rest_days"] = team_games["rest_days"].clip(upper=30.0)

    return {
        (int(r["gameId"]), int(r["teamId"])): float(r["rest_days"])
        for _, r in team_games.iterrows()
    }


def _build_conf_strength_lookup(
    ratings: pd.DataFrame,
    dates: list[pd.Timestamp],
) -> dict[tuple[str, str], float]:
    """Build conference strength lookup from gold layer ratings.

    For each unique game date, computes mean adj_net per conference using
    ratings from the day before (same cutoff as efficiency lookups).

    Args:
        ratings: Gold efficiency ratings with teamId, rating_date, adj_oe, adj_de, conference.
        dates: List of unique game dates to compute conference strengths for.

    Returns:
        Dict mapping (date_str, conference) -> avg_adj_net.
    """
    if ratings.empty or "conference" not in ratings.columns:
        return {}

    ratings = ratings.copy()
    ratings["adj_net"] = ratings["adj_oe"] - ratings["adj_de"]
    ratings["rating_date"] = pd.to_datetime(ratings["rating_date"], errors="coerce")

    lookup: dict[tuple[str, str], float] = {}
    for game_dt in dates:
        if pd.isna(game_dt):
            continue
        dt = pd.Timestamp(game_dt)
        if hasattr(dt, 'tz') and dt.tz is not None:
            dt = dt.tz_localize(None)
        cutoff = dt.normalize() - timedelta(days=1)
        # Get latest rating per team before cutoff
        eligible = ratings[ratings["rating_date"] <= cutoff]
        if eligible.empty:
            continue
        latest = eligible.sort_values("rating_date").groupby("teamId").last()
        if "conference" not in latest.columns:
            continue
        conf_means = latest.groupby("conference")["adj_net"].mean()
        date_str = cutoff.strftime("%Y-%m-%d")
        for conf, val in conf_means.items():
            lookup[(date_str, str(conf))] = float(val)

    return lookup


def _compute_scoring_variance(
    games: pd.DataFrame,
    window: int = 10,
) -> dict[tuple[int, int], float]:
    """Compute rolling standard deviation of scoring margin per team.

    Anti-leakage: .shift(1) excludes the current game. Rolling window only
    uses prior games.

    Args:
        games: DataFrame with gameId, homeTeamId, awayTeamId, homeScore,
            awayScore, startDate.
        window: Number of prior games for rolling std.

    Returns:
        Dict mapping (gameId, teamId) -> margin_std (float).
    """
    dates = pd.to_datetime(games["startDate"], errors="coerce")

    # Expand to per-team view with margin
    rows = []
    for _, g in games.iterrows():
        dt = dates[g.name]
        hs = g.get("homeScore")
        aws = g.get("awayScore")
        if pd.notna(hs) and pd.notna(aws):
            home_margin = float(hs) - float(aws)
        else:
            home_margin = np.nan
        rows.append({
            "gameId": int(g["gameId"]),
            "teamId": int(g["homeTeamId"]),
            "date": dt,
            "margin": home_margin,
        })
        rows.append({
            "gameId": int(g["gameId"]),
            "teamId": int(g["awayTeamId"]),
            "date": dt,
            "margin": -home_margin if not np.isnan(home_margin) else np.nan,
        })

    team_games = pd.DataFrame(rows)
    team_games = team_games.sort_values(["teamId", "date", "gameId"]).reset_index(drop=True)

    # Compute rolling std with shift(1) to exclude current game
    results = []
    for _tid, group in team_games.groupby("teamId"):
        g = group.copy()
        g["margin_std"] = (
            g["margin"]
            .rolling(window=window, min_periods=3)
            .std()
            .shift(1)
        )
        results.append(g)

    out = pd.concat(results, ignore_index=True)
    return {
        (int(r["gameId"]), int(r["teamId"])): float(r["margin_std"])
        for _, r in out.iterrows()
        if pd.notna(r["margin_std"])
    }


def build_features(
    season: int,
    game_date: Optional[str] = None,
    no_garbage: bool = True,
    extra_features: list[str] | None = None,
    adjust_ff: bool = False,
    adjust_prior_weight: float = 5.0,
    adjust_alpha: float = 1.0,
) -> pd.DataFrame:
    """Build the feature matrix for games in a season.

    Args:
        season: Season year (e.g. 2026).
        game_date: If provided, only build features for games on this date.
        no_garbage: If True (default), use no-garbage-time efficiency ratings.
        extra_features: Optional list of extra feature group names to include
            beyond the base 37. Valid values: rest_days, sos, conf_strength,
            form_delta, tov_rate, margin_std.
        adjust_ff: If True, opponent-adjust four-factor stats before rolling
            averages. Default False for backward compatibility.
        adjust_prior_weight: Bayesian prior weight for adjustment shrinkage.
        adjust_alpha: SOS exponent for adjustment factor.

    Returns:
        DataFrame with columns: gameId, homeTeamId, awayTeamId, startDate,
        homeScore, awayScore, + base 37 features + any requested extra features.
    """
    extra = set(extra_features or [])
    unknown = extra - EXTRA_FEATURE_GROUPS
    if unknown:
        raise ValueError(f"Unknown extra feature groups: {unknown}")

    # Load raw data
    games = load_games(season)
    if games.empty:
        return pd.DataFrame()

    need_sos = "sos" in extra
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
    ff = pd.DataFrame()
    if not boxscores.empty:
        ff = compute_game_four_factors(boxscores)
        if adjust_ff:
            ff = adjust_four_factors(
                ff,
                prior_weight=adjust_prior_weight,
                alpha=adjust_alpha,
            )
        rolling_df = compute_rolling_averages(ff)

    # Build rolling lookups:
    #   1. (gameid, teamid) -> row dict   (fast path for played games)
    #   2. teamid -> DataFrame sorted by date (as-of fallback for future games)
    rolling_lookup: dict[tuple[int, int], dict[str, float]] = {}
    rolling_team_lookup: dict[int, pd.DataFrame] = {}
    if not rolling_df.empty:
        rolling_df["_date"] = pd.to_datetime(rolling_df["startdate"], errors="coerce")
        for _, row in rolling_df.iterrows():
            key = (int(row["gameid"]), int(row["teamid"]))
            rolling_lookup[key] = row.to_dict()
        # Build per-team time-series for as-of fallback
        for tid, group in rolling_df.groupby("teamid"):
            rolling_team_lookup[int(tid)] = group.sort_values("_date").copy()

    # Build date-aware efficiency lookup: teamId -> DataFrame of dated ratings
    eff_lookup = _build_efficiency_lookup(eff_ratings, include_sos=need_sos)

    # Parse game dates once
    games["_game_dt"] = pd.to_datetime(games["startDate"], errors="coerce")

    # ── Extra feature pre-computations ─────────────────────────────
    rest_lookup: dict[tuple[int, int], float] = {}
    if "rest_days" in extra:
        # Use ALL season games for rest computation (not just date-filtered)
        # so future games can see prior games for rest day calculation
        all_games_for_rest = load_games(season) if game_date is not None else games
        rest_lookup = _compute_rest_days(all_games_for_rest)

    tov_lookup: dict[tuple[int, int], dict[str, float]] = {}
    tov_team_lookup: dict[int, pd.DataFrame] = {}
    if "tov_rate" in extra and not boxscores.empty:
        tov_df = compute_rolling_turnovers(boxscores)
        if not tov_df.empty:
            tov_df["_date"] = pd.to_datetime(tov_df["startdate"] if "startdate" in tov_df.columns else tov_df.get("_date"), errors="coerce")
            for _, row in tov_df.iterrows():
                key = (int(row["gameid"]), int(row["teamid"]))
                tov_lookup[key] = row.to_dict()
            for tid, group in tov_df.groupby("teamid"):
                tov_team_lookup[int(tid)] = group.sort_values("_date").copy()

    form_lookup: dict[tuple[int, int], float] = {}
    form_team_lookup: dict[int, pd.DataFrame] = {}
    if "form_delta" in extra and not ff.empty:
        form_df = compute_form_delta(ff)
        if not form_df.empty:
            # form_df has gameid, teamid, form_delta — need dates from ff
            ff_dates = ff[["gameid", "teamid", "startdate"]].drop_duplicates(["gameid", "teamid"])
            form_df = form_df.merge(ff_dates, on=["gameid", "teamid"], how="left")
            form_df["_date"] = pd.to_datetime(form_df["startdate"], errors="coerce")
            for _, row in form_df.iterrows():
                form_lookup[(int(row["gameid"]), int(row["teamid"]))] = float(row["form_delta"])
            for tid, group in form_df.groupby("teamid"):
                form_team_lookup[int(tid)] = group.sort_values("_date").copy()

    conf_lookup: dict[tuple[str, str], float] = {}
    team_conf: dict[int, str] = {}
    if "conf_strength" in extra:
        # Build conference mapping from gold layer ratings (has conference col)
        if "conference" in eff_ratings.columns:
            for tid, conf in zip(eff_ratings["teamId"], eff_ratings["conference"]):
                if pd.notna(conf):
                    team_conf[int(tid)] = str(conf)
        unique_dates = games["_game_dt"].dropna().unique()
        conf_lookup = _build_conf_strength_lookup(eff_ratings, list(unique_dates))

    margin_std_lookup: dict[tuple[int, int], float] = {}
    margin_std_team_lookup: dict[int, pd.DataFrame] = {}
    if "margin_std" in extra:
        # Use ALL season games for margin_std (not just date-filtered)
        all_games_for_margin = load_games(season) if game_date is not None else games
        margin_std_lookup = _compute_scoring_variance(all_games_for_margin)
        # Build per-team as-of lookup for future games
        if margin_std_lookup:
            _ms_rows = []
            for (gid_key, tid_key), val in margin_std_lookup.items():
                _ms_rows.append({"gameid": gid_key, "teamid": tid_key, "margin_std": val})
            _ms_df = pd.DataFrame(_ms_rows)
            # Get dates from all_games_for_margin
            _dates_df = all_games_for_margin[["gameId", "startDate"]].copy()
            _dates_df["_date"] = pd.to_datetime(_dates_df["startDate"], errors="coerce")
            _ms_df = _ms_df.merge(
                _dates_df.rename(columns={"gameId": "gameid"}),
                on="gameid", how="left",
            )
            for tid, group in _ms_df.groupby("teamid"):
                margin_std_team_lookup[int(tid)] = group.sort_values("_date").copy()

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
            home_eff = _get_asof_rating(eff_lookup, home_tid, game_dt, include_sos=need_sos)
            away_eff = _get_asof_rating(eff_lookup, away_tid, game_dt, include_sos=need_sos)

        # Group 1: Efficiency features
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
        if not away_rolling and not pd.isna(game_dt):
            away_rolling = _get_asof_rolling(
                rolling_team_lookup, away_tid, game_dt,
                list(AWAY_ROLLING_MAP.values()),
            )
        for feat_name, rolling_col in AWAY_ROLLING_MAP.items():
            feat[feat_name] = away_rolling.get(rolling_col)

        # Group 2: Rolling four-factor averages (home team)
        home_rolling = rolling_lookup.get((gid, home_tid), {})
        if not home_rolling and not pd.isna(game_dt):
            home_rolling = _get_asof_rolling(
                rolling_team_lookup, home_tid, game_dt,
                list(HOME_ROLLING_MAP.values()),
            )
        for feat_name, rolling_col in HOME_ROLLING_MAP.items():
            feat[feat_name] = home_rolling.get(rolling_col)

        # ── Extra features ─────────────────────────────────────────
        if "rest_days" in extra:
            h_rest = rest_lookup.get((gid, home_tid), 5.0)
            a_rest = rest_lookup.get((gid, away_tid), 5.0)
            feat["home_rest_days"] = h_rest
            feat["away_rest_days"] = a_rest
            feat["rest_advantage"] = h_rest - a_rest

        if "sos" in extra:
            feat["home_sos_oe"] = home_eff.get("sos_oe")
            feat["home_sos_de"] = home_eff.get("sos_de")
            feat["away_sos_oe"] = away_eff.get("sos_oe")
            feat["away_sos_de"] = away_eff.get("sos_de")

        if "conf_strength" in extra:
            if not pd.isna(game_dt):
                dt_norm = pd.Timestamp(game_dt)
                if hasattr(dt_norm, 'tz') and dt_norm.tz is not None:
                    dt_norm = dt_norm.tz_localize(None)
                date_key = (dt_norm.normalize() - timedelta(days=1)).strftime("%Y-%m-%d")
                h_conf = team_conf.get(home_tid, "")
                a_conf = team_conf.get(away_tid, "")
                feat["home_conf_strength"] = conf_lookup.get((date_key, h_conf))
                feat["away_conf_strength"] = conf_lookup.get((date_key, a_conf))
            else:
                feat["home_conf_strength"] = None
                feat["away_conf_strength"] = None

        if "form_delta" in extra:
            h_form = form_lookup.get((gid, home_tid))
            a_form = form_lookup.get((gid, away_tid))
            if h_form is None and not pd.isna(game_dt):
                h_asof = _get_asof_rolling(form_team_lookup, home_tid, game_dt, ["form_delta"])
                h_form = h_asof.get("form_delta")
            if a_form is None and not pd.isna(game_dt):
                a_asof = _get_asof_rolling(form_team_lookup, away_tid, game_dt, ["form_delta"])
                a_form = a_asof.get("form_delta")
            feat["home_form_delta"] = h_form
            feat["away_form_delta"] = a_form

        if "tov_rate" in extra:
            away_tov = tov_lookup.get((gid, away_tid), {})
            if not away_tov and not pd.isna(game_dt):
                away_tov = _get_asof_rolling(
                    tov_team_lookup, away_tid, game_dt,
                    list(AWAY_TOV_MAP.values()),
                )
            home_tov = tov_lookup.get((gid, home_tid), {})
            if not home_tov and not pd.isna(game_dt):
                home_tov = _get_asof_rolling(
                    tov_team_lookup, home_tid, game_dt,
                    list(HOME_TOV_MAP.values()),
                )
            for feat_name, tov_col in AWAY_TOV_MAP.items():
                feat[feat_name] = away_tov.get(tov_col)
            for feat_name, tov_col in HOME_TOV_MAP.items():
                feat[feat_name] = home_tov.get(tov_col)

        if "margin_std" in extra:
            h_ms = margin_std_lookup.get((gid, home_tid))
            a_ms = margin_std_lookup.get((gid, away_tid))
            if h_ms is None and not pd.isna(game_dt):
                h_asof = _get_asof_rolling(margin_std_team_lookup, home_tid, game_dt, ["margin_std"])
                h_ms = h_asof.get("margin_std")
            if a_ms is None and not pd.isna(game_dt):
                a_asof = _get_asof_rolling(margin_std_team_lookup, away_tid, game_dt, ["margin_std"])
                a_ms = a_asof.get("margin_std")
            feat["home_margin_std"] = h_ms
            feat["away_margin_std"] = a_ms

        # Metadata (not part of features)
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

    # Verify we have the base 37 features
    missing = [f for f in config.FEATURE_ORDER if f not in result.columns]
    if missing:
        for col in missing:
            result[col] = None

    return result


def get_feature_matrix(
    df: pd.DataFrame,
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    """Extract feature columns in the correct order.

    Args:
        df: DataFrame with feature columns + metadata.
        feature_order: Custom feature order. Defaults to config.FEATURE_ORDER (base 37).

    Returns:
        DataFrame with only the feature columns in the specified order.
    """
    order = feature_order or config.FEATURE_ORDER
    # Only include columns that exist in df
    available = [c for c in order if c in df.columns]
    return df[available].copy()


def get_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Extract training targets: spread_home and home_win."""
    out = pd.DataFrame()
    out["spread_home"] = df["homeScore"].astype(float) - df["awayScore"].astype(float)
    out["home_win"] = (out["spread_home"] > 0).astype(float)
    return out


# ── Feature names for each extra group ───────────────────────────

EXTRA_FEATURE_NAMES: dict[str, list[str]] = {
    "rest_days": ["home_rest_days", "away_rest_days", "rest_advantage"],
    "sos": ["home_sos_oe", "home_sos_de", "away_sos_oe", "away_sos_de"],
    "conf_strength": ["home_conf_strength", "away_conf_strength"],
    "form_delta": ["home_form_delta", "away_form_delta"],
    "tov_rate": ["home_tov_rate", "home_def_tov_rate", "away_tov_rate", "away_def_tov_rate"],
    "margin_std": ["home_margin_std", "away_margin_std"],
}


# ══════════════════════════════════════════════════════════════════
# V2 FEATURE PIPELINE — Extended feature set (~120 features)
# All new code below; existing functions above are untouched.
# ══════════════════════════════════════════════════════════════════

from .opponent_adjustment import opponent_adjust, build_stat_pairs
from .pbp_advanced_stats import (
    ALL_ADVANCED_STAT_COLS,
    SHOT_QUALITY_COLS,
    TURNOVER_DECOMP_COLS,
    TEMPO_COLS,
    PUTBACK_COLS,
    CLUTCH_COLS,
    DROUGHT_COLS,
    HALF_SPLIT_COLS,
    ROTATION_DEPTH_COLS,
    PRESSURE_FT_COLS,
    ZONE_COUNT_COLS,
    COMPOSITE_COLS,
    compute_advanced_stats,
)
from .schedule_features import compute_schedule_features, SCHEDULE_FEATURE_NAMES
from .pace_features import compute_pace_features, PACE_FEATURE_COLS
from .kill_shot_analysis import compute_kill_shot_metrics, KILL_SHOT_COLS
from .luck_regression import compute_luck_features, LUCK_FEATURE_COLS
from .rolling_averages import compute_rolling_averages_v2

# ── Stat pairs for opponent adjustment of advanced stats ─────────
# Offensive stats → defensive counterpart, and vice versa.

_SHOT_QUALITY_PAIRS = {
    "rim_rate": "def_rim_rate",
    "mid_range_rate": "def_mid_range_rate",
    "rim_fg_pct": "def_rim_fg_pct",
    "mid_range_fg_pct": "def_mid_range_fg_pct",
    "def_rim_rate": "rim_rate",
    "def_mid_range_rate": "mid_range_rate",
    "def_rim_fg_pct": "rim_fg_pct",
    "def_mid_range_fg_pct": "mid_range_fg_pct",
}

_TURNOVER_PAIRS = {
    "live_ball_tov_rate": "steal_rate_defense",
    "dead_ball_tov_rate": "steal_rate_defense",  # approximate pair
    "steal_rate_defense": "live_ball_tov_rate",
}

_TEMPO_PAIRS = {
    "avg_possession_length": "avg_possession_length",  # self-paired (game-level)
    "early_clock_shot_rate": "early_clock_shot_rate",
    "shot_clock_pressure_rate": "shot_clock_pressure_rate",
}

_CLUTCH_PAIRS = {
    "clutch_off_efficiency": "clutch_def_efficiency",
    "clutch_def_efficiency": "clutch_off_efficiency",
    "clutch_eff_fg_pct": "clutch_eff_fg_pct",  # self-paired
}

_HALF_SPLIT_PAIRS = {
    "first_half_off_efficiency": "first_half_def_efficiency",
    "second_half_off_efficiency": "second_half_def_efficiency",
    "first_half_def_efficiency": "first_half_off_efficiency",
    "second_half_def_efficiency": "second_half_off_efficiency",
    "half_adjustment_delta": "second_half_def_delta",
    "second_half_def_delta": "half_adjustment_delta",
}

_DROUGHT_PAIRS = {
    "off_avg_drought_length": "def_avg_drought_length",
    "off_max_drought_length": "def_max_drought_length",
    "off_drought_frequency": "def_drought_frequency",
    "def_avg_drought_length": "off_avg_drought_length",
    "def_max_drought_length": "off_max_drought_length",
    "def_drought_frequency": "off_drought_frequency",
}

_PUTBACK_PAIRS = {
    "putback_rate": "putback_rate",  # self-paired (adjust vs opp def)
    "second_chance_pts_per_oreb": "second_chance_pts_per_oreb",
}

_ROTATION_PAIRS = {
    "scoring_hhi": "scoring_hhi",
    "top2_scorer_pct": "top2_scorer_pct",
}

# All advanced stat pairs combined
_ALL_ADV_STAT_PAIRS = {}
_ALL_ADV_STAT_PAIRS.update(_SHOT_QUALITY_PAIRS)
_ALL_ADV_STAT_PAIRS.update(_TURNOVER_PAIRS)
_ALL_ADV_STAT_PAIRS.update(_TEMPO_PAIRS)
_ALL_ADV_STAT_PAIRS.update(_CLUTCH_PAIRS)
_ALL_ADV_STAT_PAIRS.update(_HALF_SPLIT_PAIRS)
_ALL_ADV_STAT_PAIRS.update(_DROUGHT_PAIRS)
_ALL_ADV_STAT_PAIRS.update(_PUTBACK_PAIRS)
_ALL_ADV_STAT_PAIRS.update(_ROTATION_PAIRS)

# Stats that should NOT be opponent-adjusted
_ADV_NO_ADJUST = {
    "transition_rate",       # Already a relative/context stat
    "unique_scorers",        # Count, not a rate to adjust
    "clutch_tov_rate",       # Too sparse for reliable adjustment
    "clutch_ft_pct",         # Opponent-independent
    "non_clutch_to_clutch_delta",  # Already a delta
    "pressure_ft_pct",       # Opponent-independent
    "non_pressure_ft_pct",   # Opponent-independent
    "ft_pressure_delta",     # Already a delta
    "three_pt_rate_corner",  # Distribution stat, hard to adjust
    "three_pt_rate_above_break",
    "assisted_fg_pct",       # Mostly about the offense
    "unassisted_fg_pct",
    # Luck features: league-relative by construction
    "efg_luck",
    "three_pt_luck",
    "two_pt_luck",
    # Composite features: derived from already-adjusted components
    "transition_scoring_efficiency",
    "expected_pts_per_shot",
    "transition_value",
}

# Per-team advanced stat columns that become rolling features (home_ / away_ prefixed)
# Exclude clutch/pressure stats that are too sparse or deltas
_ROLLING_ADV_STATS = [
    # Shot quality (Group A)
    "rim_rate", "mid_range_rate", "rim_fg_pct", "mid_range_fg_pct",
    "assisted_fg_pct",
    "def_rim_rate", "def_mid_range_rate", "def_rim_fg_pct", "def_mid_range_fg_pct",
    # Turnover decomp (Group C)
    "live_ball_tov_rate", "dead_ball_tov_rate", "steal_rate_defense", "transition_rate",
    # Tempo (Group D)
    "avg_possession_length", "early_clock_shot_rate", "shot_clock_pressure_rate",
    # Putbacks (Group G)
    "putback_rate", "second_chance_pts_per_oreb",
    # Clutch (Group H) - only the most reliable ones
    "clutch_off_efficiency", "clutch_def_efficiency", "non_clutch_to_clutch_delta",
    # Droughts (Group I)
    "off_avg_drought_length", "off_max_drought_length",
    "def_avg_drought_length", "def_max_drought_length",
    # Half splits (Group J)
    "half_adjustment_delta", "second_half_def_delta",
    # Rotation depth (Group M)
    "scoring_hhi", "top2_scorer_pct",
    # Pressure FT (Group N)
    "ft_pressure_delta",
    # Composites (NOT zone counts — those are intermediate only)
    "transition_scoring_efficiency", "expected_pts_per_shot", "transition_value",
]

# Consistency features (Group F) - computed from game-level data
_CONSISTENCY_STATS = [
    "scoring_variance", "blowout_rate",
]


def compute_rolling_advanced_stats(
    adv_stats: pd.DataFrame,
    stat_cols: list[str],
    ewm_span: int = 15,
) -> pd.DataFrame:
    """Compute EWM rolling averages of advanced stats per team.

    Same pattern as rolling_averages.compute_rolling_averages() but for
    arbitrary stat columns.

    Args:
        adv_stats: DataFrame with gameid, teamid, startdate, + stat columns.
        stat_cols: List of stat columns to compute rolling averages for.
        ewm_span: EWM span parameter.

    Returns:
        DataFrame with gameid, teamid, startdate, + rolling_{stat} columns.
    """
    df = adv_stats.copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.sort_values(["teamid", "_date", "gameid"]).reset_index(drop=True)

    rolling_cols = [f"rolling_{c}" for c in stat_cols]

    results = []
    for _tid, group in df.groupby("teamid"):
        g = group.copy()
        for stat, rcol in zip(stat_cols, rolling_cols):
            if stat in g.columns:
                g[rcol] = (
                    g[stat]
                    .ewm(span=ewm_span, min_periods=1)
                    .mean()
                    .shift(1)
                )
            else:
                g[rcol] = np.nan
        results.append(g)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    keep = ["gameid", "teamid", "startdate"] + rolling_cols
    return out[[c for c in keep if c in out.columns]].copy()


# ── Per-group EWM span optimization (V2 rolling) ─────────────────

def _load_optimal_spans() -> dict[str, int]:
    """Load optimal EWM spans from artifacts/optimal_ewm_spans.json.

    Returns empty dict if file does not exist.
    """
    path = config.ARTIFACTS_DIR / "optimal_ewm_spans.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# Mapping from individual stat column → group name (for span lookup)
_STAT_TO_GROUP: dict[str, str] = {}
_GROUP_DEFS = {
    "shot_quality": [
        "rim_rate", "mid_range_rate", "rim_fg_pct", "mid_range_fg_pct",
        "assisted_fg_pct",
        "def_rim_rate", "def_mid_range_rate", "def_rim_fg_pct", "def_mid_range_fg_pct",
    ],
    "turnover": [
        "live_ball_tov_rate", "dead_ball_tov_rate", "steal_rate_defense",
        "transition_rate", "transition_scoring_efficiency",
    ],
    "tempo": [
        "avg_possession_length", "early_clock_shot_rate", "shot_clock_pressure_rate",
    ],
    "second_chance": [
        "putback_rate", "second_chance_pts_per_oreb",
    ],
    "clutch": [
        "clutch_off_efficiency", "clutch_def_efficiency",
        "non_clutch_to_clutch_delta",
    ],
    "drought": [
        "off_avg_drought_length", "off_max_drought_length",
        "def_avg_drought_length", "def_max_drought_length",
    ],
    "half_split": [
        "half_adjustment_delta", "second_half_def_delta",
    ],
    "rotation": [
        "scoring_hhi", "top2_scorer_pct",
    ],
    "composites": [
        "transition_scoring_efficiency", "expected_pts_per_shot", "transition_value",
    ],
    "luck": [
        "efg_luck", "three_pt_luck", "two_pt_luck",
    ],
}
for _grp, _cols in _GROUP_DEFS.items():
    for _col in _cols:
        _STAT_TO_GROUP[_col] = _grp


def compute_rolling_advanced_stats_v2(
    adv_stats: pd.DataFrame,
    stat_cols: list[str],
    default_span: int = 15,
) -> pd.DataFrame:
    """Compute EWM rolling averages with per-group optimal spans.

    Same interface as compute_rolling_advanced_stats() but looks up the
    optimal EWM span for each stat's group from artifacts/optimal_ewm_spans.json.
    Falls back to default_span if no artifact or unmapped stat.

    Args:
        adv_stats: DataFrame with gameid, teamid, startdate, + stat columns.
        stat_cols: List of stat columns to compute rolling averages for.
        default_span: Fallback span if no optimal span found.

    Returns:
        DataFrame with gameid, teamid, startdate, + rolling_{stat} columns.
    """
    optimal_spans = _load_optimal_spans()

    df = adv_stats.copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.sort_values(["teamid", "_date", "gameid"]).reset_index(drop=True)

    rolling_cols = [f"rolling_{c}" for c in stat_cols]

    # Pre-compute the span for each stat
    stat_spans = []
    for stat in stat_cols:
        group = _STAT_TO_GROUP.get(stat)
        span = optimal_spans.get(group, default_span) if group else default_span
        stat_spans.append(span)

    results = []
    for _tid, group_df in df.groupby("teamid"):
        g = group_df.copy()
        for stat, rcol, span in zip(stat_cols, rolling_cols, stat_spans):
            if stat in g.columns:
                g[rcol] = (
                    g[stat]
                    .ewm(span=span, min_periods=1)
                    .mean()
                    .shift(1)
                )
            else:
                g[rcol] = np.nan
        results.append(g)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    keep = ["gameid", "teamid", "startdate"] + rolling_cols
    return out[[c for c in keep if c in out.columns]].copy()


def _compute_consistency_features(
    games: pd.DataFrame,
    window: int = 10,
) -> dict[tuple[int, int], dict[str, float]]:
    """Compute consistency/variance features (Group F) from game-level data.

    Returns dict mapping (gameId, teamId) -> {scoring_variance, blowout_rate}.
    """
    dates = pd.to_datetime(games["startDate"], errors="coerce")

    rows = []
    for _, g in games.iterrows():
        dt = dates[g.name]
        hs = g.get("homeScore")
        aws = g.get("awayScore")
        if pd.notna(hs) and pd.notna(aws):
            home_margin = float(hs) - float(aws)
        else:
            home_margin = np.nan

        rows.append({
            "gameId": int(g["gameId"]),
            "teamId": int(g["homeTeamId"]),
            "date": dt,
            "margin": home_margin,
            "is_blowout": abs(home_margin) >= 15 if not np.isnan(home_margin) else np.nan,
        })
        rows.append({
            "gameId": int(g["gameId"]),
            "teamId": int(g["awayTeamId"]),
            "date": dt,
            "margin": -home_margin if not np.isnan(home_margin) else np.nan,
            "is_blowout": abs(home_margin) >= 15 if not np.isnan(home_margin) else np.nan,
        })

    team_games = pd.DataFrame(rows)
    team_games = team_games.sort_values(["teamId", "date", "gameId"]).reset_index(drop=True)

    result_map: dict[tuple[int, int], dict[str, float]] = {}

    for _tid, group in team_games.groupby("teamId"):
        g = group.copy()
        g["scoring_variance"] = (
            g["margin"]
            .rolling(window=window, min_periods=3)
            .std()
            .shift(1)
        )
        g["blowout_rate"] = (
            g["is_blowout"]
            .rolling(window=window, min_periods=3)
            .mean()
            .shift(1)
        )
        for _, row in g.iterrows():
            d = {}
            if pd.notna(row["scoring_variance"]):
                d["scoring_variance"] = float(row["scoring_variance"])
            if pd.notna(row["blowout_rate"]):
                d["blowout_rate"] = float(row["blowout_rate"])
            if d:
                result_map[(int(row["gameId"]), int(row["teamId"]))] = d

    return result_map


def build_features_v2(
    season: int,
    game_date: Optional[str] = None,
    no_garbage: bool = True,
) -> pd.DataFrame:
    """Build the EXPANDED feature matrix (~120 features) for games in a season.

    This is the V2 pipeline that adds PBP-derived advanced stats, schedule
    features, pace features, and consistency features on top of the existing
    efficiency ratings and opponent-adjusted four factors.

    Args:
        season: Season year (e.g. 2026).
        game_date: If provided, only build features for games on this date.
        no_garbage: If True (default), use no-garbage-time efficiency ratings.

    Returns:
        DataFrame with columns: gameId, homeTeamId, awayTeamId, startDate,
        homeScore, awayScore, + all V2 features.
    """
    # Load raw data (same as existing pipeline)
    games = load_games(season)
    if games.empty:
        return pd.DataFrame()

    eff_ratings = load_efficiency_ratings(season, no_garbage=no_garbage)
    boxscores = load_boxscores(season)

    # ── Date filtering ────────────────────────────────────────────
    all_games = games.copy()  # Keep all for context lookups
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

    # ── Group 1: Efficiency ratings (same as v1) ──────────────────
    eff_lookup = _build_efficiency_lookup(eff_ratings, include_sos=True)
    games["_game_dt"] = pd.to_datetime(games["startDate"], errors="coerce")

    # ── Group 2: Opponent-adjusted rolling four factors ────────────
    rolling_df = pd.DataFrame()
    ff = pd.DataFrame()
    if not boxscores.empty:
        ff = compute_game_four_factors(boxscores)
        # Apply opponent adjustment using the existing multiplicative method
        ff = adjust_four_factors(
            ff,
            prior_weight=config.ADJUST_PRIOR,
            alpha=config.ADJUST_ALPHA,
        )
        # V2: use per-group optimal span for four factors
        ff_span = _load_optimal_spans().get("four_factors")
        rolling_df = compute_rolling_averages_v2(ff, optimal_span=ff_span)

    # Build rolling lookups
    rolling_lookup: dict[tuple[int, int], dict[str, float]] = {}
    rolling_team_lookup: dict[int, pd.DataFrame] = {}
    if not rolling_df.empty:
        rolling_df["_date"] = pd.to_datetime(rolling_df["startdate"], errors="coerce")
        for _, row in rolling_df.iterrows():
            key = (int(row["gameid"]), int(row["teamid"]))
            rolling_lookup[key] = row.to_dict()
        for tid, group in rolling_df.groupby("teamid"):
            rolling_team_lookup[int(tid)] = group.sort_values("_date").copy()

    # ── PBP Advanced Stats (Groups A, C, D, G, H, I, J, M, N) ────
    adv_rolling_lookup: dict[tuple[int, int], dict[str, float]] = {}
    adv_team_lookup: dict[int, pd.DataFrame] = {}

    # ── Luck features ────────────────────────────────────────────
    luck_lookup: dict[tuple[int, int], dict[str, float]] = {}
    luck_team_lookup: dict[int, pd.DataFrame] = {}

    # Load enriched PBP for this season
    pbp_keys = s3_reader.list_parquet_keys(
        f"{config.SILVER_PREFIX}/fct_pbp_plays_enriched/season={season}/"
    )
    if pbp_keys:
        pbp_tbl = s3_reader.read_parquet_table(pbp_keys)
        pbp_df = pbp_tbl.to_pandas()

        # Compute per-game advanced stats
        adv_stats = compute_advanced_stats(pbp_df)

        if not adv_stats.empty:
            # Compute luck features on RAW (pre-opponent-adjusted) stats
            luck_df = compute_luck_features(adv_stats)
            if not luck_df.empty:
                luck_rolling = compute_rolling_advanced_stats_v2(luck_df, LUCK_FEATURE_COLS)
                if not luck_rolling.empty:
                    luck_rolling["_date"] = pd.to_datetime(luck_rolling["startdate"], errors="coerce")
                    for _, row in luck_rolling.iterrows():
                        key = (int(row["gameid"]), int(row["teamid"]))
                        luck_lookup[key] = row.to_dict()
                    for tid, group in luck_rolling.groupby("teamid"):
                        luck_team_lookup[int(tid)] = group.sort_values("_date").copy()

            # Opponent-adjust the advanced stats
            adjustable = [s for s in _ROLLING_ADV_STATS if s not in _ADV_NO_ADJUST and s in adv_stats.columns]
            pairs_for_adj = {s: _ALL_ADV_STAT_PAIRS[s] for s in adjustable if s in _ALL_ADV_STAT_PAIRS}
            if adjustable and pairs_for_adj:
                adv_stats = opponent_adjust(
                    adv_stats,
                    stat_cols=adjustable,
                    stat_pairs=pairs_for_adj,
                    no_adjust=_ADV_NO_ADJUST,
                )

            # V2: Compute rolling averages with per-group optimal spans
            available_stats = [s for s in _ROLLING_ADV_STATS if s in adv_stats.columns]
            adv_rolling = compute_rolling_advanced_stats_v2(adv_stats, available_stats)

            if not adv_rolling.empty:
                adv_rolling["_date"] = pd.to_datetime(adv_rolling["startdate"], errors="coerce")
                for _, row in adv_rolling.iterrows():
                    key = (int(row["gameid"]), int(row["teamid"]))
                    adv_rolling_lookup[key] = row.to_dict()
                for tid, group in adv_rolling.groupby("teamid"):
                    adv_team_lookup[int(tid)] = group.sort_values("_date").copy()

    # ── Schedule features (Group L) ───────────────────────────────
    schedule_df = compute_schedule_features(all_games)
    schedule_lookup: dict[int, dict] = {}
    if not schedule_df.empty:
        for _, row in schedule_df.iterrows():
            schedule_lookup[int(row["gameId"])] = row.to_dict()

    # ── Pace features (Group K) ───────────────────────────────────
    pace_lookup: dict[tuple[int, int], dict[str, float]] = {}
    pace_team_lookup: dict[int, pd.DataFrame] = {}
    if not boxscores.empty:
        pace_df = compute_pace_features(boxscores)
        if not pace_df.empty:
            pace_df["_date"] = pd.to_datetime(pace_df["startdate"], errors="coerce")
            for _, row in pace_df.iterrows():
                key = (int(row["gameid"]), int(row["teamid"]))
                pace_lookup[key] = row.to_dict()
            for tid, group in pace_df.groupby("teamid"):
                pace_team_lookup[int(tid)] = group.sort_values("_date").copy()

    # ── Kill shot metrics (Group E) ──────────────────────────────
    ks_lookup: dict[tuple[int, int], dict[str, float]] = {}
    ks_team_lookup: dict[int, pd.DataFrame] = {}
    if pbp_keys:
        # pbp_df already loaded above for advanced stats
        ks_df = compute_kill_shot_metrics(pbp_df)
        if not ks_df.empty:
            # Compute rolling averages for kill shot metrics
            ks_rolling = compute_rolling_advanced_stats(ks_df, KILL_SHOT_COLS)
            if not ks_rolling.empty:
                ks_rolling["_date"] = pd.to_datetime(ks_rolling["startdate"], errors="coerce")
                for _, row in ks_rolling.iterrows():
                    key = (int(row["gameid"]), int(row["teamid"]))
                    ks_lookup[key] = row.to_dict()
                for tid, group in ks_rolling.groupby("teamid"):
                    ks_team_lookup[int(tid)] = group.sort_values("_date").copy()

    # ── Consistency features (Group F) ────────────────────────────
    consistency_lookup = _compute_consistency_features(all_games)

    # ── Turnover rate rolling (existing) ──────────────────────────
    tov_lookup: dict[tuple[int, int], dict[str, float]] = {}
    tov_team_lookup: dict[int, pd.DataFrame] = {}
    if not boxscores.empty:
        tov_df = compute_rolling_turnovers(boxscores)
        if not tov_df.empty:
            tov_df["_date"] = pd.to_datetime(
                tov_df["startdate"] if "startdate" in tov_df.columns else tov_df.get("_date"),
                errors="coerce",
            )
            for _, row in tov_df.iterrows():
                key = (int(row["gameid"]), int(row["teamid"]))
                tov_lookup[key] = row.to_dict()
            for tid, group in tov_df.groupby("teamid"):
                tov_team_lookup[int(tid)] = group.sort_values("_date").copy()

    # ── Form delta ────────────────────────────────────────────────
    form_lookup: dict[tuple[int, int], float] = {}
    form_team_lookup: dict[int, pd.DataFrame] = {}
    if not ff.empty:
        form_df = compute_form_delta(ff)
        if not form_df.empty:
            ff_dates = ff[["gameid", "teamid", "startdate"]].drop_duplicates(["gameid", "teamid"])
            form_df = form_df.merge(ff_dates, on=["gameid", "teamid"], how="left")
            form_df["_date"] = pd.to_datetime(form_df["startdate"], errors="coerce")
            for _, row in form_df.iterrows():
                form_lookup[(int(row["gameid"]), int(row["teamid"]))] = float(row["form_delta"])
            for tid, group in form_df.groupby("teamid"):
                form_team_lookup[int(tid)] = group.sort_values("_date").copy()

    # ── Assemble features per game ────────────────────────────────
    records = []
    for _, game in games.iterrows():
        gid = int(game["gameId"])
        home_tid = int(game["homeTeamId"])
        away_tid = int(game["awayTeamId"])
        game_dt = game["_game_dt"]
        neutral = bool(game.get("neutralSite", False))

        # Efficiency ratings (as-of)
        if pd.isna(game_dt):
            home_eff = {}
            away_eff = {}
        else:
            home_eff = _get_asof_rating(eff_lookup, home_tid, game_dt, include_sos=True)
            away_eff = _get_asof_rating(eff_lookup, away_tid, game_dt, include_sos=True)

        # Group 1: Efficiency features
        feat: dict = {
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
            "away_team_home": 0,
        }

        # Group 2: Rolling four factors (away)
        away_rolling = rolling_lookup.get((gid, away_tid), {})
        if not away_rolling and not pd.isna(game_dt):
            away_rolling = _get_asof_rolling(
                rolling_team_lookup, away_tid, game_dt,
                list(AWAY_ROLLING_MAP.values()),
            )
        for feat_name, rolling_col in AWAY_ROLLING_MAP.items():
            feat[feat_name] = away_rolling.get(rolling_col)

        # Group 2: Rolling four factors (home)
        home_rolling = rolling_lookup.get((gid, home_tid), {})
        if not home_rolling and not pd.isna(game_dt):
            home_rolling = _get_asof_rolling(
                rolling_team_lookup, home_tid, game_dt,
                list(HOME_ROLLING_MAP.values()),
            )
        for feat_name, rolling_col in HOME_ROLLING_MAP.items():
            feat[feat_name] = home_rolling.get(rolling_col)

        # ── PBP Advanced Stats (rolling) ──────────────────────────
        for prefix, tid in [("away", away_tid), ("home", home_tid)]:
            adv_row = adv_rolling_lookup.get((gid, tid), {})
            if not adv_row and not pd.isna(game_dt):
                adv_row = _get_asof_rolling(
                    adv_team_lookup, tid, game_dt,
                    [f"rolling_{s}" for s in _ROLLING_ADV_STATS],
                )
            for stat in _ROLLING_ADV_STATS:
                feat_name = f"{prefix}_{stat}"
                feat[feat_name] = adv_row.get(f"rolling_{stat}")

        # ── Schedule features (Group L) ───────────────────────────
        sched = schedule_lookup.get(gid, {})
        feat["home_days_rest"] = sched.get("home_days_rest", 5.0)
        feat["away_days_rest"] = sched.get("away_days_rest", 5.0)
        feat["rest_differential"] = sched.get("rest_differential", 0.0)
        feat["home_games_last_7"] = sched.get("home_games_last_7", 1)
        feat["away_games_last_7"] = sched.get("away_games_last_7", 1)

        # ── Pace features (Group K) ───────────────────────────────
        for prefix, tid in [("away", away_tid), ("home", home_tid)]:
            pace_row = pace_lookup.get((gid, tid), {})
            if not pace_row and not pd.isna(game_dt):
                pace_row = _get_asof_rolling(
                    pace_team_lookup, tid, game_dt,
                    PACE_FEATURE_COLS,
                )
            for col in PACE_FEATURE_COLS:
                feat[f"{prefix}_{col}"] = pace_row.get(col)

        # ── Kill shot metrics (Group E) ─────────────────────────
        for prefix, tid in [("away", away_tid), ("home", home_tid)]:
            ks_row = ks_lookup.get((gid, tid), {})
            if not ks_row and not pd.isna(game_dt):
                ks_row = _get_asof_rolling(
                    ks_team_lookup, tid, game_dt,
                    [f"rolling_{s}" for s in KILL_SHOT_COLS],
                )
            for stat in KILL_SHOT_COLS:
                feat[f"{prefix}_{stat}"] = ks_row.get(f"rolling_{stat}")

        # ── Luck features ────────────────────────────────────────
        for prefix, tid in [("away", away_tid), ("home", home_tid)]:
            luck_row = luck_lookup.get((gid, tid), {})
            if not luck_row and not pd.isna(game_dt):
                luck_row = _get_asof_rolling(
                    luck_team_lookup, tid, game_dt,
                    [f"rolling_{s}" for s in LUCK_FEATURE_COLS],
                )
            for stat in LUCK_FEATURE_COLS:
                feat[f"{prefix}_{stat}"] = luck_row.get(f"rolling_{stat}")

        # ── Consistency features (Group F) ────────────────────────
        for prefix, tid in [("away", away_tid), ("home", home_tid)]:
            cons = consistency_lookup.get((gid, tid), {})
            feat[f"{prefix}_scoring_variance"] = cons.get("scoring_variance")
            feat[f"{prefix}_blowout_rate"] = cons.get("blowout_rate")

        # ── SOS features ──────────────────────────────────────────
        feat["home_sos_oe"] = home_eff.get("sos_oe")
        feat["home_sos_de"] = home_eff.get("sos_de")
        feat["away_sos_oe"] = away_eff.get("sos_oe")
        feat["away_sos_de"] = away_eff.get("sos_de")

        # ── Turnover rate (existing) ──────────────────────────────
        away_tov = tov_lookup.get((gid, away_tid), {})
        if not away_tov and not pd.isna(game_dt):
            away_tov = _get_asof_rolling(
                tov_team_lookup, away_tid, game_dt,
                list(AWAY_TOV_MAP.values()),
            )
        home_tov = tov_lookup.get((gid, home_tid), {})
        if not home_tov and not pd.isna(game_dt):
            home_tov = _get_asof_rolling(
                tov_team_lookup, home_tid, game_dt,
                list(HOME_TOV_MAP.values()),
            )
        for feat_name, tov_col in AWAY_TOV_MAP.items():
            feat[feat_name] = away_tov.get(tov_col)
        for feat_name, tov_col in HOME_TOV_MAP.items():
            feat[feat_name] = home_tov.get(tov_col)

        # ── Form delta ────────────────────────────────────────────
        for prefix, tid in [("away", away_tid), ("home", home_tid)]:
            fd = form_lookup.get((gid, tid))
            if fd is None and not pd.isna(game_dt):
                fd_asof = _get_asof_rolling(form_team_lookup, tid, game_dt, ["form_delta"])
                fd = fd_asof.get("form_delta")
            feat[f"{prefix}_form_delta"] = fd

        # ── Conference strength ───────────────────────────────────
        if "conference" in eff_ratings.columns:
            team_conf_map: dict[int, str] = {}
            for tid_val, conf in zip(eff_ratings["teamId"], eff_ratings["conference"]):
                if pd.notna(conf):
                    team_conf_map[int(tid_val)] = str(conf)

            if not pd.isna(game_dt):
                dt_norm = pd.Timestamp(game_dt)
                if hasattr(dt_norm, 'tz') and dt_norm.tz is not None:
                    dt_norm = dt_norm.tz_localize(None)
                date_key = (dt_norm.normalize() - timedelta(days=1)).strftime("%Y-%m-%d")
                # Reuse existing conf_lookup builder
                unique_dates = games["_game_dt"].dropna().unique()
                cl = _build_conf_strength_lookup(eff_ratings, list(unique_dates))
                h_conf = team_conf_map.get(home_tid, "")
                a_conf = team_conf_map.get(away_tid, "")
                feat["home_conf_strength"] = cl.get((date_key, h_conf))
                feat["away_conf_strength"] = cl.get((date_key, a_conf))
            else:
                feat["home_conf_strength"] = None
                feat["away_conf_strength"] = None
        else:
            feat["home_conf_strength"] = None
            feat["away_conf_strength"] = None

        # ── Scoring variance (existing) ───────────────────────────
        feat["home_margin_std"] = consistency_lookup.get((gid, home_tid), {}).get("scoring_variance")
        feat["away_margin_std"] = consistency_lookup.get((gid, away_tid), {}).get("scoring_variance")

        # Metadata
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
    return result

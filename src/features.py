"""Assemble the feature vector for each game.

Combines:
  - Group 1 (11 features): efficiency metrics from gold/team_adjusted_efficiencies + fct_games
  - Group 2 (26 features): rolling four-factor averages from fct_pbp_game_teams_flat
  - Extra feature groups (optional): rest_days, sos, conf_strength, form_delta, tov_rate, margin_std
"""

from __future__ import annotations

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
    compute_venue_split_rolling,
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


def _compute_team_hca(
    games: pd.DataFrame,
    eff_lookup: dict[int, pd.DataFrame],
) -> dict[tuple[int, int], float]:
    """Compute team-specific HCA from opponent-quality-adjusted residuals.

    For each completed game, computes:
        residual = actual_margin - expected_margin
    where expected_margin = team_adj_net - opp_adj_net (from efficiency ratings).

    Residuals are split by venue (home/away), then:
        team_hca = (ewm(home_residuals) - ewm(away_residuals)) / 2

    The /2 converts the full home/away gap into a per-game venue effect.

    Anti-leakage: shift(1) on each venue subset, then forward-fill to all games.

    Args:
        games: DataFrame with gameId, homeTeamId, awayTeamId, homeScore,
            awayScore, startDate.
        eff_lookup: Dict mapping teamId -> DataFrame of dated efficiency ratings
            (from _build_efficiency_lookup). Used to compute expected margins.

    Returns:
        Dict mapping (gameId, teamId) -> team_hca (float).
    """
    dates = pd.to_datetime(games["startDate"], errors="coerce")

    # Expand to per-team view with residual and venue flag
    rows = []
    for _, g in games.iterrows():
        dt = dates[g.name]
        gid = int(g["gameId"])
        home_tid = int(g["homeTeamId"])
        away_tid = int(g["awayTeamId"])
        hs = g.get("homeScore")
        aws = g.get("awayScore")

        residual = np.nan
        if pd.notna(hs) and pd.notna(aws) and not pd.isna(dt):
            actual_home_margin = float(hs) - float(aws)
            # Look up pre-game efficiency ratings for both teams
            home_eff = _get_asof_rating(eff_lookup, home_tid, dt)
            away_eff = _get_asof_rating(eff_lookup, away_tid, dt)
            if home_eff and away_eff:
                home_net = home_eff["adj_oe"] - home_eff["adj_de"]
                away_net = away_eff["adj_oe"] - away_eff["adj_de"]
                expected_margin = home_net - away_net
                residual = actual_home_margin - expected_margin

        rows.append({
            "gameId": gid,
            "teamId": home_tid,
            "date": dt,
            "residual": residual,
            "is_home": True,
        })
        rows.append({
            "gameId": gid,
            "teamId": away_tid,
            "date": dt,
            "residual": -residual if not np.isnan(residual) else np.nan,
            "is_home": False,
        })

    team_games = pd.DataFrame(rows)
    team_games = team_games.sort_values(["teamId", "date", "gameId"]).reset_index(drop=True)

    results = []
    for _tid, group in team_games.groupby("teamId"):
        g = group.copy()
        home_mask = g["is_home"]

        # Home-only residual EWM (min_periods=3 to avoid noisy early-season values)
        g["_home_res_ewm"] = np.nan
        home_idx = g.index[home_mask]
        if len(home_idx) > 0:
            g.loc[home_idx, "_home_res_ewm"] = (
                g.loc[home_idx, "residual"]
                .ewm(span=config.EWM_SPAN, min_periods=3)
                .mean()
                .shift(1)
            )
        g["_home_res_ewm"] = g["_home_res_ewm"].ffill()

        # Away-only residual EWM
        g["_away_res_ewm"] = np.nan
        away_idx = g.index[~home_mask]
        if len(away_idx) > 0:
            g.loc[away_idx, "_away_res_ewm"] = (
                g.loc[away_idx, "residual"]
                .ewm(span=config.EWM_SPAN, min_periods=3)
                .mean()
                .shift(1)
            )
        g["_away_res_ewm"] = g["_away_res_ewm"].ffill()

        # HCA = (home_residual_avg - away_residual_avg) / 2
        g["team_hca"] = (g["_home_res_ewm"] - g["_away_res_ewm"]) / 2
        results.append(g[["gameId", "teamId", "team_hca"]])

    if not results:
        return {}
    out = pd.concat(results, ignore_index=True)
    return {
        (int(r["gameId"]), int(r["teamId"])): float(r["team_hca"])
        for _, r in out.iterrows()
        if pd.notna(r["team_hca"])
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
            beyond the base features. Valid values: rest_days, sos, conf_strength,
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

    # ── HCA and venue-split pre-computations ─────────────────────
    # Always computed (base features, not extras)
    all_games_for_hca = load_games(season) if game_date is not None else games
    hca_lookup = _compute_team_hca(all_games_for_hca, eff_lookup)
    hca_team_lookup: dict[int, pd.DataFrame] = {}
    if hca_lookup:
        _hca_rows = []
        for (gid_key, tid_key), val in hca_lookup.items():
            _hca_rows.append({"gameid": gid_key, "teamid": tid_key, "team_hca": val})
        _hca_df = pd.DataFrame(_hca_rows)
        _dates_df = all_games_for_hca[["gameId", "startDate"]].copy()
        _dates_df["_date"] = pd.to_datetime(_dates_df["startDate"], errors="coerce")
        _hca_df = _hca_df.merge(
            _dates_df.rename(columns={"gameId": "gameid"}),
            on="gameid", how="left",
        )
        for tid, group in _hca_df.groupby("teamid"):
            hca_team_lookup[int(tid)] = group.sort_values("_date").copy()

    venue_split_lookup: dict[tuple[int, int], dict[str, float]] = {}
    venue_split_team_lookup: dict[int, pd.DataFrame] = {}
    if not ff.empty:
        venue_split_df = compute_venue_split_rolling(ff)
        if not venue_split_df.empty:
            venue_split_df["_date"] = pd.to_datetime(venue_split_df["startdate"], errors="coerce")
            for _, row in venue_split_df.iterrows():
                key = (int(row["gameid"]), int(row["teamid"]))
                venue_split_lookup[key] = row.to_dict()
            for tid, group in venue_split_df.groupby("teamid"):
                venue_split_team_lookup[int(tid)] = group.sort_values("_date").copy()

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

        # ── HCA and venue-split features ───────────────────────────
        if neutral:
            feat["home_team_hca"] = 0.0
        else:
            h_hca = hca_lookup.get((gid, home_tid))
            if h_hca is None and not pd.isna(game_dt):
                h_asof = _get_asof_rolling(hca_team_lookup, home_tid, game_dt, ["team_hca"])
                h_hca = h_asof.get("team_hca")
            feat["home_team_hca"] = h_hca

        home_vs = venue_split_lookup.get((gid, home_tid), {})
        if not home_vs and not pd.isna(game_dt):
            home_vs = _get_asof_rolling(
                venue_split_team_lookup, home_tid, game_dt,
                ["rolling_home_efg"],
            )
        feat["home_team_efg_home_split"] = home_vs.get("rolling_home_efg")

        away_vs = venue_split_lookup.get((gid, away_tid), {})
        if not away_vs and not pd.isna(game_dt):
            away_vs = _get_asof_rolling(
                venue_split_team_lookup, away_tid, game_dt,
                ["rolling_away_efg"],
            )
        feat["away_team_efg_away_split"] = away_vs.get("rolling_away_efg")

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

    # Verify we have all features in FEATURE_ORDER
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
        feature_order: Custom feature order. Defaults to config.FEATURE_ORDER.

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

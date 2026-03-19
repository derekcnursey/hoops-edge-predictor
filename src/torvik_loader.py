"""Load Torvik efficiency ratings from S3 for use in the feature pipeline.

Reads daily_data from s3://hoops-edge/silver/torvik/daily_data/season={YYYY}/
and translates team names using artifacts/team_name_mapping.json.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import timedelta
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from . import config, s3_reader

logger = logging.getLogger(__name__)

# Cache: season -> DataFrame
_torvik_cache: dict[int, pd.DataFrame] = {}

# Team name mapping: gold team name -> Torvik team name
_MAPPING_PATH = config.ARTIFACTS_DIR / "team_name_mapping.json"
_gold_to_torvik: dict[str, str] = {}
_torvik_to_gold: dict[str, str] = {}

# season -> (teamId -> gold team name) loaded from fct_games
_teamid_to_name_by_season: dict[int, dict[int, str]] = {}


def _load_mapping() -> None:
    """Load the team name mapping from artifacts."""
    global _gold_to_torvik, _torvik_to_gold
    if _gold_to_torvik:
        return
    with open(_MAPPING_PATH) as f:
        data = json.load(f)
    _gold_to_torvik = data.get("gold_to_torvik", {})
    _torvik_to_gold = data.get("torvik_to_gold", {})


def _gold_name_to_torvik(gold_name: str) -> str:
    """Convert a gold-layer team name to Torvik team name."""
    _load_mapping()
    return _gold_to_torvik.get(gold_name, gold_name)


def _load_teamid_mapping(season: int) -> None:
    """Build teamId -> team name mapping from fct_games for a season."""
    if season in _teamid_to_name_by_season:
        return
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if tbl.num_rows == 0:
        return
    df = tbl.to_pandas()
    mapping: dict[int, str] = {}
    for col_id, col_name in [("homeTeamId", "homeTeam"), ("awayTeamId", "awayTeam")]:
        if col_id in df.columns and col_name in df.columns:
            for tid, name in zip(df[col_id], df[col_name]):
                if pd.notna(tid) and pd.notna(name):
                    mapping[int(tid)] = str(name)
    if mapping:
        _teamid_to_name_by_season[season] = mapping


def load_torvik_season(season: int) -> pd.DataFrame:
    """Load and cache Torvik daily_data for a season from S3.

    Returns DataFrame with columns: team_name, date, adj_oe, adj_de,
    adj_pace, BARTHAG, conference, sorted by (team_name, date).
    """
    if season in _torvik_cache:
        return _torvik_cache[season]

    prefix = f"{config.SILVER_PREFIX}/torvik/daily_data/season={season}/"
    keys = s3_reader.list_parquet_keys(prefix)
    if not keys:
        logger.warning("No Torvik daily_data for season %d", season)
        _torvik_cache[season] = pd.DataFrame()
        return _torvik_cache[season]

    tbl = s3_reader.read_parquet_table(keys)
    df = tbl.to_pandas()

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["team_name", "date"]).reset_index(drop=True)

    _torvik_cache[season] = df
    logger.info("Loaded Torvik daily_data for season %d: %d rows, %d teams",
                season, len(df), df["team_name"].nunique())
    return df


def build_torvik_efficiency_lookup(
    season: int,
) -> dict[str, pd.DataFrame]:
    """Build per-team lookup table from Torvik daily ratings.

    Returns:
        Dict mapping Torvik team_name -> DataFrame sorted by date with columns
        date, adj_oe, adj_de, adj_pace, BARTHAG, conference.
    """
    df = load_torvik_season(season)
    if df.empty:
        return {}

    keep_cols = ["date", "adj_oe", "adj_de", "adj_pace", "BARTHAG", "conference"]
    available = [c for c in keep_cols if c in df.columns]

    lookup: dict[str, pd.DataFrame] = {}
    for name, group in df.groupby("team_name"):
        lookup[str(name)] = group[available].copy()
    return lookup


def get_torvik_asof_rating(
    torvik_lookup: dict[str, pd.DataFrame],
    team_id: int,
    game_date: pd.Timestamp,
    season: int,
) -> dict:
    """Look up a team's Torvik ratings as of the day before game_date.

    Translates teamId -> gold team name -> Torvik team name, then does
    an as-of lookup for the most recent rating strictly before game_date.

    Returns dict with keys: adj_oe, adj_de, adj_tempo, barthag.
    Returns empty dict if no data found.
    """
    _load_teamid_mapping(season)
    gold_name = _teamid_to_name_by_season.get(season, {}).get(team_id)
    if gold_name is None:
        return {}

    torvik_name = _gold_name_to_torvik(gold_name)
    team_df = torvik_lookup.get(torvik_name)
    if team_df is None or team_df.empty:
        return {}

    # Normalize game_date to tz-naive for comparison
    if hasattr(game_date, 'tz') and game_date.tz is not None:
        game_date = game_date.tz_localize(None)
    cutoff = game_date.normalize() - timedelta(days=1)

    dates = team_df["date"]
    if hasattr(dates.dtype, 'tz') and dates.dtype.tz is not None:
        dates = dates.dt.tz_localize(None)

    eligible = team_df[dates <= cutoff]
    if eligible.empty:
        return {}

    row = eligible.iloc[-1]
    return {
        "adj_oe": row.get("adj_oe"),
        "adj_de": row.get("adj_de"),
        "adj_tempo": row.get("adj_pace"),
        "barthag": row.get("BARTHAG"),
    }


def build_torvik_conf_strength_lookup(
    season: int,
    game_dates: list[pd.Timestamp],
) -> dict[tuple[str, str], float]:
    """Build conference strength lookup from Torvik ratings.

    For each unique game date, computes mean adj_net per conference using
    the most recent Torvik ratings before the game date.

    Returns:
        Dict mapping (date_str, conference) -> avg_adj_net.
    """
    df = load_torvik_season(season)
    if df.empty:
        return {}

    df = df.copy()
    df["adj_net"] = df["adj_oe"] - df["adj_de"]

    lookup: dict[tuple[str, str], float] = {}
    for game_dt in game_dates:
        if pd.isna(game_dt):
            continue
        dt = pd.Timestamp(game_dt)
        if hasattr(dt, 'tz') and dt.tz is not None:
            dt = dt.tz_localize(None)
        cutoff = dt.normalize() - timedelta(days=1)

        dates = df["date"]
        if hasattr(dates.dtype, 'tz') and dates.dtype.tz is not None:
            dates = dates.dt.tz_localize(None)

        eligible = df[dates <= cutoff]
        if eligible.empty:
            continue

        # Get latest rating per team
        latest = eligible.sort_values("date").groupby("team_name").last()
        if "conference" not in latest.columns:
            continue
        conf_means = latest.groupby("conference")["adj_net"].mean()
        date_str = cutoff.strftime("%Y-%m-%d")
        for conf, val in conf_means.items():
            lookup[(date_str, str(conf))] = float(val)

    return lookup


def build_torvik_sos_lookup(
    season: int,
    games: pd.DataFrame,
) -> dict[tuple[int, str], dict[str, float]]:
    """Compute SOS from Torvik ratings: average opponent adj_oe/adj_de.

    For each (teamId, game_date), computes the average Torvik adj_oe and adj_de
    of all opponents faced up to that date.

    Returns:
        Dict mapping (teamId, date_str) -> {"sos_oe": float, "sos_de": float}.
    """
    df = load_torvik_season(season)
    if df.empty or games.empty:
        return {}

    _load_teamid_mapping(season)
    _load_mapping()

    # Build schedule: for each team, list of (game_date, opponent_id) in order
    game_dates = pd.to_datetime(games["startDate"], errors="coerce", utc=True)
    schedule: dict[int, list[tuple[pd.Timestamp, int]]] = {}
    for _, g in games.iterrows():
        dt = game_dates[g.name]
        if pd.isna(dt):
            continue
        home_tid = int(g["homeTeamId"])
        away_tid = int(g["awayTeamId"])
        schedule.setdefault(home_tid, []).append((dt, away_tid))
        schedule.setdefault(away_tid, []).append((dt, home_tid))

    # Sort each team's schedule
    for tid in schedule:
        schedule[tid].sort(key=lambda x: x[0])

    # Build Torvik lookup keyed by Torvik team name
    torvik_by_name: dict[str, pd.DataFrame] = {}
    for name, group in df.groupby("team_name"):
        torvik_by_name[str(name)] = group.sort_values("date").copy()

    def _get_torvik_rating(team_id: int, cutoff: pd.Timestamp) -> Optional[tuple[float, float]]:
        gold_name = _teamid_to_name_by_season.get(season, {}).get(team_id)
        if gold_name is None:
            return None
        torvik_name = _gold_name_to_torvik(gold_name)
        tdf = torvik_by_name.get(torvik_name)
        if tdf is None or tdf.empty:
            return None
        dates = tdf["date"]
        if hasattr(dates.dtype, 'tz') and dates.dtype.tz is not None:
            dates = dates.dt.tz_localize(None)
        eligible = tdf[dates <= cutoff]
        if eligible.empty:
            return None
        row = eligible.iloc[-1]
        oe = row.get("adj_oe")
        de = row.get("adj_de")
        if pd.isna(oe) or pd.isna(de):
            return None
        return (float(oe), float(de))

    result: dict[tuple[int, str], dict[str, float]] = {}
    for tid, sched in schedule.items():
        opp_oes: list[float] = []
        opp_des: list[float] = []
        for game_dt, opp_id in sched:
            if hasattr(game_dt, 'tz') and game_dt.tz is not None:
                game_dt_naive = game_dt.tz_localize(None)
            else:
                game_dt_naive = game_dt
            cutoff = game_dt_naive.normalize() - timedelta(days=1)
            date_str = cutoff.strftime("%Y-%m-%d")

            # Get opponent's rating as of cutoff
            opp_rating = _get_torvik_rating(opp_id, cutoff)
            if opp_rating is not None:
                opp_oes.append(opp_rating[0])
                opp_des.append(opp_rating[1])

            if opp_oes:
                result[(tid, date_str)] = {
                    "sos_oe": sum(opp_oes) / len(opp_oes),
                    "sos_de": sum(opp_des) / len(opp_des),
                }

    return result


def get_torvik_team_conference(
    team_id: int,
    season: int,
) -> Optional[str]:
    """Get conference for a team from Torvik data."""
    _load_teamid_mapping(season)
    _load_mapping()

    gold_name = _teamid_to_name_by_season.get(season, {}).get(team_id)
    if gold_name is None:
        return None

    torvik_name = _gold_name_to_torvik(gold_name)
    df = load_torvik_season(season)
    if df.empty:
        return None

    team_rows = df[df["team_name"] == torvik_name]
    if team_rows.empty:
        return None

    return str(team_rows.iloc[-1]["conference"])

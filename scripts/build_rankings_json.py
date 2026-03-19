"""Build power rankings JSON from S3 efficiency ratings and game results.

Reads:
  - preferred: gold/<config.PRODUCTION_GOLD_RATINGS_TABLE>
  - fallback:  gold/team_adjusted_efficiencies_no_garbage
  - silver/fct_games (season 2026) → win-loss records
  - silver/fct_games team names → display names + conference fallback

Outputs:
  - predictions/json/rankings.json
  - site/public/data/rankings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as standalone script
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config, s3_reader
from src.adjusted_four_factors import adjust_four_factors
from src.features import _dedupe_boxscores, _dedupe_efficiency_ratings
from src.four_factors import compute_game_four_factors
from src.trainer import load_scaler, load_tree_regressor


CURRENT_SEASON = 2026
PRIMARY_RATINGS_TABLE = config.PRODUCTION_GOLD_RATINGS_TABLE
FALLBACK_RATINGS_TABLE = "team_adjusted_efficiencies_no_garbage"
PRIMARY_SOURCE_LABEL = "Hoops Edge Ratings"
PRIMARY_SOURCE_DESCRIPTION = (
    "Best internal efficiency model. Strongest in post-Dec-15 validation."
)
PRIMARY_SOURCE_NOTE = (
    "Internal ratings source only. Torvik remains stronger on full-season pooled validation."
)
MODEL_INDEX_LABEL = "DCN INDEX"
MODEL_INDEX_DESCRIPTION = (
    "Projected neutral-court spread vs an average D-I team from the current LightGBM mean model."
)

RANK_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("adj_oe", False),
    ("adj_de", True),
    ("adj_margin", False),
    ("adj_tempo", False),
    ("three_p_pct", False),
    ("def_3p_pct", True),
    ("ft_pct", False),
    ("model_index", False),
)


def _normalize_public_tempo(adj_tempo: float) -> float:
    """Map obviously inflated pace values back into a public-facing D-I range."""
    tempo = float(adj_tempo)
    while tempo > 85:
        tempo /= 2.0
    return min(max(tempo, 45.0), 85.0)


def _add_metric_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add public-facing category ranks for the exported team metrics."""
    ranked = df.copy()
    for column, ascending in RANK_COLUMNS:
        if column not in ranked.columns:
            ranked[f"{column}_rank"] = pd.NA
            continue
        values = pd.to_numeric(ranked[column], errors="coerce")
        ranked[f"{column}_rank"] = (
            values.rank(method="min", ascending=ascending, na_option="bottom").astype("Int64")
        )
    return ranked


def _load_latest_ratings(season: int) -> tuple[pd.DataFrame, str]:
    """Load the most recent efficiency rating for each team."""
    table_name = PRIMARY_RATINGS_TABLE
    tbl = s3_reader.read_gold_table(table_name, season=season)
    if tbl.num_rows == 0:
        print(
            f"WARNING: rankings season {season} missing preferred ratings table "
            f"{PRIMARY_RATINGS_TABLE}; falling back to {FALLBACK_RATINGS_TABLE}"
        )
        table_name = FALLBACK_RATINGS_TABLE
        tbl = s3_reader.read_gold_table(table_name, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame(), table_name
    df = tbl.to_pandas()

    needed = ["teamId", "rating_date", "adj_oe", "adj_de", "adj_tempo", "barthag"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in efficiency ratings: {missing}")

    df["rating_date"] = pd.to_datetime(df["rating_date"], errors="coerce")
    df = _dedupe_efficiency_ratings(df)
    # Keep only the latest rating_date row per team
    idx = df.groupby("teamId")["rating_date"].idxmax()
    latest = df.loc[idx].copy()
    return latest, table_name


def _compute_model_index(ratings: pd.DataFrame) -> pd.DataFrame:
    """Score each team as a neutral-court matchup vs an average D-I team.

    Uses the current production tree mean model with a symmetric home/away
    construction to reduce slot bias:
      1. team as home vs average away on a neutral floor
      2. average home vs team as away on a neutral floor
      index = (pred_home_team - pred_home_average) / 2
    """
    if ratings.empty:
        return pd.DataFrame(columns=["teamId", "model_index"])

    try:
        model, feature_order, _ = load_tree_regressor()
        scaler = load_scaler()
    except Exception as exc:
        print(f"WARNING: rankings model index unavailable; tree-model artifacts missing ({exc})")
        return pd.DataFrame(columns=["teamId", "model_index"])

    if len(feature_order) != len(scaler.mean_):
        print("WARNING: rankings model index unavailable; feature contract mismatch")
        return pd.DataFrame(columns=["teamId", "model_index"])

    base = {name: float(scaler.mean_[i]) for i, name in enumerate(feature_order)}
    avg_rest = float(
        np.mean([base.get("home_rest_days", 5.0), base.get("away_rest_days", 5.0)])
    )
    base["neutral_site"] = 1.0
    base["home_team_hca"] = 0.0
    base["home_rest_days"] = avg_rest
    base["away_rest_days"] = avg_rest
    base["rest_advantage"] = 0.0

    avg_eff = {
        "adj_oe": float(ratings["adj_oe"].mean()),
        "adj_de": float(ratings["adj_de"].mean()),
        "adj_tempo": float(ratings["adj_tempo"].mean()),
        "barthag": float(ratings["barthag"].mean()),
    }
    if "sos_oe" in ratings.columns:
        avg_eff["sos_oe"] = float(ratings["sos_oe"].mean())
    if "sos_de" in ratings.columns:
        avg_eff["sos_de"] = float(ratings["sos_de"].mean())

    def apply_home(vec: dict[str, float], row: pd.Series) -> None:
        vec["home_team_adj_oe"] = float(row["adj_oe"])
        vec["home_team_adj_de"] = float(row["adj_de"])
        vec["home_team_adj_pace"] = float(row["adj_tempo"])
        vec["home_team_BARTHAG"] = float(row["barthag"])
        if "home_sos_oe" in vec and "sos_oe" in row.index and pd.notna(row["sos_oe"]):
            vec["home_sos_oe"] = float(row["sos_oe"])
        if "home_sos_de" in vec and "sos_de" in row.index and pd.notna(row["sos_de"]):
            vec["home_sos_de"] = float(row["sos_de"])

    def apply_away(vec: dict[str, float], row: pd.Series) -> None:
        vec["away_team_adj_oe"] = float(row["adj_oe"])
        vec["away_team_adj_de"] = float(row["adj_de"])
        vec["away_team_adj_pace"] = float(row["adj_tempo"])
        vec["away_team_BARTHAG"] = float(row["barthag"])
        if "away_sos_oe" in vec and "sos_oe" in row.index and pd.notna(row["sos_oe"]):
            vec["away_sos_oe"] = float(row["sos_oe"])
        if "away_sos_de" in vec and "sos_de" in row.index and pd.notna(row["sos_de"]):
            vec["away_sos_de"] = float(row["sos_de"])

    def apply_avg_home(vec: dict[str, float]) -> None:
        vec["home_team_adj_oe"] = avg_eff["adj_oe"]
        vec["home_team_adj_de"] = avg_eff["adj_de"]
        vec["home_team_adj_pace"] = avg_eff["adj_tempo"]
        vec["home_team_BARTHAG"] = avg_eff["barthag"]
        if "home_sos_oe" in vec and "sos_oe" in avg_eff:
            vec["home_sos_oe"] = avg_eff["sos_oe"]
        if "home_sos_de" in vec and "sos_de" in avg_eff:
            vec["home_sos_de"] = avg_eff["sos_de"]

    def apply_avg_away(vec: dict[str, float]) -> None:
        vec["away_team_adj_oe"] = avg_eff["adj_oe"]
        vec["away_team_adj_de"] = avg_eff["adj_de"]
        vec["away_team_adj_pace"] = avg_eff["adj_tempo"]
        vec["away_team_BARTHAG"] = avg_eff["barthag"]
        if "away_sos_oe" in vec and "sos_oe" in avg_eff:
            vec["away_sos_oe"] = avg_eff["sos_oe"]
        if "away_sos_de" in vec and "sos_de" in avg_eff:
            vec["away_sos_de"] = avg_eff["sos_de"]

    def neutralize_context_terms(vec: dict[str, float]) -> None:
        """Keep DCN focused on neutral team strength, not schedule context."""
        for name in (
            "home_sos_oe",
            "away_sos_oe",
            "home_sos_de",
            "away_sos_de",
            "home_conf_strength",
            "away_conf_strength",
        ):
            if name in vec and name in base:
                vec[name] = float(base[name])

    rows: list[dict[str, float]] = []
    for _, row in ratings.iterrows():
        home_vec = dict(base)
        away_vec = dict(base)
        apply_home(home_vec, row)
        apply_avg_away(home_vec)
        apply_avg_home(away_vec)
        apply_away(away_vec, row)
        neutralize_context_terms(home_vec)
        neutralize_context_terms(away_vec)

        X = pd.DataFrame([home_vec, away_vec], columns=feature_order)
        preds = model.predict(X.values.astype(np.float32))
        model_index = float((preds[0] - preds[1]) / 2.0)
        rows.append({"teamId": int(row["teamId"]), "model_index": model_index})

    return pd.DataFrame(rows)


def _load_records(season: int) -> pd.DataFrame:
    """Compute W-L and conference W-L from fct_games."""
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame()
    games = tbl.to_pandas()

    # Normalize columns
    rename = {}
    for target, candidates in [
        ("gameId", ["gameId"]),
        ("homeTeamId", ["homeTeamId"]),
        ("awayTeamId", ["awayTeamId"]),
        ("homeScore", ["homeScore", "homePoints"]),
        ("awayScore", ["awayScore", "awayPoints"]),
        ("homeConference", ["homeConference", "homeConferenceName"]),
        ("awayConference", ["awayConference", "awayConferenceName"]),
        ("homeTeam", ["homeTeam"]),
        ("awayTeam", ["awayTeam"]),
    ]:
        for cand in candidates:
            if cand in games.columns:
                rename[cand] = target
                break
    games = games.rename(columns=rename)

    # Only completed games (drop missing scores and 0-0 placeholders for future games)
    games = games.dropna(subset=["homeScore", "awayScore"])
    games = games[~((games["homeScore"] == 0) & (games["awayScore"] == 0))]
    games = games.drop_duplicates(subset=["gameId"], keep="last")

    has_conf = "homeConference" in games.columns and "awayConference" in games.columns

    records: dict[int, dict] = {}

    for _, g in games.iterrows():
        h_id = int(g["homeTeamId"])
        a_id = int(g["awayTeamId"])
        h_score = float(g["homeScore"])
        a_score = float(g["awayScore"])
        home_won = h_score > a_score

        is_conf_game = False
        if has_conf:
            h_conf = g.get("homeConference")
            a_conf = g.get("awayConference")
            if pd.notna(h_conf) and pd.notna(a_conf) and h_conf == a_conf:
                is_conf_game = True

        for tid, won in [(h_id, home_won), (a_id, not home_won)]:
            if tid not in records:
                records[tid] = {"W": 0, "L": 0, "conf_W": 0, "conf_L": 0}
            if won:
                records[tid]["W"] += 1
            else:
                records[tid]["L"] += 1
            if is_conf_game:
                if won:
                    records[tid]["conf_W"] += 1
                else:
                    records[tid]["conf_L"] += 1

    rec_df = pd.DataFrame.from_dict(records, orient="index")
    rec_df.index.name = "teamId"
    rec_df = rec_df.reset_index()
    return rec_df


def _load_team_info(season: int) -> pd.DataFrame:
    """Load team display names and conferences from fct_games."""
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame()
    games = tbl.to_pandas()

    rename = {}
    for target, candidates in [
        ("homeTeamId", ["homeTeamId"]),
        ("awayTeamId", ["awayTeamId"]),
        ("homeTeam", ["homeTeam"]),
        ("awayTeam", ["awayTeam"]),
        ("homeConference", ["homeConference", "homeConferenceName"]),
        ("awayConference", ["awayConference", "awayConferenceName"]),
    ]:
        for cand in candidates:
            if cand in games.columns:
                rename[cand] = target
                break
    games = games.rename(columns=rename)

    teams: dict[int, dict] = {}

    # Collect from home side
    if "homeTeamId" in games.columns and "homeTeam" in games.columns:
        for _, row in games.iterrows():
            tid = int(row["homeTeamId"])
            name = str(row["homeTeam"])
            conf = str(row.get("homeConference", "")) if pd.notna(row.get("homeConference")) else ""
            if tid not in teams or not teams[tid].get("conference"):
                teams[tid] = {"team": name, "conference": conf}

    # Collect from away side
    if "awayTeamId" in games.columns and "awayTeam" in games.columns:
        for _, row in games.iterrows():
            tid = int(row["awayTeamId"])
            name = str(row["awayTeam"])
            conf = str(row.get("awayConference", "")) if pd.notna(row.get("awayConference")) else ""
            if tid not in teams or not teams[tid].get("conference"):
                teams[tid] = {"team": name, "conference": conf}

    info_df = pd.DataFrame.from_dict(teams, orient="index")
    info_df.index.name = "teamId"
    info_df = info_df.reset_index()
    return info_df


def _compute_team_shooting_snapshot(season: int) -> pd.DataFrame:
    """Compute current FT%, adjusted 3PT%, and adjusted defensive 3PT% snapshots per team.

    FT% is intentionally opponent-independent in this codebase and is carried
    through from the raw four-factor computation. 3PT% is opponent-adjusted
    using the production four-factor settings, then summarized with the same
    EWM span used elsewhere in the feature pipeline.
    """
    box_cols = [
        "gameid",
        "teamid",
        "opponentid",
        "ishometeam",
        "startdate",
        "team_fg_made",
        "team_fg_att",
        "team_3fg_made",
        "team_3fg_att",
        "team_ft_made",
        "team_ft_att",
        "team_reb_off",
        "team_reb_def",
        "opp_fg_made",
        "opp_fg_att",
        "opp_3fg_made",
        "opp_3fg_att",
        "opp_ft_made",
        "opp_ft_att",
        "opp_reb_off",
        "opp_reb_def",
    ]
    base = f"{config.SILVER_PREFIX}/{config.TABLE_FCT_GAME_TEAMS}/season={season}/"
    keys = s3_reader.list_parquet_keys(base)
    if not keys:
        return pd.DataFrame(columns=["teamId", "ft_pct", "three_p_pct", "def_3p_pct"])
    box_tbl = s3_reader.read_parquet_table(keys, columns=box_cols)
    box = _dedupe_boxscores(box_tbl.to_pandas())
    if box.empty:
        return pd.DataFrame(columns=["teamId", "ft_pct", "three_p_pct", "def_3p_pct"])

    ff = compute_game_four_factors(box)
    if ff.empty:
        return pd.DataFrame(columns=["teamId", "ft_pct", "three_p_pct", "def_3p_pct"])

    adjusted = (
        adjust_four_factors(
            ff,
            prior_weight=config.ADJUST_PRIOR,
            alpha=config.ADJUST_ALPHA,
        )
        if config.ADJUST_FF
        else ff.copy()
    )
    adjusted["_date"] = pd.to_datetime(adjusted["startdate"], errors="coerce")
    adjusted = adjusted.sort_values(["teamid", "_date", "gameid"]).reset_index(drop=True)

    rows: list[dict[str, float | int | None]] = []
    for team_id, group in adjusted.groupby("teamid", sort=False):
        ft_series = pd.to_numeric(group["ft_pct"], errors="coerce").dropna()
        three_series = pd.to_numeric(group["three_p_pct"], errors="coerce").dropna()
        def_three_series = pd.to_numeric(group["def_3p_pct"], errors="coerce").dropna()

        ft_pct = (
            float(ft_series.ewm(span=config.EWM_SPAN, min_periods=1).mean().iloc[-1])
            if not ft_series.empty
            else None
        )
        three_p_pct = (
            float(three_series.ewm(span=config.EWM_SPAN, min_periods=1).mean().iloc[-1])
            if not three_series.empty
            else None
        )
        def_3p_pct = (
            float(def_three_series.ewm(span=config.EWM_SPAN, min_periods=1).mean().iloc[-1])
            if not def_three_series.empty
            else None
        )
        rows.append(
            {
                "teamId": int(team_id),
                "ft_pct": ft_pct,
                "three_p_pct": three_p_pct,
                "def_3p_pct": def_3p_pct,
            }
        )

    return pd.DataFrame(rows)


def build_rankings(season: int = CURRENT_SEASON) -> dict:
    """Build the full rankings payload."""
    print(f"Loading efficiency ratings for season {season}...")
    ratings, table_name = _load_latest_ratings(season)
    if ratings.empty:
        raise RuntimeError("No efficiency ratings found.")

    print(f"  {len(ratings)} teams with ratings")

    print("Computing model index...")
    model_index = _compute_model_index(ratings)
    if not model_index.empty:
        print(f"  {len(model_index)} teams scored with the promoted tree mean model")

    print("Loading game records...")
    records = _load_records(season)
    print(f"  {len(records)} teams with records")

    print("Loading team info...")
    team_info = _load_team_info(season)
    print(f"  {len(team_info)} teams with info")

    print("Computing shooting snapshots...")
    shooting = _compute_team_shooting_snapshot(season)
    print(f"  {len(shooting)} teams with FT% / adjusted 3PT% / adjusted def 3PT% snapshots")

    # Merge ratings + records + team info
    rating_cols = ["teamId", "rating_date", "adj_oe", "adj_de", "adj_tempo", "barthag"]
    for optional_col in ["ft_pct", "three_p_pct", "def_3p_pct"]:
        if optional_col in ratings.columns:
            rating_cols.append(optional_col)
    df = ratings[rating_cols].copy()
    df["adj_margin"] = df["adj_oe"] - df["adj_de"]
    if not model_index.empty:
        df = df.merge(model_index, on="teamId", how="left")
    else:
        df["model_index"] = pd.NA

    if not records.empty:
        df = df.merge(records, on="teamId", how="left")
    else:
        df["W"] = 0
        df["L"] = 0
        df["conf_W"] = 0
        df["conf_L"] = 0

    if not team_info.empty:
        df = df.merge(team_info, on="teamId", how="left")
    else:
        df["team"] = df["teamId"].astype(str)
        df["conference"] = ""

    if not shooting.empty:
        df = df.merge(
            shooting,
            on="teamId",
            how="left",
            suffixes=("", "_snapshot"),
        )
        for col in ["ft_pct", "three_p_pct", "def_3p_pct"]:
            snapshot_col = f"{col}_snapshot"
            if snapshot_col in df.columns:
                if col in df.columns:
                    df[col] = df[col].where(df[col].notna(), df[snapshot_col])
                    df = df.drop(columns=[snapshot_col])
                else:
                    df = df.rename(columns={snapshot_col: col})

    # Fill NaN records
    for col in ["W", "L", "conf_W", "conf_L"]:
        df[col] = df[col].fillna(0).astype(int)
    df["team"] = df["team"].fillna(df["teamId"].astype(str))
    df["conference"] = df["conference"].fillna("")

    # Sort by model-driven neutral spread vs average team when available.
    if "barthag" in df.columns:
        df = df.sort_values(["adj_margin", "barthag"], ascending=[False, False]).reset_index(drop=True)
    else:
        df = df.sort_values("adj_margin", ascending=False).reset_index(drop=True)
    df = _add_metric_ranks(df)

    # Determine as_of_date from the latest rating_date
    as_of = df["rating_date"].max()
    as_of_str = as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else ""

    # Build output
    teams = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        adj_oe = round(float(row["adj_oe"]), 1)
        adj_de = round(float(row["adj_de"]), 1)
        adj_margin = round(float(row["adj_margin"]), 1)
        adj_tempo = round(_normalize_public_tempo(float(row["adj_tempo"])), 1)
        model_index_value = (
            round(float(row["model_index"]), 2)
            if pd.notna(row["model_index"])
            else None
        )
        ft_pct_value = (
            round(float(row["ft_pct"]), 3)
            if "ft_pct" in row.index and pd.notna(row["ft_pct"])
            else None
        )
        three_p_pct_value = (
            round(float(row["three_p_pct"]), 3)
            if "three_p_pct" in row.index and pd.notna(row["three_p_pct"])
            else None
        )
        def_3p_pct_value = (
            round(float(row["def_3p_pct"]), 3)
            if "def_3p_pct" in row.index and pd.notna(row["def_3p_pct"])
            else None
        )
        adj_oe_rank = int(row["adj_oe_rank"]) if pd.notna(row.get("adj_oe_rank")) else None
        adj_de_rank = int(row["adj_de_rank"]) if pd.notna(row.get("adj_de_rank")) else None
        adj_margin_rank = int(row["adj_margin_rank"]) if pd.notna(row.get("adj_margin_rank")) else None
        adj_tempo_rank = int(row["adj_tempo_rank"]) if pd.notna(row.get("adj_tempo_rank")) else None
        three_p_pct_rank = int(row["three_p_pct_rank"]) if pd.notna(row.get("three_p_pct_rank")) else None
        def_3p_pct_rank = int(row["def_3p_pct_rank"]) if pd.notna(row.get("def_3p_pct_rank")) else None
        ft_pct_rank = int(row["ft_pct_rank"]) if pd.notna(row.get("ft_pct_rank")) else None
        model_index_rank = int(row["model_index_rank"]) if pd.notna(row.get("model_index_rank")) else None

        record = f"{row['W']}-{row['L']}"
        conf_record = f"{row['conf_W']}-{row['conf_L']}"

        teams.append({
            "rank": rank,
            "team": str(row["team"]),
            "team_id": int(row["teamId"]),
            "conference": str(row["conference"]),
            "record": record,
            "conf_record": conf_record,
            "adj_oe": adj_oe,
            "adj_de": adj_de,
            "adj_margin": adj_margin,
            "adj_tempo": adj_tempo,
            "model_index": model_index_value,
            "ft_pct": ft_pct_value,
            "three_p_pct": three_p_pct_value,
            "def_3p_pct": def_3p_pct_value,
            "adj_oe_rank": adj_oe_rank,
            "adj_de_rank": adj_de_rank,
            "adj_margin_rank": adj_margin_rank,
            "adj_tempo_rank": adj_tempo_rank,
            "three_p_pct_rank": three_p_pct_rank,
            "def_3p_pct_rank": def_3p_pct_rank,
            "ft_pct_rank": ft_pct_rank,
            "model_index_rank": model_index_rank,
        })

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "as_of_date": as_of_str,
        "season": season,
        "source_id": PRIMARY_SOURCE_LABEL.lower().replace(" ", "_"),
        "source_label": PRIMARY_SOURCE_LABEL,
        "source_table": table_name,
        "source_description": PRIMARY_SOURCE_DESCRIPTION,
        "source_note": PRIMARY_SOURCE_NOTE,
        "model_index_label": MODEL_INDEX_LABEL,
        "model_index_description": MODEL_INDEX_DESCRIPTION,
        "teams": teams,
    }
    return payload


def save_rankings(payload: dict, season: int = CURRENT_SEASON) -> list[Path]:
    """Write rankings JSON to both output locations.

    Writes rankings_{season}.json. When season == CURRENT_SEASON also
    writes rankings.json as a backward-compatible alias.
    """
    json_dir = config.PREDICTIONS_DIR / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    site_dir = config.PROJECT_ROOT / "site" / "public" / "data"
    site_dir.mkdir(parents=True, exist_ok=True)

    blob = json.dumps(payload, indent=2)
    written: list[Path] = []

    # Always write the season-specific file
    for base_dir in (json_dir, site_dir):
        p = base_dir / f"rankings_{season}.json"
        p.write_text(blob)
        written.append(p)

    # Also write the generic alias for the current season
    if season == CURRENT_SEASON:
        for base_dir in (json_dir, site_dir):
            p = base_dir / "rankings.json"
            p.write_text(blob)
            written.append(p)

    return written


def main():
    parser = argparse.ArgumentParser(description="Build power rankings JSON")
    parser.add_argument(
        "--season",
        type=int,
        default=CURRENT_SEASON,
        help=f"Season year (default: {CURRENT_SEASON})",
    )
    args = parser.parse_args()

    payload = build_rankings(season=args.season)
    written = save_rankings(payload, season=args.season)
    print(f"\nRankings built: {len(payload['teams'])} teams (season {args.season})")
    for p in written:
        print(f"  {p}")

    # Print top 10
    print(f"\nTop 10 (as of {payload['as_of_date']}):")
    for t in payload["teams"][:10]:
        print(
            f"  {t['rank']:>3}. {t['team']:<22} "
            f"{t['record']:>6}  "
            f"Model: {t['model_index']:+.2f}  "
            f"Net: {t['adj_margin']:+.1f}  "
            f"O: {t['adj_oe']:.1f}  D: {t['adj_de']:.1f}  "
            f"Tempo: {t['adj_tempo']:.1f}"
        )


if __name__ == "__main__":
    main()

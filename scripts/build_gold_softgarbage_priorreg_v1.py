#!/usr/bin/env python3
"""Build research-only soft-garbage prior-regularized gold tables to S3."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
ETL_ROOT = WORKSPACE_ROOT / "hoops_edge_database_etl_codex"
ETL_SRC = ETL_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ETL_SRC))

from src import config as predictor_config
from src.efficiency_research import (
    SoftGarbageConfig,
    blend_soft_garbage_team_games,
    select_min_possession_team_game_rows,
)

from cbbd_etl.config import load_config
from cbbd_etl.gold._io_helpers import dedup_by, pydict_get, read_silver_table
from cbbd_etl.gold.adjusted_efficiencies import (
    _apply_margin_cap,
    _get_margin_cap,
    _get_preseason_regression,
    _get_rating_params,
    _get_solver_type,
    _get_wls_params,
    _load_d1_team_ids,
    _load_preseason_prior,
    _load_team_info,
    _parse_date_str,
    _run_per_date_ratings,
)
from cbbd_etl.gold.iterative_ratings import GameObs
from cbbd_etl.normalize import normalize_records
from cbbd_etl.s3_io import S3IO, make_part_key


@dataclass
class BuildResult:
    season: int
    prior_k: float
    garbage_keep_fraction: float
    rows: int
    table_name: str
    s3_key: str


def _default_output_dir() -> Path:
    return predictor_config.ARTIFACTS_DIR / "efficiency_research" / "gold_softgarbage_priorreg_v1_build"


def _table_name(prior_k: float, garbage_keep_fraction: float) -> str:
    keep_pct = int(round(garbage_keep_fraction * 100))
    return f"team_adjusted_efficiencies_no_garbage_softkeep{keep_pct}_priorreg_k{int(prior_k)}_v1"


def _load_soft_garbage_games(
    s3: S3IO,
    cfg,
    season: int,
    d1_ids: set[int],
    soft_cfg: SoftGarbageConfig,
    raw_table_name: str = "fct_pbp_game_teams_flat",
    no_garbage_table_name: str = "fct_pbp_game_teams_flat_garbage_removed",
) -> dict[str, list[GameObs]]:
    """Blend raw and no-garbage PBP rows into a soft-attenuated game set."""
    fct_games = dedup_by(read_silver_table(s3, cfg, "fct_games", season=season), ["gameId"])
    game_neutral: dict[int, bool] = {}
    d1_game_ids: set[int] = set()

    if fct_games.num_rows > 0:
        g_ids = pydict_get(fct_games, "gameId")
        g_neutral = pydict_get(fct_games, "neutralSite")
        g_home = pydict_get(fct_games, "homeTeamId")
        g_away = pydict_get(fct_games, "awayTeamId")
        for i, gid in enumerate(g_ids):
            if gid is None:
                continue
            gid_int = int(gid)
            home_id = int(g_home[i]) if g_home[i] is not None else 0
            away_id = int(g_away[i]) if g_away[i] is not None else 0
            if home_id in d1_ids and away_id in d1_ids:
                d1_game_ids.add(gid_int)
                game_neutral[gid_int] = bool(g_neutral[i]) if g_neutral[i] is not None else False

    raw_tbl = read_silver_table(s3, cfg, raw_table_name, season=season)
    ng_tbl = read_silver_table(s3, cfg, no_garbage_table_name, season=season)
    if raw_tbl.num_rows == 0 or ng_tbl.num_rows == 0:
        return {}

    raw_df = select_min_possession_team_game_rows(raw_tbl.to_pandas())
    ng_df = select_min_possession_team_game_rows(ng_tbl.to_pandas())
    merged = blend_soft_garbage_team_games(raw_df, ng_df, soft_cfg)
    if merged.empty:
        return {}

    games_by_date: dict[str, list[GameObs]] = {}
    for row in merged.itertuples(index=False):
        gid = int(row.gameid)
        tid = int(row.teamid)
        if gid not in d1_game_ids:
            continue
        dt_str = _parse_date_str(row.startdate)
        if dt_str is None:
            continue

        team_poss = getattr(row, "team_possessions_formula", None)
        opp_poss = getattr(row, "opp_possessions_formula", None)
        team_pts = getattr(row, "team_points_total", None)
        opp_pts = getattr(row, "opp_points_total", None)

        if team_poss is None or team_poss <= 0 or team_pts is None:
            continue
        if opp_poss is None or opp_poss <= 0:
            opp_poss = team_poss
        if opp_pts is None:
            opp_pts = 0.0

        obs = GameObs(
            game_id=gid,
            team_id=tid,
            opp_id=int(row.opponentid) if getattr(row, "opponentid", None) is not None else 0,
            team_pts=float(team_pts),
            team_poss=float(team_poss),
            opp_pts=float(opp_pts),
            opp_poss=float(opp_poss),
            is_home=bool(row.ishometeam) if getattr(row, "ishometeam", None) is not None else False,
            is_neutral=game_neutral.get(gid, False),
            game_date=dt_str,
            weight=0.0,
        )
        games_by_date.setdefault(dt_str, []).append(obs)

    return games_by_date


def build_no_garbage_softgarbage_priorreg(
    cfg,
    season: int,
    *,
    prior_k: float,
    soft_cfg: SoftGarbageConfig,
    raw_table_name: str = "fct_pbp_game_teams_flat",
    no_garbage_table_name: str = "fct_pbp_game_teams_flat_garbage_removed",
):
    s3 = S3IO(cfg.bucket, cfg.region)
    params = _get_rating_params(cfg)
    margin_cap = _get_margin_cap(cfg)
    preseason_regression = _get_preseason_regression(cfg)
    solver_type = _get_solver_type(cfg)
    wls_params = _get_wls_params(cfg)

    d1_ids = _load_d1_team_ids(s3, cfg)
    team_info = _load_team_info(s3, cfg)
    games_by_date = _load_soft_garbage_games(
        s3,
        cfg,
        season,
        d1_ids,
        soft_cfg,
        raw_table_name=raw_table_name,
        no_garbage_table_name=no_garbage_table_name,
    )
    if not games_by_date:
        return normalize_records("team_adjusted_efficiencies_no_garbage", [])

    if margin_cap is not None and solver_type != "wls":
        games_by_date = _apply_margin_cap(games_by_date, margin_cap)

    preseason_prior = None
    if preseason_regression is not None:
        preseason_prior = _load_preseason_prior(
            s3,
            cfg,
            season,
            "team_adjusted_efficiencies_no_garbage",
            preseason_regression,
        )

    records = _run_per_date_ratings(
        games_by_date,
        team_info,
        season,
        preseason_prior=preseason_prior,
        persistent_prior=preseason_prior,
        persistent_prior_k=prior_k,
        solver_type=solver_type,
        margin_cap=margin_cap,
        **wls_params,
        **params,
    )
    if not records:
        return normalize_records("team_adjusted_efficiencies_no_garbage", [])
    return normalize_records("team_adjusted_efficiencies_no_garbage", records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build soft-garbage prior-regularized no-garbage gold tables.")
    parser.add_argument("--season-start", type=int, default=2015)
    parser.add_argument("--season-end", type=int, default=2025)
    parser.add_argument("--prior-k", type=float, default=5.0)
    parser.add_argument("--garbage-keep-fraction", type=float, default=0.25)
    parser.add_argument(
        "--etl-config",
        type=str,
        default=str(ETL_ROOT / "config.yaml"),
        help="Path to ETL config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.etl_config)
    cfg.raw.setdefault("gold", {}).setdefault("adjusted_efficiencies", {})
    cfg.raw["gold"]["adjusted_efficiencies"]["solver"] = "iterative"
    cfg.raw["gold"]["adjusted_efficiencies"]["sos_exponent"] = 0.85
    cfg.raw["gold"]["adjusted_efficiencies"]["preseason_regression"] = 0.30
    cfg.raw["gold"]["adjusted_efficiencies"]["half_life"] = None
    cfg.raw["gold"]["adjusted_efficiencies"]["shrinkage"] = 0.0

    soft_cfg = SoftGarbageConfig(garbage_keep_fraction=args.garbage_keep_fraction)
    s3 = S3IO(cfg.bucket, cfg.region)
    asof = datetime.now(timezone.utc).date().isoformat()
    table_name = _table_name(args.prior_k, args.garbage_keep_fraction)
    results: list[BuildResult] = []

    for season in range(args.season_start, args.season_end + 1):
        print(f"Building {table_name} for season {season}...", flush=True)
        table = build_no_garbage_softgarbage_priorreg(
            cfg,
            season,
            prior_k=args.prior_k,
            soft_cfg=soft_cfg,
        )
        if table.num_rows == 0:
            print(f"  -> empty table for season {season}", flush=True)
            continue
        s3_key = make_part_key(
            cfg.s3_layout["gold_prefix"],
            table_name,
            f"season={season}",
            f"asof={asof}",
            f"part-softkeep{int(round(args.garbage_keep_fraction * 100))}-k{int(args.prior_k)}.parquet",
        )
        s3.put_parquet(s3_key, table)
        print(f"  -> wrote {table.num_rows} rows to s3://{cfg.bucket}/{s3_key}", flush=True)
        results.append(
            BuildResult(
                season=season,
                prior_k=args.prior_k,
                garbage_keep_fraction=args.garbage_keep_fraction,
                rows=table.num_rows,
                table_name=table_name,
                s3_key=s3_key,
            )
        )

    protocol = {
        "season_start": args.season_start,
        "season_end": args.season_end,
        "prior_k": args.prior_k,
        "garbage_keep_fraction": args.garbage_keep_fraction,
        "fixed_params": {
            "solver": "iterative",
            "sos_exponent": 0.85,
            "preseason_regression": 0.30,
            "half_life": None,
            "shrinkage": 0.0,
            "branch": "soft garbage attenuation",
        },
        "table_name": table_name,
        "asof": asof,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (output_dir / "build_results.json").write_text(
        json.dumps([asdict(row) for row in results], indent=2)
    )


if __name__ == "__main__":
    main()

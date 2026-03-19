#!/usr/bin/env python3
"""Repair season-2025 no-garbage PBP rows with corrupted score/possession totals.

The 2025 no-garbage PBP silver table contains many team-game rows whose own
box-score totals are clearly corrupted relative to fct_games final scores. For
one-side-corrupted games, this script repairs both team rows using the actual
game score and the trustworthy side's possession count. If both team rows are
corrupted, the game is dropped from the repaired silver table rather than
guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
ETL_ROOT = WORKSPACE_ROOT / "hoops_edge_database_etl_codex"
ETL_SRC = ETL_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ETL_SRC))

from src import config as predictor_config, s3_reader
from cbbd_etl.config import load_config
from cbbd_etl.gold.adjusted_efficiencies import build_no_garbage, build_no_garbage_priorreg
from cbbd_etl.s3_io import S3IO, make_part_key


SOURCE_TABLE = "fct_pbp_game_teams_flat_garbage_removed"
REPAIRED_TABLE = "fct_pbp_game_teams_flat_garbage_removed_repaired_v1"
CURRENT_GOLD_TABLE = "team_adjusted_efficiencies_no_garbage"
PRIORREG_GOLD_TABLE = "team_adjusted_efficiencies_no_garbage_priorreg_k5_v1"
PRIOR_K = 5.0
POINT_DIFF_THRESHOLD = 20.0


@dataclass
class RepairAuditRow:
    gameid: int
    date: str
    repair_mode: str
    suspicious_rows: int
    home_teamid: int
    away_teamid: int
    shared_possessions_formula: float | None


def _default_output_dir() -> Path:
    return predictor_config.ARTIFACTS_DIR / "repairs" / "pbp_no_garbage_2025_v1"


def _load_inputs(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp = s3_reader.read_silver_table(SOURCE_TABLE, season=season).to_pandas()
    games = s3_reader.read_silver_table("fct_games", season=season).to_pandas()
    return pbp, games


def _actual_score_map(games: pd.DataFrame) -> tuple[dict[tuple[int, int], float], dict[int, dict[str, float]]]:
    team_scores: dict[tuple[int, int], float] = {}
    game_scores: dict[int, dict[str, float]] = {}
    for _, row in games.iterrows():
        gid = int(row["gameId"])
        home_tid = int(row["homeTeamId"])
        away_tid = int(row["awayTeamId"])
        home_pts = float(row["homePoints"])
        away_pts = float(row["awayPoints"])
        team_scores[(gid, home_tid)] = home_pts
        team_scores[(gid, away_tid)] = away_pts
        game_scores[gid] = {
            "home_tid": home_tid,
            "away_tid": away_tid,
            "home_pts": home_pts,
            "away_pts": away_pts,
        }
    return team_scores, game_scores


def _repair_rows(pbp: pd.DataFrame, games: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    team_scores, game_scores = _actual_score_map(games)
    pbp = pbp.copy()
    pbp["actual_points"] = pbp.apply(
        lambda r: team_scores.get((int(r["gameid"]), int(r["teamid"]))), axis=1
    )
    pbp["point_diff"] = pbp["team_points_total"] - pbp["actual_points"]
    pbp["is_suspicious"] = pbp["point_diff"].abs() >= POINT_DIFF_THRESHOLD

    repaired_groups: list[pd.DataFrame] = []
    audit_rows: list[RepairAuditRow] = []

    for gid, group in pbp.groupby("gameid", sort=False):
        grp = group.copy().reset_index(drop=True)
        susp_count = int(grp["is_suspicious"].sum())
        date = str(grp["startdate"].iloc[0])
        meta = game_scores.get(int(gid))
        if meta is None:
            continue

        if len(grp) != 2:
            # Preserve unusual shapes unchanged; this repair targets normal paired rows.
            repaired_groups.append(grp.drop(columns=["actual_points", "point_diff", "is_suspicious"]))
            continue

        if susp_count == 0:
            repaired_groups.append(grp.drop(columns=["actual_points", "point_diff", "is_suspicious"]))
            continue

        if susp_count == 2:
            audit_rows.append(
                RepairAuditRow(
                    gameid=int(gid),
                    date=date,
                    repair_mode="drop_both_sides_corrupted",
                    suspicious_rows=susp_count,
                    home_teamid=int(meta["home_tid"]),
                    away_teamid=int(meta["away_tid"]),
                    shared_possessions_formula=None,
                )
            )
            continue

        good = grp.loc[~grp["is_suspicious"]].iloc[0]
        shared_formula = float(good["team_possessions_formula"])
        shared_possessions = float(good["team_possessions"])
        shared_pace = float(shared_formula * 40.0 / good["game_minutes"]) if float(good["game_minutes"]) > 0 else float(good["pace"])

        for idx in grp.index:
            tid = int(grp.at[idx, "teamid"])
            is_home = tid == int(meta["home_tid"])
            team_pts = float(meta["home_pts"] if is_home else meta["away_pts"])
            opp_pts = float(meta["away_pts"] if is_home else meta["home_pts"])
            grp.at[idx, "team_points_total"] = team_pts
            grp.at[idx, "opp_points_total"] = opp_pts
            grp.at[idx, "team_possessions_formula"] = shared_formula
            grp.at[idx, "opp_possessions_formula"] = shared_formula
            grp.at[idx, "team_possessions"] = shared_possessions
            grp.at[idx, "opp_possessions"] = shared_possessions
            grp.at[idx, "pace"] = shared_pace

        audit_rows.append(
            RepairAuditRow(
                gameid=int(gid),
                date=date,
                repair_mode="repair_from_clean_side_and_actual_score",
                suspicious_rows=susp_count,
                home_teamid=int(meta["home_tid"]),
                away_teamid=int(meta["away_tid"]),
                shared_possessions_formula=round(shared_formula, 4),
            )
        )
        repaired_groups.append(grp.drop(columns=["actual_points", "point_diff", "is_suspicious"]))

    repaired = pd.concat(repaired_groups, ignore_index=True)
    audit_df = pd.DataFrame([asdict(r) for r in audit_rows])
    audit_df.to_csv(output_dir / "repair_audit.csv", index=False)
    summary = {
        "source_table": SOURCE_TABLE,
        "repaired_table": REPAIRED_TABLE,
        "season": 2025,
        "rows_in": int(len(pbp)),
        "rows_out": int(len(repaired)),
        "games_repaired": int((audit_df["repair_mode"] == "repair_from_clean_side_and_actual_score").sum()) if not audit_df.empty else 0,
        "games_dropped": int((audit_df["repair_mode"] == "drop_both_sides_corrupted").sum()) if not audit_df.empty else 0,
        "point_diff_threshold": POINT_DIFF_THRESHOLD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "protocol.json").write_text(json.dumps(summary, indent=2))
    return repaired


def _write_repaired_silver(cfg, repaired: pd.DataFrame, season: int) -> None:
    s3 = S3IO(cfg.bucket, cfg.region)
    for date, group in repaired.groupby("startdate", sort=True):
        key = make_part_key(
            cfg.s3_layout["silver_prefix"],
            REPAIRED_TABLE,
            f"season={season}",
            f"date={date}",
            "part-repaired.parquet",
        )
        s3.put_parquet(key, pa.Table.from_pandas(group, preserve_index=False))


def _delete_existing_gold_parts(cfg, season: int, asof: str) -> None:
    client = boto3.client("s3", region_name=cfg.region)
    for table in [CURRENT_GOLD_TABLE, PRIORREG_GOLD_TABLE]:
        prefix = f"{cfg.s3_layout['gold_prefix']}/{table}/season={season}/asof={asof}/"
        paginator = client.get_paginator("list_objects_v2")
        to_delete: list[dict[str, str]] = []
        for page in paginator.paginate(Bucket=cfg.bucket, Prefix=prefix):
            to_delete.extend({"Key": obj["Key"]} for obj in page.get("Contents", []))
        if to_delete:
            for i in range(0, len(to_delete), 1000):
                client.delete_objects(Bucket=cfg.bucket, Delete={"Objects": to_delete[i:i + 1000]})


def _rebuild_gold(cfg, season: int, output_dir: Path) -> dict[str, str]:
    s3 = S3IO(cfg.bucket, cfg.region)
    asof = datetime.now(timezone.utc).date().isoformat()
    _delete_existing_gold_parts(cfg, season, asof)

    current_table = build_no_garbage(cfg, season, pbp_table_name=REPAIRED_TABLE)
    priorreg_table = build_no_garbage_priorreg(
        cfg, season, prior_k=PRIOR_K, pbp_table_name=REPAIRED_TABLE
    )

    current_key = make_part_key(
        cfg.s3_layout["gold_prefix"],
        CURRENT_GOLD_TABLE,
        f"season={season}",
        f"asof={asof}",
        "part-repaired-v1.parquet",
    )
    priorreg_key = make_part_key(
        cfg.s3_layout["gold_prefix"],
        PRIORREG_GOLD_TABLE,
        f"season={season}",
        f"asof={asof}",
        "part-repaired-v1.parquet",
    )
    s3.put_parquet(current_key, current_table)
    s3.put_parquet(priorreg_key, priorreg_table)

    result = {
        "current_gold_key": current_key,
        "priorreg_gold_key": priorreg_key,
        "current_rows": int(current_table.num_rows),
        "priorreg_rows": int(priorreg_table.num_rows),
    }
    (output_dir / "gold_rebuild.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair 2025 no-garbage PBP silver and rebuild gold ratings.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--etl-config",
        type=str,
        default=str(ETL_ROOT / "config.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.etl_config)
    cfg.raw.setdefault("gold", {}).setdefault("adjusted_efficiencies", {})
    cfg.raw["gold"]["adjusted_efficiencies"]["solver"] = "iterative"
    cfg.raw["gold"]["adjusted_efficiencies"]["sos_exponent"] = 0.85
    cfg.raw["gold"]["adjusted_efficiencies"]["preseason_regression"] = 0.30

    pbp, games = _load_inputs(args.season)
    repaired = _repair_rows(pbp, games, output_dir)
    _write_repaired_silver(cfg, repaired, args.season)
    rebuild = _rebuild_gold(cfg, args.season, output_dir)

    print(f"Repaired {SOURCE_TABLE} season {args.season} into {REPAIRED_TABLE}")
    print(f"  rows in={len(pbp)} rows out={len(repaired)}")
    print(f"  wrote {CURRENT_GOLD_TABLE} -> s3://{cfg.bucket}/{rebuild['current_gold_key']}")
    print(f"  wrote {PRIORREG_GOLD_TABLE} -> s3://{cfg.bucket}/{rebuild['priorreg_gold_key']}")


if __name__ == "__main__":
    main()

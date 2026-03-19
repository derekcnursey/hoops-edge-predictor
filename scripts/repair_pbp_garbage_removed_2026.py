#!/usr/bin/env python3
"""Repair season-2026 garbage-removed PBP team-game rows and rebuild gold ratings.

The broken 2026 silver table contains multiple per-date parquet parts with the
same team-game rows at different scale factors (roughly 1x/2x/3x/4x). This
script stages a repaired silver table by selecting the lowest-possession part
per date, then rebuilds the 2026 gold efficiency tables from that repaired
source.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
ETL_ROOT = WORKSPACE_ROOT / "hoops_edge_database_etl_codex"
ETL_SRC = ETL_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ETL_SRC))

from src import config as predictor_config
from cbbd_etl.config import load_config
from cbbd_etl.gold.adjusted_efficiencies import build_no_garbage, build_no_garbage_priorreg
from cbbd_etl.s3_io import S3IO, make_part_key


SOURCE_TABLE = "fct_pbp_game_teams_flat_garbage_removed"
REPAIRED_TABLE = "fct_pbp_game_teams_flat_garbage_removed_repaired_v1"
CURRENT_GOLD_TABLE = "team_adjusted_efficiencies_no_garbage"
PRIORREG_GOLD_TABLE = "team_adjusted_efficiencies_no_garbage_priorreg_k5_v1"
PRIOR_K = 5.0


@dataclass
class CandidatePart:
    date: str
    key: str
    rows: int
    median_poss: float
    median_pts: float
    selected: bool


def _default_output_dir() -> Path:
    return predictor_config.ARTIFACTS_DIR / "repairs" / "pbp_no_garbage_2026_v1"


def _s3_client(region: str):
    return boto3.client("s3", region_name=region)


def _list_date_prefixes(client, bucket: str, season: int) -> list[str]:
    base = f"silver/{SOURCE_TABLE}/season={season}/"
    paginator = client.get_paginator("list_objects_v2")
    prefixes: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=base, Delimiter="/"):
        prefixes.extend(p["Prefix"] for p in page.get("CommonPrefixes", []))
    return sorted(prefixes)


def _list_parquet_keys(client, bucket: str, prefix: str) -> list[str]:
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return sorted(o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet"))


def _read_parquet_bytes(client, bucket: str, key: str) -> pa.Table:
    obj = client.get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()))


def _candidate_metrics(tbl: pa.Table) -> tuple[int, float, float]:
    df = tbl.select(["team_possessions_formula", "team_points_total"]).to_pandas()
    return (
        len(df),
        float(df["team_possessions_formula"].median()),
        float(df["team_points_total"].median()),
    )


def _write_repaired_silver(
    cfg,
    season: int,
    output_dir: Path,
) -> list[CandidatePart]:
    client = _s3_client(cfg.region)
    s3 = S3IO(cfg.bucket, cfg.region)
    asof = datetime.now(timezone.utc).date().isoformat()
    audit_rows: list[CandidatePart] = []

    for prefix in _list_date_prefixes(client, cfg.bucket, season):
        date = prefix.split("date=")[1].rstrip("/")
        keys = _list_parquet_keys(client, cfg.bucket, prefix)
        candidates: list[tuple[CandidatePart, pa.Table]] = []
        for key in keys:
            tbl = _read_parquet_bytes(client, cfg.bucket, key)
            rows, median_poss, median_pts = _candidate_metrics(tbl)
            candidates.append(
                (
                    CandidatePart(
                        date=date,
                        key=key,
                        rows=rows,
                        median_poss=median_poss,
                        median_pts=median_pts,
                        selected=False,
                    ),
                    tbl,
                )
            )

        if not candidates:
            continue

        selected_idx = min(range(len(candidates)), key=lambda i: candidates[i][0].median_poss)
        for idx, (meta, _) in enumerate(candidates):
            meta.selected = idx == selected_idx
            audit_rows.append(meta)

        selected_tbl = candidates[selected_idx][1]
        repaired_key = make_part_key(
            cfg.s3_layout["silver_prefix"],
            REPAIRED_TABLE,
            f"season={season}",
            f"date={date}",
            "part-repaired.parquet",
        )
        s3.put_parquet(repaired_key, selected_tbl)

    audit_df = pd.DataFrame([asdict(r) for r in audit_rows])
    audit_df.to_csv(output_dir / "repair_audit.csv", index=False)
    selected_df = audit_df[audit_df["selected"]].copy()
    selected_df.to_csv(output_dir / "selected_parts.csv", index=False)

    protocol = {
        "season": season,
        "source_table": SOURCE_TABLE,
        "repaired_table": REPAIRED_TABLE,
        "selection_rule": "choose per-date parquet part with smallest median team_possessions_formula",
        "generated_at": asof,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return audit_rows


def _rebuild_gold(cfg, season: int, output_dir: Path) -> dict[str, str]:
    s3 = S3IO(cfg.bucket, cfg.region)
    asof = datetime.now(timezone.utc).date().isoformat()

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
    parser = argparse.ArgumentParser(description="Repair 2026 no-garbage PBP silver and rebuild gold ratings.")
    parser.add_argument("--season", type=int, default=2026)
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

    print(f"Repairing {SOURCE_TABLE} season {args.season} into {REPAIRED_TABLE}...")
    audit_rows = _write_repaired_silver(cfg, args.season, output_dir)
    selected_count = sum(1 for r in audit_rows if r.selected)
    print(f"  wrote repaired silver partitions for {selected_count} dates")

    print("Rebuilding 2026 gold efficiency tables from repaired silver...")
    rebuild = _rebuild_gold(cfg, args.season, output_dir)
    print(f"  wrote {CURRENT_GOLD_TABLE} -> s3://{cfg.bucket}/{rebuild['current_gold_key']}")
    print(f"  wrote {PRIORREG_GOLD_TABLE} -> s3://{cfg.bucket}/{rebuild['priorreg_gold_key']}")


if __name__ == "__main__":
    main()

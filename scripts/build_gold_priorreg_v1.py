#!/usr/bin/env python3
"""Build research-only prior-regularized gold efficiency tables to S3.

This leaves production gold tables untouched and writes versioned research
tables using the same schema:

  - gold/team_adjusted_efficiencies_no_garbage_priorreg_k5_v1/
  - gold/team_adjusted_efficiencies_no_garbage_priorreg_k5_hl60_v1/
  - gold/team_adjusted_efficiencies_no_garbage_priorreg_k5_hl45_v1/
  - gold/team_adjusted_efficiencies_no_garbage_priorreg_k5_hl30_v1/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
ETL_ROOT = WORKSPACE_ROOT / "hoops_edge_database_etl_codex"
ETL_SRC = ETL_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ETL_SRC))

from src import config as predictor_config
from cbbd_etl.config import load_config
from cbbd_etl.gold.adjusted_efficiencies import build_no_garbage_priorreg
from cbbd_etl.s3_io import S3IO, make_part_key


@dataclass
class BuildResult:
    season: int
    k: int
    half_life: float | None
    rows: int
    table_name: str
    s3_key: str


def _default_output_dir() -> Path:
    return predictor_config.ARTIFACTS_DIR / "efficiency_research" / "gold_priorreg_v1_build"


def _table_name_for_variant(k: int, half_life: float | None) -> str:
    if half_life is None:
        return f"team_adjusted_efficiencies_no_garbage_priorreg_k{k}_v1"
    return f"team_adjusted_efficiencies_no_garbage_priorreg_k{k}_hl{int(half_life)}_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prior-regularized gold efficiency tables.")
    parser.add_argument("--season-start", type=int, default=2015)
    parser.add_argument("--season-end", type=int, default=2025)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument(
        "--half-lives",
        type=str,
        nargs="+",
        default=["none"],
        help="Half-life variants to build. Use 'none' for no recency weighting.",
    )
    parser.add_argument("--prior-regression", type=float, default=0.30)
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
        help="Local artifact directory for build summaries.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.etl_config)
    cfg.raw.setdefault("gold", {}).setdefault("adjusted_efficiencies", {})
    cfg.raw["gold"]["adjusted_efficiencies"]["solver"] = "iterative"
    cfg.raw["gold"]["adjusted_efficiencies"]["sos_exponent"] = 0.85
    cfg.raw["gold"]["adjusted_efficiencies"]["preseason_regression"] = args.prior_regression

    s3 = S3IO(cfg.bucket, cfg.region)
    asof = datetime.now(timezone.utc).date().isoformat()

    results: list[BuildResult] = []
    half_lives: list[float | None] = []
    for raw in args.half_lives:
        if raw.lower() == "none":
            half_lives.append(None)
        else:
            half_lives.append(float(raw))

    for season in range(args.season_start, args.season_end + 1):
        for k in args.k_values:
            for half_life in half_lives:
                cfg.raw["gold"]["adjusted_efficiencies"]["half_life"] = half_life
                table_name = _table_name_for_variant(k, half_life)
                print(f"Building {table_name} for season {season}...")
                table = build_no_garbage_priorreg(cfg, season, prior_k=float(k))
                if table.num_rows == 0:
                    print(f"  -> empty table for season {season}")
                    continue

                suffix = f"part-priorreg-k{k}"
                if half_life is not None:
                    suffix += f"-hl{int(half_life)}"
                s3_key = make_part_key(
                    cfg.s3_layout["gold_prefix"],
                    table_name,
                    f"season={season}",
                    f"asof={asof}",
                    f"{suffix}.parquet",
                )
                s3.put_parquet(s3_key, table)
                results.append(
                    BuildResult(
                        season=season,
                        k=k,
                        half_life=half_life,
                        rows=table.num_rows,
                        table_name=table_name,
                        s3_key=s3_key,
                    )
                )
                print(f"  -> wrote {table.num_rows} rows to s3://{cfg.bucket}/{s3_key}")

    protocol = {
        "season_start": args.season_start,
        "season_end": args.season_end,
        "k_values": args.k_values,
        "half_lives": args.half_lives,
        "prior_regression": args.prior_regression,
        "fixed_params": {
            "solver": "iterative",
            "sos_exponent": 0.85,
            "half_life": None,
            "shrinkage": 0.0,
            "no_garbage": True,
        },
        "asof": asof,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (output_dir / "build_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )


if __name__ == "__main__":
    main()

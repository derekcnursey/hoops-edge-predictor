#!/usr/bin/env python3
"""Assemble a benchmark-style bundle for a promoted mean-path candidate.

Creates a walk-forward bundle with:
  - mean predictions from a research parquet containing all holdout seasons
  - sigma predictions copied from existing benchmark artifacts

This keeps the site/history rebuild path stable while promoting a new mean model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MEAN_PARQUET = ROOT / "artifacts" / "research" / "objective_tail_compression_experiment_v1" / "LightGBMRegressionL2Blend_predictions.parquet"
DEFAULT_SIGMA_DIRS = [
    ROOT / "artifacts" / "benchmarks" / "canonical_walkforward_priorreg_k5_repaired_lines_neutralfix",
    ROOT / "artifacts" / "benchmarks" / "canonical_walkforward_priorreg_k5_repaired_lines_2026_neutralfix",
]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "benchmarks" / "canonical_walkforward_lgb_l2_blend_repaired_lines_neutralfix"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a benchmark-style bundle for a promoted mean model.")
    parser.add_argument("--mean-parquet", type=Path, default=DEFAULT_MEAN_PARQUET)
    parser.add_argument("--mu-model-name", default="LightGBMRegressionL2Blend")
    parser.add_argument("--sigma-model-name", default="CurrentMLP")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _load_sigma_frames(model_name: str) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for base_dir in DEFAULT_SIGMA_DIRS:
        pred_dir = base_dir / "predictions" / model_name
        for path in sorted(pred_dir.glob("season_*.parquet")):
            season = int(path.stem.split("_")[1])
            frames[season] = pd.read_parquet(path)
    if not frames:
        raise FileNotFoundError(f"No sigma parquet files found for {model_name}")
    return frames


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mean_df = pd.read_parquet(args.mean_parquet)
    if "holdout_season" not in mean_df.columns:
        raise ValueError(f"{args.mean_parquet} missing holdout_season column")

    mu_out_dir = args.output_dir / "predictions" / args.mu_model_name
    mu_out_dir.mkdir(parents=True, exist_ok=True)
    for season, season_df in mean_df.groupby("holdout_season", sort=True):
        out_path = mu_out_dir / f"season_{int(season)}.parquet"
        season_df.to_parquet(out_path, index=False)

    sigma_frames = _load_sigma_frames(args.sigma_model_name)
    sigma_out_dir = args.output_dir / "predictions" / args.sigma_model_name
    sigma_out_dir.mkdir(parents=True, exist_ok=True)
    for season, season_df in sigma_frames.items():
        out_path = sigma_out_dir / f"season_{season}.parquet"
        season_df.to_parquet(out_path, index=False)

    protocol = {
        "mean_parquet": str(args.mean_parquet),
        "mu_model_name": args.mu_model_name,
        "sigma_model_name": args.sigma_model_name,
        "sigma_sources": [str(p) for p in DEFAULT_SIGMA_DIRS],
        "holdout_seasons": sorted(int(s) for s in mean_df["holdout_season"].unique()),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

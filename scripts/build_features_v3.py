#!/usr/bin/env python3
"""Build the V3 feature matrix for one or more seasons.

Usage:
    poetry run python scripts/build_features_v3.py --seasons 2025
    poetry run python scripts/build_features_v3.py --seasons 2015-2026
    poetry run python scripts/build_features_v3.py --seasons 2023,2024,2025
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features_v2
from src.config import FEATURES_DIR, FEATURE_ORDER_V3


def parse_seasons(s: str) -> list[int]:
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Build V3 feature matrix for all seasons")
    parser.add_argument("--seasons", required=True, help="Seasons (e.g. '2015-2026' or '2023,2025')")
    parser.add_argument("--output-dir", default=str(FEATURES_DIR), help="Output directory")
    args = parser.parse_args()

    seasons = parse_seasons(args.seasons)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building V3 features for seasons: {seasons}")
    print(f"Output directory: {out_dir}")
    print(f"Expected features: {len(FEATURE_ORDER_V3)}")
    print()

    total_t0 = time.time()
    results = []

    for season in seasons:
        print(f"{'='*60}")
        print(f"Season {season}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            df = build_features_v2(season)
            elapsed = time.time() - t0

            if df.empty:
                print(f"  No games found for season {season}")
                results.append({"season": season, "games": 0, "elapsed": elapsed, "status": "empty"})
                continue

            # Count features present
            present = [f for f in FEATURE_ORDER_V3 if f in df.columns]
            missing = [f for f in FEATURE_ORDER_V3 if f not in df.columns]

            # NaN analysis
            feat_df = df[[f for f in FEATURE_ORDER_V3 if f in df.columns]]
            nan_rate = feat_df.isna().mean().mean() * 100
            complete_rows = (feat_df.isna().sum(axis=1) == 0).sum()

            print(f"  Games: {len(df)}")
            print(f"  Features: {len(present)}/{len(FEATURE_ORDER_V3)}")
            print(f"  Overall NaN rate: {nan_rate:.1f}%")
            print(f"  Complete rows: {complete_rows}/{len(df)} ({100*complete_rows/len(df):.1f}%)")
            print(f"  Time: {elapsed:.1f}s")

            if missing:
                print(f"  MISSING: {missing}")

            # Save to parquet
            out_path = out_dir / f"season_{season}_v3_features.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  Saved: {out_path}")

            results.append({
                "season": season,
                "games": len(df),
                "features": len(present),
                "nan_rate": nan_rate,
                "complete_rows": complete_rows,
                "elapsed": elapsed,
                "status": "ok",
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"season": season, "games": 0, "elapsed": elapsed, "status": f"error: {e}"})

        print()

    total_elapsed = time.time() - total_t0

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Season':>8} {'Games':>8} {'Features':>10} {'NaN%':>8} {'Complete':>10} {'Time':>8} {'Status':>10}")
    print("-" * 76)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['season']:>8} {r['games']:>8} {r['features']:>10} "
                  f"{r['nan_rate']:>7.1f}% {r['complete_rows']:>10} "
                  f"{r['elapsed']:>7.1f}s {'OK':>10}")
        else:
            print(f"{r['season']:>8} {r.get('games', 0):>8} {'':>10} "
                  f"{'':>8} {'':>10} {r['elapsed']:>7.1f}s {r['status']:>10}")

    total_games = sum(r.get("games", 0) for r in results)
    print(f"\nTotal: {total_games} games across {len(results)} seasons in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()

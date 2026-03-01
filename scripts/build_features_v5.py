#!/usr/bin/env python3
"""Build the V5 feature matrix for all seasons using the bulk pipeline.

V5 uses the same build_features_v2_bulk() pipeline as V4, which now includes:
  - Corrected efficiencies (shot_quality_oe/de, shot_quality_vs_actual)
  - Halfcourt/transition efficiency split
  - All previous V4 features

The V5 *feature set* (86 features) is a curated subset that replaces raw shot
breakdown stats with corrected efficiencies for fewer, cleaner features.

Usage:
    poetry run python scripts/build_features_v5.py --seasons 2015-2025
    poetry run python scripts/build_features_v5.py --seasons 2024,2025
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features_v2_bulk
from src.config import FEATURES_DIR, FEATURE_ORDER_V5


def parse_seasons(s: str) -> list[int]:
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Build V5 features using bulk pipeline")
    parser.add_argument("--seasons", required=True, help="Seasons (e.g. '2015-2025')")
    parser.add_argument("--output-dir", default=str(FEATURES_DIR), help="Output directory")
    args = parser.parse_args()

    seasons = parse_seasons(args.seasons)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building V5 features for seasons: {seasons}")
    print(f"Output directory: {out_dir}")
    print(f"V5 features: {len(FEATURE_ORDER_V5)}")
    print()

    total_t0 = time.time()
    results = []

    for season in seasons:
        print(f"{'='*60}")
        print(f"Season {season}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            df = build_features_v2_bulk(season)
            elapsed = time.time() - t0

            if df.empty:
                print(f"  No games found for season {season}")
                results.append({"season": season, "games": 0, "elapsed": elapsed, "status": "empty"})
                continue

            # Check V5 features present
            present = [f for f in FEATURE_ORDER_V5 if f in df.columns]
            missing = [f for f in FEATURE_ORDER_V5 if f not in df.columns]

            # NaN analysis on V5 features
            feat_df = df[[f for f in FEATURE_ORDER_V5 if f in df.columns]]
            nan_rate = feat_df.isna().mean().mean() * 100
            complete_rows = (feat_df.isna().sum(axis=1) == 0).sum()

            print(f"  Games: {len(df)}")
            print(f"  V5 features: {len(present)}/{len(FEATURE_ORDER_V5)}")
            print(f"  NaN rate: {nan_rate:.1f}%")
            print(f"  Complete rows: {complete_rows}/{len(df)} ({100*complete_rows/len(df):.1f}%)")
            print(f"  Time: {elapsed:.1f}s")

            if missing:
                print(f"  MISSING: {missing}")

            # Save full df (includes all columns from bulk pipeline)
            out_path = out_dir / f"season_{season}_v5_features.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  Saved: {out_path}")

            results.append({
                "season": season, "games": len(df), "features": len(present),
                "nan_rate": nan_rate, "complete_rows": complete_rows,
                "elapsed": elapsed, "status": "ok",
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
    print(f"{'Season':>8} {'Games':>8} {'V5 Feats':>10} {'NaN%':>8} {'Complete':>10} {'Time':>8}")
    print("-" * 66)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['season']:>8} {r['games']:>8} {r['features']:>10} "
                  f"{r['nan_rate']:>7.1f}% {r['complete_rows']:>10} {r['elapsed']:>7.1f}s")
        else:
            print(f"{r['season']:>8} {r.get('games', 0):>8} {'':>10} "
                  f"{'':>8} {'':>10} {r['elapsed']:>7.1f}s  {r['status']}")

    total_games = sum(r.get("games", 0) for r in results)
    print(f"\nTotal: {total_games} games across {len(results)} seasons in {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()

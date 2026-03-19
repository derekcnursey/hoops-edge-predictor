#!/usr/bin/env python3
"""Build raw (unadjusted) feature parquets and cache lines for Session 12.

Requires AWS credentials (derek-admin profile).
Builds features with adjust_ff=False and all extra feature groups enabled.
Also caches lines_2025.parquet for book-spread MAE computation.

Usage:
    AWS_PROFILE=derek-admin poetry run python -u scripts/build_raw_and_lines.py
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.features import build_features, load_research_lines

SEASONS = list(range(2015, 2026))
EXTRA_GROUPS = ["rest_days", "sos", "conf_strength", "form_delta", "tov_rate", "margin_std"]


def build_raw_parquets():
    """Build raw (unadjusted) feature parquets for all seasons."""
    for season in SEASONS:
        out_path = config.FEATURES_DIR / f"season_{season}_no_garbage_features.parquet"
        if out_path.exists():
            print(f"  {out_path.name} already exists — skipping")
            continue

        print(f"  Building raw features for season {season}...")
        try:
            df = build_features(
                season=season,
                no_garbage=True,
                extra_features=EXTRA_GROUPS,
                adjust_ff=False,  # RAW — no opponent adjustment
            )
            if df.empty:
                print(f"    WARNING: Empty DataFrame for season {season}")
                continue

            df.to_parquet(out_path)
            print(f"    Saved: {out_path.name} ({len(df)} games, {len(df.columns)} cols)")
        except Exception as e:
            print(f"    ERROR: {e}")


def cache_lines():
    """Cache lines data for holdout season."""
    out_path = config.FEATURES_DIR / "lines_2025.parquet"
    if out_path.exists():
        print(f"  {out_path.name} already exists — skipping")
        return

    print("  Loading lines for 2025...")
    try:
        lines = load_research_lines(2025)
        if lines is not None and not lines.empty:
            lines.to_parquet(out_path)
            print(f"    Saved: {out_path.name} ({len(lines)} lines)")
        else:
            print("    WARNING: No lines data found")
    except Exception as e:
        print(f"    ERROR: {e}")


if __name__ == "__main__":
    print("Building raw feature parquets...")
    build_raw_parquets()

    print("\nCaching lines data...")
    cache_lines()

    print("\nDone!")

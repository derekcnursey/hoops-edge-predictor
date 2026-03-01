"""Test that build_features_v2_bulk produces equivalent results to build_features_v2.

Usage:
    poetry run python scripts/test_bulk_builder.py
"""
import time
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.features import build_features_v2, build_features_v2_bulk
from src.config import FEATURE_ORDER_V3

SEASON = 2025
GAME_DATE = "2025-01-15"


def compare():
    """Compare bulk builder output with single-date builder."""
    # Build with existing pipeline (single date)
    print("=" * 60)
    print(f"Building single-date features: {GAME_DATE}")
    print("=" * 60)
    t0 = time.time()
    v2_single = build_features_v2(SEASON, game_date=GAME_DATE)
    t_single = time.time() - t0
    print(f"Single-date: {len(v2_single)} games, {len(v2_single.columns)} cols, {t_single:.1f}s\n")

    # Build with bulk pipeline (full season)
    print("=" * 60)
    print(f"Building bulk features: season {SEASON}")
    print("=" * 60)
    t0 = time.time()
    v2_bulk = build_features_v2_bulk(SEASON)
    t_bulk = time.time() - t0
    print(f"Bulk: {len(v2_bulk)} games, {len(v2_bulk.columns)} cols, {t_bulk:.1f}s\n")

    # Filter bulk to same date for comparison
    v2_bulk["_date_str"] = (
        pd.to_datetime(v2_bulk["startDate"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    bulk_filtered = v2_bulk[v2_bulk["_date_str"] == GAME_DATE].copy()
    bulk_filtered = bulk_filtered.drop(columns=["_date_str"])
    print(f"Bulk filtered to {GAME_DATE}: {len(bulk_filtered)} games")

    if len(bulk_filtered) == 0:
        print("ERROR: No games found for the test date in bulk output!")
        return

    if len(v2_single) != len(bulk_filtered):
        print(f"WARNING: Game count mismatch: single={len(v2_single)}, bulk={len(bulk_filtered)}")

    # Compare feature columns
    single_cols = set(v2_single.columns)
    bulk_cols = set(bulk_filtered.columns)

    print(f"\nFeature columns:")
    print(f"  Single-date: {len(single_cols)}")
    print(f"  Bulk: {len(bulk_cols)}")

    only_single = single_cols - bulk_cols
    only_bulk = bulk_cols - single_cols
    if only_single:
        print(f"  Only in single: {only_single}")
    if only_bulk:
        print(f"  Only in bulk: {only_bulk}")

    # Compare feature values on matching games
    common_cols = sorted(single_cols & bulk_cols)
    # Merge on gameId to align rows
    single_sorted = v2_single.sort_values("gameId").reset_index(drop=True)
    bulk_sorted = bulk_filtered.sort_values("gameId").reset_index(drop=True)

    if not set(single_sorted["gameId"]) == set(bulk_sorted["gameId"]):
        print("WARNING: gameId sets differ!")
        common_games = set(single_sorted["gameId"]) & set(bulk_sorted["gameId"])
        print(f"  Common games: {len(common_games)}")
        single_sorted = single_sorted[single_sorted["gameId"].isin(common_games)]
        bulk_sorted = bulk_sorted[bulk_sorted["gameId"].isin(common_games)]
        single_sorted = single_sorted.sort_values("gameId").reset_index(drop=True)
        bulk_sorted = bulk_sorted.sort_values("gameId").reset_index(drop=True)

    # Check V3 feature columns specifically
    v3_features = [f for f in FEATURE_ORDER_V3 if f in common_cols]
    print(f"\nV3 features to compare: {len(v3_features)}")

    mismatches = []
    matches = []
    for col in v3_features:
        s_vals = single_sorted[col].astype(float)
        b_vals = bulk_sorted[col].astype(float)

        # Handle NaN comparison
        both_nan = s_vals.isna() & b_vals.isna()
        either_nan = s_vals.isna() | b_vals.isna()
        nan_mismatch = (~both_nan & either_nan).sum()

        # Numeric comparison on non-NaN values
        valid = ~either_nan
        if valid.sum() > 0:
            s_valid = s_vals[valid]
            b_valid = b_vals[valid]
            max_diff = (s_valid - b_valid).abs().max()
            corr = np.corrcoef(s_valid, b_valid)[0, 1] if valid.sum() > 1 else 1.0
        else:
            max_diff = 0
            corr = 1.0

        if max_diff > 0.01 or nan_mismatch > 0:
            mismatches.append({
                "col": col,
                "max_diff": max_diff,
                "nan_mismatch": nan_mismatch,
                "corr": corr,
            })
        else:
            matches.append(col)

    print(f"\n  Exact matches (max_diff < 0.01): {len(matches)}")
    if mismatches:
        print(f"  Mismatches: {len(mismatches)}")
        for m in sorted(mismatches, key=lambda x: -x["max_diff"])[:20]:
            print(f"    {m['col']:45s}  diff={m['max_diff']:.6f}  "
                  f"nan_mismatch={m['nan_mismatch']}  r={m['corr']:.4f}")
    else:
        print("  ALL features match!")

    print(f"\n{'=' * 60}")
    print(f"SPEEDUP: {t_single:.1f}s (single) → {t_bulk:.1f}s (bulk full season)")
    if t_single > 0:
        per_game_single = t_single / max(len(v2_single), 1)
        estimated_full = per_game_single * len(v2_bulk)
        print(f"  Estimated single-date full season: {estimated_full:.0f}s")
        print(f"  Bulk speedup factor: {estimated_full / max(t_bulk, 0.1):.0f}x")


if __name__ == "__main__":
    compare()

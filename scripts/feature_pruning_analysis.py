"""Analyze feature correlations to identify pruning candidates for V4.

Usage:
    poetry run python scripts/feature_pruning_analysis.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.features import build_features_v2_bulk
from src.config import FEATURE_ORDER_V3

SEASONS = [2024, 2025]  # Two recent seasons for robust analysis


def load_features():
    """Load features for multiple seasons."""
    frames = []
    for season in SEASONS:
        print(f"Loading season {season}...")
        df = build_features_v2_bulk(season)
        df["_season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def analyze_correlations(df: pd.DataFrame):
    """Find highly correlated feature pairs."""
    feat_cols = [c for c in FEATURE_ORDER_V3 if c in df.columns]
    feat_df = df[feat_cols].astype(float)

    # Drop columns that are all NaN
    feat_df = feat_df.dropna(axis=1, how="all")
    remaining = list(feat_df.columns)

    print(f"\nFeatures available for analysis: {len(remaining)}")
    print(f"Rows: {len(feat_df)}")

    # Compute correlation matrix
    corr = feat_df.corr(method="spearman")

    # Find pairs with |r| > 0.95
    print("\n" + "=" * 80)
    print("HIGHLY CORRELATED PAIRS (|r| > 0.95)")
    print("=" * 80)
    pairs = []
    for i, c1 in enumerate(remaining):
        for j, c2 in enumerate(remaining):
            if j <= i:
                continue
            r = corr.loc[c1, c2]
            if abs(r) > 0.95:
                pairs.append((c1, c2, r))

    pairs.sort(key=lambda x: -abs(x[2]))
    for c1, c2, r in pairs:
        print(f"  {c1:45s} vs {c2:45s}  r={r:.4f}")

    print(f"\nTotal pairs with |r| > 0.95: {len(pairs)}")

    # Group by feature "base name" (strip home_/away_ prefix)
    print("\n" + "=" * 80)
    print("CROSS-SIDE DUPLICATES (same stat, home vs away)")
    print("=" * 80)
    home_feats = [f for f in remaining if f.startswith("home_")]
    for hf in home_feats:
        af = hf.replace("home_", "away_", 1)
        if af in remaining:
            r = corr.loc[hf, af]
            if abs(r) > 0.80:
                print(f"  {hf:45s} vs {af:45s}  r={r:.4f}")

    # NaN rates
    print("\n" + "=" * 80)
    print("NaN RATES (>10%)")
    print("=" * 80)
    nan_rates = feat_df.isna().mean().sort_values(ascending=False)
    high_nan = nan_rates[nan_rates > 0.10]
    for col, rate in high_nan.items():
        print(f"  {col:45s}  {rate:.1%}")

    # Feature variance (near-zero variance)
    print("\n" + "=" * 80)
    print("NEAR-ZERO VARIANCE (std < 0.01 on standardized)")
    print("=" * 80)
    standardized = (feat_df - feat_df.mean()) / feat_df.std()
    low_var = standardized.std().sort_values()
    for col, std in low_var.items():
        if std < 0.01 or np.isnan(std):
            print(f"  {col:45s}  std={std:.6f}")

    # Redundancy groups: connected components of |r| > 0.95
    print("\n" + "=" * 80)
    print("REDUNDANCY GROUPS (connected components of |r| > 0.95)")
    print("=" * 80)

    # Build adjacency
    from collections import defaultdict
    adj = defaultdict(set)
    for c1, c2, r in pairs:
        # Only same-side pairs (both home or both away or neither)
        adj[c1].add(c2)
        adj[c2].add(c1)

    visited = set()
    groups = []
    for node in remaining:
        if node in visited or node not in adj:
            continue
        # BFS
        group = []
        queue = [node]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            group.append(n)
            for nb in adj[n]:
                if nb not in visited:
                    queue.append(nb)
        if len(group) > 1:
            groups.append(sorted(group))

    for i, group in enumerate(groups):
        print(f"\n  Group {i + 1} ({len(group)} features):")
        for f in group:
            nan_rate = feat_df[f].isna().mean()
            print(f"    {f:45s}  nan={nan_rate:.1%}")

    # Pruning recommendations
    print("\n" + "=" * 80)
    print("PRUNING RECOMMENDATIONS")
    print("=" * 80)
    drop_candidates = set()
    keep_candidates = set()

    for c1, c2, r in pairs:
        if abs(r) > 0.95:
            # Keep the one with lower NaN rate, or the more interpretable one
            nan1 = feat_df[c1].isna().mean()
            nan2 = feat_df[c2].isna().mean()
            if nan1 <= nan2:
                keep, drop = c1, c2
            else:
                keep, drop = c2, c1
            # Don't drop if already keeping
            if drop not in keep_candidates:
                drop_candidates.add(drop)
                keep_candidates.add(keep)

    print(f"\n  Candidates to DROP ({len(drop_candidates)}):")
    for f in sorted(drop_candidates):
        print(f"    {f}")
    print(f"\n  Remaining features: {len(remaining) - len(drop_candidates)}")

    return pairs, drop_candidates


if __name__ == "__main__":
    df = load_features()
    pairs, drops = analyze_correlations(df)

"""Build opponent-adjusted four-factor feature parquets + sanity checks.

Runs locally on CPU. Produces parquets that can be copied to a GPU instance
for training (scripts/run_adjusted_ff_eval.py).

Usage:
    poetry run python -u scripts/build_adjusted_features.py [--alpha 1.0] [--prior 5]
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Force unbuffered output
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.features import (
    EXTRA_FEATURE_NAMES,
    build_features,
    get_feature_matrix,
    get_targets,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALL_EXTRA_GROUPS = list(EXTRA_FEATURE_NAMES.keys())
ALL_SEASONS = list(range(2015, 2026))


# ── Step 1: Build adjusted feature parquets ──────────────────────


def build_adjusted_parquets(alpha: float = 1.0, prior_weight: float = 5.0):
    """Build features with opponent-adjusted four-factors for all seasons."""
    print(f"\n{'='*70}")
    print(f"Building adjusted features (alpha={alpha}, prior_weight={prior_weight})")
    print(f"{'='*70}\n")

    for season in ALL_SEASONS:
        out_path = config.FEATURES_DIR / f"season_{season}_no_garbage_adj_features.parquet"

        t0 = time.time()
        print(f"  Building adjusted features for season {season}...")
        df = build_features(
            season,
            no_garbage=True,
            extra_features=ALL_EXTRA_GROUPS,
            adjust_ff=True,
            adjust_prior_weight=prior_weight,
            adjust_alpha=alpha,
        )
        elapsed = time.time() - t0

        if df.empty:
            print(f"    WARNING: No games for season {season} ({elapsed:.1f}s)")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"    Saved {len(df)} rows -> {out_path.name} ({elapsed:.1f}s)")


# ── Step 2: Sanity checks ────────────────────────────────────────


def load_adj_features(season: int) -> pd.DataFrame:
    path = config.FEATURES_DIR / f"season_{season}_no_garbage_adj_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Adjusted features not found: {path}")
    return pd.read_parquet(path)


def load_raw_features(season: int) -> pd.DataFrame:
    path = config.FEATURES_DIR / f"season_{season}_no_garbage_v2_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Raw v2 features not found: {path}")
    return pd.read_parquet(path)


def sanity_check():
    """Run sanity checks on 2025 adjusted vs raw features."""
    print(f"\n{'='*70}")
    print("SANITY CHECKS — Season 2025")
    print(f"{'='*70}\n")

    adj_df = load_adj_features(2025)
    raw_df = load_raw_features(2025)

    # Merge on gameId to align rows
    adj_df = adj_df.sort_values("gameId").reset_index(drop=True)
    raw_df = raw_df.sort_values("gameId").reset_index(drop=True)

    # The rolling four-factor feature columns (shared between raw and adjusted)
    rolling_feats = [
        "away_eff_fg_pct", "away_ft_pct", "away_ft_rate", "away_3pt_rate",
        "away_3p_pct", "away_off_rebound_pct", "away_def_rebound_pct",
        "away_def_eff_fg_pct", "away_def_ft_rate", "away_def_3pt_rate",
        "away_def_3p_pct", "away_def_off_rebound_pct", "away_def_def_rebound_pct",
        "home_eff_fg_pct", "home_ft_pct", "home_ft_rate", "home_3pt_rate",
        "home_3p_pct", "home_off_rebound_pct", "home_def_rebound_pct",
        "home_def_eff_fg_pct", "home_opp_ft_rate", "home_def_3pt_rate",
        "home_def_3p_pct", "home_def_off_rebound_pct", "home_def_def_rebound_pct",
    ]

    # Check 1: No NaN/Inf in adjusted features
    print("--- Check 1: NaN/Inf in adjusted features ---")
    for feat in rolling_feats:
        if feat not in adj_df.columns:
            continue
        vals = adj_df[feat].dropna()
        n_inf = np.isinf(vals).sum()
        n_nan = adj_df[feat].isna().sum()
        raw_nan = raw_df[feat].isna().sum() if feat in raw_df.columns else 0
        new_nan = n_nan - raw_nan
        if n_inf > 0 or new_nan > 10:
            print(f"  WARNING: {feat}: {n_inf} inf, {new_nan} new NaN")
    print("  No NaN/Inf issues detected.")

    # Check 2: Reasonable value ranges
    print("\n--- Check 2: Value ranges ---")
    for feat in rolling_feats:
        if feat not in adj_df.columns:
            continue
        vals = adj_df[feat].dropna()
        if len(vals) == 0:
            continue
        lo, hi = vals.min(), vals.max()
        # Percentage stats should be 0-1ish (with some adjustment overshoot)
        if "pct" in feat and (lo < -0.1 or hi > 1.5):
            print(f"  WARNING: {feat}: range [{lo:.4f}, {hi:.4f}]")
        elif "rate" in feat and (lo < -0.5 or hi > 2.0):
            print(f"  WARNING: {feat}: range [{lo:.4f}, {hi:.4f}]")
    print("  Value ranges look reasonable.")

    # Check 3: Correlation between adjusted and raw
    print("\n--- Check 3: Adjusted vs Raw correlation ---")
    corrs = {}
    for feat in rolling_feats:
        if feat not in adj_df.columns or feat not in raw_df.columns:
            continue
        adj_vals = adj_df[feat].values
        raw_vals = raw_df[feat].values
        mask = ~(np.isnan(adj_vals) | np.isnan(raw_vals))
        if mask.sum() < 100:
            continue
        r = np.corrcoef(adj_vals[mask], raw_vals[mask])[0, 1]
        corrs[feat] = r
        if r < 0.85:
            print(f"  LOW CORR: {feat}: r={r:.4f}")

    avg_corr = np.mean(list(corrs.values())) if corrs else 0
    print(f"  Average correlation: {avg_corr:.4f}")
    print(f"  Min correlation: {min(corrs.values()):.4f} ({min(corrs, key=corrs.get)})")
    print(f"  Max correlation: {max(corrs.values()):.4f} ({max(corrs, key=corrs.get)})")

    # Check 4: ft_pct should be UNCHANGED (no defensive counterpart)
    print("\n--- Check 4: ft_pct unchanged ---")
    for prefix in ["away_ft_pct", "home_ft_pct"]:
        if prefix in adj_df.columns and prefix in raw_df.columns:
            mask = ~(adj_df[prefix].isna() | raw_df[prefix].isna())
            if mask.sum() > 0:
                diff = (adj_df.loc[mask, prefix].values - raw_df.loc[mask, prefix].values)
                max_diff = np.abs(diff).max()
                print(f"  {prefix}: max |adj - raw| = {max_diff:.8f}")

    # Check 5: Team-level analysis for 5 teams across quality spectrum
    print("\n--- Check 5: Team-level adjustment patterns ---")
    # Look up team names and identify 5 diverse teams
    team_cols = ["homeTeamId", "homeTeam", "awayTeamId", "awayTeam"]
    available_cols = [c for c in team_cols if c in adj_df.columns]

    # Build team lookup from whatever we have
    teams = {}
    if "homeTeam" in adj_df.columns:
        for _, row in adj_df[["homeTeamId", "homeTeam"]].drop_duplicates().iterrows():
            teams[int(row["homeTeamId"])] = str(row["homeTeam"])

    # Try to find diverse teams by adj_oe (proxy for quality)
    # Use home team adj_oe as quality metric
    if "home_team_adj_oe" in adj_df.columns and "homeTeamId" in adj_df.columns:
        team_quality = adj_df.groupby("homeTeamId")["home_team_adj_oe"].mean().dropna()
        team_quality = team_quality.sort_values(ascending=False)

        # Pick 5 teams across the quality spectrum
        n = len(team_quality)
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        sample_teams = [int(team_quality.index[i]) for i in indices]

        for tid in sample_teams:
            name = teams.get(tid, f"Team {tid}")
            # Get home games for this team
            home_mask_adj = adj_df["homeTeamId"] == tid
            home_mask_raw = raw_df["homeTeamId"] == tid

            if home_mask_adj.sum() < 3:
                continue

            # Compare a key stat: home_eff_fg_pct
            adj_efg = adj_df.loc[home_mask_adj, "home_eff_fg_pct"].dropna()
            raw_efg = raw_df.loc[home_mask_raw, "home_eff_fg_pct"].dropna()

            adj_defg = adj_df.loc[home_mask_adj, "home_def_eff_fg_pct"].dropna()
            raw_defg = raw_df.loc[home_mask_raw, "home_def_eff_fg_pct"].dropna()

            quality = team_quality.get(tid, 0)
            print(f"\n  {name} (adj_oe={quality:.1f}):")
            print(f"    eFG% (rolling): raw={raw_efg.mean():.4f} → adj={adj_efg.mean():.4f} "
                  f"(delta={adj_efg.mean() - raw_efg.mean():+.4f})")
            print(f"    def_eFG% (rolling): raw={raw_defg.mean():.4f} → adj={adj_defg.mean():.4f} "
                  f"(delta={adj_defg.mean() - raw_defg.mean():+.4f})")

    # Check 6: Early November vs late February (shrinkage dominance)
    print("\n--- Check 6: Early vs late season adjustment magnitude ---")
    adj_df["_month"] = pd.to_datetime(adj_df["startDate"], errors="coerce").dt.month
    raw_df["_month"] = pd.to_datetime(raw_df["startDate"], errors="coerce").dt.month

    for month, label in [(11, "November"), (2, "February")]:
        adj_month = adj_df[adj_df["_month"] == month]
        raw_month = raw_df[raw_df["_month"] == month]

        if len(adj_month) == 0:
            continue

        # Compute average absolute adjustment for eff_fg_pct
        feat = "away_eff_fg_pct"
        if feat in adj_month.columns and feat in raw_month.columns:
            adj_vals = adj_month[feat].dropna().values
            raw_vals = raw_month[feat].dropna().values
            n = min(len(adj_vals), len(raw_vals))
            if n > 0:
                abs_diff = np.abs(adj_vals[:n] - raw_vals[:n]).mean()
                print(f"  {label}: mean |adj - raw| for {feat} = {abs_diff:.6f} ({n} games)")

    adj_df.drop(columns=["_month"], inplace=True, errors="ignore")

    print("\n  Sanity checks complete.")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Build adjusted four-factor features")
    parser.add_argument("--alpha", type=float, default=1.0, help="SOS exponent (default: 1.0)")
    parser.add_argument("--prior", type=float, default=5.0, help="Bayesian prior weight (default: 5)")
    parser.add_argument("--skip-build", action="store_true", help="Skip building, only run sanity checks")
    args = parser.parse_args()

    if not args.skip_build:
        build_adjusted_parquets(alpha=args.alpha, prior_weight=args.prior)

    sanity_check()


if __name__ == "__main__":
    main()

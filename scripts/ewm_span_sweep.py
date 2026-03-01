#!/usr/bin/env python3
"""EWM span sweep: find optimal EWM span per stat group.

For each stat group and candidate span, computes per-team EWM rolling
averages with shift(1), then measures Spearman correlation against the
next game's actual value.  Selects the span that maximises median rho
across all teams.

Usage:
    python scripts/ewm_span_sweep.py --seasons 2024 2025
    python scripts/ewm_span_sweep.py --seasons 2023 2024 2025 --output artifacts/optimal_ewm_spans.json

Output:
    artifacts/optimal_ewm_spans.json — JSON mapping group_name → best span.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, s3_reader
from src.pbp_advanced_stats import (
    SHOT_QUALITY_COLS,
    TURNOVER_DECOMP_COLS,
    TEMPO_COLS,
    PUTBACK_COLS,
    CLUTCH_COLS,
    DROUGHT_COLS,
    HALF_SPLIT_COLS,
    ROTATION_DEPTH_COLS,
    PRESSURE_FT_COLS,
    ZONE_COUNT_COLS,
    COMPOSITE_COLS,
    compute_advanced_stats,
)
from src.kill_shot_analysis import KILL_SHOT_COLS, compute_kill_shot_metrics
from src.four_factors import FOUR_FACTOR_COLS, compute_game_four_factors
from src.luck_regression import LUCK_FEATURE_COLS, compute_luck_features

# ── Stat group definitions ───────────────────────────────────────

STAT_GROUPS: dict[str, list[str]] = {
    "shot_quality": [
        "rim_rate", "mid_range_rate", "rim_fg_pct", "mid_range_fg_pct",
        "assisted_fg_pct",
        "def_rim_rate", "def_mid_range_rate", "def_rim_fg_pct", "def_mid_range_fg_pct",
    ],
    "turnover": [
        "live_ball_tov_rate", "dead_ball_tov_rate", "steal_rate_defense",
        "transition_rate", "transition_scoring_efficiency",
    ],
    "tempo": [
        "avg_possession_length", "early_clock_shot_rate", "shot_clock_pressure_rate",
    ],
    "second_chance": [
        "putback_rate", "second_chance_pts_per_oreb",
    ],
    "clutch": [
        "clutch_off_efficiency", "clutch_def_efficiency",
        "non_clutch_to_clutch_delta",
    ],
    "drought": [
        "off_avg_drought_length", "off_max_drought_length",
        "def_avg_drought_length", "def_max_drought_length",
    ],
    "half_split": [
        "half_adjustment_delta", "second_half_def_delta",
    ],
    "rotation": [
        "scoring_hhi", "top2_scorer_pct",
    ],
    "four_factors": FOUR_FACTOR_COLS,
    "kill_shot": KILL_SHOT_COLS,
    "composites": COMPOSITE_COLS,
    "luck": LUCK_FEATURE_COLS,
}

CANDIDATE_SPANS = [5, 8, 10, 12, 15, 20, 25, 30]


def _spearman_rho(series: pd.Series, span: int) -> float:
    """Compute median Spearman rho for EWM(span) + shift(1) vs actual across teams."""
    rolling = series.ewm(span=span, min_periods=1).mean().shift(1)
    # Compare rolling prediction to actual value
    valid = pd.notna(rolling) & pd.notna(series)
    if valid.sum() < 10:
        return np.nan
    rho, _ = spearmanr(rolling[valid], series[valid])
    return rho


def sweep_group(
    df: pd.DataFrame,
    stat_cols: list[str],
    spans: list[int],
) -> dict[int, float]:
    """For a set of stat columns, compute median Spearman rho per span.

    Returns dict mapping span → median rho across all stats and teams.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.sort_values(["teamid", "_date", "gameid"]).reset_index(drop=True)

    available = [c for c in stat_cols if c in df.columns]
    if not available:
        return {}

    span_rhos: dict[int, list[float]] = {s: [] for s in spans}

    for _tid, group in df.groupby("teamid"):
        if len(group) < 15:
            continue
        for stat in available:
            series = group[stat].reset_index(drop=True)
            if series.notna().sum() < 10:
                continue
            for span in spans:
                rho = _spearman_rho(series, span)
                if pd.notna(rho):
                    span_rhos[span].append(rho)

    return {
        span: float(np.median(rhos)) if rhos else np.nan
        for span, rhos in span_rhos.items()
    }


def main():
    parser = argparse.ArgumentParser(description="EWM span sweep for optimal spans per stat group")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--output", type=str, default=str(config.ARTIFACTS_DIR / "optimal_ewm_spans.json"))
    args = parser.parse_args()

    print(f"Loading data for seasons: {args.seasons}")

    # Load and concatenate per-game stats across seasons
    adv_frames = []
    ks_frames = []
    ff_frames = []
    luck_frames = []

    for season in args.seasons:
        print(f"  Processing season {season}...")

        # Four factors from boxscores
        try:
            tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAME_TEAMS, season=season)
            if tbl.num_rows > 0:
                box = tbl.to_pandas()
                ff = compute_game_four_factors(box)
                if not ff.empty:
                    ff_frames.append(ff)
        except Exception as e:
            print(f"    Warning: could not load boxscores for season {season}: {e}")

        # PBP-derived stats
        try:
            pbp_keys = s3_reader.list_parquet_keys(
                f"{config.SILVER_PREFIX}/fct_pbp_plays_enriched/season={season}/"
            )
            if pbp_keys:
                pbp_tbl = s3_reader.read_parquet_table(pbp_keys)
                pbp_df = pbp_tbl.to_pandas()

                adv = compute_advanced_stats(pbp_df)
                if not adv.empty:
                    adv_frames.append(adv)

                    # Luck features from raw (pre-adjustment) stats
                    luck = compute_luck_features(adv)
                    if not luck.empty:
                        luck_frames.append(luck)

                ks = compute_kill_shot_metrics(pbp_df)
                if not ks.empty:
                    ks_frames.append(ks)
        except Exception as e:
            print(f"    Warning: could not load PBP for season {season}: {e}")

    # Combine
    adv_all = pd.concat(adv_frames, ignore_index=True) if adv_frames else pd.DataFrame()
    ks_all = pd.concat(ks_frames, ignore_index=True) if ks_frames else pd.DataFrame()
    ff_all = pd.concat(ff_frames, ignore_index=True) if ff_frames else pd.DataFrame()
    luck_all = pd.concat(luck_frames, ignore_index=True) if luck_frames else pd.DataFrame()

    # Merge luck into adv for sweep
    if not luck_all.empty and not adv_all.empty:
        adv_all = adv_all.merge(
            luck_all[["gameid", "teamid"] + LUCK_FEATURE_COLS],
            on=["gameid", "teamid"],
            how="left",
        )

    # Run sweep per group
    optimal_spans: dict[str, int] = {}
    print("\nRunning EWM span sweep...")
    print(f"{'Group':<20} {'Best Span':>10} {'Rho':>10}  Span→Rho details")
    print("-" * 80)

    for group_name, stat_cols in STAT_GROUPS.items():
        if group_name == "four_factors":
            source = ff_all
        elif group_name == "kill_shot":
            source = ks_all
        else:
            source = adv_all

        if source.empty:
            print(f"{group_name:<20} {'N/A':>10} {'N/A':>10}  (no data)")
            optimal_spans[group_name] = 15  # fallback
            continue

        rhos = sweep_group(source, stat_cols, CANDIDATE_SPANS)
        if not rhos:
            print(f"{group_name:<20} {'N/A':>10} {'N/A':>10}  (no valid stats)")
            optimal_spans[group_name] = 15
            continue

        # Find best span
        best_span = max(rhos, key=lambda s: rhos[s] if pd.notna(rhos[s]) else -999)
        best_rho = rhos[best_span]

        detail = "  ".join(f"{s}:{rhos[s]:.3f}" for s in CANDIDATE_SPANS if pd.notna(rhos.get(s, np.nan)))
        print(f"{group_name:<20} {best_span:>10} {best_rho:>10.3f}  {detail}")
        optimal_spans[group_name] = best_span

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(optimal_spans, indent=2) + "\n")
    print(f"\nWrote optimal spans to {output_path}")


if __name__ == "__main__":
    main()

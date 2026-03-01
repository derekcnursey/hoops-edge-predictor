#!/usr/bin/env python3
"""Comprehensive validation of V3 features across all seasons.

Produces docs/session14_validation_report.md and optional plots in docs/plots/.

Usage:
    poetry run python scripts/validate_v3_features.py
    poetry run python scripts/validate_v3_features.py --seasons 2023,2024,2025
    poetry run python scripts/validate_v3_features.py --features-dir features/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, FEATURES_DIR, FEATURE_ORDER_V3

# Feature groups for validation
RATE_FEATURES = [
    f for f in FEATURE_ORDER_V3
    if any(kw in f for kw in [
        "_rate", "_pct", "blowout_rate", "rim_rate", "mid_range_rate",
    ])
]

EFFICIENCY_FEATURES = [
    f for f in FEATURE_ORDER_V3
    if any(kw in f for kw in ["eff_fg_pct", "adj_oe", "adj_de"])
]

LUCK_FEATURES = [
    f for f in FEATURE_ORDER_V3
    if any(kw in f for kw in ["efg_luck", "three_pt_luck", "two_pt_luck"])
]

COMPOSITE_FEATURES = [
    f for f in FEATURE_ORDER_V3
    if any(kw in f for kw in [
        "transition_scoring_efficiency", "expected_pts_per_shot", "transition_value",
    ])
]


def load_all_seasons(features_dir: Path, seasons: list[int]) -> dict[int, pd.DataFrame]:
    """Load V3 feature parquet files for the given seasons."""
    data = {}
    for season in seasons:
        path = features_dir / f"season_{season}_v3_features.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            data[season] = df
    return data


# ── 3a. Shape & Completeness ─────────────────────────────────────────


def check_shape_completeness(data: dict[int, pd.DataFrame]) -> list[str]:
    lines = []
    lines.append("## 3a. Shape & Completeness\n")
    lines.append("| Season | Games | Columns | Features Present | Missing Features | Overall NaN% |")
    lines.append("|--------|-------|---------|------------------|------------------|-------------|")

    for season in sorted(data.keys()):
        df = data[season]
        present = [f for f in FEATURE_ORDER_V3 if f in df.columns]
        missing = [f for f in FEATURE_ORDER_V3 if f not in df.columns]
        feat_df = df[[f for f in FEATURE_ORDER_V3 if f in df.columns]]
        nan_pct = feat_df.isna().mean().mean() * 100
        lines.append(
            f"| {season} | {len(df)} | {len(df.columns)} | "
            f"{len(present)}/{len(FEATURE_ORDER_V3)} | {len(missing)} | {nan_pct:.1f}% |"
        )

    lines.append("")

    # Per-feature NaN rates across all seasons
    all_df = pd.concat(data.values(), ignore_index=True)
    feat_df = all_df[[f for f in FEATURE_ORDER_V3 if f in all_df.columns]]
    nan_rates = feat_df.isna().mean().sort_values(ascending=False)

    high_nan = nan_rates[nan_rates > 0.30]
    if len(high_nan) > 0:
        lines.append("### Features with >30% NaN (overall)\n")
        lines.append("| Feature | NaN Rate |")
        lines.append("|---------|----------|")
        for feat, rate in high_nan.items():
            lines.append(f"| {feat} | {rate:.1%} |")
        lines.append("")

    # Per-season NaN rates for flagged features
    moderate_nan = nan_rates[nan_rates > 0.10]
    if len(moderate_nan) > 0:
        lines.append("### Features with >10% NaN by season\n")
        lines.append("| Feature | " + " | ".join(str(s) for s in sorted(data.keys())) + " |")
        lines.append("|---------|" + "|".join(["-----"] * len(data)) + "|")
        for feat in moderate_nan.index:
            rates = []
            for season in sorted(data.keys()):
                df = data[season]
                if feat in df.columns:
                    r = df[feat].isna().mean()
                    rates.append(f"{r:.1%}")
                else:
                    rates.append("N/A")
            lines.append(f"| {feat} | " + " | ".join(rates) + " |")
        lines.append("")

    return lines


# ── 3b. Distribution Sanity ──────────────────────────────────────────


def check_distributions(data: dict[int, pd.DataFrame]) -> list[str]:
    lines = []
    lines.append("## 3b. Distribution Sanity\n")

    all_df = pd.concat(data.values(), ignore_index=True)
    feat_cols = [f for f in FEATURE_ORDER_V3 if f in all_df.columns]
    feat_df = all_df[feat_cols]

    # Summary stats
    lines.append("### Summary Statistics (all seasons combined)\n")
    lines.append("| Feature | Mean | Std | Min | Max | Median |")
    lines.append("|---------|------|-----|-----|-----|--------|")
    for col in feat_cols:
        vals = feat_df[col].dropna()
        if len(vals) > 0:
            lines.append(
                f"| {col} | {vals.mean():.4f} | {vals.std():.4f} | "
                f"{vals.min():.4f} | {vals.max():.4f} | {vals.median():.4f} |"
            )
    lines.append("")

    # Zero variance features
    zero_var = []
    for col in feat_cols:
        vals = feat_df[col].dropna()
        if len(vals) > 10 and vals.std() == 0:
            zero_var.append(col)

    if zero_var:
        lines.append(f"### ALERT: Zero Variance Features\n")
        for f in zero_var:
            lines.append(f"- {f}")
        lines.append("")
    else:
        lines.append("### Zero Variance Features: None found\n")

    # Extreme outliers (>5 std from mean)
    outlier_features = []
    for col in feat_cols:
        vals = feat_df[col].dropna()
        if len(vals) > 50:
            mean, std = vals.mean(), vals.std()
            if std > 0:
                n_outliers = ((vals - mean).abs() > 5 * std).sum()
                if n_outliers > 0:
                    outlier_features.append((col, n_outliers, n_outliers / len(vals)))

    if outlier_features:
        lines.append("### Features with Extreme Outliers (>5 std)\n")
        lines.append("| Feature | Count | Rate |")
        lines.append("|---------|-------|------|")
        for col, cnt, rate in sorted(outlier_features, key=lambda x: -x[2])[:30]:
            lines.append(f"| {col} | {cnt} | {rate:.3%} |")
        lines.append("")

    # Rate features outside [0, 1]
    bad_rates = []
    for col in RATE_FEATURES:
        if col in feat_df.columns:
            vals = feat_df[col].dropna()
            if len(vals) > 0:
                below = (vals < -0.01).sum()
                above = (vals > 1.01).sum()
                if below > 0 or above > 0:
                    bad_rates.append((col, below, above, vals.min(), vals.max()))

    if bad_rates:
        lines.append("### Rate Features Outside [0, 1]\n")
        lines.append("| Feature | Below 0 | Above 1 | Min | Max |")
        lines.append("|---------|---------|---------|-----|-----|")
        for col, below, above, mn, mx in bad_rates:
            lines.append(f"| {col} | {below} | {above} | {mn:.4f} | {mx:.4f} |")
        lines.append("")
    else:
        lines.append("### Rate Features: All within [0, 1] bounds\n")

    return lines


# ── 3c. Luck Feature Validation ──────────────────────────────────────


def check_luck_features(data: dict[int, pd.DataFrame]) -> list[str]:
    lines = []
    lines.append("## 3c. Luck Feature Validation\n")

    all_df = pd.concat(data.values(), ignore_index=True)

    luck_cols = [f for f in LUCK_FEATURES if f in all_df.columns]
    if not luck_cols:
        lines.append("No luck features found in data.\n")
        return lines

    # Mean should be near zero
    lines.append("### Mean Luck Values (should be ~0)\n")
    lines.append("| Feature | Mean | Std | Min | Max | Median |")
    lines.append("|---------|------|-----|-----|-----|--------|")
    for col in luck_cols:
        vals = all_df[col].dropna()
        if len(vals) > 0:
            lines.append(
                f"| {col} | {vals.mean():.6f} | {vals.std():.6f} | "
                f"{vals.min():.6f} | {vals.max():.6f} | {vals.median():.6f} |"
            )
    lines.append("")

    # Distribution symmetry
    lines.append("### Distribution Symmetry\n")
    lines.append("| Feature | Skewness | % Positive | % Negative |")
    lines.append("|---------|----------|-----------|-----------|")
    for col in luck_cols:
        vals = all_df[col].dropna()
        if len(vals) > 10:
            skew = vals.skew()
            pct_pos = (vals > 0).mean() * 100
            pct_neg = (vals < 0).mean() * 100
            lines.append(f"| {col} | {skew:.4f} | {pct_pos:.1f}% | {pct_neg:.1f}% |")
    lines.append("")

    # Predictive signal check: correlation with next-game efficiency
    # We can check if rolling luck (as of game G) anticorrelates with next-game actual
    lines.append("### Predictive Signal (negative correlation = regression to mean = good)\n")
    lines.append("Note: Luck features represent rolling averages of per-game luck residuals. ")
    lines.append("A team with sustained positive luck is likely to regress (negative correlation).\n")

    # Use home/away efg as proxy for offensive efficiency
    eff_cols_for_corr = [
        ("home_efg_luck", "home_eff_fg_pct"),
        ("away_efg_luck", "away_eff_fg_pct"),
    ]
    lines.append("| Luck Feature | Efficiency Feature | Pearson r | Spearman rho |")
    lines.append("|-------------|-------------------|-----------|-------------|")
    for luck_col, eff_col in eff_cols_for_corr:
        if luck_col in all_df.columns and eff_col in all_df.columns:
            valid = all_df[[luck_col, eff_col]].dropna()
            if len(valid) > 30:
                from scipy.stats import spearmanr
                pearson_r = valid[luck_col].corr(valid[eff_col])
                spearman_rho, _ = spearmanr(valid[luck_col], valid[eff_col])
                lines.append(f"| {luck_col} | {eff_col} | {pearson_r:.4f} | {spearman_rho:.4f} |")
    lines.append("")

    return lines


# ── 3d. Feature Correlations ─────────────────────────────────────────


def check_correlations(data: dict[int, pd.DataFrame]) -> list[str]:
    lines = []
    lines.append("## 3d. Feature Correlations\n")

    all_df = pd.concat(data.values(), ignore_index=True)
    feat_cols = [f for f in FEATURE_ORDER_V3 if f in all_df.columns]
    feat_df = all_df[feat_cols].select_dtypes(include=[np.number])

    if len(feat_df) < 50 or len(feat_df.columns) < 10:
        lines.append("Insufficient data for correlation analysis.\n")
        return lines

    corr = feat_df.corr()

    # Find highly correlated pairs
    pairs_95 = []
    pairs_90 = []
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if i < j:
                r = corr.loc[c1, c2]
                if pd.notna(r):
                    if abs(r) > 0.95:
                        pairs_95.append((c1, c2, r))
                    elif abs(r) > 0.90:
                        pairs_90.append((c1, c2, r))

    lines.append(f"### Near-Duplicate Pairs (|r| > 0.95): {len(pairs_95)}\n")
    if pairs_95:
        lines.append("| Feature 1 | Feature 2 | Pearson r |")
        lines.append("|-----------|-----------|-----------|")
        for c1, c2, r in sorted(pairs_95, key=lambda x: -abs(x[2])):
            lines.append(f"| {c1} | {c2} | {r:.4f} |")
        lines.append("")

    lines.append(f"### High Correlation Pairs (0.90 < |r| < 0.95): {len(pairs_90)}\n")
    if pairs_90:
        lines.append("| Feature 1 | Feature 2 | Pearson r |")
        lines.append("|-----------|-----------|-----------|")
        for c1, c2, r in sorted(pairs_90, key=lambda x: -abs(x[2]))[:30]:
            lines.append(f"| {c1} | {c2} | {r:.4f} |")
        lines.append("")

    # Top 20 most correlated pairs overall
    all_pairs = pairs_95 + pairs_90
    all_pairs_extended = []
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if i < j:
                r = corr.loc[c1, c2]
                if pd.notna(r) and abs(r) > 0.80:
                    all_pairs_extended.append((c1, c2, r))

    top20 = sorted(all_pairs_extended, key=lambda x: -abs(x[2]))[:20]
    lines.append("### Top 20 Most Correlated Feature Pairs\n")
    lines.append("| Rank | Feature 1 | Feature 2 | Pearson r |")
    lines.append("|------|-----------|-----------|-----------|")
    for idx, (c1, c2, r) in enumerate(top20, 1):
        lines.append(f"| {idx} | {c1} | {c2} | {r:.4f} |")
    lines.append("")

    # Check new features vs V1 features
    v1_features = [
        f for f in FEATURE_ORDER_V3
        if not any(kw in f for kw in [
            "rim_", "mid_range_", "assisted_fg", "live_ball_tov", "dead_ball_tov",
            "steal_rate", "transition_", "avg_possession", "early_clock", "shot_clock_pressure",
            "putback_", "second_chance_pts", "clutch_", "non_clutch", "drought",
            "half_adjustment", "second_half_def", "scoring_hhi", "top2_scorer",
            "ft_pressure", "max_run", "run_frequency", "avg_run_magnitude",
            "pace_variance", "pace_conformity", "pace_mismatch",
            "scoring_variance", "blowout_rate",
            "efg_luck", "three_pt_luck", "two_pt_luck",
            "expected_pts_per_shot", "transition_value", "transition_scoring_efficiency",
        ])
    ]
    new_features = [f for f in FEATURE_ORDER_V3 if f not in v1_features]

    lines.append("### New V3 Features Correlated with V1 Features (|r| > 0.80)\n")
    v1_new_corr = []
    for new_f in new_features:
        if new_f not in corr.columns:
            continue
        for v1_f in v1_features:
            if v1_f not in corr.columns:
                continue
            r = corr.loc[new_f, v1_f]
            if pd.notna(r) and abs(r) > 0.80:
                v1_new_corr.append((new_f, v1_f, r))

    if v1_new_corr:
        lines.append("| New Feature | V1 Feature | Pearson r |")
        lines.append("|------------|-----------|-----------|")
        for n, v, r in sorted(v1_new_corr, key=lambda x: -abs(x[2]))[:20]:
            lines.append(f"| {n} | {v} | {r:.4f} |")
    else:
        lines.append("No new features highly correlated with V1 features.\n")
    lines.append("")

    # Save correlation heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(24, 20))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=4)
        ax.set_yticklabels(corr.columns, fontsize=4)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.title("V3 Feature Correlation Matrix (148 features)")
        plt.tight_layout()
        plot_path = PROJECT_ROOT / "docs" / "plots" / "correlation_heatmap.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        lines.append(f"Correlation heatmap saved to: `docs/plots/correlation_heatmap.png`\n")
    except Exception as e:
        lines.append(f"Could not generate correlation heatmap: {e}\n")

    return lines


# ── 3e. Composite Feature Validation ─────────────────────────────────


def check_composites(data: dict[int, pd.DataFrame]) -> list[str]:
    lines = []
    lines.append("## 3e. Composite Feature Validation\n")

    all_df = pd.concat(data.values(), ignore_index=True)
    comp_cols = [f for f in COMPOSITE_FEATURES if f in all_df.columns]

    if not comp_cols:
        lines.append("No composite features found.\n")
        return lines

    # expected_pts_per_shot range check
    for prefix in ["home", "away"]:
        col = f"{prefix}_expected_pts_per_shot"
        if col in all_df.columns:
            vals = all_df[col].dropna()
            lines.append(f"### {col}\n")
            lines.append(f"- Range: [{vals.min():.4f}, {vals.max():.4f}]")
            lines.append(f"- Mean: {vals.mean():.4f}, Std: {vals.std():.4f}")
            lines.append(f"- Expected range: ~0.7-1.4 pts/shot")
            in_range = ((vals >= 0.5) & (vals <= 2.0)).mean()
            lines.append(f"- % in [0.5, 2.0]: {in_range:.1%}\n")

    # transition_value check
    for prefix in ["home", "away"]:
        tv_col = f"{prefix}_transition_value"
        sr_col = f"{prefix}_steal_rate_defense"
        tse_col = f"{prefix}_transition_scoring_efficiency"
        if all(c in all_df.columns for c in [tv_col, sr_col, tse_col]):
            valid = all_df[[tv_col, sr_col, tse_col]].dropna()
            if len(valid) > 0:
                lines.append(f"### {tv_col}\n")
                lines.append(f"- Correlation with (steal_rate * trans_eff): ", )
                expected = valid[sr_col] * valid[tse_col]
                corr = valid[tv_col].corr(expected)
                lines.append(f"{corr:.4f}")
                lines.append(f"- This should be ~1.0 since transition_value = steal_rate_defense * transition_scoring_efficiency\n")

    # Correlation with offensive efficiency
    for prefix in ["home", "away"]:
        eps_col = f"{prefix}_expected_pts_per_shot"
        oe_col = f"{prefix}_team_adj_oe"
        if eps_col in all_df.columns and oe_col in all_df.columns:
            valid = all_df[[eps_col, oe_col]].dropna()
            if len(valid) > 30:
                r = valid[eps_col].corr(valid[oe_col])
                lines.append(f"- {eps_col} corr with {oe_col}: {r:.4f} (should be positive)\n")

    return lines


# ── 3f. EWM Span Impact ─────────────────────────────────────────────


def check_ewm_spans() -> list[str]:
    lines = []
    lines.append("## 3f. EWM Span Impact\n")

    spans_path = ARTIFACTS_DIR / "optimal_ewm_spans.json"
    if not spans_path.exists():
        lines.append("No optimal_ewm_spans.json found.\n")
        return lines

    with open(spans_path) as f:
        spans = json.load(f)

    lines.append("### Optimal Spans from EWM Sweep\n")
    lines.append("| Stat Group | Optimal Span | Default (15) | Change |")
    lines.append("|-----------|-------------|-------------|--------|")
    for group, span in sorted(spans.items()):
        diff = span - 15
        lines.append(f"| {group} | {span} | 15 | {diff:+d} |")
    lines.append("")

    lines.append("### Key Findings\n")
    all_five = all(v == 5 for v in spans.values())
    if all_five:
        lines.append("- All groups optimized to span=5 (shortest tested window)")
        lines.append("- This suggests recent performance is most predictive for college basketball")
        lines.append("- College basketball has high game-to-game variance, so shorter windows capture current form better")
        lines.append("- The original span=15 was over-smoothing, diluting recent signal with stale data")
    lines.append("")

    return lines


# ── 3g. No Data Leakage Check ───────────────────────────────────────


def check_leakage(data: dict[int, pd.DataFrame]) -> list[str]:
    lines = []
    lines.append("## 3g. No Data Leakage Check\n")

    # Check that first few games of each season have NaN for rolling features
    rolling_keywords = [
        "eff_fg_pct", "ft_pct", "ft_rate", "3pt_rate", "3p_pct",
        "off_rebound_pct", "def_rebound_pct",
        "rim_rate", "mid_range_rate", "rim_fg_pct",
        "scoring_hhi", "drought", "clutch",
        "efg_luck", "three_pt_luck",
    ]

    lines.append("### Rolling Feature NaN Rate for First N Games\n")
    lines.append("Rolling features should be NaN (or near-NaN) for the first game of each season.\n")
    lines.append("| Season | First Game NaN Rate | Games 1-5 NaN Rate | Games 6-30 NaN Rate | Games 31+ NaN Rate |")
    lines.append("|--------|--------------------|--------------------|--------------------|--------------------|")

    for season in sorted(data.keys()):
        df = data[season]
        if len(df) == 0:
            continue

        # Identify rolling features present
        rolling_cols = [
            f for f in FEATURE_ORDER_V3
            if f in df.columns and any(kw in f for kw in rolling_keywords)
        ]
        if not rolling_cols:
            continue

        feat_df = df[rolling_cols]

        # Sort by startDate if available
        if "startDate" in df.columns:
            sort_df = df.copy()
            sort_df["_dt"] = pd.to_datetime(sort_df["startDate"], errors="coerce")
            sort_idx = sort_df.sort_values("_dt").index
            feat_df = feat_df.loc[sort_idx]

        n = len(feat_df)
        first_nan = feat_df.iloc[0].isna().mean() if n > 0 else 0
        first5_nan = feat_df.iloc[:min(5, n)].isna().mean().mean() if n > 0 else 0
        mid_nan = feat_df.iloc[5:min(30, n)].isna().mean().mean() if n > 5 else 0
        late_nan = feat_df.iloc[30:].isna().mean().mean() if n > 30 else 0

        lines.append(
            f"| {season} | {first_nan:.1%} | {first5_nan:.1%} | {mid_nan:.1%} | {late_nan:.1%} |"
        )
    lines.append("")

    lines.append("### Interpretation\n")
    lines.append("- First game: Should be ~100% NaN (no prior data to compute rolling averages)")
    lines.append("- Games 1-5: High NaN expected (EWM warming up)")
    lines.append("- Games 6-30: Moderate NaN (some features need more history)")
    lines.append("- Games 31+: Should be very low NaN (<5%)")
    lines.append("- If first game has LOW NaN rates, there may be data leakage\n")

    return lines


# ── 3h. Season Boundary Check ───────────────────────────────────────


def check_season_boundaries(data: dict[int, pd.DataFrame]) -> list[str]:
    lines = []
    lines.append("## 3h. Season Boundary Check\n")

    lines.append("Verifying that rolling averages reset at season boundaries.\n")
    lines.append("Each season's V3 features are computed independently via `build_features_v2(season)`. ")
    lines.append("Since the pipeline loads data fresh for each season from S3 (partitioned by season), ")
    lines.append("cross-season leakage is architecturally prevented.\n")

    lines.append("### Evidence\n")
    lines.append("- `build_features_v2(season)` calls `load_games(season)`, `load_boxscores(season)`, etc.")
    lines.append("- Each call reads from `s3://hoops-edge/silver/*/season={season}/`")
    lines.append("- Rolling averages are computed within a single season's data only")
    lines.append("- EWM `.shift(1)` ensures the current game is excluded\n")

    # Verify by checking first-game NaN rates across seasons
    seasons = sorted(data.keys())
    if len(seasons) >= 2:
        lines.append("### First-Game NaN Rate by Season (should be consistently high)\n")
        lines.append("| Season | First Row NaN Rate |")
        lines.append("|--------|--------------------|")
        for season in seasons:
            df = data[season]
            if len(df) == 0:
                continue
            feat_cols = [f for f in FEATURE_ORDER_V3 if f in df.columns]
            feat_df = df[feat_cols]
            if "startDate" in df.columns:
                sort_df = df.copy()
                sort_df["_dt"] = pd.to_datetime(sort_df["startDate"], errors="coerce")
                sort_idx = sort_df.sort_values("_dt").index
                feat_df = feat_df.loc[sort_idx]
            first_nan = feat_df.iloc[0].isna().mean()
            lines.append(f"| {season} | {first_nan:.1%} |")
        lines.append("")

    return lines


# ── Plots ────────────────────────────────────────────────────────────


def generate_plots(data: dict[int, pd.DataFrame]):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    all_df = pd.concat(data.values(), ignore_index=True)
    plot_dir = PROJECT_ROOT / "docs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Luck distribution histograms
    luck_cols = [f for f in LUCK_FEATURES if f in all_df.columns]
    if luck_cols:
        fig, axes = plt.subplots(1, len(luck_cols), figsize=(5 * len(luck_cols), 4))
        if len(luck_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, luck_cols):
            vals = all_df[col].dropna()
            ax.hist(vals, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
            ax.set_title(col)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(plot_dir / "luck_distributions.png", dpi=150)
        plt.close()

    # NaN rates per season
    seasons = sorted(data.keys())
    nan_rates = []
    for season in seasons:
        df = data[season]
        feat_cols = [f for f in FEATURE_ORDER_V3 if f in df.columns]
        if feat_cols:
            nan_rates.append(df[feat_cols].isna().mean().mean() * 100)
        else:
            nan_rates.append(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(seasons)), nan_rates, tick_label=[str(s) for s in seasons])
    ax.set_ylabel("NaN Rate (%)")
    ax.set_title("Overall NaN Rate by Season")
    ax.set_ylim(0, max(nan_rates) * 1.2 if nan_rates else 100)
    for i, v in enumerate(nan_rates):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "nan_rates_by_season.png", dpi=150)
    plt.close()

    # Feature variance plot (top and bottom features)
    feat_cols = [f for f in FEATURE_ORDER_V3 if f in all_df.columns]
    feat_df = all_df[feat_cols]
    variances = feat_df.var().sort_values(ascending=False)
    # Normalized variance (coefficient of variation)
    means = feat_df.mean()
    cv = (feat_df.std() / means.abs().replace(0, np.nan)).dropna().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    top20_cv = cv.head(20)
    ax.barh(range(len(top20_cv)), top20_cv.values)
    ax.set_yticks(range(len(top20_cv)))
    ax.set_yticklabels(top20_cv.index, fontsize=7)
    ax.set_xlabel("Coefficient of Variation")
    ax.set_title("Top 20 Features by Coefficient of Variation")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_variability.png", dpi=150)
    plt.close()


# ── Main Report Generation ───────────────────────────────────────────


def generate_report(data: dict[int, pd.DataFrame], output_path: Path):
    """Generate the full validation report."""
    lines = []

    lines.append("# Session 14 V3 Feature Validation Report\n")
    lines.append(f"- **Total seasons**: {len(data)}")
    total_games = sum(len(df) for df in data.values())
    lines.append(f"- **Total games**: {total_games}")
    lines.append(f"- **Expected features**: {len(FEATURE_ORDER_V3)}")
    lines.append(f"- **Seasons**: {', '.join(str(s) for s in sorted(data.keys()))}")
    lines.append("")

    # Run all checks
    lines.extend(check_shape_completeness(data))
    lines.extend(check_distributions(data))
    lines.extend(check_luck_features(data))
    lines.extend(check_correlations(data))
    lines.extend(check_composites(data))
    lines.extend(check_ewm_spans())
    lines.extend(check_leakage(data))
    lines.extend(check_season_boundaries(data))

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default=None,
                        help="Comma-separated seasons (default: all available)")
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "session14_validation_report.md"))
    args = parser.parse_args()

    features_dir = Path(args.features_dir)

    # Discover available seasons
    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",")]
    else:
        # Auto-discover from parquet files
        files = sorted(features_dir.glob("season_*_v3_features.parquet"))
        seasons = [int(f.stem.split("_")[1]) for f in files]

    if not seasons:
        print("No seasons found. Run build_features_v3.py first.")
        sys.exit(1)

    print(f"Loading V3 features for seasons: {seasons}")
    data = load_all_seasons(features_dir, seasons)
    print(f"Loaded {len(data)} seasons, {sum(len(df) for df in data.values())} total games")

    if not data:
        print("No data loaded. Check features directory.")
        sys.exit(1)

    # Generate report
    generate_report(data, Path(args.output))

    # Generate plots
    print("Generating plots...")
    generate_plots(data)
    print("Done.")


if __name__ == "__main__":
    main()

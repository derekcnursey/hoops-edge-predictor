#!/usr/bin/env python3
"""Session 16: 3-way walk-forward comparison of V1 vs V4 vs V5.

Reuses the training/prediction/metrics infrastructure from session15_walkforward.
Adds V5 loader and produces 3-way comparison summary.

Usage:
    poetry run python scripts/session16_walkforward.py
    poetry run python scripts/session16_walkforward.py --models v1,v5   # subset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from scripts.session15_walkforward import (
    TEST_YEARS, WINNER_HP,
    load_v1_season, load_v4_season,
    run_walkforward, compute_metrics,
)


def load_v5_season(season: int):
    """Load V5 features (86-feature parquet) for a season."""
    import pandas as pd
    path = config.FEATURES_DIR / f"season_{season}_v5_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"V5 features not found: {path}")
    return pd.read_parquet(path)


def print_3way_summary(results_by_model: dict[str, list]):
    """Print side-by-side 3-way comparison."""
    models = list(results_by_model.keys())
    n_models = len(models)

    print(f"\n{'='*90}")
    print(f"  3-WAY SUMMARY: {' vs '.join(models)}")
    print(f"{'='*90}")

    # Collect per-model stats
    model_maes = {m: [] for m in models}
    model_rmses = {m: [] for m in models}
    model_ats_units = {m: 0.0 for m in models}
    model_ats_bets = {m: 0 for m in models}
    model_ats_wins = {m: 0 for m in models}

    # Header
    header_parts = [f"{'Year':>6}"]
    for m in models:
        header_parts.extend([f"{m+' MAE':>10}", f"{m+' RMSE':>11}"])
    header = " ".join(header_parts)
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for i, ty in enumerate(TEST_YEARS):
        parts = [f"{ty:>6}"]
        all_ok = True

        for m in models:
            r = results_by_model[m]
            if i < len(r) and r[i].get("status") == "ok":
                parts.append(f"{r[i]['mae']:>10.2f}")
                parts.append(f"{r[i]['rmse']:>11.2f}")
                model_maes[m].append(r[i]["mae"])
                model_rmses[m].append(r[i]["rmse"])
                model_ats_units[m] += r[i]["ats"]["units"]
                model_ats_bets[m] += r[i]["ats"]["bets"]
                model_ats_wins[m] += r[i]["ats"]["wins"]
            else:
                parts.extend([f"{'SKIP':>10}", f"{'':>11}"])
                all_ok = False

        print(f"  {' '.join(parts)}")

    # Averages
    avg_parts = [f"{'AVG':>6}"]
    for m in models:
        if model_maes[m]:
            avg_parts.append(f"{np.mean(model_maes[m]):>10.2f}")
            avg_parts.append(f"{np.mean(model_rmses[m]):>11.2f}")
        else:
            avg_parts.extend([f"{'---':>10}", f"{'---':>11}"])
    print(f"\n  {' '.join(avg_parts)}")

    # ATS summary
    print(f"\n  Pooled ATS:")
    for m in models:
        b = model_ats_bets[m]
        w = model_ats_wins[m]
        u = model_ats_units[m]
        if b > 0:
            roi = u / b * 100
            print(f"    {m}: {w}/{b} ({w/b:.1%}) ROI={roi:+.1f}% units={u:+.1f}")
        else:
            print(f"    {m}: no bets")

    # Verdict
    print(f"\n  {'='*70}")
    if len(models) >= 2 and all(model_maes[m] for m in models):
        avg_mae_by_model = {m: np.mean(model_maes[m]) for m in models}
        best_mae_model = min(avg_mae_by_model, key=avg_mae_by_model.get)
        print(f"  BEST MAE: {best_mae_model} ({avg_mae_by_model[best_mae_model]:.3f})")

        avg_rmse_by_model = {m: np.mean(model_rmses[m]) for m in models}
        best_rmse_model = min(avg_rmse_by_model, key=avg_rmse_by_model.get)
        print(f"  BEST RMSE: {best_rmse_model} ({avg_rmse_by_model[best_rmse_model]:.3f})")

        best_ats_model = max(models, key=lambda m: model_ats_units[m])
        print(f"  BEST ATS: {best_ats_model} ({model_ats_units[best_ats_model]:+.1f} units)")

    print(f"  {'='*70}")


FEATURE_COUNTS = {
    "V1": len(config.FEATURE_ORDER),
    "V4": len(config.FEATURE_ORDER_V4),
    "V5": len(config.FEATURE_ORDER_V5),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="v1,v4,v5",
                        help="Comma-separated model versions to compare")
    args = parser.parse_args()
    model_set = {m.strip().upper() for m in args.models.split(",")}

    print("=" * 70)
    print("  SESSION 16: 3-WAY WALK-FORWARD COMPARISON")
    print("=" * 70)
    print(f"  Test years: {TEST_YEARS}")
    print(f"  Models: {sorted(model_set)}")
    for m in sorted(model_set):
        print(f"    {m}: {FEATURE_COUNTS.get(m, '?')} features")
    print(f"  HP: {WINNER_HP}")

    results = {}

    if "V1" in model_set:
        results["V1"] = run_walkforward(
            "V1", load_v1_season, config.FEATURE_ORDER, hp=WINNER_HP,
        )

    if "V4" in model_set:
        results["V4"] = run_walkforward(
            "V4", load_v4_season, config.FEATURE_ORDER_V4, hp=WINNER_HP,
        )

    if "V5" in model_set:
        results["V5"] = run_walkforward(
            "V5", load_v5_season, config.FEATURE_ORDER_V5, hp=WINNER_HP,
        )

    print_3way_summary(results)

    # Save results
    out_path = config.ARTIFACTS_DIR / "session16_walkforward_results.json"
    serializable = {}
    for m, res in results.items():
        serializable[m.lower()] = res
    serializable["config"] = {
        "test_years": TEST_YEARS,
        "models": sorted(model_set),
        "hp": WINNER_HP,
        "feature_counts": {m: FEATURE_COUNTS.get(m) for m in sorted(model_set)},
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()

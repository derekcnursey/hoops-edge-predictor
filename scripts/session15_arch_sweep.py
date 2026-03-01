#!/usr/bin/env python3
"""Quick architecture sweep for V4 model on the most recent fold.

Tests several architectures on train=2015-2024, test=2025.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from scripts.session15_walkforward import (
    load_v4_season, load_multi, merge_book_spreads,
    train_model, predict, compute_metrics,
    MIN_MONTH_DAY,
)
from src.trainer import impute_column_means

ARCHITECTURES = [
    # name, hparams
    ("baseline (384/256)", {"hidden1": 384, "hidden2": 256, "dropout": 0.20, "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4}),
    ("wider (512/384)", {"hidden1": 512, "hidden2": 384, "dropout": 0.20, "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4}),
    ("wider (512/256)", {"hidden1": 512, "hidden2": 256, "dropout": 0.25, "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4}),
    ("narrow (256/128)", {"hidden1": 256, "hidden2": 128, "dropout": 0.30, "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4}),
    ("low-dropout (384/256)", {"hidden1": 384, "hidden2": 256, "dropout": 0.10, "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4}),
    ("high-wd (384/256)", {"hidden1": 384, "hidden2": 256, "dropout": 0.20, "lr": 3e-3, "batch_size": 4096, "weight_decay": 5e-4}),
    ("slower-lr (384/256)", {"hidden1": 384, "hidden2": 256, "dropout": 0.20, "lr": 1e-3, "batch_size": 4096, "weight_decay": 1e-4}),
    ("big-batch (384/256)", {"hidden1": 384, "hidden2": 256, "dropout": 0.20, "lr": 3e-3, "batch_size": 2048, "weight_decay": 1e-4}),
]


def main():
    print("=" * 70)
    print("  ARCHITECTURE SWEEP: V4 (136 features)")
    print("  Train: 2015-2024, Test: 2025")
    print("=" * 70)

    train_seasons = list(range(2015, 2025))
    val_seasons = [2025]

    # Load data once
    print("\nLoading data...")
    X_tr, y_tr, _ = load_multi(train_seasons, load_v4_season, config.FEATURE_ORDER_V4)
    X_val, y_val, df_val = load_multi(val_seasons, load_v4_season, config.FEATURE_ORDER_V4)
    book = merge_book_spreads(df_val, val_seasons)

    X_tr = impute_column_means(X_tr)
    X_val = impute_column_means(X_val)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    print(f"  Train: {len(y_tr)}, Val: {len(y_val)}, Book: {(~np.isnan(book)).sum()}")

    # Run sweep (3 seeds per architecture for stability)
    results = []
    for name, hp in ARCHITECTURES:
        print(f"\n  --- {name} ---")
        maes, rmses, epochs = [], [], []
        for seed in range(3):
            import torch
            torch.manual_seed(seed)
            np.random.seed(seed)

            t0 = time.time()
            model, best_ep = train_model(X_tr_s, y_tr, X_val_s, y_val, hp=hp, verbose=False)
            mu, sigma = predict(model, X_val_s)
            elapsed = time.time() - t0

            metrics = compute_metrics(mu, sigma, y_val, book)
            maes.append(metrics["mae"])
            rmses.append(metrics["rmse"])
            epochs.append(best_ep)

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_mae = np.mean(maes)
        avg_rmse = np.mean(rmses)
        std_mae = np.std(maes)
        print(f"    MAE={avg_mae:.3f}±{std_mae:.3f}  RMSE={avg_rmse:.3f}  "
              f"epochs={epochs}")

        results.append({
            "name": name,
            "hp": hp,
            "avg_mae": avg_mae,
            "std_mae": std_mae,
            "avg_rmse": avg_rmse,
            "maes": maes,
            "epochs": epochs,
        })

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Architecture':<30} {'MAE':>10} {'±std':>8} {'RMSE':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*10}")

    results.sort(key=lambda r: r["avg_mae"])
    for r in results:
        marker = " <-- BEST" if r == results[0] else ""
        print(f"  {r['name']:<30} {r['avg_mae']:>10.3f} {r['std_mae']:>8.3f} "
              f"{r['avg_rmse']:>10.3f}{marker}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train production V4 model: 136 features, seasons 2015-2025.

Architecture: 512/256 MLP with 25% dropout (session 15 sweep winner).
Saves to checkpoints/v4/ with V4 feature order embedded.

Usage:
    poetry run python scripts/train_production_v4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.features import get_feature_matrix, get_targets
from src.trainer import (
    fit_scaler, impute_column_means, save_checkpoint,
    train_regressor, train_classifier,
)

SEASONS = list(range(2015, 2026))
SUBDIR = "v4"

# Session 15 sweep winner (512/256, 25% dropout)
V4_REG_HP = {
    "hidden1": 512,
    "hidden2": 256,
    "dropout": 0.25,
    "lr": 3e-3,
    "batch_size": 4096,
    "weight_decay": 1e-4,
    "epochs": 150,
}

V4_CLS_HP = {
    "hidden1": 256,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 4096,
}


def load_v4_features(seasons: list[int]) -> pd.DataFrame:
    """Load and concatenate V4 feature parquets."""
    dfs = []
    for s in seasons:
        path = config.FEATURES_DIR / f"season_{s}_v4_features.parquet"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True)


def main():
    print("=" * 60)
    print("  PRODUCTION V4 MODEL TRAINING")
    print("=" * 60)
    print(f"  Seasons: {SEASONS}")
    print(f"  Features: {len(config.FEATURE_ORDER_V4)}")
    print(f"  Regressor HP: {V4_REG_HP}")
    print(f"  Classifier HP: {V4_CLS_HP}")

    # Load features
    print("\nLoading V4 features...")
    df = load_v4_features(SEASONS)
    df = df.dropna(subset=["homeScore", "awayScore"])
    df = df[(df["homeScore"] != 0) | (df["awayScore"] != 0)]
    print(f"  Training samples: {len(df)}")

    # Extract feature matrix and targets
    X = get_feature_matrix(df, feature_order=config.FEATURE_ORDER_V4).values.astype(np.float32)
    targets = get_targets(df)
    y_spread = targets["spread_home"].values.astype(np.float32)
    y_win = targets["home_win"].values.astype(np.float32)

    # NaN analysis
    nan_count = np.isnan(X).sum()
    nan_pct = nan_count / X.size * 100
    print(f"  NaN values: {nan_count} ({nan_pct:.1f}%)")

    # Impute with column means (better than nan_to_num for scaling)
    X = impute_column_means(X)

    # Fit scaler
    print("\nFitting StandardScaler...")
    scaler = fit_scaler(X, subdir=SUBDIR)
    X_scaled = scaler.transform(X).astype(np.float32)

    # Train regressor
    print("\nTraining MLPRegressor (Gaussian NLL, 512/256)...")
    regressor = train_regressor(X_scaled, y_spread, hparams=V4_REG_HP)
    save_checkpoint(regressor, "regressor", hparams=V4_REG_HP,
                    subdir=SUBDIR, feature_order=config.FEATURE_ORDER_V4)

    # Train classifier
    print("\nTraining MLPClassifier (BCE)...")
    classifier = train_classifier(X_scaled, y_win, hparams=V4_CLS_HP)
    save_checkpoint(classifier, "classifier", hparams=V4_CLS_HP,
                    subdir=SUBDIR, feature_order=config.FEATURE_ORDER_V4)

    # Save best hparams
    hp_path = config.ARTIFACTS_DIR / SUBDIR / "best_hparams.json"
    hp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hp_path, "w") as f:
        json.dump({"regressor": V4_REG_HP, "classifier": V4_CLS_HP}, f, indent=2)
    print(f"  Saved hparams: {hp_path}")

    print(f"\n{'='*60}")
    print("  PRODUCTION V4 TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Scaler:     {config.ARTIFACTS_DIR / SUBDIR / 'scaler.pkl'}")
    print(f"  Regressor:  {config.CHECKPOINTS_DIR / SUBDIR / 'regressor.pt'}")
    print(f"  Classifier: {config.CHECKPOINTS_DIR / SUBDIR / 'classifier.pt'}")
    print(f"  Hparams:    {hp_path}")


if __name__ == "__main__":
    main()

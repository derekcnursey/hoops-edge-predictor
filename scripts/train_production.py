#!/usr/bin/env python3
"""Train production model: Torvik efficiencies, a0.85_p10 adjusted, 53 features, seasons 2015-2025."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset import load_multi_season_features
from src.features import get_feature_matrix, get_targets
from src.trainer import fit_scaler, impute_column_means, save_checkpoint, train_classifier, train_regressor

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
SEASONS = list(range(2015, 2026))
VAL_FRAC = 0.15  # best-loss checkpointing

# Load best hparams from session 12
with open(config.ARTIFACTS_DIR / "best_hparams.json") as f:
    best_hp = json.load(f)

reg_hp = best_hp["regressor"]
cls_hp = best_hp["classifier"]

print(f"=== Production Training ===")
print(f"Seasons: {SEASONS}")
print(f"Efficiency source: {config.EFFICIENCY_SOURCE}")
print(f"Adj suffix: {ADJ_SUFFIX}")
print(f"Features: {len(config.FEATURE_ORDER)}")
print(f"Val frac: {VAL_FRAC} (best-loss checkpointing)")
print(f"Regressor HP: {reg_hp}")
print(f"Classifier HP: {cls_hp}")

# Load all adjusted parquets
print(f"\nLoading features...")
df = load_multi_season_features(
    SEASONS, no_garbage=True, adj_suffix=ADJ_SUFFIX,
    efficiency_source=config.EFFICIENCY_SOURCE,
)
df = df.dropna(subset=["homeScore", "awayScore"])
print(f"  Training samples: {len(df)}")

# Extract feature matrix and targets
X = get_feature_matrix(df).values.astype(np.float32)
targets = get_targets(df)
y_spread = targets["spread_home"].values.astype(np.float32)
y_win = targets["home_win"].values.astype(np.float32)

# Impute NaN with per-column means (matches inference pipeline)
nan_count = np.isnan(X).sum()
print(f"  NaN values: {nan_count}")
X = impute_column_means(X)

# Fit scaler (saves to artifacts/scaler.pkl)
print("\nFitting StandardScaler...")
scaler = fit_scaler(X)
X_scaled = scaler.transform(X)

# Train regressor with best-loss checkpointing
print("\nTraining MLPRegressor (Gaussian NLL)...")
reg_hp_full = {**reg_hp, "epochs": 150}
regressor = train_regressor(X_scaled, y_spread, hparams=reg_hp_full, val_frac=VAL_FRAC)
save_checkpoint(regressor, "regressor", hparams=reg_hp,
                feature_order=config.FEATURE_ORDER)

# Train classifier with best-loss checkpointing
print("\nTraining MLPClassifier (BCE)...")
cls_hp_full = {**cls_hp, "epochs": 150}
classifier = train_classifier(X_scaled, y_win, hparams=cls_hp_full, val_frac=VAL_FRAC)
save_checkpoint(classifier, "classifier", hparams=cls_hp,
                feature_order=config.FEATURE_ORDER)

print("\n=== Production training complete ===")
print(f"  Efficiency source: {config.EFFICIENCY_SOURCE}")
print(f"  Scaler: {config.ARTIFACTS_DIR / 'scaler.pkl'}")
print(f"  Regressor: {config.CHECKPOINTS_DIR / 'regressor.pt'}")
print(f"  Classifier: {config.CHECKPOINTS_DIR / 'classifier.pt'}")

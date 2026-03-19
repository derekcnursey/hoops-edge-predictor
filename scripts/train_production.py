#!/usr/bin/env python3
"""Train production model: Torvik efficiencies, a0.85_p10 adjusted, 53 features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset import load_multi_season_features
from src.efficiency_blend import blend_enabled
from src.features import get_feature_matrix, get_targets
from src.model_hparams import load_best_hparams, production_mu_hparams
from src.trainer import (
    fit_scaler,
    impute_column_means,
    save_checkpoint,
    save_tree_regressor,
    train_classifier,
    train_hist_gradient_boosting_regressor,
    train_lightgbm_regressor,
    train_regressor,
)

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
SEASONS = [season for season in range(2015, 2026) if season not in config.EXCLUDE_SEASONS]
VAL_FRAC = 0.15  # best-loss checkpointing


def _production_regressor_hparams(base: dict, efficiency_source: str) -> dict:
    """Use a safer sigma fit when training on the gold-backed feature set."""
    hp = {**base, "epochs": 150}
    if efficiency_source == "gold":
        hp["lr"] = min(float(hp.get("lr", 1e-3)), 1e-3)
        hp["batch_size"] = min(int(hp.get("batch_size", 1024)), 1024)
    return hp


def _temporal_train_val_split(
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
    y_spread: np.ndarray,
    y_win: np.ndarray,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_val = max(1, int(len(X_raw) * val_frac))
    n_val = min(n_val, len(X_raw) - 1)
    split_idx = len(X_raw) - n_val
    return (
        X_raw[:split_idx],
        X_raw[split_idx:],
        X_scaled[:split_idx],
        X_scaled[split_idx:],
        y_spread[:split_idx],
        y_spread[split_idx:],
        y_win[:split_idx],
        y_win[split_idx:],
    )

# Load best hparams from artifacts/best_hparams.json when present
best_hp = load_best_hparams()

reg_hp = best_hp["regressor"]
cls_hp = best_hp["classifier"]
mu_hp = production_mu_hparams()

print(f"=== Production Training ===")
print(f"Seasons: {SEASONS}")
print(f"Efficiency source: {config.EFFICIENCY_SOURCE}")
print(f"Adj suffix: {ADJ_SUFFIX}")
print(f"Features: {len(config.FEATURE_ORDER)}")
print(f"Val frac: {VAL_FRAC} (best-loss checkpointing)")
print(f"Regressor HP: {reg_hp}")
print(f"Classifier HP: {cls_hp}")
print(f"Mu HP: {mu_hp}")

# Load all adjusted parquets
print(f"\nLoading features...")
df = load_multi_season_features(
    SEASONS, no_garbage=True, adj_suffix=ADJ_SUFFIX,
    efficiency_source=config.EFFICIENCY_SOURCE,
)
df = df.dropna(subset=["homeScore", "awayScore"])
df = df[(df["homeScore"] != 0) | (df["awayScore"] != 0)]
df["startDate"] = np.array(df["startDate"], dtype="datetime64[ns]")
df = df.sort_values(["startDate", "gameId"]).reset_index(drop=True)
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
(
    X_train_raw,
    X_val_raw,
    X_train_scaled,
    X_val_scaled,
    y_spread_train,
    y_spread_val,
    y_win_train,
    y_win_val,
) = _temporal_train_val_split(X, X_scaled, y_spread, y_win, VAL_FRAC)

# Train regressor with best-loss checkpointing
print("\nTraining MLPRegressor (Gaussian NLL)...")
reg_hp_full = _production_regressor_hparams(reg_hp, config.EFFICIENCY_SOURCE)
regressor = train_regressor(
    X_scaled,
    y_spread,
    hparams=reg_hp_full,
    val_frac=VAL_FRAC,
    temporal_val_split=True,
)
save_checkpoint(regressor, "regressor", hparams=reg_hp_full,
                feature_order=config.FEATURE_ORDER)

# Train production LightGBM L2 regressor for mu on raw imputed features
print("\nTraining LightGBMRegressor (mu)...")
tree_regressor = train_lightgbm_regressor(
    X_train_raw,
    y_spread_train,
    hparams=mu_hp,
    X_val=X_val_raw,
    y_val=y_spread_val,
)
mu_hp_saved = {
    **mu_hp,
    **(
        {"best_iteration": int(tree_regressor.best_iteration_)}
        if getattr(tree_regressor, "best_iteration_", None)
        else {}
    ),
}
tree_path = save_tree_regressor(
    tree_regressor,
    feature_order=config.FEATURE_ORDER,
    hparams=mu_hp_saved,
)

torvik_tree_path = None
if blend_enabled():
    print("\nTraining Torvik LightGBMRegressor (mu blend side)...")
    torvik_df = load_multi_season_features(
        SEASONS, no_garbage=True, adj_suffix=ADJ_SUFFIX,
        efficiency_source="torvik",
    )
    torvik_df = torvik_df.dropna(subset=["homeScore", "awayScore"])
    torvik_df = torvik_df[(torvik_df["homeScore"] != 0) | (torvik_df["awayScore"] != 0)]
    torvik_df["startDate"] = np.array(torvik_df["startDate"], dtype="datetime64[ns]")
    torvik_df = torvik_df.sort_values(["startDate", "gameId"]).reset_index(drop=True)
    X_t = get_feature_matrix(torvik_df).values.astype(np.float32)
    X_t = impute_column_means(X_t)
    y_t = get_targets(torvik_df)["spread_home"].values.astype(np.float32)
    y_t_win = get_targets(torvik_df)["home_win"].values.astype(np.float32)
    X_t_scaled = scaler.transform(X_t)
    X_t_train_raw, X_t_val_raw, _x_t_train_scaled, _x_t_val_scaled, y_t_train, y_t_val, _yw_t_train, _yw_t_val = _temporal_train_val_split(
        X_t,
        X_t_scaled,
        y_t,
        y_t_win,
        VAL_FRAC,
    )
    torvik_tree = train_lightgbm_regressor(
        X_t_train_raw,
        y_t_train,
        hparams=mu_hp,
        X_val=X_t_val_raw,
        y_val=y_t_val,
    )
    torvik_mu_hp_saved = {
        **mu_hp,
        **(
            {"best_iteration": int(torvik_tree.best_iteration_)}
            if getattr(torvik_tree, "best_iteration_", None)
            else {}
        ),
    }
    torvik_tree_path = save_tree_regressor(
        torvik_tree,
        path=config.TORVIK_TREE_REGRESSOR_PATH,
        feature_order=config.FEATURE_ORDER,
        hparams=torvik_mu_hp_saved,
    )

# Train classifier with best-loss checkpointing
print("\nTraining MLPClassifier (BCE)...")
cls_hp_full = {**cls_hp, "epochs": 150}
classifier = train_classifier(
    X_scaled,
    y_win,
    hparams=cls_hp_full,
    val_frac=VAL_FRAC,
    temporal_val_split=True,
)
save_checkpoint(classifier, "classifier", hparams=cls_hp_full,
                feature_order=config.FEATURE_ORDER)

print("\n=== Production training complete ===")
print(f"  Efficiency source: {config.EFFICIENCY_SOURCE}")
print(f"  Scaler: {config.ARTIFACTS_DIR / 'scaler.pkl'}")
print(f"  Regressor: {config.CHECKPOINTS_DIR / 'regressor.pt'}")
print(f"  Tree regressor: {tree_path}")
if torvik_tree_path is not None:
    print(f"  Torvik tree regressor: {torvik_tree_path}")
print(f"  Classifier: {config.CHECKPOINTS_DIR / 'classifier.pt'}")

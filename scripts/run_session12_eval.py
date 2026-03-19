#!/usr/bin/env python3
"""Session 12: GPU Evaluation — Opponent-Adjusted Four-Factors

Runs 27 training configurations:
  - 8 adjusted combos × 3 feature sets = 24 adjusted runs
  - 3 raw (unadjusted) controls
Each with 50-trial Optuna hyperparameter search.

Then for the best adjusted combo:
  - Permutation importance
  - Backward elimination
  - Forward selection
  - Full metrics (MAE, ROI, calibration, monthly breakdown)

Usage:
    poetry run python -u scripts/run_session12_eval.py [--skip-optuna] [--resume]
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Force unbuffered output
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset
from src.features import get_feature_matrix, get_targets
from src.trainer import train_regressor

# ── Constants ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / ".session12_state"
REPORT_PATH = PROJECT_ROOT / "reports" / "adjusted_ff_evaluation.md"

TRAIN_SEASONS = list(range(2015, 2025))
HOLDOUT_SEASON = 2025

# Feature configs
V1_FEATURES: list[str] = json.loads(
    (config.ARTIFACTS_DIR / "feature_order_v1.json").read_text()
)  # 37 features

V2_FEATURES_RAW: list[str] = json.loads(
    (config.ARTIFACTS_DIR / "feature_order_v2.json").read_text()
)  # 55 features
FORCE_REMOVE = {"away_team_home"}
V2_FEATURES = [f for f in V2_FEATURES_RAW if f not in FORCE_REMOVE]  # 54 features

V0_FEATURES: list[str] = json.loads(
    (config.ARTIFACTS_DIR / "feature_order.json").read_text()
)  # 10 features (pruned)

FEATURE_CONFIGS = {
    "37feat": V1_FEATURES,
    "54feat": V2_FEATURES,
    "10feat": V0_FEATURES,
}

# 8 adjusted combos
ADJ_COMBOS = [
    {"alpha": 1.0,  "prior": 5,  "suffix": ""},
    {"alpha": 0.85, "prior": 5,  "suffix": "_a0.85_p5"},
    {"alpha": 0.7,  "prior": 5,  "suffix": "_a0.7_p5"},
    {"alpha": 0.5,  "prior": 5,  "suffix": "_a0.5_p5"},
    {"alpha": 1.0,  "prior": 3,  "suffix": "_a1.0_p3"},
    {"alpha": 1.0,  "prior": 10, "suffix": "_a1.0_p10"},
    {"alpha": 1.0,  "prior": 15, "suffix": "_a1.0_p15"},
    {"alpha": 0.85, "prior": 10, "suffix": "_a0.85_p10"},
]

# Known raw baselines
RAW_BASELINES = {
    "37feat": {"mae_book": 9.62, "mae_overall": None},
    "54feat": {"mae_book": 9.38, "mae_overall": None},
    "10feat": {"mae_book": 9.48, "mae_overall": None},
}

# Default hparams (used if Optuna is skipped)
DEFAULT_HP = {
    "hidden1": 256,
    "hidden2": 128,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 256,
}

OPTUNA_EPOCHS = 50
FINAL_EPOCHS = 100
N_FOLDS = 3
N_OPTUNA_TRIALS = 50

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Data Loading ─────────────────────────────────────────────────


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_combo_data(suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training + holdout data for an adjusted combo."""
    train_dfs = []
    for s in TRAIN_SEASONS:
        path = config.FEATURES_DIR / f"season_{s}_no_garbage_adj{suffix}_features.parquet"
        if path.exists():
            train_dfs.append(pd.read_parquet(path))
        else:
            raise FileNotFoundError(f"Missing: {path}")
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])

    holdout_path = config.FEATURES_DIR / f"season_{HOLDOUT_SEASON}_no_garbage_adj{suffix}_features.parquet"
    holdout_df = pd.read_parquet(holdout_path)
    holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])

    return train_df, holdout_df


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load raw (unadjusted) training + holdout data if available."""
    train_dfs = []
    for s in TRAIN_SEASONS:
        path = config.FEATURES_DIR / f"season_{s}_no_garbage_features.parquet"
        if not path.exists():
            return None
        train_dfs.append(pd.read_parquet(path))
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])

    holdout_path = config.FEATURES_DIR / f"season_{HOLDOUT_SEASON}_no_garbage_features.parquet"
    if not holdout_path.exists():
        return None
    holdout_df = pd.read_parquet(holdout_path)
    holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])

    return train_df, holdout_df


def prepare_data(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Extract features, handle NaN, scale."""
    X_train = get_feature_matrix(train_df, feature_order=feature_cols).values.astype(np.float32)
    targets = get_targets(train_df)
    y_spread = targets["spread_home"].values.astype(np.float32)

    X_holdout = get_feature_matrix(holdout_df, feature_order=feature_cols).values.astype(np.float32)
    targets_h = get_targets(holdout_df)
    y_test = targets_h["spread_home"].values.astype(np.float32)

    # Handle NaN in training
    nan_mask = np.isnan(X_train)
    if nan_mask.any():
        col_means = np.nanmean(X_train, axis=0)
        for j in range(X_train.shape[1]):
            X_train[nan_mask[:, j], j] = col_means[j]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Handle NaN in holdout
    nan_mask_h = np.isnan(X_holdout)
    if nan_mask_h.any():
        for j in range(X_holdout.shape[1]):
            X_holdout[nan_mask_h[:, j], j] = scaler.mean_[j]
    X_holdout_scaled = scaler.transform(X_holdout)

    return X_train_scaled, y_spread, X_holdout_scaled, y_test, scaler


# ── Book Spreads ─────────────────────────────────────────────────


def load_book_spreads() -> pd.DataFrame | None:
    """Load book spreads for holdout season."""
    # Try cached lines parquet first
    cache_path = config.FEATURES_DIR / f"lines_{HOLDOUT_SEASON}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Try S3
    try:
        from src.features import load_research_lines
        lines = load_research_lines(HOLDOUT_SEASON)
        if lines is not None and not lines.empty:
            # Cache for future use
            lines.to_parquet(cache_path)
            return lines
    except Exception:
        pass

    return None


def attach_book_spreads(preds_df: pd.DataFrame, lines: pd.DataFrame | None) -> pd.DataFrame:
    """Attach book spreads to predictions."""
    if lines is not None and not lines.empty:
        lines_dedup = lines.sort_values("provider").drop_duplicates(
            subset=["gameId"], keep="first")
        preds_df = preds_df.merge(
            lines_dedup[["gameId", "spread"]].rename(columns={"spread": "book_spread"}),
            on="gameId", how="left")
        if "predicted_spread" in preds_df.columns:
            preds_df["model_spread"] = -preds_df["predicted_spread"]
            preds_df["spread_diff"] = preds_df["model_spread"] - preds_df["book_spread"]
    else:
        preds_df["book_spread"] = np.nan
    return preds_df


# ── Optuna Tuning ────────────────────────────────────────────────


def run_optuna_search(
    X_train: np.ndarray,
    y_spread: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
    label: str = "",
) -> dict:
    """Run Optuna hyperparameter search for regressor.

    Uses direct GPU tensor batching (no DataLoader) for speed on tabular data.
    """
    device = _get_device()
    use_amp = device.type == "cuda"

    # Pre-load ALL data to GPU as tensors (avoids CPU-GPU transfer per batch)
    X_gpu = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_gpu = torch.tensor(y_spread, dtype=torch.float32, device=device)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_splits = list(kf.split(X_train))

    trial_times = []

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        hp = {
            "hidden1": trial.suggest_categorical("hidden1", [128, 256, 512]),
            "hidden2": trial.suggest_categorical("hidden2", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": OPTUNA_EPOCHS,
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        }
        val_losses = []
        for train_idx, val_idx in fold_splits:
            train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
            val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

            X_fold = X_gpu[train_idx_t]
            y_fold = y_gpu[train_idx_t]
            X_val = X_gpu[val_idx_t]
            y_val = y_gpu[val_idx_t]

            n_train = len(train_idx)
            bs = hp["batch_size"]

            model = MLPRegressor(
                input_dim=X_train.shape[1],
                hidden1=hp["hidden1"],
                hidden2=hp["hidden2"],
                dropout=hp["dropout"],
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
            amp_scaler = GradScaler(device.type, enabled=use_amp)

            model.train()
            for _ in range(hp["epochs"]):
                perm = torch.randperm(n_train, device=device)
                for start in range(0, n_train - bs + 1, bs):
                    idx = perm[start:start + bs]
                    x_batch = X_fold[idx]
                    y_batch = y_fold[idx]
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device.type, enabled=use_amp):
                        mu, raw_sigma = model(x_batch)
                        loss = gaussian_nll_loss(mu, raw_sigma, y_batch)
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()

            model.eval()
            with torch.no_grad():
                with autocast(device.type, enabled=use_amp):
                    mu_v, raw_sigma_v = model(X_val)
                    val_loss = gaussian_nll_loss(mu_v, raw_sigma_v, y_val).item()
            val_losses.append(val_loss)
            del model, amp_scaler

        elapsed = time.time() - t0
        trial_times.append(elapsed)
        if len(trial_times) % 10 == 0:
            avg = np.mean(trial_times[-10:])
            remaining = (n_trials - len(trial_times)) * avg
            print(f"      Trial {len(trial_times)}/{n_trials}: "
                  f"loss={np.mean(val_losses):.4f} ({elapsed:.1f}s, ~{remaining/60:.0f}m left)")

        return float(np.mean(val_losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Cleanup GPU memory
    del X_gpu, y_gpu
    torch.cuda.empty_cache()

    best = study.best_trial.params
    best["epochs"] = FINAL_EPOCHS
    avg_trial = np.mean(trial_times) if trial_times else 0
    print(f"    Optuna best ({label}): val_loss={study.best_trial.value:.4f} | "
          f"h1={best['hidden1']} h2={best['hidden2']} "
          f"drop={best['dropout']:.3f} lr={best['lr']:.5f} "
          f"wd={best['weight_decay']:.6f} bs={best['batch_size']} "
          f"({avg_trial:.1f}s/trial)")
    return best


# ── Train & Evaluate ─────────────────────────────────────────────


def train_and_evaluate(
    X_train: np.ndarray,
    y_spread: np.ndarray,
    X_holdout: np.ndarray,
    y_test: np.ndarray,
    hparams: dict,
    holdout_df: pd.DataFrame,
    lines: pd.DataFrame | None = None,
) -> dict:
    """Train regressor with given hparams, evaluate on holdout."""
    reg = train_regressor(X_train, y_spread, hparams=hparams)

    reg.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_holdout, dtype=torch.float32)
        mu_t, log_sigma_t = reg(X_t)
        sigma_t = torch.nn.functional.softplus(log_sigma_t) + 1e-3
        sigma_t = sigma_t.clamp(min=0.5, max=30.0)

    mu = mu_t.numpy().flatten()
    sigma = sigma_t.numpy().flatten()

    mae_overall = float(np.mean(np.abs(mu - y_test)))

    # Build predictions DataFrame
    preds = holdout_df[["gameId", "homeTeamId", "awayTeamId",
                        "homeScore", "awayScore", "startDate"]].copy()
    preds["predicted_spread"] = mu
    preds["spread_sigma"] = sigma
    preds["actual_margin"] = y_test

    preds = attach_book_spreads(preds, lines)
    mae_book = None
    n_book = 0
    with_book = preds.dropna(subset=["book_spread"])
    if len(with_book) > 0:
        mae_book = float(np.abs(with_book["predicted_spread"] - with_book["actual_margin"]).mean())
        n_book = len(with_book)

    return {
        "mae_overall": mae_overall,
        "mae_book": mae_book,
        "n_book": n_book,
        "mu": mu,
        "sigma": sigma,
        "preds": preds,
        "reg": reg,
    }


# ── Metrics ──────────────────────────────────────────────────────


def compute_roi(preds_df: pd.DataFrame, threshold: float, sigma_filter: float | None = None) -> dict:
    """Compute ATS ROI at a given threshold."""
    with_book = preds_df.dropna(subset=["book_spread"]).copy()
    if sigma_filter is not None:
        with_book = with_book[with_book["spread_sigma"] < sigma_filter]
    if len(with_book) == 0:
        return {"bets": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "roi": 0.0}

    with_book["edge"] = with_book["model_spread"] - with_book["book_spread"]
    bets = with_book[with_book["edge"].abs() >= threshold].copy()
    if len(bets) == 0:
        return {"bets": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "roi": 0.0}

    bets["bet_side"] = np.sign(bets["edge"])
    bets["cover"] = np.sign(bets["actual_margin"] + bets["book_spread"])
    bets["win"] = (bets["bet_side"] == bets["cover"]).astype(int)
    bets["push"] = (bets["cover"] == 0).astype(int)

    non_push = bets[bets["push"] == 0]
    wins = int(non_push["win"].sum())
    losses = len(non_push) - wins
    wr = wins / len(non_push) if len(non_push) > 0 else 0
    roi = (wins - losses * 1.1) / len(non_push) if len(non_push) > 0 else 0

    return {"bets": len(non_push), "wins": wins, "losses": losses,
            "win_rate": wr, "roi": roi}


def compute_calibration(preds_df: pd.DataFrame) -> dict:
    """Compute calibration by probability bucket."""
    valid = preds_df.dropna(subset=["predicted_spread"]).copy()
    if len(valid) == 0:
        return {"within_1sigma": None, "within_2sigma": None, "buckets": []}

    residual = np.abs(valid["predicted_spread"].values - valid["actual_margin"].values)
    sigma = valid["spread_sigma"].values

    within_1 = float((residual <= sigma).mean())
    within_2 = float((residual <= 2 * sigma).mean())

    # Bucket by predicted probability of being within threshold
    buckets = []
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        # Use normalized z-score = residual/sigma
        z = residual / np.clip(sigma, 0.5, None)
        mask = (z >= lo * 3) & (z < hi * 3)  # map to z-score range
        if mask.sum() > 0:
            buckets.append({
                "range": f"{lo:.1f}-{hi:.1f}",
                "count": int(mask.sum()),
                "mean_residual": float(residual[mask].mean()),
                "mean_sigma": float(sigma[mask].mean()),
            })

    return {"within_1sigma": within_1, "within_2sigma": within_2, "buckets": buckets}


def compute_monthly_mae(preds_df: pd.DataFrame, use_book: bool = True) -> list[dict]:
    """Compute MAE broken down by month."""
    valid = preds_df.copy()
    if use_book:
        valid = valid.dropna(subset=["book_spread"])
    if len(valid) == 0:
        return []

    valid["month"] = pd.to_datetime(valid["startDate"], errors="coerce").dt.to_period("M")
    results = []
    for month, group in valid.groupby("month"):
        mae = float(np.abs(group["predicted_spread"] - group["actual_margin"]).mean())
        results.append({"month": str(month), "mae": mae, "games": len(group)})
    return results


# ── Permutation Importance ───────────────────────────────────────


def permutation_importance(
    feature_cols: list[str],
    X_holdout: np.ndarray,
    y_test: np.ndarray,
    reg: MLPRegressor,
    n_repeats: int = 10,
    book_mask: np.ndarray | None = None,
) -> tuple[list[str], dict]:
    """Compute permutation importance for each feature."""
    print(f"\n--- Permutation Importance ({len(feature_cols)} features, {n_repeats} repeats) ---")

    # Baseline MAE (on book games if available, otherwise all)
    with torch.no_grad():
        X_t = torch.tensor(X_holdout, dtype=torch.float32)
        mu_base, _ = reg(X_t)
    mu_base = mu_base.numpy().flatten()

    if book_mask is not None and book_mask.sum() > 0:
        baseline_mae = float(np.abs(mu_base[book_mask] - y_test[book_mask]).mean())
        eval_mask = book_mask
    else:
        baseline_mae = float(np.abs(mu_base - y_test).mean())
        eval_mask = np.ones(len(y_test), dtype=bool)

    print(f"  Baseline MAE: {baseline_mae:.4f}")

    importance = {}
    for j, feat in enumerate(feature_cols):
        deltas = []
        for rep in range(n_repeats):
            X_perm = X_holdout.copy()
            rng = np.random.RandomState(42 + rep)
            X_perm[:, j] = rng.permutation(X_perm[:, j])

            with torch.no_grad():
                X_t = torch.tensor(X_perm, dtype=torch.float32)
                mu_p, _ = reg(X_t)
            mu_perm = mu_p.numpy().flatten()
            mae_perm = float(np.abs(mu_perm[eval_mask] - y_test[eval_mask]).mean())
            deltas.append(mae_perm - baseline_mae)

        importance[feat] = {"mean": float(np.mean(deltas)), "std": float(np.std(deltas))}
        print(f"  {feat}: +{np.mean(deltas):.4f} (+/- {np.std(deltas):.4f})")

    sorted_feats = sorted(importance.keys(), key=lambda f: importance[f]["mean"], reverse=True)
    return sorted_feats, importance


# ── Backward Elimination ─────────────────────────────────────────


def backward_elimination(
    feature_cols: list[str],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    importance_ranking: list[str],
    lines: pd.DataFrame | None = None,
    min_features: int = 8,
) -> tuple[list[str], float, list[dict]]:
    """Backward elimination: coarse (remove bottom 5) then fine (remove 1)."""
    print(f"\n--- Backward Elimination (from {len(feature_cols)} features) ---")

    current_features = [f for f in feature_cols if f not in FORCE_REMOVE]
    log = []
    search_hp = {**DEFAULT_HP, "epochs": 50}

    def eval_features(feats, label):
        X_tr, y_tr, X_ho, y_ho, sc = prepare_data(train_df, holdout_df, feats)
        result = train_and_evaluate(X_tr, y_tr, X_ho, y_ho, search_hp, holdout_df, lines)
        mae = result["mae_book"] if result["mae_book"] is not None else result["mae_overall"]
        print(f"    {label}: {len(feats)} feats -> MAE={mae:.4f}")
        return mae

    # Initial MAE
    init_mae = eval_features(current_features, "start")
    log.append({"step": "start", "n_features": len(current_features),
                "mae": init_mae, "removed": []})

    ranked = [f for f in importance_ranking if f in current_features]

    # Coarse: remove bottom 5 by importance
    while len(current_features) > 13:
        bottom5 = ranked[-5:]
        candidate = [f for f in current_features if f not in bottom5]
        mae = eval_features(candidate, f"coarse_remove_{len(bottom5)}")
        log.append({"step": f"coarse", "n_features": len(candidate),
                     "mae": mae, "removed": bottom5})
        current_features = candidate
        ranked = [f for f in ranked if f not in bottom5]

    # Fine: remove 1 at a time
    while len(current_features) > min_features:
        best_mae = None
        best_remove = None
        best_features = None

        for feat in list(current_features):
            candidate = [f for f in current_features if f != feat]
            mae = eval_features(candidate, f"try_remove_{feat}")
            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_remove = feat
                best_features = candidate

        print(f"  Best removal: {best_remove} -> MAE={best_mae:.4f}")
        current_features = best_features
        log.append({"step": f"fine_remove_{best_remove}", "n_features": len(current_features),
                     "mae": best_mae, "removed": [best_remove]})

    # Full validation with final hparams
    print(f"\n  Full validation for {len(current_features)} features...")
    X_tr, y_tr, X_ho, y_ho, sc = prepare_data(train_df, holdout_df, current_features)
    result = train_and_evaluate(X_tr, y_tr, X_ho, y_ho, DEFAULT_HP, holdout_df, lines)
    final_mae = result["mae_book"] if result["mae_book"] is not None else result["mae_overall"]

    return current_features, final_mae, log


# ── Forward Selection ─────────────────────────────────────────────


def forward_selection(
    all_features: list[str],
    seed_features: list[str],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    lines: pd.DataFrame | None = None,
    max_add: int = 5,
) -> tuple[list[str], float, list[dict]]:
    """Forward selection: try adding features back to the minimal set."""
    print(f"\n--- Forward Selection (seed={len(seed_features)} features) ---")

    current_features = list(seed_features)
    candidates = [f for f in all_features if f not in current_features and f not in FORCE_REMOVE]
    log = []
    search_hp = {**DEFAULT_HP, "epochs": 50}

    def eval_features(feats, label):
        X_tr, y_tr, X_ho, y_ho, sc = prepare_data(train_df, holdout_df, feats)
        result = train_and_evaluate(X_tr, y_tr, X_ho, y_ho, search_hp, holdout_df, lines)
        mae = result["mae_book"] if result["mae_book"] is not None else result["mae_overall"]
        return mae

    # Baseline
    base_mae = eval_features(current_features, "seed")
    log.append({"step": "seed", "n_features": len(current_features),
                "mae": base_mae, "added": []})
    print(f"  Seed MAE: {base_mae:.4f}")

    for round_num in range(max_add):
        if not candidates:
            break
        best_mae = base_mae
        best_add = None

        for feat in candidates:
            trial_feats = current_features + [feat]
            mae = eval_features(trial_feats, f"try_add_{feat}")
            if mae < best_mae:
                best_mae = mae
                best_add = feat

        if best_add is None:
            print(f"  No improvement found. Stopping.")
            break

        current_features.append(best_add)
        candidates.remove(best_add)
        base_mae = best_mae
        print(f"  Added: {best_add} -> MAE={best_mae:.4f}")
        log.append({"step": f"add_{best_add}", "n_features": len(current_features),
                     "mae": best_mae, "added": [best_add]})

    return current_features, base_mae, log


# ── State Management ─────────────────────────────────────────────


def save_state(state: dict):
    """Save incremental state to disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "session12.json", "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_state() -> dict:
    """Load state from disk."""
    path = STATE_DIR / "session12.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"results": {}, "completed": []}


# ── Report Generation ────────────────────────────────────────────


def generate_report(state: dict, ablation_state: dict | None = None):
    """Generate the comprehensive markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = state["results"]
    has_book = any(r.get("mae_book") is not None for r in results.values())
    mae_key = "mae_book" if has_book else "mae_overall"
    mae_label = "Book-Spread MAE" if has_book else "Overall MAE"

    lines = []
    lines.append("# Adjusted Four-Factors Evaluation — Session 12\n")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Training**: Seasons 2015-2024 | **Holdout**: Season 2025")
    lines.append(f"**Optuna**: {N_OPTUNA_TRIALS} trials per config | **Metric**: {mae_label}\n")

    # ── Master Comparison Table ──────────────────────────────────
    lines.append("## Master Comparison Table\n")
    lines.append(f"All 27 configs ranked by {mae_label}.\n")

    # Collect all results into rows
    rows = []
    for key, r in sorted(results.items()):
        mae = r.get(mae_key)
        if mae is None:
            mae = r.get("mae_overall") or r.get("mae_book")
        if mae is None:
            continue
        rows.append({
            "config": key,
            "n_features": r.get("n_features", "?"),
            "mae": mae,
            "mae_overall": r.get("mae_overall"),
            "mae_book": r.get("mae_book"),
            "n_book": r.get("n_book", 0),
            "hparams": r.get("best_hparams", {}),
        })

    # Add raw baselines
    for feat_cfg, baseline in RAW_BASELINES.items():
        key = f"raw_{feat_cfg}"
        if key not in results:
            n_feat = len(FEATURE_CONFIGS.get(feat_cfg, []))
            bk_mae = baseline.get("mae_book")
            ov_mae = baseline.get("mae_overall")
            mae = bk_mae if has_book and bk_mae else ov_mae if ov_mae else bk_mae
            if mae:
                rows.append({
                    "config": f"{key} (baseline)",
                    "n_features": n_feat,
                    "mae": mae,
                    "mae_overall": ov_mae,
                    "mae_book": bk_mae,
                    "n_book": 0,
                    "hparams": {},
                })

    rows.sort(key=lambda r: r["mae"])

    lines.append(f"| Rank | Config | Features | {mae_label} | Overall MAE |")
    lines.append("|------|--------|----------|" + "-" * (len(mae_label) + 2) + "|-------------|")
    for rank, row in enumerate(rows, 1):
        mae_str = f"{row['mae']:.4f}"
        overall_str = f"{row['mae_overall']:.4f}" if row['mae_overall'] else "—"
        book_str = f"{row['mae_book']:.4f}" if row['mae_book'] else "—"
        lines.append(f"| {rank} | {row['config']} | {row['n_features']} | {mae_str} | {overall_str} |")

    # ── Best Combo Identification ────────────────────────────────
    if rows:
        best = rows[0]
        lines.append(f"\n## Best Configuration\n")
        lines.append(f"**Winner**: `{best['config']}` with {mae_label} = **{best['mae']:.4f}**\n")

        # Statistical significance vs raw baselines
        lines.append("### vs Raw Baselines\n")
        lines.append("| Comparison | Adjusted MAE | Raw MAE | Delta | % Improvement |")
        lines.append("|------------|-------------|---------|-------|---------------|")
        for feat_cfg, baseline in RAW_BASELINES.items():
            raw_mae = baseline.get("mae_book") if has_book else baseline.get("mae_overall")
            if raw_mae:
                delta = best["mae"] - raw_mae
                pct = (raw_mae - best["mae"]) / raw_mae * 100
                lines.append(f"| vs raw_{feat_cfg} | {best['mae']:.4f} | {raw_mae:.4f} | "
                           f"{delta:+.4f} | {pct:+.2f}% |")

        # Best hparams
        if best.get("hparams"):
            lines.append(f"\n### Best Hyperparameters\n")
            lines.append("```json")
            lines.append(json.dumps(best["hparams"], indent=2))
            lines.append("```\n")

    # ── Per-Combo Results ────────────────────────────────────────
    lines.append("## Results by Adjustment Combo\n")

    for combo in ADJ_COMBOS:
        alpha = combo["alpha"]
        prior = combo["prior"]
        combo_label = f"alpha={alpha}, prior={prior}"
        lines.append(f"### {combo_label}\n")

        combo_rows = []
        for feat_cfg in ["37feat", "54feat", "10feat"]:
            key = f"a{alpha}_p{prior}_{feat_cfg}"
            if key in results:
                r = results[key]
                mae = r.get(mae_key, r.get("mae_overall", "—"))
                combo_rows.append(f"| {feat_cfg} | {r.get('n_features', '?')} | "
                                f"{mae if isinstance(mae, str) else f'{mae:.4f}'} | "
                                f"{r.get('mae_overall', 0):.4f} |")

        if combo_rows:
            lines.append(f"| Feature Set | N | {mae_label} | Overall MAE |")
            lines.append("|-------------|---|" + "-" * (len(mae_label) + 2) + "|-------------|")
            lines.extend(combo_rows)
        else:
            lines.append("_No results available._")
        lines.append("")

    # ── Ablation Results ─────────────────────────────────────────
    if ablation_state:
        lines.append("## Ablation Analysis (Best Combo)\n")

        if "importance" in ablation_state:
            lines.append("### Permutation Importance\n")
            lines.append("| Rank | Feature | MAE Increase | Std |")
            lines.append("|------|---------|-------------|-----|")
            imp = ablation_state["importance"]
            sorted_feats = ablation_state.get("importance_ranking", sorted(
                imp.keys(), key=lambda f: imp[f]["mean"], reverse=True))
            for rank, feat in enumerate(sorted_feats, 1):
                i = imp[feat]
                lines.append(f"| {rank} | {feat} | +{i['mean']:.4f} | {i['std']:.4f} |")

        if "backward_log" in ablation_state:
            lines.append("\n### Backward Elimination\n")
            lines.append("| Step | Features | MAE | Removed |")
            lines.append("|------|----------|-----|---------|")
            for entry in ablation_state["backward_log"]:
                removed_str = ", ".join(entry["removed"]) if entry["removed"] else "—"
                lines.append(f"| {entry['step']} | {entry['n_features']} | "
                           f"{entry['mae']:.4f} | {removed_str} |")

        if "forward_log" in ablation_state:
            lines.append("\n### Forward Selection\n")
            lines.append("| Step | Features | MAE | Added |")
            lines.append("|------|----------|-----|-------|")
            for entry in ablation_state["forward_log"]:
                added_str = ", ".join(entry["added"]) if entry["added"] else "—"
                lines.append(f"| {entry['step']} | {entry['n_features']} | "
                           f"{entry['mae']:.4f} | {added_str} |")

        if "optimal_features" in ablation_state:
            lines.append(f"\n### Optimal Feature Set ({len(ablation_state['optimal_features'])} features)\n")
            lines.append("```json")
            lines.append(json.dumps(ablation_state["optimal_features"], indent=2))
            lines.append("```\n")

    # ── Full Metrics for Best Config ─────────────────────────────
    if "best_metrics" in state:
        bm = state["best_metrics"]
        lines.append("## Detailed Metrics — Best Configuration\n")

        # MAE
        lines.append("### MAE\n")
        lines.append(f"- **Overall MAE**: {bm.get('mae_overall', '—')}")
        if bm.get("mae_book"):
            lines.append(f"- **Book-Spread MAE**: {bm['mae_book']:.4f} (n={bm.get('n_book', 0)})")

        # Calibration
        if bm.get("calibration"):
            cal = bm["calibration"]
            lines.append(f"\n### Calibration\n")
            if cal.get("within_1sigma") is not None:
                lines.append(f"- Within 1σ: **{cal['within_1sigma']:.1%}** (ideal: 68.3%)")
                lines.append(f"- Within 2σ: **{cal['within_2sigma']:.1%}** (ideal: 95.4%)")

        # ROI
        if bm.get("roi"):
            lines.append(f"\n### Sigma-Filtered ROI\n")
            lines.append("| Threshold | Bets | Wins | Losses | Win Rate | ROI |")
            lines.append("|-----------|------|------|--------|----------|-----|")
            for thresh_str, r in sorted(bm["roi"].items()):
                if r["bets"] > 0:
                    lines.append(f"| {thresh_str} | {r['bets']} | {r['wins']} | {r['losses']} | "
                               f"{r['win_rate']:.1%} | {r['roi']:+.1%} |")
                else:
                    lines.append(f"| {thresh_str} | 0 | — | — | — | — |")

        # Monthly MAE
        if bm.get("monthly_mae"):
            lines.append(f"\n### Monthly MAE Breakdown\n")
            lines.append("| Month | MAE | Games |")
            lines.append("|-------|-----|-------|")
            for m in bm["monthly_mae"]:
                lines.append(f"| {m['month']} | {m['mae']:.2f} | {m['games']} |")

    # ── Recommendation ───────────────────────────────────────────
    lines.append("\n## Production Recommendation\n")
    if rows:
        best = rows[0]
        lines.append(f"Based on this evaluation, the recommended production configuration is:\n")
        lines.append(f"- **Config**: `{best['config']}`")
        lines.append(f"- **{mae_label}**: {best['mae']:.4f}")
        if best.get("hparams"):
            lines.append(f"- **Architecture**: hidden1={best['hparams'].get('hidden1')}, "
                       f"hidden2={best['hparams'].get('hidden2')}")
        if ablation_state and "optimal_features" in ablation_state:
            lines.append(f"- **Features**: {len(ablation_state['optimal_features'])} "
                       f"(after ablation)")
    lines.append("")

    content = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(content)
    print(f"\nReport saved to: {REPORT_PATH}")


# ── Main Pipeline ────────────────────────────────────────────────


def run_sweep(skip_optuna: bool = False, resume: bool = False):
    """Run the full 27-config sweep."""
    state = load_state() if resume else {"results": {}, "completed": []}

    total_configs = len(ADJ_COMBOS) * len(FEATURE_CONFIGS) + len(FEATURE_CONFIGS)  # 24 + 3 = 27
    completed = len(state["completed"])

    print(f"\n{'='*70}")
    print(f"SESSION 12: OPPONENT-ADJUSTED FOUR-FACTORS EVALUATION")
    print(f"{'='*70}")
    print(f"Total configs: {total_configs} | Completed: {completed}")
    print(f"Optuna trials: {N_OPTUNA_TRIALS} | Folds: {N_FOLDS}")
    print(f"Device: {_get_device()}")
    print(f"{'='*70}\n")

    # Try to load book spreads
    lines = load_book_spreads()
    if lines is not None:
        print(f"Book spreads loaded: {len(lines)} lines")
    else:
        print("No book spreads available — using overall MAE for ranking")

    sweep_start = time.time()

    # ── 1. Adjusted combos (24 configs) ──────────────────────────
    for combo_idx, combo in enumerate(ADJ_COMBOS):
        alpha = combo["alpha"]
        prior = combo["prior"]
        suffix = combo["suffix"]
        combo_label = f"alpha={alpha}, prior={prior}"

        print(f"\n{'─'*60}")
        print(f"COMBO {combo_idx+1}/8: {combo_label}")
        print(f"{'─'*60}")

        # Load data once per combo
        try:
            train_df, holdout_df = load_combo_data(suffix)
            print(f"  Train: {len(train_df)} samples | Holdout: {len(holdout_df)} samples")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        for feat_name, feat_cols in FEATURE_CONFIGS.items():
            config_key = f"a{alpha}_p{prior}_{feat_name}"

            if config_key in state["completed"]:
                print(f"\n  [{config_key}] Already completed — skipping")
                continue

            config_num = completed + 1
            elapsed = time.time() - sweep_start
            if completed > 0:
                eta = elapsed / completed * (total_configs - completed)
                eta_str = f" | ETA: {eta/60:.0f} min"
            else:
                eta_str = ""

            print(f"\n  [{config_num}/{total_configs}] {config_key} ({len(feat_cols)} features){eta_str}")

            # Prepare data
            X_train, y_spread, X_holdout, y_test, scaler = prepare_data(
                train_df, holdout_df, feat_cols)

            # Optuna search
            if skip_optuna:
                best_hp = dict(DEFAULT_HP)
                print(f"    Skipping Optuna — using defaults")
            else:
                t0 = time.time()
                best_hp = run_optuna_search(X_train, y_spread, label=config_key)
                print(f"    Optuna: {time.time()-t0:.0f}s")

            # Final training + evaluation
            t0 = time.time()
            result = train_and_evaluate(
                X_train, y_spread, X_holdout, y_test, best_hp, holdout_df, lines)
            print(f"    Final train: {time.time()-t0:.0f}s")

            mae_overall = result["mae_overall"]
            mae_book = result["mae_book"]
            print(f"    MAE (overall): {mae_overall:.4f}")
            if mae_book is not None:
                print(f"    MAE (book, n={result['n_book']}): {mae_book:.4f}")

            # Save result (excluding large objects)
            state["results"][config_key] = {
                "alpha": alpha,
                "prior": prior,
                "feature_config": feat_name,
                "n_features": len(feat_cols),
                "mae_overall": mae_overall,
                "mae_book": mae_book,
                "n_book": result["n_book"],
                "best_hparams": best_hp,
            }
            state["completed"].append(config_key)
            completed += 1
            save_state(state)

    # ── 2. Raw controls (3 configs) ──────────────────────────────
    print(f"\n{'─'*60}")
    print(f"RAW CONTROLS")
    print(f"{'─'*60}")

    raw_data = load_raw_data()
    if raw_data is not None:
        train_raw, holdout_raw = raw_data
        print(f"  Raw data loaded: {len(train_raw)} train, {len(holdout_raw)} holdout")

        for feat_name, feat_cols in FEATURE_CONFIGS.items():
            config_key = f"raw_{feat_name}"

            if config_key in state["completed"]:
                print(f"\n  [{config_key}] Already completed — skipping")
                continue

            config_num = completed + 1
            print(f"\n  [{config_num}/{total_configs}] {config_key} ({len(feat_cols)} features)")

            X_train, y_spread, X_holdout, y_test, scaler = prepare_data(
                train_raw, holdout_raw, feat_cols)

            if skip_optuna:
                best_hp = dict(DEFAULT_HP)
            else:
                t0 = time.time()
                best_hp = run_optuna_search(X_train, y_spread, label=config_key)
                print(f"    Optuna: {time.time()-t0:.0f}s")

            result = train_and_evaluate(
                X_train, y_spread, X_holdout, y_test, best_hp, holdout_raw, lines)

            mae_overall = result["mae_overall"]
            mae_book = result["mae_book"]
            print(f"    MAE (overall): {mae_overall:.4f}")
            if mae_book is not None:
                print(f"    MAE (book, n={result['n_book']}): {mae_book:.4f}")

            state["results"][config_key] = {
                "alpha": None,
                "prior": None,
                "feature_config": feat_name,
                "n_features": len(feat_cols),
                "mae_overall": mae_overall,
                "mae_book": mae_book,
                "n_book": result["n_book"],
                "best_hparams": best_hp,
            }
            state["completed"].append(config_key)
            completed += 1
            save_state(state)
    else:
        print("  Raw parquets not found — using known baselines")
        for feat_name, baseline in RAW_BASELINES.items():
            config_key = f"raw_{feat_name}"
            if config_key not in state["results"]:
                state["results"][config_key] = {
                    "alpha": None,
                    "prior": None,
                    "feature_config": feat_name,
                    "n_features": len(FEATURE_CONFIGS[feat_name]),
                    "mae_overall": baseline.get("mae_overall"),
                    "mae_book": baseline.get("mae_book"),
                    "n_book": 0,
                    "best_hparams": {},
                }
        save_state(state)

    total_time = time.time() - sweep_start
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE: {completed} configs in {total_time/60:.1f} minutes")
    print(f"{'='*70}")

    return state


def run_ablation(state: dict):
    """Run ablation on the best adjusted combo."""
    print(f"\n{'='*70}")
    print(f"ABLATION ANALYSIS")
    print(f"{'='*70}")

    # Find best adjusted config
    has_book = any(r.get("mae_book") is not None
                   for r in state["results"].values() if r.get("alpha") is not None)
    mae_key = "mae_book" if has_book else "mae_overall"

    best_key = None
    best_mae = float("inf")
    for key, r in state["results"].items():
        if r.get("alpha") is None:  # skip raw
            continue
        mae = r.get(mae_key)
        if mae is not None and mae < best_mae:
            best_mae = mae
            best_key = key

    if best_key is None:
        print("  No adjusted results found — skipping ablation")
        return None

    best = state["results"][best_key]
    alpha = best["alpha"]
    prior = best["prior"]
    feat_cfg = best["feature_config"]
    feat_cols = FEATURE_CONFIGS[feat_cfg]

    print(f"  Best adjusted: {best_key} (MAE={best_mae:.4f})")
    print(f"  Alpha={alpha}, Prior={prior}, Features={feat_cfg} ({len(feat_cols)})")

    # Load data for best combo
    suffix = ""
    for combo in ADJ_COMBOS:
        if combo["alpha"] == alpha and combo["prior"] == prior:
            suffix = combo["suffix"]
            break

    train_df, holdout_df = load_combo_data(suffix)
    X_train, y_spread, X_holdout, y_test, scaler = prepare_data(
        train_df, holdout_df, feat_cols)

    lines_data = load_book_spreads()

    # Train final model for permutation importance
    best_hp = best.get("best_hparams", DEFAULT_HP)
    reg = train_regressor(X_train, y_spread, hparams=best_hp)
    reg.eval()

    # Get book mask if available
    preds = holdout_df[["gameId"]].copy()
    preds = attach_book_spreads(preds, lines_data)
    book_mask = preds["book_spread"].notna().values if "book_spread" in preds.columns else None

    # Permutation importance
    sorted_feats, importance = permutation_importance(
        feat_cols, X_holdout, y_test, reg, n_repeats=10, book_mask=book_mask)

    # Backward elimination
    bw_features, bw_mae, bw_log = backward_elimination(
        feat_cols, train_df, holdout_df, sorted_feats, lines_data)

    # Forward selection from backward result
    fw_features, fw_mae, fw_log = forward_selection(
        feat_cols, bw_features, train_df, holdout_df, lines_data)

    # Save optimal features
    optimal_features = fw_features if fw_mae <= bw_mae else bw_features
    optimal_mae = min(fw_mae, bw_mae)

    optimal_path = config.ARTIFACTS_DIR / "feature_order_session12.json"
    with open(optimal_path, "w") as f:
        json.dump(optimal_features, f, indent=2)
    print(f"\nOptimal features saved: {optimal_path} ({len(optimal_features)} features)")

    ablation_state = {
        "best_config": best_key,
        "importance": importance,
        "importance_ranking": sorted_feats,
        "backward_features": bw_features,
        "backward_mae": bw_mae,
        "backward_log": bw_log,
        "forward_features": fw_features,
        "forward_mae": fw_mae,
        "forward_log": fw_log,
        "optimal_features": optimal_features,
        "optimal_mae": optimal_mae,
    }

    # Save ablation state
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "ablation.json", "w") as f:
        json.dump(ablation_state, f, indent=2, default=str)

    return ablation_state


def compute_best_config_metrics(state: dict, ablation_state: dict | None = None):
    """Compute full metrics for the best config overall."""
    print(f"\n{'='*70}")
    print(f"FULL METRICS — BEST CONFIG")
    print(f"{'='*70}")

    has_book = any(r.get("mae_book") is not None for r in state["results"].values())
    mae_key = "mae_book" if has_book else "mae_overall"

    # Find overall best (including raw and ablation)
    best_key = None
    best_mae = float("inf")
    for key, r in state["results"].items():
        mae = r.get(mae_key)
        if mae is not None and mae < best_mae:
            best_mae = mae
            best_key = key

    if best_key is None:
        print("  No results found")
        return

    best = state["results"][best_key]
    feat_cfg = best["feature_config"]

    # Determine which features to use
    if ablation_state and ablation_state.get("optimal_mae", float("inf")) < best_mae:
        feat_cols = ablation_state["optimal_features"]
        print(f"  Using ablated features: {len(feat_cols)} features")
    else:
        feat_cols = FEATURE_CONFIGS[feat_cfg]
        print(f"  Using {feat_cfg}: {len(feat_cols)} features")

    # Load data
    alpha = best.get("alpha")
    prior = best.get("prior")
    if alpha is not None:
        suffix = ""
        for combo in ADJ_COMBOS:
            if combo["alpha"] == alpha and combo["prior"] == prior:
                suffix = combo["suffix"]
                break
        train_df, holdout_df = load_combo_data(suffix)
    else:
        raw_data = load_raw_data()
        if raw_data is None:
            print("  Cannot compute full metrics — raw data not available")
            return
        train_df, holdout_df = raw_data

    X_train, y_spread, X_holdout, y_test, scaler = prepare_data(
        train_df, holdout_df, feat_cols)

    best_hp = best.get("best_hparams", DEFAULT_HP)
    lines_data = load_book_spreads()

    result = train_and_evaluate(
        X_train, y_spread, X_holdout, y_test, best_hp, holdout_df, lines_data)

    preds = result["preds"]

    # MAE
    mae_overall = result["mae_overall"]
    mae_book = result["mae_book"]
    n_book = result["n_book"]
    print(f"  MAE (overall): {mae_overall:.4f}")
    if mae_book is not None:
        print(f"  MAE (book, n={n_book}): {mae_book:.4f}")

    # Calibration
    cal = compute_calibration(preds)
    if cal["within_1sigma"] is not None:
        print(f"  Calibration: {cal['within_1sigma']:.1%} within 1σ, {cal['within_2sigma']:.1%} within 2σ")

    # Sigma-filtered ROI
    roi_results = {}
    for threshold in [5, 7, 10]:
        # Compute sigma percentile for filtering
        valid_sigma = preds.dropna(subset=["book_spread"])["spread_sigma"] if "book_spread" in preds.columns else preds["spread_sigma"]
        sigma_p25 = float(valid_sigma.quantile(0.25)) if len(valid_sigma) > 0 else 11.5

        roi = compute_roi(preds, threshold, sigma_filter=sigma_p25)
        roi_results[f"threshold_{threshold}"] = roi
        if roi["bets"] > 0:
            print(f"  ROI @{threshold} (sigma<{sigma_p25:.1f}): {roi['roi']:+.1%} "
                  f"({roi['wins']}/{roi['bets']} = {roi['win_rate']:.1%})")

    # Monthly MAE
    monthly = compute_monthly_mae(preds, use_book=has_book)
    if monthly:
        print(f"  Monthly MAE:")
        for m in monthly:
            print(f"    {m['month']}: {m['mae']:.2f} ({m['games']} games)")

    # Save to state
    state["best_metrics"] = {
        "config": best_key,
        "features": feat_cols,
        "n_features": len(feat_cols),
        "mae_overall": mae_overall,
        "mae_book": mae_book,
        "n_book": n_book,
        "calibration": cal,
        "roi": roi_results,
        "monthly_mae": monthly,
    }
    save_state(state)


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Session 12: Adjusted FF Evaluation")
    parser.add_argument("--skip-optuna", action="store_true",
                       help="Skip Optuna search, use default hparams")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from saved state")
    parser.add_argument("--ablation-only", action="store_true",
                       help="Run only ablation (requires completed sweep)")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report from saved state")
    args = parser.parse_args()

    if args.report_only:
        state = load_state()
        ablation_path = STATE_DIR / "ablation.json"
        ablation_state = None
        if ablation_path.exists():
            with open(ablation_path) as f:
                ablation_state = json.load(f)
        generate_report(state, ablation_state)
        return

    if args.ablation_only:
        state = load_state()
        if not state["results"]:
            print("ERROR: No sweep results found. Run sweep first.")
            sys.exit(1)
        ablation_state = run_ablation(state)
        compute_best_config_metrics(state, ablation_state)
        generate_report(state, ablation_state)
        return

    # Full pipeline
    state = run_sweep(skip_optuna=args.skip_optuna, resume=args.resume)
    ablation_state = run_ablation(state)
    compute_best_config_metrics(state, ablation_state)
    generate_report(state, ablation_state)

    print(f"\n{'='*70}")
    print("SESSION 12 COMPLETE")
    print(f"{'='*70}")
    print(f"Report: {REPORT_PATH}")
    print(f"State: {STATE_DIR / 'session12.json'}")
    print(f"Optimal features: {config.ARTIFACTS_DIR / 'feature_order_session12.json'}")


if __name__ == "__main__":
    main()

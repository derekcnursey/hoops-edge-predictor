#!/usr/bin/env python3
"""Retune the production gold-backed model stack.

Tunes:
  - mu LightGBM regressor
  - sigma MLP regressor
  - home-win classifier

Searches use season-aware walk-forward validation on historical gold features.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import log_loss, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.architecture import MLPClassifier, MLPRegressor, gaussian_nll_loss
from src.dataset import load_season_features
from src.features import get_feature_matrix, get_targets
from src.model_hparams import load_best_hparams, save_best_hparams

SEED = 42
ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
TRAIN_SEASONS = [season for season in range(2015, 2026) if season not in config.EXCLUDE_SEASONS]
VALIDATION_SEASONS = [2022, 2023, 2024, 2025]
CPU_DEVICE = torch.device("cpu")


@dataclass
class FoldData:
    season: int
    train_raw: np.ndarray
    train_scaled: np.ndarray
    val_raw: np.ndarray
    val_scaled: np.ndarray
    y_spread_train: np.ndarray
    y_spread_val: np.ndarray
    y_win_train: np.ndarray
    y_win_val: np.ndarray


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _load_gold_training_frame() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in TRAIN_SEASONS:
        df = load_season_features(
            season,
            no_garbage=True,
            adj_suffix=ADJ_SUFFIX,
            efficiency_source="gold",
        ).copy()
        df["season"] = season
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["homeScore", "awayScore"]).copy()
    full = full[(full["homeScore"] != 0) | (full["awayScore"] != 0)].copy()
    full["startDate"] = pd.to_datetime(full["startDate"], utc=True, errors="coerce")
    full = full.dropna(subset=["startDate"]).sort_values(["startDate", "gameId"]).reset_index(drop=True)
    return full


def _build_folds(df: pd.DataFrame) -> list[FoldData]:
    folds: list[FoldData] = []
    for holdout in VALIDATION_SEASONS:
        train_df = df[df["season"] < holdout].copy()
        val_df = df[df["season"] == holdout].copy()
        if train_df.empty or val_df.empty:
            continue

        X_train_raw = get_feature_matrix(train_df).values.astype(np.float32)
        X_val_raw = get_feature_matrix(val_df).values.astype(np.float32)
        y_spread_train = get_targets(train_df)["spread_home"].values.astype(np.float32)
        y_spread_val = get_targets(val_df)["spread_home"].values.astype(np.float32)
        y_win_train = get_targets(train_df)["home_win"].values.astype(np.float32)
        y_win_val = get_targets(val_df)["home_win"].values.astype(np.float32)

        train_means = np.nanmean(X_train_raw, axis=0)
        train_means = np.where(np.isnan(train_means), 0.0, train_means).astype(np.float32)
        X_train = X_train_raw.copy()
        X_val = X_val_raw.copy()
        train_nan = np.isnan(X_train)
        val_nan = np.isnan(X_val)
        if train_nan.any():
            X_train[train_nan] = train_means[np.where(train_nan)[1]]
        if val_nan.any():
            X_val[val_nan] = train_means[np.where(val_nan)[1]]

        scaler_mean = X_train.mean(axis=0)
        scaler_std = X_train.std(axis=0)
        scaler_std = np.where(scaler_std > 1e-6, scaler_std, 1.0)
        X_train_scaled = ((X_train - scaler_mean) / scaler_std).astype(np.float32)
        X_val_scaled = ((X_val - scaler_mean) / scaler_std).astype(np.float32)

        folds.append(
            FoldData(
                season=holdout,
                train_raw=X_train,
                train_scaled=X_train_scaled,
                val_raw=X_val,
                val_scaled=X_val_scaled,
                y_spread_train=y_spread_train,
                y_spread_val=y_spread_val,
                y_win_train=y_win_train,
                y_win_val=y_win_val,
            )
        )
    return folds


def _train_sigma_fold(fold: FoldData, hp: dict[str, Any]) -> float:
    model = MLPRegressor(
        input_dim=fold.train_scaled.shape[1],
        hidden1=int(hp["hidden1"]),
        hidden2=int(hp["hidden2"]),
        dropout=float(hp["dropout"]),
    ).to(CPU_DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
    )
    batch_size = int(hp["batch_size"])
    x_train = torch.tensor(fold.train_scaled, dtype=torch.float32)
    y_train = torch.tensor(fold.y_spread_train, dtype=torch.float32)
    model.train()
    for _ in range(int(hp["epochs"])):
        indices = torch.randperm(len(x_train))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            if len(batch_idx) < batch_size and len(x_train) >= batch_size:
                continue
            x_batch = x_train[batch_idx].to(CPU_DEVICE)
            y_batch = y_train[batch_idx].to(CPU_DEVICE)
            optimizer.zero_grad()
            mu, raw_sigma = model(x_batch)
            loss, _ = gaussian_nll_loss(mu, raw_sigma, y_batch)
            loss.mean().backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_val = torch.tensor(fold.val_scaled, dtype=torch.float32).to(CPU_DEVICE)
        y_val = torch.tensor(fold.y_spread_val, dtype=torch.float32).to(CPU_DEVICE)
        mu, raw_sigma = model(x_val)
        loss, _ = gaussian_nll_loss(mu, raw_sigma, y_val)
        return float(loss.mean().item())


def _train_classifier_fold(fold: FoldData, hp: dict[str, Any]) -> float:
    model = MLPClassifier(
        input_dim=fold.train_scaled.shape[1],
        hidden1=int(hp["hidden1"]),
        dropout=float(hp["dropout"]),
    ).to(CPU_DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    batch_size = int(hp["batch_size"])
    x_train = torch.tensor(fold.train_scaled, dtype=torch.float32)
    y_train = torch.tensor(fold.y_win_train, dtype=torch.float32)
    model.train()
    for _ in range(int(hp["epochs"])):
        indices = torch.randperm(len(x_train))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            if len(batch_idx) < batch_size and len(x_train) >= batch_size:
                continue
            x_batch = x_train[batch_idx].to(CPU_DEVICE)
            y_batch = y_train[batch_idx].to(CPU_DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_val = torch.tensor(fold.val_scaled, dtype=torch.float32).to(CPU_DEVICE)
        logits = model(x_val).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return float(log_loss(fold.y_win_val, probs))


def _train_mu_fold(fold: FoldData, hp: dict[str, Any]) -> float:
    model = lgb.LGBMRegressor(**hp)
    model.fit(
        fold.train_raw,
        fold.y_spread_train,
        eval_set=[(fold.val_raw, fold.y_spread_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_iteration = int(model.best_iteration_ or int(hp["n_estimators"]))
    preds = model.predict(fold.val_raw, num_iteration=best_iteration)
    return float(mean_absolute_error(fold.y_spread_val, preds))


def _tune_sigma(folds: list[FoldData], trials: int) -> tuple[dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        hp = {
            "hidden1": trial.suggest_categorical("hidden1", [256, 384, 512]),
            "hidden2": trial.suggest_categorical("hidden2", [128, 256, 384]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.35),
            "lr": trial.suggest_float("lr", 3e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": 60,
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
            "loss": "gaussian",
            "arch_type": "shared",
            "sigma_param": "exp",
        }
        losses = [_train_sigma_fold(fold, hp) for fold in folds]
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    best = dict(study.best_trial.params)
    best.update({"loss": "gaussian", "arch_type": "shared", "sigma_param": "exp"})
    return best, float(study.best_trial.value)


def _tune_classifier(folds: list[FoldData], trials: int) -> tuple[dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        hp = {
            "hidden1": trial.suggest_categorical("hidden1", [256, 384, 512]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.35),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": 60,
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        }
        losses = [_train_classifier_fold(fold, hp) for fold in folds]
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    return dict(study.best_trial.params), float(study.best_trial.value)


def _tune_mu(folds: list[FoldData], trials: int) -> tuple[dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        hp = {
            "objective": "regression",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 48, 160),
            "max_depth": trial.suggest_int("max_depth", 5, 11),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": trial.suggest_float("subsample", 0.55, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "n_estimators": 5000,
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        losses = [_train_mu_fold(fold, hp) for fold in folds]
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    best = dict(study.best_trial.params)
    best.update(
        {
            "objective": "regression",
            "n_estimators": 5000,
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
    )
    return best, float(study.best_trial.value)


def _write_report(
    output_path: Path,
    mu_params: dict[str, Any] | None,
    mu_score: float | None,
    sigma_params: dict[str, Any] | None,
    sigma_score: float | None,
    classifier_params: dict[str, Any] | None,
    classifier_score: float | None,
    trial_config: dict[str, int],
) -> None:
    payload = {
        "search": {
            "seed": SEED,
            "efficiency_source": "gold",
            "adj_suffix": ADJ_SUFFIX,
            "train_seasons": TRAIN_SEASONS,
            "validation_seasons": VALIDATION_SEASONS,
            **trial_config,
        },
        "mu_regressor": {"params": mu_params, "cv_mae": mu_score} if mu_params is not None else None,
        "regressor": {"params": sigma_params, "cv_nll": sigma_score} if sigma_params is not None else None,
        "classifier": {"params": classifier_params, "cv_logloss": classifier_score} if classifier_params is not None else None,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Retune the production gold-backed stack.")
    parser.add_argument("--mu-trials", type=int, default=25, help="Optuna trials for mu")
    parser.add_argument("--sigma-trials", type=int, default=25, help="Optuna trials for sigma")
    parser.add_argument("--classifier-trials", type=int, default=25, help="Optuna trials for classifier")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "production_gold_tuning_v1.json",
        help="Write detailed tuning results here",
    )
    parser.add_argument(
        "--update-best",
        action="store_true",
        help="Write tuned params back into artifacts/best_hparams.json",
    )
    parser.add_argument("--skip-mu", action="store_true", help="Skip mu search")
    parser.add_argument("--skip-sigma", action="store_true", help="Skip sigma search")
    parser.add_argument("--skip-classifier", action="store_true", help="Skip classifier search")
    args = parser.parse_args()

    _seed_everything()
    print("Loading gold-backed production training frame...")
    df = _load_gold_training_frame()
    print(f"  Rows: {len(df)}")
    print(f"  Seasons: {sorted(df['season'].unique().tolist())}")
    folds = _build_folds(df)
    print(f"  Walk-forward folds: {[fold.season for fold in folds]}")

    mu_params = None
    mu_score = None
    sigma_params = None
    sigma_score = None
    classifier_params = None
    classifier_score = None

    if not args.skip_mu:
        print("\nTuning mu LightGBM...")
        mu_params, mu_score = _tune_mu(folds, args.mu_trials)
        print(f"  Best mu MAE: {mu_score:.4f}")
        print(f"  Params: {mu_params}")
        _write_report(
            args.output,
            mu_params,
            mu_score,
            sigma_params,
            sigma_score,
            classifier_params,
            classifier_score,
            {
                "mu_trials": args.mu_trials,
                "sigma_trials": args.sigma_trials,
                "classifier_trials": args.classifier_trials,
            },
        )

    if not args.skip_sigma:
        print("\nTuning sigma MLP...")
        sigma_params, sigma_score = _tune_sigma(folds, args.sigma_trials)
        print(f"  Best sigma NLL: {sigma_score:.4f}")
        print(f"  Params: {sigma_params}")
        _write_report(
            args.output,
            mu_params,
            mu_score,
            sigma_params,
            sigma_score,
            classifier_params,
            classifier_score,
            {
                "mu_trials": args.mu_trials,
                "sigma_trials": args.sigma_trials,
                "classifier_trials": args.classifier_trials,
            },
        )

    if not args.skip_classifier:
        print("\nTuning classifier...")
        classifier_params, classifier_score = _tune_classifier(folds, args.classifier_trials)
        print(f"  Best classifier logloss: {classifier_score:.4f}")
        print(f"  Params: {classifier_params}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_report(
        args.output,
        mu_params,
        mu_score,
        sigma_params,
        sigma_score,
        classifier_params,
        classifier_score,
        {
            "mu_trials": args.mu_trials,
            "sigma_trials": args.sigma_trials,
            "classifier_trials": args.classifier_trials,
        },
    )
    print(f"\nSaved tuning report to {args.output}")

    if args.update_best:
        best = load_best_hparams()
        if mu_params is not None:
            best["mu_regressor"] = mu_params
        if sigma_params is not None:
            best["regressor"] = {**best.get("regressor", {}), **sigma_params}
        if classifier_params is not None:
            best["classifier"] = {**best.get("classifier", {}), **classifier_params}
        save_best_hparams(best)
        print("Updated artifacts/best_hparams.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

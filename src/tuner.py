"""Optuna hyperparameter search for MLPRegressor and MLPClassifier."""

from __future__ import annotations

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .architecture import MLPClassifier, MLPRegressor, gaussian_nll_loss
from .dataset import HoopsDataset


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hp: dict,
) -> float:
    """Train regressor on train split, return validation loss."""
    device = _get_device()
    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden1=hp["hidden1"],
        hidden2=hp["hidden2"],
        dropout=hp["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    ds = HoopsDataset(X_train, spread=y_train, home_win=np.zeros(len(y_train)))
    loader = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True, drop_last=True)

    model.train()
    for _ in range(hp["epochs"]):
        for batch in loader:
            x, spread, _ = [b.to(device) for b in batch]
            optimizer.zero_grad()
            mu, raw_sigma = model(x)
            loss = gaussian_nll_loss(mu, raw_sigma, spread)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        x_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
        mu, raw_sigma = model(x_val)
        val_loss = gaussian_nll_loss(mu, raw_sigma, y_val_t).item()
    return val_loss


def _evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hp: dict,
) -> float:
    """Train classifier on train split, return validation loss."""
    device = _get_device()
    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden1=hp["hidden1"],
        dropout=hp["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()

    ds = HoopsDataset(X_train, spread=y_train, home_win=y_train)
    loader = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True, drop_last=True)

    model.train()
    for _ in range(hp["epochs"]):
        for batch in loader:
            x, _, win = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, win)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
        logits = model(x_val)
        val_loss = criterion(logits, y_val_t).item()
    return val_loss


def tune_regressor(
    X: np.ndarray,
    y_spread: np.ndarray,
    n_trials: int = 50,
    n_folds: int = 3,
) -> dict:
    """Run Optuna search for MLPRegressor hyperparameters."""

    def objective(trial: optuna.Trial) -> float:
        hp = {
            "hidden1": trial.suggest_categorical("hidden1", [128, 256, 512]),
            "hidden2": trial.suggest_categorical("hidden2", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": 50,  # fewer epochs for tuning
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        val_losses = []
        for train_idx, val_idx in kf.split(X):
            loss = _evaluate_regressor(
                X[train_idx], y_spread[train_idx],
                X[val_idx], y_spread[val_idx],
                hp,
            )
            val_losses.append(loss)
        return float(np.mean(val_losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best regressor trial: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    return study.best_trial.params


def tune_classifier(
    X: np.ndarray,
    y_win: np.ndarray,
    n_trials: int = 50,
    n_folds: int = 3,
) -> dict:
    """Run Optuna search for MLPClassifier hyperparameters."""

    def objective(trial: optuna.Trial) -> float:
        hp = {
            "hidden1": trial.suggest_categorical("hidden1", [128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": 50,
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        val_losses = []
        for train_idx, val_idx in kf.split(X):
            loss = _evaluate_classifier(
                X[train_idx], y_win[train_idx],
                X[val_idx], y_win[val_idx],
                hp,
            )
            val_losses.append(loss)
        return float(np.mean(val_losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best classifier trial: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    return study.best_trial.params

"""Training loop for MLPRegressor and MLPClassifier with mixed precision."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from . import config
from .architecture import MLPClassifier, MLPRegressor, gaussian_nll_loss
from .dataset import HoopsDataset


def fit_scaler(X: np.ndarray, subdir: str | None = None) -> StandardScaler:
    """Fit and save a StandardScaler on feature matrix X."""
    scaler = StandardScaler()
    scaler.fit(X)
    base = config.ARTIFACTS_DIR / subdir if subdir else config.ARTIFACTS_DIR
    path = base / "scaler.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


def load_scaler() -> StandardScaler:
    """Load the fitted StandardScaler from artifacts."""
    path = config.ARTIFACTS_DIR / "scaler.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_regressor(
    X_train: np.ndarray,
    y_spread: np.ndarray,
    hparams: dict | None = None,
) -> MLPRegressor:
    """Train the MLPRegressor with Gaussian NLL loss.

    Args:
        X_train: (N, 37) scaled feature matrix.
        y_spread: (N,) home spread targets.
        hparams: Optional hyperparameters override.

    Returns:
        Trained MLPRegressor on CPU.
    """
    hp = {
        "hidden1": 256,
        "hidden2": 128,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 100,
        "batch_size": 256,
        **(hparams or {}),
    }

    device = _get_device()
    use_amp = device.type == "cuda"

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden1=hp["hidden1"],
        hidden2=hp["hidden2"],
        dropout=hp["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"],
    )
    amp_scaler = GradScaler(device.type, enabled=use_amp)

    ds = HoopsDataset(X_train, spread=y_spread, home_win=np.zeros(len(y_spread)))
    loader = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True, drop_last=True)

    model.train()
    for epoch in range(hp["epochs"]):
        epoch_loss = 0.0
        for batch in loader:
            x, spread, _ = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                mu, raw_sigma = model(x)
                loss = gaussian_nll_loss(mu, raw_sigma, spread)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            avg = epoch_loss / max(len(loader), 1)
            print(f"  Regressor epoch {epoch+1}/{hp['epochs']} — loss: {avg:.4f}")

    return model.cpu()


def train_classifier(
    X_train: np.ndarray,
    y_win: np.ndarray,
    hparams: dict | None = None,
) -> MLPClassifier:
    """Train the MLPClassifier with BCEWithLogitsLoss.

    Args:
        X_train: (N, 37) scaled feature matrix.
        y_win: (N,) binary home win labels.
        hparams: Optional hyperparameters override.

    Returns:
        Trained MLPClassifier on CPU.
    """
    hp = {
        "hidden1": 256,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 100,
        "batch_size": 256,
        **(hparams or {}),
    }

    device = _get_device()
    use_amp = device.type == "cuda"

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden1=hp["hidden1"],
        dropout=hp["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    amp_scaler = GradScaler(device.type, enabled=use_amp)

    ds = HoopsDataset(X_train, spread=y_win, home_win=y_win)
    loader = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True, drop_last=True)

    model.train()
    for epoch in range(hp["epochs"]):
        epoch_loss = 0.0
        for batch in loader:
            x, _, win = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, win)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            avg = epoch_loss / max(len(loader), 1)
            print(f"  Classifier epoch {epoch+1}/{hp['epochs']} — loss: {avg:.4f}")

    return model.cpu()


def save_checkpoint(
    model: nn.Module,
    name: str,
    hparams: dict | None = None,
    subdir: str | None = None,
) -> Path:
    """Save model checkpoint with feature_order and hparams embedded."""
    base = config.CHECKPOINTS_DIR / subdir if subdir else config.CHECKPOINTS_DIR
    path = base / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_order": config.FEATURE_ORDER,
            "hparams": hparams or {},
        },
        path,
    )
    print(f"  Saved checkpoint: {path}")
    return path

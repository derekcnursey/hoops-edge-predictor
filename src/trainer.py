"""Training loop for MLPRegressor and MLPClassifier with mixed precision."""

from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from . import config
from .architecture import MLPClassifier, MLPRegressor, MLPRegressorSplit, gaussian_nll_loss
from .dataset import HoopsDataset


def impute_column_means(X: np.ndarray) -> np.ndarray:
    """Replace NaN values with per-column means computed from non-NaN values.

    This preserves the true feature distribution for downstream scaling,
    unlike np.nan_to_num(X, nan=0.0) which distorts the scaler by up to 3.7x
    on features with natural ranges far from zero (e.g. efficiency ratings ~103).
    """
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = col_means[j]
    return X


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


def train_hist_gradient_boosting_regressor(
    X_train: np.ndarray,
    y_spread: np.ndarray,
    hparams: dict | None = None,
) -> HistGradientBoostingRegressor:
    """Train the benchmark-winning HistGradientBoosting point regressor."""
    hp = {**config.HGBR_PARAMS, **(hparams or {})}
    model = HistGradientBoostingRegressor(**hp)
    model.fit(X_train, y_spread)
    return model


def train_lightgbm_regressor(
    X_train: np.ndarray,
    y_spread: np.ndarray,
    hparams: dict | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int = 100,
) -> lgb.LGBMRegressor:
    """Train the promoted LightGBM L2 point regressor."""
    hp = {**config.LGBM_REG_L2_PARAMS, **(hparams or {})}
    model = lgb.LGBMRegressor(**hp)
    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None and len(X_val) > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_metric"] = "l1"
        fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    model.fit(X_train, y_spread, **fit_kwargs)
    return model


def save_tree_regressor(
    model,
    path: Path | None = None,
    feature_order: list[str] | None = None,
    hparams: dict | None = None,
    model_type: str | None = None,
) -> Path:
    """Persist the production tree mu regressor with feature metadata."""
    out_path = path or config.TREE_REGRESSOR_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inferred_model_type = model_type
    if inferred_model_type is None:
        inferred_model_type = "lightgbm" if isinstance(model, lgb.LGBMRegressor) else "hist_gradient_boosting"
    default_hparams = config.LGBM_REG_L2_PARAMS if inferred_model_type == "lightgbm" else config.HGBR_PARAMS
    payload = {
        "model": model,
        "feature_order": feature_order or config.FEATURE_ORDER,
        "model_type": inferred_model_type,
        "hparams": hparams or default_hparams,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return out_path


def load_tree_regressor(path: Path | None = None) -> tuple[object, list[str], dict]:
    """Load the production tree mu regressor if present."""
    in_path = path or config.TREE_REGRESSOR_PATH
    with open(in_path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_order"], {
        **payload.get("hparams", {}),
        "model_type": payload.get("model_type", "hist_gradient_boosting"),
    }


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
    val_frac: float = 0.0,
    temporal_val_split: bool = False,
) -> MLPRegressor:
    """Train the MLPRegressor with Gaussian NLL loss.

    Args:
        X_train: (N, D) scaled feature matrix.
        y_spread: (N,) home spread targets.
        hparams: Optional hyperparameters override.
        val_frac: Fraction of data for validation. When > 0, saves model at
            best validation loss instead of returning last-epoch weights.

    Returns:
        Trained MLPRegressor on CPU.
    """
    hp = {
        "hidden1": 384,
        "hidden2": 256,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 100,
        "batch_size": 256,
        **(hparams or {}),
    }

    device = _get_device()
    use_amp = device.type == "cuda"

    # Validation split
    X_val, y_val = None, None
    if val_frac > 0:
        n_val = int(len(X_train) * val_frac)
        if temporal_val_split:
            train_idx = np.arange(0, len(X_train) - n_val)
            val_idx = np.arange(len(X_train) - n_val, len(X_train))
        else:
            indices = np.random.permutation(len(X_train))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_val, y_val = X_train[val_idx], y_spread[val_idx]
        X_train, y_spread = X_train[train_idx], y_spread[train_idx]

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

    best_val_loss = float("inf")
    best_state = None
    best_epoch = hp["epochs"]

    model.train()
    for epoch in range(hp["epochs"]):
        epoch_loss = 0.0
        for batch in loader:
            x, spread, _ = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                mu, log_sigma = model(x)
                nll, _sigma = gaussian_nll_loss(mu, log_sigma, spread)
                loss = nll.mean()
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()

        # Best-loss checkpointing
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vx = torch.tensor(X_val, dtype=torch.float32).to(device)
                vy = torch.tensor(y_val, dtype=torch.float32).to(device)
                v_mu, v_ls = model(vx)
                v_nll, _ = gaussian_nll_loss(v_mu, v_ls, vy)
                val_loss = v_nll.mean().item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
            model.train()

        if (epoch + 1) % 20 == 0:
            avg = epoch_loss / max(len(loader), 1)
            val_str = f", val: {val_loss:.4f}" if X_val is not None else ""
            print(f"  Regressor epoch {epoch+1}/{hp['epochs']} — loss: {avg:.4f}{val_str}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_val_loss = float(best_val_loss) if X_val is not None else None
    model.best_epoch = int(best_epoch)

    return model.cpu()


def train_classifier(
    X_train: np.ndarray,
    y_win: np.ndarray,
    hparams: dict | None = None,
    val_frac: float = 0.0,
    temporal_val_split: bool = False,
) -> MLPClassifier:
    """Train the MLPClassifier with BCEWithLogitsLoss.

    Args:
        X_train: (N, D) scaled feature matrix.
        y_win: (N,) binary home win labels.
        hparams: Optional hyperparameters override.
        val_frac: Fraction of data for validation. When > 0, saves model at
            best validation loss instead of returning last-epoch weights.

    Returns:
        Trained MLPClassifier on CPU.
    """
    hp = {
        "hidden1": 384,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 100,
        "batch_size": 256,
        **(hparams or {}),
    }

    device = _get_device()
    use_amp = device.type == "cuda"

    # Validation split
    X_val, y_val = None, None
    if val_frac > 0:
        n_val = int(len(X_train) * val_frac)
        if temporal_val_split:
            train_idx = np.arange(0, len(X_train) - n_val)
            val_idx = np.arange(len(X_train) - n_val, len(X_train))
        else:
            indices = np.random.permutation(len(X_train))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_val, y_val = X_train[val_idx], y_win[val_idx]
        X_train, y_win = X_train[train_idx], y_win[train_idx]

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

    best_val_loss = float("inf")
    best_state = None
    best_epoch = hp["epochs"]

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

        # Best-loss checkpointing
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vx = torch.tensor(X_val, dtype=torch.float32).to(device)
                vy = torch.tensor(y_val, dtype=torch.float32).to(device)
                val_loss = criterion(model(vx), vy).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
            model.train()

        if (epoch + 1) % 20 == 0:
            avg = epoch_loss / max(len(loader), 1)
            val_str = f", val: {val_loss:.4f}" if X_val is not None else ""
            print(f"  Classifier epoch {epoch+1}/{hp['epochs']} — loss: {avg:.4f}{val_str}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_val_loss = float(best_val_loss) if X_val is not None else None
    model.best_epoch = int(best_epoch)

    return model.cpu()


def save_checkpoint(
    model: nn.Module,
    name: str,
    hparams: dict | None = None,
    subdir: str | None = None,
    feature_order: list[str] | None = None,
) -> Path:
    """Save model checkpoint with feature_order, hparams, and architecture metadata."""
    base = config.CHECKPOINTS_DIR / subdir if subdir else config.CHECKPOINTS_DIR
    path = base / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect architecture type
    arch_type = "split" if isinstance(model, MLPRegressorSplit) else "shared"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_order": feature_order or config.FEATURE_ORDER,
            "hparams": {
                **(hparams or {}),
                **(
                    {"best_epoch": int(model.best_epoch)}
                    if hasattr(model, "best_epoch") and model.best_epoch is not None
                    else {}
                ),
                **(
                    {"best_val_loss": float(model.best_val_loss)}
                    if hasattr(model, "best_val_loss") and model.best_val_loss is not None
                    else {}
                ),
            },
            "arch_type": arch_type,
            "sigma_param": "exp",
        },
        path,
    )
    print(f"  Saved checkpoint: {path}")
    return path

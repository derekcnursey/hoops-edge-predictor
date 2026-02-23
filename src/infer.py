"""Load trained models and produce predictions."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from .architecture import MLPClassifier, MLPRegressor
from .trainer import load_scaler


def load_regressor(path: Path | None = None) -> tuple[MLPRegressor, dict]:
    """Load MLPRegressor from checkpoint."""
    if path is None:
        path = config.CHECKPOINTS_DIR / "regressor.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    model = MLPRegressor(
        input_dim=37,
        hidden1=hp.get("hidden1", 256),
        hidden2=hp.get("hidden2", 128),
        dropout=hp.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, hp


def load_classifier(path: Path | None = None) -> tuple[MLPClassifier, dict]:
    """Load MLPClassifier from checkpoint."""
    if path is None:
        path = config.CHECKPOINTS_DIR / "classifier.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    model = MLPClassifier(
        input_dim=37,
        hidden1=hp.get("hidden1", 256),
        dropout=hp.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, hp


@torch.no_grad()
def predict(
    features_df: pd.DataFrame,
    lines_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate predictions for a feature DataFrame.

    Args:
        features_df: DataFrame with the 37 feature columns + metadata (gameId, etc.)
        lines_df: Optional lines data to attach spread/moneyline info.

    Returns:
        DataFrame with predictions: mu, sigma, home_win_prob, plus edge metrics.
    """
    scaler = load_scaler()
    regressor, _ = load_regressor()
    classifier, _ = load_classifier()

    X = features_df[config.FEATURE_ORDER].values.astype(np.float32)

    # Handle NaN: fill with column means (from scaler)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = scaler.mean_
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_means[j]

    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Regressor: (mu, raw_sigma)
    mu_raw, log_sigma_raw = regressor(X_tensor)
    sigma = torch.nn.functional.softplus(log_sigma_raw) + 1e-3
    sigma = sigma.clamp(min=0.5, max=30.0)

    mu = mu_raw.numpy()
    sigma = sigma.numpy()

    # Classifier: home_win_prob
    logits = classifier(X_tensor)
    home_win_prob = torch.sigmoid(logits).numpy()

    # Build output — include team names if available
    meta_cols = ["gameId", "homeTeamId", "awayTeamId"]
    if "homeTeam" in features_df.columns:
        meta_cols.append("homeTeam")
    if "awayTeam" in features_df.columns:
        meta_cols.append("awayTeam")
    meta_cols.append("startDate")
    out = features_df[meta_cols].copy()
    out["predicted_spread"] = mu
    out["spread_sigma"] = sigma
    out["home_win_prob"] = home_win_prob
    out["away_win_prob"] = 1.0 - home_win_prob

    # Attach lines if available
    if lines_df is not None and not lines_df.empty:
        # Use first provider per game
        lines_dedup = lines_df.sort_values("provider").drop_duplicates(subset=["gameId"], keep="first")
        lines_dedup = lines_dedup.rename(columns={
            "spread": "book_spread",
            "overUnder": "book_total",
            "homeMoneyline": "home_moneyline",
            "awayMoneyline": "away_moneyline",
        })
        merge_cols = ["gameId", "book_spread", "book_total", "home_moneyline", "away_moneyline"]
        available = [c for c in merge_cols if c in lines_dedup.columns]
        out = out.merge(lines_dedup[available], on="gameId", how="left")

        # Edge metrics
        if "book_spread" in out.columns:
            # book_spread is from home perspective (negative = home favored)
            # predicted_spread is home_pts - away_pts (positive = home favored)
            # Convert predicted_spread to book convention: negate it
            out["model_spread"] = -out["predicted_spread"]
            out["spread_diff"] = out["model_spread"] - out["book_spread"]

    return out


def save_predictions(preds: pd.DataFrame, game_date: str | None = None) -> tuple[Path, Path]:
    """Save predictions as JSON and CSV.

    Returns:
        (json_path, csv_path)
    """
    if game_date is None:
        game_date = date.today().isoformat()

    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    json_dir = config.PREDICTIONS_DIR / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    json_path = json_dir / f"{game_date}.json"
    csv_path = config.PREDICTIONS_DIR / "preds_today.csv"

    # JSON output
    records = preds.to_dict(orient="records")
    # Convert numpy types to native Python for JSON serialization
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.bool_,)):
                rec[k] = bool(v)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    # CSV output
    preds.to_csv(csv_path, index=False)

    return json_path, csv_path

#!/usr/bin/env python3
"""Session 15: Walk-forward comparison of V1 (50 features) vs V4 (136 features).

Trains both models from scratch on each fold, using the same architecture and
hyperparameters (scaled input_dim). Compares MAE, RMSE, coverage, calibration,
and ATS record.

Usage:
    poetry run python scripts/session15_walkforward.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset
from src.features import get_feature_matrix, get_targets, load_lines
from src.trainer import impute_column_means

# ── Walk-forward config ──────────────────────────────────────────
TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
MIN_MONTH_DAY = "12-01"
MAX_EPOCHS = 500
PATIENCE = 50

# Same architecture used in session 13 validation suite
WINNER_HP = {
    "hidden1": 384, "hidden2": 256, "dropout": 0.20,
    "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4,
}


# ── Data loading ─────────────────────────────────────────────────

def _filter_by_min_date(df: pd.DataFrame, min_month_day: str) -> pd.DataFrame:
    """Filter out early-season games (before min_month_day within each season)."""
    dates = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    game_dates = dates.dt.tz_localize(None).dt.normalize()
    month, day = (int(x) for x in min_month_day.split("-"))
    game_years = game_dates.dt.year
    game_months = game_dates.dt.month
    season_year = game_years.where(game_months <= 7, game_years + 1)
    cutoff_year = season_year - 1 if month >= 8 else season_year
    cutoffs = pd.to_datetime(
        cutoff_year.astype(int).astype(str) + f"-{month:02d}-{day:02d}",
        errors="coerce",
    )
    return df[game_dates >= cutoffs].reset_index(drop=True)


def load_v1_season(season: int) -> pd.DataFrame:
    """Load V1 features (50-feature parquet) for a season."""
    adj_suffix = f"no_garbage_adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
    path = config.FEATURES_DIR / f"season_{season}_{adj_suffix}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"V1 features not found: {path}")
    return pd.read_parquet(path)


def load_v4_season(season: int) -> pd.DataFrame:
    """Load V4 features (136-feature parquet) for a season."""
    path = config.FEATURES_DIR / f"season_{season}_v4_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"V4 features not found: {path}")
    return pd.read_parquet(path)


def load_multi(seasons: list[int], loader_fn, feature_order: list[str]):
    """Load multi-season data, filter dates, extract features + targets."""
    dfs = []
    for s in seasons:
        try:
            dfs.append(loader_fn(s))
        except FileNotFoundError:
            print(f"    Warning: No features for season {s}, skipping.")
    if not dfs:
        raise FileNotFoundError(f"No features for seasons {seasons}")
    df = pd.concat(dfs, ignore_index=True)

    # Date filter
    before = len(df)
    df = _filter_by_min_date(df, MIN_MONTH_DAY)

    # Drop rows without scores
    df = df.dropna(subset=["homeScore", "awayScore"])
    df = df[(df["homeScore"] != 0) | (df["awayScore"] != 0)]

    # Extract features and targets
    X = get_feature_matrix(df, feature_order=feature_order).values.astype(np.float32)
    targets = get_targets(df)
    y = targets["spread_home"].values.astype(np.float32)

    return X, y, df


def merge_book_spreads(df: pd.DataFrame, seasons: list[int]) -> np.ndarray:
    """Merge book spreads onto a DataFrame, return array aligned with df."""
    book = np.full(len(df), np.nan, dtype=np.float64)
    for vs in seasons:
        try:
            lines_df = load_lines(vs)
            if lines_df.empty:
                continue
            ld = lines_df.sort_values("provider").drop_duplicates(
                subset=["gameId"], keep="first")
            if "spread" in ld.columns:
                merge_map = dict(zip(ld["gameId"], ld["spread"]))
                for i, gid in enumerate(df["gameId"].values):
                    if gid in merge_map:
                        book[i] = merge_map[gid]
        except Exception:
            pass
    return book


# ── Training ─────────────────────────────────────────────────────

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _val_loss(model, X_val_t, y_val_t, device):
    """Compute validation Gaussian NLL loss."""
    model.eval()
    with torch.no_grad():
        x = X_val_t.to(device)
        y = y_val_t.to(device)
        mu, ls = model(x)
        nll, _ = gaussian_nll_loss(mu, ls, y)
        loss = nll.mean().item()
    model.train()
    return loss


def train_model(X_train, y_train, X_val, y_val, hp=None, verbose=False):
    """Train MLPRegressor with early stopping. Returns (model, best_epoch)."""
    if hp is None:
        hp = WINNER_HP
    device = _get_device()
    use_amp = device.type == "cuda"

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden1=hp["hidden1"], hidden2=hp["hidden2"],
        dropout=hp["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"],
        weight_decay=hp.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    amp_scaler = GradScaler(device.type, enabled=use_amp)

    ds = HoopsDataset(X_train, spread=y_train, home_win=np.zeros(len(y_train)))
    loader = DataLoader(ds, batch_size=hp.get("batch_size", 4096),
                        shuffle=True, drop_last=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    model.train()
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            x, spread, _ = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                mu, log_sigma = model(x)
                nll, _ = gaussian_nll_loss(mu, log_sigma, spread)
                loss = nll.mean()
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()
        scheduler.step()

        val_loss = _val_loss(model, X_val_t, y_val_t, device)
        ep = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (ep % 50 == 0 or no_improve == PATIENCE):
            print(f"      ep {ep}: val={val_loss:.4f} best={best_val_loss:.4f} @{best_epoch}")

        if no_improve >= PATIENCE:
            break

    model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


@torch.no_grad()
def predict(model, X_val):
    """Return (mu, sigma) numpy arrays."""
    X_t = torch.tensor(X_val, dtype=torch.float32)
    mu_t, ls_t = model(X_t)
    sigma_t = torch.exp(ls_t).clamp(min=0.5, max=30.0)
    return mu_t.numpy(), sigma_t.numpy()


# ── Metrics ──────────────────────────────────────────────────────

def normal_cdf(z):
    """Standard normal CDF approximation."""
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


def compute_metrics(mu, sigma, y_actual, book):
    """Compute comprehensive metrics for a fold."""
    # Basic regression metrics
    mae = np.mean(np.abs(mu - y_actual))
    rmse = np.sqrt(np.mean((mu - y_actual) ** 2))
    residuals = y_actual - mu

    # Coverage: % of actuals within 1 sigma
    within_1sigma = np.mean(np.abs(residuals) <= sigma)
    within_2sigma = np.mean(np.abs(residuals) <= 2 * sigma)

    # Calibration: mean sigma vs empirical std
    mean_sigma = np.mean(sigma)
    empirical_std = np.std(residuals)
    calibration_ratio = mean_sigma / empirical_std if empirical_std > 0 else float("nan")

    # ATS record (against the spread)
    valid_book = ~np.isnan(book)
    ats_record = {"bets": 0, "wins": 0, "win_rate": 0.0, "roi": 0.0, "units": 0.0}

    if valid_book.sum() > 0:
        edge_home = mu[valid_book] + book[valid_book]
        pick_home = edge_home >= 0
        home_covered = (y_actual[valid_book] + book[valid_book]) > 0
        pick_won = np.where(pick_home, home_covered, ~home_covered)

        # Apply edge threshold (prob_edge >= 5%)
        sigma_safe = np.clip(sigma[valid_book], 0.5, None)
        edge_z = edge_home / sigma_safe
        hcp = normal_cdf(edge_z)
        pick_prob = np.where(pick_home, hcp, 1.0 - hcp)
        prob_edge = pick_prob - 0.5238  # break-even at -110

        threshold = 0.05
        bet_mask = prob_edge >= threshold
        n_bets = bet_mask.sum()

        if n_bets > 0:
            wins = pick_won[bet_mask].sum()
            profit_per_1 = 100.0 / 110.0
            units = wins * profit_per_1 - (n_bets - wins)
            ats_record = {
                "bets": int(n_bets),
                "wins": int(wins),
                "win_rate": float(wins / n_bets),
                "roi": float(units / n_bets),
                "units": float(units),
            }

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "within_1sigma": float(within_1sigma),
        "within_2sigma": float(within_2sigma),
        "mean_sigma": float(mean_sigma),
        "empirical_std": float(empirical_std),
        "calibration_ratio": float(calibration_ratio),
        "n_games": int(len(mu)),
        "ats": ats_record,
    }


# ── Main walk-forward ────────────────────────────────────────────

def run_walkforward(model_name, loader_fn, feature_order, hp=None):
    """Run full walk-forward for a given feature set."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD: {model_name} ({len(feature_order)} features)")
    print(f"{'='*70}")

    results = []

    for ty in TEST_YEARS:
        train_seasons = list(range(2015, ty))
        val_seasons = [ty]

        print(f"\n  --- {ty}: train on {train_seasons[0]}-{train_seasons[-1]} ---")
        t0 = time.time()

        try:
            X_tr, y_tr, _ = load_multi(train_seasons, loader_fn, feature_order)
            X_val, y_val, df_val = load_multi(val_seasons, loader_fn, feature_order)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            results.append({"year": ty, "status": "skip", "reason": str(e)})
            continue

        book = merge_book_spreads(df_val, val_seasons)

        # Impute + scale
        X_tr = impute_column_means(X_tr)
        X_val = impute_column_means(X_val)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)

        # Train
        model, best_ep = train_model(X_tr_s, y_tr, X_val_s, y_val, hp=hp, verbose=False)
        mu, sigma = predict(model, X_val_s)

        elapsed = time.time() - t0
        metrics = compute_metrics(mu, sigma, y_val, book)

        n_book = (~np.isnan(book)).sum()
        ats = metrics["ats"]
        print(f"    {len(df_val)} games, {n_book} with book, ep@{best_ep} [{elapsed:.0f}s]")
        print(f"    MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
              f"σ={metrics['mean_sigma']:.1f}  1σ-cov={metrics['within_1sigma']:.1%}  "
              f"cal={metrics['calibration_ratio']:.2f}")
        if ats["bets"] > 0:
            print(f"    ATS: {ats['wins']}/{ats['bets']} ({ats['win_rate']:.1%}) "
                  f"ROI={ats['roi']*100:+.1f}% units={ats['units']:+.1f}")

        results.append({
            "year": ty,
            "status": "ok",
            "elapsed": elapsed,
            "best_epoch": best_ep,
            "n_train": len(y_tr),
            "n_val": len(y_val),
            **metrics,
        })

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def print_summary(v1_results, v4_results):
    """Print side-by-side comparison of V1 vs V4."""
    print(f"\n{'='*80}")
    print("  SUMMARY: V1 (50 features) vs V4 (136 features)")
    print(f"{'='*80}")

    # Per-year comparison
    header = f"{'Year':>6} {'V1 MAE':>8} {'V4 MAE':>8} {'ΔMAE':>8} " \
             f"{'V1 RMSE':>9} {'V4 RMSE':>9} {'ΔRMSE':>8} " \
             f"{'V1 ATS':>10} {'V4 ATS':>10}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    v1_maes, v4_maes, v1_rmses, v4_rmses = [], [], [], []
    v1_ats_units, v4_ats_units = 0.0, 0.0
    v1_ats_bets, v4_ats_bets = 0, 0
    v1_ats_wins, v4_ats_wins = 0, 0

    for v1, v4 in zip(v1_results, v4_results):
        if v1.get("status") != "ok" or v4.get("status") != "ok":
            print(f"  {v1.get('year', '?'):>6}  {'SKIP':>8}  {'SKIP':>8}")
            continue

        yr = v1["year"]
        d_mae = v4["mae"] - v1["mae"]
        d_rmse = v4["rmse"] - v1["rmse"]

        v1_ats_str = f"{v1['ats']['wins']}/{v1['ats']['bets']}" if v1["ats"]["bets"] > 0 else "---"
        v4_ats_str = f"{v4['ats']['wins']}/{v4['ats']['bets']}" if v4["ats"]["bets"] > 0 else "---"

        print(f"  {yr:>6} {v1['mae']:>8.2f} {v4['mae']:>8.2f} {d_mae:>+8.2f} "
              f"{v1['rmse']:>9.2f} {v4['rmse']:>9.2f} {d_rmse:>+8.2f} "
              f"{v1_ats_str:>10} {v4_ats_str:>10}")

        v1_maes.append(v1["mae"])
        v4_maes.append(v4["mae"])
        v1_rmses.append(v1["rmse"])
        v4_rmses.append(v4["rmse"])

        v1_ats_units += v1["ats"]["units"]
        v4_ats_units += v4["ats"]["units"]
        v1_ats_bets += v1["ats"]["bets"]
        v4_ats_bets += v4["ats"]["bets"]
        v1_ats_wins += v1["ats"]["wins"]
        v4_ats_wins += v4["ats"]["wins"]

    # Averages
    if v1_maes and v4_maes:
        print(f"\n  {'AVG':>6} {np.mean(v1_maes):>8.2f} {np.mean(v4_maes):>8.2f} "
              f"{np.mean(v4_maes) - np.mean(v1_maes):>+8.2f} "
              f"{np.mean(v1_rmses):>9.2f} {np.mean(v4_rmses):>9.2f} "
              f"{np.mean(v4_rmses) - np.mean(v1_rmses):>+8.2f}")

        print(f"\n  Pooled ATS:")
        if v1_ats_bets > 0:
            v1_roi = v1_ats_units / v1_ats_bets * 100
            print(f"    V1: {v1_ats_wins}/{v1_ats_bets} ({v1_ats_wins/v1_ats_bets:.1%}) "
                  f"ROI={v1_roi:+.1f}% units={v1_ats_units:+.1f}")
        if v4_ats_bets > 0:
            v4_roi = v4_ats_units / v4_ats_bets * 100
            print(f"    V4: {v4_ats_wins}/{v4_ats_bets} ({v4_ats_wins/v4_ats_bets:.1%}) "
                  f"ROI={v4_roi:+.1f}% units={v4_ats_units:+.1f}")

    # Verdict
    print(f"\n  {'='*60}")
    if v4_maes and v1_maes:
        mae_better = np.mean(v4_maes) < np.mean(v1_maes)
        rmse_better = np.mean(v4_rmses) < np.mean(v1_rmses)
        ats_better = v4_ats_units > v1_ats_units

        wins = sum([mae_better, rmse_better, ats_better])
        if wins >= 2:
            print("  VERDICT: V4 WINS (lower MAE/RMSE or better ATS)")
        elif wins == 0:
            print("  VERDICT: V1 WINS (V4 did not improve)")
        else:
            print("  VERDICT: MIXED RESULTS — needs deeper analysis")
    print(f"  {'='*60}")


def main():
    print("=" * 70)
    print("  SESSION 15: V1 vs V4 WALK-FORWARD COMPARISON")
    print("=" * 70)
    print(f"  Test years: {TEST_YEARS}")
    print(f"  Date filter: >= {MIN_MONTH_DAY}")
    print(f"  Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")
    print(f"  HP: {WINNER_HP}")

    # Run V1 (50 features)
    v1_results = run_walkforward(
        "V1", load_v1_season, config.FEATURE_ORDER, hp=WINNER_HP,
    )

    # Run V4 (136 features)
    v4_results = run_walkforward(
        "V4", load_v4_season, config.FEATURE_ORDER_V4, hp=WINNER_HP,
    )

    # Compare
    print_summary(v1_results, v4_results)

    # Save results
    out = {
        "v1": v1_results,
        "v4": v4_results,
        "config": {
            "test_years": TEST_YEARS,
            "min_month_day": MIN_MONTH_DAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "hp": WINNER_HP,
            "v1_features": len(config.FEATURE_ORDER),
            "v4_features": len(config.FEATURE_ORDER_V4),
        },
    }
    out_path = config.ARTIFACTS_DIR / "session15_walkforward_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()

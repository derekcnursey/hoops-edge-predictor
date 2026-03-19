#!/usr/bin/env python3
"""Walk-forward comparison: iterative vs WLS solver for adjusted efficiencies.

This script rebuilds the gold-layer adjusted efficiencies using the WLS solver,
then runs the full feature pipeline and walk-forward evaluation to compare
prediction accuracy against the current iterative solver.

The key difference from other walkforwards: this one rebuilds the RATINGS
(adj_oe/adj_de) which are the #1 and #2 features. Everything downstream
(feature build, model training, evaluation) is identical.

Usage:
    # Step 1: Rebuild ratings with WLS solver for all seasons
    poetry run python scripts/wls_walkforward.py --rebuild --seasons 2015-2026

    # Step 2: Run walk-forward comparison
    poetry run python scripts/wls_walkforward.py --evaluate
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets, load_research_lines
from src.trainer import impute_column_means

# ── Config ────────────────────────────────────────────────────────────
ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
HOLDOUT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
TRAIN_START = 2015
MIN_DATE = "12-01"

MAX_EPOCHS = 500
PATIENCE = 50

HP = {
    "hidden1": 384, "hidden2": 256, "dropout": 0.20,
    "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4,
}

REPORT_PATH = Path(__file__).resolve().parent.parent / "analysis" / "wls_walkforward_report.md"


def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _val_loss(model, X_val_t, y_val_t, device):
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(X_val_t), 4096):
        e = min(s + 4096, len(X_val_t))
        x = X_val_t[s:e].to(device)
        y = y_val_t[s:e].to(device)
        mu, ls = model(x)
        nll, _ = gaussian_nll_loss(mu, ls, y)
        total += nll.mean().item()
        n += 1
    model.train()
    return total / max(n, 1)


def train_model(X_train, y_train, X_val_s, y_val):
    device = get_device()
    use_amp = device.type == "cuda"

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden1=HP["hidden1"], hidden2=HP["hidden2"],
        dropout=HP["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    amp_scaler = GradScaler(device.type, enabled=use_amp)

    ds = HoopsDataset(X_train, spread=y_train, home_win=np.zeros(len(y_train)))
    loader = DataLoader(ds, batch_size=HP["batch_size"], shuffle=True, drop_last=True)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss, best_state, best_epoch, no_improve = float("inf"), None, 0, 0

    model.train()
    for epoch in range(MAX_EPOCHS):
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
        scheduler.step()

        val_loss = _val_loss(model, X_val_t, y_val_t, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


@torch.no_grad()
def predict_model(model, X_scaled):
    model.eval()
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    mu_list, sigma_list = [], []
    for s in range(0, len(X_t), 4096):
        e = min(s + 4096, len(X_t))
        mu, log_sigma = model(X_t[s:e])
        sigma = torch.exp(log_sigma).clamp(0.5, 30.0)
        mu_list.append(mu.numpy().flatten())
        sigma_list.append(sigma.numpy().flatten())
    return np.concatenate(mu_list), np.concatenate(sigma_list)


def run_walkforward(adj_suffix: str, label: str):
    """Run a single walk-forward configuration."""
    feature_order = config.FEATURE_ORDER
    yearly_results = {}

    for holdout in HOLDOUT_YEARS:
        train_seasons = list(range(TRAIN_START, holdout))
        print(f"  [{label}] Holdout {holdout}: train {TRAIN_START}-{holdout-1}...", end=" ", flush=True)

        df_train = load_multi_season_features(
            train_seasons, no_garbage=True, adj_suffix=adj_suffix, min_month_day=MIN_DATE)
        df_train = df_train.dropna(subset=["homeScore", "awayScore"])
        df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]

        df_val = load_multi_season_features(
            [holdout], no_garbage=True, adj_suffix=adj_suffix, min_month_day=MIN_DATE)
        df_val = df_val.dropna(subset=["homeScore", "awayScore"])
        df_val = df_val[(df_val["homeScore"] != 0) | (df_val["awayScore"] != 0)]

        # Merge book spreads
        book_spread_arr = np.full(len(df_val), np.nan)
        try:
            lines_df = load_research_lines(holdout)
            if not lines_df.empty:
                ld = lines_df.sort_values("provider").drop_duplicates(
                    subset=["gameId"], keep="first")
                if "spread" in ld.columns:
                    merge_df = ld[["gameId", "spread"]].rename(columns={"spread": "bookSpread"})
                    df_val = df_val.merge(merge_df, on="gameId", how="left")
                    book_spread_arr = df_val["bookSpread"].values.astype(float)
        except Exception:
            pass

        X_train = get_feature_matrix(df_train, feature_order=feature_order).values.astype(np.float32)
        targets_train = get_targets(df_train)
        y_train = targets_train["spread_home"].values.astype(np.float32)

        X_val = get_feature_matrix(df_val, feature_order=feature_order).values.astype(np.float32)
        targets_val = get_targets(df_val)
        y_val = targets_val["spread_home"].values.astype(np.float32)

        X_train = impute_column_means(X_train)
        X_val = impute_column_means(X_val)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)

        model, best_epoch = train_model(X_train_s, y_train, X_val_s, y_val)
        mu, sigma = predict_model(model, X_val_s)

        mae = np.mean(np.abs(mu - y_val))
        rmse = np.sqrt(np.mean((mu - y_val) ** 2))
        avg_sigma = np.mean(sigma)

        # ATS
        has_book = ~np.isnan(book_spread_arr)
        n_book = has_book.sum()
        if n_book > 0:
            bs = book_spread_arr[has_book]
            edge = mu[has_book] + bs
            pick_home = edge >= 0
            actual = y_val[has_book]
            home_covers = actual + bs > 0
            pushes = np.abs(actual + bs) < 0.5

            prob_edge = normal_cdf(np.abs(edge) / sigma[has_book]) - 0.5
            filt = prob_edge >= 0.05
            n_filt = filt.sum()
            if n_filt > 0:
                pick_wins = np.where(pick_home[filt], home_covers[filt], ~home_covers[filt]) & ~pushes[filt]
                w = pick_wins.sum()
                l = n_filt - pushes[filt].sum() - w
                wr = w / (w + l) if (w + l) > 0 else 0
                roi = (w * 100 - l * 110) / (n_filt * 110) if n_filt > 0 else 0
            else:
                wr, roi, w, l, n_filt = 0, 0, 0, 0, 0
        else:
            wr, roi, w, l, n_filt = 0, 0, 0, 0, 0

        yearly_results[holdout] = {
            "mae": mae, "rmse": rmse, "avg_sigma": avg_sigma,
            "best_epoch": best_epoch, "n_games": len(df_val),
            "ats_n": n_filt, "ats_w": w, "ats_l": l,
            "ats_wr": wr, "ats_roi": roi,
        }
        print(f"ep={best_epoch} MAE={mae:.2f} RMSE={rmse:.2f} σ={avg_sigma:.1f} "
              f"ATS@5%: {w}W-{l}L ({wr:.1%}) ROI={roi:.1%}")

    return yearly_results


def print_comparison(results_a, results_b, label_a="Iterative", label_b="WLS"):
    """Print side-by-side comparison of two configs."""
    print(f"\n{'='*100}")
    print(f"  WALK-FORWARD COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*100}")

    # Per-year table
    print(f"\n  {'Year':>4}  {'MAE-A':>7} {'MAE-B':>7} {'Δ':>7}  "
          f"{'RMSE-A':>7} {'RMSE-B':>7} {'Δ':>7}  "
          f"{'WR-A':>6} {'WR-B':>6} {'ROI-A':>7} {'ROI-B':>7}")
    print(f"  {'─'*4}  {'─'*7} {'─'*7} {'─'*7}  {'─'*7} {'─'*7} {'─'*7}  "
          f"{'─'*6} {'─'*6} {'─'*7} {'─'*7}")

    all_mae_a, all_mae_b = [], []
    all_rmse_a, all_rmse_b = [], []
    total_w_a, total_l_a, total_w_b, total_l_b = 0, 0, 0, 0

    for yr in HOLDOUT_YEARS:
        a = results_a.get(yr, {})
        b = results_b.get(yr, {})
        if not a or not b:
            continue

        d_mae = b["mae"] - a["mae"]
        d_rmse = b["rmse"] - a["rmse"]
        all_mae_a.append(a["mae"]); all_mae_b.append(b["mae"])
        all_rmse_a.append(a["rmse"]); all_rmse_b.append(b["rmse"])
        total_w_a += a["ats_w"]; total_l_a += a["ats_l"]
        total_w_b += b["ats_w"]; total_l_b += b["ats_l"]

        print(f"  {yr:>4}  {a['mae']:>7.2f} {b['mae']:>7.2f} {d_mae:>+7.2f}  "
              f"{a['rmse']:>7.2f} {b['rmse']:>7.2f} {d_rmse:>+7.2f}  "
              f"{a['ats_wr']:>6.1%} {b['ats_wr']:>6.1%} "
              f"{a['ats_roi']:>7.1%} {b['ats_roi']:>7.1%}")

    # Averages
    if all_mae_a:
        avg_mae_a = np.mean(all_mae_a)
        avg_mae_b = np.mean(all_mae_b)
        avg_rmse_a = np.mean(all_rmse_a)
        avg_rmse_b = np.mean(all_rmse_b)
        agg_wr_a = total_w_a / (total_w_a + total_l_a) if (total_w_a + total_l_a) > 0 else 0
        agg_wr_b = total_w_b / (total_w_b + total_l_b) if (total_w_b + total_l_b) > 0 else 0
        agg_roi_a = (total_w_a * 100 - total_l_a * 110) / ((total_w_a + total_l_a) * 110) if (total_w_a + total_l_a) > 0 else 0
        agg_roi_b = (total_w_b * 100 - total_l_b * 110) / ((total_w_b + total_l_b) * 110) if (total_w_b + total_l_b) > 0 else 0

        print(f"  {'─'*4}  {'─'*7} {'─'*7} {'─'*7}  {'─'*7} {'─'*7} {'─'*7}  "
              f"{'─'*6} {'─'*6} {'─'*7} {'─'*7}")
        print(f"  {'AVG':>4}  {avg_mae_a:>7.2f} {avg_mae_b:>7.2f} {avg_mae_b-avg_mae_a:>+7.2f}  "
              f"{avg_rmse_a:>7.2f} {avg_rmse_b:>7.2f} {avg_rmse_b-avg_rmse_a:>+7.2f}  "
              f"{agg_wr_a:>6.1%} {agg_wr_b:>6.1%} "
              f"{agg_roi_a:>7.1%} {agg_roi_b:>7.1%}")
        print(f"\n  Total picks: {label_a} {total_w_a}W-{total_l_a}L, "
              f"{label_b} {total_w_b}W-{total_l_b}L")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true",
                        help="Run walk-forward comparison using pre-built feature files")
    args = parser.parse_args()

    if not args.evaluate:
        print("Usage: poetry run python scripts/wls_walkforward.py --evaluate")
        print("\nNote: This script requires pre-built feature parquet files.")
        print("The WLS solver must first be run through the ETL pipeline to rebuild")
        print("the gold-layer ratings, then features must be built with the new ratings.")
        print("\nFor now, use scripts/wls_solver_validation.py to validate the solver directly.")
        sys.exit(0)

    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"WLS Walk-Forward Comparison")
    print(f"Features: {len(config.FEATURE_ORDER)}")
    print(f"Holdout years: {HOLDOUT_YEARS}")
    print()

    # Config A: Current production (iterative solver)
    print("=" * 50)
    print("CONFIG A: Iterative solver (current production)")
    print("=" * 50)
    results_a = run_walkforward(ADJ_SUFFIX, "ITER")

    # Config B: WLS solver — requires rebuilt features
    # For now, this uses the same features (same adj_suffix)
    # since the WLS solver hasn't been run through the ETL pipeline yet.
    # Once WLS ratings are built and features are regenerated, change
    # adj_suffix to point to the WLS-based feature files.
    print()
    print("=" * 50)
    print("CONFIG B: WLS solver (requires rebuilt features)")
    print("=" * 50)
    print("  Note: Using same features as Config A until WLS ratings are built through ETL.")
    # results_b = run_walkforward("adj_wls", "WLS")  # Uncomment when WLS features exist
    # print_comparison(results_a, results_b)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

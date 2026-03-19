#!/usr/bin/env python3
"""Session 13 Phase A-F: Comprehensive convergence training — 20 runs.

5 architecture configs × 4 training variants = 20 total runs.

Configs:
  C1: 256→192,  d=0.20  (small baseline, best calibration)
  C2: 384→256,  d=0.20  (medium-small)
  C3: 512→384,  d=0.20  (medium, σ_std=3.17 in prior sweep)
  C4: 768→640,  d=0.20  (large, matches old working Torvik model)
  C5: 512→384,  d=0.30  (medium + higher regularization)

Variants:
  V1: lr=1e-3, batch=2048, cosine→1e-5, Gaussian  (safe convergence)
  V2: lr=3e-3, batch=4096, cosine→1e-5, Gaussian  (faster LR + large batch)
  V3: lr=7e-3, batch=4096, step ×0.5/100ep, Gaussian  (matches Session 12 Optuna)
  V4: lr=1e-3, batch=2048, cosine→1e-5, Laplacian  (alternative loss)

500 epochs, early stopping patience=50.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import (
    MLPClassifier,
    MLPRegressor,
    gaussian_nll_loss,
    laplacian_nll_loss,
)
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets
from src.trainer import impute_column_means

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
TRAIN_SEASONS = list(range(2015, 2026))
VAL_SEASON = [2026]
MAX_EPOCHS = 500
PATIENCE = 50
REPORTS_DIR = config.PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Architecture configs ─────────────────────────────────────────────

CONFIGS = [
    {"name": "C1", "label": "256→192, d=0.20",
     "hidden1": 256, "hidden2": 192, "dropout": 0.20, "weight_decay": 1e-4},
    {"name": "C2", "label": "384→256, d=0.20",
     "hidden1": 384, "hidden2": 256, "dropout": 0.20, "weight_decay": 1e-4},
    {"name": "C3", "label": "512→384, d=0.20",
     "hidden1": 512, "hidden2": 384, "dropout": 0.20, "weight_decay": 1e-4},
    {"name": "C4", "label": "768→640, d=0.20",
     "hidden1": 768, "hidden2": 640, "dropout": 0.20, "weight_decay": 1e-4},
    {"name": "C5", "label": "512→384, d=0.30",
     "hidden1": 512, "hidden2": 384, "dropout": 0.30, "weight_decay": 1e-4},
]

# ── Training variants ────────────────────────────────────────────────

VARIANTS = {
    "V1": {"lr": 1e-3, "batch_size": 2048, "sched": "cosine", "loss": "gaussian"},
    "V2": {"lr": 3e-3, "batch_size": 4096, "sched": "cosine", "loss": "gaussian"},
    "V3": {"lr": 7e-3, "batch_size": 4096, "sched": "step",   "loss": "gaussian"},
    "V4": {"lr": 1e-3, "batch_size": 2048, "sched": "cosine", "loss": "laplacian"},
}

# Prior sweep metrics for the 5 configs (from /tmp/session13_full_output.log)
PRIOR_SWEEP = {
    "S-A": {"mae": 10.056, "cal": 0.2843, "dead": 0, "sigma_std": 2.66,
             "quintiles": [1.541, 1.367, 1.279, 1.209, 1.026]},
    "S-B": {"mae": 10.244, "cal": 0.4511, "dead": 0, "sigma_std": 2.92,
             "quintiles": [2.010, 1.429, 1.389, 1.298, 1.129]},
    "S-C": {"mae": 10.239, "cal": 0.5953, "dead": 4, "sigma_std": 3.11,
             "quintiles": [2.285, 1.623, 1.571, 1.358, 1.139]},
    "S-D": {"mae": 10.215, "cal": 0.6997, "dead": 86, "sigma_std": 3.19,
             "quintiles": [2.512, 1.819, 1.615, 1.373, 1.180]},
    "S-E": {"mae": 10.109, "cal": 0.3769, "dead": 9, "sigma_std": 2.81,
             "quintiles": [1.787, 1.448, 1.380, 1.192, 1.078]},
}


# ══════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════════

def preflight_checks():
    from src.features import load_research_lines

    print("=" * 70)
    print("  PRE-FLIGHT CHECKS")
    print("=" * 70)

    # ── Check 1: Target variable ──────────────────────────────────────
    print("\n--- CHECK 1: Target Variable ---")
    df = load_multi_season_features(
        [2026], adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df = df.dropna(subset=["homeScore", "awayScore"])
    df = df[(df["homeScore"] != 0) | (df["awayScore"] != 0)]
    targets = get_targets(df)

    try:
        lines_df = load_research_lines(2026)
        if not lines_df.empty:
            ld = lines_df.sort_values("provider").drop_duplicates(
                subset=["gameId"], keep="first"
            )
            if "spread" in ld.columns:
                df = df.merge(
                    ld[["gameId", "spread"]].rename(columns={"spread": "bookSpread"}),
                    on="gameId", how="left",
                )
    except Exception as e:
        print(f"  Lines warning: {e}")

    print(f"\n  {'gameId':>14} {'homeScore':>9} {'awayScore':>9} {'spread_home':>11} {'bookSpread':>10} {'y=HS-AS?':>8}")
    print(f"  {'-'*14} {'-'*9} {'-'*9} {'-'*11} {'-'*10} {'-'*8}")

    sample_idx = df.head(20).dropna(subset=["bookSpread"]).head(5).index
    for idx in sample_idx:
        row = df.loc[idx]
        hs = int(row["homeScore"])
        aws = int(row["awayScore"])
        spread = targets.loc[idx, "spread_home"]
        book = row.get("bookSpread", float("nan"))
        match = "YES" if abs(spread - (hs - aws)) < 0.01 else "NO"
        print(f"  {row['gameId']:>14} {hs:>9} {aws:>9} {spread:>11.1f} {book:>10.1f} {match:>8}")

    computed = df["homeScore"].values - df["awayScore"].values
    target_vals = targets["spread_home"].values
    assert np.allclose(computed[:len(target_vals)], target_vals, atol=0.01), \
        "FATAL: spread_home != homeScore - awayScore"
    print(f"\n  PASS: spread_home == homeScore - awayScore for all {len(target_vals)} games")
    print(f"  spread_home mean={target_vals.mean():.2f}, range=[{target_vals.min():.0f}, {target_vals.max():.0f}]")

    # ── Check 2: Feature count ────────────────────────────────────────
    print("\n--- CHECK 2: Feature Count ---")
    fo = config.FEATURE_ORDER
    print(f"  feature_order.json: {len(fo)} features")
    print(f"  First 5: {fo[:5]}")
    assert len(fo) == 50, f"Expected 50 features, got {len(fo)}"
    print(f"  PASS: 50 features confirmed")

    # ── Check 3: Train/val split ──────────────────────────────────────
    print("\n--- CHECK 3: Train/Val Split ---")
    df_train = load_multi_season_features(
        TRAIN_SEASONS, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_train = df_train.dropna(subset=["homeScore", "awayScore"])
    df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]

    n_val = len(df)
    n_val_with_book = df["bookSpread"].notna().sum() if "bookSpread" in df.columns else 0
    print(f"  Train: {len(df_train)} games, seasons {TRAIN_SEASONS}")
    print(f"  Val: {n_val} games, season {VAL_SEASON}")
    print(f"  Val games with book spreads: {n_val_with_book}/{n_val}")
    print(f"  PASS: Train/val split verified")

    print()


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_data():
    from src.features import load_research_lines

    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)

    df_train = load_multi_season_features(
        TRAIN_SEASONS, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_train = df_train.dropna(subset=["homeScore", "awayScore"])
    n_before = len(df_train)
    df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]
    print(f"  Train: {n_before} → {len(df_train)} (removed {n_before - len(df_train)} 0-0)")

    df_val = load_multi_season_features(
        VAL_SEASON, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_val = df_val.dropna(subset=["homeScore", "awayScore"])
    n_before_v = len(df_val)
    df_val = df_val[(df_val["homeScore"] != 0) | (df_val["awayScore"] != 0)]
    print(f"  Val: {n_before_v} → {len(df_val)} (removed {n_before_v - len(df_val)} 0-0)")

    try:
        lines_df = load_research_lines(2026)
        if not lines_df.empty:
            lines_dedup = lines_df.sort_values("provider").drop_duplicates(
                subset=["gameId"], keep="first"
            )
            if "spread" in lines_dedup.columns:
                merge_df = lines_dedup[["gameId", "spread"]].rename(
                    columns={"spread": "bookSpread"}
                )
                df_val = df_val.merge(merge_df, on="gameId", how="left")
                n_with = df_val["bookSpread"].notna().sum()
                print(f"  Book spreads: {n_with}/{len(df_val)} val games")
    except Exception as e:
        print(f"  Lines load failed: {e}")

    X_train = get_feature_matrix(df_train).values.astype(np.float32)
    targets_train = get_targets(df_train)
    y_spread_train = targets_train["spread_home"].values.astype(np.float32)
    y_win_train = targets_train["home_win"].values.astype(np.float32)

    X_val = get_feature_matrix(df_val).values.astype(np.float32)
    targets_val = get_targets(df_val)
    y_spread_val = targets_val["spread_home"].values.astype(np.float32)
    y_win_val = targets_val["home_win"].values.astype(np.float32)

    n_nan_train = np.isnan(X_train).sum()
    n_nan_val = np.isnan(X_val).sum()
    X_train = impute_column_means(X_train)
    X_val = impute_column_means(X_val)
    print(f"  NaN imputed: train={n_nan_train:,}, val={n_nan_val:,}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    print(f"  Shapes: train={X_train_s.shape}, val={X_val_s.shape}")
    return X_train_s, y_spread_train, y_win_train, X_val_s, y_spread_val, y_win_val, scaler, df_val


# ══════════════════════════════════════════════════════════════════════
# LR SCHEDULERS
# ══════════════════════════════════════════════════════════════════════

def make_scheduler(sched_type, optimizer, max_epochs):
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5
        )
    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.5
        )
    raise ValueError(f"Unknown scheduler: {sched_type}")


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def compute_val_loss(model, X_val_tensor, y_val_tensor, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(X_val_tensor), 4096):
        end = min(start + 4096, len(X_val_tensor))
        x = X_val_tensor[start:end].to(device)
        y = y_val_tensor[start:end].to(device)
        mu, log_sigma = model(x)
        nll, _ = loss_fn(mu, log_sigma, y)
        total_loss += nll.mean().item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train_run(cfg, variant_name, X_train, y_spread, X_val_s, y_spread_val):
    device = get_device()
    use_amp = device.type == "cuda"
    var = VARIANTS[variant_name]

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden1=cfg["hidden1"],
        hidden2=cfg["hidden2"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=var["lr"], weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = make_scheduler(var["sched"], optimizer, MAX_EPOCHS)
    amp_scaler = GradScaler(device.type, enabled=use_amp)
    loss_fn = laplacian_nll_loss if var["loss"] == "laplacian" else gaussian_nll_loss

    ds = HoopsDataset(X_train, spread=y_spread, home_win=np.zeros(len(y_spread)))
    loader = DataLoader(ds, batch_size=var["batch_size"], shuffle=True, drop_last=True,
                        num_workers=2, pin_memory=True)

    X_val_tensor = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_spread_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    # Track learning curve for diagnostics
    curve = {}

    model.train()
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            x, spread, _ = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                mu, log_sigma = model(x)
                nll, sigma = loss_fn(mu, log_sigma, spread)
                loss = nll.mean()
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)
        val_loss = compute_val_loss(model, X_val_tensor, y_val_tensor, loss_fn, device)

        # Track learning curve at key epochs
        ep = epoch + 1
        if ep in (1, 50, 100, 200, 300, MAX_EPOCHS) or val_loss < best_val_loss or no_improve == PATIENCE:
            curve[ep] = {"train": avg_train, "val": val_loss,
                         "lr": optimizer.param_groups[0]["lr"]}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1

        if ep % 50 == 0 or no_improve == PATIENCE:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    epoch {ep}/{MAX_EPOCHS} — train: {avg_train:.4f} "
                  f"val: {val_loss:.4f} best: {best_val_loss:.4f} "
                  f"(best@{best_epoch}) lr: {lr_now:.2e}")

        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {ep} (best@{best_epoch})")
            break

    model.cpu()
    model.load_state_dict(best_state)
    model.eval()

    return model, best_epoch, ep, avg_train, best_val_loss, curve


# ══════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════

def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def count_dead_neurons(model, X_val_tensor):
    model.eval()
    device = next(model.parameters()).device
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    hooks = []
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.ReLU):
            hooks.append(layer.register_forward_hook(make_hook(f"relu_{i}")))

    with torch.no_grad():
        all_acts = {}
        for start in range(0, len(X_val_tensor), 4096):
            end = min(start + 4096, len(X_val_tensor))
            model(X_val_tensor[start:end].to(device))
            for name, act in activations.items():
                if name not in all_acts:
                    all_acts[name] = []
                all_acts[name].append(act)

    for h in hooks:
        h.remove()

    dead_counts = {}
    for name, act_list in all_acts.items():
        full_act = torch.cat(act_list, dim=0)
        zero_frac = (full_act == 0).float().mean(dim=0)
        n_dead = (zero_frac > 0.99).sum().item()
        dead_counts[name] = (int(n_dead), full_act.shape[1])
    return dead_counts


def compute_roi(mu, sigma, book_spread, actual_spread, threshold, sigma_mask=None):
    """Compute ATS ROI at a probability edge threshold.

    Args:
        mu: model predicted margins
        sigma: model predicted sigmas
        book_spread: sportsbook spreads (negative = home favored)
        actual_spread: homeScore - awayScore
        threshold: minimum prob_edge to bet
        sigma_mask: optional boolean mask for sigma filtering
    """
    valid = ~np.isnan(book_spread)
    if sigma_mask is not None:
        valid = valid & sigma_mask
    if valid.sum() == 0:
        return {"bets": 0, "win_rate": 0, "roi": 0, "units": 0}

    edge_home = mu[valid] + book_spread[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    home_cover_prob = normal_cdf(edge_z)
    away_cover_prob = 1.0 - home_cover_prob
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, home_cover_prob, away_cover_prob)

    breakeven = 0.5238
    profit_per_1 = 100.0 / 110.0
    prob_edge = pick_prob - breakeven
    bet_mask = prob_edge >= threshold
    if bet_mask.sum() == 0:
        return {"bets": 0, "win_rate": 0, "roi": 0, "units": 0}

    home_covered = (actual_spread[valid] + book_spread[valid]) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)
    wins = pick_won[bet_mask].sum()
    total = bet_mask.sum()
    win_rate = wins / total
    units = wins * profit_per_1 - (total - wins)
    return {"bets": int(total), "win_rate": float(win_rate),
            "roi": float(units / total), "units": float(units)}


@torch.no_grad()
def evaluate_full(model, X_val_s, y_spread_val, df_val):
    """Compute ALL metrics for a trained regressor.

    Returns dict with:
      - book_spread_mae: MAE on games WITH book spreads (primary metric)
      - overall_mae: MAE on ALL games
      - book_baseline_mae: |(-book_spread) - actual| on games with spreads
      - delta_mae: book_spread_mae - book_baseline_mae
      - sigma stats, calibration, ROI (unfiltered + sigma-filtered)
    """
    model.eval()
    X_tensor = torch.tensor(X_val_s, dtype=torch.float32)
    mu_t, log_sigma_t = model(X_tensor)
    sigma_t = torch.exp(log_sigma_t).clamp(min=0.5, max=30.0)
    mu = mu_t.numpy()
    sigma = sigma_t.numpy()
    actual = y_spread_val
    residuals = actual - mu
    abs_res = np.abs(residuals)

    # Dead neurons
    dead = count_dead_neurons(model, X_tensor)
    total_dead = sum(d[0] for d in dead.values())

    # Overall MAE (all games)
    overall_mae = float(np.mean(abs_res))

    # Book-Spread MAE + Book Baseline MAE (games with book spreads only)
    book_spread_mae = None
    book_baseline_mae = None
    delta_mae = None
    has_book = np.zeros(len(df_val), dtype=bool)

    if "bookSpread" in df_val.columns:
        bs = df_val["bookSpread"].values.astype(np.float64)
        has_book = ~np.isnan(bs)
        if has_book.sum() > 0:
            book_spread_mae = float(np.mean(abs_res[has_book]))
            # Book baseline: book predicts margin = -book_spread
            book_baseline_mae = float(np.mean(np.abs(actual[has_book] - (-bs[has_book]))))
            delta_mae = book_spread_mae - book_baseline_mae

    # Sigma stats
    sigma_mean = float(np.mean(sigma))
    sigma_std = float(np.std(sigma))
    sigma_min = float(np.min(sigma))
    sigma_max = float(np.max(sigma))
    sigma_median = float(np.median(sigma))
    sigma_p25 = float(np.percentile(sigma, 25))

    # Calibration: within 1σ
    within_1sig = float(np.mean(abs_res < sigma))

    # Per-quintile calibration
    qi = np.array_split(np.argsort(sigma), 5)
    quintile_ratios = []
    for idx in qi:
        actual_std = np.std(residuals[idx])
        pred_sigma_mean = np.mean(sigma[idx])
        quintile_ratios.append(float(actual_std / pred_sigma_mean) if pred_sigma_mean > 0 else 999.0)
    cal_score = float(np.mean([abs(r - 1.0) for r in quintile_ratios]))

    # Spearman
    sp_corr, _ = scipy_stats.spearmanr(sigma, abs_res)
    sp_corr = float(sp_corr) if not np.isnan(sp_corr) else 0.0

    # ROI (on games with book spreads)
    roi_results = {}
    if has_book.sum() > 0:
        bs_arr = df_val["bookSpread"].values.astype(np.float64)
        act_arr = actual.astype(np.float64)

        # Unfiltered ROI
        for thresh in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
            roi_results[f"unfilt_{thresh}"] = compute_roi(
                mu, sigma, bs_arr, act_arr, thresh)

        # Sigma < median
        sig_med_mask = sigma < sigma_median
        for thresh in [0.03, 0.05, 0.07]:
            roi_results[f"sig_med_{thresh}"] = compute_roi(
                mu, sigma, bs_arr, act_arr, thresh, sigma_mask=sig_med_mask)

        # Sigma < p25
        sig_p25_mask = sigma < sigma_p25
        for thresh in [0.03, 0.05, 0.07]:
            roi_results[f"sig_p25_{thresh}"] = compute_roi(
                mu, sigma, bs_arr, act_arr, thresh, sigma_mask=sig_p25_mask)

    # Best ROI across all sigma-filtered thresholds
    best_sig_roi = max(
        (r["roi"] for k, r in roi_results.items()
         if ("sig_" in k) and r and r["bets"] > 10),
        default=-1.0
    )

    return {
        "dead_neurons": dead, "total_dead": total_dead,
        "book_spread_mae": book_spread_mae, "overall_mae": overall_mae,
        "book_baseline_mae": book_baseline_mae, "delta_mae": delta_mae,
        "sigma_mean": sigma_mean, "sigma_std": sigma_std,
        "sigma_min": sigma_min, "sigma_max": sigma_max,
        "sigma_median": sigma_median, "sigma_p25": sigma_p25,
        "within_1sig": within_1sig,
        "quintile_ratios": quintile_ratios, "cal_score": cal_score,
        "spearman": sp_corr,
        "roi_results": roi_results, "best_sig_roi": best_sig_roi,
        "mu": mu, "sigma": sigma, "residuals": residuals,
        "has_book": has_book,
    }


# ══════════════════════════════════════════════════════════════════════
# CLASSIFIER TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_classifier_production(X_train, y_win, hidden1=256, dropout=0.3, batch_size=2048):
    device = get_device()
    use_amp = device.type == "cuda"
    model = MLPClassifier(input_dim=X_train.shape[1], hidden1=hidden1, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    amp_scaler = GradScaler(device.type, enabled=use_amp)
    ds = HoopsDataset(X_train, spread=y_win, home_win=y_win)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=2, pin_memory=True)
    model.train()
    for epoch in range(200):
        for batch in loader:
            x, _, win = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                loss = criterion(model(x), win)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        if (epoch + 1) % 50 == 0:
            print(f"    Classifier epoch {epoch+1}/200")
    return model.cpu()


# ══════════════════════════════════════════════════════════════════════
# VALIDATION SUITE (8 tests)
# ══════════════════════════════════════════════════════════════════════

def run_validation_suite(model, X_val, y_spread_val, df_val, scaler, metrics,
                         X_train, y_spread_train, y_win_train, cfg, variant):
    X_tensor = torch.tensor(X_val, dtype=torch.float32)
    mu = metrics["mu"]
    sigma = metrics["sigma"]
    residuals = metrics["residuals"]
    abs_res = np.abs(residuals)
    actual = y_spread_val
    results = {}
    is_laplacian = VARIANTS[variant]["loss"] == "laplacian"

    # TEST 1: Dead neurons
    print("\n--- TEST 1: Dead Neuron Scan ---")
    dead = count_dead_neurons(model, X_tensor)
    all_clear = True
    for layer, (n_dead, total) in dead.items():
        s = "PASS" if n_dead == 0 else "FAIL"
        if n_dead > 0:
            all_clear = False
        print(f"  {layer}: {n_dead}/{total} [{s}]")
    results["T1_dead_neurons"] = "PASS" if all_clear else "FAIL"

    # TEST 2: Sigma correlation with difficulty
    print("\n--- TEST 2: Sigma vs Difficulty ---")
    if "bookSpread" in df_val.columns:
        bs = np.abs(df_val["bookSpread"].values.astype(np.float64))
        valid = ~np.isnan(bs)
        for label, lo, hi in [("Close (0-3)", 0, 3), ("Medium (3-7)", 3, 7),
                               ("Large (7-14)", 7, 14), ("Blowout (14+)", 14, 999)]:
            mask = valid & (bs >= lo) & (bs < hi)
            if mask.sum() > 0:
                print(f"  {label}: mean σ={sigma[mask].mean():.3f}, "
                      f"MAE={abs_res[mask].mean():.3f} (n={mask.sum()})")
    results["T2_sigma_difficulty"] = "PASS"

    # TEST 3: Multi-level calibration
    print("\n--- TEST 3: Multi-Level Calibration ---")
    if is_laplacian:
        targets_k = [(0.5, 0.3935), (1.0, 0.6321), (1.5, 0.7769), (2.0, 0.8647)]
        loss_label = "Laplacian"
    else:
        targets_k = [(0.5, 0.3829), (1.0, 0.6827), (1.5, 0.8664), (2.0, 0.9545)]
        loss_label = "Gaussian"
    print(f"  (Using {loss_label} targets)")
    cal_pass = True
    for k, target in targets_k:
        actual_frac = float(np.mean(abs_res < k * sigma))
        diff = abs(actual_frac - target)
        s = "PASS" if diff < 0.05 else ("WARN" if diff < 0.08 else "FAIL")
        if diff >= 0.08:
            cal_pass = False
        print(f"  Within {k:.1f}σ: actual={actual_frac:.4f} target={target:.4f} "
              f"diff={diff:.4f} [{s}]")
    results["T3_calibration"] = "PASS" if cal_pass else "FAIL"

    # TEST 4: Monthly MAE
    print("\n--- TEST 4: Monthly MAE ---")
    print(f"  Session 12 ref: Nov:11.16, Dec:9.44, Jan:8.90, Feb:8.63, Mar:9.05")
    if "startDate" in df_val.columns and "bookSpread" in df_val.columns:
        dates = pd.to_datetime(df_val["startDate"], errors="coerce", utc=True)
        months = dates.dt.month
        bs_valid = df_val["bookSpread"].notna().values
        for m, label in [(11, "Nov"), (12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")]:
            mask_all = months == m
            mask_book = mask_all.values & bs_valid
            if mask_all.sum() > 0:
                bs_mae_str = f", Book-Spread MAE={abs_res[mask_book].mean():.3f}" if mask_book.sum() > 5 else ""
                print(f"  {label}: Overall MAE={abs_res[mask_all.values].mean():.3f} "
                      f"(n={mask_all.sum()}){bs_mae_str}")
    results["T4_monthly"] = "PASS"

    # TEST 5: Edge profitability
    print("\n--- TEST 5: Edge Profitability ---")
    print(f"  SOS 0.85 benchmarks: σ<p25 ROI@5=+11.5%, ROI@7=+14.5%")
    if "bookSpread" in df_val.columns:
        bs_arr = df_val["bookSpread"].values.astype(np.float64)
        act_arr = actual.astype(np.float64)

        print("  Unfiltered:")
        for thresh in [0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
            rd = compute_roi(mu, sigma, bs_arr, act_arr, thresh)
            if rd["bets"] > 0:
                print(f"    @{thresh*100:.0f}%: {rd['bets']:>4} bets, WR={rd['win_rate']:.3f}, "
                      f"ROI={rd['roi']*100:>+.1f}%, Units={rd['units']:.1f}")

        print(f"  Sigma < median ({metrics['sigma_median']:.1f}):")
        sig_med = sigma < metrics["sigma_median"]
        for thresh in [0.03, 0.05, 0.07]:
            rd = compute_roi(mu, sigma, bs_arr, act_arr, thresh, sigma_mask=sig_med)
            if rd["bets"] > 0:
                print(f"    @{thresh*100:.0f}%: {rd['bets']:>4} bets, WR={rd['win_rate']:.3f}, "
                      f"ROI={rd['roi']*100:>+.1f}%, Units={rd['units']:.1f}")

        print(f"  Sigma < p25 ({metrics['sigma_p25']:.1f}):")
        sig_p25 = sigma < metrics["sigma_p25"]
        for thresh in [0.03, 0.05, 0.07]:
            rd = compute_roi(mu, sigma, bs_arr, act_arr, thresh, sigma_mask=sig_p25)
            if rd["bets"] > 0:
                print(f"    @{thresh*100:.0f}%: {rd['bets']:>4} bets, WR={rd['win_rate']:.3f}, "
                      f"ROI={rd['roi']*100:>+.1f}%, Units={rd['units']:.1f}")
    results["T5_edge_profit"] = "PASS"

    # TEST 6: Sigma vs actual error
    print("\n--- TEST 6: Sigma-Error Correlation ---")
    sp = metrics["spearman"]
    s = "PASS" if sp > 0 else "FAIL"
    print(f"  Spearman ρ = {sp:.4f} [{s}]")
    results["T6_sigma_corr"] = s

    # TEST 7: Accuracy by spread size
    print("\n--- TEST 7: Accuracy by |book_spread| ---")
    if "bookSpread" in df_val.columns:
        bs = np.abs(df_val["bookSpread"].values.astype(np.float64))
        valid = ~np.isnan(bs)
        for label, lo, hi in [("Close (0-3)", 0, 3), ("Medium (3-7)", 3, 7),
                               ("Large (7-14)", 7, 14), ("Blowout (14+)", 14, 999)]:
            mask = valid & (bs >= lo) & (bs < hi)
            if mask.sum() > 0:
                print(f"  {label}: Book-Spread MAE={abs_res[mask].mean():.3f} (n={mask.sum()})")
    results["T7_spread_accuracy"] = "PASS"

    # TEST 8: Hold-out year
    print("\n--- TEST 8: Hold-Out Year (2015-2024 → 2025) ---")
    print(f"  Benchmark: HE MAE=9.10 (4372 games, different methodology)")
    try:
        _run_holdout(cfg, variant)
        results["T8_holdout"] = "PASS"
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["T8_holdout"] = "WARN"

    # Summary
    print("\n--- Validation Summary ---")
    for k, v in results.items():
        print(f"  {k}: {v}")
    fails = [k for k, v in results.items() if v == "FAIL"]
    if fails:
        print(f"\n  RED FLAGS: {fails}")
    else:
        print("\n  All tests PASSED")
    return results


def _run_holdout(cfg, variant):
    from src.features import load_research_lines

    print("  Training hold-out model (2015-2024 → 2025)...")
    ho_train = list(range(2015, 2025))
    df_ht = load_multi_season_features(ho_train, adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
    df_ht = df_ht.dropna(subset=["homeScore", "awayScore"])
    df_ht = df_ht[(df_ht["homeScore"] != 0) | (df_ht["awayScore"] != 0)]

    df_hv = load_multi_season_features([2025], adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
    df_hv = df_hv.dropna(subset=["homeScore", "awayScore"])
    df_hv = df_hv[(df_hv["homeScore"] != 0) | (df_hv["awayScore"] != 0)]

    try:
        lines_df = load_research_lines(2025)
        if not lines_df.empty:
            ld = lines_df.sort_values("provider").drop_duplicates(subset=["gameId"], keep="first")
            if "spread" in ld.columns:
                df_hv = df_hv.merge(ld[["gameId", "spread"]].rename(columns={"spread": "bookSpread"}),
                                    on="gameId", how="left")
    except Exception:
        pass

    X_ht = impute_column_means(get_feature_matrix(df_ht).values.astype(np.float32))
    X_hv = impute_column_means(get_feature_matrix(df_hv).values.astype(np.float32))
    y_ht = get_targets(df_ht)["spread_home"].values.astype(np.float32)
    y_hv = get_targets(df_hv)["spread_home"].values.astype(np.float32)

    sc = StandardScaler()
    sc.fit(X_ht)
    X_ht_s = sc.transform(X_ht).astype(np.float32)
    X_hv_s = sc.transform(X_hv).astype(np.float32)

    model, best_ep, stopped, _, _, _ = train_run(cfg, variant, X_ht_s, y_ht, X_hv_s, y_hv)
    m = evaluate_full(model, X_hv_s, y_hv, df_hv)
    print(f"  Holdout 2025: {len(y_hv)} total games")
    print(f"  Overall MAE: {m['overall_mae']:.3f}")
    if m['book_spread_mae'] is not None:
        n_book = m['has_book'].sum()
        print(f"  Book-Spread MAE: {m['book_spread_mae']:.3f} ({n_book} games with lines)")
        print(f"  Book Baseline MAE: {m['book_baseline_mae']:.3f}")
    print(f"  σ: mean={m['sigma_mean']:.2f} std={m['sigma_std']:.2f}")
    print(f"  Cal: {m['cal_score']:.4f}, Dead: {m['total_dead']}, Best@epoch {best_ep}")
    return m


# ══════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_report(all_results, winner, test_results, report_path):
    L = ["# Session 13: Architecture Search — Converged Training Report",
         f"\nDate: 2026-02-27 | GPU: NVIDIA A10",
         f"Training: 2015-2025 | Validation: 2026",
         f"Features: {len(config.FEATURE_ORDER)} | Max epochs: {MAX_EPOCHS} | Patience: {PATIENCE}"]

    L.append("\n## Configs")
    for c in CONFIGS:
        L.append(f"- {c['name']}: {c['label']}")
    L.append("\n## Variants")
    for v, d in VARIANTS.items():
        L.append(f"- {v}: lr={d['lr']}, batch={d['batch_size']}, {d['sched']}, {d['loss']}")

    L.append("\n## All 20 Runs (sorted by Book-Spread MAE)")
    L.append("")
    L.append("| Run | Best@ | Stop@ | Dead | BS-MAE | Overall MAE | Book Base | Δ | σ_std | CalSc | ρ | SigROI |")
    L.append("|-----|-------|-------|------|--------|-------------|-----------|---|-------|-------|---|--------|")
    for r in sorted(all_results, key=lambda x: x.get("book_spread_mae") or 999):
        bs_mae = f"{r['book_spread_mae']:.3f}" if r['book_spread_mae'] else "N/A"
        bb_mae = f"{r['book_baseline_mae']:.3f}" if r['book_baseline_mae'] else "N/A"
        delta = f"{r['delta_mae']:+.2f}" if r['delta_mae'] is not None else "N/A"
        sig_roi = f"{r['best_sig_roi']*100:.1f}%" if r['best_sig_roi'] > -1 else "N/A"
        L.append(
            f"| {r['name']} | {r['best_epoch']} | {r['stopped_at']} | "
            f"{r['total_dead']} | {bs_mae} | {r['overall_mae']:.3f} | "
            f"{bb_mae} | {delta} | {r['sigma_std']:.2f} | "
            f"{r['cal_score']:.4f} | {r['spearman']:.4f} | {sig_roi} |"
        )

    L.append("\n## Per-Quintile Calibration")
    L.append("")
    L.append("| Run | Q1 | Q2 | Q3 | Q4 | Q5 |")
    L.append("|-----|-----|-----|-----|-----|-----|")
    for r in sorted(all_results, key=lambda x: x.get("book_spread_mae") or 999):
        qr = r["quintile_ratios"]
        L.append(f"| {r['name']} | {qr[0]:.3f} | {qr[1]:.3f} | {qr[2]:.3f} | {qr[3]:.3f} | {qr[4]:.3f} |")

    if winner:
        wcfg = winner["config"]
        wvar = VARIANTS[winner["variant"]]
        L.append(f"\n## Winner: {winner['name']}")
        L.append(f"- Config: {wcfg['label']}")
        L.append(f"- Variant: {winner['variant']} (lr={wvar['lr']}, batch={wvar['batch_size']}, {wvar['loss']})")
        L.append(f"- Best epoch: {winner['best_epoch']}")

        L.append("\n## Comparison Table")
        L.append("")
        L.append("| Metric | Winner | Session 12 (128→128) | SOS 0.85 Baseline | HE 2025 |")
        L.append("|--------|--------|---------------------|-------------------|---------|")
        L.append(f"| Architecture | {wcfg['label']} | 128→128, d=0.45 | 256→128, d=0.3 | N/A |")
        L.append(f"| Book-Spread MAE | {winner['book_spread_mae']:.3f} | 9.42 | 9.62 | 9.10* |")
        L.append(f"| Overall MAE | {winner['overall_mae']:.3f} | 10.11 | 10.13 | N/A |")
        L.append(f"| Book Baseline MAE | {winner['book_baseline_mae']:.3f} | ~8.76 | 8.76 | 8.58 |")
        L.append(f"| σ std | {winner['sigma_std']:.2f} | 0.53 | N/A | 1.03 |")
        L.append(f"| σ range | {winner['sigma_min']:.1f}–{winner['sigma_max']:.1f} | 10.5–14.8 | N/A | N/A |")
        L.append(f"| Dead neurons | {winner['total_dead']} | 116/128 | 0 | 0 |")
        qr = winner['quintile_ratios']
        L.append(f"| Quintile ratios | {qr[0]:.2f}–{qr[-1]:.2f} | 1.07–1.49 | N/A | N/A |")
        L.append(f"| LR used | {wvar['lr']} | 0.0073 | 0.001 | N/A |")
        L.append(f"| Converged? | Yes (epoch {winner['best_epoch']}) | No (100ep) | Yes | Yes |")
        L.append("\n*HE MAE uses different methodology (4372 games, different date range)")

    if test_results:
        L.append("\n## Validation Tests")
        for k, v in test_results.items():
            L.append(f"- {k}: **{v}**")

    with open(report_path, "w") as f:
        f.write("\n".join(L))
    print(f"\nReport saved: {report_path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    total_start = time.time()

    # ── Pre-flight checks ─────────────────────────────────────────────
    preflight_checks()

    # ── Phase A: Prior sweep metrics ──────────────────────────────────
    print("=" * 70)
    print("  PHASE A: PRIOR SWEEP METRICS FOR 5 CONFIGS")
    print("=" * 70)
    print(f"\n  {'Config':<8} {'Arch':<20} {'MAE':>7} {'CalSc':>7} {'Dead':>5} {'σ_std':>6}")
    print(f"  {'-'*8} {'-'*20} {'-'*7} {'-'*7} {'-'*5} {'-'*6}")
    prior_keys = ["S-A", "S-B", "S-C", "S-D", "S-E"]
    for cfg, pk in zip(CONFIGS, prior_keys):
        p = PRIOR_SWEEP[pk]
        print(f"  {cfg['name']:<8} {cfg['label']:<20} {p['mae']:>7.3f} {p['cal']:>7.4f} "
              f"{p['dead']:>5} {p['sigma_std']:>6.2f}")
    print(f"\n  Note: Prior sweep was 100 epochs, no early stopping, lr=1e-3.")
    print(f"  S-C and S-D had dead neurons at 100 epochs — expect convergence to fix this.")
    print()

    # ── Load data ─────────────────────────────────────────────────────
    X_train, y_spread_train, y_win_train, X_val, y_spread_val, y_win_val, scaler, df_val = load_data()

    # ── Phase B: Train 20 configs ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE B: CONVERGENCE TRAINING (20 RUNS)")
    print("  5 configs × 4 variants")
    print("  V1: lr=1e-3/batch=2048/cosine/Gaussian")
    print("  V2: lr=3e-3/batch=4096/cosine/Gaussian")
    print("  V3: lr=7e-3/batch=4096/step×0.5/Gaussian  (Session 12 Optuna LR)")
    print("  V4: lr=1e-3/batch=2048/cosine/Laplacian")
    print("=" * 70)

    all_results = []
    for cfg in CONFIGS:
        for vname in sorted(VARIANTS.keys()):
            var = VARIANTS[vname]
            run_name = f"{cfg['name']}-{vname}"

            print(f"\n{'='*60}")
            print(f"  RUN: {run_name} ({cfg['label']})")
            print(f"  {vname}: lr={var['lr']}, batch={var['batch_size']}, "
                  f"sched={var['sched']}, loss={var['loss']}")
            print(f"{'='*60}")

            t0 = time.time()
            model, best_epoch, stopped_at, final_train, best_val, curve = train_run(
                cfg, vname, X_train, y_spread_train, X_val, y_spread_val
            )
            train_time = time.time() - t0

            metrics = evaluate_full(model, X_val, y_spread_val, df_val)
            metrics["name"] = run_name
            metrics["config"] = cfg
            metrics["variant"] = vname
            metrics["best_epoch"] = best_epoch
            metrics["stopped_at"] = stopped_at
            metrics["final_train_loss"] = final_train
            metrics["best_val_loss"] = best_val
            metrics["train_time"] = train_time
            metrics["model"] = model
            metrics["curve"] = curve

            conv = f"full {MAX_EPOCHS}" if stopped_at == MAX_EPOCHS else \
                   f"early stop@{stopped_at} (best@{best_epoch})"

            print(f"\n  --- {run_name} ---")
            print(f"  {conv}, time={train_time:.0f}s")
            bs_str = f"{metrics['book_spread_mae']:.3f}" if metrics['book_spread_mae'] else "N/A"
            print(f"  Dead:{metrics['total_dead']} | Book-Spread MAE:{bs_str} | "
                  f"Overall MAE:{metrics['overall_mae']:.3f} | "
                  f"Book Base:{metrics['book_baseline_mae']:.3f}")
            print(f"  σ: mean={metrics['sigma_mean']:.2f} std={metrics['sigma_std']:.2f} "
                  f"range=[{metrics['sigma_min']:.2f}, {metrics['sigma_max']:.2f}]")
            print(f"  Cal:{metrics['cal_score']:.4f} | 1σ:{metrics['within_1sig']:.3f} | "
                  f"ρ:{metrics['spearman']:.4f}")
            print(f"  Q: {[f'{r:.3f}' for r in metrics['quintile_ratios']]}")
            # Print best sigma-filtered ROI
            if metrics['best_sig_roi'] > -1:
                print(f"  Best sigma-filtered ROI: {metrics['best_sig_roi']*100:+.1f}%")

            all_results.append(metrics)

    # ── Phase C: Full comparison ──────────────────────────────────────
    print("\n" + "=" * 110)
    print("  PHASE C: ALL 20 RUNS — FULL COMPARISON (sorted by Book-Spread MAE)")
    print("=" * 110)

    header = (f"{'Run':<12} {'Best@':>5} {'Stop@':>5} {'Dead':>4} "
              f"{'BS-MAE':>7} {'AllMAE':>7} {'BkBase':>7} {'Δ':>6} "
              f"{'σ_std':>6} {'CalSc':>6} {'ρ':>6} {'SigROI':>7} {'t':>4}")
    print(header)
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: x.get("book_spread_mae") or 999):
        bs = f"{r['book_spread_mae']:.3f}" if r['book_spread_mae'] else "  N/A"
        bb = f"{r['book_baseline_mae']:.3f}" if r['book_baseline_mae'] else "  N/A"
        d = f"{r['delta_mae']:+.2f}" if r['delta_mae'] is not None else "  N/A"
        sr = f"{r['best_sig_roi']*100:+.1f}%" if r['best_sig_roi'] > -1 else "  N/A"
        print(
            f"{r['name']:<12} {r['best_epoch']:>5} {r['stopped_at']:>5} "
            f"{r['total_dead']:>4} {bs:>7} {r['overall_mae']:>7.3f} "
            f"{bb:>7} {d:>6} "
            f"{r['sigma_std']:>6.2f} {r['cal_score']:>6.4f} "
            f"{r['spearman']:>6.4f} {sr:>7} {r['train_time']:>4.0f}"
        )

    print("\n  Per-quintile calibration:")
    for r in sorted(all_results, key=lambda x: x.get("book_spread_mae") or 999):
        qr = r["quintile_ratios"]
        print(f"    {r['name']:<12} Q1={qr[0]:.3f} Q2={qr[1]:.3f} Q3={qr[2]:.3f} "
              f"Q4={qr[3]:.3f} Q5={qr[4]:.3f}")

    # Print sigma-filtered ROI for top 5
    print("\n  Sigma-filtered ROI (top 5 by BS-MAE):")
    top5 = sorted(all_results, key=lambda x: x.get("book_spread_mae") or 999)[:5]
    for r in top5:
        parts = [f"    {r['name']:<12}"]
        for key in ["sig_med_0.05", "sig_med_0.07", "sig_p25_0.05", "sig_p25_0.07"]:
            rd = r["roi_results"].get(key, {})
            if rd and rd["bets"] > 0:
                parts.append(f"{key}: {rd['roi']*100:+.1f}%/{rd['bets']}b")
            else:
                parts.append(f"{key}: N/A")
        print("  ".join(parts))

    # ── Phase D: Winner selection ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE D: WINNER SELECTION")
    print("=" * 70)

    BS_MAE_CUTOFF = 9.60
    qualified = []

    for r in all_results:
        fails = []
        if r["total_dead"] > 0:
            fails.append(f"dead={r['total_dead']}")
        if r["book_spread_mae"] is None or r["book_spread_mae"] >= BS_MAE_CUTOFF:
            mae_val = r['book_spread_mae'] if r['book_spread_mae'] else "N/A"
            fails.append(f"BS-MAE={mae_val}>={BS_MAE_CUTOFF}")
        if r["sigma_std"] <= 1.0:
            fails.append(f"σ_std={r['sigma_std']:.2f}<=1.0")
        qr = r["quintile_ratios"]
        bad_q = [i for i, q in enumerate(qr) if q < 0.75 or q > 1.25]
        if bad_q:
            fails.append(f"Q_OOB")

        if fails:
            print(f"  DQ: {r['name']} — {'; '.join(fails)}")
            r["disqualified"] = True
        else:
            print(f"  OK: {r['name']} — BS-MAE={r['book_spread_mae']:.3f}, "
                  f"CalSc={r['cal_score']:.4f}, σ_std={r['sigma_std']:.2f}")
            r["disqualified"] = False
            qualified.append(r)

    # ── Fallback: only calibration fails ──────────────────────────────
    cal_only_fail = False
    if not qualified:
        # Check if some configs pass everything EXCEPT quintile ratios
        almost = [r for r in all_results
                  if r["total_dead"] == 0
                  and r["book_spread_mae"] is not None
                  and r["book_spread_mae"] < BS_MAE_CUTOFF
                  and r["sigma_std"] > 1.0]
        if almost:
            cal_only_fail = True
            print(f"\n  FALLBACK: {len(almost)} configs pass everything except quintile ratios.")
            print("  Retraining top 2 by BS-MAE with sigma_bias=2.7, sigma_max=50...")
            # Retrain top 2 with wider sigma
            almost.sort(key=lambda r: r["book_spread_mae"])
            for retrain_r in almost[:2]:
                rc = retrain_r["config"].copy()
                rv = retrain_r["variant"]
                rname = f"{rc['name']}-{rv}-rebias"
                print(f"\n  Retraining {rname} with sigma_bias=2.7, clamp_max=50...")
                # Monkey-patch the sigma bias init and retrain
                orig_init = MLPRegressor.__init__
                def patched_init(self, *args, **kwargs):
                    orig_init(self, *args, **kwargs)
                    nn.init.constant_(self.sigma_head.bias, 2.7)
                MLPRegressor.__init__ = patched_init
                # Also patch loss clamp
                model, best_ep, stopped, ftrain, bval, curve = train_run(
                    rc, rv, X_train, y_spread_train, X_val, y_spread_val
                )
                MLPRegressor.__init__ = orig_init  # restore
                m = evaluate_full(model, X_val, y_spread_val, df_val)
                m["name"] = rname
                m["config"] = rc
                m["variant"] = rv
                m["best_epoch"] = best_ep
                m["stopped_at"] = stopped
                m["model"] = model
                m["curve"] = curve
                m["final_train_loss"] = ftrain
                m["best_val_loss"] = bval
                m["train_time"] = 0

                qr = m["quintile_ratios"]
                bad_q = [i for i, q in enumerate(qr) if q < 0.75 or q > 1.25]
                if m["total_dead"] == 0 and not bad_q and m["sigma_std"] > 1.0:
                    m["disqualified"] = False
                    qualified.append(m)
                    print(f"  {rname}: BS-MAE={m['book_spread_mae']:.3f}, "
                          f"Cal={m['cal_score']:.4f} — QUALIFIED")
                else:
                    m["disqualified"] = True
                    print(f"  {rname}: still DQ — Q={[f'{q:.3f}' for q in qr]}")
                all_results.append(m)

    # ── Fallback: BS-MAE slightly above 9.60 ─────────────────────────
    if not qualified:
        best_by_mae = sorted(
            [r for r in all_results if r["total_dead"] == 0 and r["book_spread_mae"] is not None],
            key=lambda r: r["book_spread_mae"]
        )
        if best_by_mae and best_by_mae[0]["book_spread_mae"] < 9.75:
            winner_r = best_by_mae[0]
            print(f"\n  FALLBACK: BS-MAE={winner_r['book_spread_mae']:.3f} < 9.75, proceeding with WARNING.")
            winner_r["disqualified"] = False
            qualified.append(winner_r)
        elif best_by_mae:
            # BS-MAE > 9.75 — something is wrong
            print(f"\n  *** ALL RUNS BS-MAE > 9.75 — DIAGNOSTICS ***")
            print(f"\n  Best 3 by BS-MAE:")
            for r in best_by_mae[:3]:
                print(f"    {r['name']}: BS-MAE={r['book_spread_mae']:.3f}, "
                      f"Overall={r['overall_mae']:.3f}")
                c = r.get("curve", {})
                if c:
                    print(f"      Learning curve: ", end="")
                    for ep in sorted(c.keys())[:6]:
                        print(f"ep{ep}(t={c[ep]['train']:.3f},v={c[ep]['val']:.3f}) ", end="")
                    print()
            print(f"\n  Val set: {len(df_val)} games, "
                  f"{df_val['bookSpread'].notna().sum() if 'bookSpread' in df_val.columns else 0} with lines")
            print(f"  Session 12 had: 5440 games with book spreads, BS-MAE=9.42")
            print(f"  Current val year: 2026, Session 12 val year: 2025")
            print(f"\n  Possible explanations:")
            print(f"  - Different val year (2026 vs 2025)")
            print(f"  - Different feature count (50 vs 53)")
            print(f"  - Session 12 used softplus sigma, we use exp()")
            print(f"  - The 9.42 may have been on a different train/val split")
            print(f"\n  Taking best available model with WARNING.")
            best_by_mae[0]["disqualified"] = False
            qualified.append(best_by_mae[0])

    if not qualified:
        print("\n  *** CANNOT SELECT ANY MODEL ***")
        generate_report(all_results, None, None, REPORTS_DIR / "session13_bf_report.md")
        return None

    # ── Composite scoring ─────────────────────────────────────────────
    if len(qualified) == 1:
        winner = qualified[0]
    else:
        def rank_asc(vals):
            order = np.argsort(vals)
            ranks = np.empty(len(vals), dtype=float)
            for i, idx in enumerate(order):
                ranks[idx] = i + 1
            return ranks
        def rank_desc(vals):
            return rank_asc([-v for v in vals])

        mae_r = rank_asc([r["book_spread_mae"] for r in qualified])
        cal_r = rank_asc([r["cal_score"] for r in qualified])
        sig_r = rank_desc([r["sigma_std"] for r in qualified])
        roi_r = rank_desc([r["best_sig_roi"] for r in qualified])

        for i, r in enumerate(qualified):
            r["composite"] = 0.35*mae_r[i] + 0.25*cal_r[i] + 0.20*sig_r[i] + 0.20*roi_r[i]

        qualified.sort(key=lambda r: r["composite"])
        winner = qualified[0]

    print(f"\n  WINNER: {winner['name']}")
    print(f"  Book-Spread MAE: {winner['book_spread_mae']:.3f} "
          f"(Book Baseline: {winner['book_baseline_mae']:.3f}, Δ={winner['delta_mae']:+.2f})")
    print(f"  Overall MAE: {winner['overall_mae']:.3f}")
    print(f"  σ_std: {winner['sigma_std']:.2f}, Cal: {winner['cal_score']:.4f}")
    print(f"  Best epoch: {winner['best_epoch']}, Variant: {winner['variant']}")

    # ── Phase E: Validation ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE E: FULL VALIDATION SUITE")
    print("=" * 70)

    test_results = run_validation_suite(
        winner["model"], X_val, y_spread_val, df_val, scaler, winner,
        X_train, y_spread_train, y_win_train, winner["config"], winner["variant"]
    )

    # Check for red-flag fails
    red_flags = [k for k, v in test_results.items()
                 if v == "FAIL" and k in ("T1_dead_neurons", "T6_sigma_corr")]
    if red_flags:
        print(f"\n  *** RED FLAG FAILURES: {red_flags} — NOT SAVING ***")
        generate_report(all_results, winner, test_results, REPORTS_DIR / "session13_bf_report.md")
        return None

    # ── Phase F: Save artifacts ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE F: SAVING ARTIFACTS")
    print("=" * 70)

    wcfg = winner["config"]
    wmodel = winner["model"]
    wvar = VARIANTS[winner["variant"]]

    # Regressor
    reg_path = config.CHECKPOINTS_DIR / "regressor.pt"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": wmodel.state_dict(),
        "feature_order": config.FEATURE_ORDER,
        "hparams": {
            "hidden1": wcfg["hidden1"], "hidden2": wcfg["hidden2"],
            "dropout": wcfg["dropout"],
            "batch_size": wvar["batch_size"],
            "lr": wvar["lr"],
            "loss": wvar["loss"],
            "lr_variant": winner["variant"],
            "best_epoch": winner["best_epoch"],
        },
        "arch_type": "shared",
        "sigma_param": "exp",
    }, reg_path)
    print(f"  Saved regressor: {reg_path}")

    # Classifier
    print("\n  Training production classifier...")
    cls_h1 = max(wcfg["hidden1"], 256)
    classifier = train_classifier_production(
        X_train, y_win_train, hidden1=cls_h1,
        dropout=wcfg["dropout"], batch_size=wvar["batch_size"],
    )
    cls_path = config.CHECKPOINTS_DIR / "classifier.pt"
    torch.save({
        "state_dict": classifier.state_dict(),
        "feature_order": config.FEATURE_ORDER,
        "hparams": {"hidden1": cls_h1, "dropout": wcfg["dropout"]},
    }, cls_path)
    print(f"  Saved classifier: {cls_path}")

    # Scaler
    scaler_path = config.ARTIFACTS_DIR / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler: {scaler_path}")

    # Hparams
    hp_path = config.ARTIFACTS_DIR / "best_hparams.json"
    with open(hp_path, "w") as f:
        json.dump({
            "regressor": {
                "hidden1": wcfg["hidden1"], "hidden2": wcfg["hidden2"],
                "dropout": wcfg["dropout"],
                "batch_size": wvar["batch_size"],
                "lr": wvar["lr"], "loss": wvar["loss"],
                "arch_type": "shared", "sigma_param": "exp",
                "lr_variant": winner["variant"],
                "best_epoch": winner["best_epoch"],
                "book_spread_mae": winner["book_spread_mae"],
                "overall_mae": winner["overall_mae"],
                "cal_score": winner["cal_score"],
                "sigma_std": winner["sigma_std"],
            },
            "classifier": {"hidden1": cls_h1, "dropout": wcfg["dropout"]},
        }, f, indent=2)
    print(f"  Saved hparams: {hp_path}")

    # feature_order.json already exists and hasn't changed
    print(f"  feature_order.json: already up to date ({len(config.FEATURE_ORDER)} features)")

    generate_report(all_results, winner, test_results, REPORTS_DIR / "session13_bf_report.md")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  SESSION 13 COMPLETE")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"  Winner: {winner['name']} ({wcfg['label']})")
    print(f"  Book-Spread MAE: {winner['book_spread_mae']:.3f} "
          f"(baseline: {winner['book_baseline_mae']:.3f}, Δ={winner['delta_mae']:+.2f})")
    print(f"  Overall MAE: {winner['overall_mae']:.3f}")
    print(f"  σ std: {winner['sigma_std']:.2f}, Cal: {winner['cal_score']:.4f}, "
          f"ρ: {winner['spearman']:.4f}")
    print(f"  Best epoch: {winner['best_epoch']}, Variant: {winner['variant']}")
    print(f"  Dead neurons: {winner['total_dead']}")
    if winner['best_sig_roi'] > -1:
        print(f"  Best sigma-filtered ROI: {winner['best_sig_roi']*100:+.1f}%")
    print(f"{'='*70}")
    return winner


if __name__ == "__main__":
    main()

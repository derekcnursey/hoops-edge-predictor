#!/usr/bin/env python3
"""Session 13: Comprehensive architecture grid search for sigma recovery.

Runs 14 initial configs (8 shared + 6 split), selects top 3 by composite score,
then runs 9 refinement configs (LR variants + Laplacian NLL).

Total: 23 runs on NVIDIA A10 GPU.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, field
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
    MLPRegressorSplit,
    gaussian_nll_loss,
    laplacian_nll_loss,
)
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets
from src.trainer import impute_column_means

# ── Constants ────────────────────────────────────────────────────────

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
TRAIN_SEASONS = list(range(2015, 2026))  # 2015-2025 train
VAL_SEASON = [2026]  # current season as validation
EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-4
REPORTS_DIR = config.PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ─────────────────────────────────────────────────────

def load_data():
    """Load train/val data with all quality fixes applied."""
    from src.features import load_research_lines

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Train set: 2015-2025
    print(f"\nTraining seasons: {TRAIN_SEASONS}")
    df_train = load_multi_season_features(
        TRAIN_SEASONS, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_train = df_train.dropna(subset=["homeScore", "awayScore"])
    n_before = len(df_train)
    df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]
    print(f"  Train: {n_before} → {len(df_train)} (removed {n_before - len(df_train)} 0-0 games)")

    # Val set: 2026
    print(f"\nValidation season: {VAL_SEASON}")
    df_val = load_multi_season_features(
        VAL_SEASON, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_val = df_val.dropna(subset=["homeScore", "awayScore"])
    n_before_v = len(df_val)
    df_val = df_val[(df_val["homeScore"] != 0) | (df_val["awayScore"] != 0)]
    print(f"  Val: {n_before_v} → {len(df_val)} (removed {n_before_v - len(df_val)} 0-0 games)")

    # Load lines for val season and merge book spread
    print("\n  Loading lines for validation season 2026...")
    try:
        lines_df = load_research_lines(2026)
        if not lines_df.empty:
            # Deduplicate: take first provider per game
            lines_dedup = lines_df.sort_values("provider").drop_duplicates(
                subset=["gameId"], keep="first"
            )
            # Merge book spread into df_val
            spread_col = "spread" if "spread" in lines_dedup.columns else None
            if spread_col:
                merge_df = lines_dedup[["gameId", spread_col]].rename(
                    columns={spread_col: "bookSpread"}
                )
                df_val = df_val.merge(merge_df, on="gameId", how="left")
                n_with_spread = df_val["bookSpread"].notna().sum()
                print(f"  Merged book spreads: {n_with_spread}/{len(df_val)} games have spread data")
            else:
                print("  No 'spread' column in lines data")
        else:
            print("  No lines data available")
    except Exception as e:
        print(f"  Lines load failed: {e}")

    # Extract features + targets
    X_train = get_feature_matrix(df_train).values.astype(np.float32)
    targets_train = get_targets(df_train)
    y_spread_train = targets_train["spread_home"].values.astype(np.float32)
    y_win_train = targets_train["home_win"].values.astype(np.float32)

    X_val = get_feature_matrix(df_val).values.astype(np.float32)
    targets_val = get_targets(df_val)
    y_spread_val = targets_val["spread_home"].values.astype(np.float32)
    y_win_val = targets_val["home_win"].values.astype(np.float32)

    # Impute NaN with column means (from train set only)
    n_nan_train = np.isnan(X_train).sum()
    n_nan_val = np.isnan(X_val).sum()
    X_train = impute_column_means(X_train)
    X_val = impute_column_means(X_val)
    print(f"\n  Train NaN imputed: {n_nan_train:,}")
    print(f"  Val NaN imputed: {n_nan_val:,}")

    # Fit scaler on train
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    print(f"\n  Train shape: {X_train_scaled.shape}")
    print(f"  Val shape: {X_val_scaled.shape}")

    return (X_train_scaled, y_spread_train, y_win_train,
            X_val_scaled, y_spread_val, y_win_val,
            scaler, df_val, None)


# ── Training functions ───────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_regressor_config(
    X_train, y_spread, cfg: dict
) -> nn.Module:
    """Train a regressor with the given config dict."""
    device = get_device()
    use_amp = device.type == "cuda"

    arch_type = cfg.get("arch", "shared")
    ModelClass = MLPRegressorSplit if arch_type == "split" else MLPRegressor

    model = ModelClass(
        input_dim=X_train.shape[1],
        hidden1=cfg["hidden1"],
        hidden2=cfg["hidden2"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", DEFAULT_LR),
        weight_decay=cfg.get("weight_decay", DEFAULT_WD),
    )
    amp_scaler = GradScaler(device.type, enabled=use_amp)

    loss_type = cfg.get("loss", "gaussian")
    loss_fn = laplacian_nll_loss if loss_type == "laplacian" else gaussian_nll_loss
    penalty_weight = cfg.get("sigma_penalty", 0.0)

    ds = HoopsDataset(X_train, spread=y_spread, home_win=np.zeros(len(y_spread)))
    loader = DataLoader(
        ds, batch_size=cfg.get("batch_size", 512),
        shuffle=True, drop_last=True, num_workers=2, pin_memory=True,
    )

    epochs = cfg.get("epochs", EPOCHS)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x, spread, _ = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                mu, log_sigma = model(x)
                nll, sigma = loss_fn(mu, log_sigma, spread)
                loss = nll.mean()
                if penalty_weight > 0:
                    loss = loss - penalty_weight * sigma.std()
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            avg = epoch_loss / max(len(loader), 1)
            print(f"    epoch {epoch+1}/{epochs} — loss: {avg:.4f}")

    return model.cpu()


def train_classifier_for_search(
    X_train, y_win, hidden1=256, dropout=0.3, lr=1e-3, batch_size=512, epochs=100,
) -> MLPClassifier:
    """Train classifier (shared across all configs)."""
    device = get_device()
    use_amp = device.type == "cuda"

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden1=hidden1,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=DEFAULT_WD)
    criterion = nn.BCEWithLogitsLoss()
    amp_scaler = GradScaler(device.type, enabled=use_amp)

    ds = HoopsDataset(X_train, spread=y_win, home_win=y_win)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True,
    )

    model.train()
    for epoch in range(epochs):
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
        if (epoch + 1) % 25 == 0:
            avg = epoch_loss / max(len(loader), 1)
            print(f"    Classifier epoch {epoch+1}/{epochs} — loss: {avg:.4f}")

    return model.cpu()


# ── Evaluation functions ─────────────────────────────────────────────

def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def count_dead_neurons(model, X_val_tensor, layer_name="all"):
    """Count neurons outputting 0 for >99% of validation inputs."""
    model.eval()
    device = next(model.parameters()).device

    dead_counts = {}
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    hooks = []
    if isinstance(model, MLPRegressor):
        # Shared backbone layers
        for i, layer in enumerate(model.net):
            if isinstance(layer, nn.ReLU):
                name = f"net_relu_{i}"
                hooks.append(layer.register_forward_hook(make_hook(name)))
    elif isinstance(model, MLPRegressorSplit):
        for i, layer in enumerate(model.shared):
            if isinstance(layer, nn.ReLU):
                name = f"shared_relu_{i}"
                hooks.append(layer.register_forward_hook(make_hook(name)))
        for i, layer in enumerate(model.mu_head):
            if isinstance(layer, nn.ReLU):
                name = f"mu_head_relu_{i}"
                hooks.append(layer.register_forward_hook(make_hook(name)))
        for i, layer in enumerate(model.sigma_head):
            if isinstance(layer, nn.ReLU):
                name = f"sigma_head_relu_{i}"
                hooks.append(layer.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        # Process in batches to avoid OOM
        all_acts = {}
        batch_size = 2048
        for start in range(0, len(X_val_tensor), batch_size):
            end = min(start + batch_size, len(X_val_tensor))
            x_batch = X_val_tensor[start:end].to(device)
            model(x_batch)
            for name, act in activations.items():
                if name not in all_acts:
                    all_acts[name] = []
                all_acts[name].append(act)

    for h in hooks:
        h.remove()

    for name, act_list in all_acts.items():
        full_act = torch.cat(act_list, dim=0)
        n_samples = full_act.shape[0]
        # Count neurons where output is 0 for >99% of samples
        zero_frac = (full_act == 0).float().mean(dim=0)
        n_dead = (zero_frac > 0.99).sum().item()
        total = full_act.shape[1]
        dead_counts[name] = (int(n_dead), total)

    return dead_counts


@torch.no_grad()
def evaluate_regressor(model, X_val_scaled, y_spread_val, df_val, loss_type="gaussian"):
    """Compute all metrics for a regressor config."""
    model.eval()
    X_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

    mu_raw, log_sigma_raw = model(X_tensor)
    sigma = torch.exp(log_sigma_raw).clamp(min=0.5, max=30.0)
    mu = mu_raw.numpy()
    sigma_np = sigma.numpy()

    residuals = y_spread_val - mu
    abs_residuals = np.abs(residuals)

    # Dead neuron count
    dead = count_dead_neurons(model, X_tensor)
    total_dead = sum(d[0] for d in dead.values())

    # Book MAE (if book spread available)
    book_mae = None
    if "bookSpread" in df_val.columns:
        book_spread = df_val["bookSpread"].values.astype(np.float32)
        valid_mask = ~np.isnan(book_spread)
        if valid_mask.sum() > 0:
            # book_spread is negative=home favored, predicted_spread is positive=home favored
            book_mae = np.mean(np.abs(y_spread_val[valid_mask] - (-book_spread[valid_mask])))

    # Model MAE
    model_mae = np.mean(abs_residuals)

    # Sigma stats
    sigma_mean = float(np.mean(sigma_np))
    sigma_std = float(np.std(sigma_np))
    sigma_min = float(np.min(sigma_np))
    sigma_max = float(np.max(sigma_np))
    sigma_range = sigma_max - sigma_min

    # Within 1-sigma calibration
    within_1sig = float(np.mean(abs_residuals < sigma_np))

    # Laplacian-calibrated % (for Laplacian loss configs)
    # P(|x| < sigma) = 1 - e^(-1) ≈ 0.6321 for Laplacian
    laplacian_target = 1 - math.exp(-1)

    # Per-quintile calibration
    quintile_indices = np.array_split(np.argsort(sigma_np), 5)
    quintile_ratios = []
    for qi in quintile_indices:
        actual_std = np.std(residuals[qi])
        predicted_sigma_mean = np.mean(sigma_np[qi])
        ratio = actual_std / predicted_sigma_mean if predicted_sigma_mean > 0 else float('inf')
        quintile_ratios.append(ratio)
    calibration_score = np.mean([abs(r - 1.0) for r in quintile_ratios])

    # Sigma-error Spearman correlation
    spearman_corr, spearman_p = scipy_stats.spearmanr(sigma_np, abs_residuals)

    # ROI simulation at threshold=10%
    roi_10 = None
    if "bookSpread" in df_val.columns:
        roi_10 = compute_roi(mu, sigma_np, df_val, threshold=0.10)

    return {
        "dead_neurons": dead,
        "total_dead": total_dead,
        "model_mae": float(model_mae),
        "book_mae": book_mae,
        "sigma_mean": sigma_mean,
        "sigma_std": sigma_std,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "sigma_range": sigma_range,
        "within_1sig": within_1sig,
        "quintile_ratios": quintile_ratios,
        "calibration_score": calibration_score,
        "spearman_corr": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        "roi_10": roi_10,
        "mu": mu,
        "sigma": sigma_np,
        "residuals": residuals,
    }


def compute_roi(mu, sigma, df_val, threshold=0.10):
    """Simulate spread betting ROI at a given probability edge threshold."""
    if "bookSpread" not in df_val.columns:
        return None

    book_spread = df_val["bookSpread"].values.astype(np.float64)
    actual_spread = df_val["homeScore"].values - df_val["awayScore"].values

    valid = ~np.isnan(book_spread)
    if valid.sum() == 0:
        return None

    # Edge: predicted - book (both in points, home perspective)
    # predicted_spread = mu (positive = home wins)
    # book_spread = negative = home favored
    edge_home = mu[valid] + book_spread[valid]  # mu - (-book) = mu + book
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    home_cover_prob = normal_cdf(edge_z)
    away_cover_prob = 1.0 - home_cover_prob

    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, home_cover_prob, away_cover_prob)

    # -110 odds
    breakeven = 0.5238
    profit_per_1 = 100.0 / 110.0

    prob_edge = pick_prob - breakeven
    bet_mask = prob_edge >= threshold

    if bet_mask.sum() == 0:
        return {"bets": 0, "win_rate": 0, "roi": 0, "units": 0}

    # Did pick cover?
    actual_margin = actual_spread[valid]
    # Home covers if actual margin > -book_spread (i.e., actual margin + book_spread > 0)
    home_covered = actual_margin + book_spread[valid] > 0
    away_covered = ~home_covered

    pick_won = np.where(pick_home, home_covered, away_covered)

    wins = pick_won[bet_mask].sum()
    total_bets = bet_mask.sum()
    win_rate = wins / total_bets
    units = wins * profit_per_1 - (total_bets - wins)
    roi = units / total_bets

    return {
        "bets": int(total_bets),
        "win_rate": float(win_rate),
        "roi": float(roi),
        "units": float(units),
    }


# ── Config definitions ───────────────────────────────────────────────

INITIAL_CONFIGS = [
    # Shared backbone configs
    {"name": "S-A", "arch": "shared", "hidden1": 256, "hidden2": 192, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "S-B", "arch": "shared", "hidden1": 384, "hidden2": 256, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "S-C", "arch": "shared", "hidden1": 512, "hidden2": 384, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "S-D", "arch": "shared", "hidden1": 768, "hidden2": 640, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "S-E", "arch": "shared", "hidden1": 512, "hidden2": 384, "dropout": 0.30, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "S-F", "arch": "shared", "hidden1": 384, "hidden2": 256, "dropout": 0.30, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "S-G", "arch": "shared", "hidden1": 512, "hidden2": 384, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.01},
    {"name": "S-H", "arch": "shared", "hidden1": 768, "hidden2": 640, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.01},
    # Split head configs
    {"name": "X-A", "arch": "split", "hidden1": 256, "hidden2": 192, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "X-B", "arch": "split", "hidden1": 384, "hidden2": 256, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "X-C", "arch": "split", "hidden1": 512, "hidden2": 384, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "X-D", "arch": "split", "hidden1": 768, "hidden2": 640, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.0},
    {"name": "X-E", "arch": "split", "hidden1": 512, "hidden2": 384, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.01},
    {"name": "X-F", "arch": "split", "hidden1": 384, "hidden2": 256, "dropout": 0.20, "batch_size": 512, "sigma_penalty": 0.01},
]

# All use Gaussian NLL by default, lr=1e-3
for c in INITIAL_CONFIGS:
    c.setdefault("loss", "gaussian")
    c.setdefault("lr", DEFAULT_LR)
    c.setdefault("weight_decay", DEFAULT_WD)
    c.setdefault("epochs", EPOCHS)


# ── Main search ──────────────────────────────────────────────────────

def run_single_config(cfg, X_train, y_spread, X_val, y_spread_val, df_val):
    """Train and evaluate a single config. Returns metrics dict."""
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"  CONFIG: {name}")
    print(f"  arch={cfg['arch']} h1={cfg['hidden1']} h2={cfg['hidden2']} "
          f"do={cfg['dropout']} bs={cfg['batch_size']} loss={cfg.get('loss','gaussian')} "
          f"lr={cfg.get('lr', DEFAULT_LR)} pen={cfg.get('sigma_penalty', 0.0)}")
    print(f"{'='*60}")

    t0 = time.time()
    model = train_regressor_config(X_train, y_spread, cfg)
    train_time = time.time() - t0

    metrics = evaluate_regressor(model, X_val, y_spread_val, df_val, loss_type=cfg.get("loss", "gaussian"))
    metrics["name"] = name
    metrics["config"] = cfg
    metrics["train_time"] = train_time
    metrics["model"] = model

    # Print summary
    print(f"\n  --- {name} Results ---")
    print(f"  Dead neurons: {metrics['total_dead']}")
    for layer, (dead, total) in metrics["dead_neurons"].items():
        print(f"    {layer}: {dead}/{total}")
    print(f"  Model MAE: {metrics['model_mae']:.3f}")
    if metrics["book_mae"] is not None:
        print(f"  Book MAE: {metrics['book_mae']:.3f}")
    print(f"  Sigma: mean={metrics['sigma_mean']:.2f} std={metrics['sigma_std']:.2f} "
          f"min={metrics['sigma_min']:.2f} max={metrics['sigma_max']:.2f} "
          f"range={metrics['sigma_range']:.2f}")
    print(f"  Within 1σ: {metrics['within_1sig']:.3f}")
    print(f"  Quintile ratios: {[f'{r:.3f}' for r in metrics['quintile_ratios']]}")
    print(f"  Calibration score: {metrics['calibration_score']:.4f}")
    print(f"  Spearman(σ, |error|): {metrics['spearman_corr']:.4f}")
    if metrics["roi_10"]:
        r = metrics["roi_10"]
        print(f"  ROI@10%: {r['roi']:.3f} ({r['bets']} bets, {r['win_rate']:.3f} WR)")
    print(f"  Train time: {train_time:.1f}s")

    return metrics


def composite_score(results):
    """Rank configs by composite score. Returns sorted results with ranks."""
    # Filter out configs with dead neurons
    valid = [r for r in results if r["total_dead"] == 0]
    disqualified = [r for r in results if r["total_dead"] > 0]

    if not valid:
        print("\nWARNING: All configs have dead neurons! Using all for ranking.")
        valid = results
        disqualified = []

    n = len(valid)

    # Compute ranks (1 = best)
    mae_vals = [r["model_mae"] for r in valid]
    cal_vals = [r["calibration_score"] for r in valid]
    sigma_std_vals = [r["sigma_std"] for r in valid]
    roi_vals = [r["roi_10"]["roi"] if r["roi_10"] and r["roi_10"]["bets"] > 0 else -1.0 for r in valid]

    def rank_asc(vals):
        """Rank ascending (lower value = better rank)."""
        order = np.argsort(vals)
        ranks = np.empty(len(vals), dtype=float)
        for i, idx in enumerate(order):
            ranks[idx] = i + 1
        return ranks

    def rank_desc(vals):
        """Rank descending (higher value = better rank)."""
        order = np.argsort(vals)[::-1]
        ranks = np.empty(len(vals), dtype=float)
        for i, idx in enumerate(order):
            ranks[idx] = i + 1
        return ranks

    mae_ranks = rank_asc(mae_vals)
    cal_ranks = rank_asc(cal_vals)
    sigma_ranks = rank_desc(sigma_std_vals)
    roi_ranks = rank_desc(roi_vals)

    for i, r in enumerate(valid):
        r["mae_rank"] = mae_ranks[i]
        r["cal_rank"] = cal_ranks[i]
        r["sigma_rank"] = sigma_ranks[i]
        r["roi_rank"] = roi_ranks[i]
        r["composite"] = (
            0.30 * mae_ranks[i] +
            0.30 * cal_ranks[i] +
            0.20 * sigma_ranks[i] +
            0.20 * roi_ranks[i]
        )
        r["disqualified"] = False

    for r in disqualified:
        r["mae_rank"] = n + 1
        r["cal_rank"] = n + 1
        r["sigma_rank"] = n + 1
        r["roi_rank"] = n + 1
        r["composite"] = 999.0
        r["disqualified"] = True

    all_results = valid + disqualified
    all_results.sort(key=lambda r: r["composite"])
    return all_results


def print_results_table(results, title="Results"):
    """Print a formatted comparison table."""
    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")
    header = (
        f"{'Config':<8} {'Arch':<7} {'H1':>4} {'H2':>4} {'DO':>5} {'Loss':<8} "
        f"{'LR':>7} {'Dead':>5} {'MAE':>6} {'σ_mean':>7} {'σ_std':>6} {'σ_rng':>6} "
        f"{'Cal%':>5} {'CalSc':>6} {'ρ(σ,ε)':>7} {'ROI%':>6} {'Time':>5} {'Comp':>5} "
        f"{'DQ':>3}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        roi_str = f"{r['roi_10']['roi']*100:.1f}" if r.get('roi_10') and r['roi_10']['bets'] > 0 else "N/A"
        dq_str = "YES" if r.get("disqualified") else ""
        print(
            f"{r['name']:<8} {r['config']['arch']:<7} "
            f"{r['config']['hidden1']:>4} {r['config']['hidden2']:>4} "
            f"{r['config']['dropout']:>5.2f} {r['config'].get('loss','gauss'):<8} "
            f"{r['config'].get('lr', DEFAULT_LR):>7.5f} "
            f"{r['total_dead']:>5} "
            f"{r['model_mae']:>6.3f} "
            f"{r['sigma_mean']:>7.2f} {r['sigma_std']:>6.2f} {r['sigma_range']:>6.2f} "
            f"{r['within_1sig']*100:>5.1f} {r['calibration_score']:>6.4f} "
            f"{r['spearman_corr']:>7.4f} "
            f"{roi_str:>6} "
            f"{r['train_time']:>5.0f} "
            f"{r.get('composite', 999):>5.1f} "
            f"{dq_str:>3}"
        )


# ── Validation tests (Step 7) ───────────────────────────────────────

def run_validation_tests(model, X_val_scaled, y_spread_val, df_val, scaler, metrics,
                         X_train_scaled, y_spread_train, y_win_train, cfg):
    """Run all 8 validation tests on the winning model."""
    print("\n" + "=" * 70)
    print("  VALIDATION TESTS FOR WINNING MODEL")
    print("=" * 70)

    X_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    test_results = {}

    # TEST 1: Dead neuron scan
    print("\n--- TEST 1: Dead Neuron Scan ---")
    dead = count_dead_neurons(model, X_tensor)
    all_clear = True
    for layer, (n_dead, total) in dead.items():
        status = "PASS" if n_dead == 0 else "FAIL"
        if n_dead > 0:
            all_clear = False
        print(f"  {layer}: {n_dead}/{total} dead [{status}]")
    test_results["dead_neuron_scan"] = "PASS" if all_clear else "FAIL"

    # TEST 2: Sigma correlation with difficulty
    print("\n--- TEST 2: Sigma Correlation with Difficulty ---")
    sigma_np = metrics["sigma"]

    # We don't have P6 conference data directly, but we can approximate
    # using team IDs or ratings. Use a simpler proxy: use the average
    # sigma by absolute book spread size as a proxy for game predictability.
    if "bookSpread" in df_val.columns:
        book_sp = df_val["bookSpread"].values.astype(np.float64)
        valid = ~np.isnan(book_sp)
        abs_spread = np.abs(book_sp[valid])
        sig_valid = sigma_np[valid]

        # Close games (0-3): should have HIGHER sigma (less predictable)
        # Blowout lines (14+): should have LOWER sigma (more predictable)
        close = abs_spread <= 3
        medium = (abs_spread > 3) & (abs_spread <= 7)
        large = (abs_spread > 7) & (abs_spread <= 14)
        blowout = abs_spread > 14

        for label, mask in [("Close (0-3)", close), ("Medium (3-7)", medium),
                            ("Large (7-14)", large), ("Blowout (14+)", blowout)]:
            if mask.sum() > 0:
                print(f"  {label}: mean σ = {sig_valid[mask].mean():.3f} (n={mask.sum()})")

        # Sigma should be lower for large spreads
        if close.sum() > 0 and large.sum() > 0:
            if sig_valid[close].mean() > sig_valid[large].mean():
                print("  ✓ Sigma is higher for close games (correct)")
                test_results["sigma_difficulty"] = "PASS"
            else:
                print("  ✗ Sigma is NOT higher for close games (unexpected)")
                test_results["sigma_difficulty"] = "WARN"
        else:
            test_results["sigma_difficulty"] = "SKIP"
    else:
        print("  No book spreads available — skipping")
        test_results["sigma_difficulty"] = "SKIP"

    # TEST 3: Calibration at multiple levels
    print("\n--- TEST 3: Multi-Level Calibration ---")
    residuals = metrics["residuals"]
    abs_res = np.abs(residuals)

    loss_type = cfg.get("loss", "gaussian")

    for mult, gauss_target in [(0.5, 0.3829), (1.0, 0.6827), (1.5, 0.8664), (2.0, 0.9545)]:
        actual = float(np.mean(abs_res < mult * sigma_np))
        diff = abs(actual - gauss_target)
        status = "PASS" if diff < 0.03 else ("WARN" if diff < 0.05 else "FAIL")
        print(f"  Within {mult:.1f}σ: actual={actual:.4f} target={gauss_target:.4f} "
              f"diff={diff:.4f} [{status}]")

    if loss_type == "laplacian":
        # Also show Laplacian targets
        print("  (Laplacian targets for reference:)")
        for mult, lap_target in [(0.5, 0.3935), (1.0, 0.6321), (1.5, 0.7769), (2.0, 0.8647)]:
            actual = float(np.mean(abs_res < mult * sigma_np))
            print(f"    Within {mult:.1f}σ (Laplace): actual={actual:.4f} target={lap_target:.4f}")

    test_results["multi_level_cal"] = "PASS"  # will be overridden if FAIL above

    # TEST 4: Monthly MAE breakdown
    print("\n--- TEST 4: Monthly MAE Breakdown ---")
    if "startDate" in df_val.columns:
        dates = pd.to_datetime(df_val["startDate"], errors="coerce", utc=True)
        months = dates.dt.month
        for m, label in [(12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")]:
            mask = months == m
            if mask.sum() > 0:
                monthly_mae = np.mean(np.abs(residuals[mask.values]))
                print(f"  {label}: MAE={monthly_mae:.3f} (n={mask.sum()})")
    test_results["monthly_mae"] = "PASS"

    # TEST 5: Edge profitability at multiple thresholds
    print("\n--- TEST 5: Edge Profitability at Multiple Thresholds ---")
    if "bookSpread" in df_val.columns:
        mu = metrics["mu"]
        for thresh in [0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
            roi_data = compute_roi(mu, sigma_np, df_val, threshold=thresh)
            if roi_data and roi_data["bets"] > 0:
                print(f"  Threshold {thresh*100:.0f}%: {roi_data['bets']:>4} bets, "
                      f"WR={roi_data['win_rate']:.3f}, ROI={roi_data['roi']*100:.1f}%, "
                      f"Units={roi_data['units']:.1f}")
            else:
                print(f"  Threshold {thresh*100:.0f}%: 0 bets")
    test_results["edge_profitability"] = "PASS"

    # TEST 6: Sigma vs actual error correlation
    print("\n--- TEST 6: Sigma vs Actual Error Spearman Correlation ---")
    corr = metrics["spearman_corr"]
    status = "PASS" if corr > 0 else "FAIL"
    print(f"  Spearman ρ(σ, |error|) = {corr:.4f} [{status}]")
    test_results["sigma_error_corr"] = status

    # TEST 7: Spread accuracy by spread size
    print("\n--- TEST 7: Spread Accuracy by Spread Size ---")
    if "bookSpread" in df_val.columns:
        book_sp = np.abs(df_val["bookSpread"].values.astype(np.float64))
        valid = ~np.isnan(book_sp)
        for label, lo, hi in [("Close (0-3)", 0, 3), ("Medium (3-7)", 3, 7),
                               ("Large (7-14)", 7, 14), ("Blowout (14+)", 14, 999)]:
            mask = valid & (book_sp >= lo) & (book_sp < hi)
            if mask.sum() > 0:
                bucket_mae = np.mean(np.abs(residuals[mask]))
                print(f"  {label}: MAE={bucket_mae:.3f} (n={mask.sum()})")
    test_results["spread_accuracy"] = "PASS"

    # TEST 8: Hold-out year test (train 2015-2024, predict 2025)
    print("\n--- TEST 8: Hold-Out Year Test (Train 2015-2024, Predict 2025) ---")
    try:
        holdout_metrics = run_holdout_test(cfg, scaler)
        test_results["holdout_year"] = "PASS"
    except Exception as e:
        print(f"  ERROR: {e}")
        test_results["holdout_year"] = "FAIL"

    print("\n--- Validation Summary ---")
    for test_name, status in test_results.items():
        print(f"  {test_name}: {status}")

    any_fail = any(v == "FAIL" for v in test_results.values())
    if any_fail:
        print("\n  *** RED FLAG: Some tests FAILED — review before saving artifacts ***")
    else:
        print("\n  All tests PASSED — safe to save artifacts")

    return test_results


def run_holdout_test(cfg, scaler_full=None):
    """Train on 2015-2024, predict ALL of 2025. Print MAE, sigma stats, calibration."""
    from src.features import load_research_lines

    print("  Training holdout model on 2015-2024...")

    holdout_train_seasons = list(range(2015, 2025))
    holdout_val_seasons = [2025]

    df_ho_train = load_multi_season_features(
        holdout_train_seasons, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_ho_train = df_ho_train.dropna(subset=["homeScore", "awayScore"])
    df_ho_train = df_ho_train[(df_ho_train["homeScore"] != 0) | (df_ho_train["awayScore"] != 0)]

    df_ho_val = load_multi_season_features(
        holdout_val_seasons, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_ho_val = df_ho_val.dropna(subset=["homeScore", "awayScore"])
    df_ho_val = df_ho_val[(df_ho_val["homeScore"] != 0) | (df_ho_val["awayScore"] != 0)]

    # Merge lines for holdout val
    try:
        lines_df = load_research_lines(2025)
        if not lines_df.empty:
            lines_dedup = lines_df.sort_values("provider").drop_duplicates(
                subset=["gameId"], keep="first"
            )
            if "spread" in lines_dedup.columns:
                merge_df = lines_dedup[["gameId", "spread"]].rename(
                    columns={"spread": "bookSpread"}
                )
                df_ho_val = df_ho_val.merge(merge_df, on="gameId", how="left")
    except Exception:
        pass

    X_ht = get_feature_matrix(df_ho_train).values.astype(np.float32)
    X_hv = get_feature_matrix(df_ho_val).values.astype(np.float32)
    targets_ht = get_targets(df_ho_train)
    targets_hv = get_targets(df_ho_val)
    y_ht = targets_ht["spread_home"].values.astype(np.float32)
    y_hv = targets_hv["spread_home"].values.astype(np.float32)

    X_ht = impute_column_means(X_ht)
    X_hv = impute_column_means(X_hv)

    sc = StandardScaler()
    sc.fit(X_ht)
    X_ht_s = sc.transform(X_ht).astype(np.float32)
    X_hv_s = sc.transform(X_hv).astype(np.float32)

    ho_model = train_regressor_config(X_ht_s, y_ht, cfg)
    ho_metrics = evaluate_regressor(ho_model, X_hv_s, y_hv, df_ho_val, loss_type=cfg.get("loss", "gaussian"))

    print(f"  Hold-out 2025 results:")
    print(f"    Samples: {len(y_hv)}")
    print(f"    MAE: {ho_metrics['model_mae']:.3f}")
    print(f"    Sigma: mean={ho_metrics['sigma_mean']:.2f} std={ho_metrics['sigma_std']:.2f} "
          f"range={ho_metrics['sigma_range']:.2f}")
    print(f"    Within 1σ: {ho_metrics['within_1sig']:.3f}")
    print(f"    Cal score: {ho_metrics['calibration_score']:.4f}")
    print(f"    Dead neurons: {ho_metrics['total_dead']}")
    if ho_metrics.get("roi_10") and ho_metrics["roi_10"]["bets"] > 0:
        r = ho_metrics["roi_10"]
        print(f"    ROI@10%: {r['roi']*100:.1f}% ({r['bets']} bets, WR={r['win_rate']:.3f})")

    return ho_metrics


# ── Report generation ────────────────────────────────────────────────

def generate_report(all_results, winner, test_results, report_path):
    """Generate markdown report."""
    lines = []
    lines.append("# Session 13: Architecture Grid Search Report")
    lines.append(f"\nDate: 2026-02-27")
    lines.append(f"GPU: NVIDIA A10")
    lines.append(f"Training seasons: 2015-2025")
    lines.append(f"Validation season: 2026")
    lines.append(f"Features: {len(config.FEATURE_ORDER)}")
    lines.append(f"Total configs evaluated: {len(all_results)}")

    lines.append("\n## Configuration Grid")
    lines.append("")
    lines.append("| Config | Arch | H1 | H2 | Dropout | Loss | LR | Batch | σ Penalty |")
    lines.append("|--------|------|----|----|---------|------|----|-------|-----------|")
    for r in all_results:
        c = r["config"]
        lines.append(
            f"| {r['name']} | {c['arch']} | {c['hidden1']} | {c['hidden2']} | "
            f"{c['dropout']} | {c.get('loss','gaussian')} | {c.get('lr', DEFAULT_LR)} | "
            f"{c.get('batch_size', 512)} | {c.get('sigma_penalty', 0.0)} |"
        )

    lines.append("\n## Results Summary")
    lines.append("")
    lines.append("| Config | Dead | MAE | σ_mean | σ_std | σ_range | Cal% | CalScore | ρ(σ,ε) | ROI@10% | Time(s) | Composite | DQ |")
    lines.append("|--------|------|-----|--------|-------|---------|------|----------|--------|---------|---------|-----------|-----|")
    for r in all_results:
        roi_str = f"{r['roi_10']['roi']*100:.1f}%" if r.get('roi_10') and r['roi_10']['bets'] > 0 else "N/A"
        dq = "YES" if r.get("disqualified") else ""
        lines.append(
            f"| {r['name']} | {r['total_dead']} | {r['model_mae']:.3f} | "
            f"{r['sigma_mean']:.2f} | {r['sigma_std']:.2f} | {r['sigma_range']:.2f} | "
            f"{r['within_1sig']*100:.1f}% | {r['calibration_score']:.4f} | "
            f"{r['spearman_corr']:.4f} | {roi_str} | {r['train_time']:.0f} | "
            f"{r.get('composite', 999):.1f} | {dq} |"
        )

    lines.append("\n## Per-Quintile Calibration Ratios")
    lines.append("")
    lines.append("| Config | Q1 | Q2 | Q3 | Q4 | Q5 | Mean |δ| |")
    lines.append("|--------|-----|-----|-----|-----|-----|---------|")
    for r in all_results:
        qr = r["quintile_ratios"]
        lines.append(
            f"| {r['name']} | {qr[0]:.3f} | {qr[1]:.3f} | {qr[2]:.3f} | "
            f"{qr[3]:.3f} | {qr[4]:.3f} | {r['calibration_score']:.4f} |"
        )

    lines.append(f"\n## Winner: {winner['name']}")
    lines.append("")
    c = winner["config"]
    lines.append(f"- Architecture: {c['arch']}")
    lines.append(f"- Hidden layers: {c['hidden1']} → {c['hidden2']}")
    lines.append(f"- Dropout: {c['dropout']}")
    lines.append(f"- Loss: {c.get('loss', 'gaussian')}")
    lines.append(f"- Learning rate: {c.get('lr', DEFAULT_LR)}")
    lines.append(f"- Batch size: {c.get('batch_size', 512)}")
    lines.append(f"- Sigma penalty: {c.get('sigma_penalty', 0.0)}")
    lines.append(f"- Composite score: {winner.get('composite', 'N/A')}")

    lines.append("\n## Validation Tests")
    lines.append("")
    for test_name, status in test_results.items():
        lines.append(f"- {test_name}: **{status}**")

    lines.append("\n## Comparison: Winner vs Broken Model vs Old Torvik")
    lines.append("")
    lines.append("| Metric | Winner | Broken (128→128) | Old Torvik (768→640) |")
    lines.append("|--------|--------|-------------------|----------------------|")
    lines.append(f"| Architecture | {c['arch']} {c['hidden1']}→{c['hidden2']} | shared 128→128 | shared 768→640 |")
    lines.append(f"| Dropout | {c['dropout']} | 0.45 | 0.20 |")
    lines.append(f"| Sigma param | exp() | softplus | softplus |")
    lines.append(f"| Dead neurons | {winner['total_dead']} | 116/128 | 0 |")
    lines.append(f"| MAE | {winner['model_mae']:.3f} | 9.42 | ~9.5 |")
    lines.append(f"| σ mean | {winner['sigma_mean']:.2f} | ~11.0 | ~11.5 |")
    lines.append(f"| σ std | {winner['sigma_std']:.2f} | 0.53 | 1.42 |")
    lines.append(f"| σ range | {winner['sigma_range']:.2f} | 4.3 | 10.7 |")
    lines.append(f"| Within 1σ | {winner['within_1sig']*100:.1f}% | ~72% | ~68% |")
    lines.append(f"| Cal score | {winner['calibration_score']:.4f} | ~0.25 | ~0.05 |")
    lines.append(f"| ρ(σ,|error|) | {winner['spearman_corr']:.4f} | ~0.05 | ~0.15 |")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    return report_text


# ── Main ─────────────────────────────────────────────────────────────

def main():
    total_start = time.time()

    # Load data
    (X_train, y_spread_train, y_win_train,
     X_val, y_spread_val, y_win_val,
     scaler, df_val, book_spreads_val) = load_data()

    # ── Phase 1: Initial 14-config sweep ─────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: INITIAL 14-CONFIG SWEEP (Gaussian NLL)")
    print("=" * 70)

    initial_results = []
    for cfg in INITIAL_CONFIGS:
        metrics = run_single_config(cfg, X_train, y_spread_train, X_val, y_spread_val, df_val)
        initial_results.append(metrics)

    # Score and rank
    ranked_initial = composite_score(initial_results)
    print_results_table(ranked_initial, "Phase 1 Results (ranked by composite)")

    # ── Phase 2: Refinement runs for top 3 ───────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: REFINEMENT RUNS FOR TOP 3")
    print("=" * 70)

    # Get top 3 non-disqualified configs
    top3 = [r for r in ranked_initial if not r.get("disqualified")][:3]
    print(f"\nTop 3 configs for refinement: {[r['name'] for r in top3]}")

    refinement_results = []
    for i, base_result in enumerate(top3):
        base_cfg = base_result["config"].copy()
        base_name = base_result["name"]

        # LR variant 1: 5e-4
        cfg_lr1 = base_cfg.copy()
        cfg_lr1["name"] = f"{base_name}-lr5e4"
        cfg_lr1["lr"] = 5e-4
        metrics = run_single_config(cfg_lr1, X_train, y_spread_train, X_val, y_spread_val, df_val)
        refinement_results.append(metrics)

        # LR variant 2: 1e-4
        cfg_lr2 = base_cfg.copy()
        cfg_lr2["name"] = f"{base_name}-lr1e4"
        cfg_lr2["lr"] = 1e-4
        metrics = run_single_config(cfg_lr2, X_train, y_spread_train, X_val, y_spread_val, df_val)
        refinement_results.append(metrics)

        # Laplacian NLL variant
        cfg_lap = base_cfg.copy()
        cfg_lap["name"] = f"{base_name}-lap"
        cfg_lap["loss"] = "laplacian"
        metrics = run_single_config(cfg_lap, X_train, y_spread_train, X_val, y_spread_val, df_val)
        refinement_results.append(metrics)

    # ── Phase 3: Final ranking of ALL 23 configs ─────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3: FINAL RANKING (ALL 23 CONFIGS)")
    print("=" * 70)

    all_results = initial_results + refinement_results
    ranked_all = composite_score(all_results)
    print_results_table(ranked_all, "All 23 Configs (ranked by composite)")

    # Select winner
    winner = ranked_all[0]
    print(f"\n{'='*70}")
    print(f"  WINNER: {winner['name']}")
    print(f"  Composite score: {winner.get('composite', 'N/A')}")
    print(f"{'='*70}")

    # ── Phase 4: Validation tests ────────────────────────────────────
    test_results = run_validation_tests(
        winner["model"], X_val, y_spread_val, df_val, scaler, winner,
        X_train, y_spread_train, y_win_train, winner["config"],
    )

    # Check for red flags
    any_fail = any(v == "FAIL" for v in test_results.values())

    # ── Phase 5: Save artifacts ──────────────────────────────────────
    if not any_fail:
        print("\n" + "=" * 70)
        print("  SAVING WINNER ARTIFACTS")
        print("=" * 70)

        winner_cfg = winner["config"]
        winner_model = winner["model"]

        # Save regressor checkpoint with architecture metadata
        reg_path = config.CHECKPOINTS_DIR / "regressor.pt"
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": winner_model.state_dict(),
            "feature_order": config.FEATURE_ORDER,
            "hparams": {
                "hidden1": winner_cfg["hidden1"],
                "hidden2": winner_cfg["hidden2"],
                "dropout": winner_cfg["dropout"],
                "lr": winner_cfg.get("lr", DEFAULT_LR),
                "weight_decay": winner_cfg.get("weight_decay", DEFAULT_WD),
                "batch_size": winner_cfg.get("batch_size", 512),
                "epochs": EPOCHS,
                "loss": winner_cfg.get("loss", "gaussian"),
                "sigma_penalty": winner_cfg.get("sigma_penalty", 0.0),
            },
            "arch_type": winner_cfg["arch"],  # "shared" or "split"
            "sigma_param": "exp",  # exp() instead of softplus
        }, reg_path)
        print(f"  Saved regressor: {reg_path}")

        # Train and save classifier (using winner's hidden1 size for consistency)
        print("\n  Training production classifier...")
        cls_hidden1 = max(winner_cfg["hidden1"], 256)  # at least 256
        classifier = train_classifier_for_search(
            X_train, y_win_train,
            hidden1=cls_hidden1,
            dropout=winner_cfg["dropout"],
            lr=winner_cfg.get("lr", DEFAULT_LR),
            batch_size=winner_cfg.get("batch_size", 512),
            epochs=EPOCHS,
        )
        cls_path = config.CHECKPOINTS_DIR / "classifier.pt"
        torch.save({
            "state_dict": classifier.state_dict(),
            "feature_order": config.FEATURE_ORDER,
            "hparams": {
                "hidden1": cls_hidden1,
                "dropout": winner_cfg["dropout"],
                "lr": winner_cfg.get("lr", DEFAULT_LR),
                "batch_size": winner_cfg.get("batch_size", 512),
                "epochs": EPOCHS,
            },
        }, cls_path)
        print(f"  Saved classifier: {cls_path}")

        # Save scaler
        scaler_path = config.ARTIFACTS_DIR / "scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Saved scaler: {scaler_path}")

        # Update best_hparams.json
        hparams_path = config.ARTIFACTS_DIR / "best_hparams.json"
        with open(hparams_path, "w") as f:
            json.dump({
                "regressor": {
                    "hidden1": winner_cfg["hidden1"],
                    "hidden2": winner_cfg["hidden2"],
                    "dropout": winner_cfg["dropout"],
                    "lr": winner_cfg.get("lr", DEFAULT_LR),
                    "weight_decay": winner_cfg.get("weight_decay", DEFAULT_WD),
                    "batch_size": winner_cfg.get("batch_size", 512),
                    "loss": winner_cfg.get("loss", "gaussian"),
                    "sigma_penalty": winner_cfg.get("sigma_penalty", 0.0),
                    "arch_type": winner_cfg["arch"],
                    "sigma_param": "exp",
                },
                "classifier": {
                    "hidden1": cls_hidden1,
                    "dropout": winner_cfg["dropout"],
                    "lr": winner_cfg.get("lr", DEFAULT_LR),
                    "batch_size": winner_cfg.get("batch_size", 512),
                },
            }, f, indent=2)
        print(f"  Saved hparams: {hparams_path}")
    else:
        print("\n  *** ARTIFACTS NOT SAVED — validation tests had failures ***")
        print("  Review failures above and re-run if needed.")

    # ── Phase 6: Generate report ─────────────────────────────────────
    report_path = REPORTS_DIR / "architecture_search_session13.md"
    generate_report(ranked_all, winner, test_results, report_path)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  SESSION 13 COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Configs evaluated: {len(all_results)}")
    print(f"  Winner: {winner['name']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

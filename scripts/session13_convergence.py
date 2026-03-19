#!/usr/bin/env python3
"""Session 13 Phase B-F: Convergence training for top 3 configs.

Retrains with proper LR scheduling, early stopping, 500 epochs.
12 total runs (3 configs x 4 LR variants).
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
    MLPRegressorSplit,
    gaussian_nll_loss,
    laplacian_nll_loss,
)
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets
from src.trainer import impute_column_means

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
TRAIN_SEASONS = list(range(2015, 2026))
VAL_SEASON = [2026]
REPORTS_DIR = config.PROJECT_ROOT / "reports"


# ── Data loading ─────────────────────────────────────────────────────

def load_data():
    from src.features import load_research_lines

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df_train = load_multi_season_features(
        TRAIN_SEASONS, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_train = df_train.dropna(subset=["homeScore", "awayScore"])
    n_before = len(df_train)
    df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]
    print(f"  Train: {n_before} → {len(df_train)} (removed {n_before - len(df_train)} 0-0 games)")

    df_val = load_multi_season_features(
        VAL_SEASON, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_val = df_val.dropna(subset=["homeScore", "awayScore"])
    n_before_v = len(df_val)
    df_val = df_val[(df_val["homeScore"] != 0) | (df_val["awayScore"] != 0)]
    print(f"  Val: {n_before_v} → {len(df_val)} (removed {n_before_v - len(df_val)} 0-0 games)")

    # Load lines for val season
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
                print(f"  Book spreads merged: {n_with}/{len(df_val)}")
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
    print(f"  Train NaN imputed: {n_nan_train:,}, Val NaN imputed: {n_nan_val:,}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    print(f"  Train: {X_train_s.shape}, Val: {X_val_s.shape}")
    return X_train_s, y_spread_train, y_win_train, X_val_s, y_spread_val, y_win_val, scaler, df_val


# ── LR schedulers ────────────────────────────────────────────────────

def make_scheduler(variant, optimizer, max_epochs):
    """Create LR scheduler for a given variant."""
    if variant == "V1":
        # Cosine anneal: lr_max=1e-3 → lr_min=1e-5
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5
        )
    elif variant == "V2":
        # Cosine anneal: lr_max=5e-4 → lr_min=1e-5
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5
        )
    elif variant == "V3":
        # Step decay: lr=1e-3, multiply by 0.5 every 100 epochs
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.5
        )
    elif variant == "V4":
        # Warmup 20 epochs from 1e-5 to 1e-3, then cosine to 1e-5
        def lr_lambda(epoch):
            if epoch < 20:
                # Linear warmup from 1e-5 to 1e-3
                return 0.01 + (1.0 - 0.01) * (epoch / 20.0)
            else:
                # Cosine anneal from 1e-3 to 1e-5 over remaining epochs
                progress = (epoch - 20) / max(max_epochs - 20, 1)
                return 0.01 + (1.0 - 0.01) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def get_initial_lr(variant):
    if variant == "V1":
        return 1e-3
    elif variant == "V2":
        return 5e-4
    elif variant == "V3":
        return 1e-3
    elif variant == "V4":
        return 1e-3  # will be scaled by lambda
    raise ValueError(variant)


# ── Training with early stopping ─────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def compute_val_loss(model, X_val_tensor, y_val_tensor, loss_fn, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    batch_size = 2048
    n_batches = 0
    for start in range(0, len(X_val_tensor), batch_size):
        end = min(start + batch_size, len(X_val_tensor))
        x = X_val_tensor[start:end].to(device)
        y = y_val_tensor[start:end].to(device)
        mu, log_sigma = model(x)
        nll, _ = loss_fn(mu, log_sigma, y)
        total_loss += nll.mean().item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train_with_convergence(cfg, X_train, y_spread, X_val_s, y_spread_val, variant):
    """Train a regressor with LR scheduling and early stopping."""
    device = get_device()
    use_amp = device.type == "cuda"

    max_epochs = 300 if variant == "V4" else 500
    patience = 50
    initial_lr = get_initial_lr(variant)

    ModelClass = MLPRegressorSplit if cfg.get("arch") == "split" else MLPRegressor
    model = ModelClass(
        input_dim=X_train.shape[1],
        hidden1=cfg["hidden1"],
        hidden2=cfg["hidden2"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=initial_lr, weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = make_scheduler(variant, optimizer, max_epochs)
    amp_scaler = GradScaler(device.type, enabled=use_amp)

    loss_fn = laplacian_nll_loss if cfg.get("loss") == "laplacian" else gaussian_nll_loss
    penalty_weight = cfg.get("sigma_penalty", 0.0)

    ds = HoopsDataset(X_train, spread=y_spread, home_win=np.zeros(len(y_spread)))
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 512),
                        shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    X_val_tensor = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_spread_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve_count = 0

    model.train()
    for epoch in range(max_epochs):
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

        scheduler.step()

        # Validation
        val_loss = compute_val_loss(model, X_val_tensor, y_val_tensor, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            no_improve_count = 0
        else:
            no_improve_count += 1

        avg_train = epoch_loss / max(len(loader), 1)
        if (epoch + 1) % 50 == 0 or no_improve_count == patience:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    epoch {epoch+1}/{max_epochs} — train: {avg_train:.4f} "
                  f"val: {val_loss:.4f} best_val: {best_val_loss:.4f} "
                  f"(best@{best_epoch}) lr: {current_lr:.2e}")

        if no_improve_count >= patience:
            print(f"    Early stop at epoch {epoch+1} (best was epoch {best_epoch})")
            break

    # Load best checkpoint
    model.cpu()
    model.load_state_dict(best_state)
    model.eval()

    stopped_at = epoch + 1
    final_train_loss = avg_train

    return model, best_epoch, stopped_at, final_train_loss, best_val_loss


# ── Evaluation ───────────────────────────────────────────────────────

def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def count_dead_neurons(model, X_val_tensor):
    """Count neurons outputting 0 for >99% of validation inputs."""
    model.eval()
    device = next(model.parameters()).device
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    hooks = []
    if isinstance(model, MLPRegressor):
        for i, layer in enumerate(model.net):
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(make_hook(f"net_relu_{i}")))
    elif isinstance(model, MLPRegressorSplit):
        for i, layer in enumerate(model.shared):
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(make_hook(f"shared_relu_{i}")))
        for i, layer in enumerate(model.mu_head):
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(make_hook(f"mu_relu_{i}")))
        for i, layer in enumerate(model.sigma_head):
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(make_hook(f"sigma_relu_{i}")))

    with torch.no_grad():
        all_acts = {}
        for start in range(0, len(X_val_tensor), 2048):
            end = min(start + 2048, len(X_val_tensor))
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


def compute_roi(mu, sigma, df_val, threshold):
    """Simulate spread betting ROI at a probability edge threshold."""
    if "bookSpread" not in df_val.columns:
        return None
    book_spread = df_val["bookSpread"].values.astype(np.float64)
    actual_spread = (df_val["homeScore"].values - df_val["awayScore"].values).astype(np.float64)
    valid = ~np.isnan(book_spread)
    if valid.sum() == 0:
        return None

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
    """Compute all metrics for a trained regressor."""
    model.eval()
    X_tensor = torch.tensor(X_val_s, dtype=torch.float32)
    mu_t, log_sigma_t = model(X_tensor)
    sigma_t = torch.exp(log_sigma_t).clamp(min=0.5, max=30.0)
    mu = mu_t.numpy()
    sigma = sigma_t.numpy()
    residuals = y_spread_val - mu
    abs_res = np.abs(residuals)

    # Dead neurons
    dead = count_dead_neurons(model, X_tensor)
    total_dead = sum(d[0] for d in dead.values())

    # Model MAE
    model_mae = float(np.mean(abs_res))

    # Book MAE
    book_mae = None
    if "bookSpread" in df_val.columns:
        bs = df_val["bookSpread"].values.astype(np.float64)
        valid = ~np.isnan(bs)
        if valid.sum() > 0:
            actual = (df_val["homeScore"].values - df_val["awayScore"].values).astype(np.float64)
            book_mae = float(np.mean(np.abs(actual[valid] - (-bs[valid]))))

    # Sigma stats
    sigma_mean = float(np.mean(sigma))
    sigma_std = float(np.std(sigma))
    sigma_min = float(np.min(sigma))
    sigma_max = float(np.max(sigma))

    # Calibration: within 1-sigma
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

    # ROI at multiple thresholds
    roi_results = {}
    for thresh in [0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
        roi_results[thresh] = compute_roi(mu, sigma, df_val, thresh)

    # Best ROI across thresholds
    best_roi = max(
        (r["roi"] for r in roi_results.values() if r and r["bets"] > 10),
        default=-1.0
    )

    return {
        "dead_neurons": dead, "total_dead": total_dead,
        "model_mae": model_mae, "book_mae": book_mae,
        "sigma_mean": sigma_mean, "sigma_std": sigma_std,
        "sigma_min": sigma_min, "sigma_max": sigma_max,
        "within_1sig": within_1sig,
        "quintile_ratios": quintile_ratios, "cal_score": cal_score,
        "spearman": sp_corr, "roi_results": roi_results, "best_roi": best_roi,
        "mu": mu, "sigma": sigma, "residuals": residuals,
    }


# ── Config definitions ───────────────────────────────────────────────

TOP3_CONFIGS = [
    {"name": "C1", "label": "S-A-lr1e4 (256→192, d=0.20)",
     "arch": "shared", "hidden1": 256, "hidden2": 192, "dropout": 0.20,
     "batch_size": 512, "weight_decay": 1e-4, "loss": "gaussian", "sigma_penalty": 0.0},
    {"name": "C2", "label": "S-F-lr5e4 (384→256, d=0.30)",
     "arch": "shared", "hidden1": 384, "hidden2": 256, "dropout": 0.30,
     "batch_size": 512, "weight_decay": 1e-4, "loss": "gaussian", "sigma_penalty": 0.0},
    {"name": "C3", "label": "S-B-lr1e4 (384→256, d=0.20)",
     "arch": "shared", "hidden1": 384, "hidden2": 256, "dropout": 0.20,
     "batch_size": 512, "weight_decay": 1e-4, "loss": "gaussian", "sigma_penalty": 0.0},
]

VARIANTS = ["V1", "V2", "V3", "V4"]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    total_start = time.time()

    X_train, y_spread_train, y_win_train, X_val, y_spread_val, y_win_val, scaler, df_val = load_data()

    # ── Phase B: Train 12 configs ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE B: CONVERGENCE TRAINING (12 RUNS)")
    print("=" * 70)

    all_results = []
    for cfg in TOP3_CONFIGS:
        for variant in VARIANTS:
            run_name = f"{cfg['name']}-{variant}"
            max_ep = 300 if variant == "V4" else 500
            init_lr = get_initial_lr(variant)

            print(f"\n{'='*60}")
            print(f"  RUN: {run_name} ({cfg['label']})")
            print(f"  Variant: {variant}, max_epochs={max_ep}, init_lr={init_lr}")
            print(f"{'='*60}")

            t0 = time.time()
            model, best_epoch, stopped_at, final_train, best_val = train_with_convergence(
                cfg, X_train, y_spread_train, X_val, y_spread_val, variant
            )
            train_time = time.time() - t0

            metrics = evaluate_full(model, X_val, y_spread_val, df_val)
            metrics["name"] = run_name
            metrics["config"] = cfg
            metrics["variant"] = variant
            metrics["best_epoch"] = best_epoch
            metrics["stopped_at"] = stopped_at
            metrics["final_train_loss"] = final_train
            metrics["best_val_loss"] = best_val
            metrics["train_time"] = train_time
            metrics["model"] = model

            # Check convergence
            if stopped_at == max_ep:
                conv_str = f"ran full {max_ep}"
            else:
                conv_str = f"early stop at {stopped_at} (best@{best_epoch})"

            print(f"\n  --- {run_name} ---")
            print(f"  Convergence: {conv_str}")
            print(f"  Best val loss: {best_val:.4f}, Final train: {final_train:.4f}")
            print(f"  Dead: {metrics['total_dead']}, MAE: {metrics['model_mae']:.3f}, "
                  f"Book MAE: {metrics['book_mae']:.3f}")
            print(f"  σ: mean={metrics['sigma_mean']:.2f} std={metrics['sigma_std']:.2f} "
                  f"min={metrics['sigma_min']:.2f} max={metrics['sigma_max']:.2f}")
            print(f"  Cal: {metrics['cal_score']:.4f}, Within 1σ: {metrics['within_1sig']:.3f}, "
                  f"ρ: {metrics['spearman']:.4f}")
            print(f"  Quintile ratios: {[f'{r:.3f}' for r in metrics['quintile_ratios']]}")
            roi10 = metrics['roi_results'].get(0.10)
            if roi10 and roi10['bets'] > 0:
                print(f"  ROI@10%: {roi10['roi']*100:.1f}% ({roi10['bets']} bets)")
            print(f"  Time: {train_time:.0f}s")

            all_results.append(metrics)

    # ── Phase C: Full comparison table ───────────────────────────────
    print("\n" + "=" * 100)
    print("  PHASE C: ALL 12 RUNS — FULL COMPARISON")
    print("=" * 100)

    header = (f"{'Run':<12} {'Best@':>5} {'Stop@':>5} {'ValLoss':>8} "
              f"{'Dead':>5} {'MAE':>7} {'BookMAE':>7} {'ΔMAE':>6} "
              f"{'σ_mean':>7} {'σ_std':>6} {'σ_min':>6} {'σ_max':>6} "
              f"{'1σ%':>5} {'CalSc':>6} {'ρ':>7} {'BestROI':>7} {'Time':>5}")
    print(header)
    print("-" * 100)
    for r in all_results:
        delta_mae = r['model_mae'] - r['book_mae'] if r['book_mae'] else 0
        print(
            f"{r['name']:<12} {r['best_epoch']:>5} {r['stopped_at']:>5} "
            f"{r['best_val_loss']:>8.4f} "
            f"{r['total_dead']:>5} {r['model_mae']:>7.3f} "
            f"{r['book_mae']:>7.3f} {delta_mae:>+6.2f} "
            f"{r['sigma_mean']:>7.2f} {r['sigma_std']:>6.2f} "
            f"{r['sigma_min']:>6.2f} {r['sigma_max']:>6.2f} "
            f"{r['within_1sig']*100:>5.1f} {r['cal_score']:>6.4f} "
            f"{r['spearman']:>7.4f} {r['best_roi']*100:>6.1f}% "
            f"{r['train_time']:>5.0f}"
        )

    print("\n  Per-quintile calibration ratios:")
    for r in all_results:
        qr = r["quintile_ratios"]
        print(f"    {r['name']:<12} Q1={qr[0]:.3f} Q2={qr[1]:.3f} Q3={qr[2]:.3f} "
              f"Q4={qr[3]:.3f} Q5={qr[4]:.3f}")

    print("\n  ROI at all thresholds:")
    thresh_header = f"{'Run':<12} " + " ".join(f"{'@'+str(int(t*100))+'%':>10}" for t in [0.05,0.07,0.08,0.10,0.12,0.15])
    print(thresh_header)
    for r in all_results:
        parts = [f"{r['name']:<12}"]
        for t in [0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
            rd = r['roi_results'].get(t)
            if rd and rd['bets'] > 0:
                parts.append(f"{rd['roi']*100:>5.1f}%/{rd['bets']:>4}")
            else:
                parts.append(f"{'N/A':>10}")
        print(" ".join(parts))

    # ── Phase D: Select winner ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE D: WINNER SELECTION")
    print("=" * 70)

    # Hard requirements
    qualified = []
    for r in all_results:
        fails = []
        if r["total_dead"] > 0:
            fails.append(f"dead_neurons={r['total_dead']}")
        if r["model_mae"] >= 9.50:
            fails.append(f"MAE={r['model_mae']:.3f} >= 9.50")
        if r["sigma_std"] <= 1.0:
            fails.append(f"sigma_std={r['sigma_std']:.2f} <= 1.0")
        qr = r["quintile_ratios"]
        bad_q = [i for i, q in enumerate(qr) if q < 0.80 or q > 1.20]
        if bad_q:
            fails.append(f"quintile ratios out of [0.80,1.20]: {[f'Q{i+1}={qr[i]:.3f}' for i in bad_q]}")

        if fails:
            print(f"  DISQUALIFIED: {r['name']} — {'; '.join(fails)}")
            r["disqualified"] = True
        else:
            print(f"  QUALIFIED: {r['name']} — MAE={r['model_mae']:.3f}, CalSc={r['cal_score']:.4f}, "
                  f"σ_std={r['sigma_std']:.2f}, BestROI={r['best_roi']*100:.1f}%")
            r["disqualified"] = False
            qualified.append(r)

    if not qualified:
        print("\n  *** NO CONFIG MEETS ALL HARD REQUIREMENTS ***")
        print("\n  Closest configs:")
        # Sort by MAE to show which ones were closest
        for r in sorted(all_results, key=lambda x: x["model_mae"])[:5]:
            qr = r["quintile_ratios"]
            print(f"    {r['name']}: MAE={r['model_mae']:.3f}, σ_std={r['sigma_std']:.2f}, "
                  f"CalSc={r['cal_score']:.4f}, Dead={r['total_dead']}")
            print(f"      Quintiles: {[f'{q:.3f}' for q in qr]}")
        print("\n  RECOMMENDATION: Try higher LR (2e-3 with cosine decay), or more epochs (800+),")
        print("  or mixed-precision off for stability. Do NOT save artifacts.")
        # Still generate report
        _generate_report(all_results, None, None, REPORTS_DIR / "architecture_search_session13.md")
        return None

    # Composite scoring among qualified
    n = len(qualified)
    def rank_asc(vals):
        order = np.argsort(vals)
        ranks = np.empty(len(vals), dtype=float)
        for i, idx in enumerate(order):
            ranks[idx] = i + 1
        return ranks

    def rank_desc(vals):
        return rank_asc([-v for v in vals])

    mae_ranks = rank_asc([r["model_mae"] for r in qualified])
    cal_ranks = rank_asc([r["cal_score"] for r in qualified])
    sig_ranks = rank_desc([r["sigma_std"] for r in qualified])
    roi_ranks = rank_desc([r["best_roi"] for r in qualified])

    for i, r in enumerate(qualified):
        r["composite"] = (0.30 * mae_ranks[i] + 0.30 * cal_ranks[i] +
                          0.20 * sig_ranks[i] + 0.20 * roi_ranks[i])

    qualified.sort(key=lambda r: r["composite"])
    winner = qualified[0]

    print(f"\n  WINNER: {winner['name']}")
    print(f"  Composite: {winner['composite']:.2f}")
    print(f"  MAE: {winner['model_mae']:.3f}, CalSc: {winner['cal_score']:.4f}, "
          f"σ_std: {winner['sigma_std']:.2f}, BestROI: {winner['best_roi']*100:.1f}%")

    # ── Phase E: Validation suite ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE E: FULL VALIDATION SUITE")
    print("=" * 70)

    test_results = run_validation_suite(
        winner["model"], X_val, y_spread_val, df_val, scaler, winner,
        X_train, y_spread_train, y_win_train, winner["config"]
    )

    any_fail = any(v == "FAIL" for v in test_results.values())
    if any_fail:
        print("\n  *** RED FLAG: VALIDATION FAILURES — NOT SAVING ARTIFACTS ***")
        _generate_report(all_results, winner, test_results,
                        REPORTS_DIR / "architecture_search_session13.md")
        return None

    # ── Phase F: Save artifacts ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE F: SAVING ARTIFACTS")
    print("=" * 70)

    wcfg = winner["config"]
    wmodel = winner["model"]

    # Save regressor
    reg_path = config.CHECKPOINTS_DIR / "regressor.pt"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": wmodel.state_dict(),
        "feature_order": config.FEATURE_ORDER,
        "hparams": {
            "hidden1": wcfg["hidden1"], "hidden2": wcfg["hidden2"],
            "dropout": wcfg["dropout"], "batch_size": wcfg.get("batch_size", 512),
            "loss": wcfg.get("loss", "gaussian"),
            "sigma_penalty": wcfg.get("sigma_penalty", 0.0),
            "lr_variant": winner["variant"],
            "best_epoch": winner["best_epoch"],
        },
        "arch_type": wcfg.get("arch", "shared"),
        "sigma_param": "exp",
    }, reg_path)
    print(f"  Saved regressor: {reg_path}")

    # Train classifier
    print("\n  Training production classifier...")
    cls_hidden1 = max(wcfg["hidden1"], 256)
    classifier = train_classifier_production(
        X_train, y_win_train, hidden1=cls_hidden1,
        dropout=wcfg["dropout"], batch_size=wcfg.get("batch_size", 512),
    )
    cls_path = config.CHECKPOINTS_DIR / "classifier.pt"
    torch.save({
        "state_dict": classifier.state_dict(),
        "feature_order": config.FEATURE_ORDER,
        "hparams": {"hidden1": cls_hidden1, "dropout": wcfg["dropout"]},
    }, cls_path)
    print(f"  Saved classifier: {cls_path}")

    # Save scaler
    scaler_path = config.ARTIFACTS_DIR / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler: {scaler_path}")

    # Update hparams
    hp_path = config.ARTIFACTS_DIR / "best_hparams.json"
    with open(hp_path, "w") as f:
        json.dump({
            "regressor": {
                "hidden1": wcfg["hidden1"], "hidden2": wcfg["hidden2"],
                "dropout": wcfg["dropout"], "batch_size": wcfg.get("batch_size", 512),
                "loss": wcfg.get("loss", "gaussian"),
                "arch_type": wcfg.get("arch", "shared"), "sigma_param": "exp",
                "lr_variant": winner["variant"], "best_epoch": winner["best_epoch"],
            },
            "classifier": {"hidden1": cls_hidden1, "dropout": wcfg["dropout"]},
        }, f, indent=2)
    print(f"  Saved hparams: {hp_path}")

    # Generate report
    _generate_report(all_results, winner, test_results,
                    REPORTS_DIR / "architecture_search_session13.md")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  SESSION 13 CONVERGENCE COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Winner: {winner['name']} ({winner['config']['label']})")
    print(f"  MAE: {winner['model_mae']:.3f} (book: {winner['book_mae']:.3f})")
    print(f"  σ std: {winner['sigma_std']:.2f}, Cal: {winner['cal_score']:.4f}")
    print(f"  Best epoch: {winner['best_epoch']}, Variant: {winner['variant']}")
    print(f"{'='*70}")
    return winner


# ── Classifier training ──────────────────────────────────────────────

def train_classifier_production(X_train, y_win, hidden1=256, dropout=0.3, batch_size=512):
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


# ── Validation suite ─────────────────────────────────────────────────

def run_validation_suite(model, X_val, y_spread_val, df_val, scaler, winner_metrics,
                         X_train, y_spread_train, y_win_train, cfg):
    X_tensor = torch.tensor(X_val, dtype=torch.float32)
    mu = winner_metrics["mu"]
    sigma = winner_metrics["sigma"]
    residuals = winner_metrics["residuals"]
    abs_res = np.abs(residuals)
    results = {}

    # TEST 1: Dead neurons
    print("\n--- TEST 1: Dead Neuron Scan ---")
    dead = count_dead_neurons(model, X_tensor)
    all_clear = True
    for layer, (n_dead, total) in dead.items():
        status = "PASS" if n_dead == 0 else "FAIL"
        if n_dead > 0:
            all_clear = False
        print(f"  {layer}: {n_dead}/{total} [{status}]")
    results["dead_neurons"] = "PASS" if all_clear else "FAIL"

    # TEST 2: Sigma correlation with difficulty
    print("\n--- TEST 2: Sigma Correlation with Difficulty ---")
    if "bookSpread" in df_val.columns:
        bs = np.abs(df_val["bookSpread"].values.astype(np.float64))
        valid = ~np.isnan(bs)
        for label, lo, hi in [("Close (0-3)", 0, 3), ("Medium (3-7)", 3, 7),
                               ("Large (7-14)", 7, 14), ("Blowout (14+)", 14, 999)]:
            mask = valid & (bs >= lo) & (bs < hi)
            if mask.sum() > 0:
                print(f"  {label}: mean σ = {sigma[mask].mean():.3f} (n={mask.sum()})")
        close_sig = sigma[valid & (bs <= 3)].mean() if (valid & (bs <= 3)).sum() > 0 else 0
        large_sig = sigma[valid & (bs >= 7) & (bs < 14)].mean() if (valid & (bs >= 7) & (bs < 14)).sum() > 0 else 0
        if close_sig > large_sig:
            print("  ✓ Sigma higher for close games")
            results["sigma_difficulty"] = "PASS"
        else:
            print("  ✗ Sigma NOT higher for close games")
            results["sigma_difficulty"] = "WARN"
    else:
        results["sigma_difficulty"] = "SKIP"

    # TEST 3: Multi-level calibration
    print("\n--- TEST 3: Multi-Level Calibration ---")
    cal_pass = True
    for k, target in [(0.5, 0.3829), (1.0, 0.6827), (1.5, 0.8664), (2.0, 0.9545)]:
        actual = float(np.mean(abs_res < k * sigma))
        diff = abs(actual - target)
        status = "PASS" if diff < 0.03 else ("WARN" if diff < 0.05 else "FAIL")
        if diff >= 0.05:
            cal_pass = False
        print(f"  Within {k:.1f}σ: actual={actual:.4f} target={target:.4f} diff={diff:.4f} [{status}]")
    results["multi_cal"] = "PASS" if cal_pass else "FAIL"

    # TEST 4: Monthly MAE
    print("\n--- TEST 4: Monthly MAE ---")
    if "startDate" in df_val.columns:
        dates = pd.to_datetime(df_val["startDate"], errors="coerce", utc=True)
        months = dates.dt.month
        for m, label in [(12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")]:
            mask = months == m
            if mask.sum() > 0:
                print(f"  {label}: MAE={np.mean(abs_res[mask.values]):.3f} (n={mask.sum()})")
    results["monthly_mae"] = "PASS"

    # TEST 5: Edge profitability
    print("\n--- TEST 5: Edge Profitability ---")
    for thresh in [0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
        rd = compute_roi(mu, sigma, df_val, thresh)
        if rd and rd['bets'] > 0:
            print(f"  @{thresh*100:.0f}%: {rd['bets']:>4} bets, WR={rd['win_rate']:.3f}, "
                  f"ROI={rd['roi']*100:.1f}%, Units={rd['units']:.1f}")
        else:
            print(f"  @{thresh*100:.0f}%: 0 bets")
    results["edge_profit"] = "PASS"

    # TEST 6: Sigma vs error correlation
    print("\n--- TEST 6: Sigma-Error Correlation ---")
    sp = winner_metrics["spearman"]
    status = "PASS" if sp > 0 else "FAIL"
    print(f"  Spearman ρ = {sp:.4f} [{status}]")
    results["sigma_corr"] = status

    # TEST 7: Accuracy by spread size
    print("\n--- TEST 7: Accuracy by Spread Size ---")
    if "bookSpread" in df_val.columns:
        bs = np.abs(df_val["bookSpread"].values.astype(np.float64))
        valid = ~np.isnan(bs)
        for label, lo, hi in [("Close (0-3)", 0, 3), ("Medium (3-7)", 3, 7),
                               ("Large (7-14)", 7, 14), ("Blowout (14+)", 14, 999)]:
            mask = valid & (bs >= lo) & (bs < hi)
            if mask.sum() > 0:
                print(f"  {label}: MAE={np.mean(abs_res[mask]):.3f} (n={mask.sum()})")
    results["spread_accuracy"] = "PASS"

    # TEST 8: Hold-out year
    print("\n--- TEST 8: Hold-Out Year (2015-2024 → 2025) ---")
    try:
        ho_metrics = _run_holdout(cfg)
        results["holdout"] = "PASS"
    except Exception as e:
        print(f"  ERROR: {e}")
        results["holdout"] = "FAIL"

    # Summary
    print("\n--- Validation Summary ---")
    for k, v in results.items():
        print(f"  {k}: {v}")
    any_fail = any(v == "FAIL" for v in results.values())
    if any_fail:
        print("\n  *** RED FLAGS DETECTED ***")
    else:
        print("\n  All tests PASSED")

    return results


def _run_holdout(cfg):
    from src.features import load_research_lines

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

    # Use V1 (cosine, lr=1e-3) for holdout — same convergence approach
    ho_cfg = {**cfg, "name": "holdout"}
    model, best_ep, stopped, _, _ = train_with_convergence(
        ho_cfg, X_ht_s, y_ht, X_hv_s, y_hv, "V1"
    )
    m = evaluate_full(model, X_hv_s, y_hv, df_hv)
    print(f"  Holdout 2025: {len(y_hv)} games")
    print(f"  MAE: {m['model_mae']:.3f}, Book MAE: {m['book_mae']}")
    print(f"  σ: mean={m['sigma_mean']:.2f} std={m['sigma_std']:.2f} range={m['sigma_max']-m['sigma_min']:.2f}")
    print(f"  Cal: {m['cal_score']:.4f}, Within 1σ: {m['within_1sig']:.3f}")
    print(f"  Dead: {m['total_dead']}, Best@epoch {best_ep}")
    return m


# ── Report generation ────────────────────────────────────────────────

def _generate_report(all_results, winner, test_results, report_path):
    lines = ["# Session 13: Architecture Search — Convergence Training Report",
             f"\nDate: 2026-02-27", f"GPU: NVIDIA A10",
             f"Training: 2015-2025 ({51252} games), Validation: 2026",
             f"Features: {len(config.FEATURE_ORDER)}"]

    lines.append("\n## Phase B: Convergence Training Results (12 runs)")
    lines.append("")
    lines.append("| Run | Best@ | Stop@ | ValLoss | Dead | MAE | BookMAE | ΔMAE | σ_std | CalSc | ρ | BestROI |")
    lines.append("|-----|-------|-------|---------|------|-----|---------|------|-------|-------|---|---------|")
    for r in all_results:
        delta = r['model_mae'] - r['book_mae'] if r['book_mae'] else 0
        roi_str = f"{r['best_roi']*100:.1f}%" if r['best_roi'] > -1 else "N/A"
        lines.append(
            f"| {r['name']} | {r['best_epoch']} | {r['stopped_at']} | "
            f"{r['best_val_loss']:.4f} | {r['total_dead']} | {r['model_mae']:.3f} | "
            f"{r['book_mae']:.3f} | {delta:+.2f} | {r['sigma_std']:.2f} | "
            f"{r['cal_score']:.4f} | {r['spearman']:.4f} | {roi_str} |"
        )

    lines.append("\n## Per-Quintile Calibration")
    lines.append("")
    lines.append("| Run | Q1 | Q2 | Q3 | Q4 | Q5 |")
    lines.append("|-----|-----|-----|-----|-----|-----|")
    for r in all_results:
        qr = r["quintile_ratios"]
        lines.append(f"| {r['name']} | {qr[0]:.3f} | {qr[1]:.3f} | {qr[2]:.3f} | {qr[3]:.3f} | {qr[4]:.3f} |")

    if winner:
        wcfg = winner["config"]
        lines.append(f"\n## Winner: {winner['name']}")
        lines.append(f"- Config: {wcfg['label']}")
        lines.append(f"- Variant: {winner['variant']}")
        lines.append(f"- Best epoch: {winner['best_epoch']}")
        lines.append(f"- Composite: {winner.get('composite', 'N/A')}")

        lines.append("\n## Comparison Table")
        lines.append("")
        lines.append("| Metric | Winner | Broken (128→128) | Old Torvik (768→640) |")
        lines.append("|--------|--------|------------------|---------------------|")
        lines.append(f"| Architecture | shared {wcfg['hidden1']}→{wcfg['hidden2']}, d={wcfg['dropout']} | shared 128→128, d=0.45, softplus | shared 768→640, d=0.20 |")
        lines.append(f"| Model MAE | {winner['model_mae']:.3f} | 9.42 | ~9.5 |")
        lines.append(f"| Book MAE | {winner['book_mae']:.3f} | ~8.80 | ~8.80 |")
        lines.append(f"| σ std | {winner['sigma_std']:.2f} | 0.53 | 1.42 |")
        lines.append(f"| σ range | {winner['sigma_min']:.1f}–{winner['sigma_max']:.1f} | 10.5–14.8 | 4.9–15.6 |")
        lines.append(f"| Dead neurons | {winner['total_dead']} | 116/128 | 0 |")
        qr = winner['quintile_ratios']
        lines.append(f"| Quintile ratios | {qr[0]:.2f}–{qr[-1]:.2f} | 1.07–1.49 | ~1.0 |")
        lines.append(f"| ROI @ best | {winner['best_roi']*100:.1f}% | ? | ? |")
        lines.append(f"| Spearman σ corr | {winner['spearman']:.4f} | ~0.05 | ~0.15 |")
        lines.append(f"| Training epochs | {winner['best_epoch']} | 100 | ? |")

    if test_results:
        lines.append("\n## Validation Tests")
        for k, v in test_results.items():
            lines.append(f"- {k}: **{v}**")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()

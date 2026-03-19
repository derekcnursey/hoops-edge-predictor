#!/usr/bin/env python3
"""Session 13b: Ensemble + LightGBM + Feature Engineering exploration.

Phases:
  1. Neural net ensemble (top 5 zero-dead models)
  2. LightGBM baseline + Optuna tuning
  3. Hybrid ensemble (LGB mu + NN sigma)
  4. Feature engineering for trees
  5. Temporal weighting
  6. LR warmup for neural net
  7. Full comparison table
  8. Walk-forward the winner
  9. Save best approach
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import lightgbm as lgb
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets, load_research_lines
from src.trainer import impute_column_means

optuna.logging.set_verbosity(optuna.logging.WARNING)

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
MAX_EPOCHS = 500
PATIENCE = 50
REPORTS_DIR = config.PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════

def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    """Load train/val data with book spreads. Returns unscaled X arrays + df_val."""
    df_train = load_multi_season_features(
        list(range(2015, 2026)), adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
    df_train = df_train.dropna(subset=["homeScore", "awayScore"])
    df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]

    df_val = load_multi_season_features(
        [2026], adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
    df_val = df_val.dropna(subset=["homeScore", "awayScore"])
    df_val = df_val[(df_val["homeScore"] != 0) | (df_val["awayScore"] != 0)]

    try:
        lines_df = load_research_lines(2026)
        if not lines_df.empty:
            ld = lines_df.sort_values("provider").drop_duplicates(
                subset=["gameId"], keep="first")
            if "spread" in ld.columns:
                df_val = df_val.merge(
                    ld[["gameId", "spread"]].rename(columns={"spread": "bookSpread"}),
                    on="gameId", how="left")
    except Exception:
        pass

    X_train_raw = get_feature_matrix(df_train).values.astype(np.float32)
    targets_train = get_targets(df_train)
    y_train = targets_train["spread_home"].values.astype(np.float32)

    X_val_raw = get_feature_matrix(df_val).values.astype(np.float32)
    targets_val = get_targets(df_val)
    y_val = targets_val["spread_home"].values.astype(np.float32)

    X_train_raw = impute_column_means(X_train_raw)
    X_val_raw = impute_column_means(X_val_raw)

    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train_s = scaler.transform(X_train_raw).astype(np.float32)
    X_val_s = scaler.transform(X_val_raw).astype(np.float32)

    book = df_val["bookSpread"].values.astype(np.float64) if "bookSpread" in df_val.columns else np.full(len(df_val), np.nan)
    has_book = ~np.isnan(book)

    print(f"  Train: {len(df_train)}, Val: {len(df_val)}, Book: {has_book.sum()}")
    return (X_train_raw, X_train_s, y_train, X_val_raw, X_val_s, y_val,
            scaler, book, has_book, df_val, df_train)


def train_nn(X_train_s, y_train, X_val_s, y_val, hp, verbose=True,
             sample_weights=None, warmup_epochs=0):
    """Train MLPRegressor. Returns (model, best_epoch)."""
    device = get_device()
    use_amp = device.type == "cuda"

    model = MLPRegressor(
        input_dim=X_train_s.shape[1],
        hidden1=hp["hidden1"], hidden2=hp["hidden2"],
        dropout=hp["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"],
        weight_decay=hp.get("weight_decay", 1e-4))

    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(MAX_EPOCHS - warmup_epochs, 1)
            return max(1e-5 / hp["lr"], 0.5 * (1 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    amp_scaler = GradScaler(device.type, enabled=use_amp)

    # Dataset with optional weights
    if sample_weights is not None:
        w_tensor = torch.tensor(sample_weights, dtype=torch.float32)
    else:
        w_tensor = None

    ds = HoopsDataset(X_train_s, spread=y_train, home_win=np.zeros(len(y_train)))
    loader = DataLoader(ds, batch_size=hp.get("batch_size", 4096),
                        shuffle=True, drop_last=True, num_workers=2,
                        pin_memory=True)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    model.train()
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            x, spread, _ = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                mu, log_sigma = model(x)
                nll, sigma = gaussian_nll_loss(mu, log_sigma, spread)
                loss = nll.mean()
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_loss = _val_loss_nn(model, X_val_t, y_val_t, device)
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
            if verbose:
                print(f"      Early stop at {ep} (best@{best_epoch})")
            break

    model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def train_nn_weighted(X_train_s, y_train, X_val_s, y_val, hp,
                      sample_weights, verbose=True):
    """Train MLPRegressor with per-sample loss weights."""
    device = get_device()
    use_amp = device.type == "cuda"

    model = MLPRegressor(
        input_dim=X_train_s.shape[1],
        hidden1=hp["hidden1"], hidden2=hp["hidden2"],
        dropout=hp["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"],
        weight_decay=hp.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    amp_scaler = GradScaler(device.type, enabled=use_amp)

    # Create dataset with weights as home_win channel (hacky but works)
    ds = HoopsDataset(X_train_s, spread=y_train,
                      home_win=sample_weights.astype(np.float32))
    loader = DataLoader(ds, batch_size=hp.get("batch_size", 4096),
                        shuffle=True, drop_last=True, num_workers=2,
                        pin_memory=True)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    model.train()
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            x, spread, weights = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                mu, log_sigma = model(x)
                nll, sigma = gaussian_nll_loss(mu, log_sigma, spread)
                loss = (nll * weights).mean()  # weighted NLL
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_loss = _val_loss_nn(model, X_val_t, y_val_t, device)
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
            if verbose:
                print(f"      Early stop at {ep} (best@{best_epoch})")
            break

    model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


@torch.no_grad()
def _val_loss_nn(model, X_val_t, y_val_t, device):
    model.eval()
    total = 0.0
    n = 0
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


@torch.no_grad()
def predict_nn(model, X_val_s):
    X_t = torch.tensor(X_val_s, dtype=torch.float32)
    mu_t, ls_t = model(X_t)
    sigma_t = torch.exp(ls_t).clamp(min=0.5, max=30.0)
    return mu_t.numpy(), sigma_t.numpy()


def count_dead(model, X_val_s):
    model.eval()
    activations = {}
    hooks = []
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.ReLU):
            def make_hook(name):
                def hook(mod, inp, out):
                    activations[name] = out.detach().cpu()
                return hook
            hooks.append(layer.register_forward_hook(make_hook(f"relu_{i}")))
    X_t = torch.tensor(X_val_s, dtype=torch.float32)
    all_acts = {}
    with torch.no_grad():
        for s in range(0, len(X_t), 4096):
            model(X_t[s:min(s + 4096, len(X_t))])
            for name, act in activations.items():
                all_acts.setdefault(name, []).append(act)
    for h in hooks:
        h.remove()
    total_dead = 0
    for name, acts in all_acts.items():
        full = torch.cat(acts, dim=0)
        zero_frac = (full == 0).float().mean(dim=0)
        total_dead += (zero_frac > 0.99).sum().item()
    return total_dead


def quintile_cal(sigma, residuals):
    qi = np.array_split(np.argsort(sigma), 5)
    ratios = []
    for idx in qi:
        actual_std = np.std(residuals[idx])
        pred_mean = np.mean(sigma[idx])
        ratios.append(float(actual_std / pred_mean) if pred_mean > 0 else 999.0)
    return float(np.mean([abs(r - 1.0) for r in ratios]))


def compute_roi(mu, sigma, book, actual, threshold):
    valid = ~np.isnan(book)
    if valid.sum() == 0:
        return {"bets": 0, "roi": 0.0, "units": 0.0}
    edge_home = mu[valid] + book[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    hcp = normal_cdf(edge_z)
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, hcp, 1.0 - hcp)
    prob_edge = pick_prob - 0.5238
    bet_mask = prob_edge >= threshold
    n = bet_mask.sum()
    if n == 0:
        return {"bets": 0, "roi": 0.0, "units": 0.0}
    actual_v = actual[valid]
    book_v = book[valid]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)
    w = pick_won[bet_mask].sum()
    profit_per_1 = 100.0 / 110.0
    units = w * profit_per_1 - (n - w)
    return {"bets": int(n), "roi": float(units / n), "units": float(units)}


def eval_predictions(mu, sigma, actual, book, has_book, label=""):
    """Evaluate a set of predictions. Returns dict of metrics."""
    res = actual - mu
    abs_res = np.abs(res)
    overall_mae = float(np.mean(abs_res))
    bs_mae = float(np.mean(abs_res[has_book])) if has_book.sum() > 0 else None
    sig_std = float(np.std(sigma)) if sigma is not None else None
    cal = quintile_cal(sigma, res) if sigma is not None else None

    roi_10 = compute_roi(mu, sigma, book, actual, 0.10) if sigma is not None else None
    roi_12 = compute_roi(mu, sigma, book, actual, 0.12) if sigma is not None else None
    roi_15 = compute_roi(mu, sigma, book, actual, 0.15) if sigma is not None else None

    # Spearman(sigma, |residuals|) — measures sigma usefulness
    spearman_rho = None
    if sigma is not None:
        rho, _ = scipy_stats.spearmanr(sigma, abs_res)
        spearman_rho = float(rho)

    return {
        "label": label, "bs_mae": bs_mae, "overall_mae": overall_mae,
        "sig_std": sig_std, "cal": cal,
        "roi_10": roi_10, "roi_12": roi_12, "roi_15": roi_15,
        "spearman_rho": spearman_rho,
        "mu": mu, "sigma": sigma,
    }


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: NEURAL NET ENSEMBLE
# ══════════════════════════════════════════════════════════════════════

# Top 5 zero-dead configs from the 20-run sweep
ENSEMBLE_CONFIGS = [
    {"name": "C2-V2", "hidden1": 384, "hidden2": 256, "dropout": 0.20,
     "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4, "ref_mae": 9.129},
    {"name": "C1-V1", "hidden1": 256, "hidden2": 192, "dropout": 0.20,
     "lr": 1e-3, "batch_size": 2048, "weight_decay": 1e-4, "ref_mae": 9.153},
    {"name": "C4-V1", "hidden1": 768, "hidden2": 640, "dropout": 0.20,
     "lr": 1e-3, "batch_size": 2048, "weight_decay": 1e-4, "ref_mae": 9.154},
    {"name": "C2-V1", "hidden1": 384, "hidden2": 256, "dropout": 0.20,
     "lr": 1e-3, "batch_size": 2048, "weight_decay": 1e-4, "ref_mae": 9.171},
    {"name": "C5-V1", "hidden1": 512, "hidden2": 384, "dropout": 0.30,
     "lr": 1e-3, "batch_size": 2048, "weight_decay": 1e-4, "ref_mae": 9.174},
]


def phase_1(X_train_s, y_train, X_val_s, y_val, book, has_book):
    print("\n" + "=" * 70)
    print("  PHASE 1: NEURAL NET ENSEMBLE")
    print("=" * 70)

    models = []
    individual_results = []

    for cfg in ENSEMBLE_CONFIGS:
        name = cfg["name"]
        hp = {k: v for k, v in cfg.items() if k not in ("name", "ref_mae")}
        print(f"\n  Training {name} ({cfg['hidden1']}→{cfg['hidden2']})...")
        t0 = time.time()

        model, best_ep = train_nn(X_train_s, y_train, X_val_s, y_val,
                                   hp=hp, verbose=False)
        mu, sigma = predict_nn(model, X_val_s)
        actual = y_val
        dead = count_dead(model, X_val_s)
        elapsed = time.time() - t0

        ev = eval_predictions(mu, sigma, actual, book, has_book, name)
        ev["dead"] = dead
        ev["mu"] = mu
        ev["sigma"] = sigma
        ev["best_epoch"] = best_ep
        individual_results.append(ev)
        models.append((name, model, mu, sigma))

        print(f"    BS-MAE={ev['bs_mae']:.3f} dead={dead} σ_std={ev['sig_std']:.2f} "
              f"cal={ev['cal']:.3f} ep@{best_ep} [{elapsed:.0f}s]")

    # Ensembles
    all_mu = np.stack([m[2] for m in models])  # (5, N)
    all_sigma = np.stack([m[3] for m in models])
    actual = y_val

    ensembles = {}

    # A) Mean
    mu_mean = np.mean(all_mu, axis=0)
    sig_mean = np.mean(all_sigma, axis=0)
    ensembles["A) Mean (5)"] = eval_predictions(mu_mean, sig_mean, actual, book, has_book, "Mean(5)")

    # B) Top-3 by MAE
    maes = [r["bs_mae"] for r in individual_results]
    top3_idx = np.argsort(maes)[:3]
    mu_top3 = np.mean(all_mu[top3_idx], axis=0)
    sig_top3 = np.mean(all_sigma[top3_idx], axis=0)
    top3_names = [individual_results[i]["label"] for i in top3_idx]
    ensembles["B) Top-3"] = eval_predictions(mu_top3, sig_top3, actual, book, has_book,
                                              f"Top3({','.join(top3_names)})")

    # C) Weighted by 1/MAE
    weights = np.array([1.0 / m for m in maes])
    weights /= weights.sum()
    mu_weighted = np.average(all_mu, axis=0, weights=weights)
    sig_weighted = np.average(all_sigma, axis=0, weights=weights)
    ensembles["C) Weighted"] = eval_predictions(mu_weighted, sig_weighted, actual, book, has_book, "Weighted")

    # D) Median
    mu_median = np.median(all_mu, axis=0)
    sig_median = np.median(all_sigma, axis=0)
    ensembles["D) Median"] = eval_predictions(mu_median, sig_median, actual, book, has_book, "Median")

    # Print table
    print(f"\n  ── Individual Models ──")
    print(f"  {'Model':>12} {'BS-MAE':>7} {'MAE':>7} {'σ_std':>6} {'Cal':>6} {'Dead':>5} {'ROI@12%':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5} {'-'*8}")
    for r in individual_results:
        roi12 = r["roi_12"]["roi"] * 100 if r["roi_12"] and r["roi_12"]["bets"] > 0 else 0
        print(f"  {r['label']:>12} {r['bs_mae']:>7.3f} {r['overall_mae']:>7.3f} "
              f"{r['sig_std']:>6.2f} {r['cal']:>6.3f} {r['dead']:>5} {roi12:>+7.1f}%")

    print(f"\n  ── Ensembles ──")
    print(f"  {'Method':>15} {'BS-MAE':>7} {'MAE':>7} {'σ_std':>6} {'Cal':>6} {'ROI@12%':>8}")
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*8}")
    best_ens = None
    for ename, ev in ensembles.items():
        roi12 = ev["roi_12"]["roi"] * 100 if ev["roi_12"] and ev["roi_12"]["bets"] > 0 else 0
        flag = " ***" if ev["bs_mae"] < 9.129 else ""
        print(f"  {ename:>15} {ev['bs_mae']:>7.3f} {ev['overall_mae']:>7.3f} "
              f"{ev['sig_std']:>6.2f} {ev['cal']:>6.3f} {roi12:>+7.1f}%{flag}")
        if best_ens is None or ev["bs_mae"] < best_ens[1]["bs_mae"]:
            best_ens = (ename, ev)

    if best_ens[1]["bs_mae"] < 9.129:
        print(f"\n  *** ENSEMBLE BEATS 9.129: {best_ens[0]} at {best_ens[1]['bs_mae']:.3f} ***")

    return models, individual_results, ensembles, all_mu, all_sigma


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: LIGHTGBM
# ══════════════════════════════════════════════════════════════════════

def phase_2(X_train_raw, y_train, X_val_raw, y_val, book, has_book):
    print("\n" + "=" * 70)
    print("  PHASE 2: LIGHTGBM")
    print("=" * 70)

    # 2A: Baseline
    print("\n  2A: Baseline LightGBM")
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
        "feature_pre_filter": False,
    }
    lgb_train = lgb.Dataset(X_train_raw, y_train, params=params, free_raw_data=False)
    lgb_val_ds = lgb.Dataset(X_val_raw, y_val, reference=lgb_train, free_raw_data=False)
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(500)]
    t0 = time.time()
    model_base = lgb.train(params, lgb_train, num_boost_round=5000,
                           valid_sets=[lgb_val_ds], callbacks=callbacks)
    elapsed = time.time() - t0

    mu_base = model_base.predict(X_val_raw)
    res_base = y_val - mu_base
    overall_mae_base = float(np.mean(np.abs(res_base)))
    bs_mae_base = float(np.mean(np.abs(res_base[has_book])))
    print(f"    BS-MAE: {bs_mae_base:.3f}, Overall MAE: {overall_mae_base:.3f} "
          f"[{elapsed:.1f}s, {model_base.best_iteration} rounds]")

    # 2B: Optuna-tuned LightGBM (200 trials)
    print("\n  2B: Optuna-tuned LightGBM (200 trials)")

    def objective(trial):
        p = {
            "objective": "regression_l1",
            "metric": "mae",
            "verbosity": -1,
            "n_jobs": -1,
            "seed": 42,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        cb = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        m = lgb.train(p, lgb_train, num_boost_round=5000,
                      valid_sets=[lgb_val_ds], callbacks=cb)
        pred = m.predict(X_val_raw)
        return float(np.mean(np.abs(y_val[has_book] - pred[has_book])))

    t0 = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, show_progress_bar=False)
    elapsed = time.time() - t0

    print(f"    Best BS-MAE: {study.best_value:.3f} [{elapsed:.0f}s]")
    print(f"    Best params: {json.dumps(study.best_params, indent=6)}")

    # Top 5 trials
    trials = sorted(study.trials, key=lambda t: t.value)[:5]
    print(f"\n    Top 5 trials:")
    for i, t in enumerate(trials):
        print(f"      {i+1}. BS-MAE={t.value:.3f} lr={t.params['learning_rate']:.4f} "
              f"leaves={t.params['num_leaves']} depth={t.params['max_depth']}")

    # Retrain best
    best_params = {**study.best_params, "objective": "regression_l1",
                   "metric": "mae", "verbosity": -1, "n_jobs": -1, "seed": 42}
    cb = [lgb.early_stopping(50), lgb.log_evaluation(0)]
    model_tuned = lgb.train(best_params, lgb_train, num_boost_round=5000,
                            valid_sets=[lgb_val_ds], callbacks=cb)
    mu_tuned = model_tuned.predict(X_val_raw)
    res_tuned = y_val - mu_tuned
    overall_mae_tuned = float(np.mean(np.abs(res_tuned)))
    bs_mae_tuned = float(np.mean(np.abs(res_tuned[has_book])))
    print(f"\n    Tuned: BS-MAE={bs_mae_tuned:.3f}, Overall={overall_mae_tuned:.3f}")

    # 2C: Feature importance
    print("\n  2C: LightGBM Feature Importance (top 15)")
    feat_names = config.FEATURE_ORDER
    imp_gain = model_tuned.feature_importance(importance_type="gain")
    imp_split = model_tuned.feature_importance(importance_type="split")
    imp_order = np.argsort(imp_gain)[::-1]

    print(f"  {'Rank':>4} {'Feature':>35} {'Gain':>10} {'Split':>8}")
    print(f"  {'-'*4} {'-'*35} {'-'*10} {'-'*8}")
    for i, idx in enumerate(imp_order[:15]):
        print(f"  {i+1:>4} {feat_names[idx]:>35} {imp_gain[idx]:>10.0f} {imp_split[idx]:>8}")

    return model_base, model_tuned, mu_base, mu_tuned, best_params, study


# ══════════════════════════════════════════════════════════════════════
# PHASE 3: HYBRID ENSEMBLE
# ══════════════════════════════════════════════════════════════════════

def _sigma_analysis(label, mu, sigma, actual, has_book):
    """Compute sigma calibration and usefulness for a given mu + sigma pair."""
    residuals = actual - mu
    # Quintile calibration: ratio = std(residuals) / mean(sigma) per quintile
    qi = np.array_split(np.argsort(sigma), 5)
    ratios = []
    for idx in qi:
        r_std = np.std(residuals[idx])
        s_mean = np.mean(sigma[idx])
        ratios.append(float(r_std / s_mean) if s_mean > 0 else 999.0)
    cal = float(np.mean([abs(r - 1.0) for r in ratios]))
    # Spearman: sigma vs |residuals|  (sigma useful if positive)
    rho, pval = scipy_stats.spearmanr(sigma, np.abs(residuals))
    print(f"    Sigma analysis for {label}:")
    print(f"      Quintile ratios: [{', '.join(f'{r:.2f}' for r in ratios)}]")
    print(f"      Cal (mean|ratio-1|): {cal:.3f}")
    print(f"      Spearman(σ, |resid|): ρ={rho:.3f} p={pval:.2e}")
    return {"cal": cal, "quintile_ratios": ratios, "spearman_rho": float(rho),
            "spearman_p": float(pval)}


def phase_3(mu_lgb, mu_nn, sigma_nn, y_val, book, has_book, X_val_raw):
    print("\n" + "=" * 70)
    print("  PHASE 3: HYBRID ENSEMBLE (LGB mu + NN sigma)")
    print("=" * 70)

    actual = y_val

    # Baseline sigma analysis: NN mu + NN sigma
    print("\n  Baseline sigma analysis (NN mu + NN sigma):")
    sa_nn = _sigma_analysis("NN pure", mu_nn, sigma_nn, actual, has_book)

    # 3A: LGB mu + NN sigma
    print("\n  3A: LightGBM mu + NN sigma")
    ev_3a = eval_predictions(mu_lgb, sigma_nn, actual, book, has_book, "LGB+NNσ")
    sa_3a = _sigma_analysis("LGB+NNσ", mu_lgb, sigma_nn, actual, has_book)
    ev_3a["spearman_rho"] = sa_3a["spearman_rho"]
    print(f"    BS-MAE={ev_3a['bs_mae']:.3f} σ_std={ev_3a['sig_std']:.2f} "
          f"cal={sa_3a['cal']:.3f}")

    # 3B: 50/50 blend
    print("\n  3B: 50/50 blend mu + NN sigma")
    mu_blend = 0.5 * mu_lgb + 0.5 * mu_nn
    ev_3b = eval_predictions(mu_blend, sigma_nn, actual, book, has_book, "Blend50")
    sa_3b = _sigma_analysis("Blend50", mu_blend, sigma_nn, actual, has_book)
    ev_3b["spearman_rho"] = sa_3b["spearman_rho"]
    print(f"    BS-MAE={ev_3b['bs_mae']:.3f}")

    # 3C: Optimized blend weight
    print("\n  3C: Optimized blend weight")
    best_alpha = 0.5
    best_mae = 999.0
    for alpha in np.arange(0.0, 1.01, 0.05):
        mu_b = alpha * mu_lgb + (1 - alpha) * mu_nn
        mae = float(np.mean(np.abs(actual[has_book] - mu_b[has_book])))
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    mu_opt = best_alpha * mu_lgb + (1 - best_alpha) * mu_nn
    ev_3c = eval_predictions(mu_opt, sigma_nn, actual, book, has_book,
                              f"Blend({best_alpha:.2f})")
    sa_3c = _sigma_analysis(f"Blend({best_alpha:.2f})", mu_opt, sigma_nn, actual, has_book)
    ev_3c["spearman_rho"] = sa_3c["spearman_rho"]
    print(f"    Optimal alpha={best_alpha:.2f} (LGB weight)")
    print(f"    BS-MAE={ev_3c['bs_mae']:.3f}")

    # Print alpha sweep
    print(f"\n    {'Alpha':>6} {'BS-MAE':>7}")
    print(f"    {'-'*6} {'-'*7}")
    for alpha in np.arange(0.0, 1.01, 0.1):
        mu_b = alpha * mu_lgb + (1 - alpha) * mu_nn
        mae = float(np.mean(np.abs(actual[has_book] - mu_b[has_book])))
        flag = " ***" if abs(alpha - best_alpha) < 0.01 else ""
        print(f"    {alpha:>6.1f} {mae:>7.3f}{flag}")

    # 3D: Note about stacked ensemble
    print("\n  3D: Stacked ensemble — SKIPPED")
    print("    Both NN and LGB used val set for early stopping / Optuna tuning.")
    print("    Training a Ridge on their val predictions is a data leak.")
    print("    Blend weights will be validated via walk-forward in Phase 8.")

    # Store NN baseline sigma info for Phase 7
    ev_nn_sigma = {"spearman_rho": sa_nn["spearman_rho"]}

    return ev_3a, ev_3b, ev_3c, best_alpha, ev_nn_sigma


# ══════════════════════════════════════════════════════════════════════
# PHASE 4: FEATURE ENGINEERING FOR TREES
# ══════════════════════════════════════════════════════════════════════

def phase_4(X_train_raw, y_train, X_val_raw, y_val, book, has_book,
            best_lgb_params, base_bs_mae):
    print("\n" + "=" * 70)
    print("  PHASE 4: FEATURE ENGINEERING FOR TREES")
    print("=" * 70)

    feat_names = config.FEATURE_ORDER

    def get_idx(name):
        return feat_names.index(name)

    # Build engineered features
    def add_features(X):
        feats = [X]
        # Differences
        feats.append((X[:, get_idx("home_team_BARTHAG")] -
                       X[:, get_idx("away_team_BARTHAG")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_eff_fg_pct")] -
                       X[:, get_idx("away_eff_fg_pct")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_tov_rate")] -
                       X[:, get_idx("away_tov_rate")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_off_rebound_pct")] -
                       X[:, get_idx("away_off_rebound_pct")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_ft_rate")] -
                       X[:, get_idx("away_ft_rate")]).reshape(-1, 1))
        # Ratios
        feats.append((X[:, get_idx("home_team_BARTHAG")] /
                       (X[:, get_idx("away_team_BARTHAG")] + 1e-6)).reshape(-1, 1))
        feats.append((X[:, get_idx("home_eff_fg_pct")] /
                       (X[:, get_idx("away_eff_fg_pct")] + 1e-6)).reshape(-1, 1))
        # Matchup
        feats.append(np.abs(X[:, get_idx("home_team_BARTHAG")] -
                             X[:, get_idx("away_team_BARTHAG")]).reshape(-1, 1))
        feats.append(((X[:, get_idx("home_team_BARTHAG")] +
                        X[:, get_idx("away_team_BARTHAG")]) / 2).reshape(-1, 1))
        return np.hstack(feats)

    X_train_eng = add_features(X_train_raw)
    X_val_eng = add_features(X_val_raw)
    print(f"  Features: {X_train_raw.shape[1]} → {X_train_eng.shape[1]}")

    # Optuna tuning with engineered features (100 trials)
    print(f"  Running Optuna (100 trials)...")
    lgb_train = lgb.Dataset(X_train_eng, y_train,
                             params={"feature_pre_filter": False}, free_raw_data=False)
    lgb_val_ds = lgb.Dataset(X_val_eng, y_val, reference=lgb_train, free_raw_data=False)

    def objective(trial):
        p = {
            "objective": "regression_l1",
            "metric": "mae",
            "verbosity": -1,
            "n_jobs": -1,
            "seed": 42,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        cb = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        m = lgb.train(p, lgb_train, num_boost_round=5000,
                      valid_sets=[lgb_val_ds], callbacks=cb)
        pred = m.predict(X_val_eng)
        return float(np.mean(np.abs(y_val[has_book] - pred[has_book])))

    t0 = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    elapsed = time.time() - t0

    print(f"  Best BS-MAE with eng features: {study.best_value:.3f} [{elapsed:.0f}s]")
    print(f"  Without eng features: {base_bs_mae:.3f}")
    print(f"  Delta: {study.best_value - base_bs_mae:+.3f}")

    # Retrain best
    best_p = {**study.best_params, "objective": "regression_l1",
              "metric": "mae", "verbosity": -1, "n_jobs": -1, "seed": 42}
    cb = [lgb.early_stopping(50), lgb.log_evaluation(0)]
    model_eng = lgb.train(best_p, lgb_train, num_boost_round=5000,
                          valid_sets=[lgb_val_ds], callbacks=cb)
    mu_eng = model_eng.predict(X_val_eng)

    return model_eng, mu_eng, study.best_value, X_train_eng, X_val_eng


# ══════════════════════════════════════════════════════════════════════
# PHASE 5: TEMPORAL WEIGHTING
# ══════════════════════════════════════════════════════════════════════

def phase_5(X_train_raw, X_train_s, y_train, X_val_raw, X_val_s, y_val,
            book, has_book, df_train, best_lgb_params):
    print("\n" + "=" * 70)
    print("  PHASE 5: TEMPORAL WEIGHTING")
    print("=" * 70)

    # Get year for each training game
    dates = pd.to_datetime(df_train["startDate"], errors="coerce", utc=True)
    game_years = dates.dt.year.values
    max_year = game_years.max()

    decay_factors = [0.85, 0.90, 0.95, 1.0]

    # 5A: LightGBM with sample weights
    print("\n  5A: LightGBM with temporal sample weights")
    print(f"  {'Decay':>6} {'BS-MAE':>7}")
    print(f"  {'-'*6} {'-'*7}")

    for df in decay_factors:
        if df == 1.0:
            weights = np.ones(len(y_train))
        else:
            weights = df ** (max_year - game_years).astype(float)

        lgb_train = lgb.Dataset(X_train_raw, y_train, weight=weights,
                                 params={"feature_pre_filter": False}, free_raw_data=False)
        lgb_val_ds = lgb.Dataset(X_val_raw, y_val, reference=lgb_train, free_raw_data=False)
        p = {**best_lgb_params, "objective": "regression_l1",
             "metric": "mae", "verbosity": -1, "n_jobs": -1, "seed": 42,
             "feature_pre_filter": False}
        cb = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        m = lgb.train(p, lgb_train, num_boost_round=5000,
                      valid_sets=[lgb_val_ds], callbacks=cb)
        pred = m.predict(X_val_raw)
        mae = float(np.mean(np.abs(y_val[has_book] - pred[has_book])))
        print(f"  {df:>6.2f} {mae:>7.3f}")

    # 5B: Neural Net with weighted loss
    print("\n  5B: Neural Net with temporal loss weights")
    print(f"  {'Decay':>6} {'BS-MAE':>7} {'σ_std':>6} {'Dead':>5}")
    print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*5}")

    hp = {"hidden1": 384, "hidden2": 256, "dropout": 0.20,
          "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4}

    best_nn_weighted = None
    for df in decay_factors:
        if df == 1.0:
            weights = np.ones(len(y_train))
        else:
            weights = df ** (max_year - game_years).astype(float)

        model, best_ep = train_nn_weighted(X_train_s, y_train, X_val_s, y_val,
                                            hp, weights, verbose=False)
        mu, sigma = predict_nn(model, X_val_s)
        res = y_val - mu
        mae = float(np.mean(np.abs(res[has_book])))
        dead = count_dead(model, X_val_s)
        sig_std = float(np.std(sigma))
        print(f"  {df:>6.2f} {mae:>7.3f} {sig_std:>6.2f} {dead:>5}")
        if best_nn_weighted is None or mae < best_nn_weighted["mae"]:
            best_nn_weighted = {"decay": df, "mae": mae, "mu": mu, "sigma": sigma}

        del model
        torch.cuda.empty_cache()

    return best_nn_weighted


# ══════════════════════════════════════════════════════════════════════
# PHASE 6: LR WARMUP
# ══════════════════════════════════════════════════════════════════════

def phase_6(X_train_s, y_train, X_val_s, y_val, book, has_book):
    print("\n" + "=" * 70)
    print("  PHASE 6: LR WARMUP FOR NEURAL NET")
    print("=" * 70)

    hp = {"hidden1": 384, "hidden2": 256, "dropout": 0.20,
          "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4}

    print("  Training with 10-epoch warmup...")
    model, best_ep = train_nn(X_train_s, y_train, X_val_s, y_val,
                               hp=hp, verbose=True, warmup_epochs=10)
    mu, sigma = predict_nn(model, X_val_s)
    actual = y_val
    res = actual - mu
    bs_mae = float(np.mean(np.abs(res[has_book])))
    dead = count_dead(model, X_val_s)
    sig_std = float(np.std(sigma))
    cal = quintile_cal(sigma, res)
    print(f"\n  Warmup result: BS-MAE={bs_mae:.3f} σ_std={sig_std:.2f} "
          f"cal={cal:.3f} dead={dead} ep@{best_ep}")

    del model
    torch.cuda.empty_cache()

    return {"bs_mae": bs_mae, "sig_std": sig_std, "cal": cal, "dead": dead,
            "mu": mu, "sigma": sigma}


# ══════════════════════════════════════════════════════════════════════
# PHASE 7: FULL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════

def phase_7(all_results, y_val, sigma_nn_ref):
    """Print full comparison table with sigma usefulness column.

    sigma_nn_ref: the NN sigma array used for all models that don't produce
                  their own sigma (LGB, blends, etc.)
    """
    print("\n" + "=" * 70)
    print("  PHASE 7: FULL COMPARISON TABLE")
    print("=" * 70)

    # Compute Spearman(sigma, |resid|) for every model
    for r in all_results:
        if "spearman_rho" not in r:
            mu = r.get("mu")
            sigma = r.get("sigma")
            if mu is not None and sigma is not None:
                rho, _ = scipy_stats.spearmanr(sigma, np.abs(y_val - mu))
                r["spearman_rho"] = float(rho)
            elif mu is not None and sigma_nn_ref is not None:
                rho, _ = scipy_stats.spearmanr(sigma_nn_ref, np.abs(y_val - mu))
                r["spearman_rho"] = float(rho)

    print(f"\n  {'Model':>30} {'BS-MAE':>7} {'MAE':>7} {'σ_std':>6} {'Cal':>6} "
          f"{'Dead':>5} {'ROI@12%':>8} {'σ useful?':>10}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*6} {'-'*6} "
          f"{'-'*5} {'-'*8} {'-'*10}")

    for r in all_results:
        sig = f"{r['sig_std']:>6.2f}" if r.get("sig_std") else "   N/A"
        cal = f"{r['cal']:>6.3f}" if r.get("cal") else "   N/A"
        dead = f"{r['dead']:>5}" if r.get("dead") is not None else "  N/A"
        roi = ""
        if r.get("roi_12") and r["roi_12"]["bets"] > 0:
            roi = f"{r['roi_12']['roi']*100:>+7.1f}%"
        else:
            roi = "     N/A"
        overall = f"{r['overall_mae']:>7.3f}" if r.get("overall_mae") else "    N/A"
        spear = f"  ρ={r['spearman_rho']:+.3f}" if r.get("spearman_rho") is not None else "       N/A"
        flag = " ***" if r["bs_mae"] < 9.129 else ""
        print(f"  {r['label']:>30} {r['bs_mae']:>7.3f} {overall} "
              f"{sig} {cal} {dead} {roi} {spear}{flag}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 8: WALK-FORWARD THE WINNER
# ══════════════════════════════════════════════════════════════════════

def phase_8_walkforward(winner_type, winner_info):
    """Walk-forward the best approach if it beats 9.129."""
    print("\n" + "=" * 70)
    print(f"  PHASE 8: WALK-FORWARD — {winner_type}")
    print("=" * 70)

    if winner_type == "none":
        print("  No approach beat C2-V2. Skipping walk-forward.")
        return None

    test_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    if winner_type.startswith("lgb"):
        return _wf_lgb(test_years, winner_info)
    elif winner_type.startswith("blend"):
        return _wf_blend(test_years, winner_info)
    elif winner_type.startswith("ensemble"):
        return _wf_ensemble(test_years, winner_info)
    else:
        print(f"  Unknown winner type: {winner_type}")
        return None


def _wf_lgb(test_years, info):
    """Walk-forward for LightGBM."""
    lgb_params = info["params"]
    feat_eng = info.get("feat_eng", False)

    feat_names = config.FEATURE_ORDER

    def get_idx(name):
        return feat_names.index(name)

    def add_features(X):
        feats = [X]
        feats.append((X[:, get_idx("home_team_BARTHAG")] -
                       X[:, get_idx("away_team_BARTHAG")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_eff_fg_pct")] -
                       X[:, get_idx("away_eff_fg_pct")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_tov_rate")] -
                       X[:, get_idx("away_tov_rate")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_off_rebound_pct")] -
                       X[:, get_idx("away_off_rebound_pct")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_ft_rate")] -
                       X[:, get_idx("away_ft_rate")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_team_BARTHAG")] /
                       (X[:, get_idx("away_team_BARTHAG")] + 1e-6)).reshape(-1, 1))
        feats.append((X[:, get_idx("home_eff_fg_pct")] /
                       (X[:, get_idx("away_eff_fg_pct")] + 1e-6)).reshape(-1, 1))
        feats.append(np.abs(X[:, get_idx("home_team_BARTHAG")] -
                             X[:, get_idx("away_team_BARTHAG")]).reshape(-1, 1))
        feats.append(((X[:, get_idx("home_team_BARTHAG")] +
                        X[:, get_idx("away_team_BARTHAG")]) / 2).reshape(-1, 1))
        return np.hstack(feats)

    results = []
    all_mu, all_actual, all_book = [], [], []

    for ty in test_years:
        train_seasons = list(range(2015, ty))
        print(f"\n  --- {ty}: train on {train_seasons[0]}-{train_seasons[-1]} ---")

        df_tr = load_multi_season_features(
            train_seasons, adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
        df_tr = df_tr.dropna(subset=["homeScore", "awayScore"])
        df_tr = df_tr[(df_tr["homeScore"] != 0) | (df_tr["awayScore"] != 0)]

        df_v = load_multi_season_features(
            [ty], adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
        df_v = df_v.dropna(subset=["homeScore", "awayScore"])
        df_v = df_v[(df_v["homeScore"] != 0) | (df_v["awayScore"] != 0)]

        try:
            lines_df = load_research_lines(ty)
            if not lines_df.empty:
                ld = lines_df.sort_values("provider").drop_duplicates(
                    subset=["gameId"], keep="first")
                if "spread" in ld.columns:
                    df_v = df_v.merge(
                        ld[["gameId", "spread"]].rename(columns={"spread": "bookSpread"}),
                        on="gameId", how="left")
        except Exception:
            pass

        X_tr = get_feature_matrix(df_tr).values.astype(np.float32)
        y_tr = get_targets(df_tr)["spread_home"].values.astype(np.float32)
        X_v = get_feature_matrix(df_v).values.astype(np.float32)
        y_v = get_targets(df_v)["spread_home"].values.astype(np.float32)
        X_tr = impute_column_means(X_tr)
        X_v = impute_column_means(X_v)

        if feat_eng:
            X_tr = add_features(X_tr)
            X_v = add_features(X_v)

        book = df_v["bookSpread"].values.astype(np.float64) if "bookSpread" in df_v.columns else np.full(len(df_v), np.nan)
        has_book = ~np.isnan(book)

        lgb_tr = lgb.Dataset(X_tr, y_tr,
                              params={"feature_pre_filter": False}, free_raw_data=False)
        lgb_vd = lgb.Dataset(X_v, y_v, reference=lgb_tr, free_raw_data=False)
        p = {**lgb_params, "objective": "regression_l1",
             "metric": "mae", "verbosity": -1, "n_jobs": -1, "seed": 42,
             "feature_pre_filter": False}
        cb = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        m = lgb.train(p, lgb_tr, num_boost_round=5000,
                      valid_sets=[lgb_vd], callbacks=cb)
        mu = m.predict(X_v)

        bs_mae = float(np.mean(np.abs(y_v[has_book] - mu[has_book])))
        book_base = float(np.mean(np.abs(y_v[has_book] - (-book[has_book]))))

        # For ROI we need sigma — use constant 12.0
        sigma_const = np.full_like(mu, 12.0)
        roi_12 = compute_roi(mu, sigma_const, book, y_v, 0.12)

        results.append({"year": ty, "bs_mae": bs_mae, "book_base": book_base,
                        "roi_12": roi_12})
        all_mu.append(mu)
        all_actual.append(y_v)
        all_book.append(book)

        print(f"    BS-MAE={bs_mae:.3f} Book={book_base:.3f} "
              f"ROI@12%={roi_12['roi']*100:+.1f}% ({roi_12['bets']}b)")

    # Pooled
    p_mu = np.concatenate(all_mu)
    p_actual = np.concatenate(all_actual)
    p_book = np.concatenate(all_book)
    hb = ~np.isnan(p_book)
    pooled_mae = float(np.mean(np.abs(p_actual[hb] - p_mu[hb])))

    print(f"\n  Pooled BS-MAE: {pooled_mae:.3f}")

    return results, pooled_mae


def _wf_blend(test_years, info):
    """Walk-forward for blend (LGB + NN)."""
    lgb_params = info["params"]
    alpha = info["alpha"]
    feat_eng = info.get("feat_eng", False)

    feat_names = config.FEATURE_ORDER
    def get_idx(name):
        return feat_names.index(name)
    def add_features(X):
        feats = [X]
        feats.append((X[:, get_idx("home_team_BARTHAG")] - X[:, get_idx("away_team_BARTHAG")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_eff_fg_pct")] - X[:, get_idx("away_eff_fg_pct")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_tov_rate")] - X[:, get_idx("away_tov_rate")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_off_rebound_pct")] - X[:, get_idx("away_off_rebound_pct")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_ft_rate")] - X[:, get_idx("away_ft_rate")]).reshape(-1, 1))
        feats.append((X[:, get_idx("home_team_BARTHAG")] / (X[:, get_idx("away_team_BARTHAG")] + 1e-6)).reshape(-1, 1))
        feats.append((X[:, get_idx("home_eff_fg_pct")] / (X[:, get_idx("away_eff_fg_pct")] + 1e-6)).reshape(-1, 1))
        feats.append(np.abs(X[:, get_idx("home_team_BARTHAG")] - X[:, get_idx("away_team_BARTHAG")]).reshape(-1, 1))
        feats.append(((X[:, get_idx("home_team_BARTHAG")] + X[:, get_idx("away_team_BARTHAG")]) / 2).reshape(-1, 1))
        return np.hstack(feats)

    nn_hp = {"hidden1": 384, "hidden2": 256, "dropout": 0.20,
             "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4}

    results = []
    all_mu, all_actual, all_book = [], [], []

    for ty in test_years:
        train_seasons = list(range(2015, ty))
        print(f"\n  --- {ty}: train on {train_seasons[0]}-{train_seasons[-1]} ---")
        t0 = time.time()

        df_tr = load_multi_season_features(train_seasons, adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
        df_tr = df_tr.dropna(subset=["homeScore", "awayScore"])
        df_tr = df_tr[(df_tr["homeScore"] != 0) | (df_tr["awayScore"] != 0)]
        df_v = load_multi_season_features([ty], adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
        df_v = df_v.dropna(subset=["homeScore", "awayScore"])
        df_v = df_v[(df_v["homeScore"] != 0) | (df_v["awayScore"] != 0)]
        try:
            lines_df = load_research_lines(ty)
            if not lines_df.empty:
                ld = lines_df.sort_values("provider").drop_duplicates(subset=["gameId"], keep="first")
                if "spread" in ld.columns:
                    df_v = df_v.merge(ld[["gameId", "spread"]].rename(columns={"spread": "bookSpread"}), on="gameId", how="left")
        except Exception:
            pass

        X_tr_raw = get_feature_matrix(df_tr).values.astype(np.float32)
        y_tr = get_targets(df_tr)["spread_home"].values.astype(np.float32)
        X_v_raw = get_feature_matrix(df_v).values.astype(np.float32)
        y_v = get_targets(df_v)["spread_home"].values.astype(np.float32)
        X_tr_raw = impute_column_means(X_tr_raw)
        X_v_raw = impute_column_means(X_v_raw)

        book = df_v["bookSpread"].values.astype(np.float64) if "bookSpread" in df_v.columns else np.full(len(df_v), np.nan)
        has_book = ~np.isnan(book)

        # LGB
        X_tr_lgb = add_features(X_tr_raw) if feat_eng else X_tr_raw
        X_v_lgb = add_features(X_v_raw) if feat_eng else X_v_raw
        lgb_tr = lgb.Dataset(X_tr_lgb, y_tr,
                              params={"feature_pre_filter": False}, free_raw_data=False)
        lgb_vd = lgb.Dataset(X_v_lgb, y_v, reference=lgb_tr, free_raw_data=False)
        p = {**lgb_params, "objective": "regression_l1", "metric": "mae", "verbosity": -1,
             "n_jobs": -1, "seed": 42, "feature_pre_filter": False}
        cb = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        m_lgb = lgb.train(p, lgb_tr, num_boost_round=5000, valid_sets=[lgb_vd], callbacks=cb)
        mu_lgb = m_lgb.predict(X_v_lgb)

        # NN
        scaler = StandardScaler()
        scaler.fit(X_tr_raw)
        X_tr_s = scaler.transform(X_tr_raw).astype(np.float32)
        X_v_s = scaler.transform(X_v_raw).astype(np.float32)
        m_nn, _ = train_nn(X_tr_s, y_tr, X_v_s, y_v, hp=nn_hp, verbose=False)
        mu_nn, sigma_nn = predict_nn(m_nn, X_v_s)

        # Blend
        mu_blend = alpha * mu_lgb + (1 - alpha) * mu_nn
        bs_mae = float(np.mean(np.abs(y_v[has_book] - mu_blend[has_book])))
        roi_12 = compute_roi(mu_blend, sigma_nn, book, y_v, 0.12)

        elapsed = time.time() - t0
        results.append({"year": ty, "bs_mae": bs_mae, "roi_12": roi_12})
        all_mu.append(mu_blend)
        all_actual.append(y_v)
        all_book.append(book)

        print(f"    BS-MAE={bs_mae:.3f} ROI@12%={roi_12['roi']*100:+.1f}% [{elapsed:.0f}s]")

        del m_nn
        torch.cuda.empty_cache()

    p_mu = np.concatenate(all_mu)
    p_actual = np.concatenate(all_actual)
    p_book = np.concatenate(all_book)
    hb = ~np.isnan(p_book)
    pooled_mae = float(np.mean(np.abs(p_actual[hb] - p_mu[hb])))
    print(f"\n  Pooled BS-MAE: {pooled_mae:.3f}")

    return results, pooled_mae


def _wf_ensemble(test_years, info):
    """Walk-forward for NN ensemble."""
    configs = info["configs"]
    method = info.get("method", "mean")

    results = []
    all_mu, all_actual, all_book = [], [], []

    for ty in test_years:
        train_seasons = list(range(2015, ty))
        print(f"\n  --- {ty}: train on {train_seasons[0]}-{train_seasons[-1]} ---")
        t0 = time.time()

        df_tr = load_multi_season_features(train_seasons, adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
        df_tr = df_tr.dropna(subset=["homeScore", "awayScore"])
        df_tr = df_tr[(df_tr["homeScore"] != 0) | (df_tr["awayScore"] != 0)]
        df_v = load_multi_season_features([ty], adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
        df_v = df_v.dropna(subset=["homeScore", "awayScore"])
        df_v = df_v[(df_v["homeScore"] != 0) | (df_v["awayScore"] != 0)]
        try:
            lines_df = load_research_lines(ty)
            if not lines_df.empty:
                ld = lines_df.sort_values("provider").drop_duplicates(subset=["gameId"], keep="first")
                if "spread" in ld.columns:
                    df_v = df_v.merge(ld[["gameId", "spread"]].rename(columns={"spread": "bookSpread"}), on="gameId", how="left")
        except Exception:
            pass

        X_tr_raw = get_feature_matrix(df_tr).values.astype(np.float32)
        y_tr = get_targets(df_tr)["spread_home"].values.astype(np.float32)
        X_v_raw = get_feature_matrix(df_v).values.astype(np.float32)
        y_v = get_targets(df_v)["spread_home"].values.astype(np.float32)
        X_tr_raw = impute_column_means(X_tr_raw)
        X_v_raw = impute_column_means(X_v_raw)
        scaler = StandardScaler()
        scaler.fit(X_tr_raw)
        X_tr_s = scaler.transform(X_tr_raw).astype(np.float32)
        X_v_s = scaler.transform(X_v_raw).astype(np.float32)
        book = df_v["bookSpread"].values.astype(np.float64) if "bookSpread" in df_v.columns else np.full(len(df_v), np.nan)
        has_book = ~np.isnan(book)

        mus, sigmas = [], []
        for cfg in configs:
            hp = {k: v for k, v in cfg.items() if k not in ("name", "ref_mae")}
            m, _ = train_nn(X_tr_s, y_tr, X_v_s, y_v, hp=hp, verbose=False)
            mu_i, sig_i = predict_nn(m, X_v_s)
            mus.append(mu_i)
            sigmas.append(sig_i)
            del m
            torch.cuda.empty_cache()

        mu_ens = np.mean(mus, axis=0)
        sig_ens = np.mean(sigmas, axis=0)
        bs_mae = float(np.mean(np.abs(y_v[has_book] - mu_ens[has_book])))
        roi_12 = compute_roi(mu_ens, sig_ens, book, y_v, 0.12)

        elapsed = time.time() - t0
        results.append({"year": ty, "bs_mae": bs_mae, "roi_12": roi_12})
        all_mu.append(mu_ens)
        all_actual.append(y_v)
        all_book.append(book)
        print(f"    BS-MAE={bs_mae:.3f} ROI@12%={roi_12['roi']*100:+.1f}% [{elapsed:.0f}s]")

    p_mu = np.concatenate(all_mu)
    p_actual = np.concatenate(all_actual)
    p_book = np.concatenate(all_book)
    hb = ~np.isnan(p_book)
    pooled_mae = float(np.mean(np.abs(p_actual[hb] - p_mu[hb])))
    print(f"\n  Pooled BS-MAE: {pooled_mae:.3f}")
    return results, pooled_mae


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  SESSION 13B: ENSEMBLE + LIGHTGBM + FEATURE ENGINEERING")
    print("  Baseline: C2-V2 BS-MAE = 9.129")
    print("=" * 70)
    t_start = time.time()

    # Load data once
    (X_train_raw, X_train_s, y_train, X_val_raw, X_val_s, y_val,
     scaler, book, has_book, df_val, df_train) = load_data()

    # Load C2-V2 baseline predictions
    ckpt = torch.load(config.PROJECT_ROOT / "checkpoints" / "regressor.pt",
                      map_location="cpu", weights_only=False)
    hp_winner = ckpt["hparams"]
    model_winner = MLPRegressor(
        input_dim=50, hidden1=hp_winner["hidden1"],
        hidden2=hp_winner["hidden2"], dropout=hp_winner["dropout"])
    model_winner.load_state_dict(ckpt["state_dict"])
    model_winner.eval()
    mu_c2v2, sigma_c2v2 = predict_nn(model_winner, X_val_s)

    # Collect all results for Phase 7
    all_results = []

    # C2-V2 baseline
    ev_base = eval_predictions(mu_c2v2, sigma_c2v2, y_val, book, has_book, "C2-V2 (baseline)")
    ev_base["dead"] = 0
    all_results.append(ev_base)

    # ── PHASE 1: NN Ensemble ──
    models, indiv_results, ensembles, all_mu_nn, all_sigma_nn = phase_1(
        X_train_s, y_train, X_val_s, y_val, book, has_book)

    for ename, ev in ensembles.items():
        ev["dead"] = 0
        all_results.append(ev)

    # ── PHASE 2: LightGBM ──
    model_lgb_base, model_lgb_tuned, mu_lgb_base, mu_lgb_tuned, best_lgb_params, study = phase_2(
        X_train_raw, y_train, X_val_raw, y_val, book, has_book)

    ev_lgb_base = eval_predictions(mu_lgb_base, sigma_c2v2, y_val, book, has_book, "LGB baseline")
    ev_lgb_tuned = eval_predictions(mu_lgb_tuned, sigma_c2v2, y_val, book, has_book, "LGB tuned")
    all_results.extend([ev_lgb_base, ev_lgb_tuned])

    # ── PHASE 3: Hybrid ──
    ev_3a, ev_3b, ev_3c, best_alpha, ev_nn_sigma = phase_3(
        mu_lgb_tuned, mu_c2v2, sigma_c2v2, y_val, book, has_book, X_val_raw)
    all_results.extend([ev_3a, ev_3b, ev_3c])

    # ── PHASE 4: Feature Engineering ──
    model_eng, mu_eng, bs_mae_eng, X_train_eng, X_val_eng = phase_4(
        X_train_raw, y_train, X_val_raw, y_val, book, has_book,
        best_lgb_params, study.best_value)
    ev_eng = eval_predictions(mu_eng, sigma_c2v2, y_val, book, has_book, "LGB+feat eng")
    all_results.append(ev_eng)

    # Blend with engineered LGB
    mu_eng_blend = best_alpha * mu_eng + (1 - best_alpha) * mu_c2v2
    ev_eng_blend = eval_predictions(mu_eng_blend, sigma_c2v2, y_val, book, has_book,
                                     f"Blend eng ({best_alpha:.2f})")
    all_results.append(ev_eng_blend)

    # ── PHASE 5: Temporal Weighting ──
    best_nn_weighted = phase_5(X_train_raw, X_train_s, y_train, X_val_raw, X_val_s,
                                y_val, book, has_book, df_train, best_lgb_params)
    ev_tw = eval_predictions(best_nn_weighted["mu"], best_nn_weighted["sigma"],
                              y_val, book, has_book,
                              f"NN temporal (d={best_nn_weighted['decay']})")
    ev_tw["dead"] = 0
    all_results.append(ev_tw)

    # ── PHASE 6: LR Warmup ──
    warmup_res = phase_6(X_train_s, y_train, X_val_s, y_val, book, has_book)
    ev_warmup = eval_predictions(warmup_res["mu"], warmup_res["sigma"],
                                  y_val, book, has_book, "NN warmup")
    ev_warmup["dead"] = warmup_res["dead"]
    all_results.append(ev_warmup)

    # ── PHASE 7: Comparison Table ──
    phase_7(all_results, y_val, sigma_c2v2)

    # ── Determine winner ──
    best = min(all_results, key=lambda x: x["bs_mae"])
    print(f"\n  BEST: {best['label']} at BS-MAE={best['bs_mae']:.3f}")
    beats_baseline = best["bs_mae"] < 9.129

    # ── PHASE 8: Walk-Forward ──
    if beats_baseline:
        # Determine which type
        label = best["label"]
        if "LGB" in label or "lgb" in label.lower():
            if "eng" in label.lower():
                wf_info = {"params": best_lgb_params, "feat_eng": True}
                wf_type = "lgb_eng"
            else:
                wf_info = {"params": best_lgb_params, "feat_eng": False}
                wf_type = "lgb"
        elif "Blend" in label:
            if "eng" in label.lower():
                wf_info = {"params": best_lgb_params, "alpha": best_alpha, "feat_eng": True}
            else:
                wf_info = {"params": best_lgb_params, "alpha": best_alpha, "feat_eng": False}
            wf_type = "blend"
        elif "Mean" in label or "Top" in label or "Median" in label or "Weighted" in label:
            wf_info = {"configs": ENSEMBLE_CONFIGS, "method": "mean"}
            wf_type = "ensemble"
        else:
            wf_type = "none"
            wf_info = {}

        wf_result = phase_8_walkforward(wf_type, wf_info)
    else:
        print("  No approach beats C2-V2 on val set. Skipping walk-forward.")
        wf_result = None

    # ── PHASE 9: Save ──
    print("\n" + "=" * 70)
    print("  PHASE 9: SAVE RESULTS")
    print("=" * 70)

    # Save LGB model
    lgb_path = config.PROJECT_ROOT / "artifacts" / "lgb_model.txt"
    model_lgb_tuned.save_model(str(lgb_path))
    print(f"  Saved LGB model to {lgb_path}")

    # Save report
    report_lines = [
        "# Session 13b: Ensemble + LightGBM + Feature Engineering",
        f"\nDate: 2026-02-27",
        f"\n## Baseline: C2-V2 BS-MAE = {ev_base['bs_mae']:.3f}",
        f"\n## Best approach: {best['label']} BS-MAE = {best['bs_mae']:.3f}",
        f"\n## Full comparison:\n",
        f"| Model | BS-MAE | Overall MAE | σ_std | Cal | ROI@12% | σ useful? (ρ) |",
        f"|-------|--------|-------------|-------|-----|---------|---------------|",
    ]
    for r in all_results:
        sig = f"{r['sig_std']:.2f}" if r.get("sig_std") else "N/A"
        cal = f"{r['cal']:.3f}" if r.get("cal") else "N/A"
        roi = f"{r['roi_12']['roi']*100:+.1f}%" if r.get("roi_12") and r["roi_12"]["bets"] > 0 else "N/A"
        overall = f"{r['overall_mae']:.3f}" if r.get("overall_mae") else "N/A"
        spear = f"{r['spearman_rho']:+.3f}" if r.get("spearman_rho") is not None else "N/A"
        report_lines.append(f"| {r['label']} | {r['bs_mae']:.3f} | {overall} | {sig} | {cal} | {roi} | {spear} |")

    if wf_result:
        wf_data, pooled_mae = wf_result
        report_lines.append(f"\n## Walk-Forward Results")
        report_lines.append(f"\nPooled BS-MAE: {pooled_mae:.3f}")
        for yr in wf_data:
            report_lines.append(f"- {yr['year']}: BS-MAE={yr['bs_mae']:.3f}")
    else:
        report_lines.append(f"\n## Walk-Forward: Skipped (no approach beat baseline)")

    report_lines.append(f"\n## Verdict")

    # Production complexity analysis
    is_ensemble = any(k in best["label"].lower() for k in ("blend", "stack", "mean", "top", "weighted", "median", "hybrid", "lgb+nn"))
    margin = ev_base["bs_mae"] - best["bs_mae"]

    if beats_baseline:
        if wf_result:
            _, pooled = wf_result
            if pooled < ev_base["bs_mae"] + 0.05:
                report_lines.append(f"New winner: {best['label']} (val={best['bs_mae']:.3f}, WF pooled={pooled:.3f})")
            else:
                report_lines.append(f"{best['label']} beats val but not walk-forward. C2-V2 remains production model.")
        else:
            report_lines.append(f"{best['label']} beats val. Walk-forward pending.")
    else:
        report_lines.append(f"C2-V2 remains the best model at BS-MAE={ev_base['bs_mae']:.3f}.")
        report_lines.append(f"LGB saved as secondary signal at {lgb_path}.")

    report_lines.append(f"\n## Production Complexity Note")
    if is_ensemble and margin < 0.02:
        report_lines.append(
            f"The best approach ({best['label']}) is an ensemble that wins by only "
            f"{margin:.3f} MAE. Production trade-offs to consider:")
        report_lines.append("- Inference requires running multiple models daily")
        report_lines.append("- Feature pipelines must feed all component models")
        report_lines.append("- Blend weight was optimized on a single val split")
        report_lines.append("- If any component model's artifacts are corrupted, predictions break")
        report_lines.append(f"A pure single model within 0.02 MAE of the blend is "
                            f"likely better for production reliability.")
    elif is_ensemble:
        report_lines.append(
            f"The best approach ({best['label']}) is an ensemble winning by {margin:.3f} MAE. "
            f"This is a meaningful margin. Ensemble complexity may be justified, "
            f"but monitor for drift in component models.")
    else:
        report_lines.append(f"The best approach ({best['label']}) is a single model — "
                            f"no ensemble complexity concerns.")

    report_path = REPORTS_DIR / "session13b_ensemble_report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"  Saved report to {report_path}")

    # Print final verdict with complexity note
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL PHASES COMPLETE — {elapsed/60:.1f} minutes")
    if beats_baseline:
        print(f"  BEST: {best['label']} BS-MAE={best['bs_mae']:.3f} "
              f"(beats baseline by {margin:.3f})")
        if is_ensemble and margin < 0.02:
            print(f"  NOTE: Margin is <0.02 and model is an ensemble.")
            print(f"  C2-V2 (single model) likely better for production.")
        print(f"  VERDICT: {'NEW WINNER — walk-forward needed' if not wf_result else 'SEE WALK-FORWARD RESULTS'}")
    else:
        print(f"  VERDICT: C2-V2 REMAINS CHAMPION")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

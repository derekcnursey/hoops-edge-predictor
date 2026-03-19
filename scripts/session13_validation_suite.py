#!/usr/bin/env python3
"""Session 13: Comprehensive 12-test validation suite.

Tests the C2-V2 winner (384→256, d=0.20, lr=3e-3, batch=4096, Gaussian)
with walk-forward, sigma ablation, vig sensitivity, bootstrap CI,
drawdown, correlation, feature ablation, HP stability, calibration
slices, baselines, line staleness, and bet profiling.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import sys
import time
from collections import defaultdict
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
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets, load_research_lines
from src.trainer import impute_column_means

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
MAX_EPOCHS = 500
PATIENCE = 50

# C2-V2 winner hyperparameters
WINNER_HP = {
    "hidden1": 384, "hidden2": 256, "dropout": 0.20,
    "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4,
}


# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════

def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_season_data(train_seasons, val_seasons):
    """Load train/val data and return scaled arrays + df_val."""
    df_train = load_multi_season_features(
        train_seasons, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_train = df_train.dropna(subset=["homeScore", "awayScore"])
    df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]

    df_val = load_multi_season_features(
        val_seasons, adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_val = df_val.dropna(subset=["homeScore", "awayScore"])
    df_val = df_val[(df_val["homeScore"] != 0) | (df_val["awayScore"] != 0)]

    # Merge book spreads for val
    for vs in val_seasons:
        try:
            lines_df = load_research_lines(vs)
            if not lines_df.empty:
                ld = lines_df.sort_values("provider").drop_duplicates(
                    subset=["gameId"], keep="first")
                if "spread" in ld.columns:
                    merge_df = ld[["gameId", "spread"]].rename(
                        columns={"spread": "bookSpread"})
                    df_val = df_val.merge(merge_df, on="gameId", how="left")
        except Exception:
            pass

    X_train = get_feature_matrix(df_train).values.astype(np.float32)
    targets_train = get_targets(df_train)
    y_train = targets_train["spread_home"].values.astype(np.float32)

    X_val = get_feature_matrix(df_val).values.astype(np.float32)
    targets_val = get_targets(df_val)
    y_val = targets_val["spread_home"].values.astype(np.float32)

    X_train = impute_column_means(X_train)
    X_val = impute_column_means(X_val)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    return X_train_s, y_train, X_val_s, y_val, scaler, df_val


def train_model(X_train, y_train, X_val_s, y_val, hp=None, verbose=True):
    """Train C2-V2 architecture. Returns (model, best_epoch)."""
    if hp is None:
        hp = WINNER_HP
    device = get_device()
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
            if verbose:
                print(f"      Early stop at {ep} (best@{best_epoch})")
            break

    model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


@torch.no_grad()
def _val_loss(model, X_val_t, y_val_t, device):
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
def predict(model, X_val_s):
    """Return (mu, sigma) numpy arrays."""
    X_t = torch.tensor(X_val_s, dtype=torch.float32)
    mu_t, ls_t = model(X_t)
    sigma_t = torch.exp(ls_t).clamp(min=0.5, max=30.0)
    return mu_t.numpy(), sigma_t.numpy()


def count_dead(model, X_val_s):
    """Count dead neurons (>99% zero activation on val set)."""
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
            model(X_t[s:min(s+4096, len(X_t))])
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


def compute_roi(mu, sigma, book_spread, actual, threshold,
                sigma_lo=None, sigma_hi=None):
    """Compute ATS ROI. Returns dict with bets, wins, win_rate, roi, units."""
    valid = ~np.isnan(book_spread)
    if valid.sum() == 0:
        return {"bets": 0, "wins": 0, "win_rate": 0.0, "roi": 0.0, "units": 0.0}

    edge_home = mu[valid] + book_spread[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    home_cover_prob = normal_cdf(edge_z)
    away_cover_prob = 1.0 - home_cover_prob
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, home_cover_prob, away_cover_prob)
    breakeven = 0.5238
    prob_edge = pick_prob - breakeven

    bet_mask = prob_edge >= threshold
    sigma_v = sigma[valid]
    if sigma_lo is not None:
        bet_mask = bet_mask & (sigma_v >= sigma_lo)
    if sigma_hi is not None:
        bet_mask = bet_mask & (sigma_v <= sigma_hi)

    n = bet_mask.sum()
    if n == 0:
        return {"bets": 0, "wins": 0, "win_rate": 0.0, "roi": 0.0, "units": 0.0}

    actual_v = actual[valid]
    book_v = book_spread[valid]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)
    w = pick_won[bet_mask].sum()
    profit_per_1 = 100.0 / 110.0
    units = w * profit_per_1 - (n - w)
    return {"bets": int(n), "wins": int(w), "win_rate": float(w/n),
            "roi": float(units/n), "units": float(units)}


def compute_roi_custom_vig(mu, sigma, book_spread, actual, threshold,
                           breakeven, payout, sigma_lo=None, sigma_hi=None):
    """ROI with custom vig parameters."""
    valid = ~np.isnan(book_spread)
    if valid.sum() == 0:
        return {"bets": 0, "roi": 0.0, "units": 0.0}

    edge_home = mu[valid] + book_spread[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    home_cover_prob = normal_cdf(edge_z)
    away_cover_prob = 1.0 - home_cover_prob
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, home_cover_prob, away_cover_prob)
    prob_edge = pick_prob - breakeven

    bet_mask = prob_edge >= threshold
    sigma_v = sigma[valid]
    if sigma_lo is not None:
        bet_mask = bet_mask & (sigma_v >= sigma_lo)
    if sigma_hi is not None:
        bet_mask = bet_mask & (sigma_v <= sigma_hi)

    n = bet_mask.sum()
    if n == 0:
        return {"bets": 0, "roi": 0.0, "units": 0.0}

    actual_v = actual[valid]
    book_v = book_spread[valid]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)
    w = pick_won[bet_mask].sum()
    units = w * payout - (n - w)
    return {"bets": int(n), "roi": float(units/n), "units": float(units)}


def quintile_cal(sigma, residuals):
    """Return quintile calibration ratios and cal_score."""
    qi = np.array_split(np.argsort(sigma), 5)
    ratios = []
    for idx in qi:
        actual_std = np.std(residuals[idx])
        pred_mean = np.mean(sigma[idx])
        ratios.append(float(actual_std / pred_mean) if pred_mean > 0 else 999.0)
    cal_score = float(np.mean([abs(r - 1.0) for r in ratios]))
    return ratios, cal_score


def load_winner_model():
    """Load saved C2-V2 winner checkpoint."""
    ckpt = torch.load(config.PROJECT_ROOT / "checkpoints" / "regressor.pt",
                      map_location="cpu", weights_only=False)
    hp = ckpt["hparams"]
    model = MLPRegressor(
        input_dim=50, hidden1=hp["hidden1"], hidden2=hp["hidden2"],
        dropout=hp["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_val_data():
    """Load 2026 val data with book spreads."""
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

    with open(config.PROJECT_ROOT / "artifacts" / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    X_val = get_feature_matrix(df_val).values.astype(np.float32)
    targets_val = get_targets(df_val)
    y_val = targets_val["spread_home"].values.astype(np.float32)
    X_val = impute_column_means(X_val)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    book = df_val["bookSpread"].values.astype(np.float64) if "bookSpread" in df_val.columns else np.full(len(df_val), np.nan)
    return X_val_s, y_val, book, df_val


# ══════════════════════════════════════════════════════════════════════
# STRATEGIES (used across multiple tests)
# ══════════════════════════════════════════════════════════════════════

STRATEGIES = [
    ("a) edge>=10%",       0.10, None, None),
    ("b) edge>=12%",       0.12, None, None),
    ("c) edge>=12% σ12-16", 0.12, 12.0, 16.0),
    ("d) edge>=15%",       0.15, None, None),
    ("e) edge>=7% σ>14",   0.07, 14.0, None),
]


def run_all_strategies(mu, sigma, book, actual):
    """Run all 5 strategies, return list of dicts."""
    results = []
    for name, thresh, slo, shi in STRATEGIES:
        r = compute_roi(mu, sigma, book, actual, thresh,
                        sigma_lo=slo, sigma_hi=shi)
        r["name"] = name
        results.append(r)
    return results


def print_strategy_table(results, label=""):
    """Print strategy results table."""
    if label:
        print(f"  {label}")
    print(f"  {'Strategy':>22} {'Bets':>6} {'W':>4} {'Win%':>7} {'ROI':>8} {'Units':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*4} {'-'*7} {'-'*8} {'-'*8}")
    for r in results:
        if r["bets"] > 0:
            print(f"  {r['name']:>22} {r['bets']:>6} {r['wins']:>4} "
                  f"{r['win_rate']*100:>6.1f}% {r['roi']*100:>+7.1f}% "
                  f"{r['units']:>+8.1f}")
        else:
            print(f"  {r['name']:>22}      0    -     ---      ---      ---")


# ══════════════════════════════════════════════════════════════════════
# TEST 1: WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════

def test_1_walk_forward():
    print("\n" + "=" * 70)
    print("  TEST 1: WALK-FORWARD VALIDATION")
    print("=" * 70)
    print("  Training on past only, predicting each year independently.\n")

    test_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    all_year_results = []

    # Collect pooled predictions
    pooled_mu, pooled_sigma, pooled_actual, pooled_book = [], [], [], []

    for ty in test_years:
        train_seasons = list(range(2015, ty))
        print(f"  --- {ty}: train on {train_seasons[0]}-{train_seasons[-1]} "
              f"({len(train_seasons)} seasons) ---")

        t0 = time.time()
        X_tr, y_tr, X_v, y_v, scaler, df_v = load_season_data(
            train_seasons, [ty])

        n_games = len(df_v)
        n_book = df_v["bookSpread"].notna().sum() if "bookSpread" in df_v.columns else 0
        print(f"    Games: {n_games} total, {n_book} with book spreads")

        model, best_ep = train_model(X_tr, y_tr, X_v, y_v, verbose=False)
        mu, sigma = predict(model, X_v)
        actual = y_v
        book = df_v["bookSpread"].values.astype(np.float64) if "bookSpread" in df_v.columns else np.full(n_games, np.nan)
        has_book = ~np.isnan(book)

        # Metrics
        residuals = actual - mu
        abs_res = np.abs(residuals)
        bs_mae = float(np.mean(abs_res[has_book])) if has_book.sum() > 0 else None
        book_base = float(np.mean(np.abs(actual[has_book] - (-book[has_book])))) if has_book.sum() > 0 else None
        dead = count_dead(model, X_v)
        qr, cal_sc = quintile_cal(sigma, residuals)

        strat_results = run_all_strategies(mu, sigma, book, actual)

        elapsed = time.time() - t0

        yr = {
            "year": ty, "n_games": n_games, "n_book": n_book,
            "bs_mae": bs_mae, "book_base": book_base,
            "sigma_mean": float(np.mean(sigma)), "sigma_std": float(np.std(sigma)),
            "sigma_p25": float(np.percentile(sigma, 25)),
            "sigma_med": float(np.median(sigma)),
            "cal_score": cal_sc, "dead": dead,
            "best_epoch": best_ep,
            "strategies": strat_results, "elapsed": elapsed,
        }
        all_year_results.append(yr)

        # Accumulate pooled
        pooled_mu.append(mu)
        pooled_sigma.append(sigma)
        pooled_actual.append(actual)
        pooled_book.append(book)

        # Print summary line
        sc = strat_results[2]  # strategy c
        flag = " *** FLAG: c < -5% ***" if sc["bets"] > 0 and sc["roi"] < -0.05 else ""
        print(f"    BS-MAE={bs_mae:.3f} Book={book_base:.3f} σ={float(np.mean(sigma)):.1f}±{float(np.std(sigma)):.1f} "
              f"dead={dead} cal={cal_sc:.3f} ep@{best_ep} [{elapsed:.0f}s]")
        print_strategy_table(strat_results)
        if flag:
            print(f"    {flag}")
        print()

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Summary table
    print("\n  ── Walk-Forward Summary Table ──")
    print(f"  {'Year':>6} {'BS-MAE':>7} {'Book':>7} {'σ_std':>6} {'Cal':>6} {'Dead':>5} "
          f"{'c)Bets':>7} {'c)ROI':>8} {'a)ROI':>8} {'b)ROI':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5} "
          f"{'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    for yr in all_year_results:
        sc = yr["strategies"][2]  # c
        sa = yr["strategies"][0]  # a
        sb = yr["strategies"][1]  # b
        flag = " ***" if sc["bets"] > 0 and sc["roi"] < -0.05 else ""
        print(f"  {yr['year']:>6} {yr['bs_mae']:>7.3f} {yr['book_base']:>7.3f} "
              f"{yr['sigma_std']:>6.2f} {yr['cal_score']:>6.3f} {yr['dead']:>5} "
              f"{sc['bets']:>7} {sc['roi']*100:>+7.1f}% "
              f"{sa['roi']*100:>+7.1f}% {sb['roi']*100:>+7.1f}%{flag}")

    # Pooled
    print("\n  ── Pooled Across All Years ──")
    p_mu = np.concatenate(pooled_mu)
    p_sigma = np.concatenate(pooled_sigma)
    p_actual = np.concatenate(pooled_actual)
    p_book = np.concatenate(pooled_book)
    has_b = ~np.isnan(p_book)
    p_res = p_actual - p_mu
    bs_mae_pool = float(np.mean(np.abs(p_res[has_b])))
    book_base_pool = float(np.mean(np.abs(p_actual[has_b] - (-p_book[has_b]))))
    print(f"  Total games: {len(p_mu)}, with book: {has_b.sum()}")
    print(f"  Pooled BS-MAE: {bs_mae_pool:.3f}, Book baseline: {book_base_pool:.3f}")
    pooled_strats = run_all_strategies(p_mu, p_sigma, p_book, p_actual)
    print_strategy_table(pooled_strats)

    return all_year_results


# ══════════════════════════════════════════════════════════════════════
# TEST 2: IS SIGMA ACTUALLY USEFUL?
# ══════════════════════════════════════════════════════════════════════

def test_2_sigma_ablation(model, X_val_s, y_val, book, df_val):
    print("\n" + "=" * 70)
    print("  TEST 2: IS SIGMA ACTUALLY USEFUL? (ablation)")
    print("=" * 70)

    mu, sigma = predict(model, X_val_s)
    actual = y_val

    # A) Constant sigma baseline
    print("\n  A) CONSTANT SIGMA BASELINE (σ=12.0 for all games)")
    const_sigma = np.full_like(sigma, 12.0)
    real_results = run_all_strategies(mu, sigma, book, actual)
    const_results = run_all_strategies(mu, const_sigma, book, actual)

    print(f"\n  {'Strategy':>22} {'Real σ ROI':>10} {'Const σ ROI':>12} {'Δ':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*8}")
    for r, c in zip(real_results, const_results):
        if r["bets"] > 0 and c["bets"] > 0:
            delta = r["roi"] - c["roi"]
            print(f"  {r['name']:>22} {r['roi']*100:>+9.1f}% {c['roi']*100:>+11.1f}% "
                  f"{delta*100:>+7.1f}%")
        else:
            print(f"  {r['name']:>22} {'N/A':>10} {'N/A':>12} {'N/A':>8}")

    # B) Shuffled sigma baseline (100 shuffles)
    print("\n  B) SHUFFLED SIGMA BASELINE (100 shuffles)")
    n_shuffles = 100
    shuffle_rois = {name: [] for name, _, _, _ in STRATEGIES}
    for _ in range(n_shuffles):
        shuffled_sigma = sigma.copy()
        np.random.shuffle(shuffled_sigma)
        for (name, thresh, slo, shi) in STRATEGIES:
            r = compute_roi(mu, shuffled_sigma, book, actual, thresh,
                            sigma_lo=slo, sigma_hi=shi)
            shuffle_rois[name].append(r["roi"])

    print(f"\n  {'Strategy':>22} {'Real ROI':>9} {'Shuf Mean':>10} {'Shuf Std':>9} {'Δ':>8}")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*9} {'-'*8}")
    for r in real_results:
        sr = shuffle_rois[r["name"]]
        smean = np.mean(sr)
        sstd = np.std(sr)
        delta = r["roi"] - smean
        print(f"  {r['name']:>22} {r['roi']*100:>+8.1f}% {smean*100:>+9.1f}% "
              f"{sstd*100:>8.1f}% {delta*100:>+7.1f}%")

    # C) Sigma value when model agrees/disagrees with book
    print("\n  C) SIGMA VALUE: MODEL AGREES vs DISAGREES WITH BOOK")
    has_book = ~np.isnan(book)
    if has_book.sum() > 0:
        # Model picks: mu > 0 → home, mu < 0 → away
        # Book picks: book_spread < 0 → home favored, > 0 → away favored
        model_favors_home = mu[has_book] > 0
        book_favors_home = book[has_book] < 0  # negative spread = home favored
        agrees = model_favors_home == book_favors_home

        for group, mask, label in [
            ("Agrees", agrees, "Model agrees with book"),
            ("Disagrees", ~agrees, "Model disagrees with book"),
        ]:
            n_group = mask.sum()
            # Create full-length arrays for the group
            full_mask = np.zeros(len(mu), dtype=bool)
            book_idx = np.where(has_book)[0]
            full_mask[book_idx[mask]] = True

            mu_g = mu[full_mask]
            sigma_g = sigma[full_mask]
            book_g = book[full_mask]
            actual_g = actual[full_mask]
            # Make nan-free book for this subset
            book_full = np.full(len(mu_g), np.nan)
            book_full[:] = book_g

            r_unfilt = compute_roi(mu_g, sigma_g, book_full, actual_g, 0.12)
            r_sig = compute_roi(mu_g, sigma_g, book_full, actual_g, 0.12,
                                sigma_lo=12.0, sigma_hi=16.0)
            print(f"    {label}: {n_group} games")
            print(f"      edge>=12% unfilt: {r_unfilt['bets']} bets, "
                  f"ROI={r_unfilt['roi']*100:+.1f}%")
            print(f"      edge>=12% σ12-16: {r_sig['bets']} bets, "
                  f"ROI={r_sig['roi']*100:+.1f}%")


# ══════════════════════════════════════════════════════════════════════
# TEST 3: VIG SENSITIVITY
# ══════════════════════════════════════════════════════════════════════

def test_3_vig_sensitivity(model, X_val_s, y_val, book):
    print("\n" + "=" * 70)
    print("  TEST 3: VIG SENSITIVITY")
    print("=" * 70)

    mu, sigma = predict(model, X_val_s)
    actual = y_val

    vig_levels = [
        ("-108 (reduced)",  108, 100.0/108.0, 108.0/208.0),
        ("-110 (standard)", 110, 100.0/110.0, 110.0/210.0),
        ("-112 (worse)",    112, 100.0/112.0, 112.0/212.0),
        ("-115 (worst)",    115, 100.0/115.0, 115.0/215.0),
    ]

    # Top 3 strategies to test
    top_strats = [
        ("edge>=12% unfilt", 0.12, None, None),
        ("edge>=12% σ12-16", 0.12, 12.0, 16.0),
        ("edge>=15% unfilt", 0.15, None, None),
    ]

    print(f"\n  {'Vig':>17} {'Break-even':>11} ", end="")
    for sn, _, _, _ in top_strats:
        print(f"{'|':>2} {sn:>18}", end="")
    print()
    print(f"  {'-'*17} {'-'*11} ", end="")
    for _ in top_strats:
        print(f"  {'-'*18}", end="")
    print()

    for vig_name, vig, payout, be in vig_levels:
        print(f"  {vig_name:>17} {be*100:>10.1f}% ", end="")
        for sn, thresh, slo, shi in top_strats:
            r = compute_roi_custom_vig(mu, sigma, book, actual, thresh,
                                       be, payout, sigma_lo=slo, sigma_hi=shi)
            print(f"  {r['bets']:>4}b {r['roi']*100:>+6.1f}%", end="")
        print()


# ══════════════════════════════════════════════════════════════════════
# TEST 4: BOOTSTRAP CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════

def test_4_bootstrap_ci(model, X_val_s, y_val, book):
    print("\n" + "=" * 70)
    print("  TEST 4: BOOTSTRAP CONFIDENCE INTERVALS (2000 resamples)")
    print("=" * 70)

    mu, sigma = predict(model, X_val_s)
    actual = y_val
    has_book = ~np.isnan(book)
    n_boot = 2000

    for name, thresh, slo, shi in STRATEGIES:
        # Get bet-level outcomes
        valid = has_book.copy()
        edge_home = mu[valid] + book[valid]
        sigma_safe = np.clip(sigma[valid], 0.5, None)
        edge_z = edge_home / sigma_safe
        hcp = normal_cdf(edge_z)
        pick_home = edge_home >= 0
        pick_prob = np.where(pick_home, hcp, 1.0 - hcp)
        prob_edge = pick_prob - 0.5238
        sigma_v = sigma[valid]

        bet_mask = prob_edge >= thresh
        if slo is not None:
            bet_mask = bet_mask & (sigma_v >= slo)
        if shi is not None:
            bet_mask = bet_mask & (sigma_v <= shi)

        if bet_mask.sum() < 10:
            print(f"\n  {name}: Too few bets ({bet_mask.sum()}), skipping bootstrap")
            continue

        actual_v = actual[valid]
        book_v = book[valid]
        home_covered = (actual_v + book_v) > 0
        pick_won = np.where(pick_home, home_covered, ~home_covered)

        # Extract bet outcomes (1=win, 0=loss)
        outcomes = pick_won[bet_mask].astype(float)
        n_bets = len(outcomes)
        profit_per_1 = 100.0 / 110.0

        boot_rois = []
        for _ in range(n_boot):
            idx = np.random.randint(0, n_bets, n_bets)
            sample = outcomes[idx]
            w = sample.sum()
            l = n_bets - w
            units = w * profit_per_1 - l
            boot_rois.append(units / n_bets)

        boot_rois = np.array(boot_rois)
        mean_roi = np.mean(boot_rois)
        med_roi = np.median(boot_rois)
        p5 = np.percentile(boot_rois, 5)
        p95 = np.percentile(boot_rois, 95)
        prob_pos = (boot_rois > 0).mean()
        prob_5pct = (boot_rois > 0.05).mean()

        # P(losing 20+ units in 300 bets)
        boot_units_300 = []
        for _ in range(n_boot):
            idx = np.random.randint(0, n_bets, min(300, n_bets))
            sample = outcomes[idx]
            w = sample.sum()
            l = len(idx) - w
            boot_units_300.append(w * profit_per_1 - l)
        prob_loss_20 = (np.array(boot_units_300) < -20).mean()

        print(f"\n  {name} ({n_bets} bets):")
        print(f"    Mean ROI: {mean_roi*100:+.1f}%  Median: {med_roi*100:+.1f}%")
        print(f"    90% CI: [{p5*100:+.1f}%, {p95*100:+.1f}%]")
        print(f"    P(ROI > 0%): {prob_pos*100:.1f}%  P(ROI > 5%): {prob_5pct*100:.1f}%")
        print(f"    P(losing 20+ units in ~300 bets): {prob_loss_20*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# TEST 5: DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def test_5_drawdown(model, X_val_s, y_val, book, df_val):
    print("\n" + "=" * 70)
    print("  TEST 5: DRAWDOWN ANALYSIS (12% edge + σ 12-16)")
    print("=" * 70)

    mu, sigma = predict(model, X_val_s)
    actual = y_val
    has_book = ~np.isnan(book)

    # Get bets in chronological order
    valid = has_book.copy()
    edge_home = mu[valid] + book[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    hcp = normal_cdf(edge_z)
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, hcp, 1.0 - hcp)
    prob_edge = pick_prob - 0.5238
    sigma_v = sigma[valid]

    bet_mask = (prob_edge >= 0.12) & (sigma_v >= 12.0) & (sigma_v <= 16.0)

    actual_v = actual[valid]
    book_v = book[valid]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)

    bet_indices = np.where(bet_mask)[0]
    outcomes = pick_won[bet_mask]
    n_bets = len(outcomes)

    # Get dates for chronological ordering
    dates_valid = None
    if "startDate" in df_val.columns:
        all_dates = pd.to_datetime(df_val["startDate"], errors="coerce", utc=True)
        dates_valid = all_dates.iloc[np.where(valid)[0]].values

    print(f"\n  Total bets: {n_bets}")
    profit_per_1 = 100.0 / 110.0

    # Simulate flat 1-unit bets
    pnl = []
    running = 0.0
    for i, won in enumerate(outcomes):
        if won:
            running += profit_per_1
        else:
            running -= 1.0
        pnl.append(running)

    pnl = np.array(pnl)

    # Max drawdown
    peak = np.maximum.accumulate(pnl)
    drawdown = peak - pnl
    max_dd = float(np.max(drawdown))
    max_dd_idx = int(np.argmax(drawdown))

    # Find peak before max drawdown
    peak_at_dd = float(peak[max_dd_idx])

    # Longest losing streak
    max_streak = 0
    curr_streak = 0
    for won in outcomes:
        if not won:
            curr_streak += 1
            max_streak = max(max_streak, curr_streak)
        else:
            curr_streak = 0

    # Longest winning streak
    max_win_streak = 0
    curr_win = 0
    for won in outcomes:
        if won:
            curr_win += 1
            max_win_streak = max(max_win_streak, curr_win)
        else:
            curr_win = 0

    # Max bets without new high
    max_no_high = 0
    curr_no_high = 0
    running_peak = pnl[0]
    for i, p in enumerate(pnl):
        if p > running_peak:
            running_peak = p
            max_no_high = max(max_no_high, curr_no_high)
            curr_no_high = 0
        else:
            curr_no_high += 1
    max_no_high = max(max_no_high, curr_no_high)

    print(f"  Final P&L: {pnl[-1]:+.1f} units")
    print(f"  Max drawdown: {max_dd:.1f} units (from peak of {peak_at_dd:.1f})")
    print(f"  Longest losing streak: {max_streak}")
    print(f"  Longest winning streak: {max_win_streak}")
    print(f"  Max bets without new high: {max_no_high}")

    # Print P&L curve (every ~30 bets)
    print(f"\n  P&L Curve (every ~30 bets):")
    print(f"  {'Bet#':>6} {'P&L':>8} {'W-L':>8} {'Win%':>7}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*7}")
    step = max(1, n_bets // 10)
    for i in range(0, n_bets, step):
        w = outcomes[:i+1].sum()
        l = (i+1) - w
        wr = w / (i+1)
        print(f"  {i+1:>6} {pnl[i]:>+8.1f} {int(w):>3}-{int(l):<4} {wr*100:>6.1f}%")
    # Final
    w = outcomes.sum()
    l = n_bets - w
    print(f"  {n_bets:>6} {pnl[-1]:>+8.1f} {int(w):>3}-{int(l):<4} {w/n_bets*100:>6.1f}%")


# ══════════════════════════════════════════════════════════════════════
# TEST 6: BET CORRELATION / CLUSTERING
# ══════════════════════════════════════════════════════════════════════

def test_6_bet_correlation(model, X_val_s, y_val, book, df_val):
    print("\n" + "=" * 70)
    print("  TEST 6: BET CORRELATION / CLUSTERING")
    print("=" * 70)

    mu, sigma = predict(model, X_val_s)
    actual = y_val
    has_book = ~np.isnan(book)

    # Get bet details with dates
    valid = has_book.copy()
    edge_home = mu[valid] + book[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    hcp = normal_cdf(edge_z)
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, hcp, 1.0 - hcp)
    prob_edge = pick_prob - 0.5238
    sigma_v = sigma[valid]

    bet_mask = (prob_edge >= 0.12) & (sigma_v >= 12.0) & (sigma_v <= 16.0)

    actual_v = actual[valid]
    book_v = book[valid]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)

    outcomes = pick_won[bet_mask]
    n_bets = len(outcomes)

    # Get dates
    valid_idx = np.where(valid)[0]
    bet_idx_in_df = valid_idx[np.where(bet_mask)[0]]

    dates = pd.to_datetime(df_val.iloc[bet_idx_in_df]["startDate"],
                           errors="coerce", utc=True).dt.date.values

    # Bets per day
    date_counts = pd.Series(dates).value_counts()
    n_days = len(date_counts)
    avg_per_day = n_bets / n_days
    min_per_day = date_counts.min()
    max_per_day = date_counts.max()

    print(f"\n  Total bets: {n_bets} across {n_days} days")
    print(f"  Bets per day: avg={avg_per_day:.1f}, min={min_per_day}, max={max_per_day}")

    # Bets on same day as another bet
    multi_day_mask = date_counts > 1
    n_multi_day_bets = date_counts[multi_day_mask].sum()
    print(f"  Bets on multi-bet days: {n_multi_day_bets}/{n_bets} ({n_multi_day_bets/n_bets*100:.1f}%)")

    # On days with 5+ bets, correlation of outcomes
    bet_df = pd.DataFrame({
        "date": dates,
        "outcome": outcomes,
        "idx": range(n_bets),
    })

    days_5plus = date_counts[date_counts >= 5].index.tolist()
    if days_5plus:
        print(f"\n  Days with 5+ bets: {len(days_5plus)}")
        # Compute mean win rate per day and see if outcomes cluster
        day_results = []
        for d in days_5plus:
            day_bets = bet_df[bet_df["date"] == d]
            n_day = len(day_bets)
            w = day_bets["outcome"].sum()
            day_results.append({"date": d, "n": n_day, "wins": w, "wr": w/n_day})

        day_df = pd.DataFrame(day_results)
        # If outcomes are independent, variance of win_rate should be p(1-p)/n
        overall_wr = outcomes.mean()
        expected_var = overall_wr * (1 - overall_wr) / day_df["n"].mean()
        actual_var = day_df["wr"].var()
        print(f"    Avg day win rate: {day_df['wr'].mean():.3f}")
        print(f"    Expected variance (indep): {expected_var:.4f}")
        print(f"    Actual variance: {actual_var:.4f}")
        print(f"    Overdispersion ratio: {actual_var/expected_var:.2f}x")
    else:
        print(f"\n  No days with 5+ bets for correlation analysis")

    # Effective independent bets (design effect from clustering)
    unique_dates = sorted(set(dates))
    day_outcomes = []
    day_sizes = []
    for d in unique_dates:
        mask_d = bet_df["date"] == d
        day_outcomes.append(bet_df.loc[mask_d, "outcome"].values)
        day_sizes.append(mask_d.sum())

    # ICC (intraclass correlation)
    if len(days_5plus) > 2:
        # Simple ICC estimate
        overall_mean = outcomes.mean()
        ssb = sum(n_d * (o.mean() - overall_mean)**2 for o, n_d in zip(day_outcomes, day_sizes))
        ssw = sum(((o - o.mean())**2).sum() for o in day_outcomes if len(o) > 1)
        k = len(unique_dates)
        n_avg = n_bets / k
        msb = ssb / (k - 1)
        msw = ssw / (n_bets - k) if n_bets > k else 0
        icc = (msb - msw) / (msb + (n_avg - 1) * msw) if (msb + (n_avg - 1) * msw) > 0 else 0
        icc = max(0, icc)  # clamp negative ICC to 0
        deff = 1 + (n_avg - 1) * icc
        eff_n = n_bets / deff
        print(f"\n  ICC (intraclass correlation): {icc:.4f}")
        print(f"  Design effect: {deff:.2f}")
        print(f"  Effective independent bets: {eff_n:.0f} / {n_bets}")
    else:
        eff_n = n_bets
        print(f"\n  Insufficient multi-bet days for ICC; assuming independent")

    # Block bootstrap (resample by day)
    print(f"\n  Block Bootstrap (2000 resamples by day):")
    n_boot = 2000
    profit_per_1 = 100.0 / 110.0
    boot_rois = []
    for _ in range(n_boot):
        sampled_days = np.random.choice(len(unique_dates), len(unique_dates), replace=True)
        all_outcomes = []
        for di in sampled_days:
            all_outcomes.extend(day_outcomes[di])
        all_outcomes = np.array(all_outcomes)
        w = all_outcomes.sum()
        l = len(all_outcomes) - w
        units = w * profit_per_1 - l
        boot_rois.append(units / len(all_outcomes))

    boot_rois = np.array(boot_rois)
    print(f"    Mean ROI: {np.mean(boot_rois)*100:+.1f}%")
    print(f"    90% CI: [{np.percentile(boot_rois, 5)*100:+.1f}%, "
          f"{np.percentile(boot_rois, 95)*100:+.1f}%]")
    print(f"    P(ROI > 0%): {(boot_rois > 0).mean()*100:.1f}%")
    print(f"    P(ROI > 5%): {(boot_rois > 0.05).mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# TEST 7: FEATURE ABLATION
# ══════════════════════════════════════════════════════════════════════

def test_7_feature_ablation(model, X_val_s, y_val, book, df_val):
    print("\n" + "=" * 70)
    print("  TEST 7: FEATURE ABLATION (permutation importance)")
    print("=" * 70)

    mu_base, sigma_base = predict(model, X_val_s)
    actual = y_val
    has_book = ~np.isnan(book)
    base_mae = float(np.mean(np.abs(actual[has_book] - mu_base[has_book])))
    print(f"\n  Base BS-MAE: {base_mae:.4f}")

    feature_names = config.FEATURE_ORDER
    n_shuffles = 10

    importance = []
    for fi, fname in enumerate(feature_names):
        mae_increases = []
        for _ in range(n_shuffles):
            X_shuffled = X_val_s.copy()
            np.random.shuffle(X_shuffled[:, fi])
            mu_s, _ = predict(model, X_shuffled)
            mae_s = float(np.mean(np.abs(actual[has_book] - mu_s[has_book])))
            mae_increases.append(mae_s - base_mae)
        avg_inc = np.mean(mae_increases)
        importance.append((fname, avg_inc, np.std(mae_increases)))

    # Sort by importance descending
    importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Rank':>4} {'Feature':>35} {'MAE Increase':>13} {'Std':>8}")
    print(f"  {'-'*4} {'-'*35} {'-'*13} {'-'*8}")
    n_zero = 0
    zero_features = []
    for i, (fn, inc, std) in enumerate(importance):
        flag = ""
        if inc <= 0:
            n_zero += 1
            zero_features.append(fn)
            flag = " *"
        print(f"  {i+1:>4} {fn:>35} {inc:>+12.4f} {std:>8.4f}{flag}")

    print(f"\n  Features with zero/negative importance: {n_zero}")
    if zero_features:
        print(f"  Names: {zero_features}")

    # If there are useless features, retrain without them
    if n_zero >= 3:
        print(f"\n  Retraining C2-V2 dropping {n_zero} zero/negative features...")
        keep_idx = [i for i, (fn, inc, _) in enumerate(importance) if inc > 0]
        keep_names = [importance[i][0] for i in range(len(importance)) if importance[i][1] > 0]
        # Need original feature order indices
        orig_keep = [config.FEATURE_ORDER.index(fn) for fn in keep_names]

        # Reload full data for retraining
        X_tr_s, y_tr, X_v_s, y_v, _, df_v2 = load_season_data(
            list(range(2015, 2026)), [2026])
        X_tr_sub = X_tr_s[:, orig_keep]
        X_v_sub = X_v_s[:, orig_keep]

        hp_sub = WINNER_HP.copy()
        model_sub, best_ep = train_model(X_tr_sub, y_tr, X_v_sub, y_v,
                                          hp=hp_sub, verbose=True)
        mu_sub, sigma_sub = predict(model_sub, X_v_sub)
        book2 = df_v2["bookSpread"].values.astype(np.float64) if "bookSpread" in df_v2.columns else np.full(len(df_v2), np.nan)
        hb2 = ~np.isnan(book2)
        new_mae = float(np.mean(np.abs(y_v[hb2] - mu_sub[hb2])))
        print(f"  Reduced BS-MAE: {new_mae:.4f} (was {base_mae:.4f}, Δ={new_mae-base_mae:+.4f})")

        r_orig = compute_roi(mu_base, sigma_base, book, actual, 0.12,
                             sigma_lo=12.0, sigma_hi=16.0)
        r_sub = compute_roi(mu_sub, sigma_sub, book2, y_v, 0.12,
                            sigma_lo=12.0, sigma_hi=16.0)
        print(f"  ROI (12% + σ12-16): orig={r_orig['roi']*100:+.1f}% "
              f"new={r_sub['roi']*100:+.1f}%")

        del model_sub
        torch.cuda.empty_cache()
    else:
        print(f"  Only {n_zero} zero/negative features — not worth retraining.")

    return importance


# ══════════════════════════════════════════════════════════════════════
# TEST 8: HYPERPARAMETER STABILITY
# ══════════════════════════════════════════════════════════════════════

def test_8_hp_stability():
    print("\n" + "=" * 70)
    print("  TEST 8: HYPERPARAMETER STABILITY (8 perturbations)")
    print("=" * 70)

    perturbations = [
        ("P1: lr=2e-3",        {**WINNER_HP, "lr": 2e-3}),
        ("P2: lr=4e-3",        {**WINNER_HP, "lr": 4e-3}),
        ("P3: h2=320",         {**WINNER_HP, "hidden2": 320}),
        ("P4: h2=192",         {**WINNER_HP, "hidden2": 192}),
        ("P5: d=0.15",         {**WINNER_HP, "dropout": 0.15}),
        ("P6: d=0.25",         {**WINNER_HP, "dropout": 0.25}),
        ("P7: batch=2048",     {**WINNER_HP, "batch_size": 2048}),
        ("P8: batch=8192",     {**WINNER_HP, "batch_size": 8192}),
    ]

    X_tr, y_tr, X_v, y_v, scaler, df_v = load_season_data(
        list(range(2015, 2026)), [2026])
    book = df_v["bookSpread"].values.astype(np.float64) if "bookSpread" in df_v.columns else np.full(len(df_v), np.nan)
    has_book = ~np.isnan(book)

    results = []
    for pname, hp in perturbations:
        print(f"\n  Training {pname}...")
        t0 = time.time()
        model, best_ep = train_model(X_tr, y_tr, X_v, y_v, hp=hp, verbose=False)
        mu, sigma = predict(model, X_v)
        residuals = y_v - mu
        bs_mae = float(np.mean(np.abs(residuals[has_book])))
        dead = count_dead(model, X_v)
        _, cal_sc = quintile_cal(sigma, residuals)
        r = compute_roi(mu, sigma, book, y_v, 0.12, sigma_lo=12.0, sigma_hi=16.0)

        elapsed = time.time() - t0
        results.append({
            "name": pname, "bs_mae": bs_mae, "dead": dead,
            "sigma_std": float(np.std(sigma)), "cal_score": cal_sc,
            "roi": r["roi"], "roi_bets": r["bets"],
            "best_epoch": best_ep, "elapsed": elapsed,
        })
        print(f"    BS-MAE={bs_mae:.3f} dead={dead} σ_std={float(np.std(sigma)):.2f} "
              f"cal={cal_sc:.3f} ROI={r['roi']*100:+.1f}% ({r['bets']}b) [{elapsed:.0f}s]")

        del model
        torch.cuda.empty_cache()

    # Summary table
    maes = [r["bs_mae"] for r in results]
    mae_range = max(maes) - min(maes)

    print(f"\n  ── HP Stability Summary ──")
    print(f"  {'Run':>16} {'BS-MAE':>7} {'Dead':>5} {'σ_std':>6} {'Cal':>6} {'ROI':>8} {'Epoch':>6}")
    print(f"  {'-'*16} {'-'*7} {'-'*5} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:>16} {r['bs_mae']:>7.3f} {r['dead']:>5} "
              f"{r['sigma_std']:>6.2f} {r['cal_score']:>6.3f} "
              f"{r['roi']*100:>+7.1f}% {r['best_epoch']:>6}")

    print(f"\n  BS-MAE range: {mae_range:.4f}")
    if mae_range > 0.1:
        print(f"  *** WARNING: MAE varies by >{0.1}, optimum may be UNSTABLE ***")
    else:
        print(f"  PASS: MAE range < 0.1, optimum is stable")

    return results


# ══════════════════════════════════════════════════════════════════════
# TEST 9: CALIBRATION ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════

def test_9_calibration_slices(model, X_val_s, y_val, book, df_val):
    print("\n" + "=" * 70)
    print("  TEST 9: CALIBRATION ROBUSTNESS (per-slice)")
    print("=" * 70)

    mu, sigma = predict(model, X_val_s)
    actual = y_val
    residuals = actual - mu
    has_book = ~np.isnan(book)

    # Parse dates
    dates = pd.to_datetime(df_val["startDate"], errors="coerce", utc=True)
    months = dates.dt.month.values

    # Define slices
    slices = {}

    # By month
    for m, mname in [(12, "December"), (1, "January"), (2, "February"), (3, "March")]:
        mask = months == m
        if mask.sum() >= 50:
            slices[mname] = mask

    # By book spread magnitude
    if has_book.sum() > 0:
        abs_book = np.abs(book)
        slices["|book|<5 (close)"] = has_book & (abs_book < 5)
        slices["|book| 5-10"] = has_book & (abs_book >= 5) & (abs_book <= 10)
        slices["|book|>10 (blowout)"] = has_book & (abs_book > 10)

        # Home fav vs away fav
        slices["Home favorite"] = has_book & (book < 0)
        slices["Away favorite"] = has_book & (book > 0)

    # P6 detection — check if teams are in Power 6
    # Use sigma as a proxy: P6 games tend to have lower sigma
    # Actually, let's check for team columns
    p6_conferences = {"SEC", "Big 12", "Big Ten", "ACC", "Big East", "Pac-12"}
    if "homeConference" in df_val.columns:
        home_p6 = df_val["homeConference"].isin(p6_conferences).values
        away_p6 = df_val["awayConference"].isin(p6_conferences).values
        slices["Both P6"] = home_p6 & away_p6
        slices["Neither P6"] = ~home_p6 & ~away_p6
    else:
        # Use sigma as rough proxy: bottom quartile sigma ≈ major matchups
        sig_q25 = np.percentile(sigma, 25)
        sig_q75 = np.percentile(sigma, 75)
        slices["Low σ (bot 25%)"] = sigma <= sig_q25
        slices["High σ (top 25%)"] = sigma >= sig_q75

    print(f"\n  {'Slice':>25} {'N':>6} {'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6} {'Q5':>6} {'Cal':>6} {'Flag':>5}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")

    any_flag = False
    for sname, mask in slices.items():
        n = mask.sum()
        if n < 50:
            print(f"  {sname:>25} {n:>6}   (too few for calibration)")
            continue

        s_sigma = sigma[mask]
        s_res = residuals[mask]
        qr, cal = quintile_cal(s_sigma, s_res)

        flag = ""
        for q in qr:
            if q < 0.70 or q > 1.30:
                flag = " ***"
                any_flag = True
                break

        print(f"  {sname:>25} {n:>6} {qr[0]:>6.2f} {qr[1]:>6.2f} {qr[2]:>6.2f} "
              f"{qr[3]:>6.2f} {qr[4]:>6.2f} {cal:>6.3f}{flag}")

    if any_flag:
        print("\n  *** FLAG: Some slices have quintile ratios outside 0.70-1.30")
    else:
        print("\n  PASS: All quintile ratios within 0.70-1.30")


# ══════════════════════════════════════════════════════════════════════
# TEST 10: MODEL vs SIMPLE BASELINES
# ══════════════════════════════════════════════════════════════════════

def test_10_baselines(model, X_val_s, y_val, book):
    print("\n" + "=" * 70)
    print("  TEST 10: MODEL vs SIMPLE BASELINES")
    print("=" * 70)

    mu_model, sigma_model = predict(model, X_val_s)
    actual = y_val
    has_book = ~np.isnan(book)

    # Model results
    model_mae = float(np.mean(np.abs(actual[has_book] - mu_model[has_book])))
    model_results = run_all_strategies(mu_model, sigma_model, book, actual)

    # A) Book spread baseline: mu = -book_spread, sigma = 11.5
    print(f"\n  A) BOOK SPREAD BASELINE (mu=-book_spread, σ=11.5)")
    mu_book = np.where(has_book, -book, 0.0)
    sigma_book = np.full_like(mu_model, 11.5)
    book_mae = float(np.mean(np.abs(actual[has_book] - mu_book[has_book])))
    book_results = run_all_strategies(mu_book, sigma_book, book, actual)
    print(f"  BS-MAE: {book_mae:.3f}")
    print_strategy_table(book_results)

    # B) Home advantage: mu = -book_spread + 0.3
    print(f"\n  B) HOME ADVANTAGE (mu=-book_spread+0.3, σ=11.5)")
    mu_home = np.where(has_book, -book + 0.3, 0.0)
    home_results = run_all_strategies(mu_home, sigma_book, book, actual)
    home_mae = float(np.mean(np.abs(actual[has_book] - mu_home[has_book])))
    print(f"  BS-MAE: {home_mae:.3f}")
    print_strategy_table(home_results)

    # C) Regression to mean: mu = 0.8 * (-book_spread)
    print(f"\n  C) REGRESSION TO MEAN (mu=0.8*(-book_spread), σ=11.5)")
    mu_reg = np.where(has_book, 0.8 * (-book), 0.0)
    reg_mae = float(np.mean(np.abs(actual[has_book] - mu_reg[has_book])))
    reg_results = run_all_strategies(mu_reg, sigma_book, book, actual)
    print(f"  BS-MAE: {reg_mae:.3f}")
    print_strategy_table(reg_results)

    # D) C2-V2 mu with constant sigma
    print(f"\n  D) C2-V2 MU + CONSTANT SIGMA (σ=12.0)")
    sigma_const = np.full_like(sigma_model, 12.0)
    const_results = run_all_strategies(mu_model, sigma_const, book, actual)
    print(f"  BS-MAE: {model_mae:.3f} (same mu)")
    print_strategy_table(const_results)

    # Comparison
    print(f"\n  ── Comparison (12% edge + σ12-16 ROI) ──")
    print(f"  {'Method':>30} {'BS-MAE':>7} {'ROI':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*8}")
    for name, mae, results in [
        ("C2-V2 (full model)", model_mae, model_results),
        ("Book spread baseline", book_mae, book_results),
        ("Home advantage (+0.3)", home_mae, home_results),
        ("Regression to mean (0.8x)", reg_mae, reg_results),
        ("C2-V2 mu + const σ", model_mae, const_results),
    ]:
        r = results[2]  # strategy c
        roi_str = f"{r['roi']*100:+.1f}%" if r["bets"] > 0 else "N/A"
        print(f"  {name:>30} {mae:>7.3f} {roi_str:>8}")


# ══════════════════════════════════════════════════════════════════════
# TEST 11: LINE STALENESS CHECK
# ══════════════════════════════════════════════════════════════════════

def test_11_line_staleness(df_val):
    print("\n" + "=" * 70)
    print("  TEST 11: LINE STALENESS CHECK")
    print("=" * 70)

    # Check what columns exist in lines data
    try:
        lines_df = load_research_lines(2026)
        print(f"\n  Lines table columns: {list(lines_df.columns)}")
        print(f"  Total rows: {len(lines_df)}")

        if "provider" in lines_df.columns:
            print(f"\n  Providers: {lines_df['provider'].value_counts().to_dict()}")

        # Print sample rows
        print(f"\n  Sample lines (5 games):")
        sample = lines_df.head(10)
        for col in sample.columns:
            print(f"    {col}: {sample[col].tolist()[:5]}")

        # Check for timestamp columns
        ts_cols = [c for c in lines_df.columns if "time" in c.lower() or "date" in c.lower() or "stamp" in c.lower()]
        if ts_cols:
            print(f"\n  Timestamp columns found: {ts_cols}")
            for tc in ts_cols:
                print(f"    {tc} sample: {lines_df[tc].head(5).tolist()}")
        else:
            print(f"\n  No timestamp columns found in lines data.")
            print(f"  Cannot determine if lines are opening, closing, or consensus.")

    except Exception as e:
        print(f"\n  Error loading lines: {e}")


# ══════════════════════════════════════════════════════════════════════
# TEST 12: PROFITABLE BET CHARACTERIZATION
# ══════════════════════════════════════════════════════════════════════

def test_12_bet_profiling(model, X_val_s, y_val, book, df_val):
    print("\n" + "=" * 70)
    print("  TEST 12: PROFITABLE BET CHARACTERIZATION")
    print("=" * 70)

    mu, sigma = predict(model, X_val_s)
    actual = y_val
    has_book = ~np.isnan(book)

    valid = has_book.copy()
    edge_home = mu[valid] + book[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    hcp = normal_cdf(edge_z)
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, hcp, 1.0 - hcp)
    prob_edge = pick_prob - 0.5238
    sigma_v = sigma[valid]

    bet_mask = (prob_edge >= 0.12) & (sigma_v >= 12.0) & (sigma_v <= 16.0)

    actual_v = actual[valid]
    book_v = book[valid]
    mu_v = mu[valid]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)

    # Build bet dataframe
    valid_idx = np.where(valid)[0]
    bet_idx = np.where(bet_mask)[0]
    bet_idx_in_df = valid_idx[bet_idx]

    bet_info = pd.DataFrame({
        "won": pick_won[bet_mask].astype(bool),
        "book_spread": book_v[bet_mask],
        "abs_book": np.abs(book_v[bet_mask]),
        "sigma": sigma_v[bet_mask],
        "prob_edge": prob_edge[bet_mask],
        "pick_home": pick_home[bet_mask],
        "mu": mu_v[bet_mask],
        "actual": actual_v[bet_mask],
        "margin_vs_spread": actual_v[bet_mask] + book_v[bet_mask],  # positive = home covers
    })

    # Add dates
    if "startDate" in df_val.columns:
        bet_info["date"] = pd.to_datetime(
            df_val.iloc[bet_idx_in_df]["startDate"].values,
            errors="coerce", utc=True)
        bet_info["month"] = bet_info["date"].dt.month

    n_bets = len(bet_info)
    winners = bet_info[bet_info["won"]]
    losers = bet_info[~bet_info["won"]]

    print(f"\n  Total bets: {n_bets} (W: {len(winners)}, L: {len(losers)})")

    # Winners vs losers profiling
    print(f"\n  {'Metric':>25} {'Winners':>10} {'Losers':>10} {'Δ':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for metric, col in [
        ("Mean |book_spread|", "abs_book"),
        ("Mean sigma", "sigma"),
        ("Mean prob_edge", "prob_edge"),
    ]:
        wm = winners[col].mean()
        lm = losers[col].mean()
        print(f"  {metric:>25} {wm:>10.2f} {lm:>10.2f} {wm-lm:>+10.2f}")

    # Direction breakdown
    n_home = bet_info["pick_home"].sum()
    n_away = n_bets - n_home
    home_wr = winners["pick_home"].sum() / max(n_home, 1) if n_home > 0 else 0
    away_wr = (len(winners) - winners["pick_home"].sum()) / max(n_away, 1) if n_away > 0 else 0
    print(f"\n  Direction: Home={n_home} ({n_home/n_bets*100:.1f}%), "
          f"Away={n_away} ({n_away/n_bets*100:.1f}%)")
    print(f"  Win rate: Home={home_wr*100:.1f}%, Away={away_wr*100:.1f}%")

    # Favorites vs underdogs
    bet_info["fav_home"] = bet_info["book_spread"] < 0
    bet_home_fav = bet_info[bet_info["pick_home"] & bet_info["fav_home"]]
    bet_home_dog = bet_info[bet_info["pick_home"] & ~bet_info["fav_home"]]
    bet_away_fav = bet_info[~bet_info["pick_home"] & ~bet_info["fav_home"]]
    bet_away_dog = bet_info[~bet_info["pick_home"] & bet_info["fav_home"]]

    print(f"\n  {'Pick Type':>25} {'Bets':>6} {'Win%':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*7}")
    for label, subset in [
        ("Picking home fav", bet_home_fav),
        ("Picking home dog", bet_home_dog),
        ("Picking away fav", bet_away_fav),
        ("Picking away dog", bet_away_dog),
    ]:
        n = len(subset)
        if n > 0:
            wr = subset["won"].mean()
            print(f"  {label:>25} {n:>6} {wr*100:>6.1f}%")

    # Margin analysis
    # For picks we back: margin vs spread
    # If pick_home: margin = actual + book_spread (positive = home covers)
    # If pick_away: margin = -(actual + book_spread) (positive = away covers)
    bet_info["cover_margin"] = np.where(
        bet_info["pick_home"].values,
        bet_info["actual"].values + bet_info["book_spread"].values,
        -(bet_info["actual"].values + bet_info["book_spread"].values))

    w_margin = bet_info.loc[bet_info["won"], "cover_margin"].mean()
    l_margin = bet_info.loc[~bet_info["won"], "cover_margin"].mean()
    print(f"\n  Cover margin (positive = our pick covers):")
    print(f"    Winners: {w_margin:+.1f} pts avg")
    print(f"    Losers: {l_margin:+.1f} pts avg")

    # Month breakdown
    if "month" in bet_info.columns:
        print(f"\n  Monthly breakdown:")
        profit_per_1 = 100.0 / 110.0
        print(f"  {'Month':>10} {'Bets':>6} {'W-L':>8} {'Win%':>7} {'ROI':>8} {'Units':>8}")
        print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
        for m in sorted(bet_info["month"].dropna().unique()):
            mb = bet_info[bet_info["month"] == m]
            n_m = len(mb)
            w_m = mb["won"].sum()
            l_m = n_m - w_m
            u = w_m * profit_per_1 - l_m
            roi = u / n_m if n_m > 0 else 0
            month_name = {12: "December", 1: "January", 2: "February",
                          3: "March", 4: "April"}.get(int(m), str(int(m)))
            print(f"  {month_name:>10} {n_m:>6} {int(w_m):>3}-{int(l_m):<4} "
                  f"{w_m/n_m*100:>6.1f}% {roi*100:>+7.1f}% {u:>+8.1f}")

    # Book spread buckets
    print(f"\n  By |book_spread| bucket:")
    print(f"  {'Bucket':>12} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*8}")
    profit_per_1 = 100.0 / 110.0
    for lo, hi, label in [(0, 5, "0-5"), (5, 10, "5-10"), (10, 15, "10-15"), (15, 99, "15+")]:
        mask = (bet_info["abs_book"] >= lo) & (bet_info["abs_book"] < hi)
        n_b = mask.sum()
        if n_b > 0:
            w_b = bet_info.loc[mask, "won"].sum()
            l_b = n_b - w_b
            u = w_b * profit_per_1 - l_b
            print(f"  {label:>12} {n_b:>6} {w_b/n_b*100:>6.1f}% {u/n_b*100:>+7.1f}%")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  SESSION 13: COMPREHENSIVE VALIDATION SUITE (12 TESTS)")
    print("  Model: C2-V2 (384→256, d=0.20, lr=3e-3, batch=4096, Gaussian)")
    print("=" * 70)
    t_start = time.time()

    # ── GPU Tests (priority order) ──

    # TEST 1: Walk-forward (highest priority — trains 7 models)
    wf_results = test_1_walk_forward()

    # Load saved model and val data for non-GPU tests
    print("\n  Loading saved C2-V2 model and val data...")
    model = load_winner_model()
    X_val_s, y_val, book, df_val = load_val_data()

    # TEST 2: Sigma ablation
    test_2_sigma_ablation(model, X_val_s, y_val, book, df_val)

    # TEST 4: Bootstrap CI
    test_4_bootstrap_ci(model, X_val_s, y_val, book)

    # TEST 5: Drawdown
    test_5_drawdown(model, X_val_s, y_val, book, df_val)

    # TEST 6: Bet correlation
    test_6_bet_correlation(model, X_val_s, y_val, book, df_val)

    # TEST 10: Baselines
    test_10_baselines(model, X_val_s, y_val, book)

    # TEST 3: Vig sensitivity
    test_3_vig_sensitivity(model, X_val_s, y_val, book)

    # TEST 12: Bet profiling
    test_12_bet_profiling(model, X_val_s, y_val, book, df_val)

    # TEST 7: Feature ablation (GPU)
    test_7_feature_ablation(model, X_val_s, y_val, book, df_val)

    # TEST 8: HP stability (GPU - trains 8 models)
    test_8_hp_stability()

    # TEST 9: Calibration slices (no GPU)
    test_9_calibration_slices(model, X_val_s, y_val, book, df_val)

    # TEST 11: Line staleness (no GPU)
    test_11_line_staleness(df_val)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL 12 TESTS COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

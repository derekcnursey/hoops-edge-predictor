#!/usr/bin/env python3
"""Walk-forward backtest: 3-way comparison for no_garbage + HCA feature audit.

Trains on prior seasons, predicts next season. Holdout years: 2019-2025.
Uses C2-V2 winner hyperparameters (384/256, d=0.20, lr=3e-3, batch=4096).

Runs 3 configs:
  A) 50-feat regular (old Session 13 setup)
  B) 50-feat no_garbage (isolate feature-type effect)
  C) 53-feat no_garbage (full new pipeline: HCA + no_garbage fix)

Prints Table 1: HCA impact in no_garbage world (B vs C)
Prints Table 2: Full before/after (A vs C)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets, load_lines
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

# 3 new HCA features to drop for the 50-feature baseline
HCA_FEATURES = ["home_team_hca", "home_team_efg_home_split", "away_team_efg_away_split"]

# Feature orders
FEATURE_ORDER_53 = config.FEATURE_ORDER  # 53 features (current)
FEATURE_ORDER_50 = [f for f in FEATURE_ORDER_53 if f not in HCA_FEATURES]


# ── Utilities ─────────────────────────────────────────────────────────

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


def train_model(X_train, y_train, X_val_s, y_val, hp=None):
    """Train MLPRegressor with early stopping. Returns (model, best_epoch)."""
    if hp is None:
        hp = HP
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
                        shuffle=True, drop_last=True)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

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
    """Run inference. Returns (mu, sigma) arrays."""
    model.eval()
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    mu_list, sigma_list = [], []
    for s in range(0, len(X_t), 4096):
        e = min(s + 4096, len(X_t))
        mu, log_sigma = model(X_t[s:e])
        sigma = torch.exp(log_sigma).clamp(0.5, 30.0)
        mu_list.append(mu.numpy())
        sigma_list.append(sigma.numpy())
    return np.concatenate(mu_list), np.concatenate(sigma_list)


def compute_picks(mu, sigma, book_spread):
    """Compute edge-based picks. Returns DataFrame with pick metrics."""
    # mu is predicted home spread (positive = home wins)
    # book_spread is from home perspective (negative = home favored)
    model_spread = -mu  # convert to book convention
    edge_home_pts = mu + book_spread  # positive = home undervalued

    # Probability edge
    prob_edge = normal_cdf(np.abs(edge_home_pts) / sigma) - 0.5

    # Pick side: positive edge_home_pts → pick home
    pick_home = edge_home_pts >= 0

    return pd.DataFrame({
        "mu": mu,
        "sigma": sigma,
        "book_spread": book_spread,
        "model_spread": model_spread,
        "edge_home_pts": edge_home_pts,
        "prob_edge": prob_edge,
        "pick_home": pick_home,
    })


def evaluate_picks(picks_df, actual_spread, thresholds=[0.0, 0.05, 0.10]):
    """Evaluate pick performance at various edge thresholds."""
    results = {}
    for thr in thresholds:
        mask = picks_df["prob_edge"] >= thr
        if mask.sum() == 0:
            results[thr] = {"n": 0}
            continue

        sub = picks_df[mask].copy()
        actual = actual_spread[mask]
        n = len(sub)

        # Pick side stats
        n_home = sub["pick_home"].sum()
        n_away = n - n_home
        home_pct = n_home / n if n > 0 else 0

        # ATS results
        # If pick_home: win if actual > -book_spread (home covers)
        # If pick_away: win if actual < -book_spread (away covers)
        home_covers = actual + sub["book_spread"].values > 0  # actual_margin > -book_spread
        pick_wins = np.where(sub["pick_home"].values, home_covers, ~home_covers)
        pushes = np.abs(actual + sub["book_spread"].values) < 0.5
        pick_wins = pick_wins & ~pushes

        win_rate = pick_wins.sum() / (n - pushes.sum()) if (n - pushes.sum()) > 0 else 0
        # ROI: flat $100 bets, -110 juice
        wins = pick_wins.sum()
        losses = n - pushes.sum() - wins
        roi = (wins * 100 - losses * 110) / (n * 110) if n > 0 else 0

        # Home-only ROI
        home_mask = sub["pick_home"].values
        if home_mask.sum() > 0:
            h_wins = (pick_wins & home_mask).sum()
            h_losses = home_mask.sum() - (pushes & home_mask).sum() - h_wins
            h_n = home_mask.sum()
            home_roi = (h_wins * 100 - h_losses * 110) / (h_n * 110) if h_n > 0 else 0
        else:
            home_roi = 0

        # Away-only ROI
        away_mask = ~sub["pick_home"].values
        if away_mask.sum() > 0:
            a_wins = (pick_wins & away_mask).sum()
            a_losses = away_mask.sum() - (pushes & away_mask).sum() - a_wins
            a_n = away_mask.sum()
            away_roi = (a_wins * 100 - a_losses * 110) / (a_n * 110) if a_n > 0 else 0
        else:
            away_roi = 0

        results[thr] = {
            "n": n,
            "home_pct": home_pct,
            "away_pct": 1 - home_pct,
            "win_rate": win_rate,
            "roi": roi,
            "home_roi": home_roi,
            "away_roi": away_roi,
            "n_home": int(n_home),
            "n_away": int(n_away),
        }

    return results


# ── Main Walk-Forward ─────────────────────────────────────────────────

def run_walkforward(feature_order, label="", no_garbage=False):
    """Run walk-forward across holdout years with given feature order."""
    print(f"\n{'='*70}")
    ng_tag = " [no_garbage]" if no_garbage else " [regular]"
    print(f"  WALK-FORWARD: {label} ({len(feature_order)} features){ng_tag}")
    print(f"{'='*70}")

    all_results = {}
    for holdout in HOLDOUT_YEARS:
        train_seasons = list(range(TRAIN_START, holdout))
        val_seasons = [holdout]
        print(f"\n  Holdout {holdout}: train {TRAIN_START}-{holdout-1}...", end=" ", flush=True)

        # Load data
        df_train = load_multi_season_features(
            train_seasons, no_garbage=no_garbage,
            adj_suffix=ADJ_SUFFIX, min_month_day=MIN_DATE)
        df_train = df_train.dropna(subset=["homeScore", "awayScore"])
        df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]

        df_val = load_multi_season_features(
            val_seasons, no_garbage=no_garbage,
            adj_suffix=ADJ_SUFFIX, min_month_day=MIN_DATE)
        df_val = df_val.dropna(subset=["homeScore", "awayScore"])
        df_val = df_val[(df_val["homeScore"] != 0) | (df_val["awayScore"] != 0)]

        # Merge book spreads
        try:
            lines_df = load_lines(holdout)
            if not lines_df.empty:
                ld = lines_df.sort_values("provider").drop_duplicates(
                    subset=["gameId"], keep="first")
                if "spread" in ld.columns:
                    merge_df = ld[["gameId", "spread"]].rename(
                        columns={"spread": "bookSpread"})
                    df_val = df_val.merge(merge_df, on="gameId", how="left")
        except Exception:
            pass

        # Extract features using the specified feature order
        X_train = get_feature_matrix(df_train, feature_order=feature_order).values.astype(np.float32)
        targets_train = get_targets(df_train)
        y_train = targets_train["spread_home"].values.astype(np.float32)

        X_val = get_feature_matrix(df_val, feature_order=feature_order).values.astype(np.float32)
        targets_val = get_targets(df_val)
        y_val = targets_val["spread_home"].values.astype(np.float32)

        # Impute and scale
        X_train = impute_column_means(X_train)
        X_val = impute_column_means(X_val)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)

        # Train
        model, best_epoch = train_model(X_train_s, y_train, X_val_s, y_val)
        print(f"ep={best_epoch}", end=" ", flush=True)

        # Predict
        mu, sigma = predict_model(model, X_val_s)

        # MAE vs actual
        mae = np.mean(np.abs(mu - y_val))

        # MAE vs book (for games with book spreads)
        has_book = df_val.get("bookSpread", pd.Series(dtype=float)).notna()
        if has_book.sum() > 0:
            book_vals = df_val.loc[has_book, "bookSpread"].values.astype(float)
            book_mae = np.mean(np.abs(-mu[has_book] - book_vals))  # model_spread vs book
            model_vs_book_mae = np.mean(np.abs(mu[has_book] - y_val[has_book]))

            # Compute picks for games with book lines
            picks = compute_picks(mu[has_book], sigma[has_book], book_vals)
            eval_results = evaluate_picks(picks, y_val[has_book])
        else:
            book_mae = float("nan")
            model_vs_book_mae = mae
            eval_results = {}

        all_results[holdout] = {
            "mae": mae,
            "book_mae": book_mae,
            "best_epoch": best_epoch,
            "n_games": len(df_val),
            "n_with_book": int(has_book.sum()),
            "eval": eval_results,
        }
        print(f"MAE={mae:.2f} n={len(df_val)}")

    return all_results


def _aggregate(label_a, label_b, results_a, results_b):
    """Print a comparison table between two result sets."""

    print(f"\n{'='*90}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"{'='*90}")

    print(f"\n  {'Year':>4}  {'MAE(A)':>8}  {'MAE(B)':>8}  {'Δ':>6}  "
          f"{'Home%(A)':>9}  {'Home%(B)':>9}  "
          f"{'ROI(A)':>8}  {'ROI(B)':>8}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}")

    agg = {k: {"mae": [], "hp": [], "roi": [], "hroi": [], "aroi": []}
           for k in ("a", "b")}

    for yr in HOLDOUT_YEARS:
        ra, rb = results_a[yr], results_b[yr]
        ea = ra["eval"].get(0.05, {})
        eb = rb["eval"].get(0.05, {})
        hp_a = ea.get("home_pct", 0) * 100
        hp_b = eb.get("home_pct", 0) * 100
        roi_a = ea.get("roi", 0) * 100
        roi_b = eb.get("roi", 0) * 100
        mae_d = rb["mae"] - ra["mae"]
        covid = " *" if yr == 2021 else ""

        print(f"  {yr:>4}{covid} {ra['mae']:>8.2f}  {rb['mae']:>8.2f}  "
              f"{mae_d:>+6.2f}  "
              f"{hp_a:>8.1f}%  {hp_b:>8.1f}%  "
              f"{roi_a:>+7.1f}%  {roi_b:>+7.1f}%")

        if yr != 2021:
            agg["a"]["mae"].append(ra["mae"])
            agg["b"]["mae"].append(rb["mae"])
            for side, ev in [("a", ea), ("b", eb)]:
                if ev.get("n", 0) > 0:
                    agg[side]["hp"].append(ev["home_pct"])
                    agg[side]["roi"].append(ev["roi"])
                    agg[side]["hroi"].append(ev.get("home_roi", 0))
                    agg[side]["aroi"].append(ev.get("away_roi", 0))

    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}")

    ma = np.mean(agg["a"]["mae"])
    mb = np.mean(agg["b"]["mae"])
    hpa = np.mean(agg["a"]["hp"]) * 100 if agg["a"]["hp"] else 0
    hpb = np.mean(agg["b"]["hp"]) * 100 if agg["b"]["hp"] else 0
    roia = np.mean(agg["a"]["roi"]) * 100 if agg["a"]["roi"] else 0
    roib = np.mean(agg["b"]["roi"]) * 100 if agg["b"]["roi"] else 0
    hroia = np.mean(agg["a"]["hroi"]) * 100 if agg["a"]["hroi"] else 0
    hroib = np.mean(agg["b"]["hroi"]) * 100 if agg["b"]["hroi"] else 0
    aroia = np.mean(agg["a"]["aroi"]) * 100 if agg["a"]["aroi"] else 0
    aroib = np.mean(agg["b"]["aroi"]) * 100 if agg["b"]["aroi"] else 0

    print(f"  {'AVG':>4}  {ma:>8.2f}  {mb:>8.2f}  {mb-ma:>+6.2f}  "
          f"{hpa:>8.1f}%  {hpb:>8.1f}%  "
          f"{roia:>+7.1f}%  {roib:>+7.1f}%")
    print(f"  (excluding 2021 COVID)")

    # Summary
    print(f"\n  {'─'*70}")
    print(f"  SUMMARY (excl 2021, edge >= 5%)")
    print(f"  {'─'*70}")
    print(f"  {'Metric':<25} {'A':>10} {'B':>10} {'Change':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'MAE':<25} {ma:>10.2f} {mb:>10.2f} {mb-ma:>+10.2f}")
    print(f"  {'Home pick %':<25} {hpa:>9.1f}% {hpb:>9.1f}% {hpb-hpa:>+9.1f}%")
    print(f"  {'Away pick %':<25} {100-hpa:>9.1f}% {100-hpb:>9.1f}% {hpa-hpb:>+9.1f}%")
    print(f"  {'Home ROI':<25} {hroia:>+9.1f}% {hroib:>+9.1f}% {hroib-hroia:>+9.1f}%")
    print(f"  {'Away ROI':<25} {aroia:>+9.1f}% {aroib:>+9.1f}% {aroib-aroia:>+9.1f}%")
    print(f"  {'Overall ROI':<25} {roia:>+9.1f}% {roib:>+9.1f}% {roib-roia:>+9.1f}%")

    # Edge threshold breakdown
    print(f"\n  {'─'*75}")
    print(f"  HOME/AWAY SPLIT BY EDGE THRESHOLD (avg excl 2021)")
    print(f"  {'─'*75}")
    for thr in [0.0, 0.05, 0.10]:
        hp_a_list, hp_b_list = [], []
        for yr in HOLDOUT_YEARS:
            if yr == 2021:
                continue
            ev_a = results_a[yr]["eval"].get(thr, {})
            ev_b = results_b[yr]["eval"].get(thr, {})
            if ev_a.get("n", 0) > 0:
                hp_a_list.append(ev_a["home_pct"])
            if ev_b.get("n", 0) > 0:
                hp_b_list.append(ev_b["home_pct"])
        ha = np.mean(hp_a_list) * 100 if hp_a_list else 0
        hb = np.mean(hp_b_list) * 100 if hp_b_list else 0
        print(f"  edge >= {thr*100:4.0f}%:  "
              f"A home {ha:5.1f}%/away {100-ha:5.1f}%  |  "
              f"B home {hb:5.1f}%/away {100-hb:5.1f}%  "
              f"(Δ home {hb-ha:+.1f}%)")

    # Flags
    print(f"\n  {'─'*60}")
    mae_d = mb - ma
    if mae_d > 0.1:
        print(f"  WARNING: MAE regressed by {mae_d:.2f}")
    elif mae_d < -0.05:
        print(f"  OK: MAE improved by {-mae_d:.2f}")
    else:
        print(f"  OK: MAE flat (delta {mae_d:+.2f})")

    roi_d = roib - roia
    if roi_d < -1.0:
        print(f"  WARNING: ROI dropped by {-roi_d:.1f}%")
    elif roi_d > 0.5:
        print(f"  OK: ROI improved by {roi_d:.1f}%")
    else:
        print(f"  OK: ROI flat (delta {roi_d:+.1f}%)")

    # Skew improvement: measure distance from 50%
    dist_a = abs(hpa - 50)
    dist_b = abs(hpb - 50)
    if dist_b < dist_a - 1.0:
        print(f"  OK: Pick balance improved ({hpa:.1f}% -> {hpb:.1f}% home, "
              f"closer to 50% by {dist_a - dist_b:.1f} pp)")
    elif dist_b > dist_a + 1.0:
        print(f"  WARNING: Pick balance worsened ({hpa:.1f}% -> {hpb:.1f}% home)")
    else:
        print(f"  OK: Pick balance unchanged ({hpa:.1f}% -> {hpb:.1f}% home)")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # A) 50-feat regular (old Session 13 setup)
    results_A = run_walkforward(
        FEATURE_ORDER_50, label="A: 50-feat regular", no_garbage=False)

    # B) 50-feat no_garbage (isolate feature-type effect)
    results_B = run_walkforward(
        FEATURE_ORDER_50, label="B: 50-feat no_garbage", no_garbage=True)

    # C) 53-feat no_garbage (full new pipeline)
    results_C = run_walkforward(
        FEATURE_ORDER_53, label="C: 53-feat no_garbage", no_garbage=True)

    # ── Table 1: HCA impact in no_garbage world (B vs C) ────────────
    print(f"\n\n{'#'*90}")
    print(f"  TABLE 1: HCA FEATURE IMPACT (no_garbage world)")
    print(f"  A = 50-feat no_garbage  |  B = 53-feat no_garbage")
    print(f"{'#'*90}")
    _aggregate("B: 50-feat NG", "C: 53-feat NG", results_B, results_C)

    # ── Table 2: Full before/after (A vs C) ──────────────────────────
    print(f"\n\n{'#'*90}")
    print(f"  TABLE 2: FULL BEFORE/AFTER")
    print(f"  A = 50-feat regular (Session 13 setup)  |  B = 53-feat no_garbage (new)")
    print(f"{'#'*90}")
    _aggregate("A: 50-feat reg", "C: 53-feat NG", results_A, results_C)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")

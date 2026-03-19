#!/usr/bin/env python3
"""Quick alpha sweep: patch gold-rating features at different WLS alphas, evaluate on 2025.

Only patches the 10 gold-derived features in the val set. Training features stay fixed
(from the current pipeline). This tests the effect of alpha on prediction quality without
a full pipeline rebuild.

Usage:
    poetry run python scripts/wls_alpha_sweep.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Setup paths
PROJ = Path(__file__).resolve().parent.parent
ETL = PROJ.parent / "hoops_edge_database_etl"
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(ETL / "src"))
os.chdir(ETL)  # so load_config() finds config.yaml

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets, load_research_lines
from src.trainer import impute_column_means

from cbbd_etl.config import load_config
from cbbd_etl.gold.adjusted_efficiencies import _load_d1_team_ids, _load_pbp_no_garbage_games
from cbbd_etl.gold.least_squares_ratings import solve_ratings_wls
from cbbd_etl.gold.iterative_ratings import GameObs
from cbbd_etl.s3_io import S3IO

# ── Config ─────────────────────────────────────────────────────────
ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
FEATURE_ORDER = config.FEATURE_ORDER
HOLDOUT = 2025
TRAIN_SEASONS = list(range(2015, HOLDOUT))
MIN_DATE = "12-01"
HP = {"hidden1": 384, "hidden2": 256, "dropout": 0.20,
      "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4}

ALPHAS = [0.01, 10, 50, 80, 100, 110, 120, 150, 200]

# Gold-derived feature columns to patch
ADJ_COLS = [
    "home_team_adj_oe", "home_team_adj_de", "home_team_adj_pace",
    "away_team_adj_oe", "away_team_adj_de", "away_team_adj_pace",
    "home_sos_oe", "home_sos_de", "away_sos_oe", "away_sos_de",
]


def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _val_loss(model, Xv, yv, dev):
    model.eval()
    t, n = 0.0, 0
    for s in range(0, len(Xv), 4096):
        mu, ls = model(Xv[s:s + 4096].to(dev))
        nll, _ = gaussian_nll_loss(mu, ls, yv[s:s + 4096].to(dev))
        t += nll.mean().item()
        n += 1
    model.train()
    return t / max(n, 1)


def train_eval(X_tr, y_tr, X_v, y_v):
    dev = get_device()
    use_amp = dev.type == "cuda"
    model = MLPRegressor(X_tr.shape[1], HP["hidden1"], HP["hidden2"], HP["dropout"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)
    amp_s = GradScaler(dev.type, enabled=use_amp)
    ds = HoopsDataset(X_tr, spread=y_tr, home_win=np.zeros(len(y_tr)))
    loader = DataLoader(ds, batch_size=HP["batch_size"], shuffle=True, drop_last=True)
    Xvt = torch.tensor(X_v, dtype=torch.float32)
    yvt = torch.tensor(y_v, dtype=torch.float32)
    best_vl, best_st, best_ep, no_imp = float("inf"), None, 0, 0
    model.train()
    for ep in range(500):
        for batch in loader:
            x, sp, _ = [b.to(dev) for b in batch]
            opt.zero_grad()
            with autocast(dev.type, enabled=use_amp):
                mu, ls = model(x)
                nll, _ = gaussian_nll_loss(mu, ls, sp)
                loss = nll.mean()
            amp_s.scale(loss).backward()
            amp_s.step(opt)
            amp_s.update()
        sched.step()
        vl = _val_loss(model, Xvt, yvt, dev)
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep + 1
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 50:
            break
    model.cpu()
    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_v, dtype=torch.float32)
        mu_l, sig_l = [], []
        for s in range(0, len(Xt), 4096):
            mu, ls = model(Xt[s:s + 4096])
            sig = torch.exp(ls).clamp(0.5, 30.0)
            mu_l.append(mu.numpy().flatten())
            sig_l.append(sig.numpy().flatten())
    return np.concatenate(mu_l), np.concatenate(sig_l), best_ep


def main():
    t0 = time.time()

    # Load ETL games for holdout season
    print("Loading ETL games for WLS solver...")
    etl_cfg = load_config()
    s3 = S3IO(etl_cfg.bucket, etl_cfg.region)
    d1_ids = _load_d1_team_ids(s3, etl_cfg)
    games_by_date = _load_pbp_no_garbage_games(s3, etl_cfg, HOLDOUT, d1_ids)
    all_games = []
    for d, dg in games_by_date.items():
        for g in dg:
            all_games.append(GameObs(
                game_id=g.game_id, team_id=g.team_id, opp_id=g.opp_id,
                team_pts=g.team_pts, team_poss=g.team_poss,
                opp_pts=g.opp_pts, opp_poss=g.opp_poss,
                is_home=g.is_home, is_neutral=g.is_neutral,
                game_date=g.game_date, weight=1.0,
            ))
    print(f"  {len(all_games)} game obs for {HOLDOUT}")

    # Load feature data
    print("Loading features...")
    os.chdir(PROJ)
    df_train = load_multi_season_features(
        TRAIN_SEASONS, no_garbage=True, adj_suffix=ADJ_SUFFIX, min_month_day=MIN_DATE)
    df_train = df_train.dropna(subset=["homeScore", "awayScore"])
    df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]

    df_val_base = load_multi_season_features(
        [HOLDOUT], no_garbage=True, adj_suffix=ADJ_SUFFIX, min_month_day=MIN_DATE)
    df_val_base = df_val_base.dropna(subset=["homeScore", "awayScore"])
    df_val_base = df_val_base[(df_val_base["homeScore"] != 0) | (df_val_base["awayScore"] != 0)]

    # Merge book spreads
    book_arr = np.full(len(df_val_base), np.nan)
    try:
        lines = load_research_lines(HOLDOUT)
        if not lines.empty:
            ld = lines.sort_values("provider").drop_duplicates("gameId", keep="first")
            if "spread" in ld.columns:
                df_val_base = df_val_base.merge(
                    ld[["gameId", "spread"]].rename(columns={"spread": "bookSpread"}),
                    on="gameId", how="left")
                book_arr = df_val_base["bookSpread"].values.astype(float)
    except Exception:
        pass

    # Fixed training data
    X_train = get_feature_matrix(df_train, feature_order=FEATURE_ORDER).values.astype(np.float32)
    y_train = get_targets(df_train)["spread_home"].values.astype(np.float32)
    X_train = impute_column_means(X_train)

    y_val = get_targets(df_val_base)["spread_home"].values.astype(np.float32)

    # Column indices for patching
    col_idx = {c: FEATURE_ORDER.index(c) for c in ADJ_COLS}

    # Base val features (before patching)
    X_val_base = get_feature_matrix(df_val_base, feature_order=FEATURE_ORDER).values.astype(np.float32)

    # Team IDs for lookup
    home_ids = df_val_base["homeTeamId"].values
    away_ids = df_val_base["awayTeamId"].values

    print(f"\nAlpha sweep on holdout {HOLDOUT} ({len(df_val_base)} val games, {len(df_train)} train games)")
    print(f"{'alpha':>8}  {'MAE':>6}  {'RMSE':>6}  {'σ':>5}  {'ep':>3}  "
          f"{'W':>5}  {'L':>5}  {'WR':>6}  {'ROI':>7}  {'top_net':>8}  {'mar_std':>7}")
    print(f"{'─' * 8}  {'─' * 6}  {'─' * 6}  {'─' * 5}  {'─' * 3}  "
          f"{'─' * 5}  {'─' * 5}  {'─' * 6}  {'─' * 7}  {'─' * 8}  {'─' * 7}")

    for alpha in ALPHAS:
        torch.manual_seed(42)
        np.random.seed(42)

        # Run WLS solver at this alpha
        result = solve_ratings_wls(all_games, alpha=alpha, estimate_hca=True)

        # Patch val features with this alpha's ratings
        X_val = X_val_base.copy()
        for i in range(len(X_val)):
            h = result.get(int(home_ids[i]), {})
            a = result.get(int(away_ids[i]), {})
            if h:
                X_val[i, col_idx["home_team_adj_oe"]] = h.get("adj_oe", X_val[i, col_idx["home_team_adj_oe"]])
                X_val[i, col_idx["home_team_adj_de"]] = h.get("adj_de", X_val[i, col_idx["home_team_adj_de"]])
                X_val[i, col_idx["home_team_adj_pace"]] = h.get("adj_tempo", X_val[i, col_idx["home_team_adj_pace"]])
                X_val[i, col_idx["home_sos_oe"]] = h.get("sos_oe", X_val[i, col_idx["home_sos_oe"]])
                X_val[i, col_idx["home_sos_de"]] = h.get("sos_de", X_val[i, col_idx["home_sos_de"]])
            if a:
                X_val[i, col_idx["away_team_adj_oe"]] = a.get("adj_oe", X_val[i, col_idx["away_team_adj_oe"]])
                X_val[i, col_idx["away_team_adj_de"]] = a.get("adj_de", X_val[i, col_idx["away_team_adj_de"]])
                X_val[i, col_idx["away_team_adj_pace"]] = a.get("adj_tempo", X_val[i, col_idx["away_team_adj_pace"]])
                X_val[i, col_idx["away_sos_oe"]] = a.get("sos_oe", X_val[i, col_idx["away_sos_oe"]])
                X_val[i, col_idx["away_sos_de"]] = a.get("sos_de", X_val[i, col_idx["away_sos_de"]])

        X_val = impute_column_means(X_val)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_tr_s = scaler.transform(X_train).astype(np.float32)
        X_v_s = scaler.transform(X_val).astype(np.float32)

        mu, sigma, best_ep = train_eval(X_tr_s, y_train, X_v_s, y_val)

        mae = np.mean(np.abs(mu - y_val))
        rmse = np.sqrt(np.mean((mu - y_val) ** 2))
        avg_sig = np.mean(sigma)

        # ATS
        has_book = ~np.isnan(book_arr)
        bs = book_arr[has_book]
        edge = mu[has_book] + bs
        pick_home = edge >= 0
        actual = y_val[has_book]
        home_covers = actual + bs > 0
        pushes = np.abs(actual + bs) < 0.5
        prob_edge = normal_cdf(np.abs(edge) / sigma[has_book]) - 0.5
        filt = prob_edge >= 0.05
        nf = filt.sum()
        if nf > 0:
            pw = np.where(pick_home[filt], home_covers[filt], ~home_covers[filt]) & ~pushes[filt]
            w = int(pw.sum())
            l = int(nf - pushes[filt].sum() - w)
            wr = w / (w + l) if (w + l) > 0 else 0
            roi = (w * 100 - l * 110) / (nf * 110) if nf > 0 else 0
        else:
            w, l, wr, roi = 0, 0, 0, 0

        margins = [r["adj_oe"] - r["adj_de"] for r in result.values()]
        top = max(margins)
        mar_std = np.std(margins)

        print(f"{alpha:>8.2f}  {mae:>6.2f}  {rmse:>6.2f}  {avg_sig:>5.1f}  {best_ep:>3}  "
              f"{w:>5}  {l:>5}  {wr:>5.1%}  {roi:>+6.1%}  {top:>+8.1f}  {mar_std:>7.1f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()

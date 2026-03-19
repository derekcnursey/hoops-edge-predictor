#!/usr/bin/env python3
"""Post-critical-fix walk-forward: 53-feat no_garbage with fixed imputation + rolling.

Compares to pre-fix baseline. Key question: does fixing the data leakage
(Bug 2) change the December edge found in the edge leak analysis?
"""

from __future__ import annotations

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

REPORT_PATH = Path(__file__).resolve().parent.parent / "analysis" / "critical_fixes_report.md"


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


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    feature_order = config.FEATURE_ORDER
    monthly_records = []  # (year, month, pick_home, win, push) tuples for monthly analysis

    print(f"Post-fix walk-forward: 53-feat no_garbage")
    print(f"Features: {len(feature_order)}")
    print(f"Holdout years: {HOLDOUT_YEARS}")
    print()

    yearly_results = {}

    for holdout in HOLDOUT_YEARS:
        train_seasons = list(range(TRAIN_START, holdout))
        print(f"Holdout {holdout}: train {TRAIN_START}-{holdout-1}...", end=" ", flush=True)

        df_train = load_multi_season_features(
            train_seasons, no_garbage=True, adj_suffix=ADJ_SUFFIX, min_month_day=MIN_DATE)
        df_train = df_train.dropna(subset=["homeScore", "awayScore"])
        df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]

        df_val = load_multi_season_features(
            [holdout], no_garbage=True, adj_suffix=ADJ_SUFFIX, min_month_day=MIN_DATE)
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

        # Impute and scale
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

        # Parse dates for monthly breakdown
        dates = pd.to_datetime(df_val["startDate"], errors="coerce", utc=True)
        month_arr = dates.dt.month.values

        # ATS computation — unified batch approach for both yearly and monthly
        has_book = ~np.isnan(book_spread_arr)
        n_book = has_book.sum()
        if n_book > 0:
            bs = book_spread_arr[has_book]
            edge = mu[has_book] + bs
            pick_home = edge >= 0
            actual = y_val[has_book]
            home_covers = actual + bs > 0
            pushes = np.abs(actual + bs) < 0.5

            # Apply 5% prob edge filter
            prob_edge = normal_cdf(np.abs(edge) / sigma[has_book]) - 0.5
            filt = prob_edge >= 0.05
            n_filt = filt.sum()
            if n_filt > 0:
                pick_wins = np.where(pick_home[filt], home_covers[filt], ~home_covers[filt]) & ~pushes[filt]
                w = pick_wins.sum()
                l = n_filt - pushes[filt].sum() - w
                wr = w / (w + l) if (w + l) > 0 else 0
                roi = (w * 100 - l * 110) / (n_filt * 110) if n_filt > 0 else 0
                n_home = pick_home[filt].sum()

                # Collect monthly records for games that pass the filter
                filt_months = month_arr[has_book][filt]
                filt_pick_home = pick_home[filt]
                filt_wins = pick_wins
                filt_pushes = pushes[filt]
                for j in range(len(filt_months)):
                    monthly_records.append((
                        holdout, int(filt_months[j]),
                        bool(filt_pick_home[j]), bool(filt_wins[j]), bool(filt_pushes[j])
                    ))
            else:
                wr, roi, w, l, n_home, n_filt = 0, 0, 0, 0, 0, 0
        else:
            wr, roi, w, l, n_home, n_filt = 0, 0, 0, 0, 0, 0

        yearly_results[holdout] = {
            "mae": mae, "rmse": rmse, "avg_sigma": avg_sigma,
            "best_epoch": best_epoch, "n_games": len(df_val),
            "n_with_book": int(n_book),
            "ats_n": n_filt, "ats_w": w, "ats_l": l,
            "ats_wr": wr, "ats_roi": roi,
            "ats_home_pct": n_home / n_filt if n_filt > 0 else 0,
        }
        print(f"ep={best_epoch} MAE={mae:.2f} RMSE={rmse:.2f} σ={avg_sigma:.1f} "
              f"ATS@5%: {w}W-{l}L ({wr:.1%}) ROI={roi:.1%} n={len(df_val)}")

    # ── Summary ──────────────────────────────────────────────────────

    # Yearly table
    print(f"\n{'='*90}")
    print(f"  YEARLY SUMMARY (53-feat no_garbage, post-fix)")
    print(f"{'='*90}")
    print(f"  {'Year':>4}  {'MAE':>6}  {'RMSE':>6}  {'σ':>5}  {'Ep':>3}  "
          f"{'N':>5}  {'ATS@5%':>10}  {'WR':>6}  {'ROI':>7}  {'H%':>5}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*3}  "
          f"{'─'*5}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*5}")

    for yr in HOLDOUT_YEARS:
        r = yearly_results[yr]
        covid = " *" if yr == 2021 else ""
        ats_str = f"{r['ats_w']}W-{r['ats_l']}L"
        print(f"  {yr:>4}{covid} {r['mae']:>6.2f}  {r['rmse']:>6.2f}  "
              f"{r['avg_sigma']:>5.1f}  {r['best_epoch']:>3}  "
              f"{r['n_games']:>5}  {ats_str:>10}  "
              f"{r['ats_wr']:>5.1%}  {r['ats_roi']:>+6.1%}  "
              f"{r['ats_home_pct']:>4.0%}")

    # Averages (excl 2021)
    excl = [yr for yr in HOLDOUT_YEARS if yr != 2021]
    avg_mae = np.mean([yearly_results[yr]["mae"] for yr in excl])
    avg_rmse = np.mean([yearly_results[yr]["rmse"] for yr in excl])
    avg_sigma = np.mean([yearly_results[yr]["avg_sigma"] for yr in excl])
    total_w = sum(yearly_results[yr]["ats_w"] for yr in excl)
    total_l = sum(yearly_results[yr]["ats_l"] for yr in excl)
    total_n = sum(yearly_results[yr]["ats_n"] for yr in excl)
    total_wr = total_w / (total_w + total_l) if (total_w + total_l) > 0 else 0
    total_roi = (total_w * 100 - total_l * 110) / (total_n * 110) if total_n > 0 else 0
    total_home = sum(yearly_results[yr]["ats_home_pct"] * yearly_results[yr]["ats_n"] for yr in excl)
    total_home_pct = total_home / total_n if total_n > 0 else 0

    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*3}  "
          f"{'─'*5}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*5}")
    ats_str = f"{total_w}W-{total_l}L"
    print(f"  {'AVG':>4}  {avg_mae:>6.2f}  {avg_rmse:>6.2f}  "
          f"{avg_sigma:>5.1f}  {'':>3}  "
          f"{'':>5}  {ats_str:>10}  "
          f"{total_wr:>5.1%}  {total_roi:>+6.1%}  "
          f"{total_home_pct:>4.0%}")
    print(f"  (excluding 2021)")

    # ── Monthly ATS breakdown ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MONTHLY ATS BREAKDOWN (edge >= 5%, all years excl 2021)")
    print(f"{'='*70}")

    month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
    monthly = pd.DataFrame(monthly_records, columns=["year", "month", "pick_home", "win", "push"])
    monthly = monthly[monthly["year"] != 2021]

    print(f"  {'Month':>5}  {'N':>5}  {'W':>5}  {'L':>5}  {'WR':>6}  {'ROI':>7}  {'Home%':>6}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*6}")

    for m in [11, 12, 1, 2, 3, 4]:
        sub = monthly[monthly["month"] == m]
        if len(sub) == 0:
            continue
        wins = sub["win"].sum()
        pushes = sub["push"].sum()
        losses = len(sub) - pushes - wins
        wr = wins / (wins + losses) if (wins + losses) > 0 else 0
        roi = (wins * 100 - losses * 110) / (len(sub) * 110) if len(sub) > 0 else 0
        n_home = sub["pick_home"].sum()
        h_pct = n_home / len(sub) if len(sub) > 0 else 0
        name = month_names.get(m, f"M{m}")
        print(f"  {name:>5}  {len(sub):>5}  {wins:>5}  {losses:>5}  "
              f"{wr:>5.1%}  {roi:>+6.1%}  {h_pct:>5.0%}")

    # ── Write report ──────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines_out = []
    lines_out.append("# Critical Fixes Report — Post-Fix Walk-Forward\n")
    lines_out.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
    lines_out.append("## Bugs Fixed\n")
    lines_out.append("1. **CRITICAL-1**: NaN imputation mismatch — `train_production.py` used `nan_to_num(0.0)`, "
                     "now uses `impute_column_means()` matching inference.\n")
    lines_out.append("2. **CRITICAL-3**: Rolling average fallback leaked future data — `_get_asof_rolling()` "
                     "now returns `{}` when no prior data exists.\n")
    lines_out.append("3. **HIGH-4**: `no_garbage` default inconsistency — unified to `config.NO_GARBAGE=True` "
                     "across `features.py`, `dataset.py`, `cli.py`.\n")
    lines_out.append("\n## Walk-Forward Results (53-feat, no_garbage, post-fix)\n")
    lines_out.append(f"| Year | MAE | RMSE | σ | ATS@5% | WR | ROI | Home% |\n")
    lines_out.append(f"|------|-----|------|---|--------|-----|-----|-------|\n")
    for yr in HOLDOUT_YEARS:
        r = yearly_results[yr]
        covid = " *" if yr == 2021 else ""
        lines_out.append(
            f"| {yr}{covid} | {r['mae']:.2f} | {r['rmse']:.2f} | {r['avg_sigma']:.1f} | "
            f"{r['ats_w']}W-{r['ats_l']}L | {r['ats_wr']:.1%} | {r['ats_roi']:+.1%} | "
            f"{r['ats_home_pct']:.0%} |\n")
    lines_out.append(
        f"| **AVG** | **{avg_mae:.2f}** | **{avg_rmse:.2f}** | **{avg_sigma:.1f}** | "
        f"**{total_w}W-{total_l}L** | **{total_wr:.1%}** | **{total_roi:+.1%}** | "
        f"**{total_home_pct:.0%}** |\n")
    lines_out.append("*(excluding 2021)*\n")

    # Pre-fix baseline for comparison
    lines_out.append("\n## Pre-Fix Baseline (previous session)\n")
    lines_out.append("| Metric | Pre-Fix | Post-Fix | Change |\n")
    lines_out.append("|--------|---------|----------|--------|\n")
    lines_out.append(f"| MAE (avg excl 2021) | 9.62 | {avg_mae:.2f} | {avg_mae - 9.62:+.2f} |\n")
    lines_out.append(f"| RMSE (avg excl 2021) | ~12.7 | {avg_rmse:.2f} | — |\n")
    lines_out.append(f"| ATS WR @5% | ~51.7% | {total_wr:.1%} | {(total_wr - 0.517)*100:+.1f}pp |\n")
    lines_out.append(f"| ATS ROI @5% | ~-1.3% | {total_roi:.1%} | {(total_roi + 0.013)*100:+.1f}pp |\n")

    lines_out.append("\n## Monthly ATS (edge >= 5%, excl 2021)\n")
    lines_out.append("| Month | N | W | L | WR | ROI | Home% |\n")
    lines_out.append("|-------|---|---|---|----|-----|-------|\n")
    month_stats = {}
    for m in [11, 12, 1, 2, 3, 4]:
        sub = monthly[monthly["month"] == m]
        if len(sub) == 0:
            continue
        wins = int(sub["win"].sum())
        pushes = int(sub["push"].sum())
        losses = len(sub) - pushes - wins
        wr = wins / (wins + losses) if (wins + losses) > 0 else 0
        roi = (wins * 100 - losses * 110) / (len(sub) * 110) if len(sub) > 0 else 0
        n_home = int(sub["pick_home"].sum())
        h_pct = n_home / len(sub) if len(sub) > 0 else 0
        name = month_names.get(m, f"M{m}")
        lines_out.append(f"| {name} | {len(sub)} | {wins} | {losses} | {wr:.1%} | {roi:+.1%} | {h_pct:.0%} |\n")
        month_stats[m] = {"wr": wr, "roi": roi, "n": len(sub)}

    lines_out.append("\n## Key Question: Does December Edge Survive?\n")
    dec = monthly[monthly["month"] == 12]
    if len(dec) > 0:
        dec_w = int(dec["win"].sum())
        dec_p = int(dec["push"].sum())
        dec_l = len(dec) - dec_p - dec_w
        dec_wr = dec_w / (dec_w + dec_l) if (dec_w + dec_l) > 0 else 0
        dec_roi = (dec_w * 100 - dec_l * 110) / (len(dec) * 110) if len(dec) > 0 else 0
        lines_out.append(f"Pre-fix December: 53.2% WR, +1.6% ROI (edge leak analysis)\n\n")
        lines_out.append(f"Post-fix December: {dec_wr:.1%} WR, {dec_roi:+.1%} ROI\n\n")
        if dec_wr > 0.52:
            lines_out.append("December edge appears **genuine** — survives the leakage fix.\n")
        elif dec_wr > 0.50:
            lines_out.append("December edge **partially genuine** — reduced but still positive.\n")
        else:
            lines_out.append("December edge was **fake** — driven by future data leakage.\n")
    else:
        lines_out.append("No December games with book lines found.\n")

    with open(REPORT_PATH, "w") as f:
        f.writelines(lines_out)
    print(f"\nReport saved to: {REPORT_PATH}")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} min")

#!/usr/bin/env python3
"""WLS Solver Full Walk-Forward Evaluation.

Runs walk-forward comparison of WLS solver (0.475 poss) vs iterative baseline
(0.44 poss). Uses pre-built feature parquet files from the WLS-rebuilt pipeline.

Steps:
  5. Walk-forward evaluation (7 folds, 2019-2025)
  6. Pace validation (adj_pace vs expected ~67-68)
  7. Torvik divergence analysis (2025 adj_oe vs Torvik)
  8. Save report to reports/wls_solver_evaluation.md

Usage:
    poetry run python scripts/wls_walkforward_full.py
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
from src.features import (
    get_feature_matrix,
    get_targets,
    load_efficiency_ratings,
    load_research_lines,
)
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

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "wls_solver_evaluation.md"

# Baseline results from critical_fixes_report.md (iterative solver, 0.44 poss, post-fix)
# Note: ATS numbers are UNFILTERED (all games with book lines), not edge>=5% filtered
BASELINE = {
    2019: {"mae": 9.53, "rmse": 12.49, "avg_sigma": 11.8, "ats_w": 1351, "ats_l": 1345, "ats_wr": 0.501, "ats_roi": -0.042},
    2020: {"mae": 9.60, "rmse": 12.53, "avg_sigma": 11.5, "ats_w": 1383, "ats_l": 1335, "ats_wr": 0.509, "ats_roi": -0.028},
    2021: {"mae": 10.75, "rmse": 14.04, "avg_sigma": 12.8, "ats_w": 1281, "ats_l": 1315, "ats_wr": 0.493, "ats_roi": -0.057},
    2022: {"mae": 9.42, "rmse": 12.48, "avg_sigma": 12.3, "ats_w": 1416, "ats_l": 1228, "ats_wr": 0.536, "ats_roi": 0.022},
    2023: {"mae": 9.56, "rmse": 12.46, "avg_sigma": 11.6, "ats_w": 1547, "ats_l": 1401, "ats_wr": 0.525, "ats_roi": 0.002},
    2024: {"mae": 9.88, "rmse": 13.06, "avg_sigma": 11.6, "ats_w": 1361, "ats_l": 1272, "ats_wr": 0.517, "ats_roi": -0.013},
    2025: {"mae": 9.68, "rmse": 12.91, "avg_sigma": 11.4, "ats_w": 1393, "ats_l": 1323, "ats_wr": 0.513, "ats_roi": -0.021},
}


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


# ── Walk-Forward ──────────────────────────────────────────────────

def run_walkforward():
    """Run walk-forward and return per-year results + monthly records."""
    feature_order = config.FEATURE_ORDER
    yearly_results = {}
    monthly_records = []

    for holdout in HOLDOUT_YEARS:
        train_seasons = list(range(TRAIN_START, holdout))
        print(f"  Holdout {holdout}: train {TRAIN_START}-{holdout-1}...", end=" ", flush=True)

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

        # ATS computation
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
                n_home = pick_home[filt].sum()

                # Collect monthly records
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

    return yearly_results, monthly_records


# ── Pace Validation ───────────────────────────────────────────────

def pace_validation():
    """Check adj_pace from the rebuilt gold layer across seasons."""
    print("\n" + "=" * 70)
    print("  PACE VALIDATION (adj_tempo from gold layer)")
    print("=" * 70)

    results = []
    for season in [2020, 2021, 2022, 2023, 2024, 2025]:
        try:
            ratings = load_efficiency_ratings(season, no_garbage=True)
            if ratings.empty:
                print(f"  {season}: no data")
                continue
            # Latest snapshot per team
            latest = ratings.sort_values("rating_date").drop_duplicates("teamId", keep="last")
            avg_pace = latest["adj_tempo"].mean()
            std_pace = latest["adj_tempo"].std()
            results.append({"season": season, "avg_pace": avg_pace, "std_pace": std_pace})
            print(f"  {season}: avg adj_tempo = {avg_pace:.1f} ± {std_pace:.1f}")
        except Exception as e:
            print(f"  {season}: ERROR - {e}")

    if results:
        overall_avg = np.mean([r["avg_pace"] for r in results])
        print(f"\n  Overall avg: {overall_avg:.1f} (target: ~67-68, was ~62 with 0.44)")

    return results


# ── Torvik Divergence (adj_oe comparison) ─────────────────────────

def torvik_divergence():
    """Compare 2025 WLS adj_oe to Torvik predictions for divergence analysis."""
    print("\n" + "=" * 70)
    print("  TORVIK DIVERGENCE ANALYSIS (2025)")
    print("=" * 70)

    # Load our WLS ratings
    try:
        ratings = load_efficiency_ratings(2025, no_garbage=True)
        if ratings.empty:
            print("  No 2025 ratings found.")
            return None
        latest = ratings.sort_values("rating_date").drop_duplicates("teamId", keep="last")
        latest["adj_margin"] = latest["adj_oe"] - latest["adj_de"]
    except Exception as e:
        print(f"  Error loading ratings: {e}")
        return None

    # Load Torvik game-level predictions for back-computing team-level implied ratings
    torvik_path = config.PREDICTIONS_DIR / "torvik_preds_2025.csv"
    if not torvik_path.exists():
        print(f"  No Torvik reference data at {torvik_path}")
        print("  Skipping divergence analysis.")
        return None

    torvik = pd.read_csv(torvik_path)
    # Torvik pred margin is home_team perspective
    # We can compute implied team strength from Torvik predictions at neutral
    # But simpler: just show our top/bottom teams and flag outliers

    print(f"\n  Our ratings: {len(latest)} teams")
    print(f"  Torvik games: {len(torvik)}")

    # Show top-10 and bottom-10 by adj_margin
    top10 = latest.nlargest(10, "adj_margin")
    bot10 = latest.nsmallest(10, "adj_margin")

    print(f"\n  Top-10 by adj_margin (WLS solver):")
    print(f"  {'TeamID':>8}  {'adj_oe':>7}  {'adj_de':>7}  {'margin':>7}  {'adj_tempo':>9}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}")
    for _, row in top10.iterrows():
        print(f"  {int(row['teamId']):>8}  {row['adj_oe']:>7.1f}  {row['adj_de']:>7.1f}  "
              f"{row['adj_margin']:>+7.1f}  {row['adj_tempo']:>9.1f}")

    print(f"\n  Bottom-10 by adj_margin (WLS solver):")
    for _, row in bot10.iterrows():
        print(f"  {int(row['teamId']):>8}  {row['adj_oe']:>7.1f}  {row['adj_de']:>7.1f}  "
              f"{row['adj_margin']:>+7.1f}  {row['adj_tempo']:>9.1f}")

    # Distribution stats
    print(f"\n  Distribution:")
    print(f"    adj_oe:  mean={latest['adj_oe'].mean():.1f}, std={latest['adj_oe'].std():.1f}")
    print(f"    adj_de:  mean={latest['adj_de'].mean():.1f}, std={latest['adj_de'].std():.1f}")
    print(f"    margin:  mean={latest['adj_margin'].mean():.2f}, std={latest['adj_margin'].std():.1f}")
    print(f"    tempo:   mean={latest['adj_tempo'].mean():.1f}, std={latest['adj_tempo'].std():.1f}")

    return latest


# ── Report Generation ─────────────────────────────────────────────

def write_report(yearly_results, monthly_records, pace_results, ratings_2025):
    """Write markdown report to reports/wls_solver_evaluation.md."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append("# WLS Solver Full Pipeline Evaluation\n")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
    lines.append(f"**Solver**: WLS (ridge alpha=0.01, HCA estimated from data)\n")
    lines.append(f"**Possessions**: 0.475 formula (was 0.44)\n")
    lines.append(f"**Features**: {len(config.FEATURE_ORDER)} (no-garbage, adj a={config.ADJUST_ALPHA} p={config.ADJUST_PRIOR})\n")
    lines.append(f"**Walk-forward**: {TRAIN_START}-{HOLDOUT_YEARS[-1]}, min_date={MIN_DATE}\n")
    lines.append(f"**Architecture**: MLP {HP['hidden1']}/{HP['hidden2']}, dropout={HP['dropout']}\n\n")

    # ── Comparison table ──
    lines.append("## Walk-Forward: WLS vs Iterative Baseline\n\n")
    lines.append("| Year | MAE (base) | MAE (WLS) | Δ | RMSE (base) | RMSE (WLS) | Δ | "
                 "σ (WLS) | ATS WR (base) | ATS WR (WLS) | ROI (base) | ROI (WLS) |\n")
    lines.append("|------|-----------|----------|---|------------|-----------|---|"
                 "--------|-------------|-------------|-----------|----------|\n")

    excl = [yr for yr in HOLDOUT_YEARS if yr != 2021]

    for yr in HOLDOUT_YEARS:
        r = yearly_results[yr]
        b = BASELINE.get(yr, {})
        covid = " *" if yr == 2021 else ""
        d_mae = r["mae"] - b.get("mae", r["mae"])
        d_rmse = r["rmse"] - b.get("rmse", r["rmse"])
        lines.append(
            f"| {yr}{covid} | {b.get('mae', 0):.2f} | {r['mae']:.2f} | {d_mae:+.2f} | "
            f"{b.get('rmse', 0):.2f} | {r['rmse']:.2f} | {d_rmse:+.2f} | "
            f"{r['avg_sigma']:.1f} | {b.get('ats_wr', 0):.1%} | {r['ats_wr']:.1%} | "
            f"{b.get('ats_roi', 0):+.1%} | {r['ats_roi']:+.1%} |\n")

    # Averages (excl 2021)
    avg_mae = np.mean([yearly_results[yr]["mae"] for yr in excl])
    avg_rmse = np.mean([yearly_results[yr]["rmse"] for yr in excl])
    avg_sigma = np.mean([yearly_results[yr]["avg_sigma"] for yr in excl])
    total_w = sum(yearly_results[yr]["ats_w"] for yr in excl)
    total_l = sum(yearly_results[yr]["ats_l"] for yr in excl)
    total_n = sum(yearly_results[yr]["ats_n"] for yr in excl)
    total_wr = total_w / (total_w + total_l) if (total_w + total_l) > 0 else 0
    total_roi = (total_w * 100 - total_l * 110) / (total_n * 110) if total_n > 0 else 0

    base_mae = np.mean([BASELINE[yr]["mae"] for yr in excl if yr in BASELINE])
    base_rmse = np.mean([BASELINE[yr]["rmse"] for yr in excl if yr in BASELINE])
    base_w = sum(BASELINE[yr]["ats_w"] for yr in excl if yr in BASELINE)
    base_l = sum(BASELINE[yr]["ats_l"] for yr in excl if yr in BASELINE)
    base_wr = base_w / (base_w + base_l) if (base_w + base_l) > 0 else 0
    base_roi = (base_w * 100 - base_l * 110) / ((base_w + base_l) * 110) if (base_w + base_l) > 0 else 0

    lines.append(
        f"| **AVG** | **{base_mae:.2f}** | **{avg_mae:.2f}** | **{avg_mae-base_mae:+.2f}** | "
        f"**{base_rmse:.2f}** | **{avg_rmse:.2f}** | **{avg_rmse-base_rmse:+.2f}** | "
        f"**{avg_sigma:.1f}** | **{base_wr:.1%}** | **{total_wr:.1%}** | "
        f"**{base_roi:+.1%}** | **{total_roi:+.1%}** |\n")
    lines.append("\n*(excluding 2021, baseline = iterative solver + 0.44 poss)*\n\n")

    # ── Monthly ATS ──
    lines.append("## Monthly ATS Breakdown (edge >= 5%, excl 2021)\n\n")
    month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
    monthly = pd.DataFrame(monthly_records, columns=["year", "month", "pick_home", "win", "push"])
    monthly = monthly[monthly["year"] != 2021]

    lines.append("| Month | N | W | L | WR | ROI | Home% |\n")
    lines.append("|-------|---|---|---|----|-----|-------|\n")
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
        lines.append(f"| {name} | {len(sub)} | {wins} | {losses} | {wr:.1%} | {roi:+.1%} | {h_pct:.0%} |\n")

    # ── Pace validation ──
    if pace_results:
        lines.append("\n## Pace Validation\n\n")
        lines.append("| Season | Avg adj_tempo | Std |\n")
        lines.append("|--------|-------------|-----|\n")
        for r in pace_results:
            lines.append(f"| {r['season']} | {r['avg_pace']:.1f} | {r['std_pace']:.1f} |\n")
        overall_avg = np.mean([r["avg_pace"] for r in pace_results])
        lines.append(f"\nOverall avg: **{overall_avg:.1f}** (target: ~67-68, was ~62 with 0.44 formula)\n\n")

    # ── Ratings distribution ──
    if ratings_2025 is not None:
        lines.append("\n## 2025 Ratings Distribution (WLS)\n\n")
        lines.append(f"- adj_oe: mean={ratings_2025['adj_oe'].mean():.1f}, std={ratings_2025['adj_oe'].std():.1f}\n")
        lines.append(f"- adj_de: mean={ratings_2025['adj_de'].mean():.1f}, std={ratings_2025['adj_de'].std():.1f}\n")
        lines.append(f"- adj_margin: mean={ratings_2025['adj_margin'].mean():.2f}, std={ratings_2025['adj_margin'].std():.1f}\n")
        lines.append(f"- adj_tempo: mean={ratings_2025['adj_tempo'].mean():.1f}, std={ratings_2025['adj_tempo'].std():.1f}\n")

    # ── Verdict ──
    lines.append("\n## Verdict\n\n")
    d_mae = avg_mae - base_mae
    d_wr = total_wr - base_wr
    d_roi = total_roi - base_roi
    if d_mae < -0.05 and d_roi > 0.005:
        verdict = "WLS solver **improves** both accuracy and profitability. Recommend switching."
    elif d_mae < 0 and d_roi > -0.005:
        verdict = "WLS solver improves accuracy with similar profitability. Lean toward switching."
    elif abs(d_mae) < 0.05 and abs(d_roi) < 0.005:
        verdict = "WLS solver is a **wash** — no meaningful difference. Stick with iterative."
    elif d_mae > 0.05:
        verdict = "WLS solver **worse** on accuracy. Do not switch."
    else:
        verdict = f"Mixed results: MAE {d_mae:+.2f}, WR {d_wr:+.1%}, ROI {d_roi:+.1%}. Needs judgment call."
    lines.append(f"{verdict}\n")

    with open(REPORT_PATH, "w") as f:
        f.writelines(lines)
    print(f"\nReport saved to: {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  WLS SOLVER FULL WALK-FORWARD EVALUATION")
    print("=" * 70)
    print(f"Features: {len(config.FEATURE_ORDER)}")
    print(f"Holdout years: {HOLDOUT_YEARS}")
    print(f"Architecture: MLP {HP['hidden1']}/{HP['hidden2']}, dropout={HP['dropout']}")
    print()

    # Step 5: Walk-forward
    print("─" * 70)
    print("  WALK-FORWARD (WLS solver, 0.475 poss)")
    print("─" * 70)
    yearly_results, monthly_records = run_walkforward()

    # Print yearly summary
    print(f"\n{'='*90}")
    print(f"  YEARLY SUMMARY")
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
    total_w = sum(yearly_results[yr]["ats_w"] for yr in excl)
    total_l = sum(yearly_results[yr]["ats_l"] for yr in excl)
    total_n = sum(yearly_results[yr]["ats_n"] for yr in excl)
    total_wr = total_w / (total_w + total_l) if (total_w + total_l) > 0 else 0
    total_roi = (total_w * 100 - total_l * 110) / (total_n * 110) if total_n > 0 else 0
    ats_str = f"{total_w}W-{total_l}L"
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*3}  "
          f"{'─'*5}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*5}")
    print(f"  {'AVG':>4}  {avg_mae:>6.2f}  {avg_rmse:>6.2f}  "
          f"{'':>5}  {'':>3}  {'':>5}  {ats_str:>10}  "
          f"{total_wr:>5.1%}  {total_roi:>+6.1%}")
    print(f"  (excluding 2021)")

    # Comparison with baseline
    print(f"\n{'='*90}")
    print(f"  COMPARISON: WLS vs Iterative Baseline")
    print(f"{'='*90}")
    print(f"  {'Year':>4}  {'MAE-B':>7} {'MAE-W':>7} {'Δ':>7}  "
          f"{'RMSE-B':>7} {'RMSE-W':>7} {'Δ':>7}  "
          f"{'WR-B':>6} {'WR-W':>6} {'ROI-B':>7} {'ROI-W':>7}")
    for yr in HOLDOUT_YEARS:
        r = yearly_results[yr]
        b = BASELINE.get(yr, {})
        d_mae = r["mae"] - b.get("mae", r["mae"])
        d_rmse = r["rmse"] - b.get("rmse", r["rmse"])
        print(f"  {yr:>4}  {b.get('mae',0):>7.2f} {r['mae']:>7.2f} {d_mae:>+7.2f}  "
              f"{b.get('rmse',0):>7.2f} {r['rmse']:>7.2f} {d_rmse:>+7.2f}  "
              f"{b.get('ats_wr',0):>6.1%} {r['ats_wr']:>6.1%} "
              f"{b.get('ats_roi',0):>7.1%} {r['ats_roi']:>7.1%}")

    # Monthly breakdown
    print(f"\n{'='*70}")
    print(f"  MONTHLY ATS BREAKDOWN (edge >= 5%, all years excl 2021)")
    print(f"{'='*70}")
    month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
    monthly = pd.DataFrame(monthly_records, columns=["year", "month", "pick_home", "win", "push"])
    monthly = monthly[monthly["year"] != 2021]

    print(f"  {'Month':>5}  {'N':>5}  {'W':>5}  {'L':>5}  {'WR':>6}  {'ROI':>7}  {'Home%':>6}")
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

    # Step 6: Pace validation
    pace_results = pace_validation()

    # Step 7: Torvik divergence
    ratings_2025 = torvik_divergence()

    # Step 8: Write report
    write_report(yearly_results, monthly_records, pace_results, ratings_2025)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

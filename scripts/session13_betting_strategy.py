#!/usr/bin/env python3
"""Session 13: Betting strategy exploration for calibrated exp-sigma model.

Uses the C2-V2 winner (384→256, exp-sigma) to explore profitable betting strategies
using the 2026 validation set.

Analyses:
  1. Prob_edge as primary filter (no sigma filter)
  2. Prob_edge + sigma band combinations
  3. Kelly criterion bankroll simulation
  4. Optimal strategy identification (max ROI with ≥100 bets/season)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor
from src.dataset import load_multi_season_features
from src.features import get_feature_matrix, get_targets, load_lines
from src.trainer import impute_column_means
from sklearn.preprocessing import StandardScaler
import pickle


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"


def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def load_model_and_data():
    """Load C2-V2 winner model and 2026 validation data."""
    # Load checkpoint
    ckpt = torch.load(config.PROJECT_ROOT / "checkpoints" / "regressor.pt",
                      map_location="cpu", weights_only=False)
    hp = ckpt["hparams"]
    model = MLPRegressor(
        input_dim=50,
        hidden1=hp["hidden1"],
        hidden2=hp["hidden2"],
        dropout=hp["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load scaler
    with open(config.PROJECT_ROOT / "artifacts" / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load validation data
    df_val = load_multi_season_features(
        [2026], adj_suffix=ADJ_SUFFIX, min_month_day="12-01"
    )
    df_val = df_val.dropna(subset=["homeScore", "awayScore"])
    df_val = df_val[(df_val["homeScore"] != 0) | (df_val["awayScore"] != 0)]

    # Merge book spreads
    try:
        lines_df = load_lines(2026)
        if not lines_df.empty:
            lines_dedup = lines_df.sort_values("provider").drop_duplicates(
                subset=["gameId"], keep="first"
            )
            if "spread" in lines_dedup.columns:
                merge_df = lines_dedup[["gameId", "spread"]].rename(
                    columns={"spread": "bookSpread"}
                )
                df_val = df_val.merge(merge_df, on="gameId", how="left")
    except Exception as e:
        print(f"  Lines load failed: {e}")

    X_val = get_feature_matrix(df_val).values.astype(np.float32)
    targets_val = get_targets(df_val)
    y_spread_val = targets_val["spread_home"].values.astype(np.float32)

    X_val = impute_column_means(X_val)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    # Run inference
    with torch.no_grad():
        X_tensor = torch.tensor(X_val_s, dtype=torch.float32)
        mu_t, log_sigma_t = model(X_tensor)
        sigma_t = torch.exp(log_sigma_t).clamp(min=0.5, max=30.0)
        mu = mu_t.numpy()
        sigma = sigma_t.numpy()

    actual = y_spread_val
    book_spread = df_val["bookSpread"].values.astype(np.float64) if "bookSpread" in df_val.columns else np.full(len(df_val), np.nan)
    has_book = ~np.isnan(book_spread)

    print(f"  Val games: {len(df_val)}")
    print(f"  Games with book spreads: {has_book.sum()}")
    print(f"  Sigma: mean={sigma.mean():.2f}, std={sigma.std():.2f}, "
          f"p25={np.percentile(sigma, 25):.2f}, p50={np.median(sigma):.2f}, "
          f"p75={np.percentile(sigma, 75):.2f}")

    return mu, sigma, actual, book_spread, has_book, df_val


def compute_edge_picks(mu, sigma, book_spread, has_book):
    """Compute edge probabilities and pick directions for all games with book spreads."""
    valid = has_book.copy()
    edge_home = mu[valid] + book_spread[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    home_cover_prob = normal_cdf(edge_z)
    away_cover_prob = 1.0 - home_cover_prob

    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, home_cover_prob, away_cover_prob)

    breakeven = 0.5238  # -110 juice
    prob_edge = pick_prob - breakeven

    return {
        "valid": valid,
        "edge_home": edge_home,
        "pick_home": pick_home,
        "pick_prob": pick_prob,
        "prob_edge": prob_edge,
        "sigma_valid": sigma[valid],
        "mu_valid": mu[valid],
        "book_valid": book_spread[valid],
        "actual_valid": None,  # set below
    }


def compute_roi_detailed(picks, actual, book_spread, has_book, threshold,
                         sigma_lo=None, sigma_hi=None):
    """Compute ATS ROI with optional sigma band filter.

    Returns dict with bets, wins, losses, pushes, win_rate, roi, units, avg_edge.
    """
    prob_edge = picks["prob_edge"]
    pick_home = picks["pick_home"]
    sigma_v = picks["sigma_valid"]

    bet_mask = prob_edge >= threshold
    if sigma_lo is not None:
        bet_mask = bet_mask & (sigma_v >= sigma_lo)
    if sigma_hi is not None:
        bet_mask = bet_mask & (sigma_v <= sigma_hi)

    n_bets = bet_mask.sum()
    if n_bets == 0:
        return {"bets": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "roi": 0.0, "units": 0.0, "avg_edge": 0.0, "avg_sigma": 0.0,
                "avg_prob": 0.0}

    actual_v = actual[has_book]
    book_v = book_spread[has_book]

    # Did the pick cover? (ATS: margin + spread > 0 means home covers)
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)

    bet_wins = pick_won[bet_mask]
    bet_edges = prob_edge[bet_mask]
    bet_probs = picks["pick_prob"][bet_mask]
    bet_sigmas = sigma_v[bet_mask]

    wins = bet_wins.sum()
    losses = n_bets - wins
    profit_per_1 = 100.0 / 110.0
    units = wins * profit_per_1 - losses
    roi = units / n_bets

    return {
        "bets": int(n_bets),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(wins / n_bets),
        "roi": float(roi),
        "units": float(units),
        "avg_edge": float(bet_edges.mean()),
        "avg_sigma": float(bet_sigmas.mean()),
        "avg_prob": float(bet_probs.mean()),
    }


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: PROB_EDGE AS PRIMARY FILTER
# ══════════════════════════════════════════════════════════════════════

def analysis_1(picks, actual, book_spread, has_book):
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: PROB_EDGE AS PRIMARY FILTER (no sigma filter)")
    print("=" * 70)
    print(f"\n  breakeven = 52.38% (-110 juice)")
    print(f"  prob_edge = pick_prob - 0.5238")
    print()

    thresholds = [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    print(f"  {'Threshold':>10} {'Bets':>6} {'W-L':>8} {'Win%':>7} {'ROI':>8} {'Units':>8} {'Avg Edge':>9} {'Avg σ':>7} {'Avg P':>7}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*9} {'-'*7} {'-'*7}")

    results = []
    for t in thresholds:
        r = compute_roi_detailed(picks, actual, book_spread, has_book, t)
        results.append((t, r))
        if r["bets"] > 0:
            print(f"  {t*100:>9.0f}% {r['bets']:>6} {r['wins']:>3}-{r['losses']:<4} "
                  f"{r['win_rate']*100:>6.1f}% {r['roi']*100:>7.1f}% {r['units']:>+8.1f} "
                  f"{r['avg_edge']*100:>8.1f}% {r['avg_sigma']:>7.2f} {r['avg_prob']*100:>6.1f}%")
        else:
            print(f"  {t*100:>9.0f}%      0   ---    ---      ---      ---       ---     ---     ---")

    return results


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: PROB_EDGE + SIGMA BAND COMBINATIONS
# ══════════════════════════════════════════════════════════════════════

def analysis_2(picks, actual, book_spread, has_book):
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: PROB_EDGE + SIGMA BAND COMBINATIONS")
    print("=" * 70)

    edge_thresholds = [0.08, 0.10, 0.12]
    sigma_bands = [
        (8.0, 11.0),
        (10.0, 13.0),
        (11.0, 14.0),
        (12.0, 16.0),
        (None, None),  # no filter (for comparison)
    ]

    all_results = []

    for et in edge_thresholds:
        print(f"\n  --- prob_edge >= {et*100:.0f}% ---")
        print(f"  {'Sigma Band':>12} {'Bets':>6} {'W-L':>8} {'Win%':>7} {'ROI':>8} {'Units':>8} {'Avg Edge':>9} {'Avg σ':>7}")
        print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*9} {'-'*7}")

        for slo, shi in sigma_bands:
            r = compute_roi_detailed(picks, actual, book_spread, has_book, et,
                                     sigma_lo=slo, sigma_hi=shi)
            label = f"{slo:.0f}-{shi:.0f}" if slo is not None else "none"
            all_results.append((et, slo, shi, r))

            if r["bets"] > 0:
                print(f"  {label:>12} {r['bets']:>6} {r['wins']:>3}-{r['losses']:<4} "
                      f"{r['win_rate']*100:>6.1f}% {r['roi']*100:>7.1f}% {r['units']:>+8.1f} "
                      f"{r['avg_edge']*100:>8.1f}% {r['avg_sigma']:>7.2f}")
            else:
                print(f"  {label:>12}      0   ---    ---      ---      ---       ---     ---")

    return all_results


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: KELLY CRITERION SIZING
# ══════════════════════════════════════════════════════════════════════

def analysis_3(picks, actual, book_spread, has_book, df_val):
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: KELLY CRITERION SIZING")
    print("=" * 70)
    print(f"\n  Strategy: prob_edge >= 10%, no sigma filter")
    print(f"  Odds: -110 → decimal 1.909 → b = 0.909")
    print(f"  Kelly fraction = (bp - q) / b where p = pick_prob, q = 1-p, b = 0.909")
    print(f"  Starting bankroll: 1000 units")
    print()

    prob_edge = picks["prob_edge"]
    pick_home = picks["pick_home"]
    pick_prob = picks["pick_prob"]

    threshold = 0.10
    bet_mask = prob_edge >= threshold
    n_bets = bet_mask.sum()
    if n_bets == 0:
        print("  No bets qualify.")
        return

    actual_v = actual[has_book]
    book_v = book_spread[has_book]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)

    # Get indices of bets in order (we need chronological order)
    # Use df_val date if available for ordering
    bet_indices = np.where(bet_mask)[0]

    b = 100.0 / 110.0  # profit per unit wagered at -110

    # Simulate three strategies: flat 1 unit, half Kelly, full Kelly
    bankrolls = {
        "flat_1u": [],
        "half_kelly": [],
        "full_kelly": [],
        "quarter_kelly": [],
    }
    br = {"flat_1u": 1000.0, "half_kelly": 1000.0, "full_kelly": 1000.0,
          "quarter_kelly": 1000.0}

    bet_details = []

    for idx in bet_indices:
        p = pick_prob[idx]
        q = 1.0 - p
        kelly_frac = max(0, (b * p - q) / b)
        won = pick_won[idx]

        # Flat: always 1 unit
        if won:
            br["flat_1u"] += b
        else:
            br["flat_1u"] -= 1.0

        # Full Kelly: wager = kelly_frac * bankroll
        for strat, frac in [("full_kelly", 1.0), ("half_kelly", 0.5), ("quarter_kelly", 0.25)]:
            wager = frac * kelly_frac * br[strat]
            wager = min(wager, br[strat])  # can't bet more than bankroll
            if won:
                br[strat] += wager * b
            else:
                br[strat] -= wager

        for k in bankrolls:
            bankrolls[k].append(br[k])

        bet_details.append({
            "idx": idx,
            "prob": p,
            "kelly_frac": kelly_frac,
            "won": bool(won),
        })

    print(f"  Total bets: {n_bets}")
    print(f"  Win rate: {sum(1 for d in bet_details if d['won'])/n_bets*100:.1f}%")
    print(f"  Kelly fraction stats: mean={np.mean([d['kelly_frac'] for d in bet_details]):.3f}, "
          f"max={np.max([d['kelly_frac'] for d in bet_details]):.3f}")
    print()
    print(f"  {'Strategy':>16} {'Final BR':>10} {'Return':>8} {'Max DD':>8} {'Peak':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    for strat_name, br_history in bankrolls.items():
        final = br_history[-1]
        ret = (final - 1000.0) / 1000.0
        peak = max(br_history)
        # Max drawdown
        running_peak = 1000.0
        max_dd = 0.0
        for val in br_history:
            if val > running_peak:
                running_peak = val
            dd = (running_peak - val) / running_peak
            if dd > max_dd:
                max_dd = dd
        label = strat_name.replace("_", " ").title()
        print(f"  {label:>16} {final:>10.1f} {ret*100:>+7.1f}% {max_dd*100:>7.1f}% {peak:>10.1f}")

    # Print quarterly breakdown for half-Kelly
    print(f"\n  Half-Kelly quarterly progression (starting 1000):")
    quarters = [n_bets // 4, n_bets // 2, 3 * n_bets // 4, n_bets]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    for i, (q, lbl) in enumerate(zip(quarters, labels)):
        if q > 0 and q <= len(bankrolls["half_kelly"]):
            val = bankrolls["half_kelly"][q - 1]
            print(f"    {lbl}: After bet {q:>4}: {val:>10.1f}")

    return bankrolls, bet_details


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: OPTIMAL STRATEGY (MAX ROI WITH ≥100 BETS)
# ══════════════════════════════════════════════════════════════════════

def analysis_4(picks, actual, book_spread, has_book):
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: OPTIMAL STRATEGY (max ROI with ≥100 bets/season)")
    print("=" * 70)
    print(f"\n  Scanning all combinations of prob_edge and sigma bands...")
    print(f"  Minimum: 100 bets (≈season-viable)")
    print()

    edge_thresholds = [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18]
    sigma_options = [
        (None, None, "none"),
        (8.0, 11.0, "8-11"),
        (8.0, 12.0, "8-12"),
        (9.0, 12.0, "9-12"),
        (9.0, 13.0, "9-13"),
        (10.0, 13.0, "10-13"),
        (10.0, 14.0, "10-14"),
        (11.0, 14.0, "11-14"),
        (11.0, 15.0, "11-15"),
        (12.0, 16.0, "12-16"),
        (12.0, 18.0, "12-18"),
        (None, 12.0, "<12"),
        (None, 13.0, "<13"),
        (None, 14.0, "<14"),
        (14.0, None, ">14"),
        (16.0, None, ">16"),
    ]

    all_combos = []
    for et in edge_thresholds:
        for slo, shi, slabel in sigma_options:
            r = compute_roi_detailed(picks, actual, book_spread, has_book, et,
                                     sigma_lo=slo, sigma_hi=shi)
            if r["bets"] >= 100:
                all_combos.append({
                    "edge": et,
                    "sigma_band": slabel,
                    "sigma_lo": slo,
                    "sigma_hi": shi,
                    **r,
                })

    # Sort by ROI descending
    all_combos.sort(key=lambda x: x["roi"], reverse=True)

    # Print top 20
    print(f"  {'Rank':>4} {'Edge':>6} {'σ Band':>8} {'Bets':>6} {'W-L':>8} {'Win%':>7} {'ROI':>8} {'Units':>8} {'Avg σ':>7}")
    print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for i, c in enumerate(all_combos[:20]):
        print(f"  {i+1:>4} {c['edge']*100:>5.0f}% {c['sigma_band']:>8} {c['bets']:>6} "
              f"{c['wins']:>3}-{c['losses']:<4} {c['win_rate']*100:>6.1f}% "
              f"{c['roi']*100:>7.1f}% {c['units']:>+8.1f} {c['avg_sigma']:>7.2f}")

    # Also find best per edge threshold
    print(f"\n  Best combo per edge threshold (≥100 bets):")
    print(f"  {'Edge':>6} {'σ Band':>8} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Units':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")
    seen_edges = set()
    for c in all_combos:
        e = c["edge"]
        if e not in seen_edges:
            seen_edges.add(e)
            print(f"  {e*100:>5.0f}% {c['sigma_band']:>8} {c['bets']:>6} "
                  f"{c['win_rate']*100:>6.1f}% {c['roi']*100:>7.1f}% {c['units']:>+8.1f}")

    # Minimum 50 bets analysis too
    print(f"\n  --- Relaxed: ≥50 bets ---")
    relaxed = []
    for et in edge_thresholds + [0.20, 0.22, 0.25]:
        for slo, shi, slabel in sigma_options:
            r = compute_roi_detailed(picks, actual, book_spread, has_book, et,
                                     sigma_lo=slo, sigma_hi=shi)
            if r["bets"] >= 50:
                relaxed.append({"edge": et, "sigma_band": slabel, **r})

    relaxed.sort(key=lambda x: x["roi"], reverse=True)
    print(f"  {'Rank':>4} {'Edge':>6} {'σ Band':>8} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Units':>8}")
    print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")
    for i, c in enumerate(relaxed[:15]):
        print(f"  {i+1:>4} {c['edge']*100:>5.0f}% {c['sigma_band']:>8} {c['bets']:>6} "
              f"{c['win_rate']*100:>6.1f}% {c['roi']*100:>7.1f}% {c['units']:>+8.1f}")

    return all_combos


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  SESSION 13: BETTING STRATEGY EXPLORATION")
    print("  Model: C2-V2 (384→256, exp-sigma, BS-MAE=9.129)")
    print("=" * 70)

    mu, sigma, actual, book_spread, has_book, df_val = load_model_and_data()
    picks = compute_edge_picks(mu, sigma, book_spread, has_book)

    # Print edge distribution
    pe = picks["prob_edge"]
    print(f"\n  Prob_edge distribution (games with book spreads):")
    for t in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        n = (pe >= t).sum()
        print(f"    >= {t*100:>5.1f}%: {n:>5} games ({n/len(pe)*100:.1f}%)")

    # Run all analyses
    r1 = analysis_1(picks, actual, book_spread, has_book)
    r2 = analysis_2(picks, actual, book_spread, has_book)
    r3 = analysis_3(picks, actual, book_spread, has_book, df_val)
    r4 = analysis_4(picks, actual, book_spread, has_book)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    if r4:
        best = r4[0]
        print(f"  Best strategy (≥100 bets): edge>={best['edge']*100:.0f}%, "
              f"sigma={best['sigma_band']}")
        print(f"    Bets: {best['bets']}, Win%: {best['win_rate']*100:.1f}%, "
              f"ROI: {best['roi']*100:+.1f}%, Units: {best['units']:+.1f}")

    print()


if __name__ == "__main__":
    main()

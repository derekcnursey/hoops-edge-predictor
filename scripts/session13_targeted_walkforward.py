#!/usr/bin/env python3
"""Session 13: Targeted walk-forward test for away-dog and big-spread strategies.

Reuses the trained walk-forward models (same training procedure) to test:
  S1: prob_edge >= 10% AND pick_away AND book_spread > 10
  S2: prob_edge >= 7% AND |book_spread| > 15

Also tests several related variants for context.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.session13_validation_suite import (
    load_season_data, train_model, predict, count_dead, normal_cdf,
    WINNER_HP,
)
from src import config

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"


def compute_targeted_roi(mu, sigma, book, actual, threshold,
                         pick_filter=None, abs_book_min=None,
                         abs_book_max=None, away_only=False):
    """Compute ROI with targeted filters.

    Args:
        pick_filter: None or "away" or "home"
        abs_book_min: min |book_spread| to bet
        abs_book_max: max |book_spread| to bet
        away_only: if True, only bet when picking away side
    """
    valid = ~np.isnan(book)
    if valid.sum() == 0:
        return {"bets": 0, "wins": 0, "win_rate": 0.0, "roi": 0.0, "units": 0.0}

    edge_home = mu[valid] + book[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    hcp = normal_cdf(edge_z)
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, hcp, 1.0 - hcp)
    prob_edge = pick_prob - 0.5238

    bet_mask = prob_edge >= threshold

    # Direction filter
    if pick_filter == "away":
        bet_mask = bet_mask & ~pick_home  # picking away
    elif pick_filter == "home":
        bet_mask = bet_mask & pick_home

    # Book spread magnitude filter
    abs_book_v = np.abs(book[valid])
    if abs_book_min is not None:
        bet_mask = bet_mask & (abs_book_v > abs_book_min)
    if abs_book_max is not None:
        bet_mask = bet_mask & (abs_book_v <= abs_book_max)

    n = bet_mask.sum()
    if n == 0:
        return {"bets": 0, "wins": 0, "win_rate": 0.0, "roi": 0.0, "units": 0.0}

    actual_v = actual[valid]
    book_v = book[valid]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)
    w = pick_won[bet_mask].sum()
    profit_per_1 = 100.0 / 110.0
    units = w * profit_per_1 - (n - w)
    return {"bets": int(n), "wins": int(w), "win_rate": float(w / n),
            "roi": float(units / n), "units": float(units)}


STRATEGIES = [
    # name, threshold, pick_filter, abs_book_min, abs_book_max
    ("S1: edge>=10% away |bk|>10",  0.10, "away", 10, None),
    ("S2: edge>=7% |bk|>15",        0.07, None,   15, None),
    # Context variants
    ("S3: edge>=10% |bk|>10",       0.10, None,   10, None),
    ("S4: edge>=12% away |bk|>10",  0.12, "away", 10, None),
    ("S5: edge>=10% away |bk|>15",  0.10, "away", 15, None),
    ("S6: edge>=12% |bk|>15",       0.12, None,   15, None),
    ("S7: edge>=10% away dog",       0.10, "away", None, None),
    ("S8: edge>=12% unfiltered",     0.12, None,   None, None),
]


def main():
    print("=" * 70)
    print("  TARGETED WALK-FORWARD: AWAY DOG + BIG SPREAD STRATEGIES")
    print("=" * 70)

    test_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    # Collect per-year results
    year_results = {s[0]: [] for s in STRATEGIES}
    pooled = {s[0]: {"mu": [], "sigma": [], "book": [], "actual": []} for s in STRATEGIES}

    # We need pooled predictions — collect all
    all_mu, all_sigma, all_book, all_actual = [], [], [], []

    for ty in test_years:
        train_seasons = list(range(2015, ty))
        print(f"\n  --- {ty}: train on {train_seasons[0]}-{train_seasons[-1]} ---")

        t0 = time.time()
        X_tr, y_tr, X_v, y_v, scaler, df_v = load_season_data(train_seasons, [ty])
        n_book = df_v["bookSpread"].notna().sum() if "bookSpread" in df_v.columns else 0

        model, best_ep = train_model(X_tr, y_tr, X_v, y_v, verbose=False)
        mu, sigma = predict(model, X_v)
        actual = y_v
        book = df_v["bookSpread"].values.astype(np.float64) if "bookSpread" in df_v.columns else np.full(len(df_v), np.nan)

        elapsed = time.time() - t0
        print(f"    {len(df_v)} games, {n_book} with book, ep@{best_ep} [{elapsed:.0f}s]")

        all_mu.append(mu)
        all_sigma.append(sigma)
        all_book.append(book)
        all_actual.append(actual)

        # Run each strategy
        for sname, thresh, pfilt, abmin, abmax in STRATEGIES:
            r = compute_targeted_roi(mu, sigma, book, actual, thresh,
                                     pick_filter=pfilt, abs_book_min=abmin,
                                     abs_book_max=abmax)
            year_results[sname].append({"year": ty, **r})

        del model
        torch.cuda.empty_cache()

    # Pooled
    p_mu = np.concatenate(all_mu)
    p_sigma = np.concatenate(all_sigma)
    p_book = np.concatenate(all_book)
    p_actual = np.concatenate(all_actual)

    pooled_results = {}
    for sname, thresh, pfilt, abmin, abmax in STRATEGIES:
        r = compute_targeted_roi(p_mu, p_sigma, p_book, p_actual, thresh,
                                 pick_filter=pfilt, abs_book_min=abmin,
                                 abs_book_max=abmax)
        pooled_results[sname] = r

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    for sname, _, _, _, _ in STRATEGIES:
        yrs = year_results[sname]
        pool = pooled_results[sname]

        n_positive = sum(1 for y in yrs if y["bets"] > 0 and y["roi"] > 0)
        n_with_bets = sum(1 for y in yrs if y["bets"] > 0)

        print(f"\n  ── {sname} ──")
        print(f"  {'Year':>6} {'Bets':>6} {'W':>4} {'Win%':>7} {'ROI':>8} {'Units':>8}")
        print(f"  {'-'*6} {'-'*6} {'-'*4} {'-'*7} {'-'*8} {'-'*8}")

        for y in yrs:
            if y["bets"] > 0:
                print(f"  {y['year']:>6} {y['bets']:>6} {y['wins']:>4} "
                      f"{y['win_rate']*100:>6.1f}% {y['roi']*100:>+7.1f}% "
                      f"{y['units']:>+8.1f}")
            else:
                print(f"  {y['year']:>6}      0    -     ---      ---      ---")

        if pool["bets"] > 0:
            print(f"  {'POOLED':>6} {pool['bets']:>6} {pool['wins']:>4} "
                  f"{pool['win_rate']*100:>6.1f}% {pool['roi']*100:>+7.1f}% "
                  f"{pool['units']:>+8.1f}")
        print(f"  Positive years: {n_positive}/{n_with_bets}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("  SUMMARY: WHICH STRATEGY QUALIFIES?")
    print("  Criteria: positive in 5/7 years AND pooled ROI > 3%")
    print("=" * 70)

    print(f"\n  {'Strategy':>35} {'Pooled ROI':>11} {'Pooled Bets':>12} {'Pos Yrs':>8} {'PASS?':>6}")
    print(f"  {'-'*35} {'-'*11} {'-'*12} {'-'*8} {'-'*6}")

    qualifying = []
    for sname, _, _, _, _ in STRATEGIES:
        yrs = year_results[sname]
        pool = pooled_results[sname]
        n_pos = sum(1 for y in yrs if y["bets"] > 0 and y["roi"] > 0)
        n_with = sum(1 for y in yrs if y["bets"] > 0)
        passes = n_pos >= 5 and pool["roi"] > 0.03
        label = "YES" if passes else "no"
        if passes:
            qualifying.append((sname, pool, n_pos, n_with))
        print(f"  {sname:>35} {pool['roi']*100:>+10.1f}% {pool['bets']:>12} "
              f"{n_pos}/{n_with:>5} {label:>6}")

    # Save qualifying strategies
    if qualifying:
        print(f"\n  QUALIFYING STRATEGIES: {len(qualifying)}")
        best = max(qualifying, key=lambda x: x[1]["roi"])
        print(f"  Best: {best[0]} → ROI={best[1]['roi']*100:+.1f}%, "
              f"{best[1]['bets']} bets, {best[2]}/{best[3]} positive years")

        strategy_json = {
            "name": best[0],
            "pooled_roi": best[1]["roi"],
            "pooled_bets": best[1]["bets"],
            "pooled_win_rate": best[1]["win_rate"],
            "pooled_units": best[1]["units"],
            "positive_years": best[2],
            "total_years": best[3],
            "all_qualifying": [
                {"name": s[0], "roi": s[1]["roi"], "bets": s[1]["bets"],
                 "pos_years": s[2]}
                for s in qualifying
            ],
        }

        out_path = config.PROJECT_ROOT / "artifacts" / "betting_strategy.json"
        with open(out_path, "w") as f:
            json.dump(strategy_json, f, indent=2)
        print(f"  Saved to {out_path}")
    else:
        print(f"\n  NO STRATEGIES QUALIFY.")
        print(f"  None met both criteria (5/7 positive years AND pooled ROI > 3%).")

        # Save the negative result too
        strategy_json = {
            "result": "no_qualifying_strategy",
            "note": "No strategy met criteria: 5/7 positive years AND pooled ROI > 3%",
            "best_unfiltered": {
                "name": "edge>=12% unfiltered",
                "pooled_roi": pooled_results["S8: edge>=12% unfiltered"]["roi"],
                "pooled_bets": pooled_results["S8: edge>=12% unfiltered"]["bets"],
            },
        }
        out_path = config.PROJECT_ROOT / "artifacts" / "betting_strategy.json"
        with open(out_path, "w") as f:
            json.dump(strategy_json, f, indent=2)
        print(f"  Saved negative result to {out_path}")

    return year_results, pooled_results


if __name__ == "__main__":
    main()

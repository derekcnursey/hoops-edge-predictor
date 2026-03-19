#!/usr/bin/env python3
"""Validate WLS solver against iterative solver and Torvik/KenPom.

Compares the two solver backends on a single season of data:
1. Runs both iterative and WLS solvers on the same input games
2. Compares adj_oe, adj_de, adj_margin for top/bottom teams
3. Checks HCA estimate from WLS
4. Runs parameter sweep over (half_life, margin_cap, wls_alpha)

Usage:
    poetry run python scripts/wls_solver_validation.py --season 2025
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add ETL repo to path for imports
ETL_REPO = Path(__file__).resolve().parent.parent.parent / "hoops_edge_database_etl"
sys.path.insert(0, str(ETL_REPO / "src"))
sys.path.insert(0, str(ETL_REPO))

from cbbd_etl.config import load_config
from cbbd_etl.gold.adjusted_efficiencies import (
    _get_rating_params,
    _load_d1_team_ids,
    _load_pbp_no_garbage_games,
    _load_team_info,
    _run_per_date_ratings,
    _apply_margin_cap,
)
from cbbd_etl.gold.iterative_ratings import exponential_decay_weight, GameObs
from cbbd_etl.gold.least_squares_ratings import solve_ratings_wls
from cbbd_etl.s3_io import S3IO


def load_games(cfg: Config, season: int):
    """Load game data from S3."""
    s3 = S3IO(cfg.bucket, cfg.region)
    d1_ids = _load_d1_team_ids(s3, cfg)
    team_info = _load_team_info(s3, cfg)
    games_by_date = _load_pbp_no_garbage_games(s3, cfg, season, d1_ids)
    return games_by_date, team_info, d1_ids


def run_solver(games_by_date, team_info, season, solver_type, params, half_life=None,
               margin_cap=None, wls_alpha=0.01):
    """Run a solver and return the final date's records as a DataFrame."""
    records = _run_per_date_ratings(
        games_by_date, team_info, season,
        half_life=half_life,
        hca_oe=params["hca_oe"],
        hca_de=params["hca_de"],
        barthag_exp=params["barthag_exp"],
        sos_exponent=params.get("sos_exponent", 1.0),
        shrinkage=params.get("shrinkage", 0.0),
        solver_type=solver_type,
        wls_alpha=wls_alpha,
        wls_estimate_hca=True,
        margin_cap=margin_cap,
    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Get final date's ratings
    final_date = df["rating_date"].max()
    return df[df["rating_date"] == final_date].copy()


def compare_solvers(df_iter, df_wls, top_n=15):
    """Compare top and bottom teams between solvers."""
    merged = df_iter.merge(df_wls, on="teamId", suffixes=("_iter", "_wls"))

    # Sort by iterative adj_margin
    merged = merged.sort_values("adj_margin_iter", ascending=False)

    print(f"\n{'='*100}")
    print(f"  TOP {top_n} TEAMS BY ITERATIVE ADJ_MARGIN")
    print(f"{'='*100}")
    print(f"  {'Team':<25} {'Iter OE':>8} {'WLS OE':>8} {'Iter DE':>8} {'WLS DE':>8} "
          f"{'Iter Margin':>12} {'WLS Margin':>12} {'Δ Margin':>10}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*12} {'─'*12} {'─'*10}")

    for _, row in merged.head(top_n).iterrows():
        delta = row["adj_margin_wls"] - row["adj_margin_iter"]
        print(f"  {str(row.get('team_iter', '')):<25} "
              f"{row['adj_oe_iter']:>8.2f} {row['adj_oe_wls']:>8.2f} "
              f"{row['adj_de_iter']:>8.2f} {row['adj_de_wls']:>8.2f} "
              f"{row['adj_margin_iter']:>12.2f} {row['adj_margin_wls']:>12.2f} "
              f"{delta:>+10.2f}")

    print(f"\n  BOTTOM {top_n} TEAMS")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*12} {'─'*12} {'─'*10}")
    for _, row in merged.tail(top_n).iterrows():
        delta = row["adj_margin_wls"] - row["adj_margin_iter"]
        print(f"  {str(row.get('team_iter', '')):<25} "
              f"{row['adj_oe_iter']:>8.2f} {row['adj_oe_wls']:>8.2f} "
              f"{row['adj_de_iter']:>8.2f} {row['adj_de_wls']:>8.2f} "
              f"{row['adj_margin_iter']:>12.2f} {row['adj_margin_wls']:>12.2f} "
              f"{delta:>+10.2f}")

    # Correlation stats
    corr_oe = merged["adj_oe_iter"].corr(merged["adj_oe_wls"])
    corr_de = merged["adj_de_iter"].corr(merged["adj_de_wls"])
    corr_margin = merged["adj_margin_iter"].corr(merged["adj_margin_wls"])
    mae_margin = (merged["adj_margin_iter"] - merged["adj_margin_wls"]).abs().mean()

    print(f"\n  Correlation (Iter vs WLS):")
    print(f"    adj_oe:     r = {corr_oe:.4f}")
    print(f"    adj_de:     r = {corr_de:.4f}")
    print(f"    adj_margin: r = {corr_margin:.4f}")
    print(f"    MAE(margin): {mae_margin:.2f}")

    # WLS HCA estimate
    if "estimated_hca" in df_wls.columns:
        hca = df_wls["estimated_hca"].iloc[0]
        print(f"\n  WLS estimated HCA: {hca:.2f} pts/100poss")

    return merged


def parameter_sweep(games_by_date, team_info, season, params):
    """Sweep over (half_life, margin_cap, wls_alpha) for WLS solver."""
    half_lives = [None, 30.0, 45.0, 60.0]
    margin_caps = [None, 25.0, 35.0, 50.0]
    alphas = [0.001, 0.01, 0.05, 0.1]

    print(f"\n{'='*100}")
    print(f"  WLS PARAMETER SWEEP ({len(half_lives)} × {len(margin_caps)} × {len(alphas)} = "
          f"{len(half_lives) * len(margin_caps) * len(alphas)} configs)")
    print(f"{'='*100}")
    print(f"  {'HL':>6} {'MCap':>6} {'α':>8} {'Avg OE':>8} {'Std OE':>8} "
          f"{'Avg DE':>8} {'Std DE':>8} {'HCA':>6} {'Time':>6}")
    print(f"  {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

    results = []
    for hl in half_lives:
        for mc in margin_caps:
            for alpha in alphas:
                t0 = time.time()
                df = run_solver(
                    games_by_date, team_info, season,
                    solver_type="wls", params=params,
                    half_life=hl, margin_cap=mc, wls_alpha=alpha,
                )
                elapsed = time.time() - t0

                if df.empty:
                    continue

                avg_oe = df["adj_oe"].mean()
                std_oe = df["adj_oe"].std()
                avg_de = df["adj_de"].mean()
                std_de = df["adj_de"].std()
                hca = df["estimated_hca"].iloc[0] if "estimated_hca" in df.columns else float("nan")

                hl_str = f"{hl:.0f}" if hl is not None else "None"
                mc_str = f"{mc:.0f}" if mc is not None else "None"

                print(f"  {hl_str:>6} {mc_str:>6} {alpha:>8.4f} "
                      f"{avg_oe:>8.2f} {std_oe:>8.2f} "
                      f"{avg_de:>8.2f} {std_de:>8.2f} "
                      f"{hca:>6.2f} {elapsed:>5.1f}s")

                results.append({
                    "half_life": hl, "margin_cap": mc, "alpha": alpha,
                    "avg_oe": avg_oe, "std_oe": std_oe,
                    "avg_de": avg_de, "std_de": std_de,
                    "hca": hca, "time": elapsed,
                })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(str(ETL_REPO / "config.yaml"))
    params = _get_rating_params(cfg)

    print(f"Loading games for season {args.season}...")
    t0 = time.time()
    games_by_date, team_info, d1_ids = load_games(cfg, args.season)
    n_games = sum(len(v) for v in games_by_date.values())
    n_dates = len(games_by_date)
    print(f"  Loaded {n_games} game-team observations across {n_dates} dates ({time.time()-t0:.1f}s)")

    # Run iterative solver (current production)
    print(f"\nRunning ITERATIVE solver (sos={params['sos_exponent']})...")
    t0 = time.time()
    df_iter = run_solver(
        games_by_date, team_info, args.season,
        solver_type="iterative", params=params,
        half_life=params.get("half_life"),
    )
    print(f"  Done ({time.time()-t0:.1f}s), {len(df_iter)} teams")

    # Run WLS solver
    print(f"\nRunning WLS solver (alpha=0.01)...")
    t0 = time.time()
    df_wls = run_solver(
        games_by_date, team_info, args.season,
        solver_type="wls", params=params,
        half_life=params.get("half_life"),
        wls_alpha=0.01,
    )
    print(f"  Done ({time.time()-t0:.1f}s), {len(df_wls)} teams")

    # Compare
    compare_solvers(df_iter, df_wls, top_n=args.top_n)

    # Parameter sweep
    if args.sweep:
        sweep_df = parameter_sweep(games_by_date, team_info, args.season, params)
        out_path = Path(__file__).resolve().parent.parent / "analysis" / "wls_sweep_results.csv"
        sweep_df.to_csv(out_path, index=False)
        print(f"\n  Sweep results saved to {out_path}")


if __name__ == "__main__":
    main()

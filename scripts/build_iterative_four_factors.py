#!/usr/bin/env python3
"""Build iterative opponent-adjusted four-factor stats for a season.

Usage:
    poetry run python scripts/build_iterative_four_factors.py --season 2025 --verbose
    poetry run python scripts/build_iterative_four_factors.py --seasons 2015-2026
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.features import load_boxscores
from src.four_factors import compute_game_four_factors
from src.iterative_four_factors import solve_four_factors


def build_season(season: int, n_iterations: int, prior_weight: float,
                 verbose: bool) -> Path:
    """Build iterative FF for a single season, save as parquet."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Season {season}")
    print(f"{'='*60}")

    print("  Loading boxscores from S3...")
    box = load_boxscores(season)
    if box.empty:
        print("  No boxscore data found. Skipping.")
        return None

    print(f"  Boxscore rows: {len(box)}")

    print("  Computing raw four-factor stats...")
    ff = compute_game_four_factors(box)
    n_games = len(ff)
    n_teams = ff["teamid"].nunique()
    print(f"  Games (team-rows): {n_games}, teams: {n_teams}")

    print(f"  Running iterative solver (n_iter={n_iterations}, prior={prior_weight})...")
    adj_ff = solve_four_factors(
        ff,
        n_iterations=n_iterations,
        prior_weight=prior_weight,
        verbose=verbose,
    )

    # Save
    out_path = config.FEATURES_DIR / f"iterative_ff_{season}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adj_ff.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    print(f"  Saved: {out_path}")
    print(f"  Time: {elapsed:.1f}s")

    # Sanity check: compare raw vs adjusted for a few stats
    if verbose:
        from src.four_factors import FOUR_FACTOR_COLS
        print("\n  Stat adjustment summary (mean shift):")
        for s in FOUR_FACTOR_COLS:
            raw_mean = ff[s].mean()
            adj_mean = adj_ff[s].mean()
            diff = adj_mean - raw_mean
            print(f"    {s:>25}: raw={raw_mean:.4f}  adj={adj_mean:.4f}  Δ={diff:+.4f}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build iterative four-factor adjustments")
    parser.add_argument("--season", type=int, default=None, help="Single season")
    parser.add_argument("--seasons", type=str, default=None, help="Season range (e.g. 2015-2026)")
    parser.add_argument("--iterations", type=int, default=25, help="Solver iterations (default: 25)")
    parser.add_argument("--prior-weight", type=float, default=5.0, help="Bayesian prior weight")
    parser.add_argument("--verbose", action="store_true", help="Print convergence diagnostics")
    args = parser.parse_args()

    if args.season:
        seasons = [args.season]
    elif args.seasons:
        if "-" in args.seasons:
            start, end = args.seasons.split("-")
            seasons = list(range(int(start), int(end) + 1))
        else:
            seasons = [int(s) for s in args.seasons.split(",")]
    else:
        parser.error("Must specify --season or --seasons")

    t_total = time.time()
    results = []
    for season in seasons:
        path = build_season(season, args.iterations, args.prior_weight, args.verbose)
        if path:
            results.append((season, path))

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Built {len(results)} season(s) in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    for season, path in results:
        print(f"    {season}: {path}")


if __name__ == "__main__":
    main()

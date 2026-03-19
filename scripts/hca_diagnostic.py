#!/usr/bin/env python3
"""Empirically validate Home-Court Advantage (HCA) for a given CBB season.

Compares:
1. Empirical HCA — raw average home margin and pts/100poss from game data
2. WLS solver estimated HCA — the solver's fitted HCA parameter

Usage:
    poetry run python scripts/hca_diagnostic.py --season 2025
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
)
from cbbd_etl.gold.iterative_ratings import GameObs
from cbbd_etl.s3_io import S3IO


def load_games(cfg, season: int):
    """Load game data from S3."""
    s3 = S3IO(cfg.bucket, cfg.region)
    d1_ids = _load_d1_team_ids(s3, cfg)
    team_info = _load_team_info(s3, cfg)
    games_by_date = _load_pbp_no_garbage_games(s3, cfg, season, d1_ids)
    return games_by_date, team_info, d1_ids


def compute_empirical_hca(games_by_date: dict) -> dict:
    """Compute empirical HCA from raw game observations.

    Each GameObs is a team-game row. We filter to is_home=True, is_neutral=False
    to get one row per home game from the home team's perspective.
    """
    home_margins = []
    home_poss = []
    all_poss = []

    for date_str, obs_list in games_by_date.items():
        for obs in obs_list:
            # Collect all possessions for league average
            avg_poss = (obs.team_poss + obs.opp_poss) / 2.0
            all_poss.append(avg_poss)

            if obs.is_home and not obs.is_neutral:
                margin = obs.team_pts - obs.opp_pts
                home_margins.append(margin)
                home_poss.append(avg_poss)

    home_margins = np.array(home_margins)
    home_poss_arr = np.array(home_poss)
    all_poss_arr = np.array(all_poss)

    avg_margin = np.mean(home_margins)
    avg_home_poss = np.mean(home_poss_arr)
    avg_all_poss = np.mean(all_poss_arr)
    median_margin = np.median(home_margins)
    std_margin = np.std(home_margins, ddof=1)
    win_pct = np.mean(home_margins > 0)

    # HCA in pts/100poss: avg_margin / (avg_poss / 100)
    hca_per_100 = avg_margin / (avg_home_poss / 100.0)

    return {
        "n_home_games": len(home_margins),
        "n_total_obs": len(all_poss),
        "avg_home_margin": avg_margin,
        "median_home_margin": median_margin,
        "std_home_margin": std_margin,
        "home_win_pct": win_pct,
        "avg_home_poss": avg_home_poss,
        "avg_all_poss": avg_all_poss,
        "empirical_hca_per_100": hca_per_100,
    }


def get_wls_hca(games_by_date, team_info, season, params) -> dict:
    """Run WLS solver for final date to get estimated_hca."""
    t0 = time.time()
    records = _run_per_date_ratings(
        games_by_date, team_info, season,
        half_life=params.get("half_life"),
        hca_oe=params["hca_oe"],
        hca_de=params["hca_de"],
        barthag_exp=params["barthag_exp"],
        sos_exponent=params.get("sos_exponent", 1.0),
        shrinkage=params.get("shrinkage", 0.0),
        solver_type="wls",
        wls_alpha=0.01,
        wls_estimate_hca=True,
        margin_cap=None,
    )
    elapsed = time.time() - t0

    if not records:
        return {"elapsed": elapsed}

    df = pd.DataFrame(records)
    final_date = df["rating_date"].max()
    df_final = df[df["rating_date"] == final_date]

    estimated_hca = df_final["estimated_hca"].iloc[0] if "estimated_hca" in df_final.columns else None

    return {
        "elapsed": elapsed,
        "estimated_hca": estimated_hca,
        "final_date": final_date,
        "n_teams": len(df_final),
    }


def main():
    parser = argparse.ArgumentParser(description="HCA diagnostic for CBB season")
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    cfg = load_config(str(ETL_REPO / "config.yaml"))
    params = _get_rating_params(cfg)

    # --- Load data ---
    print(f"Loading games for season {args.season}...")
    t0 = time.time()
    games_by_date, team_info, d1_ids = load_games(cfg, args.season)
    n_obs = sum(len(v) for v in games_by_date.values())
    n_dates = len(games_by_date)
    print(f"  Loaded {n_obs} team-game observations across {n_dates} dates ({time.time()-t0:.1f}s)")

    # --- Empirical HCA ---
    print(f"\nComputing empirical HCA...")
    emp = compute_empirical_hca(games_by_date)

    # --- WLS solver HCA ---
    print(f"Running WLS solver to get estimated_hca...")
    wls = get_wls_hca(games_by_date, team_info, args.season, params)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  HCA DIAGNOSTIC — {args.season} SEASON")
    print(f"{'='*70}")

    print(f"\n  DATA SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Total team-game observations:   {emp['n_total_obs']:>8,}")
    print(f"  Non-neutral home games:         {emp['n_home_games']:>8,}")
    print(f"  Game dates:                     {n_dates:>8,}")

    print(f"\n  EMPIRICAL HOME-COURT ADVANTAGE")
    print(f"  {'─'*50}")
    print(f"  Avg home margin (raw pts):      {emp['avg_home_margin']:>+8.2f}")
    print(f"  Median home margin:             {emp['median_home_margin']:>+8.2f}")
    print(f"  Std dev of home margin:         {emp['std_home_margin']:>8.2f}")
    print(f"  Home win %:                     {emp['home_win_pct']:>8.1%}")
    print(f"  Avg possessions (home games):   {emp['avg_home_poss']:>8.1f}")
    print(f"  Avg possessions (all games):    {emp['avg_all_poss']:>8.1f}")
    print(f"  Empirical HCA (pts/100poss):    {emp['empirical_hca_per_100']:>+8.2f}")

    print(f"\n  WLS SOLVER HCA")
    print(f"  {'─'*50}")
    if wls.get("estimated_hca") is not None:
        hca_wls = wls["estimated_hca"]
        # The WLS estimated_hca is per-side pts/100poss.
        # Total margin HCA = 2 * estimated_hca (one side OE boost + one side DE boost)
        hca_margin_per_100 = 2.0 * hca_wls
        hca_raw_pts = hca_margin_per_100 * (emp["avg_home_poss"] / 100.0)

        print(f"  Solver final date:              {wls['final_date']}")
        print(f"  Teams rated:                    {wls['n_teams']:>8,}")
        print(f"  estimated_hca (per-side /100):  {hca_wls:>+8.2f}")
        print(f"  Total HCA (pts/100poss):        {hca_margin_per_100:>+8.2f}")
        print(f"  Total HCA (raw pts):            {hca_raw_pts:>+8.2f}")
        print(f"  Solver time:                    {wls['elapsed']:>7.1f}s")

        print(f"\n  COMPARISON")
        print(f"  {'─'*50}")
        print(f"  Empirical HCA (pts/100poss):    {emp['empirical_hca_per_100']:>+8.2f}")
        print(f"  WLS solver HCA (pts/100poss):   {hca_margin_per_100:>+8.2f}")
        delta = emp["empirical_hca_per_100"] - hca_margin_per_100
        print(f"  Difference (emp - solver):      {delta:>+8.2f}")
        print(f"")
        print(f"  Empirical HCA (raw pts):        {emp['avg_home_margin']:>+8.2f}")
        print(f"  WLS solver HCA (raw pts):       {hca_raw_pts:>+8.2f}")
        delta_raw = emp["avg_home_margin"] - hca_raw_pts
        print(f"  Difference (emp - solver):      {delta_raw:>+8.2f}")

        print(f"\n  CONFIG PRIORS")
        print(f"  {'─'*50}")
        print(f"  hca_oe (config prior):          {params['hca_oe']:>+8.2f}")
        print(f"  hca_de (config prior):          {params['hca_de']:>+8.2f}")
        print(f"  Total prior (pts/100poss):      {params['hca_oe'] + params['hca_de']:>+8.2f}")
        prior_raw = (params["hca_oe"] + params["hca_de"]) * (emp["avg_home_poss"] / 100.0)
        print(f"  Total prior (raw pts):          {prior_raw:>+8.2f}")
    else:
        print(f"  WARNING: estimated_hca not found in solver output")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()

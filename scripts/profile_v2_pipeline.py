"""Profile build_features_v2 to quantify time spent in each stage.

Usage:
    poetry run python scripts/profile_v2_pipeline.py --season 2025 --date 2025-01-15
"""
import argparse
import time
import sys
sys.path.insert(0, ".")

import pandas as pd
from src import config, s3_reader
from src.four_factors import compute_game_four_factors
from src.adjusted_four_factors import adjust_four_factors
from src.rolling_averages import compute_rolling_averages_v2, compute_form_delta, compute_rolling_turnovers
from src.pbp_advanced_stats import compute_advanced_stats
from src.luck_regression import compute_luck_features, LUCK_FEATURE_COLS
from src.features import (
    compute_rolling_advanced_stats_v2,
    _ROLLING_ADV_STATS,
    _load_optimal_spans,
    load_games,
    load_efficiency_ratings,
    load_boxscores,
    compute_schedule_features,
    compute_pace_features,
    compute_kill_shot_metrics,
    KILL_SHOT_COLS,
    _build_conf_strength_lookup,
)
from src.opponent_adjustment import opponent_adjust
from src.features import _ADV_NO_ADJUST, _ALL_ADV_STAT_PAIRS


def profile(season: int, game_date: str | None = None):
    timings = {}

    def tick(label):
        timings[label] = time.time()

    tick("start")

    # S3 reads
    games = load_games(season)
    tick("load_games")

    eff_ratings = load_efficiency_ratings(season)
    tick("load_eff_ratings")

    boxscores = load_boxscores(season)
    tick("load_boxscores")

    pbp_keys = s3_reader.list_parquet_keys(
        f"{config.SILVER_PREFIX}/fct_pbp_plays_enriched/season={season}/"
    )
    pbp_df = s3_reader.read_parquet_table(pbp_keys).to_pandas() if pbp_keys else pd.DataFrame()
    tick("load_pbp")

    print(f"Games: {len(games)}, Boxscores: {len(boxscores)}, PBP: {len(pbp_df)}")

    # Four factors + opponent adjustment + rolling
    ff = compute_game_four_factors(boxscores) if not boxscores.empty else pd.DataFrame()
    tick("four_factors")

    if not ff.empty:
        ff = adjust_four_factors(ff, prior_weight=config.ADJUST_PRIOR, alpha=config.ADJUST_ALPHA)
    tick("adjust_four_factors")

    ff_span = _load_optimal_spans().get("four_factors")
    rolling_df = compute_rolling_averages_v2(ff, optimal_span=ff_span) if not ff.empty else pd.DataFrame()
    tick("rolling_four_factors")

    # PBP advanced stats
    adv_stats = compute_advanced_stats(pbp_df) if not pbp_df.empty else pd.DataFrame()
    tick("compute_advanced_stats")

    # Luck features
    luck_df = compute_luck_features(adv_stats) if not adv_stats.empty else pd.DataFrame()
    tick("luck_features")

    luck_rolling = pd.DataFrame()
    if not luck_df.empty:
        luck_rolling = compute_rolling_advanced_stats_v2(luck_df, LUCK_FEATURE_COLS)
    tick("luck_rolling")

    # Opponent adjustment for advanced stats
    if not adv_stats.empty:
        adjustable = [s for s in _ROLLING_ADV_STATS if s not in _ADV_NO_ADJUST and s in adv_stats.columns]
        pairs_for_adj = {s: _ALL_ADV_STAT_PAIRS[s] for s in adjustable if s in _ALL_ADV_STAT_PAIRS}
        if adjustable and pairs_for_adj:
            adv_stats = opponent_adjust(adv_stats, stat_cols=adjustable, stat_pairs=pairs_for_adj, no_adjust=_ADV_NO_ADJUST)
    tick("opponent_adjust_adv")

    # Rolling advanced stats
    available_stats = [s for s in _ROLLING_ADV_STATS if s in adv_stats.columns] if not adv_stats.empty else []
    adv_rolling = compute_rolling_advanced_stats_v2(adv_stats, available_stats) if available_stats else pd.DataFrame()
    tick("rolling_adv_stats")

    # Kill shot
    ks_df = compute_kill_shot_metrics(pbp_df) if not pbp_df.empty else pd.DataFrame()
    tick("kill_shot_compute")

    from src.features import compute_rolling_advanced_stats
    ks_rolling = compute_rolling_advanced_stats(ks_df, KILL_SHOT_COLS) if not ks_df.empty else pd.DataFrame()
    tick("kill_shot_rolling")

    # Pace
    pace_df = compute_pace_features(boxscores) if not boxscores.empty else pd.DataFrame()
    tick("pace_features")

    # Schedule
    schedule_df = compute_schedule_features(games)
    tick("schedule_features")

    # Form delta
    form_df = compute_form_delta(ff) if not ff.empty else pd.DataFrame()
    tick("form_delta")

    # Turnovers
    tov_df = compute_rolling_turnovers(boxscores) if not boxscores.empty else pd.DataFrame()
    tick("rolling_turnovers")

    # Conference strength (the suspected bottleneck)
    unique_dates = list(pd.to_datetime(games["startDate"], errors="coerce").dropna().unique())
    cl = _build_conf_strength_lookup(eff_ratings, unique_dates)
    tick("conf_strength_once")

    # Simulate the per-game loop calling conf_strength N times
    n_games = len(games)
    t0 = time.time()
    for i in range(min(n_games, 50)):  # sample 50 to estimate
        _ = _build_conf_strength_lookup(eff_ratings, unique_dates)
    t_50 = time.time() - t0
    estimated_full = t_50 / min(n_games, 50) * n_games
    tick("conf_strength_loop_sample")

    # Lookup building (iterrows)
    t0 = time.time()
    if not rolling_df.empty:
        rolling_df["_date"] = pd.to_datetime(rolling_df["startdate"], errors="coerce")
        rl = {}
        for _, row in rolling_df.iterrows():
            key = (int(row["gameid"]), int(row["teamid"]))
            rl[key] = row.to_dict()
    tick("build_rolling_lookups")

    t0 = time.time()
    if not adv_rolling.empty:
        adv_rolling["_date"] = pd.to_datetime(adv_rolling["startdate"], errors="coerce")
        al = {}
        for _, row in adv_rolling.iterrows():
            key = (int(row["gameid"]), int(row["teamid"]))
            al[key] = row.to_dict()
    tick("build_adv_lookups")

    # Print timing report
    labels = list(timings.keys())
    print(f"\n{'='*60}")
    print(f"PROFILING REPORT: season={season}")
    print(f"{'='*60}")
    total = timings[labels[-1]] - timings["start"]
    for i in range(1, len(labels)):
        dt = timings[labels[i]] - timings[labels[i-1]]
        pct = dt / total * 100
        print(f"  {labels[i]:35s}  {dt:7.1f}s  ({pct:4.1f}%)")
    print(f"{'='*60}")
    print(f"  {'TOTAL':35s}  {total:7.1f}s")
    print(f"\n  conf_strength estimated per-game loop: {estimated_full:.1f}s for {n_games} games")
    print(f"  rolling_df rows: {len(rolling_df)}")
    print(f"  adv_rolling rows: {len(adv_rolling)}")
    if not adv_rolling.empty:
        print(f"  adv_rolling cols: {len(adv_rolling.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    profile(args.season, args.date)

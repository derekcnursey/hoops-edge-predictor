#!/usr/bin/env python3
"""Backfill site prediction JSONs with the current live inference stack.

This is a research/site-only rebuild path:
  - uses the current live `predict()` path (HGBR mu if available)
  - defaults historical line attachment to the repaired research table
  - writes the normal prediction artifacts via `save_predictions()`
  - can clear only `site/public/data/predictions_*.json` before rebuilding

Production CLI behavior is unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src import config
from src.efficiency_blend import blend_enabled
from src.features import build_features, load_research_lines
from src.infer import predict, save_predictions
from src.tournament_adjustments import needs_secondary_mu_features


def _default_current_season() -> int:
    today = datetime.now(ZoneInfo("America/New_York")).date()
    return today.year if today.month <= 6 else today.year + 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild site prediction JSONs using the current live model path."
    )
    parser.add_argument(
        "--season-start",
        type=int,
        default=2019,
        help="First season to rebuild (default: 2019).",
    )
    parser.add_argument(
        "--season-end",
        type=int,
        default=_default_current_season(),
        help="Last season to rebuild (default: current season).",
    )
    parser.add_argument(
        "--lines-table",
        default=config.RESEARCH_LINES_TABLE,
        help=(
            "Lines table to attach during backfill "
            f"(default: {config.RESEARCH_LINES_TABLE})."
        ),
    )
    parser.add_argument(
        "--clear-site-predictions",
        action="store_true",
        help="Delete site/public/data/predictions_*.json before rebuilding.",
    )
    return parser.parse_args()


def _clear_site_predictions() -> int:
    deleted = 0
    for path in sorted(config.SITE_DATA_DIR.glob("predictions_*.json")):
        path.unlink()
        deleted += 1
    return deleted


def _season_dates(features_df: pd.DataFrame, season: int) -> list[str]:
    if features_df.empty:
        return []

    dates = (
        pd.to_datetime(features_df["startDate"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
    )
    date_strings = pd.Series(dates.dt.strftime("%Y-%m-%d")).dropna().unique().tolist()
    date_strings.sort()

    if season == _default_current_season():
        today_et = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        date_strings = [d for d in date_strings if d <= today_et]

    return date_strings


def _feature_dates(features_df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(features_df["startDate"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )


def _build_secondary_season_features_if_needed(
    season: int,
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    if config.EFFICIENCY_SOURCE != "gold":
        return None, None
    if not blend_enabled() or features_df.empty:
        return None, None
    if not needs_secondary_mu_features(features_df):
        return None, None
    secondary = build_features(
        season,
        no_garbage=True,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
        efficiency_source="torvik",
    )
    if secondary.empty:
        return None, None
    return secondary, _feature_dates(secondary)


def main() -> int:
    args = _parse_args()

    if args.season_end < args.season_start:
        raise SystemExit("--season-end must be >= --season-start")

    if args.clear_site_predictions:
        deleted = _clear_site_predictions()
        print(f"Deleted {deleted} existing site prediction JSON files.")

    total_dates = 0
    total_games = 0

    for season in range(args.season_start, args.season_end + 1):
        print(f"\n=== Season {season} ===")
        features_df = build_features(
            season,
            extra_features=config.EXTRA_FEATURES,
            adjust_ff=config.ADJUST_FF,
            adjust_alpha=config.ADJUST_ALPHA,
            adjust_prior_weight=config.ADJUST_PRIOR,
            efficiency_source=config.EFFICIENCY_SOURCE,
        )
        if features_df.empty:
            print("No features found, skipping.")
            continue

        lines_df = load_research_lines(season, table_name=args.lines_table)
        dates = _season_dates(features_df, season)
        print(f"Game dates: {len(dates)}")
        feature_dates = _feature_dates(features_df)
        date_mask = feature_dates.isin(dates)
        features_df = features_df[date_mask].reset_index(drop=True)
        feature_dates = feature_dates[date_mask].reset_index(drop=True)
        secondary_features_df, secondary_feature_dates = _build_secondary_season_features_if_needed(
            season, features_df
        )
        if secondary_features_df is not None and secondary_feature_dates is not None:
            secondary_mask = secondary_feature_dates.isin(dates)
            secondary_features_df = secondary_features_df[secondary_mask].reset_index(drop=True)
            secondary_feature_dates = secondary_feature_dates[secondary_mask].reset_index(drop=True)

        season_secondary_df = None
        if secondary_features_df is not None:
            season_secondary_df = secondary_features_df

        preds_df = predict(
            features_df,
            lines_df=lines_df,
            secondary_mu_features_df=season_secondary_df,
        )

        for game_date in dates:
            daily_preds = preds_df[feature_dates.eq(game_date)].copy()
            if daily_preds.empty:
                continue

            save_predictions(daily_preds, game_date=game_date)

            total_dates += 1
            total_games += len(daily_preds)
            print(f"  {game_date}: {len(daily_preds)} games")

    print(f"\nRebuilt {total_dates} dates / {total_games} game predictions.")
    print(f"Site output directory: {config.SITE_DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Re-merge betting lines into existing prediction CSVs using current provider logic.

Reads each predictions/csv/preds_*_edge.csv, drops old line/edge columns,
re-merges lines from S3 with the current provider preference (DK > ESPN BET > Bovada),
recalculates edge metrics, and overwrites the CSV + site JSON.

Usage:
  refresh_lines.py                     # refresh all seasons
  refresh_lines.py --season 2026       # refresh one season only
  refresh_lines.py --dry-run           # show what would change, don't write
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src import config
from src.features import load_lines
from src.infer import (
    american_profit_per_1,
    american_to_breakeven,
    normal_cdf,
    prob_to_american,
)

# Columns that come from lines / edge calculation (will be dropped and rebuilt)
LINE_COLS = [
    "book_spread", "book_total", "home_moneyline", "away_moneyline",
    "model_spread", "spread_diff", "edge_home_points",
    "pick_side", "pick_cover_prob", "pick_spread_odds",
    "pick_prob_edge", "pick_ev_per_1", "pick_fair_odds",
]

# Provider preference: Draft Kings > ESPN BET > Bovada
PROVIDER_RANK = {"Draft Kings": 0, "ESPN BET": 1, "Bovada": 2}

CSV_RE = re.compile(r"^preds_(\d{4})_(\d{1,2})_(\d{1,2})_edge\.csv$")


def get_season(year: int, month: int) -> int:
    """CBB season: Nov-Apr → next year. Nov 2025 → season 2026."""
    return year + 1 if month >= 11 else year


def dedup_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate lines: prefer complete data, then DK > ESPN BET > Bovada."""
    lines_df = lines_df.copy()
    lines_df["spread"] = pd.to_numeric(lines_df["spread"], errors="coerce")
    lines_df["homeMoneyline"] = pd.to_numeric(lines_df["homeMoneyline"], errors="coerce")

    # Spread sign fix: majority vote across providers
    has_spread = lines_df["spread"].notna() & (lines_df["spread"] != 0)
    spread_sign = np.sign(lines_df.loc[has_spread, "spread"])
    majority_sign = (
        spread_sign.groupby(lines_df.loc[has_spread, "gameId"])
        .sum()
        .rename("_majority_sign")
    )

    dedup = (
        lines_df
        .assign(
            _has_spread=lines_df["spread"].notna().astype(int),
            _has_total=lines_df["overUnder"].notna().astype(int),
            _prov_rank=lines_df["provider"].map(PROVIDER_RANK).fillna(99),
        )
        .sort_values(
            ["_has_spread", "_has_total", "_prov_rank"],
            ascending=[False, False, True],
        )
        .drop_duplicates(subset=["gameId"], keep="first")
        .drop(columns=["_has_spread", "_has_total", "_prov_rank"])
        .copy()
    )

    # Apply majority sign flip
    dedup = dedup.merge(majority_sign, on="gameId", how="left")
    _sp = dedup["spread"]
    _maj = dedup["_majority_sign"]
    mask = (
        _sp.notna() & _maj.notna() & (_maj != 0)
        & (abs(_sp) >= 3)
        & (np.sign(_sp) != np.sign(_maj))
    )
    dedup.loc[mask, "spread"] = -_sp[mask]

    # Moneyline cross-check for single-provider games
    _sp2 = dedup["spread"]
    _ml = dedup["homeMoneyline"]
    mask_ml = (
        _sp2.notna() & _ml.notna()
        & (~mask)
        & dedup["_majority_sign"].isna()
        & (((_sp2 > 3) & (_ml < -150)) | ((_sp2 < -3) & (_ml > 150)))
    )
    dedup.loc[mask_ml, "spread"] = -_sp2[mask_ml]
    dedup = dedup.drop(columns=["_majority_sign"])

    dedup = dedup.rename(columns={
        "spread": "book_spread",
        "overUnder": "book_total",
        "homeMoneyline": "home_moneyline",
        "awayMoneyline": "away_moneyline",
    })
    return dedup


def recalc_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate edge metrics from predicted_spread + book_spread."""
    if "book_spread" not in df.columns or df["book_spread"].isna().all():
        return df

    df["model_spread"] = -df["predicted_spread"]
    df["spread_diff"] = df["model_spread"] - df["book_spread"]
    df["edge_home_points"] = df["predicted_spread"] + df["book_spread"]

    sigma_safe = df["spread_sigma"].clip(lower=0.5)
    edge_z = df["edge_home_points"] / sigma_safe
    home_cover = normal_cdf(edge_z)
    away_cover = 1.0 - home_cover

    df["pick_side"] = np.where(df["edge_home_points"] >= 0, "HOME", "AWAY")
    df["pick_cover_prob"] = np.where(
        df["edge_home_points"] >= 0, home_cover, away_cover
    )

    pick_odds = -110
    pick_be = float(american_to_breakeven(np.array([pick_odds]))[0])
    pick_profit = float(american_profit_per_1(np.array([pick_odds]))[0])

    df["pick_spread_odds"] = pick_odds
    df["pick_prob_edge"] = df["pick_cover_prob"] - pick_be
    df["pick_ev_per_1"] = df["pick_cover_prob"] * pick_profit - (1.0 - df["pick_cover_prob"])
    df["pick_fair_odds"] = prob_to_american(df["pick_cover_prob"].values)

    return df


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Refresh betting lines in prediction CSVs")
    parser.add_argument("--season", type=int, help="Only refresh this season")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    csv_dir = config.PREDICTIONS_DIR / "csv"
    if not csv_dir.exists():
        print("No predictions/csv/ directory found.", file=sys.stderr)
        return 1

    # Group CSV files by season
    files_by_season: dict[int, list] = {}
    for f in sorted(csv_dir.iterdir()):
        m = CSV_RE.match(f.name)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        season = get_season(year, month)
        if args.season and season != args.season:
            continue
        files_by_season.setdefault(season, []).append(f)

    if not files_by_season:
        print("No matching CSV files found.")
        return 0

    total_files = sum(len(v) for v in files_by_season.values())
    print(f"Refreshing lines for {total_files} files across {len(files_by_season)} seasons...")

    changed = 0
    for season in sorted(files_by_season):
        print(f"\n  Season {season}: loading lines from S3...")
        raw_lines = load_lines(season)
        if raw_lines.empty:
            print(f"  Season {season}: no lines in S3, skipping")
            continue

        lines_dedup = dedup_lines(raw_lines)
        merge_cols = ["gameId", "book_spread", "book_total", "home_moneyline", "away_moneyline"]
        available = [c for c in merge_cols if c in lines_dedup.columns]

        for csv_path in files_by_season[season]:
            df = pd.read_csv(csv_path)

            # Save old spread for comparison
            old_spread = df["book_spread"].copy() if "book_spread" in df.columns else pd.Series(dtype=float)

            # Drop old line/edge columns
            drop = [c for c in LINE_COLS if c in df.columns]
            df = df.drop(columns=drop)

            # Merge new lines
            df = df.merge(lines_dedup[available], on="gameId", how="left")

            # Recalculate edges
            df = recalc_edges(df)

            # Check if anything changed
            new_spread = df["book_spread"] if "book_spread" in df.columns else pd.Series(dtype=float)
            if len(old_spread) == len(new_spread):
                diffs = (old_spread.fillna(0) != new_spread.fillna(0)).sum()
            else:
                diffs = len(df)

            if diffs > 0:
                changed += 1
                label = f"  {csv_path.name}: {diffs} games updated"
                if args.dry_run:
                    print(f"  [DRY RUN] {label}")
                else:
                    df.to_csv(csv_path, index=False)
                    print(label)
            # else: no changes, skip silently

    if args.dry_run:
        print(f"\nDry run complete. {changed} files would be updated.")
    else:
        print(f"\nDone. Updated {changed} CSV files.")
        if changed > 0:
            print("Run csv_to_json.py to regenerate site JSONs, or use the backfill-season command.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

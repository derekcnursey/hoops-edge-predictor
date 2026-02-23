"""Multi-season sigma-filtered backtest across all seasons 2016-2026."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2025, 2026]


def load_season_predictions(season: int) -> pd.DataFrame:
    """Load predictions from JSON and merge with actual scores."""
    json_path = config.PREDICTIONS_DIR / "json" / f"season_{season}.json"
    if not json_path.exists():
        return pd.DataFrame()

    with open(json_path) as f:
        records = json.load(f)
    preds = pd.DataFrame(records)

    # Load features to get actual scores
    feat_path = config.FEATURES_DIR / f"season_{season}_features.parquet"
    if not feat_path.exists():
        return pd.DataFrame()
    feats = pd.read_parquet(feat_path)
    scores = feats[["gameId", "homeScore", "awayScore"]].copy()
    scores = scores.dropna(subset=["homeScore", "awayScore"])
    scores["actual_margin"] = scores["homeScore"] - scores["awayScore"]

    df = preds.merge(scores[["gameId", "actual_margin"]], on="gameId", how="inner")
    return df


def compute_roi(df: pd.DataFrame, threshold: float, sigma_filter: float | None = None) -> dict:
    """Compute ROI for a given threshold and optional sigma filter."""
    with_book = df.dropna(subset=["book_spread"]).copy()

    if sigma_filter is not None:
        with_book = with_book[with_book["spread_sigma"] < sigma_filter]

    bets = with_book[with_book["spread_diff"].abs() > threshold]
    if len(bets) == 0:
        return {"n_bets": 0, "wins": 0, "losses": 0, "win_rate": 0, "roi": 0}

    wins = 0
    losses = 0
    for _, row in bets.iterrows():
        cover_margin = row["actual_margin"] + row["book_spread"]
        if row["spread_diff"] < 0:
            if cover_margin > 0:
                wins += 1
            elif cover_margin < 0:
                losses += 1
        else:
            if cover_margin < 0:
                wins += 1
            elif cover_margin > 0:
                losses += 1

    n_bets = wins + losses
    win_rate = wins / n_bets if n_bets > 0 else 0
    roi = (wins * (100 / 110) - losses) / max(n_bets, 1) * 100

    return {"n_bets": n_bets, "wins": wins, "losses": losses, "win_rate": win_rate, "roi": roi}


def main():
    all_results = []
    all_dfs = []

    print("Loading predictions for all seasons...")
    for season in SEASONS:
        df = load_season_predictions(season)
        if df.empty:
            print(f"  Season {season}: NO DATA")
            continue
        n_with_spread = df["book_spread"].notna().sum()
        print(f"  Season {season}: {len(df)} games, {n_with_spread} with spreads")
        all_dfs.append((season, df))

    print()

    # Compute per-season sigma thresholds from each season's own data
    report_lines = []
    report_lines.append("# Multi-Season Sigma-Filtered Backtest\n")

    # Table header
    header = (
        "| Season | Games w/ Spread | "
        "Unfilt ROI@3 | Unfilt ROI@5 | "
        "Sig<med ROI@3 | Sig<med ROI@5 | "
        "Sig<p25 ROI@3 | Sig<p25 ROI@5 |"
    )
    separator = "|--------|----------------|" + "-------------|" * 6

    table_lines = [header, separator]

    # Aggregate accumulators
    agg_data = {
        "unfilt_3": {"wins": 0, "losses": 0},
        "unfilt_5": {"wins": 0, "losses": 0},
        "med_3": {"wins": 0, "losses": 0},
        "med_5": {"wins": 0, "losses": 0},
        "p25_3": {"wins": 0, "losses": 0},
        "p25_5": {"wins": 0, "losses": 0},
    }

    season_results = []

    for season, df in all_dfs:
        with_book = df.dropna(subset=["book_spread"])
        n_games = len(with_book)

        if n_games == 0:
            continue

        median_sigma = with_book["spread_sigma"].median()
        p25_sigma = with_book["spread_sigma"].quantile(0.25)

        # Compute ROI for each scenario
        u3 = compute_roi(df, 3)
        u5 = compute_roi(df, 5)
        m3 = compute_roi(df, 3, sigma_filter=median_sigma)
        m5 = compute_roi(df, 5, sigma_filter=median_sigma)
        p3 = compute_roi(df, 3, sigma_filter=p25_sigma)
        p5 = compute_roi(df, 5, sigma_filter=p25_sigma)

        # Accumulate for aggregate
        for key, result in [("unfilt_3", u3), ("unfilt_5", u5),
                             ("med_3", m3), ("med_5", m5),
                             ("p25_3", p3), ("p25_5", p5)]:
            agg_data[key]["wins"] += result["wins"]
            agg_data[key]["losses"] += result["losses"]

        season_results.append({
            "season": season, "n_games": n_games,
            "median_sigma": median_sigma, "p25_sigma": p25_sigma,
            "u3": u3, "u5": u5, "m3": m3, "m5": m5, "p3": p3, "p5": p5,
        })

        def fmt_roi(r: dict) -> str:
            if r["n_bets"] == 0:
                return "-- (0)"
            return f"{r['roi']:+.1f}% ({r['n_bets']})"

        line = (
            f"| {season} | {n_games:>14} | "
            f"{fmt_roi(u3):>11} | {fmt_roi(u5):>11} | "
            f"{fmt_roi(m3):>11} | {fmt_roi(m5):>11} | "
            f"{fmt_roi(p3):>11} | {fmt_roi(p5):>11} |"
        )
        table_lines.append(line)

    # Aggregate row
    def agg_roi(key: str) -> dict:
        w, l = agg_data[key]["wins"], agg_data[key]["losses"]
        n = w + l
        wr = w / n if n > 0 else 0
        roi = (w * (100 / 110) - l) / max(n, 1) * 100
        return {"n_bets": n, "wins": w, "losses": l, "win_rate": wr, "roi": roi}

    au3 = agg_roi("unfilt_3")
    au5 = agg_roi("unfilt_5")
    am3 = agg_roi("med_3")
    am5 = agg_roi("med_5")
    ap3 = agg_roi("p25_3")
    ap5 = agg_roi("p25_5")

    def fmt_roi(r: dict) -> str:
        if r["n_bets"] == 0:
            return "-- (0)"
        return f"{r['roi']:+.1f}% ({r['n_bets']})"

    agg_line = (
        f"| **AGG** | {'':>14} | "
        f"{fmt_roi(au3):>11} | {fmt_roi(au5):>11} | "
        f"{fmt_roi(am3):>11} | {fmt_roi(am5):>11} | "
        f"{fmt_roi(ap3):>11} | {fmt_roi(ap5):>11} |"
    )
    table_lines.append(agg_line)

    table_str = "\n".join(table_lines)
    print(table_str)
    report_lines.append(table_str)

    # Summary statistics
    print("\n\n## Summary Statistics\n")
    report_lines.append("\n\n## Summary Statistics\n")

    # How many seasons ROI-positive?
    n_pos_p25_3 = sum(1 for r in season_results if r["p3"]["roi"] > 0 and r["p3"]["n_bets"] > 0)
    n_pos_p25_5 = sum(1 for r in season_results if r["p5"]["roi"] > 0 and r["p5"]["n_bets"] > 0)
    n_seasons = len(season_results)

    summary = []
    summary.append(f"Seasons with sigma<p25 ROI-positive at threshold 3: **{n_pos_p25_3}/{n_seasons}**")
    summary.append(f"Seasons with sigma<p25 ROI-positive at threshold 5: **{n_pos_p25_5}/{n_seasons}**")
    summary.append("")
    summary.append(f"Aggregate sigma<p25 threshold 3: {ap3['wins']}W-{ap3['losses']}L, "
                   f"win rate={ap3['win_rate']:.1%}, ROI={ap3['roi']:+.1f}%")
    summary.append(f"Aggregate sigma<p25 threshold 5: {ap5['wins']}W-{ap5['losses']}L, "
                   f"win rate={ap5['win_rate']:.1%}, ROI={ap5['roi']:+.1f}%")

    # Binomial tests
    summary.append("\n### Statistical Significance (Binomial Test vs 50%)\n")
    for label, data in [("Unfilt @3", au3), ("Unfilt @5", au5),
                         ("Sigma<med @3", am3), ("Sigma<med @5", am5),
                         ("Sigma<p25 @3", ap3), ("Sigma<p25 @5", ap5)]:
        if data["n_bets"] > 0:
            p_val = stats.binomtest(data["wins"], data["n_bets"], 0.5, alternative="greater").pvalue
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            summary.append(
                f"  {label:>15}: {data['wins']}W / {data['n_bets']}N = {data['win_rate']:.1%}, "
                f"p={p_val:.4f} {sig}"
            )

    # Per-season win rates for sigma<p25 @3
    summary.append("\n### Per-Season Win Rates (Sigma<p25, Threshold 3)\n")
    for r in season_results:
        p3 = r["p3"]
        if p3["n_bets"] > 0:
            summary.append(f"  {r['season']}: {p3['wins']}W-{p3['losses']}L = {p3['win_rate']:.1%} "
                           f"(ROI {p3['roi']:+.1f}%, sig_cutoff={r['p25_sigma']:.1f})")

    summary_str = "\n".join(summary)
    print(summary_str)
    report_lines.append(summary_str)

    # Save report
    report_path = Path(__file__).resolve().parent.parent / "reports" / "multi_season_sigma_backtest.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))
    print(f"\n\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

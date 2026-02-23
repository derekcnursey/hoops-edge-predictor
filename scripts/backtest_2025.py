"""Full backtest analysis for 2025 season predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config


def load_backtest_data() -> pd.DataFrame:
    """Load predictions and merge with actual scores."""
    # Load predictions
    preds = pd.read_csv(config.PREDICTIONS_DIR / "preds_today.csv")

    # Load features (has actual scores)
    feats = pd.read_parquet(config.FEATURES_DIR / "season_2025_features.parquet")
    scores = feats[["gameId", "homeScore", "awayScore"]].copy()
    scores = scores.dropna(subset=["homeScore", "awayScore"])
    scores["actual_margin"] = scores["homeScore"] - scores["awayScore"]

    # Merge
    df = preds.merge(scores[["gameId", "actual_margin", "homeScore", "awayScore"]], on="gameId", how="inner")
    return df


def compute_mae_analysis(df: pd.DataFrame) -> str:
    """Section a: Model vs Book vs Raw ratings MAE."""
    lines = []
    lines.append("## a) MODEL vs BOOK vs RAW RATINGS MAE\n")

    # Filter to games with book spread
    with_book = df.dropna(subset=["book_spread"])

    # Model MAE: predicted_spread is home_pts - away_pts; actual_margin is same convention
    model_mae = np.abs(with_book["predicted_spread"] - with_book["actual_margin"]).mean()

    # Book MAE: book_spread is from home perspective (negative = home favored)
    # actual_margin is home_pts - away_pts
    # book_spread = -expected_margin (e.g. -5.5 means home favored by 5.5)
    # So expected_margin = -book_spread
    book_mae = np.abs(-with_book["book_spread"] - with_book["actual_margin"]).mean()

    raw_baseline = 10.85

    lines.append(f"Games with closing spread: {len(with_book)}")
    lines.append(f"  Model MAE (predicted spread vs actual margin): **{model_mae:.2f}**")
    lines.append(f"  Book MAE (book spread vs actual margin): **{book_mae:.2f}**")
    lines.append(f"  Raw ratings baseline MAE: **{raw_baseline:.2f}**")
    lines.append("")

    if model_mae < raw_baseline:
        lines.append(f"  Model BEATS raw ratings baseline by {raw_baseline - model_mae:.2f} points")
    else:
        lines.append(f"  Model LOSES to raw ratings baseline by {model_mae - raw_baseline:.2f} points")

    if model_mae < book_mae:
        lines.append(f"  Model BEATS book by {book_mae - model_mae:.2f} points")
    else:
        lines.append(f"  Model LOSES to book by {model_mae - book_mae:.2f} points")

    return "\n".join(lines)


def compute_roi_table(df: pd.DataFrame, thresholds: list[float], label: str = "", sigma_filter: float | None = None) -> str:
    """Section b/c: ROI table for ATS betting."""
    lines = []

    with_book = df.dropna(subset=["book_spread"]).copy()

    if sigma_filter is not None:
        with_book = with_book[with_book["spread_sigma"] < sigma_filter]
        lines.append(f"\n### {label} (sigma < {sigma_filter:.1f}, n={len(with_book)})\n")
    else:
        lines.append(f"\n### {label}\n")

    lines.append("| Threshold | Bets | Wins | Losses | Push | Win Rate | ROI (-110) |")
    lines.append("|-----------|------|------|--------|------|----------|------------|")

    # model_spread = -predicted_spread (book convention)
    # spread_diff = model_spread - book_spread
    # If spread_diff > 0: model thinks home is worse than book → bet AWAY ATS
    # If spread_diff < 0: model thinks home is better than book → bet HOME ATS
    #
    # ATS check:
    #   Betting HOME ATS: home covers if actual_margin + book_spread > 0
    #   Betting AWAY ATS: away covers if actual_margin + book_spread < 0

    for thresh in thresholds:
        bets = with_book[with_book["spread_diff"].abs() > thresh].copy()
        if len(bets) == 0:
            lines.append(f"| {thresh} | 0 | - | - | - | - | - |")
            continue

        wins = 0
        losses = 0
        pushes = 0

        for _, row in bets.iterrows():
            # actual_margin = homeScore - awayScore
            # book_spread is from home perspective (negative = home favored)
            cover_margin = row["actual_margin"] + row["book_spread"]

            if row["spread_diff"] < 0:
                # Bet HOME to cover
                if cover_margin > 0:
                    wins += 1
                elif cover_margin < 0:
                    losses += 1
                else:
                    pushes += 1
            else:
                # Bet AWAY to cover
                if cover_margin < 0:
                    wins += 1
                elif cover_margin > 0:
                    losses += 1
                else:
                    pushes += 1

        n_bets = wins + losses  # pushes don't count
        win_rate = wins / n_bets if n_bets > 0 else 0
        # ROI at -110: win pays 100/110, loss pays -1
        roi = (wins * (100 / 110) - losses) / max(n_bets, 1) * 100

        lines.append(
            f"| {thresh} | {wins + losses + pushes} | {wins} | {losses} | {pushes} | "
            f"{win_rate:.1%} | {roi:+.1f}% |"
        )

    return "\n".join(lines)


def compute_calibration(df: pd.DataFrame) -> str:
    """Section d: Calibration check."""
    lines = []
    lines.append("\n## d) CALIBRATION CHECK\n")

    with_scores = df.dropna(subset=["homeScore", "awayScore"]).copy()
    with_scores["home_won"] = (with_scores["actual_margin"] > 0).astype(int)

    buckets = [
        ("> 0.7", with_scores[with_scores["home_win_prob"] > 0.7]),
        ("0.6 - 0.7", with_scores[(with_scores["home_win_prob"] > 0.6) & (with_scores["home_win_prob"] <= 0.7)]),
        ("0.5 - 0.6", with_scores[(with_scores["home_win_prob"] > 0.5) & (with_scores["home_win_prob"] <= 0.6)]),
        ("< 0.5", with_scores[with_scores["home_win_prob"] <= 0.5]),
    ]

    lines.append("| Predicted Prob | Games | Actual Win Rate | Calibration |")
    lines.append("|----------------|-------|-----------------|-------------|")

    for label, bucket in buckets:
        if len(bucket) == 0:
            lines.append(f"| {label} | 0 | - | - |")
            continue
        actual_rate = bucket["home_won"].mean()
        n = len(bucket)
        cal = "Good" if abs(actual_rate - bucket["home_win_prob"].mean()) < 0.05 else "Off"
        lines.append(f"| {label} | {n} | {actual_rate:.1%} | {cal} |")

    return "\n".join(lines)


def compute_biggest_edges(df: pd.DataFrame, n: int = 20) -> str:
    """Section e: Games with largest |spread_diff|."""
    lines = []
    lines.append(f"\n## e) TOP {n} LARGEST |model_spread - book_spread| GAMES\n")

    with_book = df.dropna(subset=["book_spread"]).copy()
    with_book["abs_diff"] = with_book["spread_diff"].abs()
    top = with_book.nlargest(n, "abs_diff")

    lines.append("| Game ID | Home | Away | Model Spread | Book Spread | Diff | Actual Margin | Covered? |")
    lines.append("|---------|------|------|-------------|-------------|------|---------------|----------|")

    for _, row in top.iterrows():
        cover_margin = row["actual_margin"] + row["book_spread"]
        if row["spread_diff"] < 0:
            # Bet home
            covered = "YES" if cover_margin > 0 else ("PUSH" if cover_margin == 0 else "NO")
            side = "HOME"
        else:
            # Bet away
            covered = "YES" if cover_margin < 0 else ("PUSH" if cover_margin == 0 else "NO")
            side = "AWAY"

        lines.append(
            f"| {int(row['gameId'])} | {int(row['homeTeamId'])} | {int(row['awayTeamId'])} | "
            f"{row['model_spread']:+.1f} | {row['book_spread']:+.1f} | "
            f"{row['spread_diff']:+.1f} | {row['actual_margin']:+.0f} | "
            f"{covered} ({side}) |"
        )

    return "\n".join(lines)


def main():
    print("Loading backtest data...")
    df = load_backtest_data()
    print(f"  Total games with predictions + scores: {len(df)}")
    print(f"  Games with book spread: {df['book_spread'].notna().sum()}")
    print()

    report_lines = []
    report_lines.append("# ML Backtest Report — Season 2025\n")
    report_lines.append(f"Total games: {len(df)}")
    report_lines.append(f"Games with closing spread: {df['book_spread'].notna().sum()}\n")

    # a) MAE analysis
    mae_section = compute_mae_analysis(df)
    print(mae_section)
    report_lines.append(mae_section)

    # b) ROI table
    thresholds = [1, 2, 3, 4, 5, 6, 7]
    roi_section = compute_roi_table(df, thresholds, label="b) ROI TABLE — ATS Betting")
    print(roi_section)
    report_lines.append(roi_section)

    # c) Sigma-filtered ROI
    with_book = df.dropna(subset=["book_spread"])
    median_sigma = with_book["spread_sigma"].median()
    p25_sigma = with_book["spread_sigma"].quantile(0.25)
    print(f"\nSigma stats: median={median_sigma:.2f}, p25={p25_sigma:.2f}")
    report_lines.append(f"\n## c) SIGMA-FILTERED ROI\n")
    report_lines.append(f"Sigma stats: median={median_sigma:.2f}, p25={p25_sigma:.2f}")

    sigma_med = compute_roi_table(df, thresholds, label=f"Sigma < median ({median_sigma:.1f})", sigma_filter=median_sigma)
    print(sigma_med)
    report_lines.append(sigma_med)

    sigma_p25 = compute_roi_table(df, thresholds, label=f"Sigma < p25 ({p25_sigma:.1f})", sigma_filter=p25_sigma)
    print(sigma_p25)
    report_lines.append(sigma_p25)

    # d) Calibration
    cal_section = compute_calibration(df)
    print(cal_section)
    report_lines.append(cal_section)

    # e) Biggest edges
    edges_section = compute_biggest_edges(df)
    print(edges_section)
    report_lines.append(edges_section)

    # Save report
    report_path = Path(__file__).resolve().parent.parent / "reports" / "ml_backtest_2025.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

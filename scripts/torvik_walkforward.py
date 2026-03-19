#!/usr/bin/env python3
"""Walk-forward evaluation comparing gold-layer vs Torvik efficiency sources.

For each holdout year (2019-2025, excluding 2021):
  1. Build features with --efficiency-source torvik
  2. Train regressor + classifier on all prior seasons (excl 2021)
  3. Predict holdout season
  4. Report MAE, RMSE, sigma, ATS at edge >= 5%, monthly breakdown

Usage:
    poetry run python scripts/torvik_walkforward.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.features import build_features, get_feature_matrix, get_targets, load_research_lines
from src.trainer import fit_scaler, impute_column_means, train_regressor, train_classifier
from src.architecture import MLPRegressor, MLPClassifier, gaussian_nll_loss
from src.infer import normal_cdf
import torch

HOLDOUT_YEARS = [2019, 2020, 2022, 2023, 2024, 2025]
TRAIN_START = 2015
EXCLUDE_SEASONS = config.EXCLUDE_SEASONS  # [2021]
MIN_DATE = "12-01"
EDGE_THRESHOLD = 0.05  # 5%

# Baseline from critical_fixes_report.md
BASELINE = {
    2019: {"mae": 9.53, "roi": -4.2, "home_pct": 62},
    2020: {"mae": 9.60, "roi": -2.8, "home_pct": 54},
    2022: {"mae": 9.42, "roi": 2.2, "home_pct": 52},
    2023: {"mae": 9.56, "roi": 0.2, "home_pct": 46},
    2024: {"mae": 9.88, "roi": -1.3, "home_pct": 51},
    2025: {"mae": 9.68, "roi": -2.1, "home_pct": 45},
}


def build_or_load_features(
    season: int, efficiency_source: str, cache_dir: Path
) -> pd.DataFrame:
    """Build features for a season, caching to parquet."""
    suffix = f"_{efficiency_source}"
    cache_path = cache_dir / f"season_{season}{suffix}_features.parquet"
    if cache_path.exists():
        print(f"  Loading cached features: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"  Building features for {season} (source={efficiency_source})...")
    df = build_features(
        season,
        no_garbage=True,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
        adjust_ff_method=config.ADJUST_FF_METHOD,
        efficiency_source=efficiency_source,
    )
    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"  Cached: {cache_path} ({len(df)} games)")
    return df


def filter_by_min_date(df: pd.DataFrame, min_month_day: str) -> pd.DataFrame:
    """Filter out games before min_month_day within each season."""
    dates = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    game_dates = dates.dt.tz_localize(None).dt.normalize()
    month, day = (int(x) for x in min_month_day.split("-"))
    game_years = game_dates.dt.year
    game_months = game_dates.dt.month
    season_year = game_years.where(game_months <= 7, game_years + 1)
    cutoff_year = (season_year - 1) if month >= 8 else season_year
    cutoffs = pd.to_datetime(
        cutoff_year.astype(int).astype(str) + f"-{month:02d}-{day:02d}",
        errors="coerce",
    )
    return df[game_dates >= cutoffs].reset_index(drop=True)


def evaluate_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    holdout_season: int,
) -> dict:
    """Train on train_df, evaluate on holdout_df. Return metrics dict."""
    # Prepare training data
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])
    train_df = train_df[(train_df["homeScore"] != 0) | (train_df["awayScore"] != 0)]

    X_train = get_feature_matrix(train_df).values.astype(np.float32)
    targets = get_targets(train_df)
    y_spread = targets["spread_home"].values.astype(np.float32)
    y_win = targets["home_win"].values.astype(np.float32)

    n_nan = np.isnan(X_train).sum()
    X_train = impute_column_means(X_train)
    if n_nan > 0:
        print(f"    Imputed {n_nan:,} NaN values in training data")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # Train
    reg_hp = {"epochs": 100, "hidden1": 384, "hidden2": 256, "dropout": 0.2}
    regressor = train_regressor(X_train_scaled, y_spread, hparams=reg_hp)
    cls_hp = {"epochs": 100, "hidden1": 384, "dropout": 0.2}
    classifier = train_classifier(X_train_scaled, y_win, hparams=cls_hp)

    # Prepare holdout data
    holdout_played = holdout_df.dropna(subset=["homeScore", "awayScore"])
    holdout_played = holdout_played[
        (holdout_played["homeScore"] != 0) | (holdout_played["awayScore"] != 0)
    ]
    if holdout_played.empty:
        return {}

    X_hold = get_feature_matrix(holdout_played).values.astype(np.float32)
    nan_mask = np.isnan(X_hold)
    if nan_mask.any():
        col_means = scaler.mean_
        for j in range(X_hold.shape[1]):
            X_hold[nan_mask[:, j], j] = col_means[j]

    X_hold_scaled = scaler.transform(X_hold)
    X_tensor = torch.tensor(X_hold_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        mu_raw, log_sigma_raw = regressor(X_tensor)
        sigma = torch.exp(log_sigma_raw).clamp(min=0.5, max=30.0)
        logits = classifier(X_tensor)
        home_win_prob = torch.sigmoid(logits).numpy()

    mu = mu_raw.numpy().flatten()
    sigma = sigma.numpy().flatten()

    actual_spread = (
        holdout_played["homeScore"].astype(float).values
        - holdout_played["awayScore"].astype(float).values
    )

    # Get lines for ATS eval
    lines = load_research_lines(holdout_season)
    book_spread = np.full(len(holdout_played), np.nan)
    if not lines.empty:
        lines["spread"] = pd.to_numeric(lines["spread"], errors="coerce")
        _provider_rank = {"Draft Kings": 0, "ESPN BET": 1, "Bovada": 2}
        lines_dedup = (
            lines.assign(
                _has_spread=lines["spread"].notna().astype(int),
                _prov_rank=lines["provider"].map(_provider_rank).fillna(99),
            )
            .sort_values(["_has_spread", "_prov_rank"], ascending=[False, True])
            .drop_duplicates(subset=["gameId"], keep="first")
            .drop(columns=["_has_spread", "_prov_rank"])
        )
        spread_map = dict(zip(lines_dedup["gameId"], lines_dedup["spread"]))
        for i, (_, row) in enumerate(holdout_played.iterrows()):
            gid = int(row["gameId"])
            if gid in spread_map:
                book_spread[i] = spread_map[gid]

    # Compute metrics
    has_book = ~np.isnan(book_spread)
    mae_all = np.mean(np.abs(mu - actual_spread))
    rmse_all = np.sqrt(np.mean((mu - actual_spread) ** 2))
    mean_sigma = np.mean(sigma)

    # Book-spread games only
    if has_book.sum() > 0:
        mae_book = np.mean(np.abs(mu[has_book] - actual_spread[has_book]))
        rmse_book = np.sqrt(np.mean((mu[has_book] - actual_spread[has_book]) ** 2))
    else:
        mae_book = mae_all
        rmse_book = rmse_all

    # ATS evaluation at edge >= 5%
    edge_home_points = mu + book_spread  # predicted_spread + book_spread
    sigma_safe = np.clip(sigma, 0.5, None)
    edge_z = edge_home_points / sigma_safe
    home_cover_prob = normal_cdf(edge_z)

    pick_side = np.where(edge_home_points >= 0, "HOME", "AWAY")
    pick_cover_prob = np.where(edge_home_points >= 0, home_cover_prob, 1.0 - home_cover_prob)
    pick_prob_edge = pick_cover_prob - 0.5238  # -110 breakeven

    # Filter to edge >= threshold and has book spread
    edge_mask = has_book & (pick_prob_edge >= EDGE_THRESHOLD)
    n_picks = edge_mask.sum()

    wins = 0
    losses = 0
    if n_picks > 0:
        for i in np.where(edge_mask)[0]:
            picked_home = pick_side[i] == "HOME"
            actual_margin = actual_spread[i]
            spread = book_spread[i]
            # Home covers: actual_margin + spread > 0
            home_covers = actual_margin + spread > 0
            away_covers = actual_margin + spread < 0
            push = actual_margin + spread == 0

            if push:
                continue
            if picked_home and home_covers:
                wins += 1
            elif not picked_home and away_covers:
                wins += 1
            else:
                losses += 1

    total_decided = wins + losses
    win_rate = wins / total_decided if total_decided > 0 else 0
    roi = (wins * 0.9091 - losses) / total_decided * 100 if total_decided > 0 else 0
    units = wins * 0.9091 - losses

    # Home/away pick distribution
    home_picks = np.sum((pick_side == "HOME") & edge_mask)
    away_picks = np.sum((pick_side == "AWAY") & edge_mask)
    home_pct = int(round(home_picks / n_picks * 100)) if n_picks > 0 else 0

    # Monthly breakdown
    holdout_dates = pd.to_datetime(holdout_played["startDate"], errors="coerce", utc=True)
    holdout_months = holdout_dates.dt.tz_convert("America/New_York").dt.month

    monthly = {}
    for m in [11, 12, 1, 2, 3, 4]:
        m_mask = edge_mask & (holdout_months.values == m)
        m_wins = 0
        m_losses = 0
        for i in np.where(m_mask)[0]:
            picked_home = pick_side[i] == "HOME"
            actual_margin = actual_spread[i]
            spread = book_spread[i]
            home_covers = actual_margin + spread > 0
            away_covers = actual_margin + spread < 0
            push = actual_margin + spread == 0
            if push:
                continue
            if picked_home and home_covers:
                m_wins += 1
            elif not picked_home and away_covers:
                m_wins += 1
            else:
                m_losses += 1

        m_total = m_wins + m_losses
        m_wr = m_wins / m_total if m_total > 0 else 0
        m_roi = (m_wins * 0.9091 - m_losses) / m_total * 100 if m_total > 0 else 0
        month_name = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}[m]
        monthly[month_name] = {
            "wins": m_wins, "losses": m_losses, "total": m_total,
            "win_rate": m_wr, "roi": m_roi,
        }

    return {
        "year": holdout_season,
        "mae": round(mae_book, 2),
        "rmse": round(rmse_book, 2),
        "sigma": round(mean_sigma, 1),
        "n_games": len(holdout_played),
        "n_book": int(has_book.sum()),
        "n_picks": int(n_picks),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate * 100, 1),
        "roi": round(roi, 1),
        "units": round(units, 1),
        "home_pct": home_pct,
        "monthly": monthly,
    }


def main():
    print("=" * 70)
    print("Torvik Hybrid Walk-Forward Evaluation")
    print("=" * 70)

    cache_dir = config.FEATURES_DIR / "torvik"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build features for all seasons
    all_seasons = list(range(TRAIN_START, max(HOLDOUT_YEARS) + 1))
    all_seasons = [s for s in all_seasons if s not in EXCLUDE_SEASONS]
    print(f"\nSeasons: {all_seasons}")
    print(f"Excluded: {EXCLUDE_SEASONS}")
    print(f"Holdout years: {HOLDOUT_YEARS}")

    season_dfs: dict[int, pd.DataFrame] = {}
    for s in all_seasons:
        df = build_or_load_features(s, "torvik", cache_dir)
        if df.empty:
            print(f"  WARNING: No features for season {s}")
        else:
            season_dfs[s] = df

    # Step 2: Walk-forward evaluation
    results = []
    for holdout_year in HOLDOUT_YEARS:
        print(f"\n{'='*50}")
        print(f"Holdout: {holdout_year}")
        print(f"{'='*50}")

        train_seasons = [s for s in all_seasons if s < holdout_year and s not in EXCLUDE_SEASONS]
        print(f"  Training on: {train_seasons}")

        train_dfs = []
        for s in train_seasons:
            if s in season_dfs:
                train_dfs.append(season_dfs[s])

        if not train_dfs:
            print(f"  No training data! Skipping.")
            continue

        train_df = pd.concat(train_dfs, ignore_index=True)
        train_df = filter_by_min_date(train_df, MIN_DATE)
        print(f"  Training games (after date filter): {len(train_df)}")

        holdout_df = season_dfs.get(holdout_year)
        if holdout_df is None or holdout_df.empty:
            print(f"  No holdout data! Skipping.")
            continue

        print(f"  Holdout games: {len(holdout_df)}")

        metrics = evaluate_holdout(train_df, holdout_df, holdout_year)
        if metrics:
            results.append(metrics)
            print(f"\n  Results for {holdout_year}:")
            print(f"    MAE: {metrics['mae']}")
            print(f"    RMSE: {metrics['rmse']}")
            print(f"    Sigma: {metrics['sigma']}")
            print(f"    ATS (edge>=5%): {metrics['wins']}-{metrics['losses']} "
                  f"({metrics['win_rate']}%), ROI: {metrics['roi']:.1f}%")
            print(f"    Home picks: {metrics['home_pct']}%")
            if metrics.get("monthly"):
                print(f"    Monthly:")
                for m, mv in metrics["monthly"].items():
                    if mv["total"] > 0:
                        print(f"      {m}: {mv['wins']}-{mv['losses']} "
                              f"({mv['win_rate']:.1%}), ROI: {mv['roi']:.1f}%")

    # Step 3: Print comparison table
    if results:
        print("\n" + "=" * 70)
        print("COMPARISON TABLE: A_baseline vs B_torvik")
        print("=" * 70)
        print(f"{'Year':>6} | {'A_MAE':>6} | {'B_MAE':>6} | {'dMAE':>6} | "
              f"{'A_ROI':>6} | {'B_ROI':>6} | {'dROI':>6} | "
              f"{'B_WR':>5} | {'B_W-L':>8}")
        print("-" * 80)

        a_maes, b_maes, a_rois, b_rois = [], [], [], []
        for r in results:
            yr = r["year"]
            baseline = BASELINE.get(yr, {})
            a_mae = baseline.get("mae", 0)
            b_mae = r["mae"]
            a_roi = baseline.get("roi", 0)
            b_roi = r["roi"]
            d_mae = b_mae - a_mae if a_mae else 0
            d_roi = b_roi - a_roi if a_roi else 0

            print(f"{yr:>6} | {a_mae:>6.2f} | {b_mae:>6.2f} | {d_mae:>+6.2f} | "
                  f"{a_roi:>5.1f}% | {b_roi:>5.1f}% | {d_roi:>+5.1f}% | "
                  f"{r['win_rate']:>4.1f}% | {r['wins']:>3}-{r['losses']:<3}")

            a_maes.append(a_mae)
            b_maes.append(b_mae)
            a_rois.append(a_roi)
            b_rois.append(b_roi)

        avg_a_mae = np.mean(a_maes) if a_maes else 0
        avg_b_mae = np.mean(b_maes) if b_maes else 0
        avg_a_roi = np.mean(a_rois) if a_rois else 0
        avg_b_roi = np.mean(b_rois) if b_rois else 0
        print("-" * 80)
        print(f"{'AVG':>6} | {avg_a_mae:>6.2f} | {avg_b_mae:>6.2f} | "
              f"{avg_b_mae - avg_a_mae:>+6.2f} | "
              f"{avg_a_roi:>5.1f}% | {avg_b_roi:>5.1f}% | "
              f"{avg_b_roi - avg_a_roi:>+5.1f}%")

    # Step 4: Save report
    report_path = PROJECT_ROOT / "reports" / "torvik_hybrid_evaluation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path, results)
    print(f"\nReport saved to: {report_path}")


def _write_report(path: Path, results: list[dict]):
    """Write the evaluation report as markdown."""
    lines = [
        "# Torvik Hybrid Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat()[:19]}",
        "",
        "## Configuration",
        "- Efficiency source: Torvik daily_data from S3",
        "- PBP features: Unchanged (from gold layer pipeline)",
        "- Holdout years: " + ", ".join(str(r["year"]) for r in results),
        f"- Excluded seasons: {EXCLUDE_SEASONS}",
        f"- Training date filter: >= {MIN_DATE}",
        f"- Edge threshold: {EDGE_THRESHOLD*100:.0f}%",
        "- Architecture: MLPRegressor(384/256, d=0.2) + MLPClassifier(384, d=0.2)",
        "",
        "## Comparison: A_baseline (gold layer) vs B_torvik (Torvik efficiencies)",
        "",
        "| Year | A_MAE | B_MAE | ΔMAE | A_ROI | B_ROI | ΔROI | B_WR | B_W-L |",
        "|------|-------|-------|------|-------|-------|------|------|-------|",
    ]

    a_maes, b_maes, a_rois, b_rois = [], [], [], []
    for r in results:
        yr = r["year"]
        bl = BASELINE.get(yr, {})
        a_mae = bl.get("mae", 0)
        b_mae = r["mae"]
        a_roi = bl.get("roi", 0)
        b_roi = r["roi"]
        d_mae = b_mae - a_mae
        d_roi = b_roi - a_roi
        lines.append(
            f"| {yr} | {a_mae:.2f} | {b_mae:.2f} | {d_mae:+.2f} | "
            f"{a_roi:.1f}% | {b_roi:.1f}% | {d_roi:+.1f}% | "
            f"{r['win_rate']:.1f}% | {r['wins']}-{r['losses']} |"
        )
        a_maes.append(a_mae)
        b_maes.append(b_mae)
        a_rois.append(a_roi)
        b_rois.append(b_roi)

    if a_maes:
        avg_a = np.mean(a_maes)
        avg_b = np.mean(b_maes)
        avg_ar = np.mean(a_rois)
        avg_br = np.mean(b_rois)
        lines.append(
            f"| **AVG** | **{avg_a:.2f}** | **{avg_b:.2f}** | **{avg_b-avg_a:+.2f}** | "
            f"**{avg_ar:.1f}%** | **{avg_br:.1f}%** | **{avg_br-avg_ar:+.1f}%** | | |"
        )

    lines.extend(["", "## Per-Year Details", ""])
    for r in results:
        lines.append(f"### {r['year']}")
        lines.append(f"- Games: {r['n_games']} ({r['n_book']} with book spread)")
        lines.append(f"- MAE: {r['mae']}, RMSE: {r['rmse']}, Sigma: {r['sigma']}")
        lines.append(f"- ATS picks (edge >= 5%): {r['n_picks']}")
        lines.append(f"- Record: {r['wins']}-{r['losses']} ({r['win_rate']:.1f}%)")
        lines.append(f"- ROI: {r['roi']:.1f}%, Units: {r['units']:.1f}")
        lines.append(f"- Home pick %: {r['home_pct']}%")
        if r.get("monthly"):
            lines.append("- Monthly ATS breakdown:")
            lines.append("  | Month | W-L | WR | ROI |")
            lines.append("  |-------|-----|-----|-----|")
            for m, mv in r["monthly"].items():
                if mv["total"] > 0:
                    lines.append(
                        f"  | {m} | {mv['wins']}-{mv['losses']} | "
                        f"{mv['win_rate']:.1%} | {mv['roi']:.1f}% |"
                    )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

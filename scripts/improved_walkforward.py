#!/usr/bin/env python3
"""Improved walk-forward: gold vs torvik × full vs core features.

Tests 4 configurations with best-loss checkpointing:
  A: gold  + 53 features (full)
  B: torvik + 53 features (full)
  C: gold  + 33 features (core: no extras, no HCA)
  D: torvik + 33 features (core: no extras, no HCA)

Usage:
    poetry run python scripts/improved_walkforward.py
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.features import build_features, get_feature_matrix, get_targets, load_research_lines
from src.trainer import impute_column_means, train_regressor, train_classifier
from src.architecture import gaussian_nll_loss
from src.infer import normal_cdf

HOLDOUT_YEARS = [2019, 2020, 2022, 2023, 2024, 2025]
TRAIN_START = 2015
EXCLUDE_SEASONS = config.EXCLUDE_SEASONS
MIN_DATE = "12-01"
EDGE_THRESHOLDS = [0.05, 0.10]
VAL_FRAC = 0.15

# 19 extra features to drop for "core" config
EXTRA_FEATURE_NAMES = [
    "home_rest_days", "away_rest_days", "rest_advantage",
    "home_sos_oe", "home_sos_de", "away_sos_oe", "away_sos_de",
    "home_conf_strength", "away_conf_strength",
    "home_form_delta", "away_form_delta",
    "home_tov_rate", "home_def_tov_rate", "away_tov_rate", "away_def_tov_rate",
    "home_margin_std", "away_margin_std",
    "home_team_hca", "home_team_efg_home_split", "away_team_efg_away_split",
]

# Configs to run
CONFIGS = {
    "gold_full": {"source": "gold", "trim": False},
    "torvik_full": {"source": "torvik", "trim": False},
    "gold_core": {"source": "gold", "trim": True},
    "torvik_core": {"source": "torvik", "trim": True},
}

REG_HP = {"epochs": 150, "hidden1": 384, "hidden2": 256, "dropout": 0.2,
           "lr": 1e-3, "batch_size": 512}
CLS_HP = {"epochs": 150, "hidden1": 384, "dropout": 0.2,
           "lr": 1e-3, "batch_size": 512}


def load_gold_features(season: int) -> pd.DataFrame:
    """Load cached gold-layer features (production config)."""
    path = config.FEATURES_DIR / f"season_{season}_no_garbage_adj_a0.85_p10_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Fallback: build
    print(f"  Building gold features for {season}...")
    df = build_features(
        season, no_garbage=True, extra_features=config.EXTRA_FEATURES,
        adjust_ff=True, adjust_alpha=0.85, adjust_prior_weight=10,
        adjust_ff_method="multiplicative", efficiency_source="gold",
    )
    if not df.empty:
        df.to_parquet(path, index=False)
    return df


def load_torvik_features(season: int) -> pd.DataFrame:
    """Load cached Torvik features."""
    path = config.FEATURES_DIR / "torvik" / f"season_{season}_torvik_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    print(f"  Building torvik features for {season}...")
    df = build_features(
        season, no_garbage=True, extra_features=config.EXTRA_FEATURES,
        adjust_ff=True, adjust_alpha=0.85, adjust_prior_weight=10,
        adjust_ff_method="multiplicative", efficiency_source="torvik",
    )
    if not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df


def filter_by_min_date(df: pd.DataFrame, min_month_day: str) -> pd.DataFrame:
    dates = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    game_dates = dates.dt.tz_localize(None).dt.normalize()
    month, day = (int(x) for x in min_month_day.split("-"))
    game_months = game_dates.dt.month
    game_years = game_dates.dt.year
    season_year = game_years.where(game_months <= 7, game_years + 1)
    cutoff_year = (season_year - 1) if month >= 8 else season_year
    cutoffs = pd.to_datetime(
        cutoff_year.astype(int).astype(str) + f"-{month:02d}-{day:02d}",
        errors="coerce",
    )
    return df[game_dates >= cutoffs].reset_index(drop=True)


def get_feature_cols(trim: bool) -> list[str]:
    """Return feature columns to use."""
    if trim:
        return [f for f in config.FEATURE_ORDER if f not in EXTRA_FEATURE_NAMES]
    return list(config.FEATURE_ORDER)


def evaluate_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    holdout_season: int,
    feature_cols: list[str],
    label: str,
) -> dict:
    """Train on train_df, evaluate on holdout_df with given feature columns."""
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])
    train_df = train_df[(train_df["homeScore"] != 0) | (train_df["awayScore"] != 0)]

    X_train = train_df[feature_cols].values.astype(np.float32)
    targets = get_targets(train_df)
    y_spread = targets["spread_home"].values.astype(np.float32)
    y_win = targets["home_win"].values.astype(np.float32)

    X_train = impute_column_means(X_train)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # Train with best-loss checkpointing
    reg_hp = {**REG_HP, "input_dim": len(feature_cols)}
    cls_hp = {**CLS_HP, "input_dim": len(feature_cols)}
    regressor = train_regressor(X_train_scaled, y_spread, hparams=reg_hp, val_frac=VAL_FRAC)
    classifier = train_classifier(X_train_scaled, y_win, hparams=cls_hp, val_frac=VAL_FRAC)

    # Holdout
    holdout_played = holdout_df.dropna(subset=["homeScore", "awayScore"])
    holdout_played = holdout_played[
        (holdout_played["homeScore"] != 0) | (holdout_played["awayScore"] != 0)
    ]
    if holdout_played.empty:
        return {}

    X_hold = holdout_played[feature_cols].values.astype(np.float32)
    nan_mask = np.isnan(X_hold)
    if nan_mask.any():
        for j in range(X_hold.shape[1]):
            X_hold[nan_mask[:, j], j] = scaler.mean_[j]
    X_hold_scaled = scaler.transform(X_hold)
    X_tensor = torch.tensor(X_hold_scaled, dtype=torch.float32)

    with torch.no_grad():
        regressor.eval()
        classifier.eval()
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

    # Load book spreads
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

    has_book = ~np.isnan(book_spread)
    mae = np.mean(np.abs(mu[has_book] - actual_spread[has_book])) if has_book.sum() > 0 else np.nan
    rmse = np.sqrt(np.mean((mu[has_book] - actual_spread[has_book]) ** 2)) if has_book.sum() > 0 else np.nan
    mean_sigma = np.mean(sigma)

    # ATS at multiple edge thresholds
    edge_home_points = mu + book_spread
    sigma_safe = np.clip(sigma, 0.5, None)
    edge_z = edge_home_points / sigma_safe
    home_cover_prob = normal_cdf(edge_z)
    pick_side = np.where(edge_home_points >= 0, "HOME", "AWAY")
    pick_cover_prob = np.where(edge_home_points >= 0, home_cover_prob, 1.0 - home_cover_prob)
    pick_prob_edge = pick_cover_prob - 0.5238

    ats_by_threshold = {}
    for threshold in EDGE_THRESHOLDS:
        edge_mask = has_book & (pick_prob_edge >= threshold)
        wins, losses = 0, 0
        for i in np.where(edge_mask)[0]:
            result = actual_spread[i] + book_spread[i]
            if result == 0:
                continue
            picked_home = pick_side[i] == "HOME"
            if (picked_home and result > 0) or (not picked_home and result < 0):
                wins += 1
            else:
                losses += 1
        total = wins + losses
        ats_by_threshold[threshold] = {
            "n_picks": int(edge_mask.sum()),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "roi": round((wins * 0.9091 - losses) / total * 100, 1) if total > 0 else 0,
            "units": round(wins * 0.9091 - losses, 1),
        }

    result = {
        "config": label,
        "year": holdout_season,
        "mae": round(mae, 2) if not np.isnan(mae) else None,
        "rmse": round(rmse, 2) if not np.isnan(rmse) else None,
        "sigma": round(mean_sigma, 1),
        "n_features": len(feature_cols),
        "n_games": len(holdout_played),
        "n_book": int(has_book.sum()),
    }
    # Flatten: primary threshold (5%) at top level, all thresholds in sub-dict
    primary = ats_by_threshold[EDGE_THRESHOLDS[0]]
    result.update(primary)
    result["ats"] = ats_by_threshold
    return result


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("Improved Walk-Forward: Gold vs Torvik × Full vs Core")
    print(f"Best-loss checkpointing: val_frac={VAL_FRAC}")
    print(f"Regressor: {REG_HP}")
    print(f"Classifier: {CLS_HP}")
    print("=" * 70)

    all_seasons = [s for s in range(TRAIN_START, max(HOLDOUT_YEARS) + 1)
                   if s not in EXCLUDE_SEASONS]

    # Load all features
    print("\nLoading features...")
    gold_dfs: dict[int, pd.DataFrame] = {}
    torvik_dfs: dict[int, pd.DataFrame] = {}
    for s in all_seasons:
        gold_dfs[s] = load_gold_features(s)
        torvik_dfs[s] = load_torvik_features(s)
        print(f"  Season {s}: gold={len(gold_dfs[s])}, torvik={len(torvik_dfs[s])}")

    # Run walk-forward for each config
    all_results: dict[str, list[dict]] = defaultdict(list)

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"CONFIG: {cfg_name}")
        print(f"{'='*60}")

        source_dfs = gold_dfs if cfg["source"] == "gold" else torvik_dfs
        feature_cols = get_feature_cols(cfg["trim"])
        print(f"  Features: {len(feature_cols)}")

        for holdout_year in HOLDOUT_YEARS:
            print(f"\n  Holdout {holdout_year}:")
            train_seasons = [s for s in all_seasons if s < holdout_year]
            train_dfs = [source_dfs[s] for s in train_seasons if s in source_dfs and not source_dfs[s].empty]

            if not train_dfs:
                print(f"    No training data, skipping")
                continue

            train_df = pd.concat(train_dfs, ignore_index=True)
            train_df = filter_by_min_date(train_df, MIN_DATE)

            holdout_df = source_dfs.get(holdout_year)
            if holdout_df is None or holdout_df.empty:
                print(f"    No holdout data, skipping")
                continue

            print(f"    Train: {len(train_df)} games, Holdout: {len(holdout_df)} games")

            metrics = evaluate_holdout(
                train_df, holdout_df, holdout_year, feature_cols, cfg_name,
            )
            if metrics:
                all_results[cfg_name].append(metrics)
                print(f"    MAE: {metrics['mae']}, RMSE: {metrics['rmse']}, "
                      f"Sigma: {metrics['sigma']}")
                for thr, ats in metrics["ats"].items():
                    pct = int(thr * 100)
                    print(f"    ATS @{pct}%: {ats['wins']}-{ats['losses']} "
                          f"({ats['win_rate']}%), ROI: {ats['roi']:.1f}%")

    # Print averages per threshold
    for thr in EDGE_THRESHOLDS:
        pct = int(thr * 100)
        print(f"\n{'=' * 90}")
        print(f"AVERAGES @ {pct}% EDGE THRESHOLD")
        print(f"{'=' * 90}")
        print(f"{'Config':<16} | {'Feats':>5} | {'MAE':>5} | {'RMSE':>5} | "
              f"{'Sigma':>5} | {'W-L':>9} | {'WR%':>5} | {'ROI%':>6} | {'Units':>6}")
        print("-" * 90)

        for cfg_name in CONFIGS:
            results = all_results.get(cfg_name, [])
            if not results:
                continue
            avg_mae = np.mean([r["mae"] for r in results if r["mae"] is not None])
            avg_rmse = np.mean([r["rmse"] for r in results if r["rmse"] is not None])
            avg_sigma = np.mean([r["sigma"] for r in results])
            tot_w = sum(r["ats"][thr]["wins"] for r in results)
            tot_l = sum(r["ats"][thr]["losses"] for r in results)
            tot = tot_w + tot_l
            avg_wr = tot_w / tot * 100 if tot > 0 else 0
            avg_roi = (tot_w * 0.9091 - tot_l) / tot * 100 if tot > 0 else 0
            tot_units = tot_w * 0.9091 - tot_l
            n_feats = results[0]["n_features"]
            print(f"{cfg_name:<16} | {n_feats:>5} | {avg_mae:>5.2f} | {avg_rmse:>5.2f} | "
                  f"{avg_sigma:>5.1f} | {tot_w:>4}-{tot_l:<4} | {avg_wr:>5.1f} | "
                  f"{avg_roi:>+5.1f}% | {tot_units:>+6.1f}")

    # Save report
    report_path = PROJECT_ROOT / "reports" / "improved_walkforward.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path, all_results)
    print(f"\nReport saved to: {report_path}")


def _write_report(path: Path, all_results: dict[str, list[dict]]):
    lines = [
        "# Improved Walk-Forward: Gold vs Torvik × Full vs Core",
        "",
        f"Generated: {datetime.now().isoformat()[:19]}",
        "",
        "## Configuration",
        f"- Holdout years: {HOLDOUT_YEARS}",
        f"- Excluded seasons: {list(EXCLUDE_SEASONS)}",
        f"- Best-loss checkpointing: val_frac={VAL_FRAC}",
        f"- Regressor HP: {REG_HP}",
        f"- Classifier HP: {CLS_HP}",
        f"- Edge thresholds: {[int(t*100) for t in EDGE_THRESHOLDS]}%",
        "",
        "## Configs",
        "- **gold_full**: Gold-layer efficiencies + all 53 features",
        "- **torvik_full**: Torvik efficiencies + all 53 features",
        "- **gold_core**: Gold-layer efficiencies + 33 core features (no extras/HCA)",
        "- **torvik_core**: Torvik efficiencies + 33 core features (no extras/HCA)",
        "",
        "## Per-Year Results",
    ]

    for thr in EDGE_THRESHOLDS:
        pct = int(thr * 100)
        lines.extend([
            "", f"### @ {pct}% Edge", "",
            "| Config | Feats | Year | MAE | RMSE | Sigma | W-L | WR% | ROI% | Units |",
            "|--------|-------|------|-----|------|-------|-----|-----|------|-------|",
        ])
        for cfg_name in CONFIGS:
            for r in all_results.get(cfg_name, []):
                a = r["ats"][thr]
                lines.append(
                    f"| {r['config']} | {r['n_features']} | {r['year']} | "
                    f"{r['mae']} | {r['rmse']} | {r['sigma']} | "
                    f"{a['wins']}-{a['losses']} | {a['win_rate']}% | "
                    f"{a['roi']:+.1f}% | {a['units']:+.1f} |"
                )

    for thr in EDGE_THRESHOLDS:
        pct = int(thr * 100)
        lines.extend(["", f"## Averages @ {pct}% Edge", "",
            "| Config | Feats | MAE | RMSE | Sigma | W-L | WR% | ROI% | Units |",
            "|--------|-------|-----|------|-------|-----|-----|------|-------|",
        ])

        for cfg_name in CONFIGS:
            results = all_results.get(cfg_name, [])
            if not results:
                continue
            avg_mae = np.mean([r["mae"] for r in results if r["mae"] is not None])
            avg_rmse = np.mean([r["rmse"] for r in results if r["rmse"] is not None])
            avg_sigma = np.mean([r["sigma"] for r in results])
            tot_w = sum(r["ats"][thr]["wins"] for r in results)
            tot_l = sum(r["ats"][thr]["losses"] for r in results)
            tot = tot_w + tot_l
            avg_wr = tot_w / tot * 100 if tot > 0 else 0
            avg_roi = (tot_w * 0.9091 - tot_l) / tot * 100 if tot > 0 else 0
            tot_units = tot_w * 0.9091 - tot_l
            n_feats = results[0]["n_features"]
            lines.append(
                f"| **{cfg_name}** | {n_feats} | **{avg_mae:.2f}** | **{avg_rmse:.2f}** | "
                f"{avg_sigma:.1f} | {tot_w}-{tot_l} | {avg_wr:.1f}% | "
                f"{avg_roi:+.1f}% | {tot_units:+.1f} |"
            )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

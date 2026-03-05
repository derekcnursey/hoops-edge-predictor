#!/usr/bin/env python3
"""Walk-forward backfill: train per-season models and generate site predictions.

For each holdout season, trains on all prior seasons (Torvik features),
then generates predictions and saves per-date site JSONs.

Season 2026 uses the production model (already trained on 2015-2025).

Usage:
    PYTHONUNBUFFERED=1 poetry run python scripts/walkforward_backfill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import load_season_features
from src.features import get_feature_matrix, get_targets, load_lines
from src.infer import (
    normal_cdf, american_to_breakeven, american_profit_per_1,
    prob_to_american, save_predictions,
)
from src.trainer import impute_column_means, train_regressor, train_classifier

HOLDOUT_YEARS = [2019, 2020, 2022, 2023, 2024, 2025, 2026]
TRAIN_START = 2015
ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
VAL_FRAC = 0.15

# Use production hparams
REG_HP = {"epochs": 150, "hidden1": 384, "hidden2": 256, "dropout": 0.2,
           "lr": 1e-3, "batch_size": 512}
CLS_HP = {"epochs": 150, "hidden1": 384, "dropout": 0.2,
           "lr": 1e-3, "batch_size": 512}


def load_features(season: int) -> pd.DataFrame:
    return load_season_features(
        season, no_garbage=True, adj_suffix=ADJ_SUFFIX, efficiency_source="torvik",
    )


def generate_predictions(
    regressor, classifier, scaler: StandardScaler,
    holdout_df: pd.DataFrame, holdout_season: int,
) -> pd.DataFrame:
    """Generate full prediction DataFrame matching infer.predict output."""
    X = holdout_df[config.FEATURE_ORDER].values.astype(np.float32)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = scaler.mean_[j]
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        regressor.eval()
        classifier.eval()
        mu_raw, log_sigma_raw = regressor(X_tensor)
        sigma = torch.exp(log_sigma_raw).clamp(min=0.5, max=30.0)
        logits = classifier(X_tensor)
        home_win_prob = torch.sigmoid(logits).numpy().flatten()

    mu = mu_raw.numpy().flatten()
    sigma = sigma.numpy().flatten()

    meta_cols = ["gameId", "homeTeamId", "awayTeamId"]
    if "homeTeam" in holdout_df.columns:
        meta_cols.append("homeTeam")
    if "awayTeam" in holdout_df.columns:
        meta_cols.append("awayTeam")
    meta_cols.append("startDate")
    out = holdout_df[meta_cols].copy()
    out["predicted_spread"] = mu
    out["spread_sigma"] = sigma
    out["home_win_prob"] = home_win_prob
    out["away_win_prob"] = 1.0 - home_win_prob

    # Attach lines
    lines = load_lines(holdout_season)
    if not lines.empty:
        lines_fixed = lines.copy()
        lines_fixed["spread"] = pd.to_numeric(lines_fixed["spread"], errors="coerce")
        lines_fixed["homeMoneyline"] = pd.to_numeric(
            lines_fixed["homeMoneyline"], errors="coerce")

        # Majority spread sign fix
        has_spread = lines_fixed["spread"].notna() & (lines_fixed["spread"] != 0)
        spread_sign = np.sign(lines_fixed.loc[has_spread, "spread"])
        majority_sign = (
            spread_sign.groupby(lines_fixed.loc[has_spread, "gameId"])
            .sum().rename("_majority_sign")
        )

        _provider_rank = {"Draft Kings": 0, "ESPN BET": 1, "Bovada": 2}
        lines_dedup = (
            lines_fixed.assign(
                _has_spread=lines_fixed["spread"].notna().astype(int),
                _has_total=lines_fixed["overUnder"].notna().astype(int),
                _prov_rank=lines_fixed["provider"].map(_provider_rank).fillna(99),
            )
            .sort_values(["_has_spread", "_has_total", "_prov_rank"],
                         ascending=[False, False, True])
            .drop_duplicates(subset=["gameId"], keep="first")
            .drop(columns=["_has_spread", "_has_total", "_prov_rank"])
            .copy()
        )

        lines_dedup = lines_dedup.merge(majority_sign, on="gameId", how="left")
        _sp = lines_dedup["spread"]
        _maj = lines_dedup["_majority_sign"]
        mask_majority_flip = (
            _sp.notna() & _maj.notna() & (_maj != 0)
            & (abs(_sp) >= 3) & (np.sign(_sp) != np.sign(_maj))
        )
        lines_dedup.loc[mask_majority_flip, "spread"] = -_sp[mask_majority_flip]

        _sp2 = lines_dedup["spread"]
        _ml = lines_dedup["homeMoneyline"]
        mask_ml_fix = (
            _sp2.notna() & _ml.notna()
            & (~mask_majority_flip) & _maj.isna()
            & (((_sp2 > 3) & (_ml < -150)) | ((_sp2 < -3) & (_ml > 150)))
        )
        lines_dedup.loc[mask_ml_fix, "spread"] = -_sp2[mask_ml_fix]
        lines_dedup = lines_dedup.drop(columns=["_majority_sign"])

        lines_dedup = lines_dedup.rename(columns={
            "spread": "book_spread", "overUnder": "book_total",
            "homeMoneyline": "home_moneyline", "awayMoneyline": "away_moneyline",
        })
        merge_cols = ["gameId", "book_spread", "book_total",
                      "home_moneyline", "away_moneyline"]
        available = [c for c in merge_cols if c in lines_dedup.columns]
        out = out.merge(lines_dedup[available], on="gameId", how="left")

        # Phantom edge fix
        if "book_spread" in out.columns:
            _bs = out["book_spread"]
            _ps = out["predicted_spread"]
            _phantom = _bs + _ps
            mask_phantom = (
                _bs.notna() & _ps.notna()
                & (((_bs > 0) & (_ps > 0) & (_phantom >= 9))
                   | ((_bs < 0) & (_ps < 0) & (_phantom <= -9)))
            )
            out.loc[mask_phantom, "book_spread"] = -_bs[mask_phantom]

        # Edge metrics
        if "book_spread" in out.columns:
            out["model_spread"] = -out["predicted_spread"]
            out["spread_diff"] = out["model_spread"] - out["book_spread"]
            out["edge_home_points"] = out["predicted_spread"] + out["book_spread"]

            sigma_safe = out["spread_sigma"].clip(lower=0.5)
            edge_z = out["edge_home_points"] / sigma_safe
            home_cover_prob = normal_cdf(edge_z)
            away_cover_prob = 1.0 - home_cover_prob

            out["pick_side"] = np.where(out["edge_home_points"] >= 0, "HOME", "AWAY")
            out["pick_cover_prob"] = np.where(
                out["edge_home_points"] >= 0, home_cover_prob, away_cover_prob)

            pick_spread_odds = -110
            pick_breakeven = float(american_to_breakeven(np.array([pick_spread_odds]))[0])
            pick_profit = float(american_profit_per_1(np.array([pick_spread_odds]))[0])

            out["pick_spread_odds"] = pick_spread_odds
            out["pick_prob_edge"] = out["pick_cover_prob"] - pick_breakeven
            out["pick_ev_per_1"] = (out["pick_cover_prob"] * pick_profit
                                    - (1.0 - out["pick_cover_prob"]))
            out["pick_fair_odds"] = prob_to_american(out["pick_cover_prob"].values)

    return out


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("Walk-Forward Backfill: Torvik features, per-season models")
    print(f"Holdout years: {HOLDOUT_YEARS}")
    print(f"Val frac: {VAL_FRAC} (best-loss checkpointing)")
    print("=" * 70)

    # Load all season parquets
    all_dfs: dict[int, pd.DataFrame] = {}
    all_seasons = [s for s in range(TRAIN_START, max(HOLDOUT_YEARS) + 1)
                   if s not in config.EXCLUDE_SEASONS]
    print("\nLoading features...")
    for s in all_seasons:
        try:
            all_dfs[s] = load_features(s)
            print(f"  Season {s}: {len(all_dfs[s])} games")
        except FileNotFoundError:
            print(f"  Season {s}: not found, skipping")

    total_dates = 0

    for holdout_year in HOLDOUT_YEARS:
        print(f"\n{'='*60}")
        print(f"HOLDOUT {holdout_year}")
        print(f"{'='*60}")

        holdout_df = all_dfs.get(holdout_year)
        if holdout_df is None or holdout_df.empty:
            print("  No holdout data, skipping")
            continue

        # For 2026: use the production model (already trained on 2015-2025)
        if holdout_year == 2026:
            print("  Using production model (trained on 2015-2025)")
            from src.infer import predict
            lines = load_lines(holdout_year)
            preds = predict(holdout_df, lines_df=lines)
        else:
            # Build training set from all prior seasons
            train_seasons = [s for s in all_seasons if s < holdout_year]
            train_dfs = [all_dfs[s] for s in train_seasons
                         if s in all_dfs and not all_dfs[s].empty]
            if not train_dfs:
                print("  No training data, skipping")
                continue

            train_df = pd.concat(train_dfs, ignore_index=True)
            train_df = train_df.dropna(subset=["homeScore", "awayScore"])
            train_df = train_df[(train_df["homeScore"] != 0) | (train_df["awayScore"] != 0)]
            print(f"  Train: {len(train_df)} games ({train_seasons})")
            print(f"  Holdout: {len(holdout_df)} games")

            # Train
            X_train = train_df[config.FEATURE_ORDER].values.astype(np.float32)
            targets = get_targets(train_df)
            y_spread = targets["spread_home"].values.astype(np.float32)
            y_win = targets["home_win"].values.astype(np.float32)

            X_train = impute_column_means(X_train)
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)

            regressor = train_regressor(X_train_scaled, y_spread, hparams=REG_HP,
                                        val_frac=VAL_FRAC)
            classifier = train_classifier(X_train_scaled, y_win, hparams=CLS_HP,
                                          val_frac=VAL_FRAC)

            # Generate predictions
            preds = generate_predictions(
                regressor, classifier, scaler, holdout_df, holdout_year)

        # Split by date and save
        dates = pd.to_datetime(preds["startDate"], errors="coerce", utc=True)
        preds["_date"] = dates.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        n_dates = preds["_date"].nunique()

        for game_date, group in preds.groupby("_date"):
            group = group.drop(columns=["_date"])
            save_predictions(group, game_date=game_date)

        total_dates += n_dates
        print(f"  Saved {n_dates} date files")

    # Delete prediction files for seasons before 2019 (no walk-forward model)
    import os
    data_dir = config.SITE_DATA_DIR
    removed = 0
    for f in os.listdir(data_dir):
        if not f.startswith("predictions_"):
            continue
        date_str = f.replace("predictions_", "").replace(".json", "")
        try:
            month = int(date_str[5:7])
            year = int(date_str[0:4])
            season = year + 1 if month >= 11 else year
            if season < 2019:
                os.remove(data_dir / f)
                removed += 1
        except (ValueError, IndexError):
            continue

    print(f"\nRemoved {removed} pre-2019 prediction files (no walk-forward model)")
    print(f"Total walk-forward dates saved: {total_dates}")
    print("Done!")


if __name__ == "__main__":
    main()

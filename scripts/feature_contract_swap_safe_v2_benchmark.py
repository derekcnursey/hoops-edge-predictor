#!/usr/bin/env python3
"""Research benchmark for the swap_safe_v2 feature contract.

Compares the existing canonical HGBR baseline contract against a research-only
swap-safe_v2 contract, with no training augmentation. This isolates contract
drift before any swapped-slot training changes.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import load_season_features
from src.features import build_features, get_feature_matrix, get_targets
from src.slot_augmentation import swap_feature_frame

OUTPUT_DIR = config.ARTIFACTS_DIR / "research" / "feature_contract_swap_safe_v2_benchmark_v1"
FEATURE_CACHE_DIR = config.FEATURES_DIR / "contract_research"
MIDDEC_CUTOFF = (12, 15)


def _load_canonical_module():
    script_path = PROJECT_ROOT / "scripts" / "canonical_walkforward.py"
    spec = importlib.util.spec_from_file_location("canonical_walkforward_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cw = _load_canonical_module()


def _feature_path_v2(season: int) -> Path:
    suffix = "_no_garbage_torvik_adj_a0.85_p10_swap_safe_v2_features.parquet"
    return FEATURE_CACHE_DIR / f"season_{season}{suffix}"


def _clean_games(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["homeScore", "awayScore"]).copy()
    out = out[(out["homeScore"] != 0) | (out["awayScore"] != 0)].copy()
    return out.reset_index(drop=True)


def _load_or_build_v2_features(season: int) -> pd.DataFrame:
    path = _feature_path_v2(season)
    if path.exists():
        return pd.read_parquet(path)

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = load_season_features(
            season=season,
            no_garbage=True,
            adj_suffix=cw.ADJ_SUFFIX,
            efficiency_source=cw.EFFICIENCY_SOURCE,
        ).copy()
        df = df.rename(
            columns={
                "home_opp_ft_rate": "home_def_ft_rate",
                "home_team_hca": "venue_edge",
                "home_team_efg_home_split": "home_team_efg_slot_split",
                "away_team_efg_away_split": "away_team_efg_slot_split",
            }
        )
    except FileNotFoundError:
        df = build_features(
            season=season,
            no_garbage=True,
            extra_features=config.EXTRA_FEATURES,
            adjust_ff=config.ADJUST_FF,
            adjust_prior_weight=float(config.ADJUST_PRIOR),
            adjust_alpha=float(config.ADJUST_ALPHA),
            adjust_ff_method=config.ADJUST_FF_METHOD,
            efficiency_source="torvik",
            feature_contract="swap_safe_v2",
        )
    df.to_parquet(path, index=False)
    return df


def _prepare_data(df: pd.DataFrame, feature_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = get_feature_matrix(df, feature_order=feature_order).values.astype(np.float32)
    y = get_targets(df)["spread_home"].values.astype(np.float32)
    return X, y


def _neutral_mask(df: pd.DataFrame) -> np.ndarray:
    if "neutral_site" in df.columns:
        return df["neutral_site"].fillna(0).astype(float).to_numpy() == 1.0
    return np.zeros(len(df), dtype=bool)


def _predict_hgbr_contract(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_order: list[str],
) -> np.ndarray:
    X_train_raw, y_train = _prepare_data(train_df, feature_order)
    X_test_raw, _ = _prepare_data(test_df, feature_order)
    means = cw._train_impute_means(X_train_raw)
    X_train = cw._apply_impute_means(X_train_raw, means)
    X_test = cw._apply_impute_means(X_test_raw, means)

    model = HistGradientBoostingRegressor(**cw.HGBR_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(np.float32)

    neutral_mask = _neutral_mask(test_df)
    if neutral_mask.any():
        neutral_idx = np.flatnonzero(neutral_mask)
        feature_df = get_feature_matrix(test_df.iloc[neutral_idx], feature_order=feature_order)
        swapped_feature_df = swap_feature_frame(feature_df, feature_order, neutral_only=True)
        X_swap = cw._apply_impute_means(swapped_feature_df.values.astype(np.float32), means)
        pred_swap = model.predict(X_swap).astype(np.float32)
        pred[neutral_idx] = (pred[neutral_idx] - pred_swap) / 2.0

    return pred


def _season_slice(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["startDate"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    month = dt.dt.month
    day = dt.dt.day
    cutoff_mask = (month > MIDDEC_CUTOFF[0]) | ((month == MIDDEC_CUTOFF[0]) & (day >= MIDDEC_CUTOFF[1]))

    if slice_name == "full":
        return out
    if slice_name == "pre_dec15":
        return out.loc[~cutoff_mask].copy()
    if slice_name == "dec15_plus":
        return out.loc[cutoff_mask].copy()
    raise ValueError(f"Unknown slice_name: {slice_name}")


def _run_contract_benchmark(label: str, feature_order: list[str], load_season_fn) -> pd.DataFrame:
    rows = []

    for fold in cw._folds():
        holdout = fold["holdout_season"]
        train_seasons = fold["train_seasons"]

        train_df = _clean_games(pd.concat([load_season_fn(s) for s in train_seasons], ignore_index=True))
        holdout_df = _clean_games(load_season_fn(holdout))
        holdout_df = cw._attach_book_spread(holdout_df, holdout)

        pred = _predict_hgbr_contract(train_df, holdout_df, feature_order)
        out = holdout_df[
            [
                "gameId",
                "startDate",
                "homeTeamId",
                "awayTeamId",
                "homeTeam",
                "awayTeam",
                "book_spread",
            ]
        ].copy()
        out["actual_margin"] = (
            holdout_df["homeScore"].astype(float).to_numpy()
            - holdout_df["awayScore"].astype(float).to_numpy()
        )
        out["pred_margin"] = pred
        out["pred_home_win"] = (pred > 0).astype(int)
        out["abs_error"] = np.abs(out["actual_margin"] - out["pred_margin"])
        out["holdout_season"] = holdout
        out["model"] = label
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def _summarize(label: str, df: pd.DataFrame) -> dict[str, object]:
    metrics = cw._metrics_from_predictions(df)
    metrics["model"] = label
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _run_contract_benchmark(
        "baseline_current_contract",
        list(config.FEATURE_ORDER),
        lambda season: load_season_features(
            season,
            no_garbage=True,
            adj_suffix=cw.ADJ_SUFFIX,
            efficiency_source=cw.EFFICIENCY_SOURCE,
        ),
    )
    v2 = _run_contract_benchmark(
        "swap_safe_v2_contract",
        list(config.FEATURE_ORDER_SWAP_SAFE_V2),
        _load_or_build_v2_features,
    )

    pooled_rows = [
        _summarize("baseline_current_contract", baseline),
        _summarize("swap_safe_v2_contract", v2),
    ]
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(OUTPUT_DIR / "pooled_metrics.csv", index=False)

    slice_rows: list[dict[str, object]] = []
    for label, df in [
        ("baseline_current_contract", baseline),
        ("swap_safe_v2_contract", v2),
    ]:
        for slice_name in ["full", "pre_dec15", "dec15_plus"]:
            sliced = _season_slice(df, slice_name)
            row = _summarize(label, sliced)
            row["slice"] = slice_name
            slice_rows.append(row)
    slice_df = pd.DataFrame(slice_rows)
    slice_df.to_csv(OUTPUT_DIR / "pooled_slice_metrics.csv", index=False)

    season_rows: list[dict[str, object]] = []
    for label, df in [
        ("baseline_current_contract", baseline),
        ("swap_safe_v2_contract", v2),
    ]:
        for season, season_df in df.groupby("holdout_season"):
            row = _summarize(label, season_df)
            row["holdout_season"] = season
            season_rows.append(row)
    season_df = pd.DataFrame(season_rows)
    season_df.to_csv(OUTPUT_DIR / "season_metrics.csv", index=False)

    protocol = {
        "holdout_seasons": cw.HOLDOUT_SEASONS,
        "efficiency_source": "torvik",
        "no_garbage": True,
        "feature_contracts": {
            "baseline_current_contract": "current",
            "swap_safe_v2_contract": "swap_safe_v2",
        },
        "middec_cutoff": f"{MIDDEC_CUTOFF[0]:02d}-{MIDDEC_CUTOFF[1]:02d}",
        "repaired_lines": True,
        "model": "HistGradientBoosting",
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2))

    delta_all = float(
        pooled_df.loc[pooled_df["model"] == "swap_safe_v2_contract", "MAE_all"].iloc[0]
        - pooled_df.loc[pooled_df["model"] == "baseline_current_contract", "MAE_all"].iloc[0]
    )
    delta_lined = float(
        pooled_df.loc[pooled_df["model"] == "swap_safe_v2_contract", "MAE_lined"].iloc[0]
        - pooled_df.loc[pooled_df["model"] == "baseline_current_contract", "MAE_lined"].iloc[0]
    )
    summary = "\n".join(
        [
            "# swap_safe_v2 Contract Benchmark",
            "",
            f"- Pooled MAE_all drift: `{delta_all:+.6f}`",
            f"- Pooled MAE_lined drift: `{delta_lined:+.6f}`",
            "",
            "## Pooled",
            "",
            pooled_df.to_csv(index=False),
            "",
            "## Dec15+ Slice",
            "",
            slice_df.loc[slice_df["slice"] == "dec15_plus"].to_csv(index=False),
        ]
    )
    (OUTPUT_DIR / "summary.md").write_text(summary)


if __name__ == "__main__":
    main()

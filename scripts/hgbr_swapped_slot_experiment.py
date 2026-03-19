#!/usr/bin/env python3
"""Research experiment: swapped-slot augmentation for HGBR mean model.

This experiment keeps the canonical repaired-line benchmark setup fixed and
compares:
  - baseline HGBR training
  - HGBR with swapped-slot augmentation on neutral-site training rows only

Non-neutral swapped augmentation is intentionally not used because the current
53-feature contract contains an unswappable venue-specific feature
(`home_team_hca`) with no mirrored away-team counterpart.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "canonical_walkforward",
    PROJECT_ROOT / "scripts" / "canonical_walkforward.py",
)
cw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cw)

from src import config
from src.slot_augmentation import (
    audit_feature_order,
    augment_swapped_slot_training,
    swap_feature_frame,
)


MODEL_BASELINE = "HistGradientBoosting"
MODEL_AUGMENTED = "HistGradientBoostingSwappedSlots"
MIDDEC_CUTOFF = (12, 15)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HGBR swapped-slot augmentation experiment.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS_DIR / "research" / "hgbr_swapped_slot_experiment_v1",
        help="Artifact output directory.",
    )
    return parser.parse_args()


def _neutral_mask(df: pd.DataFrame) -> np.ndarray:
    if "neutral_site" in df.columns:
        return df["neutral_site"].fillna(0).astype(float).to_numpy() == 1.0
    if "neutralSite" in df.columns:
        return df["neutralSite"].fillna(0).astype(float).to_numpy() == 1.0
    return np.zeros(len(df), dtype=bool)


def _middec_mask(df: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(df["startDate"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    return (dates.dt.month > MIDDEC_CUTOFF[0]) | (
        (dates.dt.month == MIDDEC_CUTOFF[0]) & (dates.dt.day >= MIDDEC_CUTOFF[1])
    )


def _pre_middec_mask(df: pd.DataFrame) -> np.ndarray:
    return ~_middec_mask(df)


def _train_hgbr_model(
    train_df: pd.DataFrame,
    *,
    augment_swapped_slots: bool = False,
) -> tuple[HistGradientBoostingRegressor, np.ndarray]:
    feature_df = cw.get_feature_matrix(train_df).copy()
    spread = cw.get_targets(train_df)["spread_home"].values.astype(np.float32)
    if augment_swapped_slots:
        eligible_mask = _neutral_mask(train_df)
        feature_df, spread, _ = augment_swapped_slot_training(
            feature_df,
            spread,
            eligible_mask=eligible_mask,
        )
    X_raw = feature_df.values.astype(np.float32)
    means = cw._train_impute_means(X_raw)
    X = cw._apply_impute_means(X_raw, means)
    model = HistGradientBoostingRegressor(**cw.HGBR_PARAMS)
    model.fit(X, spread)
    return model, means


def _predict_raw_margin(
    model: HistGradientBoostingRegressor,
    means: np.ndarray,
    df: pd.DataFrame,
) -> np.ndarray:
    feature_df = cw.get_feature_matrix(df).copy()
    X_raw = feature_df.values.astype(np.float32)
    X = cw._apply_impute_means(X_raw, means)
    return model.predict(X).astype(np.float32)


def _predict_swapped_from_feature_df(
    model: HistGradientBoostingRegressor,
    means: np.ndarray,
    feature_df: pd.DataFrame,
) -> np.ndarray:
    swapped_df = swap_feature_frame(feature_df, list(feature_df.columns), neutral_only=True)
    X_raw = swapped_df.values.astype(np.float32)
    X = cw._apply_impute_means(X_raw, means)
    return model.predict(X).astype(np.float32)


def _predict_symmetrized_margin(
    model: HistGradientBoostingRegressor,
    means: np.ndarray,
    df: pd.DataFrame,
) -> np.ndarray:
    pred = _predict_raw_margin(model, means, df)
    neutral_mask = _neutral_mask(df)
    if neutral_mask.any():
        neutral_idx = np.flatnonzero(neutral_mask)
        feature_df = cw.get_feature_matrix(df.iloc[neutral_idx]).copy()
        pred_swap = _predict_swapped_from_feature_df(model, means, feature_df)
        pred[neutral_idx] = (pred[neutral_idx] - pred_swap) / 2.0
    return pred.astype(np.float32)


def _prediction_frame(
    model_name: str,
    holdout_season: int,
    holdout_df: pd.DataFrame,
    pred_margin: np.ndarray,
) -> pd.DataFrame:
    return cw._build_prediction_frame(model_name, holdout_season, holdout_df, pred_margin)


def _slot_bias_frame(
    holdout_season: int,
    holdout_df: pd.DataFrame,
    raw_pred: np.ndarray,
    raw_pred_swap: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    neutral_mask = _neutral_mask(holdout_df)
    neutral_df = holdout_df.loc[neutral_mask].copy().reset_index(drop=True)
    neutral_df["pred_orig"] = raw_pred.astype(float)
    neutral_df["pred_swap"] = raw_pred_swap.astype(float)
    neutral_df["slot_bias"] = neutral_df["pred_orig"] + neutral_df["pred_swap"]
    neutral_df["abs_slot_bias"] = neutral_df["slot_bias"].abs()
    neutral_df["neutral_strength"] = ((neutral_df["pred_orig"] - neutral_df["pred_swap"]) / 2.0).abs()
    neutral_df["holdout_season"] = holdout_season
    neutral_df["model"] = model_name
    return neutral_df[
        ["model", "holdout_season", "gameId", "startDate", "homeTeam", "awayTeam", "pred_orig", "pred_swap", "slot_bias", "abs_slot_bias", "neutral_strength"]
    ].copy()


def _slot_bias_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "n_games": 0,
            "mean_slot_bias": np.nan,
            "mean_abs_slot_bias": np.nan,
            "p95_abs_slot_bias": np.nan,
            "home_slot_adv_share": np.nan,
        }
    slot = df["slot_bias"].to_numpy(dtype=float)
    abs_slot = np.abs(slot)
    return {
        "n_games": int(len(df)),
        "mean_slot_bias": float(np.mean(slot)),
        "mean_abs_slot_bias": float(np.mean(abs_slot)),
        "p95_abs_slot_bias": float(np.quantile(abs_slot, 0.95)),
        "home_slot_adv_share": float(np.mean(slot > 0)),
    }


def _strength_bucket(value: float) -> str:
    if value < 5:
        return "0-5"
    if value < 10:
        return "5-10"
    if value < 15:
        return "10-15"
    return "15+"


def _csv_block(df: pd.DataFrame) -> str:
    return df.to_csv(index=False).strip()


def _metrics_row(model_name: str, slice_name: str, df: pd.DataFrame) -> dict[str, float | str]:
    metrics = cw._metrics_from_predictions(df, probabilistic=False, lined_only=False)
    metrics["model"] = model_name
    metrics["slice"] = slice_name
    return metrics


def run_experiment(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_feature_order(config.FEATURE_ORDER)
    protocol = {
        "experiment": "hgbr_swapped_slot_experiment_v1",
        "efficiency_source": cw.EFFICIENCY_SOURCE,
        "adj_suffix": cw.ADJ_SUFFIX,
        "no_garbage": cw.NO_GARBAGE,
        "holdout_seasons": cw.HOLDOUT_SEASONS,
        "exclude_seasons": cw.EXCLUDE_SEASONS,
        "augmentation": {
            "mode": "neutral_training_rows_only",
            "reason_non_neutral_not_augmented": "home_team_hca has no mirrored away-team counterpart in the saved feature contract",
            "explicit_pairs": audit.explicit_pairs,
            "generic_pairs": audit.generic_pairs,
            "negated": audit.negated,
            "globals": audit.globals,
            "unswappable": audit.unswappable,
            "unclassified": audit.unclassified,
        },
        "middec_cutoff": f"{MIDDEC_CUTOFF[0]:02d}-{MIDDEC_CUTOFF[1]:02d}",
        "lines_table": config.RESEARCH_LINES_TABLE,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    all_predictions: dict[str, list[pd.DataFrame]] = {MODEL_BASELINE: [], MODEL_AUGMENTED: []}
    all_bias_frames: dict[str, list[pd.DataFrame]] = {MODEL_BASELINE: [], MODEL_AUGMENTED: []}
    fold_rows: list[dict] = []
    slice_rows: list[dict] = []

    for holdout_season in cw.HOLDOUT_SEASONS:
        train_seasons = [season for season in range(cw.TRAIN_START, holdout_season) if season not in cw.EXCLUDE_SEASONS]
        print(f"\n=== Holdout {holdout_season} ===")
        print(f"Train seasons: {train_seasons}")
        train_df = cw._load_fold_frame(train_seasons)
        holdout_df = cw._attach_book_spread(cw._load_fold_frame([holdout_season]), holdout_season)

        for model_name, augment in [
            (MODEL_BASELINE, False),
            (MODEL_AUGMENTED, True),
        ]:
            model, means = _train_hgbr_model(train_df, augment_swapped_slots=augment)
            pred_sym = _predict_symmetrized_margin(model, means, holdout_df)
            pred_frame = _prediction_frame(model_name, holdout_season, holdout_df, pred_sym)
            all_predictions[model_name].append(pred_frame)

            neutral_mask = _neutral_mask(holdout_df)
            if neutral_mask.any():
                raw_df = holdout_df.loc[neutral_mask].copy()
                raw_pred = _predict_raw_margin(model, means, raw_df)
                raw_pred_swap = _predict_swapped_from_feature_df(model, means, cw.get_feature_matrix(raw_df).copy())
                bias_df = _slot_bias_frame(holdout_season, raw_df, raw_pred, raw_pred_swap, model_name)
            else:
                bias_df = pd.DataFrame(columns=["model", "holdout_season", "gameId", "startDate", "homeTeam", "awayTeam", "pred_orig", "pred_swap", "slot_bias", "abs_slot_bias", "neutral_strength"])
            all_bias_frames[model_name].append(bias_df)

            metrics = cw._metrics_from_predictions(pred_frame, probabilistic=False, lined_only=False)
            metrics["model"] = model_name
            metrics["holdout_season"] = holdout_season
            fold_rows.append(metrics)

            for slice_name, mask_func in [
                ("full", lambda df: np.ones(len(df), dtype=bool)),
                ("pre_dec15", _pre_middec_mask),
                ("dec15_plus", _middec_mask),
                ("neutral_only", _neutral_mask),
            ]:
                mask = np.asarray(mask_func(pred_frame), dtype=bool)
                if mask.sum() == 0:
                    continue
                slice_df = pred_frame.loc[mask].copy()
                row = _metrics_row(model_name, slice_name, slice_df)
                row["holdout_season"] = holdout_season
                slice_rows.append(row)

            print(
                f"  {model_name:>30}: "
                f"MAE_all={metrics['MAE_all']:.4f} "
                f"MAE_lined={metrics['MAE_lined']:.4f}"
            )

    fold_metrics = pd.DataFrame(fold_rows).sort_values(["holdout_season", "model"]).reset_index(drop=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    slice_metrics = pd.DataFrame(slice_rows).sort_values(["slice", "holdout_season", "model"]).reset_index(drop=True)
    slice_metrics.to_csv(output_dir / "slice_metrics.csv", index=False)

    pooled_rows = []
    for model_name, frames in all_predictions.items():
        pooled_df = pd.concat(frames, ignore_index=True)
        metrics = cw._metrics_from_predictions(pooled_df, probabilistic=False, lined_only=False)
        metrics["model"] = model_name
        pooled_rows.append(metrics)
        pred_dir = output_dir / "predictions" / model_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        pooled_df.to_parquet(pred_dir / "all_holdouts.parquet", index=False)
    pooled_metrics = pd.DataFrame(pooled_rows)[
        ["model", "n_games", "n_lined", "MAE_all", "RMSE_all", "MedAE_all", "WinAcc_all", "MAE_lined", "BookMAE_lined", "DeltaVsBook_MAE"]
    ]
    pooled_metrics.to_csv(output_dir / "pooled_metrics.csv", index=False)

    pooled_slice_rows = []
    for model_name, frames in all_predictions.items():
        pooled_df = pd.concat(frames, ignore_index=True)
        for slice_name, mask_func in [
            ("full", lambda df: np.ones(len(df), dtype=bool)),
            ("pre_dec15", _pre_middec_mask),
            ("dec15_plus", _middec_mask),
            ("neutral_only", _neutral_mask),
        ]:
            mask = np.asarray(mask_func(pooled_df), dtype=bool)
            if mask.sum() == 0:
                continue
            row = _metrics_row(model_name, slice_name, pooled_df.loc[mask].copy())
            pooled_slice_rows.append(row)
    pooled_slice_metrics = pd.DataFrame(pooled_slice_rows).sort_values(["slice", "model"]).reset_index(drop=True)
    pooled_slice_metrics.to_csv(output_dir / "pooled_slice_metrics.csv", index=False)

    bias_frames = []
    for model_name, frames in all_bias_frames.items():
        model_bias = pd.concat(frames, ignore_index=True)
        if model_bias.empty:
            continue
        model_bias["strength_bucket"] = model_bias["neutral_strength"].map(_strength_bucket)
        bias_frames.append(model_bias)
    all_bias = pd.concat(bias_frames, ignore_index=True)
    all_bias.to_parquet(output_dir / "neutral_slot_bias_games.parquet", index=False)

    bias_season_rows = []
    for (model_name, season), group in all_bias.groupby(["model", "holdout_season"]):
        row = _slot_bias_metrics(group)
        row["model"] = model_name
        row["holdout_season"] = season
        bias_season_rows.append(row)
    bias_season = pd.DataFrame(bias_season_rows).sort_values(["holdout_season", "model"]).reset_index(drop=True)
    bias_season.to_csv(output_dir / "neutral_slot_bias_by_season.csv", index=False)

    bias_bucket_rows = []
    for (model_name, bucket), group in all_bias.groupby(["model", "strength_bucket"]):
        row = _slot_bias_metrics(group)
        row["model"] = model_name
        row["strength_bucket"] = bucket
        bias_bucket_rows.append(row)
    bias_bucket = pd.DataFrame(bias_bucket_rows).sort_values(["model", "strength_bucket"]).reset_index(drop=True)
    bias_bucket.to_csv(output_dir / "neutral_slot_bias_by_strength_bucket.csv", index=False)

    bias_pooled_rows = []
    for model_name, group in all_bias.groupby("model"):
        row = _slot_bias_metrics(group)
        row["model"] = model_name
        bias_pooled_rows.append(row)
    bias_pooled = pd.DataFrame(bias_pooled_rows).sort_values("model").reset_index(drop=True)
    bias_pooled.to_csv(output_dir / "neutral_slot_bias_pooled.csv", index=False)

    lines = [
        "# HGBR Swapped-Slot Augmentation Experiment",
        "",
        "## Pooled Metrics",
        "",
        "```csv",
        _csv_block(pooled_metrics),
        "```",
        "",
        "## Pooled Slice Metrics",
        "",
        "```csv",
        _csv_block(pooled_slice_metrics),
        "```",
        "",
        "## Neutral Slot Bias (Pooled)",
        "",
        "```csv",
        _csv_block(bias_pooled),
        "```",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nSaved experiment artifacts to {output_dir}")


def main() -> None:
    args = _parse_args()
    run_experiment(args.output_dir)


if __name__ == "__main__":
    main()

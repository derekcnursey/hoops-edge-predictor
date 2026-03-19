#!/usr/bin/env python3
"""Research experiment: neutral-site-only swapped-slot augmentation on swap_safe_v2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import load_season_features
from src.features import get_feature_matrix, get_targets
from src.slot_augmentation import audit_feature_order, augment_swapped_slot_training, swap_feature_frame


spec = importlib.util.spec_from_file_location(
    "canonical_walkforward",
    PROJECT_ROOT / "scripts" / "canonical_walkforward.py",
)
cw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cw)


MODEL_BASELINE_CURRENT = "HistGradientBoosting"
MODEL_BASELINE_V2 = "HistGradientBoostingSwapSafeV2"
MODEL_AUGMENTED_V2_NEUTRAL = "HistGradientBoostingSwapSafeV2NeutralAugmented"
MIDDEC_CUTOFF = (12, 15)
FEATURE_CACHE_DIR = config.FEATURES_DIR / "contract_research"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neutral-only swapped-slot augmentation on swap_safe_v2.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS_DIR / "research" / "hgbr_swapped_slot_v2_neutral_experiment_v1",
        help="Artifact output directory.",
    )
    return parser.parse_args()


def _clean_games(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["homeScore", "awayScore"]).copy()
    out = out[(out["homeScore"] != 0) | (out["awayScore"] != 0)].copy()
    return out.reset_index(drop=True)


def _feature_path_v2(season: int) -> Path:
    suffix = "_no_garbage_torvik_adj_a0.85_p10_swap_safe_v2_features.parquet"
    return FEATURE_CACHE_DIR / f"season_{season}{suffix}"


def _load_current_features(season: int) -> pd.DataFrame:
    return load_season_features(
        season=season,
        no_garbage=True,
        adj_suffix=cw.ADJ_SUFFIX,
        efficiency_source=cw.EFFICIENCY_SOURCE,
    ).copy()


def _load_or_build_v2_features(season: int) -> pd.DataFrame:
    path = _feature_path_v2(season)
    if path.exists():
        return pd.read_parquet(path)

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_current_features(season).rename(
        columns={
            "home_opp_ft_rate": "home_def_ft_rate",
            "home_team_hca": "venue_edge",
            "home_team_efg_home_split": "home_team_efg_slot_split",
            "away_team_efg_away_split": "away_team_efg_slot_split",
        }
    )
    df.to_parquet(path, index=False)
    return df


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


def _train_model(
    train_df: pd.DataFrame,
    feature_order: list[str],
    *,
    augment_neutral_only: bool,
) -> tuple[HistGradientBoostingRegressor, np.ndarray]:
    feature_df = get_feature_matrix(train_df, feature_order=feature_order).copy()
    spread = get_targets(train_df)["spread_home"].values.astype(np.float32)
    if augment_neutral_only:
        eligible_mask = _neutral_mask(train_df)
        feature_df, spread, _ = augment_swapped_slot_training(
            feature_df,
            spread,
            eligible_mask=eligible_mask,
            neutral_only=True,
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
    feature_order: list[str],
) -> np.ndarray:
    feature_df = get_feature_matrix(df, feature_order=feature_order)
    X_raw = feature_df.values.astype(np.float32)
    X = cw._apply_impute_means(X_raw, means)
    return model.predict(X).astype(np.float32)


def _predict_swapped_from_feature_df(
    model: HistGradientBoostingRegressor,
    means: np.ndarray,
    feature_df: pd.DataFrame,
    feature_order: list[str],
) -> np.ndarray:
    swapped_df = swap_feature_frame(feature_df, feature_order, neutral_only=True)
    X_raw = swapped_df.values.astype(np.float32)
    X = cw._apply_impute_means(X_raw, means)
    return model.predict(X).astype(np.float32)


def _predict_symmetrized_margin(
    model: HistGradientBoostingRegressor,
    means: np.ndarray,
    df: pd.DataFrame,
    feature_order: list[str],
) -> np.ndarray:
    pred = _predict_raw_margin(model, means, df, feature_order)
    neutral_mask = _neutral_mask(df)
    if neutral_mask.any():
        neutral_idx = np.flatnonzero(neutral_mask)
        feature_df = get_feature_matrix(df.iloc[neutral_idx], feature_order=feature_order).copy()
        pred_swap = _predict_swapped_from_feature_df(model, means, feature_df, feature_order)
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
        [
            "model",
            "holdout_season",
            "gameId",
            "startDate",
            "homeTeam",
            "awayTeam",
            "pred_orig",
            "pred_swap",
            "slot_bias",
            "abs_slot_bias",
            "neutral_strength",
        ]
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


def _metrics_row(model_name: str, slice_name: str, df: pd.DataFrame) -> dict[str, float | str]:
    metrics = cw._metrics_from_predictions(df, probabilistic=False, lined_only=False)
    metrics["model"] = model_name
    metrics["slice"] = slice_name
    return metrics


def run_experiment(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "experiment": "hgbr_swapped_slot_v2_neutral_experiment_v1",
        "holdout_seasons": cw.HOLDOUT_SEASONS,
        "exclude_seasons": cw.EXCLUDE_SEASONS,
        "efficiency_source": cw.EFFICIENCY_SOURCE,
        "adj_suffix": cw.ADJ_SUFFIX,
        "no_garbage": cw.NO_GARBAGE,
        "middec_cutoff": f"{MIDDEC_CUTOFF[0]:02d}-{MIDDEC_CUTOFF[1]:02d}",
        "lines_table": config.RESEARCH_LINES_TABLE,
        "feature_contracts": {
            MODEL_BASELINE_CURRENT: "current",
            MODEL_BASELINE_V2: "swap_safe_v2",
            MODEL_AUGMENTED_V2_NEUTRAL: "swap_safe_v2 + neutral-site-only swapped-slot augmentation",
        },
        "swap_safe_v2_audit": audit_feature_order(config.FEATURE_ORDER_SWAP_SAFE_V2).__dict__,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    model_specs = [
        (MODEL_BASELINE_CURRENT, list(config.FEATURE_ORDER), _load_current_features, False),
        (MODEL_BASELINE_V2, list(config.FEATURE_ORDER_SWAP_SAFE_V2), _load_or_build_v2_features, False),
        (MODEL_AUGMENTED_V2_NEUTRAL, list(config.FEATURE_ORDER_SWAP_SAFE_V2), _load_or_build_v2_features, True),
    ]

    all_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name, *_ in model_specs}
    all_bias_frames: dict[str, list[pd.DataFrame]] = {name: [] for name, *_ in model_specs}
    fold_rows: list[dict] = []
    slice_rows: list[dict] = []

    for holdout_season in cw.HOLDOUT_SEASONS:
        train_seasons = [season for season in range(cw.TRAIN_START, holdout_season) if season not in cw.EXCLUDE_SEASONS]
        print(f"\n=== Holdout {holdout_season} ===")
        print(f"Train seasons: {train_seasons}")

        holdout_current = cw._attach_book_spread(_clean_games(_load_current_features(holdout_season)), holdout_season)
        holdout_v2 = cw._attach_book_spread(_clean_games(_load_or_build_v2_features(holdout_season)), holdout_season)

        train_current = _clean_games(pd.concat([_load_current_features(s) for s in train_seasons], ignore_index=True))
        train_v2 = _clean_games(pd.concat([_load_or_build_v2_features(s) for s in train_seasons], ignore_index=True))

        for model_name, feature_order, _, augment in model_specs:
            if feature_order == list(config.FEATURE_ORDER):
                train_df = train_current
                holdout_df = holdout_current
            else:
                train_df = train_v2
                holdout_df = holdout_v2

            model, means = _train_model(train_df, feature_order, augment_neutral_only=augment)
            pred_sym = _predict_symmetrized_margin(model, means, holdout_df, feature_order)
            pred_frame = _prediction_frame(model_name, holdout_season, holdout_df, pred_sym)
            all_predictions[model_name].append(pred_frame)

            neutral_mask = _neutral_mask(holdout_df)
            if neutral_mask.any():
                raw_df = holdout_df.loc[neutral_mask].copy()
                raw_pred = _predict_raw_margin(model, means, raw_df, feature_order)
                raw_pred_swap = _predict_swapped_from_feature_df(
                    model,
                    means,
                    get_feature_matrix(raw_df, feature_order=feature_order).copy(),
                    feature_order,
                )
                bias_df = _slot_bias_frame(holdout_season, raw_df, raw_pred, raw_pred_swap, model_name)
            else:
                bias_df = pd.DataFrame(
                    columns=[
                        "model",
                        "holdout_season",
                        "gameId",
                        "startDate",
                        "homeTeam",
                        "awayTeam",
                        "pred_orig",
                        "pred_swap",
                        "slot_bias",
                        "abs_slot_bias",
                        "neutral_strength",
                    ]
                )
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
                row = _metrics_row(model_name, slice_name, pred_frame.loc[mask].copy())
                row["holdout_season"] = holdout_season
                slice_rows.append(row)

            print(
                f"  {model_name:>43}: "
                f"MAE_all={metrics['MAE_all']:.4f} "
                f"MAE_lined={metrics['MAE_lined']:.4f}"
            )

    fold_metrics = pd.DataFrame(fold_rows).sort_values(["holdout_season", "model"]).reset_index(drop=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    slice_metrics = pd.DataFrame(slice_rows).sort_values(["slice", "holdout_season", "model"]).reset_index(drop=True)
    slice_metrics.to_csv(output_dir / "slice_metrics.csv", index=False)

    pooled_rows = []
    pooled_slice_rows = []
    for model_name, frames in all_predictions.items():
        pooled_df = pd.concat(frames, ignore_index=True)
        metrics = cw._metrics_from_predictions(pooled_df, probabilistic=False, lined_only=False)
        metrics["model"] = model_name
        pooled_rows.append(metrics)

        pred_dir = output_dir / "predictions" / model_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        pooled_df.to_parquet(pred_dir / "all_holdouts.parquet", index=False)

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

    pooled_metrics = pd.DataFrame(pooled_rows)[
        [
            "model",
            "n_games",
            "n_lined",
            "MAE_all",
            "RMSE_all",
            "MedAE_all",
            "WinAcc_all",
            "MAE_lined",
            "BookMAE_lined",
            "DeltaVsBook_MAE",
        ]
    ]
    pooled_metrics.to_csv(output_dir / "pooled_metrics.csv", index=False)

    pooled_slice_metrics = pd.DataFrame(pooled_slice_rows).sort_values(["slice", "model"]).reset_index(drop=True)
    pooled_slice_metrics.to_csv(output_dir / "pooled_slice_metrics.csv", index=False)

    bias_all = pd.concat([df for frames in all_bias_frames.values() for df in frames], ignore_index=True)
    bias_all.to_parquet(output_dir / "neutral_slot_bias_games.parquet", index=False)

    pooled_bias_rows = []
    season_bias_rows = []
    bucket_bias_rows = []
    for model_name, frames in all_bias_frames.items():
        model_df = pd.concat(frames, ignore_index=True)
        pooled = _slot_bias_metrics(model_df)
        pooled["model"] = model_name
        pooled_bias_rows.append(pooled)

        for season, season_df in model_df.groupby("holdout_season"):
            row = _slot_bias_metrics(season_df)
            row["model"] = model_name
            row["holdout_season"] = season
            season_bias_rows.append(row)

        if not model_df.empty:
            bucketed = model_df.copy()
            bucketed["strength_bucket"] = bucketed["neutral_strength"].map(_strength_bucket)
            for bucket, bucket_df in bucketed.groupby("strength_bucket"):
                row = _slot_bias_metrics(bucket_df)
                row["model"] = model_name
                row["strength_bucket"] = bucket
                bucket_bias_rows.append(row)

    pd.DataFrame(pooled_bias_rows).sort_values("model").to_csv(output_dir / "neutral_slot_bias_pooled.csv", index=False)
    pd.DataFrame(season_bias_rows).sort_values(["holdout_season", "model"]).to_csv(
        output_dir / "neutral_slot_bias_by_season.csv", index=False
    )
    pd.DataFrame(bucket_bias_rows).sort_values(["strength_bucket", "model"]).to_csv(
        output_dir / "neutral_slot_bias_by_strength_bucket.csv", index=False
    )

    summary = "\n".join(
        [
            "# HGBR swap_safe_v2 neutral-only swapped-slot experiment",
            "",
            "## Pooled metrics",
            "",
            pooled_metrics.to_csv(index=False).strip(),
            "",
            "## Pooled Dec15+ slice",
            "",
            pooled_slice_metrics.loc[pooled_slice_metrics["slice"] == "dec15_plus"].to_csv(index=False).strip(),
            "",
            "## Neutral slot bias pooled",
            "",
            pd.DataFrame(pooled_bias_rows).sort_values("model").to_csv(index=False).strip(),
        ]
    )
    (output_dir / "summary.md").write_text(summary + "\n")


def main() -> None:
    args = _parse_args()
    run_experiment(args.output_dir)


if __name__ == "__main__":
    main()

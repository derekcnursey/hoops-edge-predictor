#!/usr/bin/env python3
"""Research-only objective comparison for favorite-tail compression.

Compares the current trusted blended mean path against less tail-compressive
objectives using the same repaired-line walk-forward framework.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.efficiency_blend import gold_weight_for_start_dates
from src.features import get_feature_matrix
import scripts.canonical_walkforward as cw

SEED = 42
TRAIN_START = 2015
HOLDOUT_SEASONS = [2019, 2020, 2022, 2023, 2024, 2025, 2026]
EXCLUDE_SEASONS = list(config.EXCLUDE_SEASONS)
PRIMARY_SLICE_LABEL = "dec15_plus"

BASE_BENCHMARK_TORVIK = ROOT / "artifacts" / "benchmarks" / "canonical_walkforward_v2_lgb_repaired_lines_neutralfix"
BASE_BENCHMARK_TORVIK_2026 = ROOT / "artifacts" / "benchmarks" / "canonical_walkforward_v2_lgb_repaired_lines_2026_neutralfix"
BASE_BENCHMARK_GOLD = ROOT / "artifacts" / "benchmarks" / "canonical_walkforward_priorreg_k5_repaired_lines_neutralfix"
BASE_BENCHMARK_GOLD_2026 = ROOT / "artifacts" / "benchmarks" / "canonical_walkforward_priorreg_k5_repaired_lines_2026_neutralfix"


@dataclass(frozen=True)
class SourceConfig:
    label: str
    efficiency_source: str
    gold_table_name: str | None = None


@dataclass(frozen=True)
class Variant:
    label: str
    family: str
    params: dict[str, object] | None = None


TORVIK = SourceConfig("torvik", "torvik", None)
GOLD = SourceConfig(
    "gold_priorreg_k5_v1",
    "gold",
    "team_adjusted_efficiencies_no_garbage_priorreg_k5_v1",
)

BASELINE_VARIANT = Variant("HistGradientBoostingAbsoluteBlend", "existing")
HGBR_SQUARED = Variant(
    "HistGradientBoostingSquaredBlend",
    "hgbr",
    {
        **config.HGBR_PARAMS,
        "loss": "squared_error",
    },
)
LGBM_L2 = Variant(
    "LightGBMRegressionL2Blend",
    "lgbm",
    {
        **cw.LGBM_PARAMS,
        "objective": "regression",
    },
)
VARIANTS = [BASELINE_VARIANT, HGBR_SQUARED, LGBM_L2]


def _default_output_dir() -> Path:
    return ROOT / "artifacts" / "research" / "objective_tail_compression_experiment_v1"


def _training_seasons(holdout: int) -> list[int]:
    return [s for s in range(TRAIN_START, holdout) if s not in EXCLUDE_SEASONS]


def _is_dec15_plus(start_date: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(start_date, errors="coerce", utc=True).dt.tz_convert("America/New_York")
    mmdd = dt.dt.strftime("%m-%d")
    return ((dt.dt.month >= 1) & (dt.dt.month <= 3)) | (mmdd >= "12-15")


@contextmanager
def _cw_source(source: SourceConfig):
    old_eff = cw.EFFICIENCY_SOURCE
    old_gold = cw.GOLD_TABLE_NAME
    try:
        cw.EFFICIENCY_SOURCE = source.efficiency_source
        cw.GOLD_TABLE_NAME = source.gold_table_name
        yield
    finally:
        cw.EFFICIENCY_SOURCE = old_eff
        cw.GOLD_TABLE_NAME = old_gold


def _load_fold_frame(source: SourceConfig, seasons: list[int]) -> pd.DataFrame:
    with _cw_source(source):
        return cw._load_fold_frame(seasons)


def _prepare_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = get_feature_matrix(df).values.astype(np.float32)
    y = (df["homeScore"].to_numpy(dtype=np.float32) - df["awayScore"].to_numpy(dtype=np.float32))
    return X, y


def _train_impute_means(X: np.ndarray) -> np.ndarray:
    means = np.nanmean(X, axis=0)
    means = np.where(np.isnan(means), 0.0, means).astype(np.float32)
    return means


def _apply_means(X: np.ndarray, means: np.ndarray) -> np.ndarray:
    out = X.copy()
    nan_mask = np.isnan(out)
    if nan_mask.any():
        out[nan_mask] = means[np.where(nan_mask)[1]]
    return out


def _predict_hgbr(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict[str, object]) -> np.ndarray:
    X_train_raw, y_train = _prepare_xy(train_df)
    X_test_raw, _ = _prepare_xy(test_df)
    means = _train_impute_means(X_train_raw)
    X_train = _apply_means(X_train_raw, means)
    X_test = _apply_means(X_test_raw, means)

    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(np.float32)

    def _predict_swapped(swapped_feature_df: pd.DataFrame) -> np.ndarray:
        X_swap_raw = swapped_feature_df.values.astype(np.float32)
        X_swap = _apply_means(X_swap_raw, means)
        return model.predict(X_swap).astype(np.float32)

    return cw._symmetrize_neutral_margin(test_df, pred, _predict_swapped)


def _predict_lgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict[str, object]) -> np.ndarray:
    inner_train_df, inner_val_df = cw._split_inner_train_val(train_df)
    X_train_raw, y_train = _prepare_xy(inner_train_df)
    X_val_raw, y_val = _prepare_xy(inner_val_df)
    X_test_raw, _ = _prepare_xy(test_df)

    means = _train_impute_means(X_train_raw)
    X_train = _apply_means(X_train_raw, means)
    X_val = _apply_means(X_val_raw, means)
    X_test = _apply_means(X_test_raw, means)

    model = lgb.LGBMRegressor(**params)
    eval_metric = "l2" if params.get("objective") == "regression" else "l1"
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iteration = int(model.best_iteration_ or int(params.get("n_estimators", 5000)))
    pred = model.predict(X_test, num_iteration=best_iteration).astype(np.float32)

    def _predict_swapped(swapped_feature_df: pd.DataFrame) -> np.ndarray:
        X_swap_raw = swapped_feature_df.values.astype(np.float32)
        X_swap = _apply_means(X_swap_raw, means)
        return model.predict(X_swap, num_iteration=best_iteration).astype(np.float32)

    return cw._symmetrize_neutral_margin(test_df, pred, _predict_swapped)


def _predict_source_variant(
    source: SourceConfig,
    variant: Variant,
    holdout: int,
) -> pd.DataFrame:
    train_df = _load_fold_frame(source, _training_seasons(holdout))
    with _cw_source(source):
        holdout_df = cw._attach_book_spread(cw._load_fold_frame([holdout]), holdout)

    if variant.family == "hgbr":
        pred = _predict_hgbr(train_df, holdout_df, variant.params or {})
    elif variant.family == "lgbm":
        pred = _predict_lgbm(train_df, holdout_df, variant.params or {})
    else:
        raise ValueError(f"Unsupported predictive family: {variant.family}")

    out = cw._build_prediction_frame(
        variant.label,
        holdout,
        holdout_df,
        pred,
    )
    return out


def _existing_dir_for(source: SourceConfig, season: int) -> Path:
    if source.label == TORVIK.label:
        return BASE_BENCHMARK_TORVIK_2026 if season == 2026 else BASE_BENCHMARK_TORVIK
    return BASE_BENCHMARK_GOLD_2026 if season == 2026 else BASE_BENCHMARK_GOLD


def _load_existing_baseline_source_preds(source: SourceConfig, season: int) -> pd.DataFrame:
    path = _existing_dir_for(source, season) / "predictions" / "HistGradientBoosting" / f"season_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _blend_source_predictions(
    torvik_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    variant_label: str,
) -> pd.DataFrame:
    merge_cols = [
        "gameId",
        "startDate",
        "homeTeamId",
        "awayTeamId",
        "homeTeam",
        "awayTeam",
    ]
    torvik_base = torvik_df.rename(columns={"pred_margin": "pred_torvik"})[
        merge_cols + ["actual_margin", "book_spread", "pred_torvik"]
    ]
    gold_base = gold_df.rename(columns={"pred_margin": "pred_gold"})[
        merge_cols + ["pred_gold"]
    ]
    merged = torvik_base.merge(
        gold_base,
        on=merge_cols,
        how="inner",
        validate="one_to_one",
    )
    gold_w = gold_weight_for_start_dates(merged["startDate"])
    pred = (1.0 - gold_w) * merged["pred_torvik"].to_numpy(dtype=float) + gold_w * merged["pred_gold"].to_numpy(dtype=float)

    out = merged.copy()
    out["model"] = variant_label
    out["pred_margin"] = pred.astype(float)
    out["abs_error"] = np.abs(out["pred_margin"] - out["actual_margin"])
    out["pred_home_win"] = (out["pred_margin"] > 0).astype(int)
    out["sigma"] = np.nan
    keep = [
        "model",
        "gameId",
        "startDate",
        "homeTeamId",
        "awayTeamId",
        "homeTeam",
        "awayTeam",
        "actual_margin",
        "pred_margin",
        "abs_error",
        "pred_home_win",
        "book_spread",
        "sigma",
    ]
    return out[keep].copy()


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    actual = df["actual_margin"].to_numpy(dtype=float)
    pred = df["pred_margin"].to_numpy(dtype=float)
    lined = df["book_spread"].notna().to_numpy()
    return {
        "n_games": int(len(df)),
        "n_lined": int(lined.sum()),
        "MAE_all": float(np.mean(np.abs(actual - pred))),
        "MAE_lined": float(np.mean(np.abs(actual[lined] - pred[lined]))) if lined.any() else math.nan,
    }


def _favorite_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    lined = df["book_spread"].notna().to_numpy()
    data = df.loc[lined].copy()
    market_margin = -data["book_spread"].to_numpy(dtype=float)
    fav_sign = np.where(market_margin >= 0, 1.0, -1.0)
    market_fav = np.abs(market_margin)
    model_fav = data["pred_margin"].to_numpy(dtype=float) * fav_sign
    actual_fav = data["actual_margin"].to_numpy(dtype=float) * fav_sign
    bucket = pd.cut(
        market_fav,
        bins=[0, 5, 10, 15, np.inf],
        labels=["<5", "5-10", "10-15", "15+"],
        include_lowest=True,
        right=False,
    )
    rows = []
    for label in ["<5", "5-10", "10-15", "15+"]:
        mask = bucket.astype(str) == label
        if not mask.any():
            continue
        rows.append(
            {
                "bucket": label,
                "n": int(mask.sum()),
                "mean_market_margin": float(np.mean(market_fav[mask])),
                "mean_model_margin": float(np.mean(model_fav[mask])),
                "mean_actual_margin": float(np.mean(actual_fav[mask])),
                "model_minus_market": float(np.mean(model_fav[mask] - market_fav[mask])),
                "actual_minus_model": float(np.mean(actual_fav[mask] - model_fav[mask])),
                "bucket_mae": float(np.mean(np.abs(actual_fav[mask] - model_fav[mask]))),
            }
        )
    return pd.DataFrame(rows)


def _tail_distribution_summary(df: pd.DataFrame) -> dict[str, float]:
    lined = df["book_spread"].notna().to_numpy()
    data = df.loc[lined].copy()
    market_margin = -data["book_spread"].to_numpy(dtype=float)
    fav_sign = np.where(market_margin >= 0, 1.0, -1.0)
    model_fav = data["pred_margin"].to_numpy(dtype=float) * fav_sign
    return {
        "share_pred_10plus": float(np.mean(model_fav >= 10.0)),
        "share_pred_15plus": float(np.mean(model_fav >= 15.0)),
        "q90_pred_fav": float(np.quantile(model_fav, 0.90)),
        "q95_pred_fav": float(np.quantile(model_fav, 0.95)),
        "mean_pred_fav": float(np.mean(model_fav)),
    }


def _render_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Objective-level favorite-tail compression experiment.")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "baseline": asdict(BASELINE_VARIANT),
        "candidates": [asdict(v) for v in VARIANTS if v is not BASELINE_VARIANT],
        "holdout_seasons": HOLDOUT_SEASONS,
        "exclude_seasons": EXCLUDE_SEASONS,
        "sources": [asdict(TORVIK), asdict(GOLD)],
        "blend_schedule": {
            "start_day": config.PRODUCTION_MU_BLEND_START_DAY,
            "end_day": config.PRODUCTION_MU_BLEND_END_DAY,
        },
    }
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    blended_frames: dict[str, list[pd.DataFrame]] = {v.label: [] for v in VARIANTS}
    season_rows: list[dict[str, object]] = []

    for holdout in HOLDOUT_SEASONS:
        print(f"\n=== Holdout {holdout} ===")

        baseline_torvik = _load_existing_baseline_source_preds(TORVIK, holdout)
        baseline_gold = _load_existing_baseline_source_preds(GOLD, holdout)
        baseline_blended = _blend_source_predictions(baseline_torvik, baseline_gold, BASELINE_VARIANT.label)
        baseline_blended["holdout_season"] = holdout
        blended_frames[BASELINE_VARIANT.label].append(baseline_blended)

        m = _metrics(baseline_blended)
        season_rows.append({"variant": BASELINE_VARIANT.label, "holdout_season": holdout, **m})
        print(f"  {BASELINE_VARIANT.label}: MAE_all={m['MAE_all']:.4f} MAE_lined={m['MAE_lined']:.4f}")

        for variant in VARIANTS[1:]:
            torvik_pred = _predict_source_variant(TORVIK, variant, holdout)
            gold_pred = _predict_source_variant(GOLD, variant, holdout)
            blended = _blend_source_predictions(torvik_pred, gold_pred, variant.label)
            blended["holdout_season"] = holdout
            blended_frames[variant.label].append(blended)
            m = _metrics(blended)
            season_rows.append({"variant": variant.label, "holdout_season": holdout, **m})
            print(f"  {variant.label}: MAE_all={m['MAE_all']:.4f} MAE_lined={m['MAE_lined']:.4f}")

    season_metrics = pd.DataFrame(season_rows).sort_values(["holdout_season", "variant"]).reset_index(drop=True)
    season_metrics.to_csv(out_dir / "season_metrics.csv", index=False)

    pooled_rows: list[dict[str, object]] = []
    dec15_rows: list[dict[str, object]] = []
    favorite_rows: list[pd.DataFrame] = []
    favorite_dec15_rows: list[pd.DataFrame] = []
    tail_rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        pooled = pd.concat(blended_frames[variant.label], ignore_index=True)
        pooled.to_parquet(out_dir / f"{variant.label}_predictions.parquet", index=False)
        pooled_metrics = _metrics(pooled)
        pooled_rows.append({"variant": variant.label, **pooled_metrics})

        dec_mask = _is_dec15_plus(pooled["startDate"])
        dec_df = pooled.loc[dec_mask].copy()
        dec_metrics = _metrics(dec_df)
        dec15_rows.append({"variant": variant.label, **dec_metrics})

        fav = _favorite_bucket_summary(pooled)
        fav["variant"] = variant.label
        favorite_rows.append(fav)

        fav_dec = _favorite_bucket_summary(dec_df)
        fav_dec["variant"] = variant.label
        favorite_dec15_rows.append(fav_dec)

        tail_rows.append({"variant": variant.label, **_tail_distribution_summary(pooled)})

    pooled_df = pd.DataFrame(pooled_rows).sort_values("variant").reset_index(drop=True)
    dec15_df = pd.DataFrame(dec15_rows).sort_values("variant").reset_index(drop=True)
    favorite_df = pd.concat(favorite_rows, ignore_index=True)
    favorite_dec15_df = pd.concat(favorite_dec15_rows, ignore_index=True)
    tail_df = pd.DataFrame(tail_rows).sort_values("variant").reset_index(drop=True)

    pooled_df.to_csv(out_dir / "pooled_metrics.csv", index=False)
    dec15_df.to_csv(out_dir / "dec15_metrics.csv", index=False)
    favorite_df.to_csv(out_dir / "favorite_bucket_summary.csv", index=False)
    favorite_dec15_df.to_csv(out_dir / "favorite_bucket_summary_dec15.csv", index=False)
    tail_df.to_csv(out_dir / "tail_distribution_summary.csv", index=False)

    summary_lines = [
        "# Objective Tail Compression Experiment",
        "",
        "## Pooled Metrics",
        "",
        _render_table(pooled_df),
        "",
        "## Dec 15+ Metrics",
        "",
        _render_table(dec15_df),
        "",
        "## Favorite Buckets (Pooled)",
        "",
        _render_table(favorite_df),
        "",
        "## Favorite Buckets (Dec 15+)",
        "",
        _render_table(favorite_dec15_df),
        "",
        "## Tail Distribution Summary",
        "",
        _render_table(tail_df),
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")

    print(f"\nSaved objective experiment artifacts to {out_dir}")


if __name__ == "__main__":
    main()

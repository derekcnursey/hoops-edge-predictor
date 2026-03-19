#!/usr/bin/env python3
"""Follow-up HGBR-native sigma study using log(abs_residual + eps)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import load_season_features
from src.features import get_feature_matrix
from src.sigma_calibration import apply_sigma_transform

HOLDOUT_SEASONS = [2019, 2020, 2022, 2023, 2024, 2025]
EXCLUDE_SEASONS = [2021]
NO_GARBAGE = True
EFFICIENCY_SOURCE = "torvik"
ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
BREAKEVEN = 110.0 / 210.0
WIN_PROFIT = 100.0 / 110.0
TOP_NS = [100, 200, 500]
PROB_BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
EDGE_BUCKETS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.00]
RESIDUAL_HGBR_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_iter": 300,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
}
CONST_SIGMA_BASELINE = 14.0
POSTHOC_SHRINK = {"mode": "shrink", "shrink_alpha": 0.25, "shrink_target": 14.0}
LOG_EPS = 0.25


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def _train_impute_means(X: np.ndarray) -> np.ndarray:
    means = np.nanmean(X, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    return means.astype(np.float32)


def _apply_impute_means(X: np.ndarray, means: np.ndarray) -> np.ndarray:
    out = X.copy()
    nan_mask = np.isnan(out)
    for j in range(out.shape[1]):
        out[nan_mask[:, j], j] = means[j]
    return out


def _load_feature_frame(season: int) -> pd.DataFrame:
    df = load_season_features(
        season,
        no_garbage=NO_GARBAGE,
        adj_suffix=ADJ_SUFFIX,
        efficiency_source=EFFICIENCY_SOURCE,
    )
    out = df.dropna(subset=["homeScore", "awayScore"]).copy()
    out = out[(out["homeScore"] != 0) | (out["awayScore"] != 0)].copy()
    return out.reset_index(drop=True)


def _load_holdout_predictions(benchmark_dir: Path, model_name: str, season: int) -> pd.DataFrame:
    path = benchmark_dir / "predictions" / model_name / f"season_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _build_residual_dataset(benchmark_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in HOLDOUT_SEASONS:
        if season in EXCLUDE_SEASONS:
            continue
        feat = _load_feature_frame(season)
        hgb = _load_holdout_predictions(benchmark_dir, "HistGradientBoosting", season)
        mlp = _load_holdout_predictions(benchmark_dir, "CurrentMLP", season)

        merged = feat.merge(
            hgb[["gameId", "pred_margin", "book_spread"]],
            on="gameId",
            how="inner",
        ).rename(columns={"pred_margin": "pred_margin_mu"})
        merged = merged.merge(
            mlp[["gameId", "sigma"]],
            on="gameId",
            how="inner",
        ).rename(columns={"sigma": "raw_sigma"})
        merged["actual_margin"] = (
            merged["homeScore"].astype(float) - merged["awayScore"].astype(float)
        )
        merged["residual"] = merged["actual_margin"] - merged["pred_margin_mu"]
        merged["abs_residual"] = merged["residual"].abs()
        merged["log_abs_residual_target"] = np.log(merged["abs_residual"] + LOG_EPS)
        merged["actual_edge_home"] = merged["actual_margin"] + merged["book_spread"]
        merged["home_cover_win"] = merged["actual_edge_home"] > 0
        merged["push"] = merged["actual_edge_home"] == 0
        merged["season"] = season
        merged["startDate"] = pd.to_datetime(merged["startDate"], errors="coerce", utc=True)
        merged["month"] = merged["startDate"].dt.month
        merged["phase"] = np.where(
            merged["month"].isin([11, 12]),
            "Nov-Dec",
            np.where(merged["month"].isin([1, 2, 3]), "Jan-Mar", "Other"),
        )
        rows.append(merged)
    df = pd.concat(rows, ignore_index=True)
    return df[df["book_spread"].notna()].reset_index(drop=True)


def _prepare_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return get_feature_matrix(df).values.astype(np.float32)


def _fit_residual_model(train_df: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, np.ndarray]:
    X_raw = _prepare_feature_matrix(train_df)
    means = _train_impute_means(X_raw)
    X = _apply_impute_means(X_raw, means)
    y = train_df["log_abs_residual_target"].to_numpy(dtype=np.float32)
    model = HistGradientBoostingRegressor(**RESIDUAL_HGBR_PARAMS)
    model.fit(X, y)
    return model, means


def _predict_abs_residual_proxy(model: HistGradientBoostingRegressor, means: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    X_raw = _prepare_feature_matrix(df)
    X = _apply_impute_means(X_raw, means)
    pred_log = model.predict(X).astype(np.float32)
    proxy = np.exp(pred_log) - LOG_EPS
    return np.clip(proxy, 0.5, 30.0)


def _fit_scalar_from_predictions(df: pd.DataFrame, proxy: np.ndarray) -> float:
    target = df["home_cover_win"].astype(float).to_numpy()
    edge_home_points = df["pred_margin_mu"].to_numpy(dtype=float) + df["book_spread"].to_numpy(dtype=float)
    best_c = 1.0
    best_score: tuple[float, float] | None = None
    for c in np.arange(0.50, 2.51, 0.02):
        sigma = np.clip(c * proxy, 0.5, 30.0)
        p_home_cover = np.clip(_norm_cdf(edge_home_points / sigma), 1e-9, 1.0 - 1e-9)
        logloss = float(-(target * np.log(p_home_cover) + (1.0 - target) * np.log(1.0 - p_home_cover)).mean())
        score = (logloss, float(np.mean(sigma)))
        if best_score is None or score < best_score:
            best_score = score
            best_c = float(c)
    return best_c


def _fit_affine_from_predictions(df: pd.DataFrame, proxy: np.ndarray) -> tuple[float, float]:
    target = df["home_cover_win"].astype(float).to_numpy()
    edge_home_points = df["pred_margin_mu"].to_numpy(dtype=float) + df["book_spread"].to_numpy(dtype=float)
    best = (0.0, 1.0)
    best_score: tuple[float, float] | None = None
    for a in np.arange(0.0, 6.01, 0.25):
        for b in np.arange(0.50, 2.01, 0.05):
            sigma = np.clip(a + b * proxy, 0.5, 30.0)
            p_home_cover = np.clip(_norm_cdf(edge_home_points / sigma), 1e-9, 1.0 - 1e-9)
            logloss = float(-(target * np.log(p_home_cover) + (1.0 - target) * np.log(1.0 - p_home_cover)).mean())
            score = (logloss, float(np.mean(sigma)))
            if best_score is None or score < best_score:
                best_score = score
                best = (float(a), float(b))
    return best


def _crossval_proxy_predictions(train_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    seasons = sorted(train_df["season"].unique().tolist())
    if len(seasons) == 1:
        model, means = _fit_residual_model(train_df)
        out = train_df.copy()
        out["pred_abs_residual_proxy"] = _predict_abs_residual_proxy(model, means, train_df)
        return out, "in_sample_single_season"

    oof_parts = []
    for holdout in seasons:
        fit_df = train_df[train_df["season"] != holdout].copy()
        pred_df = train_df[train_df["season"] == holdout].copy()
        model, means = _fit_residual_model(fit_df)
        pred_df = pred_df.copy()
        pred_df["pred_abs_residual_proxy"] = _predict_abs_residual_proxy(model, means, pred_df)
        oof_parts.append(pred_df)
    return pd.concat(oof_parts, ignore_index=True), "season_oof"


def _evaluate_sigma(df: pd.DataFrame, sigma: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    sigma = np.clip(np.asarray(sigma, dtype=float), 0.5, 30.0)
    err = df["actual_margin"].to_numpy(dtype=float) - df["pred_margin_mu"].to_numpy(dtype=float)
    abs_err = np.abs(err)
    edge_home_points = df["pred_margin_mu"].to_numpy(dtype=float) + df["book_spread"].to_numpy(dtype=float)
    p_home_cover = np.clip(_norm_cdf(edge_home_points / sigma), 1e-9, 1.0 - 1e-9)
    y_cover = df["home_cover_win"].astype(float).to_numpy()
    pick_home = edge_home_points >= 0.0
    pick_prob = np.where(pick_home, p_home_cover, 1.0 - p_home_cover)
    pick_win = np.where(pick_home, df["actual_edge_home"] > 0, df["actual_edge_home"] < 0)
    push = df["push"].to_numpy(bool)
    picked_profit = np.where(push, 0.0, np.where(pick_win, WIN_PROFIT, -1.0))

    metrics = {
        "n_games": int(len(df)),
        "gaussian_nll": float(np.mean(0.5 * np.log(2.0 * math.pi * sigma ** 2) + (err ** 2) / (2.0 * sigma ** 2))),
        "cover_logloss": float(-(y_cover * np.log(p_home_cover) + (1.0 - y_cover) * np.log(1.0 - p_home_cover)).mean()),
        "cover_brier": float(np.mean((p_home_cover - y_cover) ** 2)),
        "mean_sigma": float(np.mean(sigma)),
        "std_sigma": float(np.std(sigma)),
        "mean_abs_z": float(np.mean(abs_err / sigma)),
        "cov1": float(np.mean(abs_err <= sigma)),
        "cov2": float(np.mean(abs_err <= 2.0 * sigma)),
    }
    for top_n in TOP_NS:
        chosen = np.argsort(-pick_prob)[: min(top_n, len(df))]
        mask = np.zeros(len(df), dtype=bool)
        mask[chosen] = True
        wins = pick_win[mask & ~push]
        metrics[f"top{top_n}_roi"] = float(np.mean(picked_profit[mask]))
        metrics[f"top{top_n}_winrate"] = float(np.mean(wins)) if len(wins) else float("nan")
        metrics[f"top{top_n}_avg_prob"] = float(np.mean(pick_prob[mask]))

    pred_df = df[
        [
            "season",
            "phase",
            "gameId",
            "startDate",
            "book_spread",
            "pred_margin_mu",
            "actual_margin",
            "actual_edge_home",
        ]
    ].copy()
    pred_df["sigma"] = sigma
    pred_df["pick_prob"] = pick_prob
    pred_df["pick_prob_edge"] = pick_prob - BREAKEVEN
    pred_df["pick_win"] = pick_win
    pred_df["picked_profit"] = picked_profit
    pred_df["push"] = push
    return metrics, pred_df


def _bucket_rows(pred_df: pd.DataFrame, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bucket_kind, bins, series in [
        ("probability", PROB_BUCKETS, pred_df["pick_prob"]),
        ("edge", EDGE_BUCKETS, pred_df["pick_prob_edge"]),
    ]:
        bucketed = pd.cut(series, bins=bins, include_lowest=True, right=False)
        for bucket, idx in bucketed.groupby(bucketed, observed=False).groups.items():
            if len(idx) == 0:
                continue
            grp = pred_df.loc[idx]
            nongrp = grp[~grp["push"]]
            rows.append(
                {
                    "option": label,
                    "season": int(grp["season"].iloc[0]),
                    "bucket_kind": bucket_kind,
                    "bucket": str(bucket),
                    "n": int(len(grp)),
                    "avg_pick_prob": float(grp["pick_prob"].mean()),
                    "win_rate": float(nongrp["pick_win"].mean()) if len(nongrp) else float("nan"),
                    "roi": float(grp["picked_profit"].mean()),
                }
            )
    return rows


def _best_posthoc_sigma(raw_sigma: pd.Series) -> np.ndarray:
    return apply_sigma_transform(
        raw_sigma.to_numpy(dtype=float),
        mode=POSTHOC_SHRINK["mode"],
        shrink_alpha=POSTHOC_SHRINK["shrink_alpha"],
        shrink_target=POSTHOC_SHRINK["shrink_target"],
    )


def _constant_sigma(raw_sigma: pd.Series) -> np.ndarray:
    return np.full(len(raw_sigma), CONST_SIGMA_BASELINE, dtype=float)


def _pooled_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for option, grp in fold_df.groupby("option"):
        row = {"option": option}
        for col in [
            "n_games",
            "gaussian_nll",
            "cover_logloss",
            "cover_brier",
            "mean_sigma",
            "std_sigma",
            "mean_abs_z",
            "cov1",
            "cov2",
            "top100_roi",
            "top200_roi",
            "top500_roi",
            "top100_winrate",
            "top200_winrate",
            "top500_winrate",
            "top100_avg_prob",
            "top200_avg_prob",
            "top500_avg_prob",
        ]:
            row[col] = float(np.average(grp[col], weights=grp["n_games"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["top200_roi", "cover_logloss", "gaussian_nll"],
        ascending=[False, True, True],
    )


@click.command()
@click.option(
    "--benchmark-dir",
    default="artifacts/benchmarks/canonical_walkforward_v2_lgb",
    type=click.Path(path_type=Path),
    help="Canonical benchmark artifact directory with HGBR and CurrentMLP predictions.",
)
@click.option(
    "--output-dir",
    default="artifacts/hgbr_residual_sigma_log_study_v1",
    type=click.Path(path_type=Path),
    help="Output directory for the log-residual sigma study.",
)
def main(benchmark_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    study_df = _build_residual_dataset(benchmark_dir)
    study_df.to_parquet(output_dir / "residual_dataset.parquet", index=False)

    eval_seasons = [season for season in HOLDOUT_SEASONS if season not in EXCLUDE_SEASONS][1:]
    fold_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []

    for season in eval_seasons:
        train_df = study_df[study_df["season"] < season].copy()
        eval_df = study_df[study_df["season"] == season].copy()

        cv_pred_df, fit_mode = _crossval_proxy_predictions(train_df)
        scalar_c = _fit_scalar_from_predictions(
            cv_pred_df,
            cv_pred_df["pred_abs_residual_proxy"].to_numpy(dtype=float),
        )
        affine_a, affine_b = _fit_affine_from_predictions(
            cv_pred_df,
            cv_pred_df["pred_abs_residual_proxy"].to_numpy(dtype=float),
        )

        resid_model, resid_means = _fit_residual_model(train_df)
        eval_proxy = _predict_abs_residual_proxy(resid_model, resid_means, eval_df)
        eval_sigma_scalar = np.clip(scalar_c * eval_proxy, 0.5, 30.0)
        eval_sigma_affine = np.clip(affine_a + affine_b * eval_proxy, 0.5, 30.0)

        calibration_rows.append(
            {
                "season": season,
                "fit_mode": fit_mode,
                "scalar_c": scalar_c,
                "affine_a": affine_a,
                "affine_b": affine_b,
                "train_seasons": ",".join(str(s) for s in sorted(train_df["season"].unique())),
            }
        )

        option_sigmas = {
            "raw_mlp_sigma": eval_df["raw_sigma"].to_numpy(dtype=float),
            "best_posthoc_sigma": _best_posthoc_sigma(eval_df["raw_sigma"]),
            "constant_14_sigma": _constant_sigma(eval_df["raw_sigma"]),
            "hgbr_residual_log_scalar": eval_sigma_scalar,
            "hgbr_residual_log_affine": eval_sigma_affine,
        }

        season_pred_dir = output_dir / "predictions"
        season_pred_dir.mkdir(parents=True, exist_ok=True)
        raw_top200 = None
        for option, sigma in option_sigmas.items():
            metrics, pred_df = _evaluate_sigma(eval_df, sigma)
            pred_df["option"] = option
            pred_df["raw_sigma"] = eval_df["raw_sigma"].to_numpy(dtype=float)
            if option.startswith("hgbr_residual_log_"):
                pred_df["pred_abs_residual_proxy"] = eval_proxy
                pred_df["scalar_c"] = scalar_c
                pred_df["affine_a"] = affine_a
                pred_df["affine_b"] = affine_b
            pred_df.to_parquet(season_pred_dir / f"{option}_season_{season}.parquet", index=False)

            top200 = set(pred_df["pick_prob"].nlargest(min(200, len(pred_df))).index.tolist())
            if option == "raw_mlp_sigma":
                raw_top200 = top200
                overlap = len(top200)
                churn = 0
            else:
                overlap = len(raw_top200 & top200) if raw_top200 is not None else np.nan
                churn = len(raw_top200.symmetric_difference(top200)) if raw_top200 is not None else np.nan

            fold_rows.append(
                {
                    "season": season,
                    "option": option,
                    "fit_mode": fit_mode if option.startswith("hgbr_residual_log_") else "",
                    "scalar_c": scalar_c if option == "hgbr_residual_log_scalar" else np.nan,
                    "affine_a": affine_a if option == "hgbr_residual_log_affine" else np.nan,
                    "affine_b": affine_b if option == "hgbr_residual_log_affine" else np.nan,
                    "top200_overlap_vs_raw": overlap,
                    "top200_churn_vs_raw": churn,
                    **metrics,
                }
            )
            bucket_rows.extend(_bucket_rows(pred_df, option))

    fold_df = pd.DataFrame(fold_rows).sort_values(["season", "option"]).reset_index(drop=True)
    pooled_df = _pooled_metrics(fold_df)
    calibration_df = pd.DataFrame(calibration_rows)
    bucket_df = pd.DataFrame(bucket_rows)

    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    pooled_df.to_csv(output_dir / "pooled_metrics.csv", index=False)
    calibration_df.to_csv(output_dir / "selected_calibrations.csv", index=False)
    bucket_df.to_csv(output_dir / "bucket_metrics.csv", index=False)

    protocol = {
        "benchmark_dir": str(benchmark_dir),
        "dataset_rows_lined": int(len(study_df)),
        "holdout_seasons": HOLDOUT_SEASONS,
        "excluded_seasons": EXCLUDE_SEASONS,
        "eval_seasons": eval_seasons,
        "log_target": {
            "definition": "log(abs_residual + eps)",
            "eps": LOG_EPS,
        },
        "residual_model": {
            "family": "HistGradientBoostingRegressor",
            "params": RESIDUAL_HGBR_PARAMS,
        },
        "calibration": {
            "scalar_grid": {"c_min": 0.50, "c_max": 2.50, "c_step": 0.02},
            "affine_grid": {
                "a_min": 0.0,
                "a_max": 6.0,
                "a_step": 0.25,
                "b_min": 0.50,
                "b_max": 2.00,
                "b_step": 0.05,
            },
            "objective": "cover_event_logloss",
            "protocol": "fit on prior seasons only; season-OOF on prior seasons when possible",
        },
        "baselines": {
            "raw_mlp_sigma": "raw sigma from CurrentMLP holdout predictions",
            "best_posthoc_sigma": POSTHOC_SHRINK,
            "constant_14_sigma": CONST_SIGMA_BASELINE,
        },
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))

    summary = [
        "# HGBR Residual Sigma Log Study v1",
        "",
        f"- Benchmark dir: `{benchmark_dir}`",
        f"- Residual dataset rows (lined): `{len(study_df)}`",
        f"- Evaluation seasons: `{eval_seasons}`",
        f"- Log residual eps: `{LOG_EPS}`",
        "- Mu held fixed to canonical HistGradientBoosting holdout predictions",
        "- Residual model: `HistGradientBoostingRegressor`",
        "- Compare scalar and affine calibration on prior seasons only",
        "",
        "## Pooled Metrics",
        "",
        pooled_df.to_string(index=False),
        "",
        "## Selected Calibrations",
        "",
        calibration_df.to_string(index=False),
    ]
    (output_dir / "summary.md").write_text("\n".join(summary))

    click.echo(f"Log-residual sigma study written to {output_dir}")
    best = pooled_df.iloc[0]
    click.echo(
        f"Best by top200 ROI: {best['option']} "
        f"(top200_roi={best['top200_roi']:.4f}, cover_logloss={best['cover_logloss']:.4f})"
    )


if __name__ == "__main__":
    main()

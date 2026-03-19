#!/usr/bin/env python3
"""Canonical walk-forward benchmark for pure game prediction.

Protocol:
  - Predict actual game margin: homeScore - awayScore
  - Use prebuilt production-style features (Torvik, adjusted, 53 features)
  - Evaluate on fixed outer holdout seasons
  - Keep Vegas spreads out of training features
  - Use a provider-preference book spread only as an external benchmark on lined games
  - Fit preprocessing inside each fold only
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import build_features, get_feature_matrix, get_targets, load_research_lines

SEED = 42
TRAIN_START = 2015
EXCLUDE_SEASONS = [2021]
HOLDOUT_SEASONS = [2019, 2020, 2022, 2023, 2024, 2025]
INNER_VAL_FRAC = 0.15
ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
NO_GARBAGE = True
EFFICIENCY_SOURCE = "torvik"
GOLD_TABLE_NAME: str | None = None

RIDGE_ALPHA = 10.0
HGBR_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_iter": 300,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": SEED,
}
LGBM_PARAMS = {
    "objective": "regression_l1",
    "learning_rate": 0.02019353198222356,
    "num_leaves": 110,
    "max_depth": 9,
    "min_child_samples": 60,
    "subsample": 0.6469253206264255,
    "colsample_bytree": 0.4684413154048659,
    "reg_alpha": 1.561017503434852,
    "reg_lambda": 0.004257796528177626,
    "n_estimators": 5000,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
MLP_HP = {
    "hidden1": 384,
    "hidden2": 256,
    "dropout": 0.2,
    "batch_size": 4096,
    "lr": 0.003,
    "weight_decay": 1e-4,
    "epochs": 150,
    "sigma_param": "exp",
}
LINE_PROVIDER_RANK = {"Draft Kings": 0, "ESPN BET": 1, "Bovada": 2}
LINE_PROVIDER_PREFERENCE = list(LINE_PROVIDER_RANK.keys())
BOOK_BENCHMARK_LABEL = "PreferredBookSpread"
LINE_SELECTION_RULE = (
    "Select one spread per game by taking the first non-null spread after sorting "
    "by preferred providers (Draft Kings, ESPN BET, Bovada), then by provider name."
)
MODEL_ORDER = [
    "HomeMarginMean",
    "Ridge",
    "HistGradientBoosting",
    "LightGBM",
    "CurrentMLP",
    BOOK_BENCHMARK_LABEL,
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_output_dir() -> Path:
    return config.ARTIFACTS_DIR / "benchmarks" / "canonical_walkforward_v1"


def _feature_cache_dir() -> Path:
    return config.FEATURES_DIR / "canonical_walkforward"


def _feature_cache_path(season: int) -> Path:
    source_slug = EFFICIENCY_SOURCE
    if GOLD_TABLE_NAME:
        source_slug += "_" + GOLD_TABLE_NAME
    source_slug = source_slug.replace("/", "_")
    return _feature_cache_dir() / source_slug / f"season_{season}.parquet"


def _folds(selected_holdout: int | None = None) -> list[dict]:
    if selected_holdout is not None and selected_holdout not in HOLDOUT_SEASONS:
        if selected_holdout < TRAIN_START:
            raise ValueError(
                f"holdout season {selected_holdout} is before TRAIN_START={TRAIN_START}"
            )
        if selected_holdout in EXCLUDE_SEASONS:
            raise ValueError(
                f"holdout season {selected_holdout} is excluded by config: {EXCLUDE_SEASONS}"
            )
        train_seasons = [
            season for season in range(TRAIN_START, selected_holdout)
            if season not in EXCLUDE_SEASONS
        ]
        return [{
            "holdout_season": selected_holdout,
            "train_seasons": train_seasons,
        }]

    folds = []
    for holdout in HOLDOUT_SEASONS:
        if selected_holdout is not None and holdout != selected_holdout:
            continue
        train_seasons = [
            season for season in range(TRAIN_START, holdout)
            if season not in EXCLUDE_SEASONS
        ]
        folds.append({
            "holdout_season": holdout,
            "train_seasons": train_seasons,
        })
    if selected_holdout is not None and not folds:
        raise ValueError(
            f"holdout season {selected_holdout} is not in canonical holdouts {HOLDOUT_SEASONS}"
        )
    return folds


def _load_fold_frame(seasons: list[int]) -> pd.DataFrame:
    if GOLD_TABLE_NAME is not None:
        frames = []
        for season in seasons:
            cache_path = _feature_cache_path(season)
            if cache_path.exists():
                frames.append(pd.read_parquet(cache_path))
                continue
            df = build_features(
                season,
                no_garbage=NO_GARBAGE,
                extra_features=config.EXTRA_FEATURES,
                adjust_ff=config.ADJUST_FF,
                adjust_alpha=config.ADJUST_ALPHA,
                adjust_prior_weight=config.ADJUST_PRIOR,
                adjust_ff_method=config.ADJUST_FF_METHOD,
                efficiency_source=EFFICIENCY_SOURCE,
                gold_table_name=GOLD_TABLE_NAME,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not df.empty:
                df.to_parquet(cache_path, index=False)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return _clean_games(pd.concat(frames, ignore_index=True))
    df = load_multi_season_features(
        seasons,
        no_garbage=NO_GARBAGE,
        adj_suffix=ADJ_SUFFIX,
        efficiency_source=EFFICIENCY_SOURCE,
    )
    return _clean_games(df)


def _clean_games(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["homeScore", "awayScore"]).copy()
    out = out[(out["homeScore"] != 0) | (out["awayScore"] != 0)].copy()
    return out.reset_index(drop=True)


def _sort_games(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_sort_date"] = pd.to_datetime(out["startDate"], errors="coerce", utc=True)
    out = out.sort_values(["_sort_date", "gameId"], kind="mergesort").reset_index(drop=True)
    return out.drop(columns=["_sort_date"])


def _split_inner_train_val(df: pd.DataFrame, val_frac: float = INNER_VAL_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = _sort_games(df)
    n_rows = len(ordered)
    if n_rows < 2:
        raise ValueError("Need at least 2 rows to create temporal inner validation split")
    n_val = max(1, int(n_rows * val_frac))
    n_val = min(n_val, n_rows - 1)
    return ordered.iloc[:-n_val].copy(), ordered.iloc[-n_val:].copy()


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


def _prepare_point_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = get_feature_matrix(df).values.astype(np.float32)
    y = get_targets(df)["spread_home"].values.astype(np.float32)
    return X, y


def _neutral_mask(df: pd.DataFrame) -> np.ndarray:
    if "neutral_site" in df.columns:
        return df["neutral_site"].fillna(0).astype(float).to_numpy() == 1.0
    if "neutralSite" in df.columns:
        return df["neutralSite"].fillna(0).astype(float).to_numpy() == 1.0
    return np.zeros(len(df), dtype=bool)


def _swap_feature_frame(feature_df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    swapped = feature_df[feature_order].copy()
    used: set[str] = set()
    explicit_pairs = [
        ("home_opp_ft_rate", "away_def_ft_rate"),
        ("home_team_efg_home_split", "away_team_efg_away_split"),
    ]
    for left, right in explicit_pairs:
        if left in swapped.columns and right in swapped.columns:
            tmp = swapped[left].copy()
            swapped[left] = swapped[right]
            swapped[right] = tmp
            used.add(left)
            used.add(right)

    for col in feature_order:
        if col in used:
            continue
        if col.startswith("home_"):
            other = "away_" + col[len("home_") :]
            if other in swapped.columns:
                tmp = swapped[col].copy()
                swapped[col] = swapped[other]
                swapped[other] = tmp
                used.add(col)
                used.add(other)

    if "rest_advantage" in swapped.columns:
        swapped["rest_advantage"] = -swapped["rest_advantage"]
    if "home_team_hca" in swapped.columns:
        swapped["home_team_hca"] = 0.0
    if "neutral_site" in swapped.columns:
        swapped["neutral_site"] = 1.0

    return swapped


def _symmetrize_neutral_margin(
    test_df: pd.DataFrame,
    pred_margin: np.ndarray,
    predict_swapped_margin: callable,
) -> np.ndarray:
    neutral_mask = _neutral_mask(test_df)
    if not neutral_mask.any():
        return pred_margin.astype(np.float32)

    neutral_idx = np.flatnonzero(neutral_mask)
    feature_df = get_feature_matrix(test_df.iloc[neutral_idx]).copy()
    swapped_feature_df = _swap_feature_frame(feature_df, list(feature_df.columns))
    pred_swap = np.asarray(predict_swapped_margin(swapped_feature_df), dtype=np.float32)

    out = pred_margin.astype(np.float32).copy()
    out[neutral_idx] = (out[neutral_idx] - pred_swap) / 2.0
    return out


def _symmetrize_neutral_gaussian(
    test_df: pd.DataFrame,
    pred_margin: np.ndarray,
    sigma: np.ndarray,
    predict_swapped_gaussian: callable,
) -> tuple[np.ndarray, np.ndarray]:
    neutral_mask = _neutral_mask(test_df)
    if not neutral_mask.any():
        return pred_margin.astype(np.float32), sigma.astype(np.float32)

    neutral_idx = np.flatnonzero(neutral_mask)
    feature_df = get_feature_matrix(test_df.iloc[neutral_idx]).copy()
    swapped_feature_df = _swap_feature_frame(feature_df, list(feature_df.columns))
    pred_swap, sigma_swap = predict_swapped_gaussian(swapped_feature_df)
    pred_swap = np.asarray(pred_swap, dtype=np.float32)
    sigma_swap = np.asarray(sigma_swap, dtype=np.float32)

    mu_out = pred_margin.astype(np.float32).copy()
    sigma_out = sigma.astype(np.float32).copy()

    mu_orig = mu_out[neutral_idx].copy()
    sigma_orig = sigma_out[neutral_idx].copy()
    mu_out[neutral_idx] = (mu_orig - pred_swap) / 2.0
    sigma_var = (
        0.5 * (sigma_orig ** 2 + sigma_swap ** 2)
        + ((mu_orig + pred_swap) ** 2) / 4.0
    )
    sigma_out[neutral_idx] = np.sqrt(np.maximum(sigma_var, 0.25)).astype(np.float32)
    return mu_out, sigma_out


def _line_selection_metadata() -> dict[str, object]:
    return {
        "benchmark_label": BOOK_BENCHMARK_LABEL,
        "provider_preference_order": LINE_PROVIDER_PREFERENCE,
        "uses_true_closing_timestamps": False,
        "selection_rule": LINE_SELECTION_RULE,
    }


def _dedupe_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    lines = lines_df.copy()
    if lines.empty or "gameId" not in lines.columns or "spread" not in lines.columns:
        return pd.DataFrame(columns=["gameId", "book_spread"])
    lines["spread"] = pd.to_numeric(lines["spread"], errors="coerce")
    if "provider" not in lines.columns:
        lines["provider"] = ""
    lines["provider"] = lines["provider"].fillna("").astype(str)
    lines["_has_spread"] = lines["spread"].notna().astype(int)
    lines["_prov_rank"] = lines["provider"].map(LINE_PROVIDER_RANK).fillna(99)
    lines["_provider_name"] = lines["provider"].str.casefold()
    lines = (
        lines.sort_values(
            ["_has_spread", "_prov_rank", "_provider_name"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["gameId"], keep="first")
        .rename(columns={"spread": "book_spread"})
    )
    return lines[["gameId", "book_spread"]].copy()


def _attach_book_spread(df: pd.DataFrame, season: int) -> pd.DataFrame:
    try:
        lines = load_research_lines(season)
    except Exception:
        lines = pd.DataFrame()
    lines = _dedupe_lines(lines)
    if lines.empty:
        out = df.copy()
        out["book_spread"] = np.nan
        return out
    return df.merge(lines, on="gameId", how="left")


def _gaussian_nll_numpy(actual: np.ndarray, pred: np.ndarray, sigma: np.ndarray) -> float:
    sigma_safe = np.clip(np.asarray(sigma, dtype=float), 0.5, 30.0)
    err = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    nll = 0.5 * np.log(2.0 * math.pi * sigma_safe ** 2) + (err ** 2) / (2.0 * sigma_safe ** 2)
    return float(np.mean(nll))


def _metrics_from_predictions(
    df: pd.DataFrame,
    probabilistic: bool = False,
    lined_only: bool = False,
) -> dict[str, float]:
    actual = df["actual_margin"].to_numpy(dtype=float)
    pred = df["pred_margin"].to_numpy(dtype=float)
    valid = ~np.isnan(pred)
    lined = df["book_spread"].notna() if "book_spread" in df.columns else pd.Series(False, index=df.index)
    if lined_only:
        valid = valid & lined.to_numpy()

    metrics = {
        "n_games": int(valid.sum()),
        "n_lined": int(lined.sum()) if "book_spread" in df.columns else 0,
        "MAE_all": np.nan,
        "RMSE_all": np.nan,
        "MedAE_all": np.nan,
        "WinAcc_all": np.nan,
        "MAE_lined": np.nan,
        "BookMAE_lined": np.nan,
        "DeltaVsBook_MAE": np.nan,
        "GaussianNLL": np.nan,
        "MeanAbsZ": np.nan,
        "Coverage_1sigma": np.nan,
        "Coverage_2sigma": np.nan,
    }

    if valid.any() and not lined_only:
        err = actual[valid] - pred[valid]
        abs_err = np.abs(err)
        metrics["MAE_all"] = float(np.mean(abs_err))
        metrics["RMSE_all"] = float(np.sqrt(np.mean(err ** 2)))
        metrics["MedAE_all"] = float(np.median(abs_err))
        metrics["WinAcc_all"] = float(np.mean((pred[valid] > 0) == (actual[valid] > 0)))

        if probabilistic and "sigma" in df.columns:
            sigma = df.loc[valid, "sigma"].to_numpy(dtype=float)
            sigma_safe = np.clip(sigma, 0.5, 30.0)
            z = abs_err / sigma_safe
            metrics["GaussianNLL"] = _gaussian_nll_numpy(actual[valid], pred[valid], sigma_safe)
            metrics["MeanAbsZ"] = float(np.mean(z))
            metrics["Coverage_1sigma"] = float(np.mean(abs_err <= sigma_safe))
            metrics["Coverage_2sigma"] = float(np.mean(abs_err <= (2.0 * sigma_safe)))

    if "book_spread" in df.columns:
        lined_valid = lined.to_numpy() & (~np.isnan(pred))
        if lined_valid.any():
            actual_lined = df.loc[lined_valid, "actual_margin"].to_numpy(dtype=float)
            pred_lined = df.loc[lined_valid, "pred_margin"].to_numpy(dtype=float)
            book_pred = -df.loc[lined_valid, "book_spread"].to_numpy(dtype=float)
            metrics["MAE_lined"] = float(np.mean(np.abs(actual_lined - pred_lined)))
            metrics["BookMAE_lined"] = float(np.mean(np.abs(actual_lined - book_pred)))
            metrics["DeltaVsBook_MAE"] = float(metrics["MAE_lined"] - metrics["BookMAE_lined"])

    return metrics


def _monthly_metrics(
    df: pd.DataFrame,
    probabilistic: bool = False,
    lined_only: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    out["month"] = pd.to_datetime(out["startDate"], errors="coerce", utc=True).dt.to_period("M").astype(str)
    rows = []
    for month, group in out.groupby("month", dropna=False):
        metrics = _metrics_from_predictions(
            group,
            probabilistic=probabilistic,
            lined_only=lined_only,
        )
        metrics["month"] = month
        rows.append(metrics)
    return pd.DataFrame(rows)


def _predict_mean(train_y: np.ndarray, n_rows: int) -> np.ndarray:
    return np.full(n_rows, float(np.mean(train_y)), dtype=np.float32)


def _predict_ridge(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    X_train_raw, y_train = _prepare_point_data(train_df)
    X_test_raw, _ = _prepare_point_data(test_df)
    means = _train_impute_means(X_train_raw)
    X_train = _apply_impute_means(X_train_raw, means)
    X_test = _apply_impute_means(X_test_raw, means)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled).astype(np.float32)

    def _predict_swapped(swapped_feature_df: pd.DataFrame) -> np.ndarray:
        X_swap_raw = swapped_feature_df.values.astype(np.float32)
        X_swap = _apply_impute_means(X_swap_raw, means)
        X_swap_scaled = scaler.transform(X_swap)
        return model.predict(X_swap_scaled).astype(np.float32)

    return _symmetrize_neutral_margin(test_df, pred, _predict_swapped)


def _predict_hgbr(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    X_train_raw, y_train = _prepare_point_data(train_df)
    X_test_raw, _ = _prepare_point_data(test_df)
    means = _train_impute_means(X_train_raw)
    X_train = _apply_impute_means(X_train_raw, means)
    X_test = _apply_impute_means(X_test_raw, means)
    model = HistGradientBoostingRegressor(**HGBR_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(np.float32)

    def _predict_swapped(swapped_feature_df: pd.DataFrame) -> np.ndarray:
        X_swap_raw = swapped_feature_df.values.astype(np.float32)
        X_swap = _apply_impute_means(X_swap_raw, means)
        return model.predict(X_swap).astype(np.float32)

    return _symmetrize_neutral_margin(test_df, pred, _predict_swapped)


def _predict_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, int]:
    inner_train_df, inner_val_df = _split_inner_train_val(train_df)
    X_train_raw, y_train = _prepare_point_data(inner_train_df)
    X_val_raw, y_val = _prepare_point_data(inner_val_df)
    X_test_raw, _ = _prepare_point_data(test_df)

    means = _train_impute_means(X_train_raw)
    X_train = _apply_impute_means(X_train_raw, means)
    X_val = _apply_impute_means(X_val_raw, means)
    X_test = _apply_impute_means(X_test_raw, means)

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    callbacks = [lgb.early_stopping(50, verbose=False)]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=callbacks,
    )

    best_iteration = int(model.best_iteration_ or LGBM_PARAMS["n_estimators"])
    pred = model.predict(X_test, num_iteration=best_iteration).astype(np.float32)

    def _predict_swapped(swapped_feature_df: pd.DataFrame) -> np.ndarray:
        X_swap_raw = swapped_feature_df.values.astype(np.float32)
        X_swap = _apply_impute_means(X_swap_raw, means)
        return model.predict(X_swap, num_iteration=best_iteration).astype(np.float32)

    pred = _symmetrize_neutral_margin(test_df, pred, _predict_swapped)
    return pred, best_iteration


def _train_temporal_mlp(train_df: pd.DataFrame) -> tuple[MLPRegressor, np.ndarray, StandardScaler, int]:
    inner_train_df, inner_val_df = _split_inner_train_val(train_df)
    X_train_raw, y_train = _prepare_point_data(inner_train_df)
    X_val_raw, y_val = _prepare_point_data(inner_val_df)

    means = _train_impute_means(X_train_raw)
    X_train = _apply_impute_means(X_train_raw, means)
    X_val = _apply_impute_means(X_val_raw, means)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    fold_seed = SEED + int(inner_val_df["gameId"].iloc[0]) % 10_000
    _set_seed(fold_seed)

    device = _device()
    model = MLPRegressor(
        input_dim=X_train_scaled.shape[1],
        hidden1=MLP_HP["hidden1"],
        hidden2=MLP_HP["hidden2"],
        dropout=MLP_HP["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MLP_HP["lr"],
        weight_decay=MLP_HP["weight_decay"],
    )

    ds = HoopsDataset(
        X_train_scaled,
        spread=y_train,
        home_win=np.zeros(len(y_train), dtype=np.float32),
    )
    loader = DataLoader(
        ds,
        batch_size=MLP_HP["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(MLP_HP["epochs"]):
        model.train()
        for batch in loader:
            x, spread, _ = [b.to(device) for b in batch]
            optimizer.zero_grad()
            mu, log_sigma = model(x)
            nll, _ = gaussian_nll_loss(mu, log_sigma, spread)
            loss = nll.mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_val, log_sigma_val = model(X_val_t)
            val_nll, _ = gaussian_nll_loss(mu_val, log_sigma_val, y_val_t)
            val_loss = float(val_nll.mean().item())

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint")

    model = model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    return model, means, scaler, best_epoch


def _predict_mlp(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    model, means, scaler, best_epoch = _train_temporal_mlp(train_df)
    X_test_raw, _ = _prepare_point_data(test_df)
    X_test = _apply_impute_means(X_test_raw, means)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    with torch.no_grad():
        mu_t, log_sigma_t = model(X_test_t)
        sigma_t = torch.exp(log_sigma_t).clamp(min=0.5, max=30.0)

    pred = mu_t.numpy().astype(np.float32)
    sigma = sigma_t.numpy().astype(np.float32)

    def _predict_swapped(swapped_feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_swap_raw = swapped_feature_df.values.astype(np.float32)
        X_swap = _apply_impute_means(X_swap_raw, means)
        X_swap_scaled = scaler.transform(X_swap).astype(np.float32)
        X_swap_t = torch.tensor(X_swap_scaled, dtype=torch.float32)
        with torch.no_grad():
            mu_swap_t, log_sigma_swap_t = model(X_swap_t)
            sigma_swap_t = torch.exp(log_sigma_swap_t).clamp(min=0.5, max=30.0)
        return (
            mu_swap_t.numpy().astype(np.float32),
            sigma_swap_t.numpy().astype(np.float32),
        )

    pred, sigma = _symmetrize_neutral_gaussian(test_df, pred, sigma, _predict_swapped)
    return pred, sigma, best_epoch


def _build_prediction_frame(
    model_name: str,
    holdout_season: int,
    holdout_df: pd.DataFrame,
    pred_margin: np.ndarray | pd.Series,
    sigma: np.ndarray | None = None,
) -> pd.DataFrame:
    out = holdout_df.copy()
    out["actual_margin"] = (
        out["homeScore"].astype(float) - out["awayScore"].astype(float)
    )
    out["pred_margin"] = np.asarray(pred_margin, dtype=float)
    out["abs_error"] = np.abs(out["pred_margin"] - out["actual_margin"])
    out["pred_home_win"] = (out["pred_margin"] > 0).astype(int)
    out["model"] = model_name
    out["holdout_season"] = holdout_season
    out["sigma"] = np.asarray(sigma, dtype=float) if sigma is not None else np.nan

    keep = [
        "model",
        "holdout_season",
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
    available = [col for col in keep if col in out.columns]
    return out[available].copy()


def _build_book_spread_benchmark_frame(holdout_season: int, holdout_df: pd.DataFrame) -> pd.DataFrame:
    pred_margin = -holdout_df["book_spread"].to_numpy(dtype=float)
    out = holdout_df.copy()
    out["actual_margin"] = (
        out["homeScore"].astype(float) - out["awayScore"].astype(float)
    )
    out["pred_margin"] = pred_margin
    out["abs_error"] = np.abs(out["pred_margin"] - out["actual_margin"])
    out.loc[out["pred_margin"].isna(), "abs_error"] = np.nan
    out["pred_home_win"] = np.where(out["pred_margin"].notna(), (out["pred_margin"] > 0).astype(int), np.nan)
    out["model"] = BOOK_BENCHMARK_LABEL
    out["holdout_season"] = holdout_season
    out["sigma"] = np.nan
    keep = [
        "model",
        "holdout_season",
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
    available = [col for col in keep if col in out.columns]
    return out[available].copy()


def _save_prediction_frame(output_dir: Path, model_name: str, holdout_season: int, df: pd.DataFrame) -> None:
    pred_dir = output_dir / "predictions" / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pred_dir / f"season_{holdout_season}.parquet", index=False)


def _format_float(value: float) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(_format_float(val))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _write_summary(output_dir: Path, fold_metrics: pd.DataFrame, pooled_metrics: pd.DataFrame) -> None:
    line_meta = _line_selection_metadata()
    lines = [
        "# Canonical Walk-Forward Benchmark v1",
        "",
        "## Protocol",
        "",
        f"- Holdouts: {', '.join(str(x) for x in HOLDOUT_SEASONS)}",
        f"- Excluded seasons: {', '.join(str(x) for x in EXCLUDE_SEASONS)}",
        f"- Features: {EFFICIENCY_SOURCE}{f' ({GOLD_TABLE_NAME})' if GOLD_TABLE_NAME else ''} + adjusted + 53 features (`{ADJ_SUFFIX}`)",
        "- Primary target: homeScore - awayScore",
        f"- External book benchmark: `{line_meta['benchmark_label']}`",
        f"- Uses true closing timestamps: {line_meta['uses_true_closing_timestamps']}",
        f"- Provider preference order: {', '.join(line_meta['provider_preference_order'])}",
        f"- Line selection rule: {line_meta['selection_rule']}",
        "",
        "## Pooled Metrics",
        "",
        _markdown_table(
            pooled_metrics,
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
            ],
        ),
        "",
        "## Fold Metrics",
        "",
        _markdown_table(
            fold_metrics,
            [
                "model",
                "holdout_season",
                "n_games",
                "n_lined",
                "MAE_all",
                "RMSE_all",
                "MedAE_all",
                "WinAcc_all",
                "MAE_lined",
                "BookMAE_lined",
                "DeltaVsBook_MAE",
            ],
        ),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run_benchmark(output_dir: Path, holdout_season: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(SEED)
    folds = _folds(holdout_season)

    config_payload = {
        "benchmark": "canonical_walkforward_v1",
        "seed": SEED,
        "train_start": TRAIN_START,
        "exclude_seasons": EXCLUDE_SEASONS,
        "holdout_seasons": HOLDOUT_SEASONS,
        "feature_config": {
            "no_garbage": NO_GARBAGE,
            "efficiency_source": EFFICIENCY_SOURCE,
            "gold_table_name": GOLD_TABLE_NAME,
            "adj_suffix": ADJ_SUFFIX,
            "feature_order_size": len(config.FEATURE_ORDER),
        },
        "inner_validation": {
            "type": "temporal_tail",
            "val_frac": INNER_VAL_FRAC,
            "sort": ["startDate", "gameId"],
        },
        "external_book_benchmark": _line_selection_metadata(),
        "models": {
            "HomeMarginMean": {},
            "Ridge": {"alpha": RIDGE_ALPHA},
            "HistGradientBoosting": HGBR_PARAMS,
            "LightGBM": LGBM_PARAMS,
            "CurrentMLP": MLP_HP,
            BOOK_BENCHMARK_LABEL: {"external_benchmark": True},
        },
        "folds": folds,
    }
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")

    fold_rows: list[dict] = []
    monthly_rows: list[dict] = []
    predictions_by_model: dict[str, list[pd.DataFrame]] = {name: [] for name in MODEL_ORDER}

    for fold in folds:
        holdout_season = fold["holdout_season"]
        train_seasons = fold["train_seasons"]

        print(f"\n=== Holdout {holdout_season} ===")
        print(f"Train seasons: {train_seasons}")

        train_df = _load_fold_frame(train_seasons)
        holdout_df = _load_fold_frame([holdout_season])
        holdout_df = _attach_book_spread(holdout_df, holdout_season)

        y_train = get_targets(train_df)["spread_home"].values.astype(np.float32)

        model_frames: dict[str, pd.DataFrame] = {}

        mean_pred = _predict_mean(y_train, len(holdout_df))
        model_frames["HomeMarginMean"] = _build_prediction_frame(
            "HomeMarginMean", holdout_season, holdout_df, mean_pred
        )

        ridge_pred = _predict_ridge(train_df, holdout_df)
        model_frames["Ridge"] = _build_prediction_frame(
            "Ridge", holdout_season, holdout_df, ridge_pred
        )

        hgbr_pred = _predict_hgbr(train_df, holdout_df)
        model_frames["HistGradientBoosting"] = _build_prediction_frame(
            "HistGradientBoosting", holdout_season, holdout_df, hgbr_pred
        )

        lgb_pred, lgb_best_iteration = _predict_lightgbm(train_df, holdout_df)
        model_frames["LightGBM"] = _build_prediction_frame(
            "LightGBM", holdout_season, holdout_df, lgb_pred
        )

        mlp_pred, mlp_sigma, best_epoch = _predict_mlp(train_df, holdout_df)
        model_frames["CurrentMLP"] = _build_prediction_frame(
            "CurrentMLP", holdout_season, holdout_df, mlp_pred, sigma=mlp_sigma
        )

        model_frames[BOOK_BENCHMARK_LABEL] = _build_book_spread_benchmark_frame(holdout_season, holdout_df)

        for model_name in MODEL_ORDER:
            pred_df = model_frames[model_name]
            predictions_by_model[model_name].append(pred_df)
            _save_prediction_frame(output_dir, model_name, holdout_season, pred_df)

            probabilistic = model_name == "CurrentMLP"
            lined_only = model_name == BOOK_BENCHMARK_LABEL
            metrics = _metrics_from_predictions(
                pred_df,
                probabilistic=probabilistic,
                lined_only=lined_only,
            )
            metrics["model"] = model_name
            metrics["holdout_season"] = holdout_season
            if model_name == "CurrentMLP":
                metrics["best_epoch"] = best_epoch
            elif model_name == "LightGBM":
                metrics["best_epoch"] = lgb_best_iteration
            else:
                metrics["best_epoch"] = np.nan
            fold_rows.append(metrics)

            monthly = _monthly_metrics(
                pred_df,
                probabilistic=probabilistic,
                lined_only=lined_only,
            )
            monthly["model"] = model_name
            monthly["holdout_season"] = holdout_season
            monthly_rows.append(monthly)

            print(
                f"  {model_name:>20}: "
                f"MAE_all={_format_float(metrics['MAE_all'])} "
                f"MAE_lined={_format_float(metrics['MAE_lined'])} "
                f"DeltaVsBook={_format_float(metrics['DeltaVsBook_MAE'])}"
            )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics = fold_metrics.sort_values(["holdout_season", "model"]).reset_index(drop=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    monthly_metrics = pd.concat(monthly_rows, ignore_index=True)
    monthly_metrics = monthly_metrics.sort_values(["holdout_season", "model", "month"]).reset_index(drop=True)
    monthly_metrics.to_csv(output_dir / "monthly_metrics.csv", index=False)

    pooled_rows = []
    for model_name in MODEL_ORDER:
        pooled_df = pd.concat(predictions_by_model[model_name], ignore_index=True)
        probabilistic = model_name == "CurrentMLP"
        lined_only = model_name == BOOK_BENCHMARK_LABEL
        metrics = _metrics_from_predictions(
            pooled_df,
            probabilistic=probabilistic,
            lined_only=lined_only,
        )
        metrics["model"] = model_name
        pooled_rows.append(metrics)
    pooled_metrics = pd.DataFrame(pooled_rows)
    pooled_metrics = pooled_metrics[[
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
        "GaussianNLL",
        "MeanAbsZ",
        "Coverage_1sigma",
        "Coverage_2sigma",
    ]]
    pooled_metrics.to_csv(output_dir / "pooled_metrics.csv", index=False)

    _write_summary(output_dir, fold_metrics, pooled_metrics)
    print(f"\nSaved benchmark artifacts to {output_dir}")


def main() -> None:
    global EFFICIENCY_SOURCE, GOLD_TABLE_NAME
    parser = argparse.ArgumentParser(description="Run canonical walk-forward benchmark v1.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Artifact output directory",
    )
    parser.add_argument(
        "--holdout-season",
        type=int,
        default=None,
        help="Run only one canonical holdout season (must be one of 2019, 2020, 2022, 2023, 2024, 2025)",
    )
    parser.add_argument(
        "--efficiency-source",
        type=str,
        default=EFFICIENCY_SOURCE,
        choices=["gold", "torvik"],
        help="Efficiency source for this benchmark run",
    )
    parser.add_argument(
        "--gold-table-name",
        type=str,
        default=None,
        help="Optional explicit gold ratings table when --efficiency-source=gold",
    )
    args = parser.parse_args()
    EFFICIENCY_SOURCE = args.efficiency_source
    GOLD_TABLE_NAME = args.gold_table_name
    run_benchmark(args.output_dir, holdout_season=args.holdout_season)


if __name__ == "__main__":
    main()

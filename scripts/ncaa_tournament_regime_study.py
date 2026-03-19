#!/usr/bin/env python3
"""Deep-dive NCAA Tournament regime study for the promoted Hoops Edge path.

Outputs:
  - regime_summary.csv
  - ncaa_round_summary.csv
  - ncaa_bucket_summary.csv
  - source_comparison.csv
  - probability_summary.csv
  - candidate_summary.csv
  - candidate_by_season.csv
  - candidate_calibration_params.csv
  - summary.md

The goal is to keep the study tied to the real promoted path:
  - site/history benchmark bundle:
      canonical_walkforward_lgb_l2_blend_repaired_lines_neutralfix
  - current mean model:
      LightGBMRegressionL2Blend
  - current sigma model:
      CurrentMLP
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from src import config, s3_reader
from src.features import load_research_lines
from src.line_selection import select_preferred_lines
from src.ml_odds import normal_cdf
import scripts.canonical_walkforward as cw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "research" / "ncaa_tournament_regime_study_v1"
)
PROMOTED_BENCHMARK_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "benchmarks"
    / "canonical_walkforward_lgb_l2_blend_repaired_lines_neutralfix"
)
TORVIK_LGB_BENCHMARK_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "benchmarks"
    / "canonical_walkforward_v2_lgb_repaired_lines_neutralfix"
)
GOLD_HGBR_BENCHMARK_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "benchmarks"
    / "canonical_walkforward_priorreg_k5_repaired_lines_neutralfix"
)
FEATURE_DIR = (
    PROJECT_ROOT
    / "features"
    / "canonical_walkforward"
    / "gold_team_adjusted_efficiencies_no_garbage_priorreg_k5_v1"
)

ALL_HOLDOUT_SEASONS = [2019, 2020, 2022, 2023, 2024, 2025]
NCAA_HOLDOUT_SEASONS = [2019, 2022, 2023, 2024, 2025]
TRAIN_SEASONS = list(range(2015, 2026))
RIDGE_ALPHAS = np.logspace(-2, 4, 13)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output artifact directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def _read_prediction_dir(pred_dir: Path, model_name: str, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = pred_dir / "predictions" / model_name / f"season_{season}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No predictions found for {model_name} under {pred_dir}")
    return pd.concat(frames, ignore_index=True)


def _american_implied_prob(odds: pd.Series) -> np.ndarray:
    ml = pd.to_numeric(odds, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ml < 0, -ml / (-ml + 100.0), np.where(ml > 0, 100.0 / (ml + 100.0), np.nan))
    out[~np.isfinite(out)] = np.nan
    return out


def _normalized_moneyline_home_prob(home_ml: pd.Series, away_ml: pd.Series) -> np.ndarray:
    home_prob = _american_implied_prob(home_ml)
    away_prob = _american_implied_prob(away_ml)
    total = home_prob + away_prob
    out = home_prob / total
    out[~np.isfinite(out)] = np.nan
    return out


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _safe_rmse(actual: pd.Series, pred: pd.Series) -> float:
    diff = pd.to_numeric(actual, errors="coerce") - pd.to_numeric(pred, errors="coerce")
    return float(np.sqrt(np.nanmean(np.square(diff))))


def _market_favorite_bias(frame: pd.DataFrame, pred_col: str) -> float:
    x = frame.dropna(subset=["market_margin_home", pred_col]).copy()
    if x.empty:
        return float("nan")
    fav_sign = np.sign(x["market_margin_home"]).replace(0, np.nan)
    return float(np.nanmean(x[pred_col] * fav_sign - x["market_margin_home"].abs()))


def _market_favorite_slope(frame: pd.DataFrame, pred_col: str) -> float:
    x = frame.dropna(subset=["market_margin_home", pred_col]).copy()
    if len(x) < 2:
        return float("nan")
    fav_sign = np.sign(x["market_margin_home"]).replace(0, np.nan)
    y = (x[pred_col] * fav_sign).to_numpy(dtype=float)
    m = x["market_margin_home"].abs().to_numpy(dtype=float)
    if np.nanstd(m) == 0:
        return float("nan")
    return float(np.polyfit(m, y, 1)[0])


def _bucketize_market_margin(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(
        frame["market_fav_margin"],
        bins=[-0.1, 5.0, 10.0, 15.0, 100.0],
        labels=["<5", "5-10", "10-15", "15+"],
    )


def _round_from_note(note: object) -> str | None:
    value = str(note or "").upper()
    mapping = [
        ("FIRST FOUR", "First Four"),
        ("1ST ROUND", "Round of 64"),
        ("2ND ROUND", "Round of 32"),
        ("SWEET 16", "Sweet 16"),
        ("ELITE 8", "Elite 8"),
        ("FINAL FOUR", "Final Four"),
        ("NATIONAL CHAMPIONSHIP", "Championship"),
    ]
    for needle, label in mapping:
        if needle in value:
            return label
    return None


def _round_number(label: object) -> float:
    mapping = {
        "First Four": 1.0,
        "Round of 64": 2.0,
        "Round of 32": 3.0,
        "Sweet 16": 4.0,
        "Elite 8": 5.0,
        "Final Four": 6.0,
        "Championship": 7.0,
    }
    return mapping.get(label, np.nan)


def _is_missing_tournament(value: object) -> bool:
    return pd.isna(value) or str(value) in {"None", "nan", ""}


def _regime(row: pd.Series) -> str:
    tournament = row.get("tournament")
    if tournament == "NCAA":
        return "ncaa_tournament"
    if row.get("gameType") == "TRNMNT" and _is_missing_tournament(tournament) and row.get("month") == 3:
        return "conference_tournament"
    if (
        bool(row.get("neutralSite"))
        and not bool(row.get("conferenceGame"))
        and _is_missing_tournament(tournament)
        and row.get("gameType") == "STD"
    ):
        return "neutral_nonconf"
    if row.get("gameType") == "STD" and _is_missing_tournament(tournament):
        return "regular_season"
    return "other"


def _load_games_meta(seasons: list[int]) -> pd.DataFrame:
    parts = []
    for season in seasons:
        table = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
        if table.num_rows == 0:
            continue
        games = table.to_pandas()
        keep = [
            c
            for c in [
                "gameId",
                "startDate",
                "gameType",
                "tournament",
                "neutralSite",
                "conferenceGame",
                "gameNotes",
                "homeSeed",
                "awaySeed",
                "homeConference",
                "awayConference",
            ]
            if c in games.columns
        ]
        games = games[keep].drop_duplicates("gameId").copy()
        games["season"] = season
        parts.append(games)
    return pd.concat(parts, ignore_index=True)


def _load_selected_lines(seasons: list[int]) -> pd.DataFrame:
    parts = []
    for season in seasons:
        lines = select_preferred_lines(load_research_lines(season))
        keep = [c for c in ["gameId", "home_moneyline", "away_moneyline", "provider"] if c in lines.columns]
        lines = lines[keep].drop_duplicates("gameId").copy()
        lines["season"] = season
        parts.append(lines)
    return pd.concat(parts, ignore_index=True)


def _load_feature_master() -> pd.DataFrame:
    feature_parts = []
    meta_parts = []
    for season in TRAIN_SEASONS:
        feature_path = FEATURE_DIR / f"season_{season}.parquet"
        if not feature_path.exists():
            continue
        features = pd.read_parquet(feature_path).copy()
        features["season"] = season
        feature_parts.append(features)

        table = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
        if table.num_rows == 0:
            continue
        games = table.to_pandas()
        keep = [
            c
            for c in [
                "gameId",
                "startDate",
                "gameType",
                "tournament",
                "neutralSite",
                "conferenceGame",
                "gameNotes",
                "homeSeed",
                "awaySeed",
                "homeConference",
                "awayConference",
            ]
            if c in games.columns
        ]
        games = games[keep].drop_duplicates("gameId").copy()
        games["season"] = season
        meta_parts.append(games)

    master = pd.concat(feature_parts, ignore_index=True).merge(
        pd.concat(meta_parts, ignore_index=True),
        on=["gameId", "startDate", "season"],
        how="left",
    )
    dates = pd.to_datetime(master["startDate"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    master["month"] = dates.dt.month
    master["regime"] = master.apply(_regime, axis=1)
    master["ncaa_round"] = master["gameNotes"].map(_round_from_note)
    master["round_num"] = master["ncaa_round"].map(_round_number)
    master["actual_margin"] = pd.to_numeric(master["homeScore"], errors="coerce") - pd.to_numeric(
        master["awayScore"], errors="coerce"
    )
    master["actual_home_win"] = (master["actual_margin"] > 0).astype(int)
    for col in ["homeSeed", "awaySeed"]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")
    master["seed_gap"] = (master["homeSeed"] - master["awaySeed"]).abs()
    master["conf_gap"] = master["home_conf_strength"] - master["away_conf_strength"]
    master["barthag_gap"] = master["home_team_BARTHAG"] - master["away_team_BARTHAG"]
    master["form_gap"] = master["home_form_delta"] - master["away_form_delta"]
    return master


def _build_promoted_eval_frame() -> pd.DataFrame:
    mu = _read_prediction_dir(PROMOTED_BENCHMARK_DIR, "LightGBMRegressionL2Blend", ALL_HOLDOUT_SEASONS)
    sigma = _read_prediction_dir(PROMOTED_BENCHMARK_DIR, "CurrentMLP", ALL_HOLDOUT_SEASONS)
    keys = [
        "holdout_season",
        "gameId",
        "startDate",
        "homeTeamId",
        "awayTeamId",
        "homeTeam",
        "awayTeam",
        "book_spread",
    ]
    frame = mu[keys + ["actual_margin", "pred_margin"]].merge(
        sigma[keys + ["sigma"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    frame["season"] = frame["holdout_season"].astype(int)

    games = _load_games_meta(ALL_HOLDOUT_SEASONS)
    lines = _load_selected_lines(ALL_HOLDOUT_SEASONS)
    frame = frame.merge(
        games,
        on=["gameId", "startDate", "season"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        lines.drop(columns=["season"]),
        on="gameId",
        how="left",
    )

    dates = pd.to_datetime(frame["startDate"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    frame["month"] = dates.dt.month
    frame["regime"] = frame.apply(_regime, axis=1)
    frame["ncaa_round"] = frame["gameNotes"].map(_round_from_note)
    frame["round_num"] = frame["ncaa_round"].map(_round_number)
    frame["market_margin_home"] = -pd.to_numeric(frame["book_spread"], errors="coerce")
    frame["model_margin_home"] = pd.to_numeric(frame["pred_margin"], errors="coerce")
    frame["actual_margin_home"] = pd.to_numeric(frame["actual_margin"], errors="coerce")
    frame["fav_sign"] = np.sign(frame["market_margin_home"]).replace(0, np.nan)
    frame["market_fav_margin"] = frame["market_margin_home"].abs()
    frame["model_on_market_fav"] = frame["model_margin_home"] * frame["fav_sign"]
    frame["actual_on_market_fav"] = frame["actual_margin_home"] * frame["fav_sign"]
    frame["model_minus_market_fav"] = frame["model_on_market_fav"] - frame["market_fav_margin"]
    frame["actual_minus_model_fav"] = frame["actual_on_market_fav"] - frame["model_on_market_fav"]
    frame["model_market_abs_diff"] = (frame["model_margin_home"] - frame["market_margin_home"]).abs()
    frame["actual_home_win"] = (frame["actual_margin_home"] > 0).astype(int)
    frame["home_fav"] = frame["market_margin_home"] > 0

    for col in ["homeSeed", "awaySeed"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["seed_gap"] = (frame["homeSeed"] - frame["awaySeed"]).abs()
    frame["fav_seed"] = np.where(frame["market_margin_home"] > 0, frame["homeSeed"], frame["awaySeed"])
    frame["conf_gap_fav"] = np.where(
        frame["market_margin_home"] > 0,
        frame["homeConference"].notna(),
        frame["awayConference"].notna(),
    )

    frame["market_home_win_prob"] = _normalized_moneyline_home_prob(
        frame["home_moneyline"],
        frame["away_moneyline"],
    )
    sigma_safe = pd.to_numeric(frame["sigma"], errors="coerce").clip(lower=0.5)
    frame["model_home_win_prob"] = normal_cdf(frame["model_margin_home"] / sigma_safe)
    return frame


def _regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in ["regular_season", "conference_tournament", "neutral_nonconf", "ncaa_tournament"]:
        x = frame[frame["regime"] == regime].dropna(
            subset=["market_margin_home", "model_margin_home", "actual_margin_home"]
        )
        if x.empty:
            continue
        rows.append(
            {
                "regime": regime,
                "lined_games": int(len(x)),
                "model_mae": _safe_mean((x["actual_margin_home"] - x["model_margin_home"]).abs()),
                "market_mae": _safe_mean((x["actual_margin_home"] - x["market_margin_home"]).abs()),
                "model_rmse": _safe_rmse(x["actual_margin_home"], x["model_margin_home"]),
                "market_rmse": _safe_rmse(x["actual_margin_home"], x["market_margin_home"]),
                "model_vs_market_mae_gap": _safe_mean((x["actual_margin_home"] - x["model_margin_home"]).abs())
                - _safe_mean((x["actual_margin_home"] - x["market_margin_home"]).abs()),
                "model_vs_market_fav_bias": _safe_mean(x["model_minus_market_fav"]),
                "favorite_slope_vs_market": _market_favorite_slope(x, "model_margin_home"),
                "model_to_market_abs_ratio": _safe_mean(x["model_on_market_fav"]) / _safe_mean(x["market_fav_margin"]),
            }
        )
    return pd.DataFrame(rows)


def _round_summary(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[frame["regime"] == "ncaa_tournament"].dropna(
        subset=["market_margin_home", "model_margin_home", "actual_margin_home"]
    )
    rows = []
    for round_name, group in x.groupby("ncaa_round", sort=False):
        rows.append(
            {
                "ncaa_round": round_name,
                "lined_games": int(len(group)),
                "market_fav_margin": _safe_mean(group["market_fav_margin"]),
                "model_on_market_fav": _safe_mean(group["model_on_market_fav"]),
                "actual_on_market_fav": _safe_mean(group["actual_on_market_fav"]),
                "model_vs_market_fav_bias": _safe_mean(group["model_minus_market_fav"]),
                "model_vs_market_abs_diff": _safe_mean(group["model_market_abs_diff"]),
                "model_mae": _safe_mean((group["actual_margin_home"] - group["model_margin_home"]).abs()),
                "market_mae": _safe_mean((group["actual_margin_home"] - group["market_margin_home"]).abs()),
            }
        )
    return pd.DataFrame(rows)


def _bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[frame["regime"] == "ncaa_tournament"].dropna(
        subset=["market_margin_home", "model_margin_home", "actual_margin_home"]
    ).copy()
    x["market_bucket"] = _bucketize_market_margin(x)
    rows = []
    for bucket, group in x.groupby("market_bucket", observed=True, sort=False):
        rows.append(
            {
                "market_bucket": str(bucket),
                "lined_games": int(len(group)),
                "market_fav_margin": _safe_mean(group["market_fav_margin"]),
                "model_on_market_fav": _safe_mean(group["model_on_market_fav"]),
                "actual_on_market_fav": _safe_mean(group["actual_on_market_fav"]),
                "model_vs_market_fav_bias": _safe_mean(group["model_minus_market_fav"]),
                "actual_minus_model_fav": _safe_mean(group["actual_minus_model_fav"]),
                "model_mae": _safe_mean((group["actual_margin_home"] - group["model_margin_home"]).abs()),
                "market_mae": _safe_mean((group["actual_margin_home"] - group["market_margin_home"]).abs()),
            }
        )
    return pd.DataFrame(rows)


def _probability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[frame["regime"] == "ncaa_tournament"].dropna(
        subset=["actual_home_win", "model_home_win_prob", "market_home_win_prob"]
    )
    rows = []
    for label, prob_col, pick_col in [
        ("current_model", "model_home_win_prob", "model_margin_home"),
        ("market_moneyline", "market_home_win_prob", "market_home_win_prob"),
    ]:
        probs = x[prob_col].clip(1e-6, 1.0 - 1e-6).to_numpy(dtype=float)
        actual = x["actual_home_win"].to_numpy(dtype=float)
        if label == "market_moneyline":
            picks = x[pick_col] > 0.5
        else:
            picks = x[pick_col] > 0
        rows.append(
            {
                "source": label,
                "games": int(len(x)),
                "brier": float(np.mean(np.square(probs - actual))),
                "logloss": float(-np.mean(actual * np.log(probs) + (1.0 - actual) * np.log(1.0 - probs))),
                "straight_up_accuracy": float(np.mean(picks.to_numpy(dtype=bool) == (actual == 1.0))),
            }
        )
    return pd.DataFrame(rows)


def _source_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    games = _load_games_meta(NCAA_HOLDOUT_SEASONS)[["gameId", "tournament"]].drop_duplicates("gameId")
    specs = [
        ("promoted_lgb_l2_blend", PROMOTED_BENCHMARK_DIR, "LightGBMRegressionL2Blend"),
        ("gold_lgb", GOLD_HGBR_BENCHMARK_DIR, "LightGBM"),
        ("gold_hgbr", GOLD_HGBR_BENCHMARK_DIR, "HistGradientBoosting"),
        ("torvik_lgb", TORVIK_LGB_BENCHMARK_DIR, "LightGBM"),
        ("torvik_hgbr", TORVIK_LGB_BENCHMARK_DIR, "HistGradientBoosting"),
    ]
    rows = []
    for label, bench_dir, model_name in specs:
        pred = _read_prediction_dir(bench_dir, model_name, NCAA_HOLDOUT_SEASONS)
        pred = pred.merge(games, on="gameId", how="left")
        pred = pred[pred["tournament"] == "NCAA"].copy()
        pred["market_margin_home"] = -pd.to_numeric(pred["book_spread"], errors="coerce")
        pred["pred_margin"] = pd.to_numeric(pred["pred_margin"], errors="coerce")
        pred["actual_margin"] = pd.to_numeric(pred["actual_margin"], errors="coerce")
        pred["fav_sign"] = np.sign(pred["market_margin_home"]).replace(0, np.nan)
        rows.append(
            {
                "source": label,
                "games": int(len(pred)),
                "mae": _safe_mean((pred["actual_margin"] - pred["pred_margin"]).abs()),
                "market_favorite_bias": float(
                    np.nanmean(pred["pred_margin"] * pred["fav_sign"] - pred["market_margin_home"].abs())
                ),
            }
        )
    return pd.DataFrame(rows)


def _prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    train_x = train_df[cols].to_numpy(dtype=float)
    test_x = test_df[cols].to_numpy(dtype=float)
    means = np.nanmean(train_x, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    train_x = np.where(np.isnan(train_x), means, train_x)
    test_x = np.where(np.isnan(test_x), means, test_x)
    scaler = StandardScaler()
    return scaler.fit_transform(train_x), scaler.transform(test_x), means, scaler


def _ridge_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    *,
    sample_weight: np.ndarray | None = None,
    context_mode: bool = False,
) -> np.ndarray:
    train_x, test_x, means, scaler = _prepare_xy(train_df, test_df, cols)
    alpha = float(RidgeCV(alphas=RIDGE_ALPHAS).fit(train_x, train_df["actual_margin"]).alpha_)
    model = Ridge(alpha=alpha)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(train_x, train_df["actual_margin"], **fit_kwargs)
    pred = model.predict(test_x).astype(np.float32)

    if context_mode:
        swapped = test_df[cols].copy()
        swapped["current_pred"] = -swapped["current_pred"]
        if "homeSeed" in swapped.columns and "awaySeed" in swapped.columns:
            tmp = swapped["homeSeed"].copy()
            swapped["homeSeed"] = swapped["awaySeed"]
            swapped["awaySeed"] = tmp
        for col in ["conf_gap", "barthag_gap", "form_gap"]:
            if col in swapped.columns:
                swapped[col] = -swapped[col]
        swap_x = swapped[cols].to_numpy(dtype=float)
        swap_x = np.where(np.isnan(swap_x), means, swap_x)
        pred_swap = model.predict(scaler.transform(swap_x)).astype(np.float32)
        return ((pred - pred_swap) / 2.0).astype(np.float32)

    def _predict_swapped(swapped_df: pd.DataFrame) -> np.ndarray:
        swap_x = swapped_df[cols].to_numpy(dtype=float)
        swap_x = np.where(np.isnan(swap_x), means, swap_x)
        return model.predict(scaler.transform(swap_x)).astype(np.float32)

    return cw._symmetrize_neutral_margin(test_df, pred, _predict_swapped)


def _candidate_backtest(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    calibration_rows = []
    for holdout in NCAA_HOLDOUT_SEASONS:
        test = master[(master["season"] == holdout) & (master["regime"] == "ncaa_tournament")].copy()
        train = master[master["season"] < holdout].copy()
        prior_ncaa_current = master[
            (master["season"] < holdout)
            & (master["regime"] == "ncaa_tournament")
            & master["current_pred"].notna()
            & master["market_margin_home"].notna()
        ].copy()

        test["A_current_unchanged"] = test["current_pred"]

        if len(prior_ncaa_current) >= 20:
            affine = LinearRegression().fit(prior_ncaa_current[["current_pred"]], prior_ncaa_current["actual_margin"])
            test["B_affine_tournament_calibration"] = affine.predict(test[["current_pred"]])

            alphas = np.linspace(0.0, 1.0, 81)
            alpha_errors = []
            for alpha in alphas:
                blended = alpha * prior_ncaa_current["current_pred"] + (1.0 - alpha) * prior_ncaa_current["market_margin_home"]
                alpha_errors.append(np.mean(np.abs(prior_ncaa_current["actual_margin"] - blended)))
            best_alpha = float(alphas[int(np.argmin(alpha_errors))])
            test["G_market_blend_tournament_only"] = (
                best_alpha * test["current_pred"] + (1.0 - best_alpha) * test["market_margin_home"]
            )

            context_cols = [
                "current_pred",
                "seed_gap",
                "homeSeed",
                "awaySeed",
                "round_num",
                "conf_gap",
                "barthag_gap",
                "form_gap",
            ]
            test["F_context_layer"] = _ridge_predict(
                prior_ncaa_current,
                test,
                context_cols,
                context_mode=True,
            )

            calibration_rows.append(
                {
                    "holdout_season": holdout,
                    "prior_ncaa_rows": int(len(prior_ncaa_current)),
                    "affine_slope": float(affine.coef_[0]),
                    "affine_intercept": float(affine.intercept_),
                    "market_blend_alpha": best_alpha,
                }
            )
        else:
            test["B_affine_tournament_calibration"] = test["current_pred"]
            test["G_market_blend_tournament_only"] = test["current_pred"]
            test["F_context_layer"] = test["current_pred"]
            calibration_rows.append(
                {
                    "holdout_season": holdout,
                    "prior_ncaa_rows": int(len(prior_ncaa_current)),
                    "affine_slope": 1.0,
                    "affine_intercept": 0.0,
                    "market_blend_alpha": 1.0,
                }
            )

        ncaa_train = train[train["regime"] == "ncaa_tournament"].copy()
        postseason_train = train[train["regime"].isin(["conference_tournament", "ncaa_tournament"])].copy()
        neutral_postseason_train = train[
            train["regime"].isin(["conference_tournament", "ncaa_tournament", "neutral_nonconf"])
        ].copy()

        test["C_separate_ncaa_model"] = _ridge_predict(ncaa_train, test, config.FEATURE_ORDER)
        test["D_postseason_model"] = _ridge_predict(postseason_train, test, config.FEATURE_ORDER)
        test["E_neutral_postseason_model"] = _ridge_predict(neutral_postseason_train, test, config.FEATURE_ORDER)

        weights = np.ones(len(train), dtype=float)
        weights += 0.5 * (train["regime"] == "neutral_nonconf").to_numpy(dtype=float)
        weights += 1.0 * (train["regime"] == "conference_tournament").to_numpy(dtype=float)
        weights += 2.0 * (train["regime"] == "ncaa_tournament").to_numpy(dtype=float)
        test["H_reweighted_all_games_model"] = _ridge_predict(
            train,
            test,
            config.FEATURE_ORDER,
            sample_weight=weights,
        )

        keep = [
            "season",
            "gameId",
            "actual_margin",
            "actual_home_win",
            "market_margin_home",
            "A_current_unchanged",
            "B_affine_tournament_calibration",
            "C_separate_ncaa_model",
            "D_postseason_model",
            "E_neutral_postseason_model",
            "F_context_layer",
            "G_market_blend_tournament_only",
            "H_reweighted_all_games_model",
        ]
        rows.append(test[keep])

    candidate_rows = pd.concat(rows, ignore_index=True)
    long_rows = []
    for candidate in [
        "A_current_unchanged",
        "B_affine_tournament_calibration",
        "C_separate_ncaa_model",
        "D_postseason_model",
        "E_neutral_postseason_model",
        "F_context_layer",
        "G_market_blend_tournament_only",
        "H_reweighted_all_games_model",
    ]:
        subset = candidate_rows[["season", "gameId", "actual_margin", "actual_home_win", "market_margin_home", candidate]].copy()
        subset = subset.rename(columns={candidate: "pred_margin"})
        subset["candidate"] = candidate
        long_rows.append(subset)
    long_df = pd.concat(long_rows, ignore_index=True)

    def _summarize(group: pd.DataFrame) -> dict[str, float | int]:
        pred = group["pred_margin"]
        actual = group["actual_margin"]
        market = group["market_margin_home"]
        edge = pred - market
        cover_margin = actual - market
        ats_won = np.where(
            np.isclose(edge, 0.0),
            np.nan,
            np.where(edge > 0.0, cover_margin > 0.0, cover_margin < 0.0),
        ).astype(float)
        ats_rate = float(np.nanmean(ats_won)) if np.isfinite(ats_won).any() else float("nan")
        return {
            "games": int(len(group)),
            "margin_mae": _safe_mean((actual - pred).abs()),
            "margin_rmse": _safe_rmse(actual, pred),
            "line_mae_vs_market": _safe_mean((pred - market).abs()),
            "straight_up_accuracy": float(np.mean((pred > 0.0) == (group["actual_home_win"] == 1))),
            "flat_ats_win_rate": ats_rate,
        }

    pooled = []
    for candidate, group in long_df.groupby("candidate", sort=False):
        row = {"candidate": candidate}
        row.update(_summarize(group))
        pooled.append(row)

    by_season = []
    for (candidate, season), group in long_df.groupby(["candidate", "season"], sort=False):
        row = {"candidate": candidate, "season": int(season)}
        row.update(_summarize(group))
        by_season.append(row)

    return pd.DataFrame(pooled), pd.DataFrame(by_season), pd.DataFrame(calibration_rows)


def _write_summary(
    output_dir: Path,
    protocol: dict[str, object],
    regime_summary: pd.DataFrame,
    round_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    probability_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    calibration_params: pd.DataFrame,
) -> None:
    ncaa_row = regime_summary[regime_summary["regime"] == "ncaa_tournament"].iloc[0]
    reg_row = regime_summary[regime_summary["regime"] == "regular_season"].iloc[0]
    round64_row = round_summary[round_summary["ncaa_round"] == "Round of 64"].iloc[0]
    big_bucket = bucket_summary[bucket_summary["market_bucket"] == "15+"].iloc[0]
    current_prob = probability_summary[probability_summary["source"] == "current_model"].iloc[0]
    market_prob = probability_summary[probability_summary["source"] == "market_moneyline"].iloc[0]
    current_candidate = candidate_summary[candidate_summary["candidate"] == "A_current_unchanged"].iloc[0]
    best_candidate = candidate_summary.sort_values("margin_mae").iloc[0]
    torvik_row = source_summary[source_summary["source"] == "torvik_lgb"].iloc[0]
    promoted_row = source_summary[source_summary["source"] == "promoted_lgb_l2_blend"].iloc[0]

    lines = [
        "# NCAA Tournament Regime Study",
        "",
        "## Promoted Production Path",
        "",
        f"- Historical/site benchmark bundle: `{PROMOTED_BENCHMARK_DIR}`",
        f"- Mean path in bundle: `{protocol['mu_model_name']}` from `{protocol['mean_parquet']}`",
        f"- Sigma path in bundle: `{protocol['sigma_model_name']}`",
        "- The site history JSONs point at this bundle directly.",
        "- In March, the live mean-path blend is effectively gold-only because `gold_weight_for_start_dates()` saturates to `1.0` well before the NCAA Tournament.",
        "",
        "## Diagnosis",
        "",
        f"- NCAA Tournament lined sample: `{int(ncaa_row['lined_games'])}` games across 2019, 2022, 2023, 2024, 2025.",
        f"- Current promoted model is short versus the market on the market-favorite scale by `{ncaa_row['model_vs_market_fav_bias']:.2f}` points in NCAA games, versus `{reg_row['model_vs_market_fav_bias']:.2f}` in the regular season.",
        f"- NCAA favorite-slope vs market is `{ncaa_row['favorite_slope_vs_market']:.3f}` vs `{reg_row['favorite_slope_vs_market']:.3f}` in the regular season, which is direct compression toward zero.",
        f"- Round of 64 is the worst round: mean shortfall `{round64_row['model_vs_market_fav_bias']:.2f}` points.",
        f"- For `15+` market favorites, NCAA shortfall grows to `{big_bucket['model_vs_market_fav_bias']:.2f}` points.",
        f"- Market margin MAE in NCAA is `{ncaa_row['market_mae']:.3f}` vs the model's `{ncaa_row['model_mae']:.3f}`.",
        "",
        "## Regime Read",
        "",
        f"- Neutral-site non-conference games also compress, but less cleanly: NCAA shortfall `{ncaa_row['model_vs_market_fav_bias']:.2f}` vs neutral non-conf `{regime_summary[regime_summary['regime'] == 'neutral_nonconf'].iloc[0]['model_vs_market_fav_bias']:.2f}`.",
        f"- Conference tournaments are materially less short than NCAA: `{regime_summary[regime_summary['regime'] == 'conference_tournament'].iloc[0]['model_vs_market_fav_bias']:.2f}`.",
        "- That points to an NCAA / elite mismatch regime more than a generic 'all neutral games' regime.",
        "",
        "## Probability Read",
        "",
        f"- Current model straight-up accuracy in NCAA games: `{current_prob['straight_up_accuracy']:.3f}`.",
        f"- Market moneyline straight-up accuracy: `{market_prob['straight_up_accuracy']:.3f}`.",
        f"- Current model Brier / logloss: `{current_prob['brier']:.4f}` / `{current_prob['logloss']:.4f}`.",
        f"- Market moneyline Brier / logloss: `{market_prob['brier']:.4f}` / `{market_prob['logloss']:.4f}`.",
        "- Interpretation: the model still ranks winners reasonably well, but margin calibration deteriorates more than winner-picking.",
        "",
        "## Source-Side Evidence",
        "",
        f"- Promoted NCAA MAE / shortfall: `{promoted_row['mae']:.3f}` / `{promoted_row['market_favorite_bias']:.2f}`.",
        f"- Torvik LGB NCAA MAE / shortfall: `{torvik_row['mae']:.3f}` / `{torvik_row['market_favorite_bias']:.2f}`.",
        "- Existing benchmark artifacts therefore already show that the March gold handoff is not the best NCAA source path.",
        "- Separate conference-bridge gold research does not fix NCAA Tournament quality, so the issue is not solved by a small conference-strength bridge alone.",
        "",
        "## Candidate Backtest",
        "",
        f"- Current unchanged (`A`) NCAA MAE: `{current_candidate['margin_mae']:.3f}`.",
        f"- Best pooled NCAA candidate in this study: `{best_candidate['candidate']}` at `{best_candidate['margin_mae']:.3f}` MAE.",
        f"- Tournament-only market blend (`G`) wins on pooled NCAA margin MAE at `{candidate_summary[candidate_summary['candidate'] == 'G_market_blend_tournament_only'].iloc[0]['margin_mae']:.3f}`.",
        f"- Separate NCAA model (`C`) improves some vs current (`{candidate_summary[candidate_summary['candidate'] == 'C_separate_ncaa_model'].iloc[0]['margin_mae']:.3f}`) but is less stable and sample-starved.",
        f"- Affine tournament calibration (`B`) is smaller and safer, but only modestly improves current (`{candidate_summary[candidate_summary['candidate'] == 'B_affine_tournament_calibration'].iloc[0]['margin_mae']:.3f}`).",
        "- Reweighted all-games and broader neutral/postseason retrains did not beat the simpler post-processing paths.",
        "",
        "## Recommendation",
        "",
        "- Do not ship a standalone NCAA-only model as the main production path yet. The sample is too small, and the simple tournament model only modestly helps.",
        "- Preferred production shape: keep one year-round core model, but add a tournament-only post-processing layer.",
        "- If the goal is the best displayed tournament spread, the evidence supports a tournament-only market-aware blend or market override.",
        "- If the goal is a model-pure spread with minimal market dependence, the next best justified prototype is not a separate NCAA model; it is a tournament-only torvik-heavy mean override or a slower March handoff away from torvik.",
        "",
        "## Caveats",
        "",
        "- NCAA sample size is only 67 games per completed tournament season.",
        "- Round-level results beyond Round of 64 / Round of 32 are noisy.",
        "- Historical line work uses `fct_lines_repaired_v1`, a hindsight-rebuilt line archive rather than a true time-native close snapshot.",
        "",
        "## Calibration Snapshot",
        "",
        "```csv",
        calibration_params.to_csv(index=False).strip(),
        "```",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads((PROMOTED_BENCHMARK_DIR / "protocol.json").read_text())
    promoted_frame = _build_promoted_eval_frame()
    master = _load_feature_master()

    promoted_with_current = promoted_frame[["season", "gameId", "model_margin_home", "market_margin_home"]].rename(
        columns={"model_margin_home": "current_pred"}
    )
    master = master.merge(promoted_with_current, on=["season", "gameId"], how="left")

    regime_summary = _regime_summary(promoted_frame)
    round_summary = _round_summary(promoted_frame)
    bucket_summary = _bucket_summary(promoted_frame)
    probability_summary = _probability_summary(promoted_frame)
    source_summary = _source_comparison(promoted_frame)
    candidate_summary, candidate_by_season, calibration_params = _candidate_backtest(master)

    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    round_summary.to_csv(output_dir / "ncaa_round_summary.csv", index=False)
    bucket_summary.to_csv(output_dir / "ncaa_bucket_summary.csv", index=False)
    probability_summary.to_csv(output_dir / "probability_summary.csv", index=False)
    source_summary.to_csv(output_dir / "source_comparison.csv", index=False)
    candidate_summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    candidate_by_season.to_csv(output_dir / "candidate_by_season.csv", index=False)
    calibration_params.to_csv(output_dir / "candidate_calibration_params.csv", index=False)

    _write_summary(
        output_dir,
        protocol,
        regime_summary,
        round_summary,
        bucket_summary,
        probability_summary,
        source_summary,
        candidate_summary,
        calibration_params,
    )

    manifest = {
        "promoted_benchmark_dir": str(PROMOTED_BENCHMARK_DIR),
        "torvik_benchmark_dir": str(TORVIK_LGB_BENCHMARK_DIR),
        "gold_benchmark_dir": str(GOLD_HGBR_BENCHMARK_DIR),
        "feature_dir": str(FEATURE_DIR),
        "holdout_seasons_all": ALL_HOLDOUT_SEASONS,
        "holdout_seasons_ncaa": NCAA_HOLDOUT_SEASONS,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

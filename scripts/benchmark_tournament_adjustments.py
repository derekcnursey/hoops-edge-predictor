#!/usr/bin/env python3
"""Benchmark tournament-only post-processing candidates on NCAA holdouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, s3_reader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "research" / "tournament_adjustment_benchmark_v1"
PROMOTED_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "benchmarks"
    / "canonical_walkforward_lgb_l2_blend_repaired_lines_neutralfix"
)
TORVIK_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "benchmarks"
    / "canonical_walkforward_v2_lgb_repaired_lines_neutralfix"
)
SEASONS = [2019, 2022, 2023, 2024, 2025]
PROMOTED_MODEL = "LightGBMRegressionL2Blend"
TORVIK_MODEL = "LightGBM"
MARKET_WEIGHT_GRID = np.round(np.linspace(0.0, 0.8, 17), 3)
PRIMARY_WEIGHT_GRID = np.round(np.linspace(0.35, 1.0, 14), 3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output artifact directory (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args()


def _read_predictions(pred_dir: Path, model_name: str) -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        path = pred_dir / "predictions" / model_name / f"season_{season}.parquet"
        if path.exists():
            frame = pd.read_parquet(path).copy()
            frame["season"] = season
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No predictions found for {model_name} under {pred_dir}")
    return pd.concat(frames, ignore_index=True)


def _load_games_meta() -> pd.DataFrame:
    parts = []
    for season in SEASONS:
        table = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
        if table.num_rows == 0:
            continue
        games = table.to_pandas()
        keep = [
            c
            for c in [
                "gameId",
                "startDate",
                "gameNotes",
                "gameType",
                "tournament",
                "conferenceGame",
                "neutralSite",
            ]
            if c in games.columns
        ]
        games = games[keep].drop_duplicates("gameId").copy()
        games["season"] = season
        parts.append(games)
    return pd.concat(parts, ignore_index=True)


def _round_from_note(note: object) -> str:
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
    return "Unknown"


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def _safe_rmse(actual: pd.Series, pred: pd.Series) -> float:
    diff = pd.to_numeric(actual, errors="coerce") - pd.to_numeric(pred, errors="coerce")
    arr = np.asarray(diff, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(np.square(finite)))) if finite.size else float("nan")


def _market_favorite_bias(frame: pd.DataFrame, pred_col: str) -> float:
    subset = frame.dropna(subset=["market_margin_home", pred_col]).copy()
    if subset.empty:
        return float("nan")
    fav_sign = np.sign(subset["market_margin_home"]).replace(0, np.nan)
    fav_pred = pd.to_numeric(subset[pred_col], errors="coerce") * fav_sign
    return float(np.nanmean(fav_pred - subset["market_margin_home"].abs()))


def _favorite_bucket(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(
        frame["market_fav_margin"],
        bins=[-0.1, 5.0, 10.0, 15.0, 100.0],
        labels=["<5", "5-10", "10-15", "15+"],
    )


def _build_master() -> pd.DataFrame:
    current = _read_predictions(PROMOTED_DIR, PROMOTED_MODEL).rename(
        columns={"pred_margin": "current_pred"}
    )
    torvik = _read_predictions(TORVIK_DIR, TORVIK_MODEL)[
        ["season", "gameId", "pred_margin"]
    ].rename(columns={"pred_margin": "torvik_pred"})
    meta = _load_games_meta()

    merged = current.merge(torvik, on=["season", "gameId"], how="left")
    merged = merged.merge(meta, on=["season", "gameId"], how="left", suffixes=("", "_meta"))
    merged = merged[merged["tournament"] == "NCAA"].copy()
    merged["market_margin_home"] = -pd.to_numeric(merged["book_spread"], errors="coerce")
    merged["market_fav_margin"] = merged["market_margin_home"].abs()
    merged["ncaa_round"] = merged["gameNotes"].map(_round_from_note)
    merged["favorite_bucket"] = _favorite_bucket(merged)
    return merged


def _margin_summary(frame: pd.DataFrame, pred_col: str) -> dict[str, float | int]:
    subset = frame.dropna(subset=[pred_col, "actual_margin"]).copy()
    pred = pd.to_numeric(subset[pred_col], errors="coerce")
    actual = pd.to_numeric(subset["actual_margin"], errors="coerce")
    market = pd.to_numeric(subset["market_margin_home"], errors="coerce")
    ats_rate = float("nan")
    if subset["book_spread"].notna().any():
        edge = pred - market
        cover_margin = actual - market
        ats = np.where(
            np.isclose(edge, 0.0),
            np.nan,
            np.where(edge > 0.0, cover_margin > 0.0, cover_margin < 0.0),
        ).astype(float)
        if np.isfinite(ats).any():
            ats_rate = float(np.nanmean(ats))
    return {
        "games": int(len(subset)),
        "margin_mae": _safe_mean((actual - pred).abs()),
        "margin_rmse": _safe_rmse(actual, pred),
        "line_mae_vs_market": _safe_mean((pred - market).abs()),
        "shortfall_vs_market_favorite": _market_favorite_bias(subset, pred_col),
        "straight_up_accuracy": float(np.mean((pred > 0.0) == (actual > 0.0))),
        "flat_ats_win_rate": ats_rate,
    }


def _best_market_weight(train: pd.DataFrame) -> float:
    if train.empty:
        return 0.0
    errors = []
    current = pd.to_numeric(train["current_pred"], errors="coerce")
    market = pd.to_numeric(train["market_margin_home"], errors="coerce")
    actual = pd.to_numeric(train["actual_margin"], errors="coerce")
    for weight in MARKET_WEIGHT_GRID:
        pred = (1.0 - weight) * current + weight * market
        errors.append(float(np.nanmean(np.abs(actual - pred))))
    return float(MARKET_WEIGHT_GRID[int(np.argmin(errors))])


def _best_primary_weight(train: pd.DataFrame) -> float:
    if train.empty:
        return 1.0
    current = pd.to_numeric(train["current_pred"], errors="coerce")
    torvik = pd.to_numeric(train["torvik_pred"], errors="coerce")
    actual = pd.to_numeric(train["actual_margin"], errors="coerce")
    errors = []
    for weight in PRIMARY_WEIGHT_GRID:
        pred = weight * current + (1.0 - weight) * torvik
        errors.append(float(np.nanmean(np.abs(actual - pred))))
    return float(PRIMARY_WEIGHT_GRID[int(np.argmin(errors))])


def _run_candidates(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    market_fixed_weight = _best_market_weight(master)
    primary_fixed_weight = _best_primary_weight(master)
    market_config_weight = float(np.clip(config.NCAA_TOURNAMENT_MARKET_WEIGHT, 0.0, 1.0))
    primary_config_weight = float(np.clip(config.NCAA_TOURNAMENT_PRIMARY_WEIGHT, 0.0, 1.0))

    rows = []
    calibration_rows = []
    for holdout in SEASONS:
        test = master[master["season"] == holdout].copy()
        train = master[master["season"] < holdout].copy()

        market_loo_weight = _best_market_weight(train)
        primary_loo_weight = _best_primary_weight(train)

        test["current_promoted"] = test["current_pred"]
        test["market_blend_fixed"] = (
            (1.0 - market_fixed_weight) * test["current_pred"]
            + market_fixed_weight * test["market_margin_home"]
        )
        test["market_blend_config"] = (
            (1.0 - market_config_weight) * test["current_pred"]
            + market_config_weight * test["market_margin_home"]
        )
        test["market_blend_loo"] = (
            (1.0 - market_loo_weight) * test["current_pred"]
            + market_loo_weight * test["market_margin_home"]
        )
        test["torvik_fallback_fixed"] = (
            primary_fixed_weight * test["current_pred"]
            + (1.0 - primary_fixed_weight) * test["torvik_pred"]
        )
        test["torvik_fallback_config"] = (
            primary_config_weight * test["current_pred"]
            + (1.0 - primary_config_weight) * test["torvik_pred"]
        )
        test["torvik_fallback_loo"] = (
            primary_loo_weight * test["current_pred"]
            + (1.0 - primary_loo_weight) * test["torvik_pred"]
        )

        calibration_rows.append(
            {
                "holdout_season": holdout,
                "market_weight_loo": market_loo_weight,
                "primary_weight_loo": primary_loo_weight,
                "prior_games": int(len(train)),
            }
        )
        rows.append(test)

    return (
        pd.concat(rows, ignore_index=True),
        pd.DataFrame(calibration_rows),
        market_fixed_weight,
        primary_fixed_weight,
    )


def _candidate_summary(frame: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        row = {"candidate": candidate}
        row.update(_margin_summary(frame, candidate))
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_breakdown(frame: pd.DataFrame, candidates: list[str], group_col: str) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        for value, group in frame.groupby(group_col, dropna=False, sort=False, observed=False):
            row = {"candidate": candidate, group_col: value}
            row.update(_margin_summary(group, candidate))
            rows.append(row)
    return pd.DataFrame(rows)


def _weight_grid_summary(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_rows = []
    for weight in MARKET_WEIGHT_GRID:
        pred_col = f"market_weight_{weight:.3f}"
        tmp = master.copy()
        tmp[pred_col] = (
            (1.0 - weight) * tmp["current_pred"] + weight * tmp["market_margin_home"]
        )
        row = {"market_weight": weight}
        row.update(_margin_summary(tmp, pred_col))
        market_rows.append(row)

    fallback_rows = []
    for weight in PRIMARY_WEIGHT_GRID:
        pred_col = f"primary_weight_{weight:.3f}"
        tmp = master.copy()
        tmp[pred_col] = weight * tmp["current_pred"] + (1.0 - weight) * tmp["torvik_pred"]
        row = {"primary_weight": weight}
        row.update(_margin_summary(tmp, pred_col))
        fallback_rows.append(row)

    return pd.DataFrame(market_rows), pd.DataFrame(fallback_rows)


def _write_summary(
    output_dir: Path,
    overall: pd.DataFrame,
    by_round: pd.DataFrame,
    by_bucket: pd.DataFrame,
    calibration: pd.DataFrame,
    market_grid: pd.DataFrame,
    fallback_grid: pd.DataFrame,
    market_fixed_weight: float,
    primary_fixed_weight: float,
) -> None:
    current = overall[overall["candidate"] == "current_promoted"].iloc[0]
    market_fixed = overall[overall["candidate"] == "market_blend_fixed"].iloc[0]
    market_config = overall[overall["candidate"] == "market_blend_config"].iloc[0]
    fallback_fixed = overall[overall["candidate"] == "torvik_fallback_fixed"].iloc[0]
    fallback_config = overall[overall["candidate"] == "torvik_fallback_config"].iloc[0]
    market_loo = overall[overall["candidate"] == "market_blend_loo"].iloc[0]
    fallback_loo = overall[overall["candidate"] == "torvik_fallback_loo"].iloc[0]
    market_weights_text = ", ".join(
        f"{int(row.holdout_season)}={row.market_weight_loo:.2f}"
        for row in calibration.itertuples()
    )
    primary_weights_text = ", ".join(
        f"{int(row.holdout_season)}={row.primary_weight_loo:.2f}"
        for row in calibration.itertuples()
    )
    round64 = by_round[
        (by_round["candidate"] == "current_promoted")
        & (by_round["ncaa_round"] == "Round of 64")
    ].iloc[0]
    big_favs = by_bucket[
        (by_bucket["candidate"] == "current_promoted")
        & (by_bucket["favorite_bucket"] == "15+")
    ].iloc[0]

    lines = [
        "# Tournament Adjustment Benchmark",
        "",
        "## Current Baseline",
        f"- NCAA sample: `{int(current['games'])}` games across `{', '.join(str(s) for s in SEASONS)}`.",
        f"- Current promoted NCAA margin MAE: `{current['margin_mae']:.3f}`.",
        f"- Current promoted shortfall vs market favorite: `{current['shortfall_vs_market_favorite']:.2f}`.",
        f"- Current Round of 64 shortfall: `{round64['shortfall_vs_market_favorite']:.2f}`.",
        f"- Current 15+ favorite shortfall: `{big_favs['shortfall_vs_market_favorite']:.2f}`.",
        "",
        "## Market-Aware Candidate",
        f"- Best pooled fixed NCAA market weight: `{market_fixed_weight:.2f}`.",
        f"- Fixed market blend NCAA margin MAE: `{market_fixed['margin_mae']:.3f}`.",
        f"- Fixed market blend shortfall vs market favorite: `{market_fixed['shortfall_vs_market_favorite']:.2f}`.",
        f"- Config-capped market blend (`{config.NCAA_TOURNAMENT_MARKET_WEIGHT:.2f}` market weight) NCAA margin MAE: `{market_config['margin_mae']:.3f}`.",
        f"- Config-capped market blend shortfall vs market favorite: `{market_config['shortfall_vs_market_favorite']:.2f}`.",
        f"- Leave-one-season-out market blend NCAA margin MAE: `{market_loo['margin_mae']:.3f}`.",
        f"- Leave-one-season-out market weights by season: `{market_weights_text}`.",
        "- Production use: display-only tournament spreads are appropriate because the candidate depends on the current market line.",
        "- Betting research use: not appropriate as the main model signal because it intentionally shrinks toward market.",
        "",
        "## Non-Market Fallback",
        f"- Best pooled fixed NCAA primary weight vs Torvik fallback: `{primary_fixed_weight:.2f}`.",
        f"- Fixed Torvik fallback NCAA margin MAE: `{fallback_fixed['margin_mae']:.3f}`.",
        f"- Fixed Torvik fallback shortfall vs market favorite: `{fallback_fixed['shortfall_vs_market_favorite']:.2f}`.",
        f"- Config Torvik fallback (`{config.NCAA_TOURNAMENT_PRIMARY_WEIGHT:.2f}` primary weight) NCAA margin MAE: `{fallback_config['margin_mae']:.3f}`.",
        f"- Config Torvik fallback shortfall vs market favorite: `{fallback_config['shortfall_vs_market_favorite']:.2f}`.",
        f"- Leave-one-season-out Torvik fallback NCAA margin MAE: `{fallback_loo['margin_mae']:.3f}`.",
        f"- Leave-one-season-out primary weights by season: `{primary_weights_text}`.",
        "- Production use: suitable as a core-path NCAA-only fallback because it does not consume market data.",
        "",
        "## Recommendation",
        "- Smallest production-safe improvement: add a tournament-only display blend for NCAA Tournament games only, with a capped configurable market weight.",
        "- Core model path: keep the year-round model unchanged by default; if a non-market NCAA-only core adjustment is needed, use the Torvik fallback override instead of a separate tournament model.",
        "- Do not use the market-aware blend as the primary betting-research signal.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = _build_master()
    candidate_frame, calibration, market_fixed_weight, primary_fixed_weight = _run_candidates(master)
    market_grid, fallback_grid = _weight_grid_summary(master)

    candidates = [
        "current_promoted",
        "market_blend_fixed",
        "market_blend_config",
        "market_blend_loo",
        "torvik_fallback_fixed",
        "torvik_fallback_config",
        "torvik_fallback_loo",
    ]
    overall = _candidate_summary(candidate_frame, candidates)
    by_round = _candidate_breakdown(candidate_frame, candidates, "ncaa_round")
    by_bucket = _candidate_breakdown(candidate_frame, candidates, "favorite_bucket")

    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)
    by_round.to_csv(args.output_dir / "by_round_summary.csv", index=False)
    by_bucket.to_csv(args.output_dir / "by_favorite_bucket_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "candidate_calibration.csv", index=False)
    market_grid.to_csv(args.output_dir / "market_weight_grid_summary.csv", index=False)
    fallback_grid.to_csv(args.output_dir / "fallback_weight_grid_summary.csv", index=False)
    candidate_frame.to_csv(args.output_dir / "candidate_game_level.csv", index=False)

    _write_summary(
        args.output_dir,
        overall,
        by_round,
        by_bucket,
        calibration,
        market_grid,
        fallback_grid,
        market_fixed_weight,
        primary_fixed_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

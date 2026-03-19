#!/usr/bin/env python3
"""Benchmark efficiency sources under the canonical HGBR protocol.

This is a research-only source swap benchmark:
  - same HGBR model
  - same repaired lines
  - same holdout seasons
  - same features outside the efficiency source
  - only the efficiency source changes
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.features import build_features, load_research_lines

SEED = 42
TRAIN_START = 2015
EXCLUDE_SEASONS = config.EXCLUDE_SEASONS
HOLDOUT_SEASONS = [2019, 2020, 2022, 2023, 2024, 2025]
NO_GARBAGE = True
LINE_PROVIDER_RANK = {"Draft Kings": 0, "ESPN BET": 1, "Bovada": 2}


@dataclass(frozen=True)
class EfficiencySource:
    label: str
    efficiency_source: str
    gold_table_name: str | None = None


SOURCES: list[EfficiencySource] = [
    EfficiencySource("torvik", "torvik"),
    EfficiencySource("gold_current", "gold"),
    EfficiencySource("gold_priorreg_k5_v1", "gold", "team_adjusted_efficiencies_no_garbage_priorreg_k5_v1"),
    EfficiencySource("gold_priorreg_k5_hl60_v1", "gold", "team_adjusted_efficiencies_no_garbage_priorreg_k5_hl60_v1"),
    EfficiencySource("gold_priorreg_k5_hl45_v1", "gold", "team_adjusted_efficiencies_no_garbage_priorreg_k5_hl45_v1"),
    EfficiencySource("gold_priorreg_k5_hl30_v1", "gold", "team_adjusted_efficiencies_no_garbage_priorreg_k5_hl30_v1"),
]


def _default_output_dir() -> Path:
    return config.ARTIFACTS_DIR / "efficiency_research" / "efficiency_source_benchmark_v1"


def _cache_dir() -> Path:
    return config.FEATURES_DIR / "efficiency_research"


def _feature_path(source: EfficiencySource, season: int) -> Path:
    return _cache_dir() / source.label / f"season_{season}_features.parquet"


def _existing_feature_cache(source: EfficiencySource, season: int) -> Path | None:
    if source.label == "torvik":
        path = config.FEATURES_DIR / "torvik" / f"season_{season}_torvik_features.parquet"
        return path if path.exists() else None
    if source.label == "gold_current":
        path = config.FEATURES_DIR / f"season_{season}_no_garbage_adj_a0.85_p10_features.parquet"
        return path if path.exists() else None
    return None


def _load_or_build_features(source: EfficiencySource, season: int) -> pd.DataFrame:
    path = _feature_path(source, season)
    if path.exists():
        return pd.read_parquet(path)

    existing = _existing_feature_cache(source, season)
    if existing is not None:
        df = pd.read_parquet(existing)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df

    print(f"Building features for {source.label} season {season}...")
    df = build_features(
        season,
        no_garbage=NO_GARBAGE,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
        adjust_ff_method=config.ADJUST_FF_METHOD,
        efficiency_source=source.efficiency_source,
        gold_table_name=source.gold_table_name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(path, index=False)
    return df


def _training_seasons(holdout: int) -> list[int]:
    return [s for s in range(TRAIN_START, holdout) if s not in EXCLUDE_SEASONS]


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.dropna(subset=["homeScore", "awayScore"]).copy()
    out = out[(out["homeScore"] != 0) | (out["awayScore"] != 0)].copy()
    return out.reset_index(drop=True)


def _train_impute_means(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    arr = frame[feature_cols].to_numpy(dtype=np.float32, copy=True)
    means = np.nanmean(arr, axis=0)
    means = np.where(np.isnan(means), 0.0, means).astype(np.float32)
    return means


def _apply_imputation(frame: pd.DataFrame, feature_cols: list[str], means: np.ndarray) -> np.ndarray:
    arr = frame[feature_cols].to_numpy(dtype=np.float32, copy=True)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = means[np.where(nan_mask)[1]]
    return arr


def _dedupe_lines(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return lines
    df = lines.copy()
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    df["_has_spread"] = df["spread"].notna().astype(int)
    df["_provider_rank"] = df["provider"].map(LINE_PROVIDER_RANK).fillna(99)
    df["_provider_name"] = df["provider"].fillna("")
    df = df.sort_values(
        ["_has_spread", "_provider_rank", "_provider_name"],
        ascending=[False, True, True],
    )
    return df.drop_duplicates(subset=["gameId"], keep="first").drop(
        columns=["_has_spread", "_provider_rank", "_provider_name"]
    )


def _attach_lines(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    lines = load_research_lines(season)
    if lines.empty:
        out = frame.copy()
        out["book_spread"] = np.nan
        return out
    deduped = _dedupe_lines(lines)
    spreads = deduped[["gameId", "spread"]].rename(columns={"spread": "book_spread"})
    return frame.merge(spreads, on="gameId", how="left")


def _season_phase(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("America/New_York")
    month = dt.dt.month
    phase = np.where(month.isin([11, 12]), "nov_dec", np.where(month.isin([1, 2, 3]), "jan_mar", "other"))
    return pd.Series(phase, index=series.index)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    actual = df["actual_margin"].to_numpy(dtype=float)
    pred = df["pred_margin"].to_numpy(dtype=float)
    lined = df["book_spread"].notna().to_numpy()
    mae_all = float(np.mean(np.abs(pred - actual))) if len(df) else math.nan
    mae_lined = float(np.mean(np.abs(pred[lined] - actual[lined]))) if lined.any() else math.nan
    book_mae = float(np.mean(np.abs((-df.loc[lined, "book_spread"].to_numpy(dtype=float)) - actual[lined]))) if lined.any() else math.nan
    return {
        "games": float(len(df)),
        "lined_games": float(lined.sum()),
        "mae_all": mae_all,
        "mae_lined": mae_lined,
        "book_mae_lined": book_mae,
        "delta_vs_book_mae": mae_lined - book_mae if lined.any() else math.nan,
    }


def _render_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "```"


def _fit_predict(source: EfficiencySource, holdout: int) -> pd.DataFrame:
    feature_cols = list(config.FEATURE_ORDER)
    train_frames = []
    for season in _training_seasons(holdout):
        train_frames.append(_prepare_frame(_load_or_build_features(source, season)))
    holdout_df = _prepare_frame(_load_or_build_features(source, holdout))
    if not train_frames or holdout_df.empty:
        return pd.DataFrame()

    train_df = pd.concat(train_frames, ignore_index=True)
    means = _train_impute_means(train_df, feature_cols)
    X_train = _apply_imputation(train_df, feature_cols, means)
    y_train = (train_df["homeScore"].to_numpy(dtype=float) - train_df["awayScore"].to_numpy(dtype=float)).astype(np.float32)

    model = HistGradientBoostingRegressor(**config.HGBR_PARAMS)
    model.fit(X_train, y_train)

    X_holdout = _apply_imputation(holdout_df, feature_cols, means)
    pred = model.predict(X_holdout)
    out = holdout_df[["gameId", "startDate", "homeTeam", "awayTeam", "homeScore", "awayScore"]].copy()
    out["actual_margin"] = out["homeScore"].astype(float) - out["awayScore"].astype(float)
    out["pred_margin"] = pred.astype(float)
    out = _attach_lines(out, holdout)
    out["season"] = holdout
    out["phase"] = _season_phase(out["startDate"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark efficiency sources with fixed HGBR stack.")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--holdout-season", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    holdouts = [args.holdout_season] if args.holdout_season else HOLDOUT_SEASONS
    fold_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []

    predictions_root = output_dir / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        source_frames = []
        for holdout in holdouts:
            print(f"Evaluating {source.label} holdout {holdout}...")
            pred_df = _fit_predict(source, holdout)
            if pred_df.empty:
                continue
            source_frames.append(pred_df)
            season_dir = predictions_root / source.label
            season_dir.mkdir(parents=True, exist_ok=True)
            pred_df.to_parquet(season_dir / f"season_{holdout}.parquet", index=False)

            row = {"source": source.label, "holdout_season": holdout, **_metrics(pred_df)}
            fold_rows.append(row)
            for phase, phase_df in pred_df.groupby("phase"):
                phase_rows.append(
                    {"source": source.label, "holdout_season": holdout, "phase": phase, **_metrics(phase_df)}
                )

        if source_frames:
            pooled = pd.concat(source_frames, ignore_index=True)
            pooled_rows.append({"source": source.label, **_metrics(pooled)})

    fold_df = pd.DataFrame(fold_rows).sort_values(["source", "holdout_season"])
    phase_df = pd.DataFrame(phase_rows).sort_values(["source", "holdout_season", "phase"])
    pooled_df = pd.DataFrame(pooled_rows).sort_values("mae_all")

    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    phase_df.to_csv(output_dir / "phase_metrics.csv", index=False)
    pooled_df.to_csv(output_dir / "pooled_metrics.csv", index=False)

    priorreg_only = pooled_df[pooled_df["source"].str.startswith("gold_priorreg_")].copy()
    best_priorreg = priorreg_only.sort_values("mae_all").head(1)
    protocol = {
        "benchmark": "efficiency_source_benchmark_v1",
        "holdout_seasons": holdouts,
        "excluded_seasons": EXCLUDE_SEASONS,
        "fixed_stack": {
            "model": "HistGradientBoosting",
            "feature_order": "production_53",
            "repaired_lines": True,
            "no_garbage": NO_GARBAGE,
            "adjust_ff": config.ADJUST_FF,
            "adjust_alpha": config.ADJUST_ALPHA,
            "adjust_prior": config.ADJUST_PRIOR,
            "adjust_ff_method": config.ADJUST_FF_METHOD,
        },
        "sources": [asdict(source) for source in SOURCES],
        "best_priorreg_source": None if best_priorreg.empty else best_priorreg.iloc[0]["source"],
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))

    lines = [
        "# Efficiency Source Benchmark v1",
        "",
        "Fixed stack: HistGradientBoosting + repaired lines + production-style features.",
        "",
        "## Pooled",
        "",
        _render_table(pooled_df),
        "",
        "## By Season",
        "",
        _render_table(fold_df),
        "",
        "## By Phase",
        "",
        _render_table(phase_df),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

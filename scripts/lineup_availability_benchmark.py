#!/usr/bin/env python3
"""Research-only benchmark for rotation continuity / availability features."""

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
from src.rotation_availability import (
    ROTATION_FEATURE_COLUMNS,
    build_rotation_availability_team_features,
    build_team_game_player_participation_v1,
    merge_rotation_availability_features,
    spine_is_usable,
)

SEED = 42
TRAIN_START = 2015
HOLDOUT_SEASONS = [2019, 2020, 2022, 2023, 2024, 2025]
NO_GARBAGE = True
LINE_PROVIDER_RANK = {"Draft Kings": 0, "ESPN BET": 1, "Bovada": 2}
CUTOFF_MM_DD = "12-15"
PRIMARY_CUTOFF_LABEL = "dec15_plus"


@dataclass(frozen=True)
class EfficiencySource:
    label: str
    efficiency_source: str
    gold_table_name: str | None = None


@dataclass(frozen=True)
class Variant:
    label: str
    source: EfficiencySource
    add_rotation: bool


TORVIK = EfficiencySource("torvik", "torvik")
GOLD_PRIORREG = EfficiencySource(
    "gold_priorreg_k5_v1",
    "gold",
    "team_adjusted_efficiencies_no_garbage_priorreg_k5_v1",
)

VARIANTS = [
    Variant("torvik", TORVIK, False),
    Variant("torvik_rotation_availability_v1", TORVIK, True),
    Variant("gold_priorreg_k5_v1", GOLD_PRIORREG, False),
    Variant("gold_priorreg_k5_v1_rotation_availability_v1", GOLD_PRIORREG, True),
]


def _default_output_dir() -> Path:
    return config.ARTIFACTS_DIR / "lineup_research" / "rotation_availability_benchmark_v1"


def _feature_cache_dir() -> Path:
    return config.FEATURES_DIR / "lineup_research"


def _feature_path(source: EfficiencySource, season: int) -> Path:
    return _feature_cache_dir() / source.label / f"season_{season}_features.parquet"


def _existing_feature_cache(source: EfficiencySource, season: int) -> Path | None:
    if source.label == "torvik":
        path = config.FEATURES_DIR / "torvik" / f"season_{season}_torvik_features.parquet"
        return path if path.exists() else None
    if source.label == "gold_priorreg_k5_v1":
        path = config.FEATURES_DIR / "efficiency_research" / source.label / f"season_{season}_features.parquet"
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
    return [s for s in range(TRAIN_START, holdout) if s not in config.EXCLUDE_SEASONS]


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


def _slice_label(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("America/New_York")
    md = dt.dt.strftime("%m-%d")
    out = np.where(md >= CUTOFF_MM_DD, PRIMARY_CUTOFF_LABEL, "pre_dec15")
    return pd.Series(out, index=series.index)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    actual = df["actual_margin"].to_numpy(dtype=float)
    pred = df["pred_margin"].to_numpy(dtype=float)
    lined = df["book_spread"].notna().to_numpy()
    mae_all = float(np.mean(np.abs(pred - actual))) if len(df) else math.nan
    mae_lined = float(np.mean(np.abs(pred[lined] - actual[lined]))) if lined.any() else math.nan
    return {
        "games": float(len(df)),
        "lined_games": float(lined.sum()),
        "mae_all": mae_all,
        "mae_lined": mae_lined,
    }


def _render_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "```"


def _season_features(source: EfficiencySource, season: int, team_rotation: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_frame(_load_or_build_features(source, season))
    if df.empty:
        return df
    return merge_rotation_availability_features(df, team_rotation)


def _fit_predict(variant: Variant, holdout: int, team_rotation: pd.DataFrame) -> pd.DataFrame:
    train_frames = []
    for season in _training_seasons(holdout):
        train_frames.append(_season_features(variant.source, season, team_rotation))

    holdout_df = _season_features(variant.source, holdout, team_rotation)
    if not train_frames or holdout_df.empty:
        return pd.DataFrame()

    train_df = pd.concat(train_frames, ignore_index=True)
    feature_cols = list(config.FEATURE_ORDER)
    if variant.add_rotation:
        feature_cols = feature_cols + ROTATION_FEATURE_COLUMNS

    means = _train_impute_means(train_df, feature_cols)
    X_train = _apply_imputation(train_df, feature_cols, means)
    y_train = (
        train_df["homeScore"].to_numpy(dtype=float) - train_df["awayScore"].to_numpy(dtype=float)
    ).astype(np.float32)

    model = HistGradientBoostingRegressor(**config.HGBR_PARAMS)
    model.fit(X_train, y_train)

    X_holdout = _apply_imputation(holdout_df, feature_cols, means)
    pred = model.predict(X_holdout)
    out = holdout_df[["gameId", "startDate", "homeTeam", "awayTeam", "homeScore", "awayScore"]].copy()
    out["actual_margin"] = out["homeScore"].astype(float) - out["awayScore"].astype(float)
    out["pred_margin"] = pred.astype(float)
    out = _attach_lines(out, holdout)
    out["season"] = holdout
    out["slice"] = _slice_label(out["startDate"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark lineup/availability features with fixed HGBR stack.")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--holdout-season", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    holdouts = [args.holdout_season] if args.holdout_season else HOLDOUT_SEASONS
    seasons_needed = sorted({s for holdout in holdouts for s in _training_seasons(holdout) + [holdout]})

    spine_dir = output_dir / "spine"
    flat_df, audit_df = build_team_game_player_participation_v1(seasons_needed, output_dir=spine_dir)
    usable, reason = spine_is_usable(audit_df)
    protocol = {
        "holdout_seasons": holdouts,
        "cutoff_mm_dd": CUTOFF_MM_DD,
        "variants": [asdict(v.source) | {"variant_label": v.label, "add_rotation": v.add_rotation} for v in VARIANTS],
        "spine_usable": usable,
        "spine_reason": reason,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))

    if not usable:
        print(f"Stopping early: player-game spine not usable ({reason})")
        return

    team_rotation = build_rotation_availability_team_features(flat_df, output_dir=spine_dir)

    fold_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    pooled_slice_rows: list[dict[str, object]] = []
    predictions_root = output_dir / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        variant_frames = []
        for holdout in holdouts:
            print(f"Evaluating {variant.label} holdout {holdout}...")
            pred_df = _fit_predict(variant, holdout, team_rotation)
            if pred_df.empty:
                continue
            variant_frames.append(pred_df)
            season_dir = predictions_root / variant.label
            season_dir.mkdir(parents=True, exist_ok=True)
            pred_df.to_parquet(season_dir / f"season_{holdout}.parquet", index=False)

            fold_rows.append({"variant": variant.label, "holdout_season": holdout, **_metrics(pred_df)})
            for slice_name, slice_df in pred_df.groupby("slice"):
                slice_rows.append(
                    {
                        "variant": variant.label,
                        "holdout_season": holdout,
                        "slice": slice_name,
                        **_metrics(slice_df),
                    }
                )

        if variant_frames:
            pooled = pd.concat(variant_frames, ignore_index=True)
            pooled_rows.append({"variant": variant.label, **_metrics(pooled)})
            for slice_name, slice_df in pooled.groupby("slice"):
                pooled_slice_rows.append({"variant": variant.label, "slice": slice_name, **_metrics(slice_df)})

    fold_df = pd.DataFrame(fold_rows).sort_values(["variant", "holdout_season"]).reset_index(drop=True)
    slice_df = pd.DataFrame(slice_rows).sort_values(["variant", "holdout_season", "slice"]).reset_index(drop=True)
    pooled_df = pd.DataFrame(pooled_rows).sort_values("variant").reset_index(drop=True)
    pooled_slice_df = pd.DataFrame(pooled_slice_rows).sort_values(["variant", "slice"]).reset_index(drop=True)

    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    slice_df.to_csv(output_dir / "season_slice_metrics.csv", index=False)
    pooled_df.to_csv(output_dir / "pooled_metrics.csv", index=False)
    pooled_slice_df.to_csv(output_dir / "pooled_slice_metrics.csv", index=False)

    summary_lines = [
        "# Rotation Availability Benchmark v1",
        "",
        f"- Player-game spine usable: `{usable}` ({reason})",
        f"- Primary evaluation slice: `{PRIMARY_CUTOFF_LABEL}` with cutoff `{CUTOFF_MM_DD}`",
        "",
        "## Pooled full-season metrics",
        _render_table(pooled_df.round(4)),
        "",
        f"## Pooled {PRIMARY_CUTOFF_LABEL} metrics",
        _render_table(
            pooled_slice_df[pooled_slice_df["slice"] == PRIMARY_CUTOFF_LABEL]
            .drop(columns=["slice"])
            .round(4)
        ),
        "",
        f"## By-season {PRIMARY_CUTOFF_LABEL} metrics",
        _render_table(
            slice_df[slice_df["slice"] == PRIMARY_CUTOFF_LABEL]
            .drop(columns=["slice"])
            .round(4)
        ),
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()

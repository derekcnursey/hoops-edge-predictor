#!/usr/bin/env python3
"""Targeted lined-games tie-break analysis for canonical benchmark artifacts.

Compares HistGradientBoosting vs LightGBM on lined games only and reports:
  - pooled lined-only MAE by model
  - MAE difference (LightGBM - HistGradientBoosting) with bootstrap CI
  - per-holdout lined-only results
  - phase splits: Nov-Dec vs Jan-Mar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HGBR_LABEL = "HistGradientBoosting"
LGB_LABEL = "LightGBM"
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 2000


def _phase_label(start_date: pd.Series) -> pd.Series:
    dt = pd.to_datetime(start_date, errors="coerce", utc=True)
    month = dt.dt.month
    out = pd.Series(pd.NA, index=start_date.index, dtype="object")
    out = out.mask(month.isin([11, 12]), "Nov-Dec")
    out = out.mask(month.isin([1, 2, 3]), "Jan-Mar")
    return out


def _load_model_predictions(benchmark_dir: Path, model_name: str) -> pd.DataFrame:
    pred_dir = benchmark_dir / "predictions" / model_name
    parts = sorted(pred_dir.glob("season_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No prediction parquets found for {model_name} in {pred_dir}")
    frames = [pd.read_parquet(path) for path in parts]
    out = pd.concat(frames, ignore_index=True)
    out["phase"] = _phase_label(out["startDate"])
    return out


def _paired_lined_games(benchmark_dir: Path) -> pd.DataFrame:
    hgbr = _load_model_predictions(benchmark_dir, HGBR_LABEL)
    lgb = _load_model_predictions(benchmark_dir, LGB_LABEL)

    keep = ["holdout_season", "gameId", "startDate", "phase", "actual_margin", "book_spread", "abs_error"]
    hgbr = hgbr[keep].rename(columns={"abs_error": "hgbr_abs_error"})
    lgb = lgb[keep].rename(columns={"abs_error": "lgb_abs_error"})

    merged = hgbr.merge(
        lgb,
        on=["holdout_season", "gameId", "startDate", "phase", "actual_margin", "book_spread"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged[merged["book_spread"].notna()].copy()
    merged["mae_diff_lgb_minus_hgbr"] = merged["lgb_abs_error"] - merged["hgbr_abs_error"]
    return merged


def _bootstrap_ci(values: np.ndarray, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(draws, len(vals)))
    samples = vals[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _summarize_slice(df: pd.DataFrame, label: str) -> dict[str, float | str]:
    diff = df["mae_diff_lgb_minus_hgbr"].to_numpy(dtype=float)
    ci_lo, ci_hi = _bootstrap_ci(diff)
    return {
        "slice": label,
        "n_games": int(len(df)),
        "hgbr_mae_lined": float(df["hgbr_abs_error"].mean()) if len(df) else np.nan,
        "lgb_mae_lined": float(df["lgb_abs_error"].mean()) if len(df) else np.nan,
        "mae_diff_lgb_minus_hgbr": float(diff.mean()) if len(diff) else np.nan,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "lgb_game_win_rate": float((diff < 0).mean()) if len(diff) else np.nan,
        "tie_rate": float((diff == 0).mean()) if len(diff) else np.nan,
    }


def _summary_tables(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows = [
        _summarize_slice(paired, "All lined games"),
        _summarize_slice(paired[paired["phase"] == "Nov-Dec"], "Nov-Dec"),
        _summarize_slice(paired[paired["phase"] == "Jan-Mar"], "Jan-Mar"),
    ]
    pooled = pd.DataFrame(pooled_rows)

    season_rows = []
    for season, group in paired.groupby("holdout_season"):
        season_rows.append(_summarize_slice(group, f"{season} all"))
        season_rows[-1]["holdout_season"] = int(season)
    season_df = pd.DataFrame(season_rows)

    phase_rows = []
    by = paired.dropna(subset=["phase"]).groupby(["holdout_season", "phase"])
    for (holdout_season, phase), group in by:
        row = _summarize_slice(group, f"{holdout_season} {phase}")
        row["holdout_season"] = int(holdout_season)
        row["phase"] = phase
        phase_rows.append(row)
    season_phase = pd.DataFrame(phase_rows)
    return pooled, season_df, season_phase


def _format_num(value: float) -> str:
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
                vals.append(_format_num(val))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(benchmark_dir: Path, pooled: pd.DataFrame, season: pd.DataFrame, season_phase: pd.DataFrame) -> None:
    lines = [
        "# Lined-Games Tie-Break: HistGradientBoosting vs LightGBM",
        "",
        "This analysis uses lined games only from the completed canonical walk-forward benchmark.",
        f"- Bootstrap draws: {BOOTSTRAP_DRAWS}",
        f"- Bootstrap seed: {BOOTSTRAP_SEED}",
        "- Difference convention: `LightGBM MAE - HistGradientBoosting MAE`",
        "- Negative difference favors LightGBM",
        "",
        "## Pooled",
        "",
        _markdown_table(
            pooled,
            [
                "slice",
                "n_games",
                "hgbr_mae_lined",
                "lgb_mae_lined",
                "mae_diff_lgb_minus_hgbr",
                "ci95_lo",
                "ci95_hi",
                "lgb_game_win_rate",
            ],
        ),
        "",
        "## By Holdout Season",
        "",
        _markdown_table(
            season.sort_values("holdout_season"),
            [
                "holdout_season",
                "n_games",
                "hgbr_mae_lined",
                "lgb_mae_lined",
                "mae_diff_lgb_minus_hgbr",
                "ci95_lo",
                "ci95_hi",
                "lgb_game_win_rate",
            ],
        ),
        "",
        "## By Holdout Season And Phase",
        "",
        _markdown_table(
            season_phase.sort_values(["holdout_season", "phase"]),
            [
                "holdout_season",
                "phase",
                "n_games",
                "hgbr_mae_lined",
                "lgb_mae_lined",
                "mae_diff_lgb_minus_hgbr",
                "ci95_lo",
                "ci95_hi",
                "lgb_game_win_rate",
            ],
        ),
    ]
    (benchmark_dir / "lined_tiebreak_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HGBR vs LightGBM on lined games only.")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Completed canonical benchmark artifact directory",
    )
    args = parser.parse_args()

    paired = _paired_lined_games(args.benchmark_dir)
    pooled, season, season_phase = _summary_tables(paired)

    pooled.to_csv(args.benchmark_dir / "lined_tiebreak_pooled.csv", index=False)
    season.to_csv(args.benchmark_dir / "lined_tiebreak_by_season.csv", index=False)
    season_phase.to_csv(args.benchmark_dir / "lined_tiebreak_by_season_phase.csv", index=False)
    write_report(args.benchmark_dir, pooled, season, season_phase)

    summary = {
        "benchmark_dir": str(args.benchmark_dir),
        "n_lined_games": int(len(paired)),
        "pooled_diff_lgb_minus_hgbr": float(pooled.loc[pooled["slice"] == "All lined games", "mae_diff_lgb_minus_hgbr"].iloc[0]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

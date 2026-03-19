#!/usr/bin/env python3
"""Direct post-hoc cover-probability calibration study with fixed HGBR mu."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.cover_probability_calibration import (
    apply_probability_calibration,
    fit_isotonic_calibrator,
    fit_logistic_calibrator,
    normal_cdf_from_z,
)
from src.sigma_calibration import apply_sigma_transform

BREAKEVEN = 110.0 / 210.0
WIN_PROFIT = 100.0 / 110.0
PROB_BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
EDGE_BUCKETS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.00]
TOP_NS = [100, 200, 500]
EVAL_SEASONS = [2020, 2022, 2023, 2024, 2025]


@dataclass(frozen=True)
class OptionSpec:
    label: str
    base_score: str
    method: str


def _load_sigma_winner_spec(sigma_study_dir: Path) -> dict[str, object]:
    summary = json.loads((sigma_study_dir / "dataset_summary.json").read_text())
    winner_spec = dict(summary["winner_spec"])
    return {
        key: value
        for key, value in winner_spec.items()
        if key
        not in {"selected_label"}
        and value is not None
        and not (isinstance(value, float) and np.isnan(value))
    }


def _load_dataset(benchmark_dir: Path, sigma_study_dir: Path) -> pd.DataFrame:
    winner_spec = _load_sigma_winner_spec(sigma_study_dir)
    pred_dir = benchmark_dir / "predictions"
    frames: list[pd.DataFrame] = []

    for season_path in sorted((pred_dir / "HistGradientBoosting").glob("season_*.parquet")):
        season = int(season_path.stem.split("_")[1])
        hgb = pd.read_parquet(season_path)
        mlp = pd.read_parquet(pred_dir / "CurrentMLP" / f"season_{season}.parquet")
        merged = hgb[
            [
                "gameId",
                "holdout_season",
                "startDate",
                "homeTeam",
                "awayTeam",
                "actual_margin",
                "pred_margin",
                "book_spread",
            ]
        ].rename(columns={"pred_margin": "pred_margin_mu"})
        merged = merged.merge(mlp[["gameId", "sigma"]], on="gameId", how="inner")
        merged["season"] = season
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["book_spread"].notna()].copy()
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    df["month"] = df["startDate"].dt.month
    df["phase"] = np.where(
        df["month"].isin([11, 12]),
        "Nov-Dec",
        np.where(df["month"].isin([1, 2, 3]), "Jan-Mar", "Other"),
    )
    df["actual_edge_home"] = df["actual_margin"] + df["book_spread"]
    df["edge_home_points"] = df["pred_margin_mu"] + df["book_spread"]
    df["home_cover_win"] = df["actual_edge_home"] > 0
    df["push"] = df["actual_edge_home"] == 0
    df["raw_sigma"] = df["sigma"].astype(float)
    df["best_posthoc_sigma"] = apply_sigma_transform(
        df["raw_sigma"].to_numpy(float),
        mode=str(winner_spec["family"]),
        **{
            key: float(value)
            for key, value in winner_spec.items()
            if key not in {"family"}
        },
    )
    df["constant14_sigma"] = 14.0
    df["z_raw"] = df["edge_home_points"] / df["raw_sigma"]
    df["z_best"] = df["edge_home_points"] / df["best_posthoc_sigma"]
    df["z_const14"] = df["edge_home_points"] / df["constant14_sigma"]
    return df.reset_index(drop=True)


def _option_specs() -> list[OptionSpec]:
    return [
        OptionSpec("raw_mlp_normal", "z_raw", "normal"),
        OptionSpec("best_posthoc_normal", "z_best", "normal"),
        OptionSpec("constant14_normal", "z_const14", "normal"),
        OptionSpec("logistic_raw_z", "z_raw", "logistic"),
        OptionSpec("isotonic_raw_z", "z_raw", "isotonic"),
        OptionSpec("logistic_best_posthoc_z", "z_best", "logistic"),
        OptionSpec("isotonic_best_posthoc_z", "z_best", "isotonic"),
    ]


def _fit_calibration(train_df: pd.DataFrame, option: OptionSpec) -> dict[str, object]:
    if option.method == "normal":
        return {"method": "normal"}

    train_nonpush = train_df[~train_df["push"]].copy()
    z_score = train_nonpush[option.base_score].to_numpy(float)
    outcome = train_nonpush["home_cover_win"].astype(int).to_numpy()
    if option.method == "logistic":
        return fit_logistic_calibrator(z_score, outcome)
    if option.method == "isotonic":
        return fit_isotonic_calibrator(z_score, outcome)
    raise ValueError(option.method)


def _pick_probability(df: pd.DataFrame, p_home_cover: np.ndarray) -> np.ndarray:
    edge = df["edge_home_points"].to_numpy(float)
    return np.where(edge >= 0.0, p_home_cover, 1.0 - p_home_cover)


def _evaluate_option(df: pd.DataFrame, p_home_cover: np.ndarray) -> dict[str, float]:
    nonpush = ~df["push"].to_numpy(bool)
    y = df.loc[nonpush, "home_cover_win"].astype(float).to_numpy()
    p_nonpush = np.clip(p_home_cover[nonpush], 1e-6, 1.0 - 1e-6)
    cover_logloss = float(
        -(y * np.log(p_nonpush) + (1.0 - y) * np.log(1.0 - p_nonpush)).mean()
    )
    cover_brier = float(np.mean((p_nonpush - y) ** 2))

    pick_prob = _pick_probability(df, p_home_cover)
    pick_home = df["edge_home_points"].to_numpy(float) >= 0.0
    pick_win = np.where(
        pick_home,
        df["actual_edge_home"].to_numpy(float) > 0,
        df["actual_edge_home"].to_numpy(float) < 0,
    )
    push = df["push"].to_numpy(bool)
    picked_profit = np.where(push, 0.0, np.where(pick_win, WIN_PROFIT, -1.0))
    pick_prob_edge = pick_prob - BREAKEVEN

    out = {
        "cover_logloss": cover_logloss,
        "cover_brier": cover_brier,
        "pick_prob_mean": float(np.mean(pick_prob)),
        "pick_prob_std": float(np.std(pick_prob)),
        "pick_prob_edge_mean": float(np.mean(pick_prob_edge)),
        "prob80_n": int(np.sum(pick_prob >= 0.80)),
        "prob80_winrate": float(np.mean(pick_win[(pick_prob >= 0.80) & ~push]))
        if np.any((pick_prob >= 0.80) & ~push)
        else float("nan"),
        "prob80_roi": float(np.mean(picked_profit[pick_prob >= 0.80]))
        if np.any(pick_prob >= 0.80)
        else float("nan"),
        "prob85_n": int(np.sum(pick_prob >= 0.85)),
        "prob85_winrate": float(np.mean(pick_win[(pick_prob >= 0.85) & ~push]))
        if np.any((pick_prob >= 0.85) & ~push)
        else float("nan"),
        "prob85_roi": float(np.mean(picked_profit[pick_prob >= 0.85]))
        if np.any(pick_prob >= 0.85)
        else float("nan"),
        "edge10_n": int(np.sum(pick_prob_edge >= 0.10)),
        "edge10_roi": float(np.mean(picked_profit[pick_prob_edge >= 0.10]))
        if np.any(pick_prob_edge >= 0.10)
        else float("nan"),
        "edge15_n": int(np.sum(pick_prob_edge >= 0.15)),
        "edge15_roi": float(np.mean(picked_profit[pick_prob_edge >= 0.15]))
        if np.any(pick_prob_edge >= 0.15)
        else float("nan"),
    }
    for top_n in TOP_NS:
        chosen = np.argsort(-pick_prob)[: min(top_n, len(df))]
        chosen_mask = np.zeros(len(df), dtype=bool)
        chosen_mask[chosen] = True
        wins = pick_win[chosen_mask & ~push]
        out[f"top{top_n}_roi"] = float(np.mean(picked_profit[chosen_mask]))
        out[f"top{top_n}_winrate"] = float(np.mean(wins)) if len(wins) else float("nan")
        out[f"top{top_n}_avg_prob"] = float(np.mean(pick_prob[chosen_mask]))
    return out


def _bucket_rows(
    df: pd.DataFrame,
    p_home_cover: np.ndarray,
    *,
    bucket_kind: str,
    option_label: str,
    season: int,
) -> list[dict[str, object]]:
    pick_prob = pd.Series(_pick_probability(df, p_home_cover), index=df.index)
    pick_home = df["edge_home_points"].to_numpy(float) >= 0.0
    pick_win = pd.Series(
        np.where(pick_home, df["actual_edge_home"] > 0, df["actual_edge_home"] < 0),
        index=df.index,
    )
    push = df["push"].astype(bool)
    picked_profit = pd.Series(
        np.where(push, 0.0, np.where(pick_win, WIN_PROFIT, -1.0)),
        index=df.index,
    )

    if bucket_kind == "probability":
        bucketed = pd.cut(pick_prob, bins=PROB_BUCKETS, include_lowest=True, right=False)
    else:
        bucketed = pd.cut(pick_prob - BREAKEVEN, bins=EDGE_BUCKETS, include_lowest=True, right=False)

    rows: list[dict[str, object]] = []
    for bucket, idx in bucketed.groupby(bucketed, observed=False).groups.items():
        if len(idx) == 0:
            continue
        nongrp = ~push.loc[idx]
        rows.append(
            {
                "season": season,
                "option": option_label,
                "bucket_kind": bucket_kind,
                "bucket": str(bucket),
                "n": int(len(idx)),
                "avg_pick_prob": float(pick_prob.loc[idx].mean()),
                "win_rate": float(pick_win.loc[idx][nongrp].mean()) if np.any(nongrp) else float("nan"),
                "roi": float(picked_profit.loc[idx].mean()),
            }
        )
    return rows


def _top_sets(df: pd.DataFrame, p_home_cover: np.ndarray) -> dict[int, set[int]]:
    pick_prob = pd.Series(_pick_probability(df, p_home_cover), index=df.index)
    top_sets: dict[int, set[int]] = {}
    for top_n in TOP_NS:
        top_sets[top_n] = set(pick_prob.nlargest(min(top_n, len(df))).index.tolist())
    return top_sets


def _serialize_calibration(
    calibration: dict[str, object],
    *,
    option: OptionSpec,
    season: int,
    train_seasons: list[int],
) -> dict[str, object]:
    train_seasons_int = [int(season_id) for season_id in train_seasons]
    row = {
        "season": season,
        "option": option.label,
        "base_score": option.base_score,
        "method": calibration["method"],
        "train_seasons": json.dumps(train_seasons_int),
    }
    for key, value in calibration.items():
        if isinstance(value, list):
            row[key] = json.dumps(value)
        else:
            row[key] = value
    return row


@click.command()
@click.option(
    "--benchmark-dir",
    default="artifacts/benchmarks/canonical_walkforward_v2_lgb",
    type=click.Path(path_type=Path),
    help="Canonical walk-forward benchmark artifact directory.",
)
@click.option(
    "--sigma-study-dir",
    default="artifacts/sigma_calibration_study_v1",
    type=click.Path(path_type=Path),
    help="Sigma-study artifact directory used to define the best current heuristic.",
)
@click.option(
    "--output-dir",
    default="artifacts/cover_probability_calibration_study_v1",
    type=click.Path(path_type=Path),
    help="Where to write probability-calibration study artifacts.",
)
def main(benchmark_dir: Path, sigma_study_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "calibrators").mkdir(exist_ok=True)

    df = _load_dataset(benchmark_dir, sigma_study_dir)
    option_specs = _option_specs()

    fold_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    season_prob_lookup: dict[tuple[int, str], np.ndarray] = {}

    for season in EVAL_SEASONS:
        eval_df = df[df["season"] == season].copy().reset_index(drop=True)
        train_seasons = [past for past in sorted(df["season"].unique()) if past < season]
        train_df = df[df["season"].isin(train_seasons)].copy()

        for option in option_specs:
            calibration = _fit_calibration(train_df, option)
            z_eval = eval_df[option.base_score].to_numpy(float)
            p_home_cover = apply_probability_calibration(z_eval, calibration)
            season_prob_lookup[(season, option.label)] = p_home_cover

            metrics = _evaluate_option(eval_df, p_home_cover)
            fold_rows.append(
                {
                    "season": season,
                    "option": option.label,
                    "base_score": option.base_score,
                    "calibration_method": calibration["method"],
                    **metrics,
                }
            )
            bucket_rows.extend(
                _bucket_rows(
                    eval_df,
                    p_home_cover,
                    bucket_kind="probability",
                    option_label=option.label,
                    season=season,
                )
            )
            bucket_rows.extend(
                _bucket_rows(
                    eval_df,
                    p_home_cover,
                    bucket_kind="edge",
                    option_label=option.label,
                    season=season,
                )
            )
            calibration_rows.append(
                _serialize_calibration(
                    calibration,
                    option=option,
                    season=season,
                    train_seasons=train_seasons,
                )
            )

            option_pred_dir = output_dir / "predictions" / option.label
            option_pred_dir.mkdir(parents=True, exist_ok=True)
            pred_out = eval_df[
                [
                    "gameId",
                    "season",
                    "startDate",
                    "homeTeam",
                    "awayTeam",
                    "actual_margin",
                    "pred_margin_mu",
                    "book_spread",
                    "actual_edge_home",
                    "push",
                    "phase",
                ]
            ].copy()
            pred_out["p_home_cover"] = p_home_cover
            pred_out["pick_prob"] = _pick_probability(eval_df, p_home_cover)
            pred_out["pick_prob_edge"] = pred_out["pick_prob"] - BREAKEVEN
            pred_out.to_parquet(option_pred_dir / f"season_{season}.parquet", index=False)

            calib_dir = output_dir / "calibrators" / option.label
            calib_dir.mkdir(parents=True, exist_ok=True)
            (calib_dir / f"season_{season}.json").write_text(
                json.dumps(
                    {
                        "season": season,
                        "option": option.label,
                        "base_score": option.base_score,
                        "train_seasons": [int(season_id) for season_id in train_seasons],
                        "calibration": calibration,
                    },
                    indent=2,
                )
            )

    fold_df = pd.DataFrame(fold_rows)
    calibration_df = pd.DataFrame(calibration_rows)

    overlap_rows: list[dict[str, object]] = []
    for season in EVAL_SEASONS:
        eval_df = df[df["season"] == season].copy().reset_index(drop=True)
        raw_top = _top_sets(eval_df, season_prob_lookup[(season, "raw_mlp_normal")])
        best_top = _top_sets(eval_df, season_prob_lookup[(season, "best_posthoc_normal")])
        for option in option_specs:
            top_sets = _top_sets(eval_df, season_prob_lookup[(season, option.label)])
            row = {"season": season, "option": option.label}
            for top_n in TOP_NS:
                row[f"top{top_n}_overlap_vs_raw"] = len(top_sets[top_n] & raw_top[top_n])
                row[f"top{top_n}_churn_vs_raw"] = len(top_sets[top_n].symmetric_difference(raw_top[top_n]))
                row[f"top{top_n}_overlap_vs_best"] = len(top_sets[top_n] & best_top[top_n])
                row[f"top{top_n}_churn_vs_best"] = len(top_sets[top_n].symmetric_difference(best_top[top_n]))
            overlap_rows.append(row)
    overlap_df = pd.DataFrame(overlap_rows)
    fold_df = fold_df.merge(overlap_df, on=["season", "option"], how="left")

    pooled_rows: list[dict[str, object]] = []
    for option, grp in fold_df.groupby("option"):
        row = {"option": option}
        for col in [
            "cover_logloss",
            "cover_brier",
            "pick_prob_mean",
            "pick_prob_std",
            "pick_prob_edge_mean",
            "prob80_winrate",
            "prob80_roi",
            "prob85_winrate",
            "prob85_roi",
            "edge10_roi",
            "edge15_roi",
            "top100_roi",
            "top200_roi",
            "top500_roi",
            "top100_winrate",
            "top200_winrate",
            "top500_winrate",
            "top100_avg_prob",
            "top200_avg_prob",
            "top500_avg_prob",
            "top100_overlap_vs_raw",
            "top200_overlap_vs_raw",
            "top500_overlap_vs_raw",
            "top100_overlap_vs_best",
            "top200_overlap_vs_best",
            "top500_overlap_vs_best",
            "top100_churn_vs_raw",
            "top200_churn_vs_raw",
            "top500_churn_vs_raw",
            "top100_churn_vs_best",
            "top200_churn_vs_best",
            "top500_churn_vs_best",
        ]:
            row[col] = float(np.nanmean(grp[col]))
        pooled_rows.append(row)
    pooled_df = pd.DataFrame(pooled_rows).sort_values(
        ["top200_roi", "cover_logloss", "cover_brier"],
        ascending=[False, True, True],
    )

    dataset_summary = {
        "benchmark_dir": str(benchmark_dir),
        "sigma_study_dir": str(sigma_study_dir),
        "rows_lined": int(len(df)),
        "rows_nonpush": int((~df["push"]).sum()),
        "evaluation_seasons": EVAL_SEASONS,
        "base_scores": {
            "z_raw": "edge_home_points / raw_mlp_sigma",
            "z_best": "edge_home_points / best current sigma-study heuristic",
            "z_const14": "edge_home_points / 14.0",
        },
        "options": [option.label for option in option_specs],
        "metrics_note": {
            "cover_logloss_and_brier": "non-push lined games only",
            "roi_metrics": "pushes count as 0 profit",
        },
    }

    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    pooled_df.to_csv(output_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(output_dir / "bucket_metrics.csv", index=False)
    calibration_df.to_csv(output_dir / "selected_calibrators.csv", index=False)

    summary_lines = [
        "# Cover Probability Calibration Study",
        "",
        f"- Benchmark dir: `{benchmark_dir}`",
        f"- Sigma-study dir: `{sigma_study_dir}`",
        f"- Lined rows: `{len(df)}`",
        f"- Non-push rows: `{int((~df['push']).sum())}`",
        f"- Evaluation seasons: `{EVAL_SEASONS}`",
        "",
        "## Pooled Results",
        "",
        pooled_df.to_string(index=False),
        "",
        "## Per-Season Results",
        "",
        fold_df.to_string(index=False),
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines))

    click.echo(f"Probability calibration study written to {output_dir}")


if __name__ == "__main__":
    main()

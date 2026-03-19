#!/usr/bin/env python3
"""Post-hoc sigma calibration study on canonical walk-forward lined holdouts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.sigma_calibration import apply_sigma_transform

BREAKEVEN = 110.0 / 210.0
WIN_PROFIT = 100.0 / 110.0
PROB_BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
EDGE_BUCKETS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.00]
TOP_NS = [100, 200, 500]


@dataclass(frozen=True)
class TransformSpec:
    family: str
    params: dict[str, float]

    @property
    def label(self) -> str:
        if self.family == "raw":
            return "raw"
        if self.family == "const":
            return f"const_{self.params['sigma_const']:.2f}"
        if self.family == "cap":
            return f"cap_{self.params['cap_max']:.2f}"
        if self.family == "scale":
            return f"scale_{self.params['scale']:.3f}"
        if self.family == "affine":
            return f"affine_a{self.params['affine_a']:.2f}_b{self.params['affine_b']:.3f}"
        if self.family == "shrink":
            return (
                f"shrink_alpha{self.params['shrink_alpha']:.2f}"
                f"_to_{self.params['shrink_target']:.2f}"
            )
        raise ValueError(self.family)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(erf)
    return 0.5 * (1.0 + erf_vec(x / sqrt(2.0)))


def _load_study_dataset(benchmark_dir: Path) -> pd.DataFrame:
    pred_dir = benchmark_dir / "predictions"
    frames: list[pd.DataFrame] = []
    for season_path in sorted((pred_dir / "HistGradientBoosting").glob("season_*.parquet")):
        season = int(season_path.stem.split("_")[1])
        hgb = pd.read_parquet(season_path)
        mlp = pd.read_parquet(pred_dir / "CurrentMLP" / f"season_{season}.parquet")
        merged = hgb[
            ["gameId", "startDate", "actual_margin", "pred_margin", "book_spread"]
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
    df["home_cover_win"] = df["actual_edge_home"] > 0
    df["push"] = df["actual_edge_home"] == 0
    return df.reset_index(drop=True)


def _candidate_specs() -> list[TransformSpec]:
    specs: list[TransformSpec] = [TransformSpec("raw", {})]
    specs.extend(
        TransformSpec("const", {"sigma_const": sigma_const})
        for sigma_const in np.arange(11.0, 14.01, 0.25)
    )
    specs.extend(
        TransformSpec("cap", {"cap_max": cap_max})
        for cap_max in np.arange(13.0, 17.01, 0.25)
    )
    specs.extend(
        TransformSpec("scale", {"scale": scale})
        for scale in np.arange(0.85, 1.001, 0.01)
    )
    specs.extend(
        TransformSpec("affine", {"affine_a": affine_a, "affine_b": affine_b})
        for affine_a in np.arange(0.0, 3.01, 0.25)
        for affine_b in np.arange(0.75, 1.001, 0.025)
    )
    specs.extend(
        TransformSpec(
            "shrink",
            {"shrink_alpha": alpha, "shrink_target": target},
        )
        for alpha in np.arange(0.25, 1.001, 0.05)
        for target in np.arange(11.0, 14.01, 0.25)
    )
    return specs


def _transform_sigma(raw_sigma: pd.Series, spec: TransformSpec) -> np.ndarray:
    if spec.family == "raw":
        return apply_sigma_transform(raw_sigma.to_numpy(float), mode="raw")
    if spec.family == "const":
        return np.full(len(raw_sigma), spec.params["sigma_const"], dtype=float)
    if spec.family == "cap":
        return apply_sigma_transform(raw_sigma.to_numpy(float), mode="cap", **spec.params)
    if spec.family == "scale":
        return apply_sigma_transform(raw_sigma.to_numpy(float), mode="scale", **spec.params)
    if spec.family == "affine":
        return apply_sigma_transform(raw_sigma.to_numpy(float), mode="affine", **spec.params)
    if spec.family == "shrink":
        return apply_sigma_transform(raw_sigma.to_numpy(float), mode="shrink", **spec.params)
    raise ValueError(spec.family)


def _cover_metrics(df: pd.DataFrame, sigma_t: np.ndarray) -> dict[str, float]:
    edge_home_points = df["pred_margin_mu"].to_numpy(float) + df["book_spread"].to_numpy(float)
    p_home_cover = np.clip(_norm_cdf(edge_home_points / sigma_t), 1e-9, 1.0 - 1e-9)
    y = df["home_cover_win"].astype(float).to_numpy()
    cover_logloss = float(-(y * np.log(p_home_cover) + (1.0 - y) * np.log(1.0 - p_home_cover)).mean())
    cover_brier = float(np.mean((p_home_cover - y) ** 2))

    err = df["actual_margin"].to_numpy(float) - df["pred_margin_mu"].to_numpy(float)
    abs_err = np.abs(err)
    gaussian_nll = float(
        np.mean(0.5 * np.log(2.0 * np.pi * sigma_t**2) + (err**2) / (2.0 * sigma_t**2))
    )
    z_abs = abs_err / sigma_t

    pick_home = edge_home_points >= 0.0
    pick_prob = np.where(pick_home, p_home_cover, 1.0 - p_home_cover)
    pick_win = np.where(pick_home, df["actual_edge_home"] > 0, df["actual_edge_home"] < 0)
    push = df["push"].to_numpy(bool)
    picked_profit = np.where(push, 0.0, np.where(pick_win, WIN_PROFIT, -1.0))

    out = {
        "gaussian_nll": gaussian_nll,
        "cover_logloss": cover_logloss,
        "cover_brier": cover_brier,
        "mean_sigma": float(np.mean(sigma_t)),
        "std_sigma": float(np.std(sigma_t)),
        "mean_abs_z": float(np.mean(z_abs)),
        "cov1": float(np.mean(abs_err <= sigma_t)),
        "cov2": float(np.mean(abs_err <= 2.0 * sigma_t)),
        "pick_prob_mean": float(np.mean(pick_prob)),
        "pick_prob_edge_mean": float(np.mean(pick_prob - BREAKEVEN)),
        "pick_prob_85_plus_n": int(np.sum(pick_prob >= 0.85)),
        "pick_prob_85_plus_roi": float(np.mean(picked_profit[pick_prob >= 0.85])) if np.any(pick_prob >= 0.85) else float("nan"),
        "edge_15_plus_n": int(np.sum((pick_prob - BREAKEVEN) >= 0.15)),
        "edge_15_plus_roi": float(np.mean(picked_profit[(pick_prob - BREAKEVEN) >= 0.15])) if np.any((pick_prob - BREAKEVEN) >= 0.15) else float("nan"),
    }
    for top_n in TOP_NS:
        top = np.argsort(-pick_prob)[: min(top_n, len(df))]
        chosen = np.zeros(len(df), dtype=bool)
        chosen[top] = True
        wins = pick_win[chosen & ~push]
        out[f"top{top_n}_roi"] = float(np.mean(picked_profit[chosen]))
        out[f"top{top_n}_winrate"] = float(np.mean(wins)) if len(wins) else float("nan")
        out[f"top{top_n}_avg_prob"] = float(np.mean(pick_prob[chosen]))
    return out


def _bucket_rows(df: pd.DataFrame, sigma_t: np.ndarray, *, bucket_kind: str, option_label: str, season: int) -> list[dict]:
    edge_home_points = df["pred_margin_mu"].to_numpy(float) + df["book_spread"].to_numpy(float)
    p_home_cover = _norm_cdf(edge_home_points / sigma_t)
    pick_home = edge_home_points >= 0.0
    pick_prob = pd.Series(np.where(pick_home, p_home_cover, 1.0 - p_home_cover), index=df.index)
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

    rows: list[dict] = []
    for bucket, idx in bucketed.groupby(bucketed, observed=False).groups.items():
        if len(idx) == 0:
            continue
        grp = df.loc[idx]
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


def _select_best_spec(train_df: pd.DataFrame, family: str, specs: list[TransformSpec]) -> TransformSpec:
    family_specs = [spec for spec in specs if spec.family == family]
    if family == "raw":
        return family_specs[0]

    best_spec = family_specs[0]
    best_score: tuple[float, float, float] | None = None
    for spec in family_specs:
        sigma_t = _transform_sigma(train_df["sigma"], spec)
        metrics = _cover_metrics(train_df, sigma_t)
        score = (
            metrics["cover_logloss"],
            metrics["gaussian_nll"],
            metrics["mean_sigma"],
        )
        if best_score is None or score < best_score:
            best_score = score
            best_spec = spec
    return best_spec


def _season_protocol_summary(seasons: list[int]) -> list[dict[str, object]]:
    rows = []
    for idx, season in enumerate(seasons):
        train_seasons = seasons[:idx]
        rows.append(
            {
                "eval_season": season,
                "train_seasons": train_seasons,
                "used_for_parameter_fit": bool(train_seasons),
            }
        )
    return rows


@click.command()
@click.option(
    "--benchmark-dir",
    default="artifacts/benchmarks/canonical_walkforward_v2_lgb",
    type=click.Path(path_type=Path),
    help="Canonical walk-forward benchmark artifact directory.",
)
@click.option(
    "--output-dir",
    default="artifacts/sigma_calibration_study_v1",
    type=click.Path(path_type=Path),
    help="Where to write sigma study artifacts.",
)
def main(benchmark_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_study_dataset(benchmark_dir)
    seasons = sorted(df["season"].unique().tolist())
    specs = _candidate_specs()
    families = ["raw", "const", "cap", "scale", "affine", "shrink"]

    fold_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    selected_specs: list[dict[str, object]] = []
    eval_frames: list[pd.DataFrame] = []

    for idx, season in enumerate(seasons):
        eval_df = df[df["season"] == season].copy()
        train_df = df[df["season"].isin(seasons[:idx])].copy()
        if train_df.empty:
            # First season is descriptive only; keep raw as fallback.
            for family in families:
                chosen = TransformSpec("raw", {}) if family == "raw" else _select_best_spec(df[df["season"] != season], family, specs)
                sigma_t = _transform_sigma(eval_df["sigma"], chosen)
                metrics = _cover_metrics(eval_df, sigma_t)
                fold_rows.append({"season": season, "family": family, "selected_label": chosen.label, **chosen.params, **metrics})
            continue

        for family in families:
            chosen = _select_best_spec(train_df, family, specs)
            sigma_t = _transform_sigma(eval_df["sigma"], chosen)
            metrics = _cover_metrics(eval_df, sigma_t)
            fold_rows.append({"season": season, "family": family, "selected_label": chosen.label, **chosen.params, **metrics})
            bucket_rows.extend(_bucket_rows(eval_df, sigma_t, bucket_kind="probability", option_label=family, season=season))
            bucket_rows.extend(_bucket_rows(eval_df, sigma_t, bucket_kind="edge", option_label=family, season=season))
            if family == "raw":
                raw_probs = _norm_cdf((eval_df["pred_margin_mu"].to_numpy(float) + eval_df["book_spread"].to_numpy(float)) / sigma_t)
                eval_base = eval_df.copy()
                eval_base["raw_pick_prob"] = np.where(
                    (eval_df["pred_margin_mu"].to_numpy(float) + eval_df["book_spread"].to_numpy(float)) >= 0.0,
                    raw_probs,
                    1.0 - raw_probs,
                )
                eval_frames.append(eval_base)

    fold_df = pd.DataFrame(fold_rows)
    raw_top_lookup = {
        season: set(
            eval_frame["raw_pick_prob"].nlargest(min(200, len(eval_frame))).index.tolist()
        )
        for season, eval_frame in zip(seasons[1:], eval_frames)
    }

    # Add churn versus raw for comparable eval seasons.
    churn_rows: list[dict[str, object]] = []
    for season in seasons[1:]:
        eval_df = df[df["season"] == season].copy()
        raw_sigma = _transform_sigma(eval_df["sigma"], TransformSpec("raw", {}))
        raw_probs = _norm_cdf((eval_df["pred_margin_mu"].to_numpy(float) + eval_df["book_spread"].to_numpy(float)) / raw_sigma)
        raw_pick_prob = np.where(
            (eval_df["pred_margin_mu"].to_numpy(float) + eval_df["book_spread"].to_numpy(float)) >= 0.0,
            raw_probs,
            1.0 - raw_probs,
        )
        raw_top = set(pd.Series(raw_pick_prob, index=eval_df.index).nlargest(min(200, len(eval_df))).index.tolist())

        for family in families:
            chosen_row = fold_df[(fold_df["season"] == season) & (fold_df["family"] == family)].iloc[0]
            spec = TransformSpec(
                family="raw" if family == "raw" else family,
                params={k: chosen_row[k] for k in ["sigma_const", "cap_max", "scale", "affine_a", "affine_b", "shrink_alpha", "shrink_target"] if k in chosen_row and not pd.isna(chosen_row[k])},
            )
            sigma_t = _transform_sigma(eval_df["sigma"], spec)
            probs = _norm_cdf((eval_df["pred_margin_mu"].to_numpy(float) + eval_df["book_spread"].to_numpy(float)) / sigma_t)
            pick_prob = np.where(
                (eval_df["pred_margin_mu"].to_numpy(float) + eval_df["book_spread"].to_numpy(float)) >= 0.0,
                probs,
                1.0 - probs,
            )
            top = set(pd.Series(pick_prob, index=eval_df.index).nlargest(min(200, len(eval_df))).index.tolist())
            churn_rows.append(
                {
                    "season": season,
                    "family": family,
                    "top200_overlap_vs_raw": len(raw_top & top),
                    "top200_churn_vs_raw": len(raw_top.symmetric_difference(top)),
                }
            )

    churn_df = pd.DataFrame(churn_rows)
    fold_df = fold_df.merge(churn_df, on=["season", "family"], how="left")

    eval_only = fold_df[fold_df["season"].isin(seasons[1:])].copy()
    pooled_rows: list[dict[str, object]] = []
    for family, grp in eval_only.groupby("family"):
        n = np.ones(len(grp))
        row = {"family": family}
        for col in [
            "gaussian_nll",
            "cover_logloss",
            "cover_brier",
            "mean_sigma",
            "std_sigma",
            "mean_abs_z",
            "cov1",
            "cov2",
            "pick_prob_mean",
            "pick_prob_edge_mean",
            "pick_prob_85_plus_roi",
            "edge_15_plus_roi",
            "top100_roi",
            "top200_roi",
            "top500_roi",
            "top100_winrate",
            "top200_winrate",
            "top500_winrate",
            "top100_avg_prob",
            "top200_avg_prob",
            "top500_avg_prob",
            "top200_overlap_vs_raw",
            "top200_churn_vs_raw",
        ]:
            row[col] = float(np.nanmean(grp[col]))
        row["season_wins_vs_raw_top200_roi"] = int(
            (
                grp.set_index("season")["top200_roi"]
                > eval_only[eval_only["family"] == "raw"].set_index("season")["top200_roi"]
            ).sum()
        )
        pooled_rows.append(row)
    pooled_df = pd.DataFrame(pooled_rows).sort_values(
        ["top200_roi", "cover_logloss", "mean_sigma"],
        ascending=[False, True, True],
    )

    # Final fit on all seasons for deployment candidate per family.
    for family in families:
        chosen = _select_best_spec(df, family, specs)
        selected_specs.append(
            {
                "family": family,
                "selected_label": chosen.label,
                **chosen.params,
            }
        )
    selected_df = pd.DataFrame(selected_specs)

    # Recommend a practical winner: high top200 ROI, good proper score, sharper than raw.
    raw_row = pooled_df[pooled_df["family"] == "raw"].iloc[0]
    contenders = pooled_df[
        (pooled_df["top200_roi"] >= raw_row["top200_roi"])
        & (pooled_df["cover_logloss"] <= raw_row["cover_logloss"] + 0.003)
        & (pooled_df["mean_sigma"] <= raw_row["mean_sigma"])
    ].copy()
    winner_family = contenders.sort_values(
        ["top200_roi", "cover_logloss", "mean_sigma"],
        ascending=[False, True, True],
    ).iloc[0]["family"]
    winner_spec = selected_df[selected_df["family"] == winner_family].iloc[0].to_dict()

    # Persist outputs.
    dataset_summary = {
        "benchmark_dir": str(benchmark_dir),
        "rows_lined": int(len(df)),
        "seasons": seasons,
        "protocol": {
            "selection": "rolling prior-season fit on lined holdout seasons only",
            "evaluation_seasons": seasons[1:],
            "season_protocol": _season_protocol_summary(seasons),
            "top_ns": TOP_NS,
        },
        "winner_family": winner_family,
        "winner_spec": winner_spec,
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    pooled_df.to_csv(output_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(output_dir / "bucket_metrics.csv", index=False)
    selected_df.to_csv(output_dir / "selected_transforms.csv", index=False)

    summary_lines = [
        "# Sigma Calibration Study",
        "",
        f"- Benchmark dir: `{benchmark_dir}`",
        f"- Lined holdout rows: `{len(df)}`",
        f"- Evaluation seasons: `{seasons[1:]}`",
        f"- Winner family: `{winner_family}`",
        f"- Winner selected transform: `{winner_spec['selected_label']}`",
        "",
        "## Pooled Results",
        "",
        pooled_df.to_string(index=False),
        "",
        "## Final Selected Transforms",
        "",
        selected_df.to_string(index=False),
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines))

    click.echo(f"Sigma study written to {output_dir}")
    click.echo(f"Winner family: {winner_family}")
    click.echo(f"Winner transform: {winner_spec['selected_label']}")


if __name__ == "__main__":
    main()

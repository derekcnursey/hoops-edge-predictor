#!/usr/bin/env python3
"""Downstream promotion audit for LightGBMRegressionL2Blend.

Compares the current blended absolute-error HGBR mean path against the L2 blend
using the same repaired-line sigma framework already trusted for ATS evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.sigma_calibration import apply_sigma_transform

ROOT = Path(__file__).resolve().parent.parent
OBJECTIVE_DIR = ROOT / "artifacts" / "research" / "objective_tail_compression_experiment_v1"
SIGMA_STUDY_DIR = ROOT / "artifacts" / "sigma_calibration_study_repaired_v1"
BENCHMARK_2019_2025 = ROOT / "artifacts" / "benchmarks" / "canonical_walkforward_v2_lgb_repaired_lines_neutralfix"
BENCHMARK_2026 = ROOT / "artifacts" / "benchmarks" / "canonical_walkforward_v2_lgb_repaired_lines_2026_neutralfix"
OUT_DIR = ROOT / "artifacts" / "research" / "objective_tail_promotion_audit_v1"

VARIANTS = [
    "HistGradientBoostingAbsoluteBlend",
    "LightGBMRegressionL2Blend",
]
BREAKEVEN = 110.0 / 210.0
WIN_PROFIT = 100.0 / 110.0
TOP_NS = [100, 200, 500]
PROB_BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
EDGE_BUCKETS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.00]


def _load_sigma_spec() -> dict[str, object]:
    obj = json.loads((SIGMA_STUDY_DIR / "dataset_summary.json").read_text())
    spec = dict(obj["winner_spec"])
    return {
        key: value
        for key, value in spec.items()
        if key != "selected_label"
        and value is not None
        and not (isinstance(value, float) and math.isnan(value))
    }


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _is_dec15_plus(date_series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(date_series, errors="coerce", utc=True).dt.tz_convert("America/New_York")
    mmdd = dt.dt.strftime("%m-%d")
    return (((dt.dt.month >= 1) & (dt.dt.month <= 3)) | (mmdd >= "12-15")).to_numpy()


def _load_sigma_frames() -> pd.DataFrame:
    frames = []
    for bench_dir in [BENCHMARK_2019_2025, BENCHMARK_2026]:
        pred_dir = bench_dir / "predictions" / "CurrentMLP"
        if not pred_dir.exists():
            continue
        for path in sorted(pred_dir.glob("season_*.parquet")):
            season = int(path.stem.split("_")[1])
            df = pd.read_parquet(path)[["gameId", "startDate", "sigma"]].copy()
            df["holdout_season"] = season
            frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["holdout_season", "gameId"])


def _load_variant_frame(label: str) -> pd.DataFrame:
    path = OBJECTIVE_DIR / f"{label}_predictions.parquet"
    df = pd.read_parquet(path).copy()
    season_map = (
        pd.to_datetime(df["startDate"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.year
    )
    if "holdout_season" not in df.columns:
        df["holdout_season"] = np.nan
    missing = df["holdout_season"].isna()
    if missing.any():
        dt = pd.to_datetime(df.loc[missing, "startDate"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
        df.loc[missing, "holdout_season"] = np.where(dt.dt.month >= 11, dt.dt.year + 1, dt.dt.year)
    df["holdout_season"] = df["holdout_season"].astype(int)
    return df


def _favorite_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    data = df[df["book_spread"].notna()].copy()
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


def _evaluate(df: pd.DataFrame) -> dict[str, float]:
    lined_df = df[df["book_spread"].notna()].copy().reset_index(drop=True)
    nonpush = ~lined_df["push"].to_numpy(bool)
    y = lined_df.loc[nonpush, "home_cover_win"].astype(float).to_numpy()
    p_nonpush = np.clip(lined_df.loc[nonpush, "p_home_cover"].to_numpy(float), 1e-6, 1.0 - 1e-6)
    cover_logloss = float(-(y * np.log(p_nonpush) + (1.0 - y) * np.log(1.0 - p_nonpush)).mean())
    cover_brier = float(np.mean((p_nonpush - y) ** 2))

    pick_prob = lined_df["pick_prob"].to_numpy(float)
    pick_win = lined_df["pick_win"].to_numpy(bool)
    push = lined_df["push"].to_numpy(bool)
    picked_profit = np.where(push, 0.0, np.where(pick_win, WIN_PROFIT, -1.0))
    pick_prob_edge = pick_prob - BREAKEVEN

    out = {
        "rows": int(len(df)),
        "lined_rows": int(len(lined_df)),
        "cover_logloss": cover_logloss,
        "cover_brier": cover_brier,
        "all_pick_winrate": float(np.mean(pick_win[~push])) if np.any(~push) else float("nan"),
        "all_pick_roi": float(np.mean(picked_profit)),
        "pick_prob_edge_mean": float(np.mean(pick_prob_edge)),
        "prob80_n": int(np.sum(pick_prob >= 0.80)),
        "prob80_roi": float(np.mean(picked_profit[pick_prob >= 0.80])) if np.any(pick_prob >= 0.80) else float("nan"),
        "prob85_n": int(np.sum(pick_prob >= 0.85)),
        "prob85_roi": float(np.mean(picked_profit[pick_prob >= 0.85])) if np.any(pick_prob >= 0.85) else float("nan"),
        "edge10_n": int(np.sum(pick_prob_edge >= 0.10)),
        "edge10_roi": float(np.mean(picked_profit[pick_prob_edge >= 0.10])) if np.any(pick_prob_edge >= 0.10) else float("nan"),
        "edge15_n": int(np.sum(pick_prob_edge >= 0.15)),
        "edge15_roi": float(np.mean(picked_profit[pick_prob_edge >= 0.15])) if np.any(pick_prob_edge >= 0.15) else float("nan"),
    }
    for top_n in TOP_NS:
        idx = np.argsort(-pick_prob)[: min(top_n, len(lined_df))]
        chosen = np.zeros(len(lined_df), dtype=bool)
        chosen[idx] = True
        wins = pick_win[chosen & ~push]
        out[f"top{top_n}_roi"] = float(np.mean(picked_profit[chosen]))
        out[f"top{top_n}_winrate"] = float(np.mean(wins)) if len(wins) else float("nan")
        out[f"top{top_n}_avg_prob"] = float(np.mean(pick_prob[chosen]))
    return out


def _bucket_rows(df: pd.DataFrame, variant: str, kind: str) -> list[dict[str, object]]:
    lined_df = df[df["book_spread"].notna()].copy().reset_index(drop=True)
    picked_profit = np.where(lined_df["push"], 0.0, np.where(lined_df["pick_win"], WIN_PROFIT, -1.0))
    if kind == "probability":
        bucketed = pd.cut(lined_df["pick_prob"], bins=PROB_BUCKETS, include_lowest=True, right=False)
    else:
        bucketed = pd.cut(lined_df["pick_prob"] - BREAKEVEN, bins=EDGE_BUCKETS, include_lowest=True, right=False)
    rows = []
    for bucket, idx in bucketed.groupby(bucketed, observed=False).groups.items():
        if len(idx) == 0:
            continue
        nongrp = ~lined_df.loc[idx, "push"]
        rows.append(
            {
                "variant": variant,
                "bucket_kind": kind,
                "bucket": str(bucket),
                "n": int(len(idx)),
                "avg_pick_prob": float(lined_df.loc[idx, "pick_prob"].mean()),
                "win_rate": float(lined_df.loc[idx, "pick_win"][nongrp].mean()) if np.any(nongrp) else float("nan"),
                "roi": float(np.mean(picked_profit[idx])),
            }
        )
    return rows


def _render_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "```"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sigma_spec = _load_sigma_spec()
    sigma_frames = _load_sigma_frames()

    protocol = {
        "baseline_variant": VARIANTS[0],
        "candidate_variant": VARIANTS[1],
        "sigma_spec": sigma_spec,
        "sigma_source": "CurrentMLP repaired-line benchmark predictions",
        "benchmark_dirs": [str(BENCHMARK_2019_2025), str(BENCHMARK_2026)],
        "objective_experiment_dir": str(OBJECTIVE_DIR),
    }
    (OUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    pooled_rows = []
    dec15_rows = []
    season_rows = []
    favorite_rows = []
    favorite_dec15_rows = []
    bucket_rows = []

    for variant in VARIANTS:
        pred = _load_variant_frame(variant)
        pred = pred.drop(columns=["sigma"], errors="ignore")
        merged = pred.merge(
            sigma_frames,
            on=["holdout_season", "gameId", "startDate"],
            how="inner",
            validate="one_to_one",
        )
        sigma_mode = str(sigma_spec["family"])
        sigma_kwargs = {k: v for k, v in sigma_spec.items() if k != "family"}
        merged["sigma_adj"] = apply_sigma_transform(
            merged["sigma"].to_numpy(float),
            mode=sigma_mode,
            **sigma_kwargs,
        )
        merged["edge_home_points"] = merged["pred_margin"] + merged["book_spread"]
        z = merged["edge_home_points"].to_numpy(float) / merged["sigma_adj"].clip(lower=0.5)
        merged["p_home_cover"] = _normal_cdf(z)
        merged["actual_edge_home"] = merged["actual_margin"] + merged["book_spread"]
        merged["home_cover_win"] = merged["actual_edge_home"] > 0
        merged["push"] = merged["actual_edge_home"] == 0
        merged["pick_home"] = merged["edge_home_points"] >= 0
        merged["pick_prob"] = np.where(merged["pick_home"], merged["p_home_cover"], 1.0 - merged["p_home_cover"])
        merged["pick_win"] = np.where(merged["pick_home"], merged["actual_edge_home"] > 0, merged["actual_edge_home"] < 0)

        merged.to_parquet(OUT_DIR / f"{variant}_downstream.parquet", index=False)

        pooled_rows.append({"variant": variant, **_evaluate(merged)})

        dec_mask = _is_dec15_plus(merged["startDate"])
        dec_df = merged.loc[dec_mask].copy()
        dec15_rows.append({"variant": variant, **_evaluate(dec_df)})

        for season, grp in merged.groupby("holdout_season"):
            season_rows.append(
                {
                    "variant": variant,
                    "holdout_season": int(season),
                    **_evaluate(grp.loc[_is_dec15_plus(grp["startDate"])]),
                }
            )

        fav = _favorite_bucket_summary(merged)
        fav["variant"] = variant
        favorite_rows.append(fav)

        fav_dec = _favorite_bucket_summary(dec_df)
        fav_dec["variant"] = variant
        favorite_dec15_rows.append(fav_dec)

        bucket_rows.extend(_bucket_rows(merged, variant, "probability"))
        bucket_rows.extend(_bucket_rows(merged, variant, "edge"))

    pooled_df = pd.DataFrame(pooled_rows).sort_values("variant").reset_index(drop=True)
    dec15_df = pd.DataFrame(dec15_rows).sort_values("variant").reset_index(drop=True)
    season_df = pd.DataFrame(season_rows).sort_values(["holdout_season", "variant"]).reset_index(drop=True)
    favorite_df = pd.concat(favorite_rows, ignore_index=True)
    favorite_dec15_df = pd.concat(favorite_dec15_rows, ignore_index=True)
    bucket_df = pd.DataFrame(bucket_rows)

    pooled_df.to_csv(OUT_DIR / "pooled_metrics.csv", index=False)
    dec15_df.to_csv(OUT_DIR / "dec15_metrics.csv", index=False)
    season_df.to_csv(OUT_DIR / "season_dec15_metrics.csv", index=False)
    favorite_df.to_csv(OUT_DIR / "favorite_bucket_summary.csv", index=False)
    favorite_dec15_df.to_csv(OUT_DIR / "favorite_bucket_summary_dec15.csv", index=False)
    bucket_df.to_csv(OUT_DIR / "bucket_metrics.csv", index=False)

    summary_lines = [
        "# Objective Tail Promotion Audit",
        "",
        "## Pooled Downstream Metrics",
        "",
        _render_table(pooled_df),
        "",
        "## Dec 15+ Downstream Metrics",
        "",
        _render_table(dec15_df),
        "",
        "## Dec 15+ By Season",
        "",
        _render_table(season_df),
        "",
        "## Favorite Buckets (Pooled)",
        "",
        _render_table(favorite_df),
        "",
        "## Favorite Buckets (Dec 15+)",
        "",
        _render_table(favorite_dec15_df),
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"Saved downstream promotion audit to {OUT_DIR}")


if __name__ == "__main__":
    main()

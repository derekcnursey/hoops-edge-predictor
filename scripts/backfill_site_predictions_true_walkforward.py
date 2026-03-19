#!/usr/bin/env python3
"""Rebuild site prediction JSONs from true walk-forward benchmark artifacts.

Historical seasons are sourced from the promoted canonical walk-forward bundle:
  - mu from LightGBMRegressionL2Blend holdout predictions
  - sigma from CurrentMLP holdout predictions

Current-season files are left untouched by default so the live site can keep
showing today's slate, while historical performance uses true out-of-sample
predictions only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src import config
from src.efficiency_blend import blend_enabled, gold_weight_for_start_dates
from src.infer import american_profit_per_1, american_to_breakeven, normal_cdf, prob_to_american
from src.sigma_calibration import apply_sigma_transform

BOOK_ODDS = -110
MU_MODEL = "LightGBMRegressionL2Blend"
SIGMA_MODEL = "CurrentMLP"
DEFAULT_BENCHMARK_DIR = (
    config.ARTIFACTS_DIR
    / "benchmarks"
    / "canonical_walkforward_lgb_l2_blend_repaired_lines_neutralfix"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild site historical predictions from the canonical walk-forward benchmark."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help=f"Benchmark artifact directory (default: {DEFAULT_BENCHMARK_DIR}).",
    )
    parser.add_argument(
        "--mu-model-name",
        default=MU_MODEL,
        help=f"Mean-model prediction directory name inside the benchmark bundle (default: {MU_MODEL}).",
    )
    parser.add_argument(
        "--extra-benchmark-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional benchmark directories to merge into the true walk-forward archive.",
    )
    parser.add_argument(
        "--torvik-benchmark-dir",
        type=Path,
        default=None,
        help="Optional Torvik benchmark directory for mu blending.",
    )
    parser.add_argument(
        "--torvik-extra-benchmark-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional Torvik benchmark directories for mu blending.",
    )
    parser.add_argument(
        "--site-data-dir",
        type=Path,
        default=config.SITE_DATA_DIR,
        help=f"Site data directory (default: {config.SITE_DATA_DIR}).",
    )
    parser.add_argument(
        "--keep-current-season",
        action="store_true",
        help="Preserve existing current-season prediction files instead of deleting them.",
    )
    parser.add_argument(
        "--sigma-mode",
        default="raw",
        choices=["raw", "cap", "scale", "affine", "shrink"],
        help="Optional post-hoc sigma transform to apply to benchmark sigma (default: raw).",
    )
    parser.add_argument("--sigma-cap-max", type=float, default=None)
    parser.add_argument("--sigma-scale", type=float, default=None)
    parser.add_argument("--sigma-affine-a", type=float, default=None)
    parser.add_argument("--sigma-affine-b", type=float, default=None)
    parser.add_argument("--sigma-shrink-alpha", type=float, default=None)
    parser.add_argument("--sigma-shrink-target", type=float, default=None)
    return parser.parse_args()


def _season_from_date(date_str: str) -> int:
    year = int(date_str[:4])
    month = int(date_str[5:7])
    return year + 1 if month >= 11 else year


def _current_season() -> int:
    today = datetime.now(ZoneInfo("America/New_York")).date()
    return today.year if today.month <= 6 else today.year + 1


def _slugify(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _to_native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if not np.isnan(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_predictions(benchmark_dirs: list[Path], model_name: str) -> pd.DataFrame:
    frames = []
    for benchmark_dir in benchmark_dirs:
        pred_dir = benchmark_dir / "predictions" / model_name
        for path in sorted(pred_dir.glob("season_*.parquet")):
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No prediction parquet files found for {model_name} in {benchmark_dirs}")
    out = pd.concat(frames, ignore_index=True)
    dedupe_cols = ["holdout_season", "gameId", "startDate", "homeTeam", "awayTeam"]
    out = out.sort_values(dedupe_cols, kind="mergesort").drop_duplicates(
        subset=dedupe_cols,
        keep="last",
    )
    return out.reset_index(drop=True)


def _apply_sigma_mode(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    sigma = df["sigma"].astype(float).to_numpy(copy=True)
    if args.sigma_mode == "raw":
        return pd.Series(sigma, index=df.index, dtype=float)
    transformed = apply_sigma_transform(
        sigma,
        mode=args.sigma_mode,
        cap_max=args.sigma_cap_max,
        scale=args.sigma_scale,
        affine_a=args.sigma_affine_a,
        affine_b=args.sigma_affine_b,
        shrink_alpha=args.sigma_shrink_alpha,
        shrink_target=args.sigma_shrink_target,
    )
    return pd.Series(transformed, index=df.index, dtype=float)


def _build_site_games(merged: pd.DataFrame, game_date: str) -> list[dict]:
    out = merged.copy()
    out["model_mu_home"] = out["pred_margin_mu"].astype(float)
    out["pred_sigma"] = out["sigma_calibrated"].astype(float)

    sigma_safe = out["pred_sigma"].clip(lower=0.5)
    out["pred_home_win_prob"] = normal_cdf(out["model_mu_home"] / sigma_safe)

    has_book = out["book_spread"].notna()
    out["edge_home_points"] = np.where(
        has_book, out["model_mu_home"] + out["book_spread"], np.nan
    )
    out["pick_side"] = np.where(
        has_book,
        np.where(out["edge_home_points"] >= 0, "HOME", "AWAY"),
        None,
    )
    edge_z = np.where(has_book, out["edge_home_points"] / sigma_safe, np.nan)
    home_cover_prob = normal_cdf(edge_z)
    away_cover_prob = 1.0 - home_cover_prob
    out["pick_cover_prob"] = np.where(
        has_book,
        np.where(out["edge_home_points"] >= 0, home_cover_prob, away_cover_prob),
        np.nan,
    )
    breakeven = float(american_to_breakeven(np.array([BOOK_ODDS]))[0])
    profit = float(american_profit_per_1(np.array([BOOK_ODDS]))[0])
    out["pick_spread_odds"] = np.where(has_book, BOOK_ODDS, np.nan)
    out["pick_prob_edge"] = np.where(has_book, out["pick_cover_prob"] - breakeven, np.nan)
    out["pick_ev_per_1"] = np.where(
        has_book,
        out["pick_cover_prob"] * profit - (1.0 - out["pick_cover_prob"]),
        np.nan,
    )
    fair_odds = prob_to_american(out["pick_cover_prob"].fillna(0.5).values)
    out["pick_fair_odds"] = np.where(has_book, fair_odds, np.nan)

    games = []
    ordered = out.sort_values(["startDate", "homeTeam", "awayTeam"]).reset_index(drop=True)
    for rec in ordered.to_dict(orient="records"):
        away = str(rec["awayTeam"])
        home = str(rec["homeTeam"])
        game = {
            "away_team": away,
            "edge_home_points": _to_native(rec["edge_home_points"]),
            "game_id": _slugify(f"{game_date}_{away}_{home}"),
            "home_team": home,
            "market_spread_home": _to_native(rec["book_spread"]),
            "model_mu_home": _to_native(rec["model_mu_home"]),
            "pick_cover_prob": _to_native(rec["pick_cover_prob"]),
            "pick_ev_per_1": _to_native(rec["pick_ev_per_1"]),
            "pick_fair_odds": _to_native(rec["pick_fair_odds"]),
            "pick_prob_edge": _to_native(rec["pick_prob_edge"]),
            "pick_side": rec["pick_side"],
            "pick_spread_odds": _to_native(rec["pick_spread_odds"]),
            "pred_home_win_prob": _to_native(rec["pred_home_win_prob"]),
            "pred_sigma": _to_native(rec["pred_sigma"]),
            "start_time": rec["startDate"],
        }
        games.append(game)
    return games


def _clear_historical_predictions(site_data_dir: Path, keep_current_season: bool) -> int:
    current = _current_season()
    deleted = 0
    for path in sorted(site_data_dir.glob("predictions_*.json")):
        date_str = path.stem.replace("predictions_", "")
        season = _season_from_date(date_str)
        if keep_current_season and season >= current:
            continue
        path.unlink()
        deleted += 1
    return deleted


def main() -> int:
    args = _parse_args()
    args.site_data_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dirs = [args.benchmark_dir, *args.extra_benchmark_dir]

    mu_df = _load_predictions(benchmark_dirs, args.mu_model_name)
    sigma_df = _load_predictions(benchmark_dirs, SIGMA_MODEL)

    keys = ["holdout_season", "gameId", "startDate", "homeTeam", "awayTeam", "book_spread"]
    mu_cols = keys + ["pred_margin"]
    sigma_cols = keys + ["sigma"]
    merged = mu_df[mu_cols].merge(
        sigma_df[sigma_cols],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_mu", "_sigma"),
    )
    merged = merged.rename(columns={"pred_margin": "pred_margin_mu"})
    if blend_enabled() and args.torvik_benchmark_dir is not None:
        torvik_dirs = [args.torvik_benchmark_dir, *args.torvik_extra_benchmark_dir]
        torvik_mu_df = _load_predictions(torvik_dirs, MU_MODEL)
        torvik_cols = keys + ["pred_margin"]
        torvik_mu_df = torvik_mu_df[torvik_cols].rename(columns={"pred_margin": "pred_margin_mu_torvik"})
        merged = merged.merge(
            torvik_mu_df,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        gold_w = gold_weight_for_start_dates(merged["startDate"])
        merged["pred_margin_mu"] = (
            gold_w * merged["pred_margin_mu"].astype(float).to_numpy()
            + (1.0 - gold_w) * merged["pred_margin_mu_torvik"].astype(float).to_numpy()
        )
    merged["sigma_calibrated"] = _apply_sigma_mode(merged, args)
    merged["site_date"] = (
        pd.to_datetime(merged["startDate"], utc=True, errors="coerce")
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    merged = merged.dropna(subset=["site_date"]).copy()

    deleted = _clear_historical_predictions(args.site_data_dir, args.keep_current_season)
    current_season = _current_season()

    written = 0
    for game_date, daily in merged.groupby("site_date", sort=True):
        if args.keep_current_season and _season_from_date(game_date) >= current_season:
            continue
        games = _build_site_games(daily, game_date)
        payload = {
            "date": game_date,
            "generated_at": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
            "source": "true_walkforward_historical",
            "benchmark_dir": str(args.benchmark_dir),
            "extra_benchmark_dirs": [str(p) for p in args.extra_benchmark_dir],
            "torvik_benchmark_dir": str(args.torvik_benchmark_dir) if args.torvik_benchmark_dir else None,
            "torvik_extra_benchmark_dirs": [str(p) for p in args.torvik_extra_benchmark_dir],
            "mu_model": args.mu_model_name,
            "sigma_model": SIGMA_MODEL,
            "sigma_mode": args.sigma_mode,
            "games": games,
        }
        out_path = args.site_data_dir / f"predictions_{game_date}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        written += 1

    summary = {
        "benchmark_dir": str(args.benchmark_dir),
        "extra_benchmark_dirs": [str(p) for p in args.extra_benchmark_dir],
        "mu_model": args.mu_model_name,
        "sigma_model": SIGMA_MODEL,
        "sigma_mode": args.sigma_mode,
        "dates_written": written,
        "walkforward_dates": sorted(str(d) for d in merged["site_date"].dropna().unique()),
        "rows_written": int(len(merged)),
        "historical_files_deleted": deleted,
        "keep_current_season": args.keep_current_season,
        "holdout_seasons": sorted(int(s) for s in merged["holdout_season"].unique()),
    }
    summary_path = args.site_data_dir / "true_walkforward_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

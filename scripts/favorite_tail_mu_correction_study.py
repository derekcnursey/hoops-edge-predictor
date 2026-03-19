from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import load_lines
from src.line_selection import _append_consensus_rows, _fix_spread_signs, _prepare_lines, _provider_rank

SITE_DATA = ROOT / "site" / "public" / "data"
OUT_DIR = ROOT / "artifacts" / "research" / "favorite_tail_mu_correction_study_v1"


def _season_from_date(date_str: str) -> int:
    year, month, _ = map(int, date_str.split("-"))
    return year + 1 if month >= 11 else year


def _is_post_dec15(date_str: str) -> int:
    _, month, day = map(int, date_str.split("-"))
    if month == 11:
        return 0
    if month == 12:
        return 1 if day >= 15 else 0
    return 1 if 1 <= month <= 3 else 0


def _normalize_team(name: str | None) -> str:
    if not name:
        return ""
    text = str(name).lower().strip().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\bsaint\b", "st", text)
    text = re.sub(r"\bst\.?\b", "st", text)
    text = re.sub(r"\buniversity\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _select_preferred_spreads(lines_df: pd.DataFrame) -> pd.DataFrame:
    lines = _append_consensus_rows(_fix_spread_signs(_prepare_lines(lines_df)))

    def _has(col: str) -> pd.Series:
        if col in lines.columns:
            return lines[col].notna().astype(int)
        return pd.Series(0, index=lines.index, dtype=int)

    selected = (
        lines.assign(
            _has_spread=_has("spread"),
            _prov_rank=_provider_rank(lines["provider"]),
        )
        .sort_values(
            ["_has_spread", "_prov_rank", "provider"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .drop_duplicates("gameId")
        .rename(columns={"spread": "book_spread", "homeTeam": "home_team", "awayTeam": "away_team"})
        .copy()
    )
    selected["date"] = pd.to_datetime(selected["startDate"]).dt.strftime("%Y-%m-%d")
    selected["home_key"] = selected["home_team"].map(_normalize_team)
    selected["away_key"] = selected["away_team"].map(_normalize_team)
    return selected[["season", "date", "home_key", "away_key", "book_spread"]]


def _load_lines() -> pd.DataFrame:
    manifest = json.loads((SITE_DATA / "true_walkforward_manifest.json").read_text())
    seasons = sorted({_season_from_date(d) for d in manifest["walkforward_dates"]} | {2026})
    parts = []
    for season in seasons:
        df = load_lines(season)
        if df is None or df.empty:
            continue
        parts.append(_select_preferred_spreads(df))
    return pd.concat(parts, ignore_index=True)


def _load_walkforward_dataset() -> pd.DataFrame:
    manifest = json.loads((SITE_DATA / "true_walkforward_manifest.json").read_text())
    rows: list[dict[str, object]] = []
    for date_str in manifest["walkforward_dates"]:
        pred_path = SITE_DATA / f"predictions_{date_str}.json"
        final_path = SITE_DATA / f"final_scores_{date_str}.json"
        if not pred_path.exists() or not final_path.exists():
            continue
        pred_obj = json.loads(pred_path.read_text())
        final_obj = json.loads(final_path.read_text())
        preds = pred_obj.get("games", pred_obj.get("predictions", []))
        finals = final_obj.get("games", final_obj.get("final_scores", []))
        final_lookup = {str(g.get("game_id")): g for g in finals if g.get("game_id") is not None}
        season = _season_from_date(date_str)
        for pred in preds:
            game_id = str(pred.get("game_id"))
            final = final_lookup.get(game_id)
            if final is None:
                continue
            try:
                mu = float(pred["model_mu_home"])
                home_score = float(final["home_score"])
                away_score = float(final["away_score"])
            except Exception:
                continue
            home_team = pred.get("home_team")
            away_team = pred.get("away_team")
            rows.append(
                {
                    "date": date_str,
                    "season": season,
                    "game_id": game_id,
                    "home_key": _normalize_team(home_team),
                    "away_key": _normalize_team(away_team),
                    "mu": mu,
                    "actual_margin": home_score - away_score,
                    "post_dec15": _is_post_dec15(date_str),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(["date", "game_id"])
    lines = _load_lines()
    df = df.merge(lines, on=["season", "date", "home_key", "away_key"], how="left")
    df["market_margin"] = -pd.to_numeric(df["book_spread"], errors="coerce")
    return df


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    threshold: float | None = None


def _favorite_side_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fav_sign = np.where(out["mu"].to_numpy() >= 0, 1.0, -1.0)
    out["pred_sign"] = fav_sign
    out["pred_fav_margin"] = np.abs(out["mu"].to_numpy())
    out["actual_fav_margin"] = out["actual_margin"].to_numpy() * fav_sign
    return out


def _fit_slope_tail(train: pd.DataFrame, threshold: float) -> float:
    tail = train[train["pred_fav_margin"] >= threshold]
    if tail.empty:
        return 1.0
    x = tail["pred_fav_margin"].to_numpy() - threshold
    y = tail["actual_fav_margin"].to_numpy() - threshold
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 1.0
    slope = float(np.dot(x, y) / denom)
    return float(np.clip(slope, 1.0, 2.5))


def _apply_slope_tail(df: pd.DataFrame, threshold: float, slope: float) -> np.ndarray:
    x = df["pred_fav_margin"].to_numpy()
    corrected = x.copy()
    mask = x >= threshold
    corrected[mask] = threshold + slope * (x[mask] - threshold)
    return corrected


def _fit_isotonic_tail(train: pd.DataFrame, threshold: float) -> IsotonicRegression | None:
    tail = train[train["pred_fav_margin"] >= threshold]
    if tail["pred_fav_margin"].nunique() < 2:
        return None
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(tail["pred_fav_margin"].to_numpy(), tail["actual_fav_margin"].to_numpy())
    return iso


def _apply_isotonic_tail(df: pd.DataFrame, threshold: float, iso: IsotonicRegression | None) -> np.ndarray:
    x = df["pred_fav_margin"].to_numpy()
    corrected = x.copy()
    if iso is None:
        return corrected
    mask = x >= threshold
    if mask.any():
        corrected_tail = iso.predict(x[mask])
        corrected[mask] = np.maximum(x[mask], corrected_tail)
    return corrected


def _predict_corrected_mu(df: pd.DataFrame, variant: Variant, params: object | None = None) -> np.ndarray:
    if variant.kind == "identity":
        corrected_fav = df["pred_fav_margin"].to_numpy()
    elif variant.kind == "slope_tail":
        corrected_fav = _apply_slope_tail(df, float(variant.threshold), float(params))
    elif variant.kind == "isotonic_tail":
        corrected_fav = _apply_isotonic_tail(df, float(variant.threshold), params)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown variant kind: {variant.kind}")
    return corrected_fav * df["pred_sign"].to_numpy()


def _metrics_from_mu(df: pd.DataFrame, mu_col: str) -> dict[str, float]:
    actual = df["actual_margin"].to_numpy()
    pred = df[mu_col].to_numpy()
    out = {
        "MAE_all": float(np.mean(np.abs(actual - pred))),
        "MAE_lined": np.nan,
        "Dec15_MAE_all": float(np.mean(np.abs(actual[df["post_dec15"] == 1] - pred[df["post_dec15"] == 1]))),
        "Dec15_MAE_lined": np.nan,
    }
    lined = df["market_margin"].notna()
    if lined.any():
        out["MAE_lined"] = float(np.mean(np.abs(actual[lined] - pred[lined])))
        dec_lined = lined & (df["post_dec15"] == 1)
        if dec_lined.any():
            out["Dec15_MAE_lined"] = float(np.mean(np.abs(actual[dec_lined] - pred[dec_lined])))
    return out


def _favorite_bucket_summary(df: pd.DataFrame, mu_col: str) -> pd.DataFrame:
    fav_sign = np.where(df["market_margin"].to_numpy() >= 0, 1.0, -1.0)
    actual_fav = df["actual_margin"].to_numpy() * fav_sign
    pred_fav = df[mu_col].to_numpy() * fav_sign
    market_fav = df["market_margin"].to_numpy() * fav_sign
    market_abs = np.abs(df["market_margin"].to_numpy())
    bucket = pd.cut(
        market_abs,
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
                "mean_model_margin": float(np.mean(pred_fav[mask])),
                "mean_actual_margin": float(np.mean(actual_fav[mask])),
                "model_minus_market": float(np.mean(pred_fav[mask] - market_fav[mask])),
                "actual_minus_model": float(np.mean(actual_fav[mask] - pred_fav[mask])),
                "actual_minus_market": float(np.mean(actual_fav[mask] - market_fav[mask])),
                "mae_model": float(np.mean(np.abs(actual_fav[mask] - pred_fav[mask]))),
                "mae_market": float(np.mean(np.abs(actual_fav[mask] - market_fav[mask]))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_walkforward_dataset()
    df = _favorite_side_view(df)

    eval_seasons = [2020, 2022, 2023, 2024, 2025, 2026]
    variants = [
        Variant("baseline", "identity"),
        Variant("slope_tail8", "slope_tail", threshold=8.0),
        Variant("slope_tail10", "slope_tail", threshold=10.0),
        Variant("isotonic_tail8", "isotonic_tail", threshold=8.0),
        Variant("isotonic_tail10", "isotonic_tail", threshold=10.0),
    ]

    eval_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    for season in eval_seasons:
        train = df[df["season"] < season].copy()
        test = df[df["season"] == season].copy()
        if train.empty or test.empty:
            continue
        for variant in variants:
            params: object | None = None
            if variant.kind == "slope_tail":
                params = _fit_slope_tail(train, float(variant.threshold))
            elif variant.kind == "isotonic_tail":
                params = _fit_isotonic_tail(train, float(variant.threshold))

            test_variant = test.copy()
            test_variant["variant"] = variant.name
            test_variant["mu_corrected"] = _predict_corrected_mu(test_variant, variant, params)
            eval_frames.append(test_variant)
            fit_rows.append(
                {
                    "season": season,
                    "variant": variant.name,
                    "threshold": variant.threshold,
                    "param_repr": str(params),
                }
            )

    study = pd.concat(eval_frames, ignore_index=True)
    pooled_rows = []
    season_rows = []
    for variant, grp in study.groupby("variant"):
        row = {"variant": variant}
        row.update(_metrics_from_mu(grp, "mu_corrected"))
        pooled_rows.append(row)

        for season, sgrp in grp.groupby("season"):
            srow = {"variant": variant, "season": season}
            srow.update(_metrics_from_mu(sgrp, "mu_corrected"))
            season_rows.append(srow)

    pd.DataFrame(pooled_rows).to_csv(OUT_DIR / "pooled_metrics.csv", index=False)
    pd.DataFrame(season_rows).to_csv(OUT_DIR / "season_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(OUT_DIR / "fit_params.csv", index=False)

    bucket_rows = []
    for variant, grp in study.groupby("variant"):
        for scope_name, scope_df in [("pooled", grp), ("dec15", grp[grp["post_dec15"] == 1])]:
            if scope_df.empty:
                continue
            b = _favorite_bucket_summary(scope_df[scope_df["market_margin"].notna()], "mu_corrected")
            b.insert(0, "scope", scope_name)
            b.insert(0, "variant", variant)
            bucket_rows.append(b)
    pd.concat(bucket_rows, ignore_index=True).to_csv(OUT_DIR / "favorite_bucket_summary.csv", index=False)

    summary_lines = [
        "# Favorite-Tail Mu Correction Study",
        "",
        "Baseline: current trusted walk-forward mean path from site historical archive.",
        "Correction fit in favorite-side margin space using prior seasons only.",
        "",
        "Variants:",
        "- baseline",
        "- slope_tail8",
        "- slope_tail10",
        "- isotonic_tail8",
        "- isotonic_tail10",
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines))


if __name__ == "__main__":
    main()

"""Generate a compact suspicious-game report for the live slate.

This is a diagnostic-only report. It does not modify predictions.
It reuses the current production prediction path and adds:
  - current-vs-market spread gap ranking
  - ATS edge ranking
  - current-vs-old-baseline mean ranking
  - grouped LightGBM contribution summaries
  - bug-triage hints for extreme model-vs-market gaps
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import click
import numpy as np
import pandas as pd

from src import config
from src.cli import _build_secondary_mu_features_if_needed
from src.efficiency_blend import blend_enabled, gold_weight_for_start_dates
from src.features import build_features, load_lines
from src.infer import (
    _fill_nan_with_scaler_means,
    _predict_mu_values,
    _swap_feature_frame,
    load_mu_regressor,
    load_torvik_mu_regressor,
    predict,
)
from src.line_selection import select_preferred_lines
from src.trainer import load_scaler, load_tree_regressor


EXTREME_THRESHOLDS = [15.0, 20.0, 25.0, 30.0]
KEY_FEATURES = [
    "neutral_site",
    "home_team_adj_oe",
    "away_team_adj_oe",
    "home_team_adj_de",
    "away_team_adj_de",
    "home_team_BARTHAG",
    "away_team_BARTHAG",
    "home_team_adj_pace",
    "away_team_adj_pace",
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "home_sos_oe",
    "away_sos_oe",
    "home_sos_de",
    "away_sos_de",
    "home_conf_strength",
    "away_conf_strength",
    "home_form_delta",
    "away_form_delta",
    "home_tov_rate",
    "away_tov_rate",
    "home_def_tov_rate",
    "away_def_tov_rate",
    "home_def_eff_fg_pct",
    "away_def_eff_fg_pct",
    "home_team_efg_home_split",
    "away_team_efg_away_split",
]
CRITICAL_FEATURES = [
    "home_team_adj_oe",
    "away_team_adj_oe",
    "home_team_adj_de",
    "away_team_adj_de",
]
IMPORTANT_SECONDARY_FEATURES = [
    "home_form_delta",
    "away_form_delta",
    "home_tov_rate",
    "away_tov_rate",
    "home_def_tov_rate",
    "away_def_tov_rate",
    "home_team_efg_home_split",
    "away_team_efg_away_split",
    "home_sos_oe",
    "away_sos_oe",
    "home_sos_de",
    "away_sos_de",
    "home_def_eff_fg_pct",
    "away_def_eff_fg_pct",
]


@dataclass
class MeanBundle:
    primary_model: object
    primary_feature_order: list[str]
    primary_model_type: str
    secondary_model: object | None
    secondary_feature_order: list[str] | None
    secondary_model_type: str | None


def _format_matchup(row: pd.Series) -> str:
    return f"{row['awayTeam']} @ {row['homeTeam']}"


def _feature_group(feature_name: str) -> str:
    if feature_name in {"home_team_BARTHAG", "away_team_BARTHAG"}:
        return "BARTHAG"
    if feature_name in {
        "home_team_adj_oe",
        "away_team_adj_oe",
        "home_team_adj_de",
        "away_team_adj_de",
    }:
        return "core_efficiency"
    if feature_name in {"home_team_adj_pace", "away_team_adj_pace"}:
        return "tempo_pace"
    if feature_name in {
        "home_sos_oe",
        "away_sos_oe",
        "home_sos_de",
        "away_sos_de",
        "home_conf_strength",
        "away_conf_strength",
    }:
        return "sos_conf"
    if feature_name in {
        "home_eff_fg_pct",
        "away_eff_fg_pct",
        "home_ft_pct",
        "away_ft_pct",
        "home_ft_rate",
        "away_ft_rate",
        "home_3pt_rate",
        "away_3pt_rate",
        "home_3p_pct",
        "away_3p_pct",
        "home_off_rebound_pct",
        "away_off_rebound_pct",
    }:
        return "offense_terms"
    if feature_name in {
        "home_def_rebound_pct",
        "away_def_rebound_pct",
    }:
        return "defense_core_terms"
    if feature_name in {
        "home_def_eff_fg_pct",
        "away_def_eff_fg_pct",
        "home_opp_ft_rate",
        "away_def_ft_rate",
        "home_def_3pt_rate",
        "away_def_3pt_rate",
        "home_def_3p_pct",
        "away_def_3p_pct",
        "home_def_off_rebound_pct",
        "away_def_off_rebound_pct",
    }:
        return "defensive_shot_quality"
    if feature_name in {
        "home_tov_rate",
        "away_tov_rate",
        "home_def_tov_rate",
        "away_def_tov_rate",
    }:
        return "turnover_features"
    if feature_name in {
        "home_form_delta",
        "away_form_delta",
        "home_margin_std",
        "away_margin_std",
        "home_team_efg_home_split",
        "away_team_efg_away_split",
    }:
        return "rolling_form_split"
    if feature_name in {"home_rest_days", "away_rest_days", "rest_advantage", "home_team_hca", "neutral_site"}:
        return "rest_hca_context"
    return "other"


def _swap_mapping(feature_order: Iterable[str]) -> dict[str, str]:
    mapping = {col: col for col in feature_order}
    explicit_pairs = [
        ("home_opp_ft_rate", "away_def_ft_rate"),
        ("home_team_efg_home_split", "away_team_efg_away_split"),
    ]
    for left, right in explicit_pairs:
        mapping[left] = right
        mapping[right] = left
    for col in feature_order:
        if col in mapping and mapping[col] != col:
            continue
        if col.startswith("home_"):
            other = "away_" + col[len("home_") :]
            if other in mapping:
                mapping[col] = other
                mapping[other] = col
    return mapping


def _align_swap_contribs(
    swap_contribs: np.ndarray,
    feature_order: list[str],
) -> np.ndarray:
    mapping = _swap_mapping(feature_order)
    idx = {name: i for i, name in enumerate(feature_order)}
    aligned = np.zeros_like(swap_contribs)
    for name in feature_order:
        aligned[idx[name]] = swap_contribs[idx[mapping[name]]]
    return aligned


def _predict_lgbm_contribs_for_bundle(
    model: object,
    model_type: str,
    features_df: pd.DataFrame,
    feature_order: list[str],
    scaler,
) -> tuple[np.ndarray, np.ndarray]:
    X_df = features_df[feature_order].copy()
    X_raw = _fill_nan_with_scaler_means(X_df, scaler)
    X_scaled = scaler.transform(X_raw)
    mu = _predict_mu_values(model, model_type, X_raw, X_scaled)

    if model_type != "lightgbm":
        raise ValueError("Contribution extraction currently only supports LightGBM models")

    contrib_raw = model.predict(X_raw, pred_contrib=True).astype(np.float32)
    feature_contribs = contrib_raw[:, :-1]

    neutral_col = "neutral_site" if "neutral_site" in features_df.columns else None
    if neutral_col is None and "neutralSite" in features_df.columns:
        neutral_col = "neutralSite"
    if neutral_col is None:
        return mu.astype(np.float32), feature_contribs

    neutral_mask = features_df[neutral_col].fillna(0).astype(float).to_numpy() == 1.0
    if not neutral_mask.any():
        return mu.astype(np.float32), feature_contribs

    neutral_idx = np.flatnonzero(neutral_mask)
    X_swap_df = _swap_feature_frame(X_df.iloc[neutral_idx], feature_order)
    X_swap_raw = _fill_nan_with_scaler_means(X_swap_df, scaler)
    X_swap_scaled = scaler.transform(X_swap_raw)
    mu_swap = _predict_mu_values(model, model_type, X_swap_raw, X_swap_scaled)
    swap_contrib_raw = model.predict(X_swap_raw, pred_contrib=True).astype(np.float32)[:, :-1]

    mu_sym = mu.astype(np.float32).copy()
    mu_sym[neutral_idx] = (mu_sym[neutral_idx] - mu_swap.astype(np.float32)) / 2.0

    contrib_sym = feature_contribs.copy()
    for local_i, row_i in enumerate(neutral_idx):
        aligned_swap = _align_swap_contribs(swap_contrib_raw[local_i], feature_order)
        contrib_sym[row_i] = 0.5 * (feature_contribs[row_i] - aligned_swap)
    return mu_sym, contrib_sym


def _load_current_bundle() -> MeanBundle:
    primary_model, primary_feature_order, primary_meta = load_tree_regressor(config.TREE_REGRESSOR_PATH)
    secondary_loaded = load_torvik_mu_regressor()
    secondary_model = None
    secondary_feature_order = None
    secondary_model_type = None
    if secondary_loaded is not None:
        secondary_model, secondary_feature_order, secondary_model_type = secondary_loaded
    return MeanBundle(
        primary_model=primary_model,
        primary_feature_order=primary_feature_order,
        primary_model_type=primary_meta.get("model_type", "lightgbm"),
        secondary_model=secondary_model,
        secondary_feature_order=secondary_feature_order,
        secondary_model_type=secondary_model_type,
    )


def _load_old_baseline_bundle() -> MeanBundle:
    old_primary = config.CHECKPOINTS_DIR / "regressor_hgbr.pkl"
    old_secondary = config.CHECKPOINTS_DIR / "regressor_hgbr_torvik.pkl"
    primary_model, primary_feature_order, primary_meta = load_tree_regressor(old_primary)
    secondary_model = None
    secondary_feature_order = None
    secondary_model_type = None
    if old_secondary.exists():
        secondary_model, secondary_feature_order, secondary_meta = load_tree_regressor(old_secondary)
        secondary_model_type = secondary_meta.get("model_type", "hist_gradient_boosting")
    return MeanBundle(
        primary_model=primary_model,
        primary_feature_order=primary_feature_order,
        primary_model_type=primary_meta.get("model_type", "hist_gradient_boosting"),
        secondary_model=secondary_model,
        secondary_feature_order=secondary_feature_order,
        secondary_model_type=secondary_model_type,
    )


def _predict_bundle_mu(
    bundle: MeanBundle,
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame | None,
    scaler,
) -> np.ndarray:
    feature_order = bundle.primary_feature_order
    X_df = primary_df[feature_order].copy()
    X_raw = _fill_nan_with_scaler_means(X_df, scaler)
    X_scaled = scaler.transform(X_raw)
    mu_primary = _predict_mu_values(bundle.primary_model, bundle.primary_model_type, X_raw, X_scaled)

    # Symmetrize neutral rows.
    neutral_col = "neutral_site" if "neutral_site" in primary_df.columns else "neutralSite"
    if neutral_col in primary_df.columns:
        neutral_mask = primary_df[neutral_col].fillna(0).astype(float).to_numpy() == 1.0
        if neutral_mask.any():
            neutral_idx = np.flatnonzero(neutral_mask)
            X_swap_df = _swap_feature_frame(X_df.iloc[neutral_idx], feature_order)
            X_swap_raw = _fill_nan_with_scaler_means(X_swap_df, scaler)
            X_swap_scaled = scaler.transform(X_swap_raw)
            mu_swap = _predict_mu_values(bundle.primary_model, bundle.primary_model_type, X_swap_raw, X_swap_scaled)
            mu_primary = mu_primary.astype(np.float32)
            mu_primary[neutral_idx] = (mu_primary[neutral_idx] - mu_swap.astype(np.float32)) / 2.0

    if blend_enabled() and secondary_df is not None and bundle.secondary_model is not None:
        sec_df = secondary_df.set_index("gameId").reindex(primary_df["gameId"]).reset_index()
        sec_X_df = sec_df[feature_order].copy()
        sec_X_raw = _fill_nan_with_scaler_means(sec_X_df, scaler)
        sec_X_scaled = scaler.transform(sec_X_raw)
        mu_secondary = _predict_mu_values(bundle.secondary_model, bundle.secondary_model_type, sec_X_raw, sec_X_scaled)
        if neutral_col in sec_df.columns:
            neutral_mask = sec_df[neutral_col].fillna(0).astype(float).to_numpy() == 1.0
            if neutral_mask.any():
                neutral_idx = np.flatnonzero(neutral_mask)
                sec_swap_df = _swap_feature_frame(sec_X_df.iloc[neutral_idx], feature_order)
                sec_swap_raw = _fill_nan_with_scaler_means(sec_swap_df, scaler)
                sec_swap_scaled = scaler.transform(sec_swap_raw)
                mu_swap = _predict_mu_values(bundle.secondary_model, bundle.secondary_model_type, sec_swap_raw, sec_swap_scaled)
                mu_secondary = mu_secondary.astype(np.float32)
                mu_secondary[neutral_idx] = (mu_secondary[neutral_idx] - mu_swap.astype(np.float32)) / 2.0
        gold_w = gold_weight_for_start_dates(primary_df["startDate"])
        return (gold_w * mu_primary) + ((1.0 - gold_w) * mu_secondary)

    return mu_primary.astype(np.float32)


def _predict_current_mu_and_contribs(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scaler = load_scaler()
    bundle = _load_current_bundle()
    feature_order = bundle.primary_feature_order
    mu_primary, contrib_primary = _predict_lgbm_contribs_for_bundle(
        bundle.primary_model,
        bundle.primary_model_type,
        primary_df,
        feature_order,
        scaler,
    )
    if blend_enabled() and secondary_df is not None and bundle.secondary_model is not None:
        sec_df = secondary_df.set_index("gameId").reindex(primary_df["gameId"]).reset_index()
        mu_secondary, contrib_secondary = _predict_lgbm_contribs_for_bundle(
            bundle.secondary_model,
            bundle.secondary_model_type,
            sec_df,
            feature_order,
            scaler,
        )
        gold_w = gold_weight_for_start_dates(primary_df["startDate"]).astype(np.float32)
        mu = (gold_w * mu_primary) + ((1.0 - gold_w) * mu_secondary)
        contrib = (gold_w[:, None] * contrib_primary) + ((1.0 - gold_w)[:, None] * contrib_secondary)
        return mu.astype(np.float32), contrib.astype(np.float32), feature_order
    return mu_primary.astype(np.float32), contrib_primary.astype(np.float32), feature_order


def _group_contribs(contrib_row: np.ndarray, feature_order: list[str]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for feature_name, value in zip(feature_order, contrib_row, strict=True):
        group = _feature_group(feature_name)
        grouped[group] = grouped.get(group, 0.0) + float(value)
    return dict(sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True))


def _key_feature_snapshot(row: pd.Series, scaler, feature_order: list[str]) -> list[dict[str, object]]:
    idx = {name: i for i, name in enumerate(feature_order)}
    snapshots = []
    for feature in KEY_FEATURES:
        if feature not in row.index:
            continue
        value = row[feature]
        mean_filled = pd.isna(value)
        baseline = float(scaler.mean_[idx[feature]]) if feature in idx else None
        snapshots.append(
            {
                "feature": feature,
                "value": None if pd.isna(value) else float(value),
                "mean_filled": bool(mean_filled),
                "baseline_mean": baseline,
            }
        )
    return snapshots


def _important_mean_fills(row: pd.Series) -> list[str]:
    return [feature for feature in IMPORTANT_SECONDARY_FEATURES if feature in row.index and pd.isna(row[feature])]


def _triage_label(
    merged_row: pd.Series,
    feat_row: pd.Series,
    grouped_current: dict[str, float],
    important_mean_fills: list[str],
) -> str:
    if pd.isna(merged_row.get("market_spread_home")):
        return "likely data/input bug"
    if any(col not in feat_row.index or pd.isna(feat_row.get(col)) for col in CRITICAL_FEATURES):
        return "likely data/input bug"
    abs_gap = float(abs(merged_row.get("abs_market_gap", 0.0)))
    old_delta = float(abs(merged_row.get("mu_delta_vs_old", 0.0)))
    if len(important_mean_fills) >= 4:
        return "likely feature-path anomaly"
    if abs_gap >= 20 and old_delta >= 5:
        return "likely feature-path anomaly"
    top_groups = list(grouped_current.keys())[:2]
    if abs_gap >= 15 and any(group in {"rolling_form_split", "defensive_shot_quality"} for group in top_groups):
        return "likely feature-path anomaly"
    if abs_gap >= 15:
        return "likely real model disagreement"
    return "unclear / needs deeper audit"


def _select_suspicious_games(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    candidates = []
    lined = df[df["market_spread_home"].notna()].copy()
    if not lined.empty:
        candidates.append(lined.nlargest(top_n, "abs_market_gap"))
        candidates.append(lined.nlargest(top_n, "abs_ats_edge"))
    candidates.append(df.nlargest(top_n, "abs_mu_delta_vs_old"))
    out = pd.concat(candidates, ignore_index=False).drop_duplicates(subset=["gameId"], keep="first")
    sort_cols = [c for c in ["abs_market_gap", "abs_mu_delta_vs_old", "abs_ats_edge"] if c in out.columns]
    return out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)


def _format_grouped_contribs(grouped: dict[str, float], top_n: int = 6) -> str:
    items = list(grouped.items())[:top_n]
    return ", ".join(f"{k}={v:+.2f}" for k, v in items)


def _write_report(
    report_path: Path,
    season: int,
    game_date: str,
    suspicious: pd.DataFrame,
    details: dict[str, dict[str, object]],
) -> None:
    lines = [
        f"# Suspicious Game Monitor",
        "",
        f"- season: `{season}`",
        f"- date: `{game_date}`",
        f"- suspicious games: `{len(suspicious)}`",
        "",
        "## Shortlist",
        "",
        "| Matchup | Market | Current mu | Old mu | |gap| | ATS edge | Label |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in suspicious.iterrows():
        label = details[row["gameId"]]["triage_label"]
        market = row["market_spread_home"]
        market_str = "NA" if pd.isna(market) else f"{market:+.2f}"
        lines.append(
            f"| {row['awayTeam']} @ {row['homeTeam']} | {market_str} | "
            f"{row['model_mu_home']:+.2f} | {row['old_mu_home']:+.2f} | "
            f"{row['abs_market_gap']:.2f} | {row['abs_ats_edge']:.2f} | {label} |"
        )
    lines.extend(["", "## Extreme-gap bug triage", ""])
    extreme = suspicious[suspicious["abs_market_gap"] >= EXTREME_THRESHOLDS[0]].copy()
    if extreme.empty:
        lines.append("No games met the extreme-gap threshold.")
    else:
        for _, row in extreme.iterrows():
            detail = details[row["gameId"]]
            lines.extend(
                [
                    f"### {row['awayTeam']} @ {row['homeTeam']}",
                    "",
                    f"- triage: `{detail['triage_label']}`",
                    f"- market spread (home): `{row['market_spread_home']:+.2f}`",
                    f"- current mu (home): `{row['model_mu_home']:+.2f}`",
                    f"- old baseline mu (home): `{row['old_mu_home']:+.2f}`",
                    f"- absolute market gap: `{row['abs_market_gap']:.2f}`",
                    f"- ATS edge magnitude: `{row['abs_ats_edge']:.2f}`",
                    f"- selected line provider: `{detail['line_provider']}`",
                    f"- neutral site: `{detail['neutral_site']}`",
                    f"- important mean-filled features: `{', '.join(detail['important_mean_fills']) or 'none'}`",
                    f"- grouped contributions: `{_format_grouped_contribs(detail['grouped_contribs'])}`",
                    "",
                ]
            )

    report_path.write_text("\n".join(lines))


@click.command()
@click.option("--season", required=True, type=int)
@click.option("--date", "game_date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("artifacts/diagnostics/suspicious_game_monitor"),
    show_default=True,
)
def main(season: int, game_date, output_dir: Path) -> None:
    game_date_str = game_date.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    features_df = build_features(
        season,
        game_date=game_date_str,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
        efficiency_source=config.EFFICIENCY_SOURCE,
    )
    secondary_df = _build_secondary_mu_features_if_needed(season, features_df, game_date=game_date_str)
    lines_df = load_lines(season)
    preferred_lines = select_preferred_lines(lines_df)
    current_preds = predict(features_df, lines_df, secondary_df).rename(
        columns={
            "predicted_spread": "model_mu_home",
            "spread_sigma": "pred_sigma",
            "home_win_prob": "pred_home_win_prob",
        }
    )

    scaler = load_scaler()
    current_mu_manual, current_contribs, feature_order = _predict_current_mu_and_contribs(features_df, secondary_df)
    current_preds["model_mu_home_manual"] = current_mu_manual

    old_bundle = _load_old_baseline_bundle()
    old_mu = _predict_bundle_mu(old_bundle, features_df, secondary_df, scaler)
    current_preds["old_mu_home"] = old_mu

    merged = current_preds.copy()
    merged = merged.merge(
        preferred_lines[["gameId", "provider", "book_spread", "home_moneyline", "away_moneyline"]]
        if not preferred_lines.empty
        else pd.DataFrame(columns=["gameId", "provider", "book_spread", "home_moneyline", "away_moneyline"]),
        on="gameId",
        how="left",
        suffixes=("", "_preferred"),
    )
    if "provider" not in merged.columns and "provider_preferred" in merged.columns:
        merged["provider"] = merged["provider_preferred"]

    merged["market_spread_home"] = merged["book_spread"]
    merged["market_gap_home"] = merged["model_mu_home"] + merged["market_spread_home"]
    merged["abs_market_gap"] = merged["market_gap_home"].abs()
    merged["abs_ats_edge"] = merged["edge_home_points"].abs() if "edge_home_points" in merged.columns else np.nan
    merged["mu_delta_vs_old"] = merged["model_mu_home"] - merged["old_mu_home"]
    merged["abs_mu_delta_vs_old"] = merged["mu_delta_vs_old"].abs()

    suspicious = _select_suspicious_games(merged)
    details: dict[str, dict[str, object]] = {}
    feature_rows = features_df.set_index("gameId")
    for _, row in suspicious.iterrows():
        game_id = row["gameId"]
        feat_row = feature_rows.loc[game_id]
        if isinstance(feat_row, pd.DataFrame):
            feat_row = feat_row.iloc[0]
        feat_snapshot = _key_feature_snapshot(feat_row, scaler, feature_order)
        important_fills = _important_mean_fills(feat_row)
        contrib_idx = int(np.flatnonzero(features_df["gameId"].to_numpy() == game_id)[0])
        grouped = _group_contribs(current_contribs[contrib_idx], feature_order)
        details[game_id] = {
            "game_id": game_id,
            "matchup": _format_matchup(row),
            "line_provider": row.get("provider") if pd.notna(row.get("provider")) else "NA",
            "neutral_site": bool(row.get("neutral_site", row.get("neutralSite", False))),
            "grouped_contribs": grouped,
            "key_features": feat_snapshot,
            "important_mean_fills": important_fills,
            "triage_label": _triage_label(row, feat_row, grouped, important_fills),
        }

    csv_cols = [
        "gameId",
        "awayTeam",
        "homeTeam",
        "market_spread_home",
        "model_mu_home",
        "old_mu_home",
        "abs_market_gap",
        "abs_ats_edge",
        "mu_delta_vs_old",
        "pick_side",
        "pick_prob_edge",
        "pred_sigma",
        "pred_home_win_prob",
    ]
    suspicious[csv_cols].to_csv(output_dir / f"suspicious_games_{game_date_str}.csv", index=False)
    (output_dir / f"suspicious_game_details_{game_date_str}.json").write_text(json.dumps(details, indent=2))
    _write_report(output_dir / f"suspicious_games_{game_date_str}.md", season, game_date_str, suspicious, details)

    click.echo(f"Wrote suspicious-game report to {output_dir}")
    click.echo(f"Top suspicious games for {game_date_str}:")
    for _, row in suspicious.head(10).iterrows():
        label = details[row["gameId"]]["triage_label"]
        market = "NA" if pd.isna(row["market_spread_home"]) else f"{row['market_spread_home']:+.2f}"
        click.echo(
            f"  - {_format_matchup(row)} | market {market} | current {row['model_mu_home']:+.2f} | "
            f"old {row['old_mu_home']:+.2f} | |gap| {row['abs_market_gap']:.2f} | "
            f"|edge| {row['abs_ats_edge']:.2f} | {label}"
        )


if __name__ == "__main__":
    main()

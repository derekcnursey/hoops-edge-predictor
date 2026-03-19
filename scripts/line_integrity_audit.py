#!/usr/bin/env python3
"""Audit stored sportsbook lines for likely corruption and betting-metric contamination."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src import config as app_config, s3_reader
from src.features import load_efficiency_ratings, load_games, load_lines
from src.sigma_calibration import apply_sigma_transform

BREAKEVEN = 110.0 / 210.0
WIN_PROFIT = 100.0 / 110.0
PROB_BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
LINE_PROVIDER_RANK = {"Draft Kings": 0, "ESPN BET": 1, "Bovada": 2}
DEFAULT_EVAL_SEASONS = [2019, 2020, 2022, 2023, 2024, 2025]


@dataclass(frozen=True)
class AuditConfig:
    eval_seasons: list[int]
    all_seasons: list[int]
    strong_ml_home_favorite: int = -250
    strong_ml_away_underdog: int = 200
    strong_ml_min_abs_spread: float = 4.0
    medium_ml_home_favorite: int = -180
    medium_ml_away_underdog: int = 150
    medium_ml_min_abs_spread: float = 3.0
    majority_min_abs_spread: float = 3.0
    majority_min_net_votes: int = 2


def _load_sigma_winner_spec(sigma_study_dir: Path) -> tuple[str, dict[str, float]]:
    summary = json.loads((sigma_study_dir / "dataset_summary.json").read_text())
    winner_spec = dict(summary["winner_spec"])
    mode = str(winner_spec["family"])
    params = {
        key: float(value)
        for key, value in winner_spec.items()
        if key not in {"family", "selected_label"}
        and value is not None
        and not (isinstance(value, float) and math.isnan(value))
    }
    return mode, params


def _dedupe_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    lines = lines_df.copy()
    if lines.empty:
        return lines
    lines["spread"] = pd.to_numeric(lines["spread"], errors="coerce")
    lines["_has_spread"] = lines["spread"].notna().astype(int)
    lines["_prov_rank"] = lines["provider"].map(LINE_PROVIDER_RANK).fillna(99)
    lines["_provider_name"] = lines["provider"].fillna("").astype(str).str.casefold()
    lines = (
        lines.sort_values(
            ["_has_spread", "_prov_rank", "_provider_name"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["gameId"], keep="first")
        .drop(columns=["_has_spread", "_prov_rank", "_provider_name"])
    )
    return lines.reset_index(drop=True)


def _load_lines_for_table(season: int, table_name: str) -> pd.DataFrame:
    if table_name == "fct_lines":
        return load_lines(season)
    tbl = s3_reader.read_silver_table(table_name, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame()
    return tbl.to_pandas()


def _build_strength_lookup(season: int) -> dict[int, pd.DataFrame]:
    ratings = load_efficiency_ratings(season)
    if ratings.empty:
        return {}
    ratings = ratings.copy()
    ratings["rating_date"] = pd.to_datetime(ratings["rating_date"], errors="coerce")
    lookup: dict[int, pd.DataFrame] = {}
    for team_id, group in ratings.groupby("teamId"):
        lookup[int(team_id)] = group.sort_values("rating_date")[
            ["rating_date", "adj_margin", "barthag"]
        ].copy()
    return lookup


def _get_asof_strength(
    strength_lookup: dict[int, pd.DataFrame],
    team_id: int | float | None,
    game_date: object,
) -> pd.Series | None:
    if team_id is None or pd.isna(team_id):
        return None
    team_df = strength_lookup.get(int(team_id))
    if team_df is None or team_df.empty:
        return None
    date_value = pd.to_datetime(game_date, errors="coerce", utc=True)
    if pd.isna(date_value):
        return None
    cutoff = date_value.tz_localize(None).normalize() - pd.Timedelta(days=1)
    eligible = team_df[team_df["rating_date"] <= cutoff]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _selected_line_rows_for_season(season: int, config: AuditConfig, lines_table: str) -> pd.DataFrame:
    lines = _load_lines_for_table(season, lines_table)
    if lines.empty:
        return pd.DataFrame()
    lines = lines.copy()
    lines["spread"] = pd.to_numeric(lines["spread"], errors="coerce")
    lines["homeMoneyline"] = pd.to_numeric(lines.get("homeMoneyline"), errors="coerce")
    lines["awayMoneyline"] = pd.to_numeric(lines.get("awayMoneyline"), errors="coerce")
    games = load_games(season)
    strength_lookup = _build_strength_lookup(season)

    duplicate_spread_conflict = (
        lines.groupby(["gameId", "provider"])["spread"]
        .agg(lambda values: values.dropna().nunique())
        .rename("duplicate_spread_nunique")
    )
    has_spread = lines[lines["spread"].notna() & (lines["spread"] != 0)].copy()
    has_spread["spread_sign"] = np.sign(has_spread["spread"])
    majority = has_spread.groupby("gameId")["spread_sign"].agg(["sum", "count"]).rename(
        columns={"sum": "majority_sign_sum", "count": "provider_count"}
    )

    selected = _dedupe_lines(lines)
    selected = selected.merge(majority, left_on="gameId", right_index=True, how="left")
    if not games.empty:
        selected = selected.merge(
            games[
                [
                    "gameId",
                    "homeTeam",
                    "awayTeam",
                    "homeTeamId",
                    "awayTeamId",
                    "neutralSite",
                    "startDate",
                ]
            ],
            on="gameId",
            how="left",
        )

    reason_rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        reasons: list[str] = []
        confidence = "none"
        game_id = int(row["gameId"])
        provider = str(row.get("provider", ""))
        spread = row.get("spread")
        hml = row.get("homeMoneyline")
        aml = row.get("awayMoneyline")
        majority_sum = row.get("majority_sign_sum")
        provider_count = row.get("provider_count")

        duplicate_key = (game_id, provider)
        if (
            duplicate_key in duplicate_spread_conflict.index
            and duplicate_spread_conflict.loc[duplicate_key] > 1
        ):
            reasons.append("duplicate_provider_conflict")
            confidence = "high"

        if (
            pd.notna(spread)
            and pd.notna(majority_sum)
            and pd.notna(provider_count)
            and provider_count >= 2
            and abs(float(majority_sum)) >= config.majority_min_net_votes
            and abs(float(spread)) >= config.majority_min_abs_spread
            and np.sign(float(spread)) != np.sign(float(majority_sum))
        ):
            reasons.append("majority_sign_conflict")
            confidence = "high"

        if pd.notna(spread) and pd.notna(hml) and pd.notna(aml):
            if (
                hml <= config.strong_ml_home_favorite
                and aml >= config.strong_ml_away_underdog
                and spread > config.strong_ml_min_abs_spread
            ) or (
                aml <= config.strong_ml_home_favorite
                and hml >= config.strong_ml_away_underdog
                and spread < -config.strong_ml_min_abs_spread
            ):
                reasons.append("strong_moneyline_conflict")
                confidence = "high"
            elif confidence == "none" and (
                (
                    hml <= config.medium_ml_home_favorite
                    and aml >= config.medium_ml_away_underdog
                    and spread > config.medium_ml_min_abs_spread
                )
                or (
                    aml <= config.medium_ml_home_favorite
                    and hml >= config.medium_ml_away_underdog
                    and spread < -config.medium_ml_min_abs_spread
                )
            ):
                reasons.append("moneyline_conflict")
                confidence = "medium"

        rating_edge = np.nan
        home_strength = _get_asof_strength(
            strength_lookup,
            row.get("homeTeamId"),
            row.get("startDate"),
        )
        away_strength = _get_asof_strength(
            strength_lookup,
            row.get("awayTeamId"),
            row.get("startDate"),
        )
        if home_strength is not None and away_strength is not None and pd.notna(spread):
            hca = 0.0 if bool(row.get("neutralSite")) else 3.0
            rating_edge = float(home_strength["adj_margin"] - away_strength["adj_margin"] + hca)
            if (
                confidence == "none"
                and abs(rating_edge) >= 12.0
                and abs(float(spread)) >= 6.0
                and np.sign(rating_edge) != np.sign(-float(spread))
            ):
                reasons.append("extreme_rating_context_conflict")
                confidence = "low"

        reason_rows.append(
            {
                "confidence": confidence,
                "reason_flag": "|".join(reasons),
                "rating_edge": rating_edge,
            }
        )

    selected = pd.concat([selected.reset_index(drop=True), pd.DataFrame(reason_rows)], axis=1)
    selected["season"] = season
    return selected


def _load_eval_predictions(
    benchmark_dir: Path,
    sigma_study_dir: Path,
    eval_seasons: list[int],
    selected_lines: pd.DataFrame,
) -> pd.DataFrame:
    mode, params = _load_sigma_winner_spec(sigma_study_dir)
    pred_dir = benchmark_dir / "predictions"
    frames: list[pd.DataFrame] = []
    for season in eval_seasons:
        hgb = pd.read_parquet(pred_dir / "HistGradientBoosting" / f"season_{season}.parquet")
        mlp = pd.read_parquet(pred_dir / "CurrentMLP" / f"season_{season}.parquet")[
            ["gameId", "sigma"]
        ].rename(columns={"sigma": "mlp_sigma"})
        pred = hgb.merge(mlp, on="gameId", how="inner")
        season_lines = selected_lines[selected_lines["season"] == season][
            ["gameId", "provider", "spread", "homeMoneyline", "awayMoneyline"]
        ].rename(
            columns={
                "spread": "book_spread",
                "homeMoneyline": "home_moneyline",
                "awayMoneyline": "away_moneyline",
            }
        )
        pred = pred.drop(
            columns=[c for c in ["book_spread", "home_moneyline", "away_moneyline"] if c in pred.columns],
            errors="ignore",
        )
        pred = pred.merge(season_lines, on="gameId", how="inner")
        pred = pred[pred["book_spread"].notna()].copy()
        sigma = apply_sigma_transform(pred["mlp_sigma"].to_numpy(float), mode=mode, **params)
        edge_home_points = pred["pred_margin"].to_numpy(float) + pred["book_spread"].to_numpy(float)
        home_cover_prob = 0.5 * (
            1.0 + np.vectorize(math.erf)(edge_home_points / sigma / math.sqrt(2.0))
        )
        actual_edge_home = pred["actual_margin"].to_numpy(float) + pred["book_spread"].to_numpy(float)
        bet_win = np.where(
            actual_edge_home == 0.0,
            np.nan,
            np.where(edge_home_points >= 0.0, actual_edge_home > 0.0, actual_edge_home < 0.0),
        )
        pred["season"] = season
        pred["sigma_used"] = sigma
        pred["edge_home_points"] = edge_home_points
        pred["pick_prob"] = np.where(edge_home_points >= 0.0, home_cover_prob, 1.0 - home_cover_prob)
        pred["pick_prob_edge"] = pred["pick_prob"] - BREAKEVEN
        pred["bet_win"] = bet_win
        pred["profit"] = np.where(
            actual_edge_home == 0.0,
            0.0,
            np.where(pd.Series(bet_win).fillna(False), WIN_PROFIT, -1.0),
        )
        frames.append(pred)
    return pd.concat(frames, ignore_index=True)


def _scenario_metrics(df: pd.DataFrame, scenario: str) -> dict[str, object]:
    row: dict[str, object] = {"scenario": scenario, "n_rows": int(len(df))}
    by_season = df.groupby("season", sort=True)
    for top_n in [100, 200, 500]:
        rois = []
        winrates = []
        avg_edges = []
        for _, season_df in by_season:
            top = season_df.sort_values("pick_prob", ascending=False).head(min(top_n, len(season_df)))
            non_push = top["bet_win"].notna()
            rois.append(float(top["profit"].mean()))
            winrates.append(float(top.loc[non_push, "bet_win"].mean()) if non_push.any() else float("nan"))
            avg_edges.append(float(top["edge_home_points"].abs().mean()))
        row[f"top{top_n}_roi"] = float(np.nanmean(rois))
        row[f"top{top_n}_winrate"] = float(np.nanmean(winrates))
        row[f"top{top_n}_avg_edge"] = float(np.nanmean(avg_edges))

    edge_abs = df["edge_home_points"].abs()
    for threshold in [10, 15, 20, 25, 30]:
        counts = by_season.apply(
            lambda season_df: int((season_df["edge_home_points"].abs() >= threshold).sum()),
            include_groups=False,
        )
        row[f"avg_edge_ge_{threshold}_count"] = float(counts.mean())

    bucket_series = pd.cut(
        df["pick_prob"],
        bins=PROB_BUCKETS,
        include_lowest=True,
        right=False,
    )
    bucket_rows = []
    for bucket, indices in bucket_series.groupby(bucket_series, observed=False).groups.items():
        if len(indices) == 0:
            continue
        sub = df.loc[indices]
        non_push = sub["bet_win"].notna()
        bucket_rows.append(
            {
                "scenario": scenario,
                "bucket": str(bucket),
                "n": int(len(sub)),
                "avg_pick_prob": float(sub["pick_prob"].mean()),
                "win_rate": float(sub.loc[non_push, "bet_win"].mean()) if non_push.any() else float("nan"),
                "roi": float(sub["profit"].mean()),
            }
        )

    return row | {"bucket_rows": bucket_rows}


def _summary_markdown(
    impact_df: pd.DataFrame,
    provider_scope_df: pd.DataFrame,
    season_scope_df: pd.DataFrame,
    flagged_eval_df: pd.DataFrame,
) -> str:
    full = impact_df.set_index("scenario").loc["full"]
    excl_high = impact_df.set_index("scenario").loc["exclude_high"]
    excl_hm = impact_df.set_index("scenario").loc["exclude_high_medium"]
    return "\n".join(
        [
            "# Line Integrity Audit",
            "",
            "## Flagged Scope",
            "",
            f"- Likely suspect evaluation rows: `{len(flagged_eval_df)}`",
            f"- High-confidence rows: `{int((flagged_eval_df['confidence'] == 'high').sum())}`",
            f"- Medium-confidence rows: `{int((flagged_eval_df['confidence'] == 'medium').sum())}`",
            "",
            "## Betting Impact",
            "",
            f"- Full top200 ROI: `{full['top200_roi']:.4f}`",
            f"- Excluding high-confidence rows: `{excl_high['top200_roi']:.4f}`",
            f"- Excluding high+medium-confidence rows: `{excl_hm['top200_roi']:.4f}`",
            f"- Full top100 ROI: `{full['top100_roi']:.4f}`",
            f"- Excluding high-confidence rows: `{excl_high['top100_roi']:.4f}`",
            f"- Excluding high+medium-confidence rows: `{excl_hm['top100_roi']:.4f}`",
            "",
            "## Provider Scope",
            "",
            provider_scope_df.to_string(index=False),
            "",
            "## Season Scope",
            "",
            season_scope_df.to_string(index=False),
            "",
            "## Example Flagged Rows",
            "",
            flagged_eval_df.head(25).to_string(index=False),
        ]
    )


@click.command()
@click.option(
    "--benchmark-dir",
    default="artifacts/benchmarks/canonical_walkforward_v2_lgb",
    type=click.Path(path_type=Path),
    help="Canonical benchmark directory used for historical betting evaluation.",
)
@click.option(
    "--sigma-study-dir",
    default="artifacts/sigma_calibration_study_v1",
    type=click.Path(path_type=Path),
    help="Sigma-study directory that defines the current betting probability path.",
)
@click.option(
    "--output-dir",
    default="artifacts/line_integrity_audit_v1",
    type=click.Path(path_type=Path),
    help="Where to write line-audit artifacts.",
)
@click.option(
    "--lines-table",
    default=app_config.RESEARCH_LINES_TABLE,
    help="Silver table name to audit and use for selected historical lines.",
)
def main(benchmark_dir: Path, sigma_study_dir: Path, output_dir: Path, lines_table: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AuditConfig(
        eval_seasons=DEFAULT_EVAL_SEASONS,
        all_seasons=list(range(2016, 2027)),
    )

    # Raw selected-line scope across all seasons.
    all_selected_rows = [
        _selected_line_rows_for_season(season, config, lines_table)
        for season in config.all_seasons
    ]
    all_selected = pd.concat([df for df in all_selected_rows if not df.empty], ignore_index=True)
    all_flagged = all_selected[all_selected["confidence"] != "none"].copy()

    provider_scope = (
        all_flagged.groupby(["provider", "confidence"]).size().unstack(fill_value=0).reset_index()
    )
    season_scope = (
        all_flagged.groupby(["season", "confidence"]).size().unstack(fill_value=0).reset_index()
    )

    # Historical betting contamination on canonical holdouts.
    eval_predictions = _load_eval_predictions(
        benchmark_dir,
        sigma_study_dir,
        config.eval_seasons,
        all_selected[all_selected["season"].isin(config.eval_seasons)].copy(),
    )
    eval_flags = all_selected[all_selected["season"].isin(config.eval_seasons)].copy()
    eval = eval_predictions.merge(
        eval_flags[
            [
                "season",
                "gameId",
                "provider",
                "spread",
                "homeMoneyline",
                "awayMoneyline",
                "confidence",
                "reason_flag",
                "rating_edge",
            ]
        ],
        on=["season", "gameId", "provider"],
        how="left",
    )
    eval["confidence"] = eval["confidence"].fillna("none")
    eval["reason_flag"] = eval["reason_flag"].fillna("")

    flagged_eval = (
        eval[eval["confidence"] != "none"]
        .sort_values(["confidence", "pick_prob"], ascending=[True, False])
        .copy()
    )

    scenarios = {
        "full": eval,
        "exclude_high": eval[eval["confidence"] != "high"].copy(),
        "exclude_high_medium": eval[~eval["confidence"].isin(["high", "medium"])].copy(),
    }
    impact_rows = []
    bucket_rows: list[dict[str, object]] = []
    for scenario_name, scenario_df in scenarios.items():
        result = _scenario_metrics(scenario_df, scenario_name)
        bucket_rows.extend(result.pop("bucket_rows"))
        impact_rows.append(result)

    impact_df = pd.DataFrame(impact_rows)
    bucket_df = pd.DataFrame(bucket_rows)

    # Save artifacts.
    all_flagged.to_csv(output_dir / "flagged_selected_rows_all_seasons.csv", index=False)
    flagged_eval.to_csv(output_dir / "flagged_eval_rows.csv", index=False)
    provider_scope.to_csv(output_dir / "provider_scope_summary.csv", index=False)
    season_scope.to_csv(output_dir / "season_scope_summary.csv", index=False)
    impact_df.to_csv(output_dir / "impact_summary.csv", index=False)
    bucket_df.to_csv(output_dir / "probability_bucket_impact.csv", index=False)
    (output_dir / "audit_config.json").write_text(
        json.dumps(
            {
                "benchmark_dir": str(benchmark_dir),
                "sigma_study_dir": str(sigma_study_dir),
                "lines_table": lines_table,
                "eval_seasons": config.eval_seasons,
                "all_seasons": config.all_seasons,
                "detection_logic": {
                    "high": [
                        "duplicate_provider_conflict",
                        "majority_sign_conflict",
                        "strong_moneyline_conflict",
                    ],
                    "medium": ["moneyline_conflict"],
                    "low": ["extreme_rating_context_conflict"],
                },
            },
            indent=2,
        )
    )
    (output_dir / "summary.md").write_text(
        _summary_markdown(
            impact_df=impact_df,
            provider_scope_df=provider_scope,
            season_scope_df=season_scope,
            flagged_eval_df=flagged_eval[
                [
                    "season",
                    "gameId",
                    "homeTeam",
                    "awayTeam",
                    "provider",
                    "book_spread",
                    "homeMoneyline",
                    "awayMoneyline",
                    "pick_prob",
                    "edge_home_points",
                    "confidence",
                    "reason_flag",
                ]
            ],
        )
    )

    click.echo(f"Line integrity audit written to {output_dir}")
    click.echo(f"Flagged eval rows: {len(flagged_eval)}")


if __name__ == "__main__":
    main()

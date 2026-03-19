#!/usr/bin/env python3
"""Rebuild a staged historical fct_lines table from bronze with conservative repairs."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow as pa

from src import config, s3_reader
from src.features import load_games

BRONZE_LINES_PREFIX = "bronze/lines"
STAGED_TABLE = "fct_lines_repaired_v1"
AUDIT_PREFIX = "ref/fct_lines_repaired_v1_audit"
NUMERIC_COLS = [
    "spread",
    "spreadOpen",
    "overUnder",
    "overUnderOpen",
    "homeMoneyline",
    "awayMoneyline",
]
ALL_SEASONS = list(range(2016, 2027))


@dataclass(frozen=True)
class RepairConfig:
    seasons: list[int]
    strong_ml_home_favorite: int = -250
    strong_ml_away_underdog: int = 200
    strong_ml_min_abs_spread: float = 4.0
    medium_ml_home_favorite: int = -180
    medium_ml_away_underdog: int = 150
    medium_ml_min_abs_spread: float = 3.0
    corroborating_min_abs_spread: float = 3.0
    majority_min_net_votes: int = 2


def _parse_asof_from_key(key: str) -> str | None:
    match = re.search(r"/asof=([0-9]{4}-[0-9]{2}-[0-9]{2})/", key)
    return match.group(1) if match else None


def _to_table(df: pd.DataFrame) -> pa.Table:
    if df.empty:
        return pa.table({})
    prepared = df.copy()
    for col in prepared.columns:
        if prepared[col].dtype == "object":
            prepared[col] = prepared[col].where(prepared[col].notna(), None)
    return pa.Table.from_pandas(prepared, preserve_index=False)


def _expand_bronze_records(df: pd.DataFrame, source_key: str, source_asof: str, season: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rec in df.to_dict("records"):
        lines = rec.get("lines")
        if isinstance(lines, str):
            raw = lines.strip()
            if raw and raw not in {"[]", "null", "None"}:
                try:
                    lines = json.loads(raw)
                except Exception:
                    try:
                        lines = ast.literal_eval(raw)
                    except Exception:
                        lines = None
        if not isinstance(lines, list) or not lines:
            continue
        base = {k: v for k, v in rec.items() if k != "lines"}
        base["source_key"] = source_key
        base["source_asof"] = source_asof
        base["season"] = season
        for line in lines:
            if not isinstance(line, dict):
                continue
            provider = line.get("provider") or line.get("providerName") or line.get("source")
            if not provider:
                continue
            row = dict(base)
            row.update(line)
            row["provider"] = provider
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "gameId" in out.columns:
        out["gameId"] = pd.to_numeric(out["gameId"], errors="coerce").astype("Int64")
    out["source_asof"] = pd.to_datetime(out["source_asof"], errors="coerce")
    return out


def _load_bronze_lines_for_season(season: int) -> pd.DataFrame:
    prefix = f"{BRONZE_LINES_PREFIX}/season={season}/"
    keys = s3_reader.list_parquet_keys(prefix)
    frames: list[pd.DataFrame] = []
    for key in keys:
        asof = _parse_asof_from_key(key)
        if asof is None:
            continue
        tbl = s3_reader.read_parquet_table([key])
        if tbl.num_rows == 0:
            continue
        expanded = _expand_bronze_records(tbl.to_pandas(), key, asof, season)
        if not expanded.empty:
            frames.append(expanded)
    if not frames:
        return pd.DataFrame()
    lines = pd.concat(frames, ignore_index=True)
    lines = lines[lines["gameId"].notna() & lines["provider"].notna()].copy()
    return lines


def _best_provider_rows(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return lines

    def _has_col(col: str) -> pd.Series:
        if col in lines.columns:
            return lines[col].notna().astype(int)
        return pd.Series(0, index=lines.index, dtype=int)

    ordered = (
        lines.assign(
            _has_spread=_has_col("spread"),
            _has_home_ml=_has_col("homeMoneyline"),
            _has_away_ml=_has_col("awayMoneyline"),
            _has_total=_has_col("overUnder"),
            _has_spread_open=_has_col("spreadOpen"),
            _has_total_open=_has_col("overUnderOpen"),
        )
        .sort_values(
            [
                "gameId",
                "provider",
                "_has_spread",
                "_has_home_ml",
                "_has_away_ml",
                "_has_total",
                "_has_spread_open",
                "_has_total_open",
                "source_asof",
                "source_key",
            ],
            ascending=[True, True, False, False, False, False, False, False, False, False],
            kind="mergesort",
        )
        .drop(columns=[
            "_has_spread",
            "_has_home_ml",
            "_has_away_ml",
            "_has_total",
            "_has_spread_open",
            "_has_total_open",
        ])
    )
    latest = ordered.drop_duplicates(["gameId", "provider"], keep="first").copy()
    latest["gameId"] = latest["gameId"].astype(int)
    return latest.reset_index(drop=True)


def _latest_provider_rows(lines: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for older callers/tests."""
    return _best_provider_rows(lines)


def _spread_sign(series: pd.Series, min_abs: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mask = values.notna() & (values.abs() >= min_abs)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[mask] = np.sign(values.loc[mask])
    return out


def _moneyline_conflict(spread: float, home_ml: float, away_ml: float, cfg: RepairConfig) -> str | None:
    if pd.isna(spread) or pd.isna(home_ml) or pd.isna(away_ml):
        return None
    if (
        home_ml <= cfg.strong_ml_home_favorite
        and away_ml >= cfg.strong_ml_away_underdog
        and spread > cfg.strong_ml_min_abs_spread
    ) or (
        away_ml <= cfg.strong_ml_home_favorite
        and home_ml >= cfg.strong_ml_away_underdog
        and spread < -cfg.strong_ml_min_abs_spread
    ):
        return "strong"
    if (
        home_ml <= cfg.medium_ml_home_favorite
        and away_ml >= cfg.medium_ml_away_underdog
        and spread > cfg.medium_ml_min_abs_spread
    ) or (
        away_ml <= cfg.medium_ml_home_favorite
        and home_ml >= cfg.medium_ml_away_underdog
        and spread < -cfg.medium_ml_min_abs_spread
    ):
        return "medium"
    return None


def _repair_latest_rows(latest: pd.DataFrame, games: pd.DataFrame, cfg: RepairConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if latest.empty:
        return latest, pd.DataFrame()

    game_meta = games[
        ["gameId", "startDate", "homeTeam", "awayTeam", "homeTeamId", "awayTeamId"]
    ].drop_duplicates("gameId", keep="last") if not games.empty else pd.DataFrame()
    if not game_meta.empty:
        latest = latest.merge(game_meta, on="gameId", how="left")

    repaired_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for game_id, game_df in latest.groupby("gameId", sort=False):
        working = game_df.copy()
        working["spread_sign_for_votes"] = _spread_sign(
            working["spread"], cfg.corroborating_min_abs_spread
        )
        majority_sum = float(working["spread_sign_for_votes"].fillna(0.0).sum())

        for _, row in working.iterrows():
            original_spread = row.get("spread")
            original_spread_open = row.get("spreadOpen")
            row_sign = (
                float(np.sign(original_spread))
                if pd.notna(original_spread) and abs(float(original_spread)) >= cfg.corroborating_min_abs_spread
                else np.nan
            )
            peers = working[working["provider"] != row.get("provider")].copy()
            peer_sign = peers["spread_sign_for_votes"]
            same_count = int((peer_sign == row_sign).sum()) if pd.notna(row_sign) else 0
            opposite_count = int((peer_sign == -row_sign).sum()) if pd.notna(row_sign) else 0
            non_consensus_peers = peers[peers["provider"] != "consensus"]
            non_consensus_sign = non_consensus_peers["spread_sign_for_votes"]
            non_consensus_same = int((non_consensus_sign == row_sign).sum()) if pd.notna(row_sign) else 0
            non_consensus_opp = int((non_consensus_sign == -row_sign).sum()) if pd.notna(row_sign) else 0

            ml_conflict = _moneyline_conflict(
                pd.to_numeric(row.get("spread"), errors="coerce"),
                pd.to_numeric(row.get("homeMoneyline"), errors="coerce"),
                pd.to_numeric(row.get("awayMoneyline"), errors="coerce"),
                cfg,
            )
            majority_conflict = (
                pd.notna(row_sign)
                and abs(majority_sum) >= cfg.majority_min_net_votes
                and np.sign(majority_sum) != row_sign
            )

            action = "keep"
            confidence = "none"
            reasons: list[str] = []
            repaired_spread = original_spread
            repaired_spread_open = original_spread_open

            corroborated_flip = False
            if ml_conflict == "strong":
                if majority_conflict:
                    corroborated_flip = True
                    reasons.append("strong_moneyline_conflict")
                    reasons.append("majority_sign_conflict")
                elif (
                    row.get("provider") == "consensus"
                    and non_consensus_opp >= 1
                    and non_consensus_same == 0
                ):
                    corroborated_flip = True
                    reasons.append("strong_moneyline_conflict")
                    reasons.append("corroborating_non_consensus_opposite_sign")
                elif row.get("provider") != "consensus" and opposite_count >= 2 and same_count == 0:
                    corroborated_flip = True
                    reasons.append("strong_moneyline_conflict")
                    reasons.append("cross_provider_opposite_sign")

            if corroborated_flip:
                action = "flip_sign"
                confidence = "high"
                if pd.notna(repaired_spread):
                    repaired_spread = -float(repaired_spread)
                if pd.notna(repaired_spread_open):
                    repaired_spread_open = -float(repaired_spread_open)
            elif ml_conflict == "strong":
                action = "exclude"
                confidence = "high"
                reasons.append("strong_moneyline_conflict")
            elif ml_conflict == "medium" or (
                row.get("provider") == "consensus" and majority_conflict
            ):
                action = "exclude"
                confidence = "medium"
                if ml_conflict == "medium":
                    reasons.append("moneyline_conflict")
                if row.get("provider") == "consensus" and majority_conflict:
                    reasons.append("majority_sign_conflict")

            audit_row = dict(row)
            audit_row.update(
                {
                    "action": action,
                    "confidence": confidence,
                    "repair_reason": "|".join(reasons),
                    "majority_sign_sum": majority_sum,
                    "peer_same_sign_count": same_count,
                    "peer_opposite_sign_count": opposite_count,
                    "non_consensus_same_sign_count": non_consensus_same,
                    "non_consensus_opposite_sign_count": non_consensus_opp,
                    "original_spread": original_spread,
                    "repaired_spread": repaired_spread,
                    "original_spreadOpen": original_spread_open,
                    "repaired_spreadOpen": repaired_spread_open,
                }
            )
            audit_rows.append(audit_row)

            if action == "exclude":
                continue

            repaired = dict(row)
            repaired["spread"] = repaired_spread
            if "spreadOpen" in repaired:
                repaired["spreadOpen"] = repaired_spread_open
            repaired["repair_action"] = action
            repaired["repair_confidence"] = confidence
            repaired["repair_reason"] = "|".join(reasons)
            repaired_rows.append(repaired)

    repaired_df = pd.DataFrame(repaired_rows)
    audit_df = pd.DataFrame(audit_rows)
    return repaired_df, audit_df


def _write_tables(
    repaired_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    season: int,
    staged_table: str,
    audit_prefix: str,
) -> None:
    if not repaired_df.empty:
        repaired_key = f"{config.SILVER_PREFIX}/{staged_table}/season={season}/part-rebuilt.parquet"
        s3_reader.write_parquet_to_s3(_to_table(repaired_df), repaired_key)
    if not audit_df.empty:
        audit_key = f"{audit_prefix}/season={season}/part-rebuilt.parquet"
        s3_reader.write_parquet_to_s3(_to_table(audit_df), audit_key)


@click.command()
@click.option(
    "--start-season",
    default=min(ALL_SEASONS),
    type=int,
    help="First season (season-end year) to rebuild.",
)
@click.option(
    "--end-season",
    default=max(ALL_SEASONS),
    type=int,
    help="Last season (season-end year) to rebuild.",
)
@click.option(
    "--staged-table",
    default=STAGED_TABLE,
    help="Silver table name to write repaired provider rows into.",
)
@click.option(
    "--audit-prefix",
    default=AUDIT_PREFIX,
    help="S3 prefix to write repair audit trail into.",
)
@click.option(
    "--output-dir",
    default="artifacts/fct_lines_repaired_v1_rebuild",
    type=click.Path(path_type=Path),
    help="Local artifact directory for rebuild summaries.",
)
def main(
    start_season: int,
    end_season: int,
    staged_table: str,
    audit_prefix: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(range(start_season, end_season + 1))
    cfg = RepairConfig(seasons=seasons)

    summary_rows: list[dict[str, object]] = []
    audit_frames: list[pd.DataFrame] = []
    for season in seasons:
        click.echo(f"Rebuilding season {season}...")
        bronze = _load_bronze_lines_for_season(season)
        selected = _best_provider_rows(bronze)
        games = load_games(season)
        repaired, audit = _repair_latest_rows(selected, games, cfg)
        _write_tables(repaired, audit, season, staged_table, audit_prefix)
        if not audit.empty:
            audit_frames.append(audit)
        summary_rows.append(
            {
                "season": season,
                "bronze_rows": int(len(bronze)),
                "selected_provider_rows": int(len(selected)),
                "repaired_rows_written": int(len(repaired)),
                "audit_rows": int(len(audit)),
                "flip_sign_rows": int((audit["action"] == "flip_sign").sum()) if not audit.empty else 0,
                "excluded_rows": int((audit["action"] == "exclude").sum()) if not audit.empty else 0,
                "high_confidence_actions": int((audit["confidence"] == "high").sum()) if not audit.empty else 0,
                "medium_confidence_actions": int((audit["confidence"] == "medium").sum()) if not audit.empty else 0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "repair_summary.csv", index=False)
    if audit_frames:
        full_audit = pd.concat(audit_frames, ignore_index=True)
        full_audit.to_csv(output_dir / "repair_audit_rows.csv", index=False)
    else:
        full_audit = pd.DataFrame()

    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "bronze_prefix": BRONZE_LINES_PREFIX,
                "staged_table": staged_table,
                "audit_prefix": audit_prefix,
                "seasons": seasons,
                "repair_logic": {
                    "flip_sign": [
                        "strong_moneyline_conflict + majority_sign_conflict",
                        "strong_moneyline_conflict + corroborating_non_consensus_opposite_sign for consensus rows",
                        "strong_moneyline_conflict + unanimous_opposite_peer_signals for non-consensus rows",
                    ],
                    "exclude": [
                        "strong_moneyline_conflict without corroborated repair",
                        "moneyline_conflict",
                        "consensus majority_sign_conflict without strong corroboration",
                    ],
                    "keep": ["all other selected provider rows"],
                },
            },
            indent=2,
        )
    )
    click.echo(f"Rebuild summary written to {output_dir}")
    click.echo(f"Staged repaired rows written to s3://{config.S3_BUCKET}/{config.SILVER_PREFIX}/{staged_table}/")


if __name__ == "__main__":
    main()

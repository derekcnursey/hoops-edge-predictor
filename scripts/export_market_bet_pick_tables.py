#!/usr/bin/env python3
"""Export historical internal betting pick tables by strategy bucket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.features import load_games
from src.market_bet_filter import (
    PROMOTED_INTERNAL_FILTER_THRESHOLD,
    RAW_EDGE_BASELINE_THRESHOLD,
    classify_signal_driver,
)


def _pick_team(row: pd.Series) -> str:
    return row["homeTeam"] if row["model_pick_side"] == "HOME" else row["awayTeam"]


def _pick_line(row: pd.Series) -> float:
    """Return sportsbook-style line for the picked team."""
    book_spread = float(row["book_spread"])
    return book_spread if row["model_pick_side"] == "HOME" else -book_spread


def _model_line_for_pick(row: pd.Series) -> float:
    """Return sportsbook-style Hoops Edge line for the picked team."""
    model_spread = float(row["model_spread"])
    return model_spread if row["model_pick_side"] == "HOME" else -model_spread


def _build_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    game_parts: list[pd.DataFrame] = []
    for season in sorted(out["season"].dropna().astype(int).unique().tolist()):
        season_games = load_games(season)
        if season_games.empty:
            continue
        keep_cols = [c for c in ["gameId", "homeScore", "awayScore"] if c in season_games.columns]
        if "gameId" not in keep_cols:
            continue
        season_frame = season_games[keep_cols].copy()
        season_frame["season"] = season
        game_parts.append(season_frame)
    if game_parts:
        scores = pd.concat(game_parts, ignore_index=True).drop_duplicates(["season", "gameId"], keep="last")
        out = out.merge(scores, on=["season", "gameId"], how="left", validate="many_to_one")

    out["game"] = out["awayTeam"].astype(str) + " at " + out["homeTeam"].astype(str)
    out["pick_team"] = out.apply(_pick_team, axis=1)
    out["market_line_for_pick"] = out.apply(_pick_line, axis=1)
    out["hoops_edge_line_for_pick"] = out.apply(_model_line_for_pick, axis=1)
    out["final_score"] = np.where(
        out["homeScore"].notna() & out["awayScore"].notna(),
        out["awayTeam"].astype(str) + " " + out["awayScore"].fillna(0).astype(int).astype(str)
        + " - "
        + out["homeTeam"].astype(str) + " " + out["homeScore"].fillna(0).astype(int).astype(str),
        None,
    )
    out["filter_pass"] = out["disagreement_logit_score"] >= PROMOTED_INTERNAL_FILTER_THRESHOLD
    out["raw_edge_pass"] = out["pick_prob_edge"] >= RAW_EDGE_BASELINE_THRESHOLD
    out["signal_driver"] = out.apply(classify_signal_driver, axis=1)
    out["disagreement_context"] = np.select(
        [
            out["pick_team_new_disagreement"].fillna(False),
            out["pick_team_persistent_disagreement"].fillna(False),
        ],
        [
            "new/transient",
            "persistent",
        ],
        default="none",
    )
    out["bet_result"] = np.where(
        out["bet_won"].isna(),
        "push",
        np.where(out["bet_won"].astype(float) == 1.0, "win", "loss"),
    )
    keep_cols = [
        "season",
        "game_date",
        "slice",
        "gameId",
        "game",
        "final_score",
        "pick_team",
        "model_pick_side",
        "hoops_edge_line_for_pick",
        "market_line_for_pick",
        "pick_prob_edge",
        "disagreement_logit_score",
        "he_market_edge_for_pick",
        "abs_he_vs_market_edge",
        "disagreement_context",
        "pick_team_recent_same_sign_count_21d",
        "pick_team_prior_same_sign_streak",
        "neutral_site_flag",
        "is_conference_tournament",
        "is_ncaa_tournament",
        "signal_driver",
        "bet_result",
        "roi_per_1_at_minus_110",
    ]
    out = out[keep_cols].sort_values(
        ["season", "game_date", "disagreement_logit_score", "pick_prob_edge", "gameId"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export historical internal bet lists by choice bucket.")
    parser.add_argument(
        "--scored-bets",
        type=Path,
        default=config.ARTIFACTS_DIR / "market_bet_filter_v1" / "historical_scored_bets.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS_DIR / "market_bet_pick_tables_v1",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.scored_bets).copy()
    df = _build_export_frame(df)

    filter_pass = df["disagreement_logit_score"] >= PROMOTED_INTERNAL_FILTER_THRESHOLD
    raw_pass = df["pick_prob_edge"] >= RAW_EDGE_BASELINE_THRESHOLD
    ncaa = df["is_ncaa_tournament"].fillna(False)

    groups = {
        "promoted_internal_filter": filter_pass & ~ncaa,
        "raw_edge_baseline": raw_pass,
        "overlap": filter_pass & raw_pass & ~ncaa,
        "filter_only": filter_pass & ~raw_pass & ~ncaa,
        "raw_only": raw_pass & ~filter_pass,
        "ncaa_caution": filter_pass & ncaa,
    }

    summary_rows: list[dict[str, object]] = []
    for group_name, mask in groups.items():
        out = df[mask].copy()
        out.to_csv(output_dir / f"{group_name}.csv", index=False)
        summary_rows.append(
            {
                "group": group_name,
                "rows": int(len(out)),
                "date_min": str(out["game_date"].min().date()) if not out.empty else None,
                "date_max": str(out["game_date"].max().date()) if not out.empty else None,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("group").reset_index(drop=True)
    summary.to_csv(output_dir / "summary.csv", index=False)

    manifest = {
        "scored_bets_path": str(args.scored_bets),
        "promoted_threshold": PROMOTED_INTERNAL_FILTER_THRESHOLD,
        "raw_edge_threshold": RAW_EDGE_BASELINE_THRESHOLD,
        "groups": list(groups.keys()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

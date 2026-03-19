#!/usr/bin/env python3
"""Build NCAA bracket-builder data from a canonical tournament field input.

Default flow:
  1. Read `site/public/data/ncaa_field_input_<season>.json`
  2. Validate the 68-team bracket field and First Four structure
  3. Enrich the field with rankings metadata used by the current frontend
  4. Precompute neutral-site matchup predictions with the promoted model

Dev-only fallback:
  Pass `--use-rankings-fallback` to derive a temporary field from the current
  rankings JSON. This is never used implicitly when a canonical field file is
  missing; it must be requested explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch

# Avoid pathological CPU thread contention during batch inference.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.build_rankings_json import _load_latest_ratings
from scripts.rebuild_tourney_jsons import _build_synthetic_rows, _predict_pairwise_probability
from src import config, s3_reader
from src.features import build_features, load_lines, load_research_lines
from src.infer import (
    _fill_nan_with_impute_means,
    _fill_nan_with_scaler_means,
    _predict_mu_values,
    american_to_breakeven,
    load_regressor,
    load_mu_regressor,
    normal_cdf,
    prob_to_american,
)
from src.line_selection import select_preferred_lines
from src.mean_model_variants import (
    TEAM_AB_ELITE_TAIL_ROUND64_V1,
    build_mean_model_feature_frame,
    build_team_ab_elite_tail_round64_contract,
    build_team_ab_source,
    swap_team_ab_source,
)
from src.ml_odds import site_home_win_prob_from_mu_sigma
from src.ncaa_synthetic_fallback import synthetic_fallback_margin
from src.tournament_adjustments import market_blended_display_margin
from src.trainer import load_scaler, load_tree_regressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "site" / "public" / "data"

CURRENT_SEASON = 2026
REGIONS = ["East", "West", "South", "Midwest"]
REGION_PAIRINGS = (
    {"seed_a": 1, "seed_b": 16, "slot": 1},
    {"seed_a": 8, "seed_b": 9, "slot": 2},
    {"seed_a": 5, "seed_b": 12, "slot": 3},
    {"seed_a": 4, "seed_b": 13, "slot": 4},
    {"seed_a": 6, "seed_b": 11, "slot": 5},
    {"seed_a": 3, "seed_b": 14, "slot": 6},
    {"seed_a": 7, "seed_b": 10, "slot": 7},
    {"seed_a": 2, "seed_b": 15, "slot": 8},
)
SEED_TO_REGION_SLOT = {
    pairing["seed_a"]: pairing["slot"] for pairing in REGION_PAIRINGS
} | {
    pairing["seed_b"]: pairing["slot"] for pairing in REGION_PAIRINGS
}
FINAL_FOUR_REGION_PAIRS = {
    frozenset({"East", "South"}),
    frozenset({"West", "Midwest"}),
}
REGIONAL_SWEET_16_DATES = {
    2026: {
        "West": "2026-03-26",
        "South": "2026-03-26",
        "East": "2026-03-27",
        "Midwest": "2026-03-27",
    }
}
REGIONAL_ELITE_8_DATES = {
    2026: {
        "West": "2026-03-28",
        "South": "2026-03-28",
        "East": "2026-03-29",
        "Midwest": "2026-03-29",
    }
}
FINAL_FOUR_DATES = {2026: "2026-04-04"}
NATIONAL_TITLE_DATES = {2026: "2026-04-06"}
FIELD_INPUT_TEMPLATE = "ncaa_field_input_{season}.json"
FIELD_OUTPUT_TEMPLATE = "ncaa_bracket_builder_{season}.json"
MATCHUPS_OUTPUT_TEMPLATE = "ncaa_matchup_predictions_{season}.json"
RANKINGS_TEMPLATE = "rankings_{season}.json"
BRACKET_MODEL_VARIANT_LEGACY_SYNTHETIC = "legacy_synthetic"
BRACKET_MODEL_VARIANT_TEAM_AB = TEAM_AB_ELITE_TAIL_ROUND64_V1
SUPPORTED_BRACKET_MODEL_VARIANTS = {
    BRACKET_MODEL_VARIANT_LEGACY_SYNTHETIC,
    BRACKET_MODEL_VARIANT_TEAM_AB,
}


def _read_json(path: Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _season_path(template: str, season: int) -> Path:
    return DATA_DIR / template.format(season=season)


def _region_slug(region: str) -> str:
    return region.lower().replace(" ", "-")


def _label_for_round_id(round_id: str | None) -> str | None:
    mapping = {
        "first-four": "First Four",
        "round-of-64": "Round of 64",
        "round-of-32": "Round of 32",
        "sweet-16": "Sweet 16",
        "elite-8": "Elite 8",
        "final-four": "Final Four",
        "national-championship": "National Championship",
    }
    return mapping.get(round_id)


def _round_timestamp_et(date_text: str | None) -> pd.Timestamp | None:
    if not date_text:
        return None
    return pd.Timestamp(f"{date_text} 12:00:00", tz="America/New_York")


def _main_bracket_slot(region: str, seed: int) -> str:
    return f"{_region_slug(region)}-{seed}"


def _seed_region_order(seed: int) -> list[str]:
    return REGIONS if seed % 2 == 1 else list(reversed(REGIONS))


def _load_rankings_rows(season: int) -> dict[int, dict[str, Any]]:
    rankings_path = _season_path(RANKINGS_TEMPLATE, season)
    payload = _read_json(rankings_path)
    rows = payload.get("teams", [])
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_id[int(row["team_id"])] = row
    return by_id


def _normalize_bracket_model_variant(value: str | None) -> str:
    variant = (value or BRACKET_MODEL_VARIANT_LEGACY_SYNTHETIC).strip().lower()
    if variant not in SUPPORTED_BRACKET_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported bracket matchup model variant {variant!r}. "
            f"Expected one of {sorted(SUPPORTED_BRACKET_MODEL_VARIANTS)}."
        )
    return variant


def _normalize_efficiency_source(value: str | None, *, env_name: str) -> str:
    source = (value or "gold").strip().lower()
    if source not in config.SUPPORTED_EFFICIENCY_SOURCES:
        raise ValueError(
            f"Unsupported {env_name} {source!r}. "
            f"Expected one of {sorted(config.SUPPORTED_EFFICIENCY_SOURCES)}."
        )
    return source


def _build_bracket_ratings_frame(
    season: int,
    *,
    efficiency_source: str,
    gold_table_name: str | None,
) -> tuple[pd.DataFrame, dict[str, float], str]:
    rankings = pd.DataFrame.from_records(list(_load_rankings_rows(season).values()))
    ratings, ratings_source = _load_latest_ratings(
        season,
        efficiency_source=efficiency_source,
        gold_table_name=gold_table_name,
    )
    if ratings.empty:
        raise ValueError(f"Missing latest ratings rows for NCAA bracket builder season {season}")

    ratings = ratings.copy()
    ratings["team_id"] = ratings["teamId"].astype(int)
    ratings["barthag_rank"] = ratings["barthag"].rank(ascending=False, method="first")
    ratings["adj_net_model"] = pd.to_numeric(ratings["adj_oe"], errors="coerce") - pd.to_numeric(
        ratings["adj_de"], errors="coerce"
    )
    rename_map = {
        "adj_oe": "adj_oe_model",
        "adj_de": "adj_de_model",
        "adj_tempo": "adj_tempo_model",
        "barthag": "barthag_model",
        "sos_oe": "sos_oe_model",
        "sos_de": "sos_de_model",
    }
    ratings = ratings.rename(columns=rename_map)

    conf_frame = rankings[["team_id", "conference"]].drop_duplicates("team_id").copy()
    ratings = ratings.merge(conf_frame, on="team_id", how="left", suffixes=("", "_rankings"))
    if "conference" not in ratings.columns and "conference_rankings" in ratings.columns:
        ratings["conference"] = ratings["conference_rankings"]
    elif "conference_rankings" in ratings.columns:
        ratings["conference"] = ratings["conference"].fillna(ratings["conference_rankings"])
    conf_strength_lookup = (
        ratings.dropna(subset=["conference", "adj_net_model"])
        .groupby("conference")["adj_net_model"]
        .mean()
        .to_dict()
    )
    return ratings, conf_strength_lookup, ratings_source


def _build_team_ab_bracket_source(
    team_a: pd.Series,
    team_b: pd.Series,
    *,
    season: int,
    conf_strength_lookup: dict[str, float],
    round_label: str | None,
    start_time: object,
    team_a_rest_days: float | None = None,
    team_b_rest_days: float | None = None,
    team_a_state: dict[str, Any] | None = None,
    team_b_state: dict[str, Any] | None = None,
) -> pd.DataFrame:
    fallback_ts = pd.Timestamp(year=season, month=3, day=15, tz="UTC")
    start_ts = pd.to_datetime(start_time, errors="coerce", utc=True)
    if pd.isna(start_ts):
        start_ts = fallback_ts

    def _row_value(row: pd.Series, preferred: str, fallback: str | None = None) -> float | None:
        value = row.get(preferred)
        if pd.isna(value) and fallback is not None:
            value = row.get(fallback)
        if pd.isna(value):
            return None
        return float(value)

    def _state_value(
        state: dict[str, Any] | None,
        key: str,
    ) -> Any:
        if state is None:
            return None
        value = state.get(key)
        if value is None or pd.isna(value):
            return None
        return value

    def _coalesce(*values: Any) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            return value
        return None

    def _team_payload(prefix: str, row: pd.Series, state: dict[str, Any] | None) -> dict[str, Any]:
        conf = str(row.get("conference") or "")
        return {
            f"{prefix}_team_id": int(_coalesce(_state_value(state, "team_id"), row["team_id"])),
            f"{prefix}_name": str(_coalesce(_state_value(state, "name"), row.get("team"), row.get("team_name"), "")),
            f"{prefix}_adj_oe": _coalesce(_state_value(state, "adj_oe"), _row_value(row, "adj_oe_model", "adj_oe")),
            f"{prefix}_adj_de": _coalesce(_state_value(state, "adj_de"), _row_value(row, "adj_de_model", "adj_de")),
            f"{prefix}_barthag": _coalesce(_state_value(state, "barthag"), _row_value(row, "barthag_model", "barthag")),
            f"{prefix}_conf_strength": _coalesce(_state_value(state, "conf_strength"), conf_strength_lookup.get(conf)),
            f"{prefix}_sos_oe": _coalesce(_state_value(state, "sos_oe"), _row_value(row, "sos_oe_model")),
            f"{prefix}_sos_de": _coalesce(_state_value(state, "sos_de"), _row_value(row, "sos_de_model")),
            f"{prefix}_form_delta": _state_value(state, "form_delta"),
            f"{prefix}_rest_days": team_a_rest_days if prefix == "team_a" else team_b_rest_days,
            f"{prefix}_eff_fg_pct": _state_value(state, "eff_fg_pct"),
            f"{prefix}_ft_rate": _state_value(state, "ft_rate"),
            f"{prefix}_off_rebound_pct": _state_value(state, "off_rebound_pct"),
            f"{prefix}_tov_rate": _state_value(state, "tov_rate"),
            f"{prefix}_margin_std": _state_value(state, "margin_std"),
            f"{prefix}_barthag_rank": _coalesce(_state_value(state, "barthag_rank"), _row_value(row, "barthag_rank", "rank")),
            f"{prefix}_seed": _coalesce(_state_value(state, "seed"), _row_value(row, "seed")),
        }

    team_a_payload = _team_payload("team_a", team_a, team_a_state)
    team_b_payload = _team_payload("team_b", team_b, team_b_state)
    row = {
        "season": int(season),
        "gameId": -1,
        "startDate": start_ts,
        "actual_margin": np.nan,
        "homeScore": np.nan,
        "awayScore": np.nan,
        "target_margin_ab": np.nan,
        "neutral_site": 1.0,
        "team_a_is_home_non_neutral": 0.0,
        "team_a_hca": 0.0,
        "home_team_hca": 0.0,
        "tournament": "NCAA",
        "gameType": "TRNMNT",
        "conferenceGame": False,
        "gameNotes": round_label,
        "neutral_subtype": "ncaa_neutral",
        "round_label": round_label,
        "pair_augmented": 0,
        "homeTeamId": team_a_payload["team_a_team_id"],
        "awayTeamId": team_b_payload["team_b_team_id"],
        "homeTeam": team_a_payload["team_a_name"],
        "awayTeam": team_b_payload["team_b_name"],
        "home_team_adj_oe": team_a_payload["team_a_adj_oe"],
        "away_team_adj_oe": team_b_payload["team_b_adj_oe"],
        "home_team_adj_de": team_a_payload["team_a_adj_de"],
        "away_team_adj_de": team_b_payload["team_b_adj_de"],
        "home_team_BARTHAG": team_a_payload["team_a_barthag"],
        "away_team_BARTHAG": team_b_payload["team_b_barthag"],
        "home_conf_strength": team_a_payload["team_a_conf_strength"],
        "away_conf_strength": team_b_payload["team_b_conf_strength"],
        "home_sos_oe": team_a_payload["team_a_sos_oe"],
        "away_sos_oe": team_b_payload["team_b_sos_oe"],
        "home_sos_de": team_a_payload["team_a_sos_de"],
        "away_sos_de": team_b_payload["team_b_sos_de"],
        "home_form_delta": team_a_payload["team_a_form_delta"],
        "away_form_delta": team_b_payload["team_b_form_delta"],
        "home_rest_days": team_a_payload["team_a_rest_days"],
        "away_rest_days": team_b_payload["team_b_rest_days"],
        "home_eff_fg_pct": team_a_payload["team_a_eff_fg_pct"],
        "away_eff_fg_pct": team_b_payload["team_b_eff_fg_pct"],
        "home_ft_rate": team_a_payload["team_a_ft_rate"],
        "away_ft_rate": team_b_payload["team_b_ft_rate"],
        "home_off_rebound_pct": team_a_payload["team_a_off_rebound_pct"],
        "away_off_rebound_pct": team_b_payload["team_b_off_rebound_pct"],
        "home_tov_rate": team_a_payload["team_a_tov_rate"],
        "away_tov_rate": team_b_payload["team_b_tov_rate"],
        "home_margin_std": team_a_payload["team_a_margin_std"],
        "away_margin_std": team_b_payload["team_b_margin_std"],
        "home_barthag_rank": team_a_payload["team_a_barthag_rank"],
        "away_barthag_rank": team_b_payload["team_b_barthag_rank"],
        "homeSeed": team_a_payload["team_a_seed"],
        "awaySeed": team_b_payload["team_b_seed"],
    }
    row.update(team_a_payload)
    row.update(team_b_payload)
    return pd.DataFrame([row])


def _predict_team_ab_pairwise_margin(
    team_a: pd.Series,
    team_b: pd.Series,
    *,
    season: int,
    round_label: str | None,
    start_time: object,
    conf_strength_lookup: dict[str, float],
    mu_regressor: object,
    mu_feature_order: list[str],
    mu_model_type: str,
    mu_impute_means: np.ndarray | None,
    team_a_rest_days: float | None = None,
    team_b_rest_days: float | None = None,
    team_a_state: dict[str, Any] | None = None,
    team_b_state: dict[str, Any] | None = None,
) -> float:
    source = _build_team_ab_bracket_source(
        team_a,
        team_b,
        season=season,
        conf_strength_lookup=conf_strength_lookup,
        round_label=round_label,
        start_time=start_time,
        team_a_rest_days=team_a_rest_days,
        team_b_rest_days=team_b_rest_days,
        team_a_state=team_a_state,
        team_b_state=team_b_state,
    )
    team_ab_source = build_team_ab_source(source)
    feature_frame = build_team_ab_elite_tail_round64_contract(team_ab_source)
    X_raw = _fill_nan_with_impute_means(feature_frame[mu_feature_order].copy(), mu_impute_means)
    mu = _predict_mu_values(mu_regressor, mu_model_type, X_raw, X_raw)
    neutral_mask = (
        pd.to_numeric(team_ab_source["neutral_site"], errors="coerce").fillna(0.0).to_numpy()
        == 1.0
    )
    if neutral_mask.any():
        swap_source = swap_team_ab_source(team_ab_source.iloc[np.flatnonzero(neutral_mask)].reset_index(drop=True))
        swap_frame = build_team_ab_elite_tail_round64_contract(swap_source)
        swap_X_raw = _fill_nan_with_impute_means(swap_frame[mu_feature_order].copy(), mu_impute_means)
        mu_swap = _predict_mu_values(mu_regressor, mu_model_type, swap_X_raw, swap_X_raw)
        mu = np.asarray(mu, dtype=np.float32)
        mu[np.flatnonzero(neutral_mask)] = (mu[np.flatnonzero(neutral_mask)] - mu_swap) / 2.0
    return float(mu[0])


def _load_scheduled_feature_lookup(
    season: int,
    scheduled_lookup: dict[str, dict[str, Any]],
    *,
    efficiency_source: str,
    gold_table_name: str | None = None,
) -> dict[int, pd.Series]:
    scheduled_game_ids = sorted(
        {
            int(info["scheduled_game_id"])
            for info in scheduled_lookup.values()
            if info.get("scheduled_game_id") is not None
        }
    )
    if not scheduled_game_ids:
        return {}

    game_dates = sorted(
        {
            pd.to_datetime(info["start_time"], errors="coerce", utc=True)
            .tz_convert("America/New_York")
            .strftime("%Y-%m-%d")
            for info in scheduled_lookup.values()
            if info.get("start_time") is not None and not pd.isna(pd.to_datetime(info["start_time"], errors="coerce", utc=True))
        }
    )
    if not game_dates:
        return {}

    frames: list[pd.DataFrame] = []
    for game_date in game_dates:
        frame = build_features(
            season,
            game_date=game_date,
            no_garbage=True,
            extra_features=config.EXTRA_FEATURES,
            adjust_ff=config.ADJUST_FF,
            adjust_alpha=config.ADJUST_ALPHA,
            adjust_prior_weight=config.ADJUST_PRIOR,
            efficiency_source=efficiency_source,
            gold_table_name=gold_table_name,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["gameId"].isin(scheduled_game_ids)].copy()
    if combined.empty:
        return {}
    combined = combined.drop_duplicates(subset=["gameId"], keep="last")
    return {int(row["gameId"]): row for _, row in combined.iterrows()}


def _build_team_state_lookup(feature_lookup: dict[int, pd.Series]) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}

    def _extract(row: pd.Series, side: str) -> dict[str, Any]:
        prefix = "home" if side == "home" else "away"
        return {
            "team_id": int(row[f"{prefix}TeamId"]),
            "name": row.get(f"{prefix}Team"),
            "adj_oe": row.get(f"{prefix}_team_adj_oe"),
            "adj_de": row.get(f"{prefix}_team_adj_de"),
            "barthag": row.get(f"{prefix}_team_BARTHAG"),
            "conf_strength": row.get(f"{prefix}_conf_strength"),
            "sos_oe": row.get(f"{prefix}_sos_oe"),
            "sos_de": row.get(f"{prefix}_sos_de"),
            "form_delta": row.get(f"{prefix}_form_delta"),
            "rest_days": row.get(f"{prefix}_rest_days"),
            "eff_fg_pct": row.get(f"{prefix}_eff_fg_pct"),
            "ft_rate": row.get(f"{prefix}_ft_rate"),
            "off_rebound_pct": row.get(f"{prefix}_off_rebound_pct"),
            "tov_rate": row.get(f"{prefix}_tov_rate"),
            "margin_std": row.get(f"{prefix}_margin_std"),
            "barthag_rank": row.get(f"{prefix}_barthag_rank"),
            "seed": row.get(f"{prefix}Seed"),
            "start_time": row.get("startDate"),
        }

    ordered_rows = sorted(
        feature_lookup.values(),
        key=lambda row: pd.to_datetime(row.get("startDate"), errors="coerce", utc=True),
    )
    for row in ordered_rows:
        for side in ("home", "away"):
            state = _extract(row, side)
            latest[int(state["team_id"])] = state
    return latest


def _predict_team_ab_scheduled_margin(
    feature_row: pd.Series,
    *,
    team_a_id: int,
    team_b_id: int,
    mu_regressor: object,
    mu_feature_order: list[str],
    mu_model_type: str,
    mu_impute_means: np.ndarray | None,
) -> float:
    if int(feature_row["homeTeamId"]) == team_a_id and int(feature_row["awayTeamId"]) == team_b_id:
        oriented_row = feature_row.to_dict()
    elif int(feature_row["homeTeamId"]) == team_b_id and int(feature_row["awayTeamId"]) == team_a_id:
        oriented_row = feature_row.to_dict()
        swaps = [
            (key, key.replace("home", "away", 1))
            for key in list(oriented_row.keys())
            if key.startswith("home") and key.replace("home", "away", 1) in oriented_row
        ]
        for home_key, away_key in swaps:
            oriented_row[home_key], oriented_row[away_key] = oriented_row[away_key], oriented_row[home_key]
    else:
        raise ValueError(
            f"Scheduled feature row gameId={feature_row['gameId']} does not match expected teams "
            f"{team_a_id} vs {team_b_id}"
        )

    oriented = pd.DataFrame([oriented_row])
    source = build_team_ab_source(oriented)
    feature_frame = build_team_ab_elite_tail_round64_contract(source)
    X_raw = _fill_nan_with_impute_means(feature_frame[mu_feature_order].copy(), mu_impute_means)
    mu = _predict_mu_values(mu_regressor, mu_model_type, X_raw, X_raw)
    neutral_mask = (
        pd.to_numeric(source["neutral_site"], errors="coerce").fillna(0.0).to_numpy()
        == 1.0
    )
    if neutral_mask.any():
        swap_source = swap_team_ab_source(source.iloc[np.flatnonzero(neutral_mask)].reset_index(drop=True))
        swap_frame = build_team_ab_elite_tail_round64_contract(swap_source)
        swap_X_raw = _fill_nan_with_impute_means(swap_frame[mu_feature_order].copy(), mu_impute_means)
        mu_swap = _predict_mu_values(mu_regressor, mu_model_type, swap_X_raw, swap_X_raw)
        mu = np.asarray(mu, dtype=np.float32)
        mu[np.flatnonzero(neutral_mask)] = (mu[np.flatnonzero(neutral_mask)] - mu_swap) / 2.0
    return float(mu[0])


def _extract_region_from_game_notes(note: object) -> str | None:
    value = str(note or "").upper()
    for region in sorted(REGIONS, key=len, reverse=True):
        if f"{region.upper()} REGION" in value:
            return region
    return None


def _build_slot_candidate_lookup(field: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    play_in_team_ids = {
        str(game["id"]): {int(team["team_id"]) for team in game["teams"]}
        for game in field["first_four"]
    }
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for region in field["regions"]:
        region_name = str(region["name"])
        entry_by_seed = {int(entry["seed"]): entry for entry in region["entries"]}
        for pairing in REGION_PAIRINGS:
            entry_a = entry_by_seed[pairing["seed_a"]]
            entry_b = entry_by_seed[pairing["seed_b"]]
            if entry_a["source"] == "team":
                side_a_ids = {int(entry_a["team_id"])}
            else:
                side_a_ids = play_in_team_ids[str(entry_a["play_in_game_id"])]
            if entry_b["source"] == "team":
                side_b_ids = {int(entry_b["team_id"])}
            else:
                side_b_ids = play_in_team_ids[str(entry_b["play_in_game_id"])]
            lookup[(region_name, int(pairing["slot"]))] = {
                "side_a_ids": side_a_ids,
                "side_b_ids": side_b_ids,
            }
    return lookup


def _build_team_bracket_context(field: dict[str, Any]) -> dict[int, dict[str, Any]]:
    team_context: dict[int, dict[str, Any]] = {}
    for region in field["regions"]:
        region_name = str(region["name"])
        for entry in region["entries"]:
            seed = int(entry["seed"])
            slot = SEED_TO_REGION_SLOT[seed]
            if entry["source"] == "team":
                team_context[int(entry["team_id"])] = {
                    "region": region_name,
                    "seed": seed,
                    "slot": slot,
                    "r32_slot": (slot + 1) // 2,
                    "s16_slot": ((slot + 1) // 2 + 1) // 2,
                    "first_four_game_id": None,
                }
    for game in field["first_four"]:
        region_name = str(game["region"])
        seed = int(game["seed"])
        slot = SEED_TO_REGION_SLOT[seed]
        for team in game["teams"]:
            team_context[int(team["team_id"])] = {
                "region": region_name,
                "seed": seed,
                "slot": slot,
                "r32_slot": (slot + 1) // 2,
                "s16_slot": ((slot + 1) // 2 + 1) // 2,
                "first_four_game_id": str(game["id"]),
            }
    return team_context


def _load_bracket_schedule_context(
    field: dict[str, Any],
    season: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, pd.Timestamp]]]:
    team_context = _build_team_bracket_context(field)
    slot_candidates = _build_slot_candidate_lookup(field)
    first_four_candidates = {
        str(game["id"]): {int(team["team_id"]) for team in game["teams"]}
        for game in field["first_four"]
    }

    games_table = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if games_table.num_rows == 0:
        return team_context, {team_id: {} for team_id in team_context}
    games = games_table.to_pandas()
    games = games[games["tournament"].eq("NCAA")].copy()
    if games.empty:
        return team_context, {team_id: {} for team_id in team_context}

    games["round_info"] = games["gameNotes"].map(_round_from_game_notes)
    games["scheduled_round_id"] = games["round_info"].map(lambda item: item[0])
    games["region_name"] = games["gameNotes"].map(_extract_region_from_game_notes)
    games["start_ts"] = pd.to_datetime(games["startDate"], errors="coerce", utc=True).dt.tz_convert(
        "America/New_York"
    )

    first_four_times: dict[str, pd.Timestamp] = {}
    round64_times: dict[tuple[str, int], pd.Timestamp] = {}

    first_four_games = games[games["scheduled_round_id"].eq("first-four")].copy()
    for _, row in first_four_games.iterrows():
        if pd.isna(row.get("start_ts")):
            continue
        participants = {int(row["homeTeamId"]), int(row["awayTeamId"])}
        matches = [
            game_id for game_id, team_ids in first_four_candidates.items() if team_ids == participants
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Unable to map NCAA First Four gameId={row['gameId']} "
                f"{row.get('homeTeam')} vs {row.get('awayTeam')} to a unique bracket play-in"
            )
        first_four_times[matches[0]] = row["start_ts"]

    round64_games = games[games["scheduled_round_id"].eq("round-of-64")].copy()
    for _, row in round64_games.iterrows():
        region_name = row.get("region_name")
        if not isinstance(region_name, str) or pd.isna(row.get("start_ts")):
            continue
        home_id = int(row["homeTeamId"])
        away_id = int(row["awayTeamId"])
        matches: list[int] = []
        for (candidate_region, slot), candidate in slot_candidates.items():
            if candidate_region != region_name:
                continue
            side_a_ids = candidate["side_a_ids"]
            side_b_ids = candidate["side_b_ids"]
            if (home_id in side_a_ids and away_id in side_b_ids) or (
                home_id in side_b_ids and away_id in side_a_ids
            ):
                matches.append(slot)
        if len(matches) != 1:
            raise ValueError(
                f"Unable to map NCAA Round of 64 gameId={row['gameId']} "
                f"{row.get('homeTeam')} vs {row.get('awayTeam')} to a unique bracket slot"
            )
        round64_times[(region_name, matches[0])] = row["start_ts"]

    regional_sweet_16 = REGIONAL_SWEET_16_DATES.get(season, {})
    regional_elite_8 = REGIONAL_ELITE_8_DATES.get(season, {})
    final_four_ts = _round_timestamp_et(FINAL_FOUR_DATES.get(season))
    title_ts = _round_timestamp_et(NATIONAL_TITLE_DATES.get(season))

    team_round_times: dict[int, dict[str, pd.Timestamp]] = {}
    for team_id, info in team_context.items():
        region_name = str(info["region"])
        slot = int(info["slot"])
        r32_slot = int(info["r32_slot"])
        round_times: dict[str, pd.Timestamp] = {}
        first_four_game_id = info.get("first_four_game_id")
        if isinstance(first_four_game_id, str) and first_four_game_id in first_four_times:
            round_times["first-four"] = first_four_times[first_four_game_id]
        round64_ts = round64_times.get((region_name, slot))
        if round64_ts is not None:
            round_times["round-of-64"] = round64_ts
        r32_source_slots = (r32_slot * 2 - 1, r32_slot * 2)
        r32_source_times = [
            round64_times[(region_name, source_slot)]
            for source_slot in r32_source_slots
            if (region_name, source_slot) in round64_times
        ]
        if len(r32_source_times) == 2:
            round_times["round-of-32"] = max(r32_source_times) + pd.Timedelta(days=2)
        sweet_16_ts = _round_timestamp_et(regional_sweet_16.get(region_name))
        if sweet_16_ts is not None:
            round_times["sweet-16"] = sweet_16_ts
        elite_8_ts = _round_timestamp_et(regional_elite_8.get(region_name))
        if elite_8_ts is not None:
            round_times["elite-8"] = elite_8_ts
        if final_four_ts is not None:
            round_times["final-four"] = final_four_ts
        if title_ts is not None:
            round_times["national-championship"] = title_ts
        team_round_times[team_id] = round_times
    return team_context, team_round_times


def _matchup_round_context(
    team_a_id: int,
    team_b_id: int,
    *,
    team_context: dict[int, dict[str, Any]],
    team_round_times: dict[int, dict[str, pd.Timestamp]],
) -> dict[str, Any]:
    team_a_info = team_context[team_a_id]
    team_b_info = team_context[team_b_id]

    if (
        team_a_info.get("first_four_game_id") is not None
        and team_a_info.get("first_four_game_id") == team_b_info.get("first_four_game_id")
    ):
        round_id = "first-four"
    elif team_a_info["region"] == team_b_info["region"]:
        if team_a_info["slot"] == team_b_info["slot"]:
            round_id = "round-of-64"
        elif team_a_info["r32_slot"] == team_b_info["r32_slot"]:
            round_id = "round-of-32"
        elif team_a_info["s16_slot"] == team_b_info["s16_slot"]:
            round_id = "sweet-16"
        else:
            round_id = "elite-8"
    elif frozenset({team_a_info["region"], team_b_info["region"]}) in FINAL_FOUR_REGION_PAIRS:
        round_id = "final-four"
    else:
        round_id = "national-championship"

    previous_round = {
        "round-of-64": "first-four",
        "round-of-32": "round-of-64",
        "sweet-16": "round-of-32",
        "elite-8": "sweet-16",
        "final-four": "elite-8",
        "national-championship": "final-four",
    }.get(round_id)

    current_time = team_round_times.get(team_a_id, {}).get(round_id)
    if current_time is None:
        current_time = team_round_times.get(team_b_id, {}).get(round_id)

    def _rest_days(team_id: int) -> float | None:
        if previous_round is None or current_time is None:
            return None
        previous_time = team_round_times.get(team_id, {}).get(previous_round)
        if previous_time is None:
            return None
        return float((current_time - previous_time).total_seconds() / 86400.0)

    return {
        "round_id": round_id,
        "round_label": _label_for_round_id(round_id),
        "start_time": None if current_time is None else current_time.isoformat(),
        "team_a_rest_days": _rest_days(team_a_id),
        "team_b_rest_days": _rest_days(team_b_id),
    }


def _site_probability_from_mu_sigma(
    mu: float,
    sigma: float,
    *,
    month: int = 3,
    day: int = 15,
) -> float:
    return float(
        site_home_win_prob_from_mu_sigma(
            float(mu),
            float(sigma),
            start_month=month,
            start_day=day,
            neutral_site=True,
            odds_mode="meta_small_v1",
        )
    )


def _build_slot_plan() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reserved = {
        ("East", 11): "ff1",
        ("West", 11): "ff2",
        ("South", 16): "ff3",
        ("Midwest", 16): "ff4",
    }
    direct_slots: list[dict[str, Any]] = []
    play_in_slots: list[dict[str, Any]] = []
    for seed in range(1, 17):
        for region in _seed_region_order(seed):
            game_id = reserved.get((region, seed))
            if game_id:
                play_in_slots.append(
                    {
                        "region": region,
                        "seed": seed,
                        "winner_to_slot": _main_bracket_slot(region, seed),
                        "game_id": game_id,
                    }
                )
            else:
                direct_slots.append(
                    {
                        "region": region,
                        "seed": seed,
                        "slot": _main_bracket_slot(region, seed),
                    }
                )
    return direct_slots, play_in_slots


def _load_rankings_fallback_input(season: int) -> dict[str, Any]:
    rankings = list(_load_rankings_rows(season).values())
    rankings.sort(key=lambda row: int(row["rank"]))
    if len(rankings) < 68:
        raise ValueError(f"Rankings fallback requires at least 68 teams, found {len(rankings)}")

    direct_slots, play_in_slots = _build_slot_plan()
    direct_teams = rankings[:60]
    play_in_teams = rankings[60:68]

    entries: list[dict[str, Any]] = []
    for slot, team in zip(direct_slots, direct_teams):
        entries.append(
            {
                "team_id": int(team["team_id"]),
                "team_name": str(team["team"]),
                "seed": int(slot["seed"]),
                "region": str(slot["region"]),
                "slot": str(slot["slot"]),
                "is_first_four": False,
                "first_four_game_id": None,
                "feeder_slot": None,
            }
        )

    eleven_play_in = play_in_teams[:4]
    sixteen_play_in = play_in_teams[4:]
    play_in_pairs = {
        "ff1": [eleven_play_in[0], eleven_play_in[3]],
        "ff2": [eleven_play_in[1], eleven_play_in[2]],
        "ff3": [sixteen_play_in[0], sixteen_play_in[3]],
        "ff4": [sixteen_play_in[1], sixteen_play_in[2]],
    }

    first_four_games: list[dict[str, Any]] = []
    for slot in play_in_slots:
        first_four_games.append(
            {
                "id": str(slot["game_id"]),
                "region": str(slot["region"]),
                "seed": int(slot["seed"]),
                "winner_to_slot": str(slot["winner_to_slot"]),
            }
        )
        for index, team in enumerate(play_in_pairs[str(slot["game_id"])], start=1):
            entries.append(
                {
                    "team_id": int(team["team_id"]),
                    "team_name": str(team["team"]),
                    "seed": int(slot["seed"]),
                    "region": str(slot["region"]),
                    "slot": f"{slot['game_id']}-team-{index}",
                    "is_first_four": True,
                    "first_four_game_id": str(slot["game_id"]),
                    "feeder_slot": str(slot["winner_to_slot"]),
                }
            )

    return {
        "season": season,
        "source": "rankings_fallback",
        "note": (
            "Development helper only. Replace with the official Selection Sunday "
            "field input before publishing a real NCAA bracket."
        ),
        "first_four_games": sorted(first_four_games, key=lambda game: game["id"]),
        "entries": entries,
    }


def _validate_canonical_field_input(payload: dict[str, Any], season: int) -> None:
    errors: list[str] = []

    payload_season = payload.get("season")
    if payload_season != season:
        errors.append(f"Canonical field season {payload_season} does not match requested season {season}")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Canonical field input must contain an entries array")
    if len(entries) != 68:
        errors.append(f"Expected exactly 68 field entries, found {len(entries)}")

    first_four_games = payload.get("first_four_games")
    if not isinstance(first_four_games, list):
        raise ValueError("Canonical field input must contain a first_four_games array")
    if len(first_four_games) != 4:
        errors.append(f"Expected 4 First Four games, found {len(first_four_games)}")

    game_by_id: dict[str, dict[str, Any]] = {}
    winner_slot_to_game: dict[str, str] = {}
    for game in first_four_games:
        game_id = str(game.get("id") or "")
        region = str(game.get("region") or "")
        seed = game.get("seed")
        winner_to_slot = str(game.get("winner_to_slot") or "")
        if not game_id:
            errors.append("First Four game id is required")
            continue
        if game_id in game_by_id:
            errors.append(f"Duplicate First Four game id {game_id}")
        if region not in REGIONS:
            errors.append(f"First Four game {game_id} has invalid region {region}")
        if not isinstance(seed, int) or not 1 <= seed <= 16:
            errors.append(f"First Four game {game_id} has invalid seed {seed}")
        expected_slot = _main_bracket_slot(region, seed) if region in REGIONS and isinstance(seed, int) else None
        if winner_to_slot != expected_slot:
            errors.append(
                f"First Four game {game_id} winner_to_slot must be {expected_slot}, found {winner_to_slot}"
            )
        if winner_to_slot in winner_slot_to_game:
            errors.append(
                f"First Four winner slot {winner_to_slot} is assigned to both "
                f"{winner_slot_to_game[winner_to_slot]} and {game_id}"
            )
        winner_slot_to_game[winner_to_slot] = game_id
        game_by_id[game_id] = game

    team_ids: set[int] = set()
    occupied_slots: set[str] = set()
    direct_slot_owners: dict[str, int] = {}
    play_in_entries_by_game: dict[str, list[dict[str, Any]]] = {}
    main_bracket_assignments: dict[tuple[str, int], str] = {}
    region_seeds: dict[str, set[int]] = {region: set() for region in REGIONS}
    direct_count = 0
    first_four_count = 0

    for index, entry in enumerate(entries, start=1):
        prefix = f"Entry {index}"
        try:
            team_id = int(entry["team_id"])
        except Exception:
            errors.append(f"{prefix} is missing a valid team_id")
            continue
        team_name = str(entry.get("team_name") or "").strip()
        seed = entry.get("seed")
        region = str(entry.get("region") or "")
        slot = str(entry.get("slot") or "")
        is_first_four = bool(entry.get("is_first_four"))
        first_four_game_id = entry.get("first_four_game_id")
        feeder_slot = entry.get("feeder_slot")

        if not team_name:
            errors.append(f"{prefix} team_id {team_id} is missing team_name")
        if team_id in team_ids:
            errors.append(f"Duplicate team_id {team_id}")
        team_ids.add(team_id)

        if region not in REGIONS:
            errors.append(f"{prefix} team_id {team_id} has invalid region {region}")
        if not isinstance(seed, int) or not 1 <= seed <= 16:
            errors.append(f"{prefix} team_id {team_id} has invalid seed {seed}")
        if not slot:
            errors.append(f"{prefix} team_id {team_id} is missing slot")
        elif slot in occupied_slots:
            errors.append(f"Duplicate occupied slot {slot}")
        occupied_slots.add(slot)

        if not is_first_four:
            direct_count += 1
            expected_slot = _main_bracket_slot(region, seed) if region in REGIONS and isinstance(seed, int) else None
            if slot != expected_slot:
                errors.append(f"{prefix} team_id {team_id} must occupy slot {expected_slot}, found {slot}")
            if first_four_game_id not in (None, ""):
                errors.append(f"{prefix} team_id {team_id} must not include first_four_game_id")
            if feeder_slot not in (None, ""):
                errors.append(f"{prefix} team_id {team_id} must not include feeder_slot")
            if region in REGIONS and isinstance(seed, int):
                key = (region, seed)
                if key in main_bracket_assignments:
                    errors.append(
                        f"Main bracket seed {region} {seed} is assigned twice "
                        f"({main_bracket_assignments[key]} and direct team {team_id})"
                    )
                main_bracket_assignments[key] = f"direct team {team_id}"
                region_seeds[region].add(seed)
                direct_slot_owners[slot] = team_id
            continue

        first_four_count += 1
        game_id = str(first_four_game_id or "")
        feeder_slot_value = str(feeder_slot or "")
        if not game_id:
            errors.append(f"{prefix} team_id {team_id} is marked First Four but missing first_four_game_id")
            continue
        if game_id not in game_by_id:
            errors.append(f"{prefix} team_id {team_id} references unknown First Four game {game_id}")
            continue
        game = game_by_id[game_id]
        if region != game["region"]:
            errors.append(
                f"{prefix} team_id {team_id} region {region} does not match First Four game {game_id} region {game['region']}"
            )
        if seed != game["seed"]:
            errors.append(
                f"{prefix} team_id {team_id} seed {seed} does not match First Four game {game_id} seed {game['seed']}"
            )
        if feeder_slot_value != game["winner_to_slot"]:
            errors.append(
                f"{prefix} team_id {team_id} feeder_slot must be {game['winner_to_slot']}, found {feeder_slot_value}"
            )
        if region in REGIONS and isinstance(seed, int):
            key = (region, seed)
            owner = main_bracket_assignments.get(key)
            game_label = f"First Four game {game_id}"
            if owner and owner != game_label:
                errors.append(f"Main bracket seed {region} {seed} is assigned twice ({owner} and {game_label})")
            main_bracket_assignments[key] = game_label
            region_seeds[region].add(seed)
        play_in_entries_by_game.setdefault(game_id, []).append(entry)

    if direct_count != 60:
        errors.append(f"Expected 60 direct bracket teams, found {direct_count}")
    if first_four_count != 8:
        errors.append(f"Expected 8 First Four participants, found {first_four_count}")
    if len(team_ids) != 68:
        errors.append(f"Expected 68 unique team_ids, found {len(team_ids)}")

    for region in REGIONS:
        seeds = sorted(region_seeds[region])
        if seeds != list(range(1, 17)):
            errors.append(f"Region {region} must occupy bracket seeds 1-16 exactly once; found {seeds}")

    for game_id, game in game_by_id.items():
        linked = play_in_entries_by_game.get(game_id, [])
        if len(linked) != 2:
            errors.append(f"First Four game {game_id} must have exactly 2 participant entries, found {len(linked)}")
        if game["winner_to_slot"] in direct_slot_owners:
            errors.append(
                f"First Four game {game_id} winner slot {game['winner_to_slot']} conflicts with direct team "
                f"{direct_slot_owners[game['winner_to_slot']]}"
            )

    if errors:
        raise ValueError("Canonical NCAA field input is invalid:\n- " + "\n- ".join(errors))


def _enrich_entry(
    entry: dict[str, Any],
    season: int,
    rankings_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    team_id = int(entry["team_id"])
    rankings_row = rankings_by_id.get(team_id)
    if not rankings_row:
        raise ValueError(
            f"Canonical field team_id {team_id} ({entry['team_name']}) is missing from rankings_{season}.json"
        )

    return {
        "team_id": team_id,
        "team": str(rankings_row["team"]),
        "rank": int(rankings_row["rank"]),
        "conference": str(rankings_row.get("conference") or ""),
        "record": str(rankings_row.get("record") or ""),
        "conf_record": str(rankings_row.get("conf_record") or ""),
        "adj_oe": float(rankings_row["adj_oe"]),
        "adj_de": float(rankings_row["adj_de"]),
        "adj_margin": float(rankings_row["adj_margin"]),
        "adj_tempo": float(rankings_row["adj_tempo"]),
        "model_index": None
        if rankings_row.get("model_index") is None
        else float(rankings_row["model_index"]),
        "adj_oe_rank": None
        if rankings_row.get("adj_oe_rank") is None
        else int(rankings_row["adj_oe_rank"]),
        "adj_de_rank": None
        if rankings_row.get("adj_de_rank") is None
        else int(rankings_row["adj_de_rank"]),
        "adj_margin_rank": None
        if rankings_row.get("adj_margin_rank") is None
        else int(rankings_row["adj_margin_rank"]),
        "adj_tempo_rank": None
        if rankings_row.get("adj_tempo_rank") is None
        else int(rankings_row["adj_tempo_rank"]),
        "ft_pct": None
        if rankings_row.get("ft_pct") is None
        else float(rankings_row["ft_pct"]),
        "three_p_pct": None
        if rankings_row.get("three_p_pct") is None
        else float(rankings_row["three_p_pct"]),
        "def_3p_pct": None
        if rankings_row.get("def_3p_pct") is None
        else float(rankings_row["def_3p_pct"]),
        "three_p_pct_rank": None
        if rankings_row.get("three_p_pct_rank") is None
        else int(rankings_row["three_p_pct_rank"]),
        "def_3p_pct_rank": None
        if rankings_row.get("def_3p_pct_rank") is None
        else int(rankings_row["def_3p_pct_rank"]),
        "ft_pct_rank": None
        if rankings_row.get("ft_pct_rank") is None
        else int(rankings_row["ft_pct_rank"]),
        "model_index_rank": None
        if rankings_row.get("model_index_rank") is None
        else int(rankings_row["model_index_rank"]),
    }


def _build_field_payload(
    canonical_input: dict[str, Any],
    season: int,
    rankings_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    regions: dict[str, list[dict[str, Any]]] = {region: [] for region in REGIONS}
    first_four_entries_by_game: dict[str, list[dict[str, Any]]] = {}

    for entry in canonical_input["entries"]:
        region = str(entry["region"])
        seed = int(entry["seed"])
        if entry["is_first_four"]:
            first_four_entries_by_game.setdefault(str(entry["first_four_game_id"]), []).append(entry)
            continue
        regions[region].append(
            {
                "seed": seed,
                "source": "team",
                **_enrich_entry(entry, season, rankings_by_id),
            }
        )

    first_four: list[dict[str, Any]] = []
    for game in sorted(canonical_input["first_four_games"], key=lambda row: row["id"]):
        game_id = str(game["id"])
        region = str(game["region"])
        seed = int(game["seed"])
        participants = sorted(first_four_entries_by_game[game_id], key=lambda row: row["slot"])
        first_four.append(
            {
                "id": game_id,
                "label": "First Four",
                "region": region,
                "seed": seed,
                "teams": [_enrich_entry(entry, season, rankings_by_id) for entry in participants],
            }
        )
        regions[region].append(
            {
                "seed": seed,
                "source": "play_in",
                "play_in_game_id": game_id,
            }
        )

    out_regions = []
    for region in REGIONS:
        entries = sorted(regions[region], key=lambda row: int(row["seed"]))
        out_regions.append({"name": region, "entries": entries})

    input_source = str(canonical_input.get("source") or "canonical_field_input")
    note_suffix = str(canonical_input.get("note") or "").strip()
    note = "Generated from canonical NCAA field input."
    if note_suffix:
        note = f"{note} {note_suffix}"

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "season": season,
        "source": input_source,
        "note": note,
        "regions": out_regions,
        "first_four": first_four,
    }


def _round_from_game_notes(note: object) -> tuple[str | None, str | None]:
    value = str(note or "").upper()
    if "FIRST FOUR" in value:
        return "first-four", "First Four"
    if "1ST ROUND" in value:
        return "round-of-64", "Round of 64"
    return None, None


def _preferred_ncaa_lines(season: int) -> pd.DataFrame:
    live_lines = select_preferred_lines(load_lines(season))
    research_lines = select_preferred_lines(load_research_lines(season))
    combined = pd.concat([live_lines, research_lines], ignore_index=True, sort=False)
    if combined.empty:
        return combined

    provider_rank = pd.Series(1, index=combined.index, dtype=int)
    provider_rank.loc[combined["provider"].fillna("").eq("Hard Rock Bet")] = 0
    provider_rank.loc[combined["provider"].fillna("").eq("consensus")] = 2
    provider_rank.loc[combined["book_spread"].isna()] = 3
    combined = combined.assign(_provider_rank=provider_rank)
    combined = combined.sort_values(
        ["gameId", "_provider_rank"],
        ascending=[True, True],
        kind="stable",
    )
    return combined.drop_duplicates("gameId", keep="first").drop(columns=["_provider_rank"])


def _load_opening_round_schedule_frame(season: int) -> pd.DataFrame:
    games_table = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if games_table.num_rows == 0:
        return pd.DataFrame()
    games = games_table.to_pandas()
    keep = [
        c
        for c in [
            "gameId",
            "homeTeamId",
            "awayTeamId",
            "homeTeam",
            "awayTeam",
            "startDate",
            "gameNotes",
            "tournament",
        ]
        if c in games.columns
    ]
    games = games[keep].drop_duplicates("gameId").copy()
    games = games[games["tournament"].eq("NCAA")].copy()
    if games.empty:
        return games

    round_info = games["gameNotes"].map(_round_from_game_notes)
    games["scheduled_round_id"] = round_info.map(lambda item: item[0])
    games["scheduled_round_label"] = round_info.map(lambda item: item[1])
    games = games[games["scheduled_round_id"].isin(["first-four", "round-of-64"])].copy()
    return games


def _load_opening_round_schedule_lookup(season: int) -> dict[str, dict[str, Any]]:
    games = _load_opening_round_schedule_frame(season)
    if games.empty:
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in games.iterrows():
        home_id = int(row["homeTeamId"])
        away_id = int(row["awayTeamId"])
        team1_id, team2_id = sorted([home_id, away_id])
        key = f"{team1_id}::{team2_id}"
        lookup[key] = {
            "scheduled_game_id": int(row["gameId"]),
            "scheduled_round_id": row["scheduled_round_id"],
            "scheduled_round_label": row["scheduled_round_label"],
            "start_time": row.get("startDate"),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_name": row.get("homeTeam"),
            "away_team_name": row.get("awayTeam"),
        }
    return lookup


def _load_opening_round_market_lookup(season: int) -> dict[str, dict[str, Any]]:
    games = _load_opening_round_schedule_frame(season)
    if games.empty:
        return {}

    lines = _preferred_ncaa_lines(season)
    if lines.empty:
        return {}

    merged = games.merge(
        lines[
            [
                c
                for c in [
                    "gameId",
                    "book_spread",
                    "home_moneyline",
                    "away_moneyline",
                    "provider",
                ]
                if c in lines.columns
            ]
        ],
        on="gameId",
        how="left",
    )
    merged["book_spread"] = pd.to_numeric(merged["book_spread"], errors="coerce")
    merged = merged.dropna(subset=["book_spread", "homeTeamId", "awayTeamId"]).copy()
    if merged.empty:
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in merged.iterrows():
        home_id = int(row["homeTeamId"])
        away_id = int(row["awayTeamId"])
        team1_id, team2_id = sorted([home_id, away_id])
        market_margin = -float(row["book_spread"]) if team1_id == home_id else float(row["book_spread"])
        key = f"{team1_id}::{team2_id}"
        lookup[key] = {
            "market_mu_team1_minus_team2": market_margin,
            "market_spread_home": float(row["book_spread"]),
            "market_home_team_id": home_id,
            "market_away_team_id": away_id,
            "market_home_moneyline": None if pd.isna(row.get("home_moneyline")) else float(row["home_moneyline"]),
            "market_away_moneyline": None if pd.isna(row.get("away_moneyline")) else float(row["away_moneyline"]),
            "market_line_source": row.get("provider"),
        }
    return lookup


def _predict_pairwise_projection(
    team_a: pd.Series,
    team_b: pd.Series,
    feature_order: list[str],
    scaler,
    tree_model,
    sigma_model,
    sigma_param: str,
    month: int,
    day: int,
) -> tuple[float, float, float]:
    mu, win_prob_a = _predict_pairwise_probability(
        team_a,
        team_b,
        feature_order,
        scaler,
        tree_model,
        sigma_model,
        sigma_param,
        month,
        day,
    )

    row_ab, row_ba = _build_synthetic_rows(team_a, team_b, feature_order, scaler)
    X_ab = _fill_nan_with_scaler_means(row_ab, scaler)
    X_ba = _fill_nan_with_scaler_means(row_ba, scaler)
    mu_ab = float(tree_model.predict(X_ab.astype(np.float32))[0])
    mu_ba = float(tree_model.predict(X_ba.astype(np.float32))[0])

    X_ab_scaled = scaler.transform(X_ab)
    X_ba_scaled = scaler.transform(X_ba)
    X_ab_tensor = torch.tensor(X_ab_scaled, dtype=torch.float32)
    X_ba_tensor = torch.tensor(X_ba_scaled, dtype=torch.float32)
    with torch.no_grad():
        _, log_sigma_ab = sigma_model(X_ab_tensor)
        _, log_sigma_ba = sigma_model(X_ba_tensor)
        if sigma_param == "exp":
            sigma_ab = np.exp(log_sigma_ab.numpy())[0]
            sigma_ba = np.exp(log_sigma_ba.numpy())[0]
        else:
            sigma_ab = (torch.nn.functional.softplus(log_sigma_ab) + 1e-3).numpy()[0]
            sigma_ba = (torch.nn.functional.softplus(log_sigma_ba) + 1e-3).numpy()[0]
    sigma_var = 0.5 * (sigma_ab**2 + sigma_ba**2) + ((mu_ab + mu_ba) ** 2) / 4.0
    sigma = float(max(math.sqrt(max(sigma_var, 0.25)), 0.5))
    return float(mu), sigma, float(win_prob_a)


def _validate_field_payload(field: dict[str, Any]) -> None:
    if len(field["regions"]) != 4:
        raise ValueError(f"Expected 4 regions, found {len(field['regions'])}")
    if len(field["first_four"]) != 4:
        raise ValueError(f"Expected 4 First Four games, found {len(field['first_four'])}")

    seen_ids: set[int] = set()
    play_in_ids: set[str] = set()
    region_play_in_refs: set[str] = set()

    for region in field["regions"]:
        seeds = sorted(entry["seed"] for entry in region["entries"])
        if seeds != list(range(1, 17)):
            raise ValueError(f"Region {region['name']} must contain seeds 1-16 exactly once")
        for entry in region["entries"]:
            if entry["source"] == "team":
                team_id = int(entry["team_id"])
                if team_id in seen_ids:
                    raise ValueError(f"Duplicate team id {team_id} in field payload")
                seen_ids.add(team_id)
            else:
                region_play_in_refs.add(str(entry["play_in_game_id"]))

    for game in field["first_four"]:
        play_in_ids.add(str(game["id"]))
        if len(game["teams"]) != 2:
            raise ValueError(f"First Four game {game['id']} must contain 2 teams")
        for team in game["teams"]:
            team_id = int(team["team_id"])
            if team_id in seen_ids:
                raise ValueError(f"Duplicate play-in team id {team_id} in field payload")
            seen_ids.add(team_id)

    if play_in_ids != region_play_in_refs:
        raise ValueError("Generated field payload has mismatched First Four references")
    if len(seen_ids) != 68:
        raise ValueError(f"Expected 68 field teams, found {len(seen_ids)}")


def _build_matchup_payload(
    field: dict[str, Any],
    season: int,
    *,
    bracket_model_variant: str,
    team_ab_efficiency_source: str,
    team_ab_gold_ratings_table: str | None,
) -> dict[str, Any]:
    print("Collecting selected NCAA teams", flush=True)
    team_rows = []
    for region in field["regions"]:
        for entry in region["entries"]:
            if entry["source"] == "team":
                team_rows.append(entry)
    for play_in in field["first_four"]:
        team_rows.extend(play_in["teams"])

    unique_rows: dict[int, dict[str, Any]] = {}
    for row in team_rows:
        unique_rows[int(row["team_id"])] = row

    selected_ids = sorted(unique_rows)
    selected = pd.DataFrame.from_records(list(unique_rows.values()))
    print("Loading latest ratings", flush=True)

    def _merge_selected_with_ratings(ratings_frame: pd.DataFrame) -> pd.DataFrame:
        merged = selected.merge(
            ratings_frame,
            on="team_id",
            how="left",
            suffixes=("_field", ""),
        )
        missing = merged.loc[merged["adj_oe_model"].isna(), "team"].tolist()
        if missing:
            raise ValueError(f"Missing ratings rows for selected NCAA teams: {missing[:10]}")
        for model_col, legacy_col in [
            ("adj_oe_model", "adj_oe"),
            ("adj_de_model", "adj_de"),
            ("adj_tempo_model", "adj_tempo"),
            ("barthag_model", "barthag"),
            ("sos_oe_model", "sos_oe"),
            ("sos_de_model", "sos_de"),
        ]:
            if model_col in merged.columns:
                merged[legacy_col] = merged[model_col]
        return merged

    legacy_ratings, legacy_conf_strength_lookup, legacy_ratings_source = _build_bracket_ratings_frame(
        season,
        efficiency_source="gold",
        gold_table_name=config.PRODUCTION_GOLD_RATINGS_TABLE,
    )
    legacy_merged = _merge_selected_with_ratings(legacy_ratings)

    print("Loading model artifacts", flush=True)
    scaler = load_scaler()
    tree_model, feature_order, _ = load_tree_regressor()
    sigma_model, _, sigma_feature_order, sigma_param = load_regressor()
    if sigma_feature_order != feature_order:
        raise ValueError("Tree and sigma feature orders do not match")
    team_ab_loaded: tuple[object, list[str], str, dict] | None = None
    try:
        team_ab_loaded = load_mu_regressor(TEAM_AB_ELITE_TAIL_ROUND64_V1)
    except FileNotFoundError:
        if bracket_model_variant == BRACKET_MODEL_VARIANT_TEAM_AB:
            raise

    team_lookup_legacy = {int(row["team_id"]): row for _, row in legacy_merged.iterrows()}
    team_lookup_team_ab = team_lookup_legacy
    team_lookup_team_ab_internal = team_lookup_legacy
    conf_strength_lookup = legacy_conf_strength_lookup
    team_ab_ratings_source = legacy_ratings_source
    team_ab_internal_conf_strength_lookup = legacy_conf_strength_lookup
    team_ab_internal_ratings_source = legacy_ratings_source
    internal_gold_table_name = team_ab_gold_ratings_table or config.BRACKET_TEAM_AB_GOLD_RATINGS_TABLE
    if team_ab_loaded is not None and (
        team_ab_efficiency_source != "gold"
        or (team_ab_gold_ratings_table or config.PRODUCTION_GOLD_RATINGS_TABLE) != config.PRODUCTION_GOLD_RATINGS_TABLE
    ):
        team_ab_ratings, conf_strength_lookup, team_ab_ratings_source = _build_bracket_ratings_frame(
            season,
            efficiency_source=team_ab_efficiency_source,
            gold_table_name=team_ab_gold_ratings_table,
        )
        team_ab_merged = _merge_selected_with_ratings(team_ab_ratings)
        team_lookup_team_ab = {int(row["team_id"]): row for _, row in team_ab_merged.iterrows()}
    if team_ab_loaded is not None:
        team_ab_internal_ratings, team_ab_internal_conf_strength_lookup, team_ab_internal_ratings_source = (
            _build_bracket_ratings_frame(
                season,
                efficiency_source="gold",
                gold_table_name=internal_gold_table_name,
            )
        )
        team_ab_internal_merged = _merge_selected_with_ratings(team_ab_internal_ratings)
        team_lookup_team_ab_internal = {
            int(row["team_id"]): row for _, row in team_ab_internal_merged.iterrows()
        }
    team_bracket_context, team_round_times = _load_bracket_schedule_context(field, season)
    opening_round_schedule = _load_opening_round_schedule_lookup(season)
    opening_round_lines = _load_opening_round_market_lookup(season)
    scheduled_team_ab_lookup: dict[int, pd.Series] = {}
    scheduled_team_ab_internal_lookup: dict[int, pd.Series] = {}
    team_state_lookup_team_ab: dict[int, dict[str, Any]] = {}
    team_state_lookup_team_ab_internal: dict[int, dict[str, Any]] = {}
    if team_ab_loaded is not None and opening_round_schedule:
        scheduled_team_ab_lookup = _load_scheduled_feature_lookup(
            season,
            opening_round_schedule,
            efficiency_source=team_ab_efficiency_source,
            gold_table_name=team_ab_gold_ratings_table if team_ab_efficiency_source == "gold" else None,
        )
        scheduled_team_ab_internal_lookup = _load_scheduled_feature_lookup(
            season,
            opening_round_schedule,
            efficiency_source="gold",
            gold_table_name=internal_gold_table_name,
        )
        team_state_lookup_team_ab = _build_team_state_lookup(scheduled_team_ab_lookup)
        team_state_lookup_team_ab_internal = _build_team_state_lookup(scheduled_team_ab_internal_lookup)

    predictions: dict[str, Any] = {}
    total_pairs = len(selected_ids) * (len(selected_ids) - 1) // 2
    for pair_index, (team_a_id, team_b_id) in enumerate(combinations(selected_ids, 2), start=1):
        team_a_legacy = team_lookup_legacy[team_a_id]
        team_b_legacy = team_lookup_legacy[team_b_id]
        legacy_mu, sigma, _legacy_win_prob_a = _predict_pairwise_projection(
            team_a_legacy,
            team_b_legacy,
            feature_order,
            scaler,
            tree_model,
            sigma_model,
            sigma_param,
            3,
            15,
        )
        canonical_key = f"{team_a_id}::{team_b_id}"
        schedule_info = opening_round_schedule.get(canonical_key)
        line_info = opening_round_lines.get(canonical_key)
        matchup_context = _matchup_round_context(
            team_a_id,
            team_b_id,
            team_context=team_bracket_context,
            team_round_times=team_round_times,
        )
        round_id = matchup_context["round_id"]
        round_label = matchup_context["round_label"]
        start_time = (
            schedule_info["start_time"]
            if schedule_info is not None
            else matchup_context["start_time"]
        )
        team_ab_mu = None
        team_ab_internal_mu = None
        active_mu_source_mode = None
        active_mu_source_detail = None
        active_mu_base_simple_margin = None
        team_ab_internal_mu_source_mode = None
        team_ab_internal_mu_source_detail = None
        team_ab_internal_mu_base_simple_margin = None
        if team_ab_loaded is not None:
            team_ab_regressor, team_ab_feature_order, team_ab_model_type, team_ab_meta = team_ab_loaded
            scheduled_game_id = None if schedule_info is None else schedule_info.get("scheduled_game_id")
            scheduled_team_ab_row = (
                None
                if scheduled_game_id is None
                else scheduled_team_ab_lookup.get(int(scheduled_game_id))
            )
            scheduled_team_ab_internal_row = (
                None
                if scheduled_game_id is None
                else scheduled_team_ab_internal_lookup.get(int(scheduled_game_id))
            )
            if scheduled_team_ab_row is not None:
                team_ab_mu = _predict_team_ab_scheduled_margin(
                    scheduled_team_ab_row,
                    team_a_id=team_a_id,
                    team_b_id=team_b_id,
                    mu_regressor=team_ab_regressor,
                    mu_feature_order=team_ab_feature_order,
                    mu_model_type=team_ab_model_type,
                    mu_impute_means=team_ab_meta.get("impute_means"),
                )
                active_mu_source_mode = "scheduled_team_ab_model"
                active_mu_source_detail = bracket_model_variant
            else:
                team_a_team_ab = team_lookup_team_ab[team_a_id]
                team_b_team_ab = team_lookup_team_ab[team_b_id]
                team_ab_mu, active_mu_base_simple_margin, active_mu_source_detail = synthetic_fallback_margin(
                    team_a_team_ab,
                    team_b_team_ab,
                    ratings_source=team_ab_efficiency_source,
                    round_label=round_label,
                )
                active_mu_source_mode = "synthetic_ratings_map"
            if scheduled_team_ab_internal_row is not None:
                team_ab_internal_mu = _predict_team_ab_scheduled_margin(
                    scheduled_team_ab_internal_row,
                    team_a_id=team_a_id,
                    team_b_id=team_b_id,
                    mu_regressor=team_ab_regressor,
                    mu_feature_order=team_ab_feature_order,
                    mu_model_type=team_ab_model_type,
                    mu_impute_means=team_ab_meta.get("impute_means"),
                )
                team_ab_internal_mu_source_mode = "scheduled_team_ab_model"
                team_ab_internal_mu_source_detail = internal_gold_table_name
            else:
                team_a_team_ab_internal = team_lookup_team_ab_internal[team_a_id]
                team_b_team_ab_internal = team_lookup_team_ab_internal[team_b_id]
                team_ab_internal_mu, team_ab_internal_mu_base_simple_margin, team_ab_internal_mu_source_detail = synthetic_fallback_margin(
                    team_a_team_ab_internal,
                    team_b_team_ab_internal,
                    ratings_source="gold",
                    round_label=round_label,
                )
                team_ab_internal_mu_source_mode = "synthetic_ratings_map"

        legacy_win_prob_a = _site_probability_from_mu_sigma(float(legacy_mu), float(sigma))
        team_ab_win_prob_a = (
            None if team_ab_mu is None else _site_probability_from_mu_sigma(float(team_ab_mu), float(sigma))
        )
        team_ab_internal_win_prob_a = (
            None
            if team_ab_internal_mu is None
            else _site_probability_from_mu_sigma(float(team_ab_internal_mu), float(sigma))
        )
        active_mu = (
            float(team_ab_mu)
            if bracket_model_variant == BRACKET_MODEL_VARIANT_TEAM_AB and team_ab_mu is not None
            else float(legacy_mu)
        )
        win_prob_a = (
            float(team_ab_win_prob_a)
            if bracket_model_variant == BRACKET_MODEL_VARIANT_TEAM_AB and team_ab_win_prob_a is not None
            else float(legacy_win_prob_a)
        )
        display_mu = float(active_mu)
        if config.NCAA_TOURNAMENT_MARKET_BLEND_ENABLED and line_info is not None:
            display_mu = market_blended_display_margin(
                float(active_mu),
                float(line_info["market_mu_team1_minus_team2"]),
            )
        model_mu_home = None
        display_model_mu_home = None
        edge_home_points = None
        display_edge_home_points = None
        legacy_model_mu_home = None
        team_ab_model_mu_home = None
        team_ab_internal_model_mu_home = None
        pick_side = None
        pick_cover_prob = None
        pick_prob_edge = None
        pick_fair_odds = None
        if schedule_info is not None:
            home_team_id = int(schedule_info["home_team_id"])
            model_mu_home = float(active_mu) if home_team_id == team_a_id else -float(active_mu)
            display_model_mu_home = float(display_mu) if home_team_id == team_a_id else -float(display_mu)
            legacy_model_mu_home = float(legacy_mu) if home_team_id == team_a_id else -float(legacy_mu)
            if team_ab_mu is not None:
                team_ab_model_mu_home = float(team_ab_mu) if home_team_id == team_a_id else -float(team_ab_mu)
            if team_ab_internal_mu is not None:
                team_ab_internal_model_mu_home = (
                    float(team_ab_internal_mu) if home_team_id == team_a_id else -float(team_ab_internal_mu)
                )
        if line_info is not None and schedule_info is not None:
            book_spread = float(line_info["market_spread_home"])
            edge_home_points = model_mu_home + book_spread
            display_edge_home_points = display_model_mu_home + book_spread
            sigma_safe = max(float(sigma), 0.5)
            edge_z = edge_home_points / sigma_safe
            home_cover_prob = float(normal_cdf(edge_z))
            away_cover_prob = 1.0 - home_cover_prob
            pick_side = "HOME" if edge_home_points >= 0 else "AWAY"
            pick_cover_prob = home_cover_prob if pick_side == "HOME" else away_cover_prob
            pick_breakeven = float(american_to_breakeven(np.array([-110.0]))[0])
            pick_prob_edge = pick_cover_prob - pick_breakeven
            pick_fair_odds = float(prob_to_american(np.array([pick_cover_prob]))[0])
        predictions[f"{team_a_id}::{team_b_id}"] = {
            "team1_id": int(team_a_id),
            "team1_name": str(team_a_legacy["team"]),
            "team2_id": int(team_b_id),
            "team2_name": str(team_b_legacy["team"]),
            "matchup_model_variant_active": bracket_model_variant,
            "mu_team1_minus_team2": float(active_mu),
            "mu_team1_minus_team2_legacy_synthetic": float(legacy_mu),
            "mu_team1_minus_team2_team_ab_elite_tail_round64_v1": None if team_ab_mu is None else float(team_ab_mu),
            "mu_team1_minus_team2_team_ab_internal": (
                None if team_ab_internal_mu is None else float(team_ab_internal_mu)
            ),
            "active_mu_source_mode": active_mu_source_mode,
            "active_mu_source_detail": active_mu_source_detail,
            "active_mu_base_simple_margin": (
                None if active_mu_base_simple_margin is None else float(active_mu_base_simple_margin)
            ),
            "team_ab_internal_mu_source_mode": team_ab_internal_mu_source_mode,
            "team_ab_internal_mu_source_detail": team_ab_internal_mu_source_detail,
            "team_ab_internal_mu_base_simple_margin": (
                None
                if team_ab_internal_mu_base_simple_margin is None
                else float(team_ab_internal_mu_base_simple_margin)
            ),
            "display_mu_team1_minus_team2": display_mu,
            "win_prob_team1": float(win_prob_a),
            "win_prob_team1_legacy_synthetic": float(legacy_win_prob_a),
            "win_prob_team1_team_ab_elite_tail_round64_v1": (
                None if team_ab_win_prob_a is None else float(team_ab_win_prob_a)
            ),
            "win_prob_team1_team_ab_internal": (
                None if team_ab_internal_win_prob_a is None else float(team_ab_internal_win_prob_a)
            ),
            "pred_sigma": float(sigma),
            "scheduled_game_id": None if schedule_info is None else schedule_info["scheduled_game_id"],
            "scheduled_round_id": round_id,
            "scheduled_round_label": round_label,
            "start_time": start_time,
            "home_team_id": None if schedule_info is None else schedule_info["home_team_id"],
            "away_team_id": None if schedule_info is None else schedule_info["away_team_id"],
            "home_team_name": None if schedule_info is None else schedule_info["home_team_name"],
            "away_team_name": None if schedule_info is None else schedule_info["away_team_name"],
            "model_mu_home": model_mu_home,
            "model_mu_home_legacy_synthetic": legacy_model_mu_home,
            "model_mu_home_team_ab_elite_tail_round64_v1": team_ab_model_mu_home,
            "model_mu_home_team_ab_internal": team_ab_internal_model_mu_home,
            "display_model_mu_home": display_model_mu_home,
            "edge_home_points": edge_home_points,
            "display_edge_home_points": display_edge_home_points,
            "pick_side": pick_side,
            "pick_cover_prob": pick_cover_prob,
            "pick_prob_edge": pick_prob_edge,
            "pick_fair_odds": pick_fair_odds,
            "market_mu_team1_minus_team2": None if line_info is None else line_info["market_mu_team1_minus_team2"],
            "market_spread_home": None if line_info is None else line_info["market_spread_home"],
            "market_home_team_id": None if line_info is None else line_info["market_home_team_id"],
            "market_away_team_id": None if line_info is None else line_info["market_away_team_id"],
            "market_home_moneyline": None if line_info is None else line_info["market_home_moneyline"],
            "market_away_moneyline": None if line_info is None else line_info["market_away_moneyline"],
            "market_line_source": None if line_info is None else line_info["market_line_source"],
        }
        if pair_index % 250 == 0 or pair_index == total_pairs:
            print(f"Computed {pair_index}/{total_pairs} matchup predictions", flush=True)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "season": season,
        "neutral_site": True,
        "source": "production_matchup_model",
        "matchup_model_variant_active": bracket_model_variant,
        "team_ab_efficiency_source_active": team_ab_efficiency_source,
        "team_ab_gold_ratings_table_active": team_ab_gold_ratings_table,
        "legacy_ratings_source": legacy_ratings_source,
        "team_ab_ratings_source": team_ab_ratings_source,
        "team_ab_internal_ratings_source_compare": team_ab_internal_ratings_source,
        "team_ab_internal_gold_ratings_table_compare": internal_gold_table_name,
        "matchup_model_variants_available": sorted(
            [
                BRACKET_MODEL_VARIANT_LEGACY_SYNTHETIC,
                *([BRACKET_MODEL_VARIANT_TEAM_AB] if team_ab_loaded is not None else []),
            ]
        ),
        "synthetic_fallback_active": (
            "source-aware direct ratings mapping for unscheduled NCAA matchups"
            if team_ab_loaded is not None
            else None
        ),
        "note": (
            "Neutral-site pairwise predictions generated from the NCAA bracket-builder matchup cache. "
            "The legacy synthetic tree spread, the active Team A/B tournament-engine spread, and the "
            "Team A/B internal-efficiency comparison spread are cached side by side when available; "
            "active spread selection follows the configured bracket matchup "
            "model variant. Scheduled NCAA games use the active Team A/B model on real feature rows; "
            "unscheduled NCAA matchups fall back to a source-aware direct ratings map. "
            "Sigma and site win-probability logic remain on the legacy bracket uncertainty path. "
            f"Legacy ratings source: {legacy_ratings_source}. "
            f"Team A/B ratings source: {team_ab_ratings_source}. "
            f"Team A/B internal comparison source: {team_ab_internal_ratings_source}."
        ),
        "predictions": predictions,
    }


def _validate_matchup_payload(field: dict[str, Any], payload: dict[str, Any]) -> None:
    team_ids = set()
    for region in field["regions"]:
        for entry in region["entries"]:
            if entry["source"] == "team":
                team_ids.add(int(entry["team_id"]))
    for game in field["first_four"]:
        for team in game["teams"]:
            team_ids.add(int(team["team_id"]))

    expected = len(team_ids) * (len(team_ids) - 1) // 2
    if len(payload["predictions"]) != expected:
        raise ValueError(f"Expected {expected} matchup predictions, found {len(payload['predictions'])}")

    for key, entry in payload["predictions"].items():
        team_a_id, team_b_id = (int(part) for part in key.split("::"))
        if team_a_id >= team_b_id:
            raise ValueError(f"Non-canonical matchup key {key}")
        if entry["team1_id"] != team_a_id or entry["team2_id"] != team_b_id:
            raise ValueError(f"Mismatch between matchup key {key} and cached team ids")
        if team_a_id not in team_ids or team_b_id not in team_ids:
            raise ValueError(f"Matchup key {key} references team outside field")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument(
        "--bracket-model-variant",
        type=str,
        default=None,
        help=(
            "Bracket matchup mean-model path. Defaults to HOOPS_BRACKET_MATCHUP_MODEL_VARIANT "
            f"({config.BRACKET_MATCHUP_MODEL_VARIANT})."
        ),
    )
    parser.add_argument(
        "--field-input",
        type=Path,
        default=None,
        help="Canonical field input JSON path. Defaults to site/public/data/ncaa_field_input_<season>.json.",
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=None,
        help="Generated bracket-builder JSON output path.",
    )
    parser.add_argument(
        "--matchups-output",
        type=Path,
        default=None,
        help="Generated matchup cache JSON output path.",
    )
    parser.add_argument(
        "--use-rankings-fallback",
        action="store_true",
        help="Dev helper only. Build a temporary field from rankings instead of a canonical field input file.",
    )
    parser.add_argument(
        "--team-ab-efficiency-source",
        type=str,
        default=None,
        help=(
            "Efficiency source for the Team A/B bracket scorer. Defaults to "
            "HOOPS_BRACKET_TEAM_AB_EFFICIENCY_SOURCE "
            f"({config.BRACKET_TEAM_AB_EFFICIENCY_SOURCE})."
        ),
    )
    parser.add_argument(
        "--team-ab-gold-ratings-table",
        type=str,
        default=None,
        help=(
            "Explicit gold ratings table for the Team A/B bracket scorer when "
            'the source is "gold". Defaults to '
            "HOOPS_BRACKET_TEAM_AB_GOLD_RATINGS_TABLE "
            f"({config.BRACKET_TEAM_AB_GOLD_RATINGS_TABLE})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    season = int(args.season)
    bracket_model_variant = _normalize_bracket_model_variant(
        args.bracket_model_variant or config.BRACKET_MATCHUP_MODEL_VARIANT
    )
    team_ab_efficiency_source = _normalize_efficiency_source(
        args.team_ab_efficiency_source or config.BRACKET_TEAM_AB_EFFICIENCY_SOURCE,
        env_name="Team A/B bracket efficiency source",
    )
    team_ab_gold_ratings_table = (
        (args.team_ab_gold_ratings_table or config.BRACKET_TEAM_AB_GOLD_RATINGS_TABLE).strip()
        if team_ab_efficiency_source == "gold"
        else None
    )
    field_input_path = args.field_input or _season_path(FIELD_INPUT_TEMPLATE, season)
    field_output_path = args.field_output or _season_path(FIELD_OUTPUT_TEMPLATE, season)
    matchups_output_path = args.matchups_output or _season_path(MATCHUPS_OUTPUT_TEMPLATE, season)

    if args.use_rankings_fallback:
        print("Building canonical field input from rankings fallback", flush=True)
        canonical_input = _load_rankings_fallback_input(season)
    else:
        if not field_input_path.exists():
            raise FileNotFoundError(
                f"Canonical field input not found: {field_input_path}. "
                "Provide the official field file or rerun with --use-rankings-fallback."
            )
        print(f"Loading canonical field input from {field_input_path.name}", flush=True)
        canonical_input = _read_json(field_input_path)

    _validate_canonical_field_input(canonical_input, season)
    rankings_by_id = _load_rankings_rows(season)

    print("Building field payload", flush=True)
    field = _build_field_payload(canonical_input, season, rankings_by_id)
    _validate_field_payload(field)

    print("Building matchup payload", flush=True)
    matchup_payload = _build_matchup_payload(
        field,
        season,
        bracket_model_variant=bracket_model_variant,
        team_ab_efficiency_source=team_ab_efficiency_source,
        team_ab_gold_ratings_table=team_ab_gold_ratings_table,
    )
    _validate_matchup_payload(field, matchup_payload)

    _write_json(field_output_path, field)
    _write_json(matchups_output_path, matchup_payload)
    print(f"Wrote {field_output_path.name} and {matchups_output_path.name}", flush=True)


if __name__ == "__main__":
    main()

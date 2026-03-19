"""Research-only rotation continuity / availability features."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config, s3_reader

ROTATION_WINDOW_GAMES = 5
ROTATION_CORE_SIZE = 8
ROTATION_STARTER_SIZE = 5
MIN_PRIOR_GAMES = 3

ROTATION_TEAM_FEATURES = [
    "core_minutes_return_rate_5",
    "rotation_overlap_5",
    "missing_core_minutes_share_5",
    "rotation_volatility_5",
    "starter_overlap_5",
]

ROTATION_FEATURE_COLUMNS = [
    f"{side}_{name}"
    for side in ("home", "away")
    for name in ROTATION_TEAM_FEATURES
]

AVAILABILITY_SHOCK_TEAM_FEATURES = [
    "missing_top1_minutes_last_game",
    "missing_top2_minutes_last_game",
    "missing_top3_minutes_last_game",
    "top1_minutes_share_change_1",
    "top3_minutes_share_change_1",
    "likely_starter_missing_flag",
]

AVAILABILITY_SHOCK_FEATURE_COLUMNS = [
    f"{side}_{name}"
    for side in ("home", "away")
    for name in AVAILABILITY_SHOCK_TEAM_FEATURES
]


@dataclass(frozen=True)
class SpineAuditSummary:
    season: int
    team_rows: int
    unique_game_team_rows: int
    duplicate_game_team_rows: int
    parse_success_rows: int
    parse_success_rate: float
    player_rows: int
    avg_players_per_team_game: float
    minutes_nonnull_rate: float
    starter_nonnull_rate: float


def parse_players_payload(payload: Any) -> list[dict[str, Any]] | None:
    """Parse mixed-format player payloads from fct_game_players."""
    if payload is None or (isinstance(payload, float) and pd.isna(payload)):
        return []
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if not isinstance(payload, str):
        return None

    for parser in (ast.literal_eval, json.loads):
        try:
            obj = parser(payload)
        except Exception:
            continue
        if isinstance(obj, list):
            return [p for p in obj if isinstance(p, dict)]
    return None


def _row_to_flat_records(row: pd.Series, players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for player in players:
        player_id = player.get("athleteId")
        if player_id is None:
            continue
        minutes = pd.to_numeric(player.get("minutes"), errors="coerce")
        starter = player.get("starter")
        starter_flag = bool(starter) if starter is not None else None
        records.append(
            {
                "season": int(row["season"]),
                "gameId": int(row["gameId"]),
                "teamId": int(row["teamId"]),
                "team": row.get("team"),
                "opponentId": row.get("opponentId"),
                "opponent": row.get("opponent"),
                "isHome": row.get("isHome"),
                "startDate": row.get("startDate"),
                "playerId": int(player_id),
                "playerSourceId": player.get("athleteSourceId"),
                "playerName": player.get("name"),
                "minutes": None if pd.isna(minutes) else float(minutes),
                "starter": starter_flag,
                "appeared": 1,
            }
        )
    return records


def build_team_game_player_participation_v1(
    seasons: list[int],
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten fct_game_players into one row per (gameId, teamId, playerId)."""
    flat_records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for season in seasons:
        table = s3_reader.read_silver_table("fct_game_players", season=season)
        frame = table.to_pandas()
        if frame.empty or "players" not in frame.columns:
            audit_rows.append(
                SpineAuditSummary(
                    season=season,
                    team_rows=len(frame),
                    unique_game_team_rows=0,
                    duplicate_game_team_rows=0,
                    parse_success_rows=0,
                    parse_success_rate=0.0,
                    player_rows=0,
                    avg_players_per_team_game=0.0,
                    minutes_nonnull_rate=0.0,
                    starter_nonnull_rate=0.0,
                ).__dict__
            )
            continue

        frame = frame.drop_duplicates(subset=["gameId", "teamId"], keep="last").copy()
        parse_success = 0
        player_rows = 0
        minutes_nonnull = 0
        starter_nonnull = 0

        for _, row in frame.iterrows():
            players = parse_players_payload(row.get("players"))
            if players is None:
                continue
            parse_success += 1
            player_rows += len(players)
            for player in players:
                if player.get("minutes") is not None:
                    minutes_nonnull += 1
                if player.get("starter") is not None:
                    starter_nonnull += 1
            flat_records.extend(_row_to_flat_records(row, players))

        audit_rows.append(
            SpineAuditSummary(
                season=season,
                team_rows=int(table.num_rows),
                unique_game_team_rows=int(len(frame)),
                duplicate_game_team_rows=int(table.num_rows - len(frame)),
                parse_success_rows=parse_success,
                parse_success_rate=round(parse_success / max(len(frame), 1), 4),
                player_rows=player_rows,
                avg_players_per_team_game=round(player_rows / max(parse_success, 1), 2),
                minutes_nonnull_rate=round(minutes_nonnull / max(player_rows, 1), 4),
                starter_nonnull_rate=round(starter_nonnull / max(player_rows, 1), 4),
            ).__dict__
        )

    flat_df = pd.DataFrame(flat_records)
    audit_df = pd.DataFrame(audit_rows).sort_values("season").reset_index(drop=True)

    if not flat_df.empty:
        flat_df["startDate"] = pd.to_datetime(flat_df["startDate"], errors="coerce", utc=True)
        flat_df = flat_df.sort_values(["season", "teamId", "startDate", "gameId", "playerId"]).reset_index(drop=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        flat_df.to_parquet(output_dir / "team_game_player_participation_v1.parquet", index=False)
        audit_df.to_csv(output_dir / "spine_audit_summary.csv", index=False)

    return flat_df, audit_df


def spine_is_usable(audit_df: pd.DataFrame) -> tuple[bool, str]:
    """Conservative usability gate for the player spine."""
    if audit_df.empty:
        return False, "no audit rows generated"
    if (audit_df["parse_success_rate"] < 0.95).any():
        bad = audit_df.loc[audit_df["parse_success_rate"] < 0.95, "season"].tolist()
        return False, f"parse success below 0.95 in seasons {bad}"
    if (audit_df["avg_players_per_team_game"] < 7.0).any():
        bad = audit_df.loc[audit_df["avg_players_per_team_game"] < 7.0, "season"].tolist()
        return False, f"too few players per team-game in seasons {bad}"
    if (audit_df["minutes_nonnull_rate"] < 0.9).any():
        bad = audit_df.loc[audit_df["minutes_nonnull_rate"] < 0.9, "season"].tolist()
        return False, f"minutes coverage below 0.9 in seasons {bad}"
    return True, "usable"


def _top_n_players_by_minutes(minutes_by_player: dict[int, float], n: int) -> list[int]:
    ordered = sorted(minutes_by_player.items(), key=lambda kv: (-kv[1], kv[0]))
    return [player_id for player_id, _ in ordered[:n]]


def _jaccard_churn(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return np.nan
    return 1.0 - (len(left & right) / len(union))


def build_rotation_availability_team_features(
    flat_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Build team-level rotation continuity features using prior games only."""
    if flat_df.empty:
        return pd.DataFrame(columns=["season", "gameId", "teamId", *ROTATION_TEAM_FEATURES])

    grouped_records: list[dict[str, Any]] = []
    grouped = flat_df.groupby(["season", "gameId", "teamId"], sort=False)
    for (season, game_id, team_id), group in grouped:
        group = group.sort_values(["minutes", "playerId"], ascending=[False, True])
        minutes_by_player = {
            int(row.playerId): float(0.0 if pd.isna(row.minutes) else row.minutes)
            for row in group.itertuples(index=False)
        }
        top_players = set(_top_n_players_by_minutes(minutes_by_player, ROTATION_CORE_SIZE))
        starter_set = set(
            int(row.playerId)
            for row in group.itertuples(index=False)
            if row.starter is True
        )
        grouped_records.append(
            {
                "season": int(season),
                "gameId": int(game_id),
                "teamId": int(team_id),
                "startDate": group["startDate"].iloc[0],
                "minutes_by_player": minutes_by_player,
                "present_players": set(minutes_by_player.keys()),
                "top_rotation_players": top_players,
                "starter_set": starter_set,
            }
        )

    game_df = pd.DataFrame(grouped_records).sort_values(["teamId", "startDate", "gameId"]).reset_index(drop=True)

    team_feature_rows: list[dict[str, Any]] = []
    for team_id, team_games in game_df.groupby("teamId", sort=False):
        rows = team_games.to_dict("records")
        for idx, current in enumerate(rows):
            if idx < MIN_PRIOR_GAMES:
                team_feature_rows.append(
                    {
                        "season": current["season"],
                        "gameId": current["gameId"],
                        "teamId": int(team_id),
                        **{name: np.nan for name in ROTATION_TEAM_FEATURES},
                    }
                )
                continue

            prior = rows[max(0, idx - ROTATION_WINDOW_GAMES):idx]
            last_game = rows[idx - 1]

            prior_minutes: dict[int, float] = {}
            prior_starter_counts: dict[int, int] = {}
            prior_starter_minutes: dict[int, float] = {}
            rotation_sets: list[set[int]] = []
            starter_reliable = True

            for game in prior:
                rotation_sets.append(set(game["top_rotation_players"]))
                for player_id, minutes in game["minutes_by_player"].items():
                    prior_minutes[player_id] = prior_minutes.get(player_id, 0.0) + minutes
                if len(game["starter_set"]) >= ROTATION_STARTER_SIZE:
                    for player_id in game["starter_set"]:
                        prior_starter_counts[player_id] = prior_starter_counts.get(player_id, 0) + 1
                        prior_starter_minutes[player_id] = prior_starter_minutes.get(player_id, 0.0) + game["minutes_by_player"].get(player_id, 0.0)
                else:
                    starter_reliable = False

            core_players = _top_n_players_by_minutes(prior_minutes, ROTATION_CORE_SIZE)
            core_minutes = sum(prior_minutes.get(pid, 0.0) for pid in core_players)
            present_last = last_game["present_players"]
            returned_minutes = sum(prior_minutes.get(pid, 0.0) for pid in core_players if pid in present_last)
            core_return = returned_minutes / core_minutes if core_minutes > 0 else np.nan

            overlap = (
                len(set(core_players) & set(last_game["top_rotation_players"])) / max(len(core_players), 1)
                if core_players else np.nan
            )

            churn_values = [
                _jaccard_churn(rotation_sets[i - 1], rotation_sets[i])
                for i in range(1, len(rotation_sets))
            ]
            churn_values = [v for v in churn_values if not pd.isna(v)]
            volatility = float(np.mean(churn_values)) if churn_values else np.nan

            starter_overlap = np.nan
            if starter_reliable and len(last_game["starter_set"]) >= ROTATION_STARTER_SIZE and prior_starter_counts:
                ordered_starters = sorted(
                    prior_starter_counts.items(),
                    key=lambda kv: (-kv[1], -prior_starter_minutes.get(kv[0], 0.0), kv[0]),
                )
                core_starters = [pid for pid, _ in ordered_starters[:ROTATION_STARTER_SIZE]]
                if core_starters:
                    starter_overlap = len(set(core_starters) & set(last_game["starter_set"])) / len(core_starters)

            team_feature_rows.append(
                {
                    "season": current["season"],
                    "gameId": current["gameId"],
                    "teamId": int(team_id),
                    "core_minutes_return_rate_5": core_return,
                    "rotation_overlap_5": overlap,
                    "missing_core_minutes_share_5": (1.0 - core_return) if not pd.isna(core_return) else np.nan,
                    "rotation_volatility_5": volatility,
                    "starter_overlap_5": starter_overlap,
                }
            )

    team_features = pd.DataFrame(team_feature_rows).sort_values(["season", "teamId", "gameId"]).reset_index(drop=True)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        team_features.to_parquet(output_dir / "rotation_availability_team_features_v1.parquet", index=False)
    return team_features


def merge_rotation_availability_features(
    feature_df: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach home/away rotation availability features to a game feature table."""
    if feature_df.empty:
        return feature_df

    home = team_features.rename(
        columns={
            "teamId": "homeTeamId",
            **{name: f"home_{name}" for name in ROTATION_TEAM_FEATURES},
        }
    )
    away = team_features.rename(
        columns={
            "teamId": "awayTeamId",
            **{name: f"away_{name}" for name in ROTATION_TEAM_FEATURES},
        }
    )

    out = feature_df.merge(
        home[["gameId", "homeTeamId", *[f"home_{name}" for name in ROTATION_TEAM_FEATURES]]],
        on=["gameId", "homeTeamId"],
        how="left",
    )
    out = out.merge(
        away[["gameId", "awayTeamId", *[f"away_{name}" for name in ROTATION_TEAM_FEATURES]]],
        on=["gameId", "awayTeamId"],
        how="left",
    )
    return out


def build_availability_shock_team_features(
    flat_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Build direct last-game availability-shock features using prior games only."""
    if flat_df.empty:
        return pd.DataFrame(columns=["season", "gameId", "teamId", *AVAILABILITY_SHOCK_TEAM_FEATURES])

    grouped_records: list[dict[str, Any]] = []
    grouped = flat_df.groupby(["season", "gameId", "teamId"], sort=False)
    for (season, game_id, team_id), group in grouped:
        group = group.sort_values(["minutes", "playerId"], ascending=[False, True])
        minutes_by_player = {
            int(row.playerId): float(0.0 if pd.isna(row.minutes) else row.minutes)
            for row in group.itertuples(index=False)
        }
        total_minutes = float(sum(minutes_by_player.values()))
        minute_share_by_player = {
            pid: (minutes / total_minutes if total_minutes > 0 else 0.0)
            for pid, minutes in minutes_by_player.items()
        }
        starter_set = set(
            int(row.playerId)
            for row in group.itertuples(index=False)
            if row.starter is True
        )
        grouped_records.append(
            {
                "season": int(season),
                "gameId": int(game_id),
                "teamId": int(team_id),
                "startDate": group["startDate"].iloc[0],
                "minutes_by_player": minutes_by_player,
                "minute_share_by_player": minute_share_by_player,
                "present_players": set(minutes_by_player.keys()),
                "starter_set": starter_set,
            }
        )

    game_df = pd.DataFrame(grouped_records).sort_values(["teamId", "startDate", "gameId"]).reset_index(drop=True)

    team_feature_rows: list[dict[str, Any]] = []
    for team_id, team_games in game_df.groupby("teamId", sort=False):
        rows = team_games.to_dict("records")
        for idx, current in enumerate(rows):
            if idx < ROTATION_WINDOW_GAMES:
                team_feature_rows.append(
                    {
                        "season": current["season"],
                        "gameId": current["gameId"],
                        "teamId": int(team_id),
                        **{name: np.nan for name in AVAILABILITY_SHOCK_TEAM_FEATURES},
                    }
                )
                continue

            prior = rows[idx - ROTATION_WINDOW_GAMES:idx]
            baseline_games = prior[:-1]
            last_game = prior[-1]
            if len(baseline_games) < ROTATION_WINDOW_GAMES - 1:
                team_feature_rows.append(
                    {
                        "season": current["season"],
                        "gameId": current["gameId"],
                        "teamId": int(team_id),
                        **{name: np.nan for name in AVAILABILITY_SHOCK_TEAM_FEATURES},
                    }
                )
                continue

            baseline_minutes: dict[int, float] = {}
            baseline_share_sum: dict[int, float] = {}
            baseline_starter_counts: dict[int, int] = {}

            for game in baseline_games:
                for pid, minutes in game["minutes_by_player"].items():
                    baseline_minutes[pid] = baseline_minutes.get(pid, 0.0) + minutes
                for pid, share in game["minute_share_by_player"].items():
                    baseline_share_sum[pid] = baseline_share_sum.get(pid, 0.0) + share
                for pid in game["starter_set"]:
                    baseline_starter_counts[pid] = baseline_starter_counts.get(pid, 0) + 1

            baseline_top = _top_n_players_by_minutes(baseline_minutes, 3)
            last_present = last_game["present_players"]
            last_shares = last_game["minute_share_by_player"]

            missing_top1 = float(int(len(baseline_top) >= 1 and baseline_top[0] not in last_present))
            missing_top2 = float(sum(pid not in last_present for pid in baseline_top[:2]))
            missing_top3 = float(sum(pid not in last_present for pid in baseline_top[:3]))

            top1_pid = baseline_top[0] if baseline_top else None
            baseline_top1_share = (
                baseline_share_sum.get(top1_pid, 0.0) / len(baseline_games)
                if top1_pid is not None else np.nan
            )
            last_top1_share = last_shares.get(top1_pid, 0.0) if top1_pid is not None else np.nan
            top1_share_change = (
                last_top1_share - baseline_top1_share
                if top1_pid is not None else np.nan
            )

            baseline_top3_share = (
                sum(baseline_share_sum.get(pid, 0.0) / len(baseline_games) for pid in baseline_top)
                if baseline_top else np.nan
            )
            last_top3_share = sum(last_shares.get(pid, 0.0) for pid in baseline_top)
            top3_share_change = (
                last_top3_share - baseline_top3_share
                if baseline_top else np.nan
            )

            likely_starters = sorted(
                baseline_starter_counts.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:ROTATION_STARTER_SIZE]
            likely_starter_ids = {pid for pid, count in likely_starters if count >= 2}
            likely_starter_missing = float(
                int(bool(likely_starter_ids) and any(pid not in last_present for pid in likely_starter_ids))
            )

            team_feature_rows.append(
                {
                    "season": current["season"],
                    "gameId": current["gameId"],
                    "teamId": int(team_id),
                    "missing_top1_minutes_last_game": missing_top1,
                    "missing_top2_minutes_last_game": missing_top2,
                    "missing_top3_minutes_last_game": missing_top3,
                    "top1_minutes_share_change_1": top1_share_change,
                    "top3_minutes_share_change_1": top3_share_change,
                    "likely_starter_missing_flag": likely_starter_missing,
                }
            )

    team_features = pd.DataFrame(team_feature_rows).sort_values(["season", "teamId", "gameId"]).reset_index(drop=True)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        team_features.to_parquet(output_dir / "availability_shock_team_features_v1.parquet", index=False)
    return team_features


def merge_availability_shock_features(
    feature_df: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach home/away availability shock features to a game feature table."""
    if feature_df.empty:
        return feature_df

    home = team_features.rename(
        columns={
            "teamId": "homeTeamId",
            **{name: f"home_{name}" for name in AVAILABILITY_SHOCK_TEAM_FEATURES},
        }
    )
    away = team_features.rename(
        columns={
            "teamId": "awayTeamId",
            **{name: f"away_{name}" for name in AVAILABILITY_SHOCK_TEAM_FEATURES},
        }
    )

    out = feature_df.merge(
        home[["gameId", "homeTeamId", *[f"home_{name}" for name in AVAILABILITY_SHOCK_TEAM_FEATURES]]],
        on=["gameId", "homeTeamId"],
        how="left",
    )
    out = out.merge(
        away[["gameId", "awayTeamId", *[f"away_{name}" for name in AVAILABILITY_SHOCK_TEAM_FEATURES]]],
        on=["gameId", "awayTeamId"],
        how="left",
    )
    return out

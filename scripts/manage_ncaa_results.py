#!/usr/bin/env python3
"""Validate and update NCAA tournament results files used by the bracket builder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "site" / "public" / "data"

CURRENT_SEASON = 2026
RESULTS_VERSION = 1
REGION_PAIRINGS = [
    {"seed_a": 1, "seed_b": 16, "slot": 1},
    {"seed_a": 8, "seed_b": 9, "slot": 2},
    {"seed_a": 5, "seed_b": 12, "slot": 3},
    {"seed_a": 4, "seed_b": 13, "slot": 4},
    {"seed_a": 6, "seed_b": 11, "slot": 5},
    {"seed_a": 3, "seed_b": 14, "slot": 6},
    {"seed_a": 7, "seed_b": 10, "slot": 7},
    {"seed_a": 2, "seed_b": 15, "slot": 8},
]
VALID_STATUSES = {"pending", "in_progress", "final"}


def _read_json(path: Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _field_path(season: int) -> Path:
    return DATA_DIR / f"ncaa_bracket_builder_{season}.json"


def _results_path(season: int) -> Path:
    return DATA_DIR / f"ncaa_results_{season}.json"


def _team_lookup(field: dict[str, Any]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for region in field["regions"]:
        for entry in region["entries"]:
            if entry["source"] == "team":
                lookup[int(entry["team_id"])] = str(entry["team"])
    for game in field["first_four"]:
        for team in game["teams"]:
            lookup[int(team["team_id"])] = str(team["team"])
    return lookup


def _region_source(entry: dict[str, Any], region_name: str) -> dict[str, Any]:
    if entry["source"] == "team":
        return {"type": "team", "team_id": int(entry["team_id"])}
    return {"type": "winner", "game_id": str(entry["play_in_game_id"])}


def _build_games(field: dict[str, Any]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []

    for index, game in enumerate(field["first_four"], start=1):
        games.append(
            {
                "id": str(game["id"]),
                "round_id": "first-four",
                "round_order": 0,
                "round_label": "First Four",
                "matchup_order": index,
                "source_a": {"type": "team", "team_id": int(game["teams"][0]["team_id"])},
                "source_b": {"type": "team", "team_id": int(game["teams"][1]["team_id"])},
            }
        )

    for region_index, region in enumerate(field["regions"]):
        region_name = str(region["name"])
        region_slug = region_name.lower().replace(" ", "-")
        entries_by_seed = {int(entry["seed"]): entry for entry in region["entries"]}

        for pairing in REGION_PAIRINGS:
            entry_a = entries_by_seed[pairing["seed_a"]]
            entry_b = entries_by_seed[pairing["seed_b"]]
            games.append(
                {
                    "id": f"{region_slug}-r64-{pairing['slot']}",
                    "round_id": "round-of-64",
                    "round_order": 1,
                    "round_label": "Round of 64",
                    "matchup_order": region_index * 8 + pairing["slot"],
                    "source_a": _region_source(entry_a, region_name),
                    "source_b": _region_source(entry_b, region_name),
                }
            )

        for slot in range(1, 5):
            games.append(
                {
                    "id": f"{region_slug}-r32-{slot}",
                    "round_id": "round-of-32",
                    "round_order": 2,
                    "round_label": "Round of 32",
                    "matchup_order": region_index * 4 + slot,
                    "source_a": {"type": "winner", "game_id": f"{region_slug}-r64-{slot * 2 - 1}"},
                    "source_b": {"type": "winner", "game_id": f"{region_slug}-r64-{slot * 2}"},
                }
            )

        for slot in range(1, 3):
            games.append(
                {
                    "id": f"{region_slug}-s16-{slot}",
                    "round_id": "sweet-16",
                    "round_order": 3,
                    "round_label": "Sweet 16",
                    "matchup_order": region_index * 2 + slot,
                    "source_a": {"type": "winner", "game_id": f"{region_slug}-r32-{slot * 2 - 1}"},
                    "source_b": {"type": "winner", "game_id": f"{region_slug}-r32-{slot * 2}"},
                }
            )

        games.append(
            {
                "id": f"{region_slug}-e8",
                "round_id": "elite-8",
                "round_order": 4,
                "round_label": "Elite 8",
                "matchup_order": region_index + 1,
                "source_a": {"type": "winner", "game_id": f"{region_slug}-s16-1"},
                "source_b": {"type": "winner", "game_id": f"{region_slug}-s16-2"},
            }
        )

    games.append(
        {
            "id": "final-four-1",
            "round_id": "final-four",
            "round_order": 5,
            "round_label": "Final Four",
            "matchup_order": 1,
            "source_a": {"type": "winner", "game_id": "east-e8"},
            "source_b": {"type": "winner", "game_id": "west-e8"},
        }
    )
    games.append(
        {
            "id": "final-four-2",
            "round_id": "final-four",
            "round_order": 5,
            "round_label": "Final Four",
            "matchup_order": 2,
            "source_a": {"type": "winner", "game_id": "south-e8"},
            "source_b": {"type": "winner", "game_id": "midwest-e8"},
        }
    )
    games.append(
        {
            "id": "national-championship",
            "round_id": "national-championship",
            "round_order": 6,
            "round_label": "National Championship",
            "matchup_order": 1,
            "source_a": {"type": "winner", "game_id": "final-four-1"},
            "source_b": {"type": "winner", "game_id": "final-four-2"},
        }
    )

    return games


def _collect_possible_source_teams(
    source: dict[str, Any],
    games_by_id: dict[str, dict[str, Any]],
    cache: dict[str, set[int]],
) -> set[int]:
    if source["type"] == "team":
        return {int(source["team_id"])}

    game_id = str(source["game_id"])
    if game_id in cache:
        return set(cache[game_id])

    feeder = games_by_id[game_id]
    teams = _collect_possible_source_teams(feeder["source_a"], games_by_id, cache) | _collect_possible_source_teams(
        feeder["source_b"],
        games_by_id,
        cache,
    )
    cache[game_id] = set(teams)
    return teams


def _possible_teams_by_game(games: list[dict[str, Any]]) -> dict[str, dict[str, set[int]]]:
    games_by_id = {str(game["id"]): game for game in games}
    cache: dict[str, set[int]] = {}
    out: dict[str, dict[str, set[int]]] = {}
    for game in games:
        source_a = _collect_possible_source_teams(game["source_a"], games_by_id, cache)
        source_b = _collect_possible_source_teams(game["source_b"], games_by_id, cache)
        out[str(game["id"])] = {"source_a": source_a, "source_b": source_b, "all": source_a | source_b}
    return out


def _team_labels(team_ids: set[int], team_lookup: dict[int, str]) -> str:
    ordered = sorted(team_ids)
    labels = [f"{team_id} ({team_lookup.get(team_id, 'Unknown')})" for team_id in ordered]
    if len(labels) > 8:
        return ", ".join(labels[:8]) + ", ..."
    return ", ".join(labels)


def _empty_results_payload(season: int) -> dict[str, Any]:
    return {"version": RESULTS_VERSION, "season": season, "games": {}}


def _validate_results_payload(
    payload: Any,
    season: int,
    games: list[dict[str, Any]],
    team_lookup: dict[int, str],
) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Results file must be a JSON object"], None
    if payload.get("version") != RESULTS_VERSION:
        errors.append(f"Expected results version {RESULTS_VERSION}, found {payload.get('version')}")
    if payload.get("season") != season:
        errors.append(f"Expected results season {season}, found {payload.get('season')}")

    games_payload = payload.get("games")
    if not isinstance(games_payload, dict):
        errors.append("Results file must include a top-level games object")
        return errors, None

    bracket_games = {str(game["id"]): game for game in games}
    possible = _possible_teams_by_game(games)

    for game_id, result in games_payload.items():
        if game_id not in bracket_games:
            sample = ", ".join(list(bracket_games.keys())[:8])
            errors.append(f"Unknown game id {game_id}. Known ids include: {sample}, ...")
            continue
        if not isinstance(result, dict):
            errors.append(f"Game {game_id} result must be an object")
            continue

        status = result.get("status")
        if status not in VALID_STATUSES:
            allowed = ", ".join(sorted(VALID_STATUSES))
            errors.append(f"Game {game_id} has invalid status {status!r}. Allowed statuses: {allowed}")
            continue

        winner = result.get("winner_team_id")
        loser = result.get("loser_team_id")
        if status != "final":
            if winner is not None or loser is not None:
                errors.append(f"Game {game_id} is {status}; only final games may include winner_team_id and loser_team_id")
            continue

        if not isinstance(winner, int) or not isinstance(loser, int):
            errors.append(f"Game {game_id} is final and must include integer winner_team_id and loser_team_id")
            continue
        if winner == loser:
            errors.append(f"Game {game_id} winner_team_id and loser_team_id must be different")
            continue

        source_a = possible[game_id]["source_a"]
        source_b = possible[game_id]["source_b"]
        valid_orientation = (winner in source_a and loser in source_b) or (winner in source_b and loser in source_a)
        if not valid_orientation:
            errors.append(
                f"Game {game_id} cannot be {winner} over {loser}. "
                f"Expected one team from source A [{_team_labels(source_a, team_lookup)}] "
                f"and one from source B [{_team_labels(source_b, team_lookup)}]."
            )

    normalized = {
        "version": RESULTS_VERSION,
        "season": season,
        "games": {
            game["id"]: games_payload[game["id"]]
            for game in games
            if game["id"] in games_payload
        },
    }
    return errors, normalized


def _load_context(season: int, field_path: Path, results_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, str], dict[str, Any]]:
    if not field_path.exists():
        raise FileNotFoundError(
            f"Missing bracket field file {field_path}. Generate ncaa_bracket_builder_{season}.json before updating results."
        )
    field = _read_json(field_path)
    games = _build_games(field)
    team_lookup = _team_lookup(field)
    payload = _read_json(results_path) if results_path.exists() else _empty_results_payload(season)
    return field, games, team_lookup, payload


def _validate_command(args: argparse.Namespace) -> int:
    _, games, team_lookup, payload = _load_context(args.season, args.field_path, args.results_path)
    errors, normalized = _validate_results_payload(payload, args.season, games, team_lookup)
    if errors:
        print("NCAA results validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if normalized is None:
        print("NCAA results validation failed: internal normalization error", file=sys.stderr)
        return 1

    if args.write:
        _write_json(args.results_path, normalized)
        print(f"Validated and wrote {args.results_path}")
    else:
        print(json.dumps(normalized, indent=2))
    return 0


def _finalize_command(args: argparse.Namespace) -> int:
    _, games, team_lookup, payload = _load_context(args.season, args.field_path, args.results_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("games"), dict):
        payload = _empty_results_payload(args.season)

    payload["version"] = RESULTS_VERSION
    payload["season"] = args.season
    payload.setdefault("games", {})
    payload["games"][args.game_id] = {
        "winner_team_id": args.winner_team_id,
        "loser_team_id": args.loser_team_id,
        "status": "final",
    }

    errors, normalized = _validate_results_payload(payload, args.season, games, team_lookup)
    if errors:
        print("Could not mark NCAA game final:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if normalized is None:
        print("Could not mark NCAA game final: internal normalization error", file=sys.stderr)
        return 1

    _write_json(args.results_path, normalized)
    print(
        f"Updated {args.results_path}: {args.game_id} = "
        f"{args.winner_team_id} ({team_lookup.get(args.winner_team_id, 'Unknown')}) over "
        f"{args.loser_team_id} ({team_lookup.get(args.loser_team_id, 'Unknown')})"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate and optionally pretty-print a season results file")
    validate.add_argument("--season", type=int, default=CURRENT_SEASON)
    validate.add_argument("--write", action="store_true", help="Rewrite the results file in normalized pretty-printed form")
    validate.add_argument("--field-path", type=Path, default=None)
    validate.add_argument("--results-path", type=Path, default=None)
    validate.set_defaults(func=_validate_command)

    finalize = subparsers.add_parser("finalize", help="Mark a bracket game final in the season results file")
    finalize.add_argument("--season", type=int, default=CURRENT_SEASON)
    finalize.add_argument("--game-id", required=True)
    finalize.add_argument("--winner-team-id", type=int, required=True)
    finalize.add_argument("--loser-team-id", type=int, required=True)
    finalize.add_argument("--field-path", type=Path, default=None)
    finalize.add_argument("--results-path", type=Path, default=None)
    finalize.set_defaults(func=_finalize_command)

    args = parser.parse_args()
    args.field_path = args.field_path or _field_path(args.season)
    args.results_path = args.results_path or _results_path(args.season)
    return args


def main() -> int:
    args = _parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

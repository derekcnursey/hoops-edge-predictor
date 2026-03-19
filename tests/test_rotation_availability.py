from __future__ import annotations

import pandas as pd

from src.rotation_availability import (
    build_rotation_availability_team_features,
    merge_rotation_availability_features,
    parse_players_payload,
)


def test_parse_players_payload_handles_python_and_json_strings():
    python_literal = "[{'athleteId': 1, 'minutes': 30, 'starter': True}]"
    json_literal = '[{"athleteId": 2, "minutes": 25, "starter": true}]'

    parsed_python = parse_players_payload(python_literal)
    parsed_json = parse_players_payload(json_literal)

    assert parsed_python == [{"athleteId": 1, "minutes": 30, "starter": True}]
    assert parsed_json == [{"athleteId": 2, "minutes": 25, "starter": True}]


def test_build_rotation_availability_team_features_uses_prior_games_only():
    rows = []
    for game_id, start_date, players in [
        (
            1,
            "2024-11-01T18:00:00Z",
            [(1, 30, True), (2, 28, True), (3, 22, True), (4, 20, True), (5, 18, True), (6, 10, False)],
        ),
        (
            2,
            "2024-11-05T18:00:00Z",
            [(1, 31, True), (2, 27, True), (3, 21, True), (4, 18, True), (5, 17, True), (6, 12, False)],
        ),
        (
            3,
            "2024-11-10T18:00:00Z",
            [(1, 32, True), (2, 26, True), (3, 20, True), (4, 19, True), (5, 16, True), (6, 11, False)],
        ),
        (
            4,
            "2024-11-15T18:00:00Z",
            [(2, 25, True), (3, 19, True), (4, 18, True), (5, 15, True), (6, 10, False), (7, 14, True)],
        ),
    ]:
        for player_id, minutes, starter in players:
            rows.append(
                {
                    "season": 2025,
                    "gameId": game_id,
                    "teamId": 10,
                    "startDate": pd.Timestamp(start_date),
                    "playerId": player_id,
                    "minutes": minutes,
                    "starter": starter,
                }
            )

    rows.extend(
        {
            "season": 2025,
            "gameId": 5,
            "teamId": 10,
            "startDate": pd.Timestamp("2024-11-20T18:00:00Z"),
            "playerId": player_id,
            "minutes": minutes,
            "starter": starter,
        }
        for player_id, minutes, starter in [(2, 30, True), (3, 28, True), (4, 24, True), (5, 18, True), (6, 15, True), (7, 12, False)]
    )

    features = build_rotation_availability_team_features(pd.DataFrame(rows))
    game5 = features.loc[features["gameId"] == 5].iloc[0]

    assert game5["core_minutes_return_rate_5"] < 1.0
    assert game5["missing_core_minutes_share_5"] > 0.0
    assert 0.0 <= game5["rotation_overlap_5"] <= 1.0
    assert 0.0 <= game5["starter_overlap_5"] <= 1.0


def test_merge_rotation_availability_features_adds_home_and_away_columns():
    base = pd.DataFrame(
        [
            {
                "gameId": 100,
                "homeTeamId": 1,
                "awayTeamId": 2,
                "homeScore": 70,
                "awayScore": 65,
            }
        ]
    )
    team_features = pd.DataFrame(
        [
            {"gameId": 100, "teamId": 1, "core_minutes_return_rate_5": 0.8, "rotation_overlap_5": 0.75, "missing_core_minutes_share_5": 0.2, "rotation_volatility_5": 0.1, "starter_overlap_5": 0.8},
            {"gameId": 100, "teamId": 2, "core_minutes_return_rate_5": 0.7, "rotation_overlap_5": 0.5, "missing_core_minutes_share_5": 0.3, "rotation_volatility_5": 0.2, "starter_overlap_5": 0.6},
        ]
    )

    merged = merge_rotation_availability_features(base, team_features)

    assert merged.loc[0, "home_core_minutes_return_rate_5"] == 0.8
    assert merged.loc[0, "away_core_minutes_return_rate_5"] == 0.7

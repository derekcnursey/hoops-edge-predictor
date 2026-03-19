from __future__ import annotations

import pandas as pd

from src.rotation_availability import (
    build_availability_shock_team_features,
    merge_availability_shock_features,
)


def test_build_availability_shock_team_features_flags_missing_top_players():
    rows = []
    prior_games = [
        (1, "2024-11-01T18:00:00Z", [(1, 32, True), (2, 28, True), (3, 24, True), (4, 18, True), (5, 16, True), (6, 8, False)]),
        (2, "2024-11-05T18:00:00Z", [(1, 31, True), (2, 27, True), (3, 25, True), (4, 17, True), (5, 15, True), (6, 9, False)]),
        (3, "2024-11-10T18:00:00Z", [(1, 30, True), (2, 29, True), (3, 23, True), (4, 18, True), (5, 14, True), (6, 10, False)]),
        (4, "2024-11-15T18:00:00Z", [(1, 33, True), (2, 26, True), (3, 22, True), (4, 18, True), (5, 13, True), (6, 9, False)]),
        (5, "2024-11-20T18:00:00Z", [(2, 30, True), (3, 26, True), (4, 22, True), (5, 18, True), (6, 12, True), (7, 10, False)]),
    ]
    for game_id, start_date, players in prior_games:
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
            "gameId": 6,
            "teamId": 10,
            "startDate": pd.Timestamp("2024-11-25T18:00:00Z"),
            "playerId": player_id,
            "minutes": minutes,
            "starter": starter,
        }
        for player_id, minutes, starter in [(2, 28, True), (3, 27, True), (4, 22, True), (5, 19, True), (6, 12, True), (7, 12, False)]
    )

    features = build_availability_shock_team_features(pd.DataFrame(rows))
    game6 = features.loc[features["gameId"] == 6].iloc[0]

    assert game6["missing_top1_minutes_last_game"] == 1.0
    assert game6["missing_top2_minutes_last_game"] >= 1.0
    assert game6["missing_top3_minutes_last_game"] >= 1.0
    assert game6["top1_minutes_share_change_1"] < 0.0
    assert game6["likely_starter_missing_flag"] == 1.0


def test_merge_availability_shock_features_adds_home_and_away_columns():
    base = pd.DataFrame([{"gameId": 1, "homeTeamId": 10, "awayTeamId": 20}])
    team_features = pd.DataFrame(
        [
            {
                "gameId": 1,
                "teamId": 10,
                "missing_top1_minutes_last_game": 1.0,
                "missing_top2_minutes_last_game": 1.0,
                "missing_top3_minutes_last_game": 2.0,
                "top1_minutes_share_change_1": -0.2,
                "top3_minutes_share_change_1": -0.15,
                "likely_starter_missing_flag": 1.0,
            },
            {
                "gameId": 1,
                "teamId": 20,
                "missing_top1_minutes_last_game": 0.0,
                "missing_top2_minutes_last_game": 0.0,
                "missing_top3_minutes_last_game": 0.0,
                "top1_minutes_share_change_1": 0.05,
                "top3_minutes_share_change_1": 0.04,
                "likely_starter_missing_flag": 0.0,
            },
        ]
    )

    merged = merge_availability_shock_features(base, team_features)

    assert merged.loc[0, "home_missing_top1_minutes_last_game"] == 1.0
    assert merged.loc[0, "away_top1_minutes_share_change_1"] == 0.05

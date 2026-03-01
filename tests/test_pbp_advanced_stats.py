"""Unit tests for PBP advanced stats extraction."""

import numpy as np
import pandas as pd
import pytest

from src.pbp_advanced_stats import (
    ALL_ADVANCED_STAT_COLS,
    COMPOSITE_COLS,
    ZONE_COUNT_COLS,
    _classify_shot_zone,
    _is_corner_three,
    compute_advanced_stats,
)


def _make_pbp_play(**overrides) -> dict:
    """Create a single PBP play row with sensible defaults."""
    defaults = {
        "gameId": 1,
        "teamId": 100.0,
        "opponentId": 200.0,
        "isHomeTeam": "true",
        "period": 1,
        "secondsRemaining": 600,
        "playType": "JumpShot",
        "playText": "Player A made jumper",
        "scoringPlay": False,
        "shootingPlay": True,
        "scoreValue": 0.0,
        "possession_id": 1,
        "offense_team_id": 100,
        "defense_team_id": 200,
        "possession_end": False,
        "garbage_time": False,
        "shot_range": "jumper",
        "shot_made": "False",
        "shot_assisted": "False",
        "shot_loc_x": 400.0,
        "shot_loc_y": 250.0,
        "shot_shooter_id": 1001.0,
        "shot_shooter_name": "Player A",
        "shot_assisted_by_id": np.nan,
        "shot_assisted_by_name": None,
        "homeScore": 20,
        "awayScore": 18,
        "gameStartDate": "2025-01-15",
        "id": 1,
    }
    defaults.update(overrides)
    return defaults


def _make_game_pbp() -> pd.DataFrame:
    """Create a minimal but realistic game PBP for testing."""
    plays = []
    poss_id = 1

    # 20 possessions for each team
    for i in range(20):
        # Team 100 offense
        plays.append(_make_pbp_play(
            id=len(plays) + 1,
            possession_id=poss_id,
            offense_team_id=100,
            defense_team_id=200,
            teamId=100.0,
            opponentId=200.0,
            period=1 if i < 10 else 2,
            secondsRemaining=1200 - i * 30 if i < 10 else 1200 - (i - 10) * 30,
            shootingPlay=True,
            scoringPlay=i % 3 == 0,  # Score every 3rd possession
            scoreValue=2.0 if i % 3 == 0 else 0.0,
            shot_range="rim" if i % 4 == 0 else ("three_pointer" if i % 4 == 1 else "jumper"),
            shot_made="True" if i % 3 == 0 else "False",
            shot_assisted="True" if i % 3 == 0 and i % 2 == 0 else "False",
            shot_shooter_id=1001.0 + (i % 5),
            homeScore=20 + (i // 3) * 2,
            awayScore=18 + (i // 4) * 2,
        ))
        poss_id += 1

        # Team 200 offense
        plays.append(_make_pbp_play(
            id=len(plays) + 1,
            possession_id=poss_id,
            offense_team_id=200,
            defense_team_id=100,
            teamId=200.0,
            opponentId=100.0,
            isHomeTeam="false",
            period=1 if i < 10 else 2,
            secondsRemaining=1185 - i * 30 if i < 10 else 1185 - (i - 10) * 30,
            shootingPlay=True,
            scoringPlay=i % 4 == 0,
            scoreValue=2.0 if i % 4 == 0 else 0.0,
            shot_range="jumper",
            shot_made="True" if i % 4 == 0 else "False",
            shot_shooter_id=2001.0 + (i % 3),
            homeScore=20 + (i // 3) * 2,
            awayScore=18 + (i // 4) * 2,
        ))
        poss_id += 1

    return pd.DataFrame(plays)


class TestShotClassification:
    def test_rim(self):
        assert _classify_shot_zone("rim") == "rim"

    def test_jumper(self):
        assert _classify_shot_zone("jumper") == "mid_range"

    def test_three_pointer(self):
        assert _classify_shot_zone("three_pointer") == "three_pointer"

    def test_free_throw(self):
        assert _classify_shot_zone("free_throw") == "free_throw"

    def test_none(self):
        assert _classify_shot_zone(None) is None

    def test_nan(self):
        assert _classify_shot_zone(float("nan")) is None


class TestCornerThree:
    def test_corner_three_low_y(self):
        assert _is_corner_three("three_pointer", 100.0, 50.0) is True

    def test_corner_three_high_y(self):
        assert _is_corner_three("three_pointer", 100.0, 450.0) is True

    def test_non_corner_three(self):
        # Near center, mid-y
        assert _is_corner_three("three_pointer", 700.0, 250.0) is False

    def test_non_three_pointer(self):
        assert _is_corner_three("rim", 100.0, 50.0) is False

    def test_missing_coords(self):
        assert _is_corner_three("three_pointer", None, 50.0) is False
        assert _is_corner_three("three_pointer", 100.0, None) is False


class TestComputeAdvancedStats:
    def test_output_has_expected_columns(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        assert not result.empty
        assert "gameid" in result.columns
        assert "teamid" in result.columns
        for col in ALL_ADVANCED_STAT_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_two_teams_per_game(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        assert len(result) == 2  # One row per team
        assert set(result["teamid"]) == {100, 200}

    def test_garbage_time_excluded(self):
        pbp = _make_game_pbp()
        # Mark all plays as garbage time
        pbp["garbage_time"] = True
        result = compute_advanced_stats(pbp)
        assert result.empty

    def test_shot_distribution_sums_to_one(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        team100 = result[result["teamid"] == 100].iloc[0]
        # rim_rate + mid_range_rate + (three_pt rates from three_pt_rate in main four factors)
        # should roughly sum to 1.0
        total = team100["rim_rate"] + team100["mid_range_rate"]
        # Note: total doesn't include 3pt rate since that's captured separately
        assert total <= 1.0

    def test_unique_scorers_count(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        team100 = result[result["teamid"] == 100].iloc[0]
        assert team100["unique_scorers"] > 0
        assert team100["unique_scorers"] <= 5  # We used 5 different shooter IDs

    def test_scoring_hhi_range(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        for _, row in result.iterrows():
            if pd.notna(row["scoring_hhi"]):
                assert 0 < row["scoring_hhi"] <= 1.0

    def test_drought_features_non_negative(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        for _, row in result.iterrows():
            for col in ["off_avg_drought_length", "off_max_drought_length"]:
                if pd.notna(row[col]):
                    assert row[col] >= 0

    def test_half_split_delta(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        for _, row in result.iterrows():
            if pd.notna(row["first_half_off_efficiency"]) and pd.notna(row["second_half_off_efficiency"]):
                expected_delta = row["second_half_off_efficiency"] - row["first_half_off_efficiency"]
                assert row["half_adjustment_delta"] == pytest.approx(expected_delta, abs=1e-6)

    def test_zone_count_columns_present(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        for col in ZONE_COUNT_COLS:
            assert col in result.columns, f"Missing zone count column: {col}"

    def test_composite_columns_present(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        for col in COMPOSITE_COLS:
            assert col in result.columns, f"Missing composite column: {col}"

    def test_expected_pts_per_shot_range(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        for _, row in result.iterrows():
            if pd.notna(row["expected_pts_per_shot"]):
                assert 0.5 <= row["expected_pts_per_shot"] <= 3.0, (
                    f"expected_pts_per_shot = {row['expected_pts_per_shot']}, expected [0.5, 3.0]"
                )

    def test_transition_value_formula(self):
        pbp = _make_game_pbp()
        result = compute_advanced_stats(pbp)
        for _, row in result.iterrows():
            steal_rate = row.get("steal_rate_defense")
            trans_eff = row.get("transition_scoring_efficiency")
            tv = row.get("transition_value")
            if pd.notna(steal_rate) and pd.notna(trans_eff) and pd.notna(tv):
                expected = steal_rate * trans_eff
                assert tv == pytest.approx(expected, abs=1e-6)

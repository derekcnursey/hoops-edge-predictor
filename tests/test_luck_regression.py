"""Unit tests for luck / regression-to-the-mean features."""

import numpy as np
import pandas as pd
import pytest

from src.luck_regression import LUCK_FEATURE_COLS, compute_luck_features


def _make_adv_stats(
    n_games: int = 20,
    n_teams: int = 4,
    start_date: str = "2025-01-01",
) -> pd.DataFrame:
    """Create synthetic advanced stats for testing luck features.

    Each team plays n_games games, with slightly different shooting profiles.
    """
    dates = pd.date_range(start_date, periods=n_games, freq="D")
    rows = []
    game_id = 1

    for d_idx, date in enumerate(dates):
        # Two games per date, each involving 2 teams
        for g in range(n_teams // 2):
            team_a = g * 2
            team_b = g * 2 + 1

            for team_id in [team_a, team_b]:
                opp_id = team_b if team_id == team_a else team_a

                # Deterministic shooting: team 0 is a good shooter, team 1 is average, etc.
                base_fg = 0.45 + team_id * 0.02

                rim_fga = 15
                rim_fgm = int(rim_fga * (base_fg + 0.15))  # Rim ~60%+
                mid_fga = 8
                mid_fgm = int(mid_fga * base_fg)
                three_fga = 12
                three_fgm = int(three_fga * (base_fg - 0.10))
                total_fga = rim_fga + mid_fga + three_fga

                two_pt_fga = rim_fga + mid_fga
                two_pt_fgm = rim_fgm + mid_fgm

                rows.append({
                    "gameid": game_id,
                    "teamid": team_id,
                    "opponentid": opp_id,
                    "startdate": date.strftime("%Y-%m-%d"),
                    "rim_fga": rim_fga,
                    "rim_fgm": rim_fgm,
                    "mid_fga": mid_fga,
                    "mid_fgm": mid_fgm,
                    "three_fga": three_fga,
                    "three_fgm": three_fgm,
                    "rim_rate": rim_fga / total_fga,
                    "mid_range_rate": mid_fga / total_fga,
                    "rim_fg_pct": rim_fgm / rim_fga,
                    "three_pt_fg_pct": three_fgm / three_fga if three_fga > 0 else np.nan,
                    "two_pt_fg_pct": two_pt_fgm / two_pt_fga if two_pt_fga > 0 else np.nan,
                })

            game_id += 1

    return pd.DataFrame(rows)


class TestComputeLuckFeatures:
    def test_output_columns(self):
        adv = _make_adv_stats()
        result = compute_luck_features(adv)
        assert "gameid" in result.columns
        assert "teamid" in result.columns
        assert "startdate" in result.columns
        for col in LUCK_FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_first_game_has_nan_luck(self):
        """First games of the season should have NaN luck (no league avg yet)."""
        adv = _make_adv_stats()
        result = compute_luck_features(adv)

        # Get first date
        first_date = sorted(result["startdate"].unique())[0]
        first_day = result[result["startdate"] == first_date]

        for col in LUCK_FEATURE_COLS:
            assert first_day[col].isna().all(), f"{col} should be NaN on first day"

    def test_later_games_have_values(self):
        """Games after the first day should have non-NaN luck values."""
        adv = _make_adv_stats(n_games=10)
        result = compute_luck_features(adv)

        # Get a later date
        dates = sorted(result["startdate"].unique())
        later = result[result["startdate"] == dates[-1]]

        for col in LUCK_FEATURE_COLS:
            assert later[col].notna().any(), f"{col} should have values on later days"

    def test_shooting_above_league_avg_gives_positive_luck(self):
        """A team shooting above league average should have positive luck."""
        # Create a scenario: 2 teams, one shoots much better than the other
        rows = []
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        game_id = 1

        for d in dates:
            # Team 0: bad shooter (below league avg)
            rows.append({
                "gameid": game_id, "teamid": 0, "opponentid": 1,
                "startdate": d.strftime("%Y-%m-%d"),
                "rim_fga": 15, "rim_fgm": 7,
                "mid_fga": 8, "mid_fgm": 2,
                "three_fga": 12, "three_fgm": 2,
                "rim_rate": 15 / 35, "mid_range_rate": 8 / 35,
                "rim_fg_pct": 7 / 15, "three_pt_fg_pct": 2 / 12,
                "two_pt_fg_pct": 9 / 23,
            })
            # Team 1: great shooter (above league avg)
            rows.append({
                "gameid": game_id, "teamid": 1, "opponentid": 0,
                "startdate": d.strftime("%Y-%m-%d"),
                "rim_fga": 15, "rim_fgm": 12,
                "mid_fga": 8, "mid_fgm": 6,
                "three_fga": 12, "three_fgm": 7,
                "rim_rate": 15 / 35, "mid_range_rate": 8 / 35,
                "rim_fg_pct": 12 / 15, "three_pt_fg_pct": 7 / 12,
                "two_pt_fg_pct": 18 / 23,
            })
            game_id += 1

        adv = pd.DataFrame(rows)
        result = compute_luck_features(adv)

        # After a few games, team 1 (good shooter) should have positive luck
        late = result[(result["startdate"] == dates[-1].strftime("%Y-%m-%d"))]
        team1_luck = late[late["teamid"] == 1]["efg_luck"].iloc[0]
        team0_luck = late[late["teamid"] == 0]["efg_luck"].iloc[0]

        assert team1_luck > 0, "Good shooter should have positive efg_luck"
        assert team0_luck < 0, "Bad shooter should have negative efg_luck"

    def test_causal_ordering(self):
        """Verify that luck values on day N use only data from before day N."""
        adv = _make_adv_stats(n_games=5, n_teams=2)
        result = compute_luck_features(adv)
        dates = sorted(result["startdate"].unique())

        # Day 0: NaN (no prior data)
        day0 = result[result["startdate"] == dates[0]]
        assert day0["efg_luck"].isna().all()

        # Day 1: uses only day 0 data
        day1 = result[result["startdate"] == dates[1]]
        assert day1["efg_luck"].notna().any()

    def test_mean_luck_near_zero(self):
        """Mean luck across all teams should be approximately 0 over a full season."""
        adv = _make_adv_stats(n_games=30, n_teams=4)
        result = compute_luck_features(adv)

        # Exclude first day (NaN)
        valid = result[result["efg_luck"].notna()]
        mean_luck = valid["efg_luck"].mean()

        # Mean luck should be close to zero (within tolerance)
        assert abs(mean_luck) < 0.15, f"Mean efg_luck = {mean_luck:.4f}, expected near 0"

    def test_empty_input(self):
        """Empty input should return empty DataFrame with correct columns."""
        result = compute_luck_features(pd.DataFrame())
        assert result.empty
        expected_cols = {"gameid", "teamid", "startdate"} | set(LUCK_FEATURE_COLS)
        assert set(result.columns) == expected_cols

    def test_missing_columns(self):
        """Missing required columns should return empty DataFrame."""
        df = pd.DataFrame({"gameid": [1], "teamid": [1]})
        result = compute_luck_features(df)
        assert result.empty

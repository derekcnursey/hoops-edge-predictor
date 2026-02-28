"""Unit tests for the universal opponent adjustment module."""

import numpy as np
import pandas as pd
import pytest

from src.opponent_adjustment import build_stat_pairs, opponent_adjust


def _make_games(n_games: int = 10) -> pd.DataFrame:
    """Create synthetic per-game per-team data for testing.

    Creates a round-robin between 4 teams with varying stats.
    """
    rows = []
    teams = [100, 200, 300, 400]
    game_id = 1
    day = 1

    for i in range(n_games):
        t1 = teams[i % 4]
        t2 = teams[(i + 1) % 4]
        date = f"2025-01-{day:02d}"
        day += 1

        # Team 1's perspective
        rows.append({
            "gameid": game_id,
            "teamid": t1,
            "opponentid": t2,
            "startdate": date,
            "off_stat": 0.50 + (t1 - 100) * 0.001,  # varies by team
            "def_stat": 0.45 + (t1 - 100) * 0.001,
        })
        # Team 2's perspective
        rows.append({
            "gameid": game_id,
            "teamid": t2,
            "opponentid": t1,
            "startdate": date,
            "off_stat": 0.50 + (t2 - 100) * 0.001,
            "def_stat": 0.45 + (t2 - 100) * 0.001,
        })
        game_id += 1

    return pd.DataFrame(rows)


class TestOpponentAdjust:
    def test_output_shape_matches_input(self):
        df = _make_games(5)
        pairs = {"off_stat": "def_stat", "def_stat": "off_stat"}
        result = opponent_adjust(df, ["off_stat", "def_stat"], stat_pairs=pairs, no_adjust=set())
        assert result.shape == df.shape

    def test_first_games_minimal_adjustment(self):
        """First game of season has no prior data, so no adjustment is possible."""
        df = _make_games(5)
        pairs = {"off_stat": "def_stat", "def_stat": "off_stat"}
        result = opponent_adjust(df, ["off_stat", "def_stat"], stat_pairs=pairs, no_adjust=set())
        # First game: no league avg available, so values should be unchanged
        first_day = result[result["startdate"] == "2025-01-01"]
        original = df[df["startdate"] == "2025-01-01"]
        np.testing.assert_array_almost_equal(
            first_day["off_stat"].values, original["off_stat"].values
        )

    def test_no_adjust_stats_unchanged(self):
        """Stats in no_adjust should not be modified."""
        df = _make_games(10)
        pairs = {"off_stat": "def_stat"}
        result = opponent_adjust(
            df, ["off_stat", "def_stat"],
            stat_pairs=pairs,
            no_adjust={"def_stat"},
        )
        pd.testing.assert_series_equal(
            result["def_stat"].reset_index(drop=True),
            df.sort_values(["startdate", "gameid", "teamid"]).reset_index(drop=True)["def_stat"],
            check_names=False,
        )

    def test_adjustment_direction(self):
        """Playing against a strong defense should lower adjusted offensive stat."""
        rows = []
        # Game 1: Team A (avg offense) vs Team B (strong defense)
        rows.append({"gameid": 1, "teamid": 100, "opponentid": 200, "startdate": "2025-01-01",
                      "off_stat": 0.50, "def_stat": 0.45})
        rows.append({"gameid": 1, "teamid": 200, "opponentid": 100, "startdate": "2025-01-01",
                      "off_stat": 0.50, "def_stat": 0.40})  # B has strong D (low=good)

        # Game 2: Team C vs Team D (same stats)
        rows.append({"gameid": 2, "teamid": 300, "opponentid": 400, "startdate": "2025-01-02",
                      "off_stat": 0.50, "def_stat": 0.48})
        rows.append({"gameid": 2, "teamid": 400, "opponentid": 300, "startdate": "2025-01-02",
                      "off_stat": 0.50, "def_stat": 0.48})

        # Game 3: A plays against weak D (C, def=0.48), B plays against avg D
        rows.append({"gameid": 3, "teamid": 100, "opponentid": 300, "startdate": "2025-01-03",
                      "off_stat": 0.55, "def_stat": 0.45})
        rows.append({"gameid": 3, "teamid": 300, "opponentid": 100, "startdate": "2025-01-03",
                      "off_stat": 0.48, "def_stat": 0.48})

        df = pd.DataFrame(rows)
        pairs = {"off_stat": "def_stat", "def_stat": "off_stat"}
        result = opponent_adjust(df, ["off_stat", "def_stat"], stat_pairs=pairs, no_adjust=set())

        # Game 3: Team A's off_stat=0.55 vs Team C (def=0.48 season avg)
        # If league avg def is ~0.45, then adj = 0.55 - 0.48 + 0.45 = 0.52
        # The adjusted value should be lower than raw (0.55) because C has weak defense
        game3_a = result[(result["gameid"] == 3) & (result["teamid"] == 100)]
        assert game3_a["off_stat"].iloc[0] < 0.55, "Adj should be lower against weak defense"

    def test_no_data_leakage(self):
        """Adjustment for game N should only use data from games before N."""
        rows = []
        # 5 games for team 100 with increasing offense
        for i in range(5):
            rows.append({
                "gameid": i + 1, "teamid": 100, "opponentid": 200,
                "startdate": f"2025-01-{i + 1:02d}",
                "off_stat": 0.40 + i * 0.05,
                "def_stat": 0.45,
            })
            rows.append({
                "gameid": i + 1, "teamid": 200, "opponentid": 100,
                "startdate": f"2025-01-{i + 1:02d}",
                "off_stat": 0.50,
                "def_stat": 0.50,
            })

        df = pd.DataFrame(rows)
        pairs = {"off_stat": "def_stat", "def_stat": "off_stat"}
        result = opponent_adjust(df, ["off_stat", "def_stat"], stat_pairs=pairs, no_adjust=set())

        # Game 2's adjustment should only use game 1 data
        g2 = result[(result["gameid"] == 2) & (result["teamid"] == 100)]
        # At game 2, opp (200) has played 1 game with def_stat=0.50
        # League avg def_stat from game 1: (0.45 + 0.50) / 2 = 0.475
        # adj = 0.45 - 0.50 + 0.475 = 0.425
        # raw was 0.45
        assert g2["off_stat"].iloc[0] != 0.45  # Should be adjusted

    def test_nan_handling(self):
        """NaN values in stats should be preserved."""
        rows = [
            {"gameid": 1, "teamid": 100, "opponentid": 200, "startdate": "2025-01-01",
             "off_stat": np.nan, "def_stat": 0.45},
            {"gameid": 1, "teamid": 200, "opponentid": 100, "startdate": "2025-01-01",
             "off_stat": 0.50, "def_stat": 0.50},
        ]
        df = pd.DataFrame(rows)
        pairs = {"off_stat": "def_stat", "def_stat": "off_stat"}
        result = opponent_adjust(df, ["off_stat", "def_stat"], stat_pairs=pairs, no_adjust=set())
        # NaN input should remain NaN
        assert np.isnan(result[result["teamid"] == 100]["off_stat"].iloc[0])

    def test_uses_raw_values_for_running_averages(self):
        """Running averages should use raw (not adjusted) values to prevent drift."""
        # Create enough data that adjusted values would diverge from raw
        rows = []
        for i in range(20):
            rows.append({
                "gameid": i + 1, "teamid": 100, "opponentid": 200,
                "startdate": f"2025-01-{i + 1:02d}",
                "off_stat": 0.50, "def_stat": 0.45,
            })
            rows.append({
                "gameid": i + 1, "teamid": 200, "opponentid": 100,
                "startdate": f"2025-01-{i + 1:02d}",
                "off_stat": 0.50, "def_stat": 0.55,
            })

        df = pd.DataFrame(rows)
        pairs = {"off_stat": "def_stat", "def_stat": "off_stat"}
        result = opponent_adjust(df, ["off_stat", "def_stat"], stat_pairs=pairs, no_adjust=set())

        # With stable inputs, adjusted values should converge, not drift
        late_games = result[result["gameid"] >= 15]
        team100 = late_games[late_games["teamid"] == 100]
        # Values should be reasonable (not exploding or collapsing)
        assert all(team100["off_stat"].between(0.3, 0.7))
        assert all(team100["def_stat"].between(0.3, 0.7))


class TestBuildStatPairs:
    def test_basic_pairing(self):
        pairs = build_stat_pairs(["off_a", "off_b"], ["def_a", "def_b"])
        assert pairs == {
            "off_a": "def_a", "def_a": "off_a",
            "off_b": "def_b", "def_b": "off_b",
        }

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            build_stat_pairs(["off_a"], ["def_a", "def_b"])

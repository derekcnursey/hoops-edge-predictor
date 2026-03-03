"""Unit tests for feature assembly."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import _compute_barthag, build_features, get_feature_matrix


class TestBarthag:
    def test_known_values(self):
        """BARTHAG for adj_oe=115, adj_de=95 should be high (~0.89+)."""
        result = _compute_barthag(115.0, 95.0)
        assert result is not None
        assert 0.85 < result < 1.0, f"Expected high BARTHAG, got {result}"

    def test_equal_efficiencies(self):
        """When adj_oe == adj_de, BARTHAG should be 0.5."""
        result = _compute_barthag(100.0, 100.0)
        assert result == pytest.approx(0.5, rel=1e-6)

    def test_none_input(self):
        assert _compute_barthag(None, 100.0) is None
        assert _compute_barthag(100.0, None) is None

    def test_exponent_is_11_5(self):
        """Verify the BARTHAG formula uses exponent 11.5."""
        oe, de = 110.0, 95.0
        expected = oe**11.5 / (oe**11.5 + de**11.5)
        result = _compute_barthag(oe, de)
        assert result == pytest.approx(expected, rel=1e-10)


class TestFeatureAssembly:
    """Test that build_features produces the expected columns."""

    @patch("src.features.load_lines")
    @patch("src.features.load_boxscores")
    @patch("src.features.load_efficiency_ratings")
    @patch("src.features.load_games")
    def test_feature_order_columns(self, mock_games, mock_ratings, mock_box, mock_lines):
        # Mock games
        mock_games.return_value = pd.DataFrame([{
            "gameId": 1,
            "homeTeamId": 100,
            "awayTeamId": 200,
            "homeScore": 75,
            "awayScore": 70,
            "neutralSite": False,
            "startDate": "2025-01-15",
        }])

        # Mock efficiency ratings (gold table schema with rating_date)
        mock_ratings.return_value = pd.DataFrame([
            {"teamId": 100, "rating_date": pd.Timestamp("2025-01-14"), "adj_oe": 115.0, "adj_de": 95.0, "adj_tempo": 70.0, "barthag": 0.89},
            {"teamId": 200, "rating_date": pd.Timestamp("2025-01-14"), "adj_oe": 105.0, "adj_de": 100.0, "adj_tempo": 68.0, "barthag": 0.58},
        ])

        # Mock boxscores with enough games for rolling averages
        box_rows = []
        for i in range(5):
            for tid, oid, is_home in [(100, 200, True), (200, 100, False)]:
                box_rows.append({
                    "gameid": i + 1,
                    "teamid": tid,
                    "opponentid": oid,
                    "ishometeam": is_home,
                    "startdate": f"2025-01-{10 + i:02d}",
                    "team_fg_made": 25.0 + i,
                    "team_fg_att": 60.0,
                    "team_3fg_made": 8.0,
                    "team_3fg_att": 20.0,
                    "team_ft_made": 15.0,
                    "team_ft_att": 20.0,
                    "team_reb_off": 10.0,
                    "team_reb_def": 25.0,
                    "opp_fg_made": 22.0,
                    "opp_fg_att": 55.0,
                    "opp_3fg_made": 6.0,
                    "opp_3fg_att": 18.0,
                    "opp_ft_made": 12.0,
                    "opp_ft_att": 16.0,
                    "opp_reb_off": 8.0,
                    "opp_reb_def": 22.0,
                })
        mock_box.return_value = pd.DataFrame(box_rows)

        df = build_features(2025)
        assert not df.empty

        # Feature matrix should match config.FEATURE_ORDER
        n_features = len(config.FEATURE_ORDER)
        feat = get_feature_matrix(df, feature_order=config.FEATURE_ORDER)
        assert feat.shape[1] == n_features, f"Expected {n_features} columns, got {feat.shape[1]}"
        assert list(feat.columns) == config.FEATURE_ORDER

    @patch("src.features.load_lines")
    @patch("src.features.load_boxscores")
    @patch("src.features.load_efficiency_ratings")
    @patch("src.features.load_games")
    def test_neutral_site_flag(self, mock_games, mock_ratings, mock_box, mock_lines):
        mock_games.return_value = pd.DataFrame([{
            "gameId": 1,
            "homeTeamId": 100,
            "awayTeamId": 200,
            "homeScore": 75,
            "awayScore": 70,
            "neutralSite": True,
            "startDate": "2025-03-15",
        }])
        mock_ratings.return_value = pd.DataFrame([
            {"teamId": 100, "rating_date": pd.Timestamp("2025-03-14"), "adj_oe": 115.0, "adj_de": 95.0, "adj_tempo": 70.0, "barthag": 0.89},
            {"teamId": 200, "rating_date": pd.Timestamp("2025-03-14"), "adj_oe": 105.0, "adj_de": 100.0, "adj_tempo": 68.0, "barthag": 0.58},
        ])
        mock_box.return_value = pd.DataFrame()

        df = build_features(2025)
        assert df.iloc[0]["neutral_site"] == 1
        assert df.iloc[0]["home_team_home"] == 0
        assert df.iloc[0]["away_team_home"] == 0
        assert df.iloc[0]["home_team_hca"] == 0.0

    @patch("src.features.load_lines")
    @patch("src.features.load_boxscores")
    @patch("src.features.load_efficiency_ratings")
    @patch("src.features.load_games")
    def test_home_team_home_when_not_neutral(self, mock_games, mock_ratings, mock_box, mock_lines):
        mock_games.return_value = pd.DataFrame([{
            "gameId": 1,
            "homeTeamId": 100,
            "awayTeamId": 200,
            "homeScore": 75,
            "awayScore": 70,
            "neutralSite": False,
            "startDate": "2025-01-15",
        }])
        mock_ratings.return_value = pd.DataFrame([
            {"teamId": 100, "rating_date": pd.Timestamp("2025-01-14"), "adj_oe": 115.0, "adj_de": 95.0, "adj_tempo": 70.0, "barthag": 0.89},
            {"teamId": 200, "rating_date": pd.Timestamp("2025-01-14"), "adj_oe": 105.0, "adj_de": 100.0, "adj_tempo": 68.0, "barthag": 0.58},
        ])
        mock_box.return_value = pd.DataFrame()

        df = build_features(2025)
        assert df.iloc[0]["neutral_site"] == 0
        assert df.iloc[0]["home_team_home"] == 1
        assert df.iloc[0]["away_team_home"] == 0
        # home_team_hca should exist (may be None for single-game mock data)
        assert "home_team_hca" in df.columns
        assert "home_team_efg_home_split" in df.columns
        assert "away_team_efg_away_split" in df.columns

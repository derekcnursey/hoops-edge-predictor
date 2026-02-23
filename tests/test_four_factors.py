"""Unit tests for four-factor computation."""

import numpy as np
import pandas as pd
import pytest

from src.four_factors import FOUR_FACTOR_COLS, compute_game_four_factors


def _make_boxscore(**overrides) -> pd.DataFrame:
    """Create a single-row boxscore DataFrame with sensible defaults."""
    defaults = {
        "gameid": 1,
        "teamid": 100,
        "opponentid": 200,
        "ishometeam": True,
        "startdate": "2025-01-15",
        # Team offense
        "team_fg_made": 25.0,
        "team_fg_att": 60.0,
        "team_3fg_made": 8.0,
        "team_3fg_att": 20.0,
        "team_ft_made": 15.0,
        "team_ft_att": 20.0,
        "team_reb_off": 10.0,
        "team_reb_def": 25.0,
        # Opponent
        "opp_fg_made": 22.0,
        "opp_fg_att": 55.0,
        "opp_3fg_made": 6.0,
        "opp_3fg_att": 18.0,
        "opp_ft_made": 12.0,
        "opp_ft_att": 16.0,
        "opp_reb_off": 8.0,
        "opp_reb_def": 22.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestComputeGameFourFactors:
    def test_basic_output_shape(self):
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        assert result.shape == (1, 4 + 13)  # 4 id cols + 13 stats

    def test_all_four_factor_columns_present(self):
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        for col in FOUR_FACTOR_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_eff_fg_pct(self):
        """eff_fg_pct = (FGM + 0.5 * 3PM) / FGA = (25 + 0.5*8) / 60 = 29/60."""
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        expected = (25 + 0.5 * 8) / 60
        assert result["eff_fg_pct"].iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_ft_pct(self):
        """ft_pct = FTM / FTA = 15/20 = 0.75."""
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        assert result["ft_pct"].iloc[0] == pytest.approx(0.75, rel=1e-6)

    def test_ft_rate(self):
        """ft_rate = FTA / FGA = 20/60."""
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        assert result["ft_rate"].iloc[0] == pytest.approx(20 / 60, rel=1e-6)

    def test_3pt_rate(self):
        """3pt_rate = 3PA / FGA = 20/60."""
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        assert result["three_pt_rate"].iloc[0] == pytest.approx(20 / 60, rel=1e-6)

    def test_3p_pct(self):
        """3p_pct = 3PM / 3PA = 8/20."""
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        assert result["three_p_pct"].iloc[0] == pytest.approx(8 / 20, rel=1e-6)

    def test_off_rebound_pct(self):
        """off_rebound_pct = team_off_reb / (team_off_reb + opp_def_reb) = 10 / (10+22)."""
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        assert result["off_rebound_pct"].iloc[0] == pytest.approx(10 / 32, rel=1e-6)

    def test_def_rebound_pct(self):
        """def_rebound_pct = team_def_reb / (team_def_reb + opp_off_reb) = 25 / (25+8)."""
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        assert result["def_rebound_pct"].iloc[0] == pytest.approx(25 / 33, rel=1e-6)

    def test_def_eff_fg_pct(self):
        """def_eff_fg_pct = (opp_FGM + 0.5*opp_3PM) / opp_FGA = (22+3)/55."""
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        expected = (22 + 0.5 * 6) / 55
        assert result["def_eff_fg_pct"].iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_def_ft_rate(self):
        """def_ft_rate = opp_FTA / opp_FGA = 16/55."""
        box = _make_boxscore()
        result = compute_game_four_factors(box)
        assert result["def_ft_rate"].iloc[0] == pytest.approx(16 / 55, rel=1e-6)

    def test_zero_denominator_returns_nan(self):
        """If FGA is 0, ratios should be NaN, not error."""
        box = _make_boxscore(team_fg_att=0.0, team_3fg_att=0.0)
        result = compute_game_four_factors(box)
        assert np.isnan(result["eff_fg_pct"].iloc[0])
        assert np.isnan(result["ft_rate"].iloc[0])
        assert np.isnan(result["three_pt_rate"].iloc[0])

    def test_multiple_games(self):
        """Verify computation works for multiple rows."""
        box = pd.concat([
            _make_boxscore(gameid=1, teamid=100),
            _make_boxscore(gameid=2, teamid=100, team_fg_made=30.0),
        ], ignore_index=True)
        result = compute_game_four_factors(box)
        assert len(result) == 2
        # Second game has different eff_fg_pct
        assert result["eff_fg_pct"].iloc[1] == pytest.approx((30 + 0.5 * 8) / 60, rel=1e-6)

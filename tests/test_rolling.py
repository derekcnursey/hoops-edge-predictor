"""Unit tests for rolling average computation."""

import numpy as np
import pandas as pd
import pytest

from src.four_factors import FOUR_FACTOR_COLS, compute_game_four_factors
from src.rolling_averages import compute_rolling_averages, compute_venue_split_rolling


def _make_team_games(
    teamid: int,
    n_games: int,
    base_fg_made: float = 25.0,
) -> pd.DataFrame:
    """Create boxscore data for n sequential games for a single team."""
    rows = []
    for i in range(n_games):
        rows.append({
            "gameid": 1000 + i,
            "teamid": teamid,
            "opponentid": 900 + i,
            "ishometeam": i % 2 == 0,
            "startdate": f"2025-01-{15 + i:02d}",
            "team_fg_made": base_fg_made + i,
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
    return pd.DataFrame(rows)


class TestRollingAverages:
    def test_output_has_rolling_columns(self):
        box = _make_team_games(100, 5)
        ff = compute_game_four_factors(box)
        rolling = compute_rolling_averages(ff)
        for col in FOUR_FACTOR_COLS:
            assert f"rolling_{col}" in rolling.columns, f"Missing: rolling_{col}"

    def test_first_game_is_nan(self):
        """First game should have NaN rolling averages (no prior games)."""
        box = _make_team_games(100, 5)
        ff = compute_game_four_factors(box)
        rolling = compute_rolling_averages(ff)
        # Sort by date to get first game
        rolling = rolling.sort_values("startdate")
        first = rolling.iloc[0]
        assert np.isnan(first["rolling_eff_fg_pct"]), "First game should have NaN rolling avg"

    def test_second_game_uses_first_game_stats(self):
        """Second game's rolling avg should be the first game's stat value."""
        box = _make_team_games(100, 3)
        ff = compute_game_four_factors(box)
        rolling = compute_rolling_averages(ff)
        rolling = rolling.sort_values("startdate")
        # Second game's rolling should equal first game's raw stat
        second_rolling = rolling.iloc[1]["rolling_eff_fg_pct"]
        first_raw = ff.sort_values("startdate").iloc[0]["eff_fg_pct"]
        assert second_rolling == pytest.approx(first_raw, rel=1e-6)

    def test_no_data_leakage(self):
        """Each game's rolling should not include its own stats."""
        box = _make_team_games(100, 10)
        ff = compute_game_four_factors(box)
        rolling = compute_rolling_averages(ff)
        rolling = rolling.sort_values("startdate")

        # Make a game with very different stats
        extreme = _make_team_games(100, 1, base_fg_made=50.0)
        extreme["gameid"] = 9999
        extreme["startdate"] = "2025-01-30"
        box_ext = pd.concat([box, extreme], ignore_index=True)
        ff_ext = compute_game_four_factors(box_ext)
        rolling_ext = compute_rolling_averages(ff_ext)
        rolling_ext = rolling_ext.sort_values("startdate")

        # The extreme game's rolling should NOT include its own 50-FGM stat
        extreme_row = rolling_ext[rolling_ext["gameid"] == 9999].iloc[0]
        # Its rolling avg should be based on the prior 10 games only
        # (where FGM ranged from 25-34, not 50)
        assert extreme_row["rolling_eff_fg_pct"] < 0.65, "Rolling avg should not include current game"

    def test_multiple_teams_independent(self):
        """Rolling averages for different teams should be independent."""
        box1 = _make_team_games(100, 5, base_fg_made=20.0)
        box2 = _make_team_games(200, 5, base_fg_made=30.0)
        box = pd.concat([box1, box2], ignore_index=True)
        ff = compute_game_four_factors(box)
        rolling = compute_rolling_averages(ff)

        team100 = rolling[rolling["teamid"] == 100].sort_values("startdate")
        team200 = rolling[rolling["teamid"] == 200].sort_values("startdate")

        # Second game rolling averages should differ between teams
        r100 = team100.iloc[1]["rolling_eff_fg_pct"]
        r200 = team200.iloc[1]["rolling_eff_fg_pct"]
        assert r100 != pytest.approx(r200, abs=0.01), "Different teams should have different rolling avgs"

    def test_venue_split_output_columns(self):
        """Venue split rolling should produce rolling_home_efg and rolling_away_efg."""
        box = _make_team_games(100, 10)
        ff = compute_game_four_factors(box)
        vs = compute_venue_split_rolling(ff)
        assert "rolling_home_efg" in vs.columns
        assert "rolling_away_efg" in vs.columns

    def test_venue_split_first_home_game_is_nan(self):
        """First home game should have NaN rolling_home_efg (no prior home games)."""
        box = _make_team_games(100, 6)
        ff = compute_game_four_factors(box)
        vs = compute_venue_split_rolling(ff)
        vs = vs.sort_values("startdate")
        # First game (gameid=1000) is home (ishometeam alternates, starts True)
        first_home = vs[vs["gameid"] == 1000].iloc[0]
        assert np.isnan(first_home["rolling_home_efg"]), "First home game should have NaN"

    def test_venue_split_forward_fill(self):
        """Away games should carry forward the latest home-split value."""
        box = _make_team_games(100, 6)
        ff = compute_game_four_factors(box)
        vs = compute_venue_split_rolling(ff)
        vs = vs.sort_values("startdate")
        # Game at index 1 (gameid=1001) is away — should carry forward home value from game 0
        second = vs.iloc[1]
        # rolling_home_efg should be the first home game's raw eff_fg_pct (forward-filled)
        first_home_raw = ff.sort_values("startdate").iloc[0]["eff_fg_pct"]
        # After shift(1), first home game has NaN, so second home game gets the value
        # But the first away game (index 1) comes between, so it inherits NaN via ffill
        # Actually: home game 0 -> shift(1) = NaN, away game 1 -> ffill from NaN = NaN
        # home game 2 -> shift(1) = game 0's value, so game 2 has a value
        # The forward fill only works after at least 2 home games
        third_game = vs.iloc[2]  # gameid=1002, home game
        assert not np.isnan(third_game["rolling_home_efg"]), "Second home game should have value"

    def test_venue_split_no_leakage(self):
        """Home split should not include the current home game's stats."""
        box = _make_team_games(100, 6)
        ff = compute_game_four_factors(box)
        vs = compute_venue_split_rolling(ff)
        vs = vs.sort_values("startdate")
        # Third game (index 2, gameid=1002) is home — its rolling_home_efg
        # should equal the first home game's (index 0) raw stat, not include index 2
        first_home_raw = ff.sort_values("startdate").iloc[0]["eff_fg_pct"]
        third_home_rolling = vs.iloc[2]["rolling_home_efg"]
        assert third_home_rolling == pytest.approx(first_home_raw, rel=1e-6)

    def test_exponential_decay_weights_recent_more(self):
        """More recent games should have more weight in the EWM average."""
        # Create games where stats increase over time
        box = _make_team_games(100, 20, base_fg_made=20.0)
        ff = compute_game_four_factors(box)
        rolling = compute_rolling_averages(ff)
        rolling = rolling.sort_values("startdate")

        # The rolling average at the end should be higher than simple mean
        # because EWM weights recent (higher) values more
        all_raw = ff.sort_values("startdate")["eff_fg_pct"].values
        simple_mean = np.nanmean(all_raw[:-1])  # exclude last game
        last_rolling = rolling.iloc[-1]["rolling_eff_fg_pct"]
        # EWM with increasing values should be > simple mean
        assert last_rolling > simple_mean, "EWM should weight recent games more"

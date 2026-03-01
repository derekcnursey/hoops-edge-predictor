"""Unit tests for corrected efficiency features."""

import numpy as np
import pandas as pd
import pytest

from src.corrected_efficiencies import (
    CORRECTED_EFFICIENCY_COLS,
    compute_corrected_efficiencies,
)


def _make_adv_stats(n_dates: int = 10, n_teams: int = 4) -> pd.DataFrame:
    """Create synthetic advanced stats for testing.

    Generates n_dates game dates with n_teams/2 games per date (each game
    produces 2 team rows).
    """
    rows = []
    game_id = 1
    teams = list(range(100, 100 + n_teams))

    for d in range(n_dates):
        date_str = f"2025-01-{d + 1:02d}T19:00:00Z"
        # Pair teams into games
        for g in range(0, n_teams, 2):
            t1, t2 = teams[g], teams[g + 1]

            for tid in [t1, t2]:
                # Vary shot distribution by team and date for realism
                np.random.seed(tid * 100 + d)
                rim_fga = np.random.randint(8, 20)
                mid_fga = np.random.randint(5, 15)
                three_fga = np.random.randint(10, 25)

                rim_fgm = int(rim_fga * np.random.uniform(0.50, 0.70))
                mid_fgm = int(mid_fga * np.random.uniform(0.30, 0.50))
                three_fgm = int(three_fga * np.random.uniform(0.25, 0.40))

                total_fga = rim_fga + mid_fga + three_fga

                rows.append({
                    "gameid": game_id,
                    "teamid": tid,
                    "opponentid": t2 if tid == t1 else t1,
                    "startdate": date_str,
                    "rim_fga": rim_fga,
                    "rim_fgm": rim_fgm,
                    "mid_fga": mid_fga,
                    "mid_fgm": mid_fgm,
                    "three_fga": three_fga,
                    "three_fgm": three_fgm,
                    "rim_rate": rim_fga / total_fga,
                    "mid_range_rate": mid_fga / total_fga,
                    "rim_fg_pct": rim_fgm / rim_fga if rim_fga > 0 else 0,
                    "three_pt_fg_pct": three_fgm / three_fga if three_fga > 0 else 0,
                    "two_pt_fg_pct": (rim_fgm + mid_fgm) / (rim_fga + mid_fga)
                        if (rim_fga + mid_fga) > 0 else 0,
                    "def_rim_rate": np.random.uniform(0.20, 0.40),
                    "def_mid_range_rate": np.random.uniform(0.15, 0.30),
                })
            game_id += 1

    return pd.DataFrame(rows)


class TestComputeCorrectedEfficiencies:
    def test_output_columns(self):
        df = _make_adv_stats()
        result = compute_corrected_efficiencies(df)
        for col in CORRECTED_EFFICIENCY_COLS:
            assert col in result.columns, f"Missing column: {col}"
        assert "gameid" in result.columns
        assert "teamid" in result.columns
        assert "startdate" in result.columns

    def test_first_date_has_nan(self):
        """First game date should have NaN — no league averages available yet."""
        df = _make_adv_stats()
        result = compute_corrected_efficiencies(df)
        first_date = pd.to_datetime(result["startdate"]).min()
        first_rows = result[pd.to_datetime(result["startdate"]) == first_date]
        for col in CORRECTED_EFFICIENCY_COLS:
            assert first_rows[col].isna().all(), (
                f"{col} should be NaN on first date but has values"
            )

    def test_second_date_has_values(self):
        """After first date, league averages exist so values should be non-NaN."""
        df = _make_adv_stats()
        result = compute_corrected_efficiencies(df)
        dates = sorted(pd.to_datetime(result["startdate"]).unique())
        if len(dates) >= 2:
            second_rows = result[pd.to_datetime(result["startdate"]) == dates[1]]
            assert second_rows["shot_quality_oe"].notna().all()
            assert second_rows["shot_quality_de"].notna().all()

    def test_shot_quality_oe_range(self):
        """Expected pts/FGA should be in a reasonable basketball range."""
        df = _make_adv_stats()
        result = compute_corrected_efficiencies(df)
        valid = result["shot_quality_oe"].dropna()
        assert len(valid) > 0
        # pts/FGA: rim ~1.3, mid ~0.8, three ~1.05 → weighted avg ~0.8-1.4
        assert valid.min() >= 0.5, f"shot_quality_oe min={valid.min()}"
        assert valid.max() <= 2.0, f"shot_quality_oe max={valid.max()}"

    def test_shot_quality_de_range(self):
        """Defensive shot quality should be in same range as offensive."""
        df = _make_adv_stats()
        result = compute_corrected_efficiencies(df)
        valid = result["shot_quality_de"].dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.5
        assert valid.max() <= 2.0

    def test_shot_quality_vs_actual_mean_near_zero(self):
        """Mean luck across all teams over a season should be near zero."""
        df = _make_adv_stats(n_dates=30, n_teams=8)
        result = compute_corrected_efficiencies(df)
        valid = result["shot_quality_vs_actual"].dropna()
        assert len(valid) > 0
        # Mean should be close to zero (not exactly, due to causal lag)
        assert abs(valid.mean()) < 0.15, (
            f"Mean shot_quality_vs_actual = {valid.mean()}, expected near 0"
        )

    def test_causal_ordering(self):
        """League averages should only use data from prior dates."""
        df = _make_adv_stats(n_dates=5, n_teams=2)
        result = compute_corrected_efficiencies(df)

        dates = sorted(pd.to_datetime(result["startdate"]).unique())
        # Values on date 2 should use only date 1 data
        # Values on date 3 should use dates 1-2 data
        # This means shot_quality_oe might differ between dates even if
        # team shot distribution is similar, because league averages change.
        d2_vals = result[pd.to_datetime(result["startdate"]) == dates[1]]["shot_quality_oe"]
        d3_vals = result[pd.to_datetime(result["startdate"]) == dates[2]]["shot_quality_oe"]
        # Both should be non-NaN
        assert d2_vals.notna().all()
        assert d3_vals.notna().all()

    def test_missing_columns_returns_empty(self):
        """If required columns are missing, return empty DataFrame with schema."""
        df = pd.DataFrame({"gameid": [1], "teamid": [100], "startdate": ["2025-01-01"]})
        result = compute_corrected_efficiencies(df)
        assert len(result) == 0
        for col in CORRECTED_EFFICIENCY_COLS:
            assert col in result.columns

    def test_row_count_preserved(self):
        """Output should have same number of rows as input."""
        df = _make_adv_stats(n_dates=5, n_teams=4)
        result = compute_corrected_efficiencies(df)
        assert len(result) == len(df)

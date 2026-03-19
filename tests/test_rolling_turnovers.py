import pandas as pd

from src.rolling_averages import compute_rolling_turnovers


def test_compute_rolling_turnovers_preserves_startdate_for_asof_lookup():
    box = pd.DataFrame(
        [
            {
                "gameid": 1,
                "teamid": 10,
                "startdate": "2025-11-01T00:00:00Z",
                "team_tov_ratio": 0.2,
                "opp_tov_ratio": 0.18,
            },
            {
                "gameid": 2,
                "teamid": 10,
                "startdate": "2025-11-03T00:00:00Z",
                "team_tov_ratio": 0.25,
                "opp_tov_ratio": 0.16,
            },
        ]
    )

    out = compute_rolling_turnovers(box)

    assert "startdate" in out.columns
    assert out["startdate"].tolist() == box["startdate"].tolist()
    assert pd.isna(out.loc[out["gameid"] == 1, "rolling_tov_rate"]).all()
    assert out.loc[out["gameid"] == 2, "rolling_tov_rate"].notna().all()

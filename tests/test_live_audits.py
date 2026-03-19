import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.live_audits import audit_hrb_lines, audit_live_feature_drift, audit_ratings_asof


def _fit_test_scaler(columns: list[str]) -> StandardScaler:
    base = np.tile(np.arange(len(columns), dtype=float), (30, 1))
    scaler = StandardScaler().fit(base)
    return scaler


def test_live_feature_drift_flags_mean_fill_and_drift():
    features = pd.DataFrame(
        {
            "home_team_adj_oe": [20.0, 20.5, np.nan],
            "away_team_adj_oe": [1.0, 1.0, 1.0],
            "home_team_adj_de": [2.0, 2.0, 2.0],
            "away_team_adj_de": [3.0, 3.0, 3.0],
        }
    )
    scaler = _fit_test_scaler(list(features.columns))

    report = audit_live_feature_drift(
        features,
        scaler,
        features.columns.tolist(),
        ["home_team_adj_oe", "away_team_adj_oe", "home_team_adj_de", "away_team_adj_de"],
    )

    assert report.info
    assert any("worst mean-z drift" in line for line in report.info)
    assert any("feature(s) exceed mean-z drift" in msg for msg in report.warnings)
    assert any("critical feature fill present" in msg for msg in report.warnings)


def test_ratings_asof_reports_missing_and_stale_rows():
    slate = pd.DataFrame(
        [
            {
                "gameId": 1,
                "awayTeam": "BYU",
                "homeTeam": "Houston",
                "awayTeamId": 20,
                "homeTeamId": 10,
                "startDate": "2026-03-14T18:00:00Z",
            },
            {
                "gameId": 2,
                "awayTeam": "Iona",
                "homeTeam": "Akron",
                "awayTeamId": 40,
                "homeTeamId": 30,
                "startDate": "2026-03-14T20:00:00Z",
            },
        ]
    )
    ratings = pd.DataFrame(
        [
            {"teamId": 10, "rating_date": "2026-03-13", "adj_oe": 1, "adj_de": 1, "adj_tempo": 65, "barthag": 0.8},
            {"teamId": 20, "rating_date": "2026-03-09", "adj_oe": 1, "adj_de": 1, "adj_tempo": 66, "barthag": 0.7},
            {"teamId": 30, "rating_date": "2026-03-13", "adj_oe": 1, "adj_de": 1, "adj_tempo": 67, "barthag": 0.6},
        ]
    )

    report = audit_ratings_asof(slate, ratings)

    assert any("median age" in line for line in report.info)
    assert any("stale ratings older than" in msg for msg in report.warnings)
    assert any("missing away rating history" in msg for msg in report.errors)


def test_hrb_line_audit_flags_sign_conflict_and_peer_dislocation():
    slate = pd.DataFrame(
        [
            {
                "gameId": 1,
                "awayTeam": "BYU",
                "homeTeam": "Houston",
                "startDate": "2026-03-14T18:00:00Z",
            }
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "gameId": 1,
                "provider": "Hard Rock Bet",
                "awayTeam": "BYU",
                "homeTeam": "Houston",
                "spread": 6.5,
                "homeMoneyline": -220,
                "awayMoneyline": 180,
                "overUnder": 139.5,
            },
            {
                "gameId": 1,
                "provider": "Draft Kings",
                "awayTeam": "BYU",
                "homeTeam": "Houston",
                "spread": -1.0,
                "homeMoneyline": -120,
                "awayMoneyline": 100,
                "overUnder": 139.5,
            },
        ]
    )
    preferred = pd.DataFrame(
        [
            {
                "gameId": 1,
                "provider": "Hard Rock Bet",
                "book_spread": 6.5,
                "home_moneyline": -220,
                "away_moneyline": 180,
            }
        ]
    )

    report = audit_hrb_lines(slate, lines, preferred)

    assert any("selected as preferred" in line for line in report.info)
    assert any("Houston" in msg for msg in report.errors)
    assert any(f">= {config.HRB_SPREAD_DISLOCATION_WARN:.1f}" in msg for msg in report.warnings)

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src import hrb_odds


def test_current_cbb_season_rollover():
    assert hrb_odds.current_cbb_season(datetime(2026, 3, 13)) == 2026
    assert hrb_odds.current_cbb_season(datetime(2026, 11, 10)) == 2027


def test_normalize_team_name_handles_common_sportsbook_variants():
    assert hrb_odds._normalize_team_name("1 UConn") == "connecticut"
    assert hrb_odds._normalize_team_name("Southern University") == "southern"
    assert hrb_odds._normalize_team_name("Texas Arlington") == "utarlington"
    assert hrb_odds._normalize_team_name("Saint Joseph's") == "stjosephs"


def test_team_key_variants_include_common_alias_forms():
    assert "usf" in hrb_odds._team_key_variants("South Florida")
    assert "southflorida" in hrb_odds._team_key_variants("USF")
    assert "saintlouis" in hrb_odds._team_key_variants("Saint Louis")


def test_match_event_to_game_uses_alias_variants():
    start = pd.Timestamp.now(tz="UTC").floor("min")
    schedule = pd.DataFrame(
        [
            {
                "gameId": 1,
                "awayTeam": "Dayton",
                "homeTeam": "Saint Louis",
                "startDate": start,
                "away_keys": hrb_odds._team_key_variants("Dayton"),
                "home_keys": hrb_odds._team_key_variants("Saint Louis"),
            },
            {
                "gameId": 2,
                "awayTeam": "Charlotte",
                "homeTeam": "South Florida",
                "startDate": start + pd.Timedelta(hours=2),
                "away_keys": hrb_odds._team_key_variants("Charlotte"),
                "home_keys": hrb_odds._team_key_variants("South Florida"),
            },
        ]
    )
    event = {
        "eventTime": (start + pd.Timedelta(hours=2)).value / 1_000_000,
        "participants": [
            {"name": "USF", "position": 0},
            {"name": "5 Charlotte", "position": 1},
        ],
    }

    matched = hrb_odds._match_event_to_game(event, schedule)

    assert matched is not None
    assert int(matched.game["gameId"]) == 2
    assert matched.slot_to_side == {"A": "home", "B": "away"}


def test_build_line_row_maps_hrb_event_to_home_perspective_lines():
    event = {
        "id": "evt-1",
        "eventTime": 1773511200000.0,
        "participants": [
            {"name": "2 Harvard", "position": 0},
            {"name": "3 Pennsylvania", "position": 1},
        ],
        "markets": [
            {
                "type": "BASKETBALL:FTOT:ML",
                "selection": [
                    {"name": "2 Harvard", "type": "A", "rootIdx": 60},
                    {"name": "3 Pennsylvania", "type": "B", "rootIdx": 80},
                ],
            },
            {
                "type": "BASKETBALL:FTOT:SPRD",
                "subtype": "M#-3.5",
                "selection": [
                    {"name": "2 Harvard -3.5", "type": "A", "rootIdx": 69},
                    {"name": "3 Pennsylvania +3.5", "type": "B", "rootIdx": 71},
                ],
            },
            {
                "type": "BASKETBALL:FTOT:OU",
                "subtype": "M#144.5",
                "selection": [
                    {"name": "Over 144.5", "type": "Over", "rootIdx": 70},
                    {"name": "Under 144.5", "type": "Under", "rootIdx": 70},
                ],
            },
        ],
    }
    matched = hrb_odds._MatchedGame(
        game=pd.Series(
            {
                "gameId": 372320,
                "awayTeam": "Pennsylvania",
                "homeTeam": "Harvard",
                "startDate": pd.Timestamp("2026-03-13T18:00:00Z"),
            }
        ),
        slot_to_side={"A": "home", "B": "away"},
    )
    ladder = {
        60: {"decimal": 1.8, "moneyline": -125},
        69: {"decimal": 1.91, "moneyline": -110},
        70: {"decimal": 1.91, "moneyline": -110},
        71: {"decimal": 1.91, "moneyline": -110},
        80: {"decimal": 2.1, "moneyline": 110},
    }

    row = hrb_odds._build_line_row(event, matched, ladder)

    assert row is not None
    assert row["gameId"] == 372320
    assert row["provider"] == "Hard Rock Bet"
    assert row["spread"] == -3.5
    assert row["overUnder"] == 144.5
    assert row["homeMoneyline"] == -125
    assert row["awayMoneyline"] == 110

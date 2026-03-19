from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.tournament_adjustments import (
    add_tournament_market_display_columns,
    is_ncaa_tournament,
)


def test_is_ncaa_tournament_prefers_explicit_tournament_flag() -> None:
    frame = pd.DataFrame(
        {
            "tournament": ["NCAA", None, None],
            "gameType": ["TRNMNT", "TRNMNT", "TRNMNT"],
            "conferenceGame": [False, False, True],
            "neutralSite": [True, True, True],
            "startDate": [
                "2025-03-20T18:00:00Z",
                "2025-03-19T18:00:00Z",
                "2025-03-12T18:00:00Z",
            ],
        }
    )

    mask = is_ncaa_tournament(frame)
    assert mask.tolist() == [True, True, False]


def test_add_tournament_market_display_columns_only_changes_ncaa_rows(monkeypatch) -> None:
    monkeypatch.setattr(config, "NCAA_TOURNAMENT_MARKET_BLEND_ENABLED", True)
    monkeypatch.setattr(config, "NCAA_TOURNAMENT_MARKET_WEIGHT", 0.4)

    frame = pd.DataFrame(
        {
            "predicted_spread": [10.0, 5.0],
            "book_spread": [-12.0, -7.0],
            "tournament": ["NCAA", None],
            "gameType": ["TRNMNT", "STD"],
            "conferenceGame": [False, False],
            "neutralSite": [True, False],
            "startDate": ["2025-03-21T18:00:00Z", "2025-02-01T18:00:00Z"],
        }
    )

    out = add_tournament_market_display_columns(frame)

    assert out.loc[0, "display_predicted_spread"] == 10.8
    assert out.loc[0, "display_model_spread"] == -10.8
    assert out.loc[0, "display_edge_home_points"] == pytest.approx(-1.2)
    assert out.loc[1, "display_predicted_spread"] == 5.0
    assert out.loc[1, "display_model_spread"] == -5.0

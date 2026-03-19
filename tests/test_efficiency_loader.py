from __future__ import annotations

import pandas as pd

from src.features import _dedupe_efficiency_ratings


def test_dedupe_efficiency_ratings_prefers_plausible_tempo_row() -> None:
    df = pd.DataFrame(
        {
            "teamId": [1, 1, 2],
            "rating_date": pd.to_datetime(["2026-03-11", "2026-03-11", "2026-03-11"]),
            "adj_oe": [120.0, 121.0, 110.0],
            "adj_de": [95.0, 96.0, 101.0],
            "adj_tempo": [121.2, 60.3, 68.0],
            "barthag": [0.9, 0.91, 0.7],
        }
    )

    out = _dedupe_efficiency_ratings(df)

    assert len(out) == 2
    chosen = out[out["teamId"] == 1].iloc[0]
    assert chosen["adj_tempo"] == 60.3
    assert chosen["adj_oe"] == 121.0


def test_dedupe_efficiency_ratings_prefers_smaller_tempo_if_all_implausible() -> None:
    df = pd.DataFrame(
        {
            "teamId": [1, 1],
            "rating_date": pd.to_datetime(["2026-03-11", "2026-03-11"]),
            "adj_oe": [120.0, 121.0],
            "adj_de": [95.0, 96.0],
            "adj_tempo": [128.0, 192.0],
            "barthag": [0.9, 0.91],
        }
    )

    out = _dedupe_efficiency_ratings(df)

    assert len(out) == 1
    chosen = out.iloc[0]
    assert chosen["adj_tempo"] == 128.0
    assert chosen["adj_oe"] == 120.0

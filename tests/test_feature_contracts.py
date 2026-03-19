from __future__ import annotations

from src import config


def test_swap_safe_v2_feature_order_shape_and_names() -> None:
    current = config.FEATURE_ORDER
    v2 = config.FEATURE_ORDER_SWAP_SAFE_V2

    assert len(current) == len(v2) == 53

    assert "home_team_hca" in current
    assert "home_team_hca" not in v2
    assert "venue_edge" in v2

    assert "home_opp_ft_rate" in current
    assert "home_opp_ft_rate" not in v2
    assert "home_def_ft_rate" in v2

    assert "home_team_efg_home_split" in current
    assert "away_team_efg_away_split" in current
    assert "home_team_efg_slot_split" in v2
    assert "away_team_efg_slot_split" in v2

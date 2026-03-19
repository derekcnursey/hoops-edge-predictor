from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.slot_augmentation import (
    augment_swapped_slot_training,
    audit_feature_order,
    swap_feature_frame,
)


def test_feature_order_audit_flags_unswappable_hca():
    feature_order = [
        "neutral_site",
        "home_team_adj_oe",
        "away_team_adj_oe",
        "home_opp_ft_rate",
        "away_def_ft_rate",
        "home_team_hca",
        "rest_advantage",
    ]
    audit = audit_feature_order(feature_order)

    assert ("home_team_adj_oe", "away_team_adj_oe") in audit.generic_pairs
    assert ("home_opp_ft_rate", "away_def_ft_rate") in audit.explicit_pairs
    assert "home_team_hca" in audit.unswappable
    assert "rest_advantage" in audit.negated
    assert "neutral_site" in audit.globals


def test_swap_feature_frame_is_involution_for_neutral_rows():
    df = pd.DataFrame(
        [
            {
                "neutral_site": 1.0,
                "home_team_adj_oe": 120.0,
                "away_team_adj_oe": 110.0,
                "home_opp_ft_rate": 0.22,
                "away_def_ft_rate": 0.18,
                "home_team_efg_home_split": 0.54,
                "away_team_efg_away_split": 0.49,
                "home_team_hca": 0.0,
                "rest_advantage": 2.0,
            }
        ]
    )
    feature_order = list(df.columns)

    swapped = swap_feature_frame(df, feature_order)
    roundtrip = swap_feature_frame(swapped, feature_order)

    pd.testing.assert_frame_equal(roundtrip[feature_order], df[feature_order])


def test_augment_swapped_slot_training_negates_targets_and_flips_home_win():
    feature_df = pd.DataFrame(
        [
            {
                "neutral_site": 1.0,
                "home_team_adj_oe": 120.0,
                "away_team_adj_oe": 110.0,
                "home_team_hca": 0.0,
                "rest_advantage": 1.5,
            },
            {
                "neutral_site": 0.0,
                "home_team_adj_oe": 118.0,
                "away_team_adj_oe": 112.0,
                "home_team_hca": 3.0,
                "rest_advantage": -1.0,
            },
        ]
    )
    spread = np.array([4.5, 2.0], dtype=np.float32)
    home_win = np.array([1.0, 1.0], dtype=np.float32)
    eligible_mask = np.array([True, False])

    aug_df, aug_spread, aug_home_win = augment_swapped_slot_training(
        feature_df,
        spread,
        home_win=home_win,
        eligible_mask=eligible_mask,
    )

    assert len(aug_df) == 3
    assert np.allclose(aug_spread, np.array([4.5, 2.0, -4.5], dtype=np.float32))
    assert np.allclose(aug_home_win, np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert aug_df.iloc[2]["neutral_site"] == 1.0
    assert aug_df.iloc[2]["home_team_hca"] == 0.0
    assert aug_df.iloc[2]["rest_advantage"] == -1.5


def test_swap_safe_v2_contract_has_no_unswappable_fields():
    audit = audit_feature_order(config.FEATURE_ORDER_SWAP_SAFE_V2)
    assert audit.unswappable == []
    assert "venue_edge" in audit.negated


def test_swap_feature_frame_supports_all_games_for_swap_safe_v2():
    cols = config.FEATURE_ORDER_SWAP_SAFE_V2
    row = {col: 0.0 for col in cols}
    row.update(
        {
            "neutral_site": 0.0,
            "home_team_adj_oe": 110.0,
            "away_team_adj_oe": 100.0,
            "home_def_ft_rate": 0.21,
            "away_def_ft_rate": 0.18,
            "venue_edge": 2.7,
            "home_team_efg_slot_split": 0.54,
            "away_team_efg_slot_split": 0.49,
            "rest_advantage": 1.0,
        }
    )
    df = pd.DataFrame([row])
    swapped = swap_feature_frame(df, cols, neutral_only=False)

    assert float(swapped.loc[0, "home_team_adj_oe"]) == 100.0
    assert float(swapped.loc[0, "away_team_adj_oe"]) == 110.0
    assert float(swapped.loc[0, "home_def_ft_rate"]) == 0.18
    assert float(swapped.loc[0, "away_def_ft_rate"]) == 0.21
    assert float(swapped.loc[0, "venue_edge"]) == -2.7
    assert float(swapped.loc[0, "home_team_efg_slot_split"]) == 0.49
    assert float(swapped.loc[0, "away_team_efg_slot_split"]) == 0.54
    assert float(swapped.loc[0, "rest_advantage"]) == -1.0
    assert float(swapped.loc[0, "neutral_site"]) == 0.0

    swapped_twice = swap_feature_frame(swapped, cols, neutral_only=False)
    pd.testing.assert_frame_equal(swapped_twice.reset_index(drop=True), df.reset_index(drop=True))

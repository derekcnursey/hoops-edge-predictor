from __future__ import annotations

import pandas as pd
import pytest

from src.ncaa_synthetic_fallback import (
    direct_margin_fit_for_round,
    simple_efficiency_margin,
    synthetic_fallback_margin,
)


def _team(adj_oe: float, adj_de: float, adj_tempo: float) -> pd.Series:
    return pd.Series(
        {
            "adj_oe": adj_oe,
            "adj_de": adj_de,
            "adj_tempo": adj_tempo,
        }
    )


def test_simple_efficiency_margin_uses_net_and_average_tempo() -> None:
    team_a = _team(120.0, 95.0, 70.0)
    team_b = _team(110.0, 100.0, 66.0)

    margin = simple_efficiency_margin(team_a, team_b)

    assert margin == pytest.approx((25.0 - 10.0) * (68.0 / 100.0))


def test_direct_margin_fit_for_round_uses_round64_gold_coefficients() -> None:
    fit, label = direct_margin_fit_for_round("gold", "Round of 64")

    assert fit is not None
    assert fit.intercept == pytest.approx(1.517813)
    assert fit.coef_simple_eff == pytest.approx(1.137470)
    assert label == "gold_direct_fitted_round64_v1"


def test_synthetic_fallback_margin_keeps_torvik_identity() -> None:
    team_a = _team(118.0, 96.0, 69.0)
    team_b = _team(108.0, 101.0, 67.0)

    mapped, simple, label = synthetic_fallback_margin(
        team_a,
        team_b,
        ratings_source="torvik",
        round_label="Elite 8",
    )

    assert mapped == pytest.approx(simple)
    assert label == "torvik_simple_identity_v1"


def test_synthetic_fallback_margin_uses_fitted_gold_mapping() -> None:
    team_a = _team(121.0, 94.0, 71.0)
    team_b = _team(107.0, 103.0, 67.0)

    mapped, simple, label = synthetic_fallback_margin(
        team_a,
        team_b,
        ratings_source="team_adjusted_efficiencies_no_garbage_softkeep25_priorreg_k5_v1",
        round_label="Sweet 16",
    )

    expected_simple = simple_efficiency_margin(team_a, team_b)
    expected_mapped = 0.953555 + 1.082141 * expected_simple

    assert simple == pytest.approx(expected_simple)
    assert mapped == pytest.approx(expected_mapped)
    assert label == "gold_direct_fitted_ncaa_v1"

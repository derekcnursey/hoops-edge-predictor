from src.ml_odds import (
    fair_american_odds,
    mu_sigma_home_win_prob,
    site_home_win_prob_from_mu_sigma,
    stabilize_sigma_for_ml,
)


def test_cap14_sigma_stabilization():
    assert stabilize_sigma_for_ml(12.0, mode="cap14") == 12.0
    assert stabilize_sigma_for_ml(20.0, mode="cap14") == 14.0


def test_mu_sigma_probability_uses_cap14_default():
    raw = mu_sigma_home_win_prob(7.0, 20.0, sigma_mode="raw")
    capped = mu_sigma_home_win_prob(7.0, 20.0)
    assert capped > raw


def test_fair_american_odds():
    assert round(fair_american_odds(0.6), 1) == -150.0
    assert round(fair_american_odds(0.4), 1) == 150.0


def test_neutral_site_probability_calibration_only_applies_to_neutral():
    non_neutral = site_home_win_prob_from_mu_sigma(
        5.0,
        12.0,
        start_month=3,
        start_day=20,
        neutral_site=False,
    )
    neutral = site_home_win_prob_from_mu_sigma(
        5.0,
        12.0,
        start_month=3,
        start_day=20,
        neutral_site=True,
    )
    assert neutral is not None
    assert non_neutral is not None
    assert neutral != non_neutral


def test_neutral_site_probability_calibration_vectorized_mask():
    probs = site_home_win_prob_from_mu_sigma(
        [5.0, 5.0],
        [12.0, 12.0],
        start_month=3,
        start_day=20,
        neutral_site=[False, True],
    )
    assert probs[0] != probs[1]

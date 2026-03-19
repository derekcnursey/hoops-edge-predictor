import numpy as np

from src.cover_probability_calibration import (
    apply_probability_calibration,
    fit_isotonic_calibrator,
    fit_logistic_calibrator,
    normal_cdf_from_z,
)


def test_normal_cdf_from_z_is_monotonic_and_centered():
    z = np.array([-1.0, 0.0, 1.0])
    probs = normal_cdf_from_z(z)
    assert probs[0] < probs[1] < probs[2]
    assert np.isclose(probs[1], 0.5)


def test_logistic_calibrator_produces_bounded_monotonic_probs():
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.array([0, 0, 0, 1, 1])
    calibration = fit_logistic_calibrator(z, y)
    probs = apply_probability_calibration(z, calibration)
    assert calibration["method"] in {"logistic", "constant"}
    assert np.all(probs > 0.0)
    assert np.all(probs < 1.0)
    assert np.all(np.diff(probs) >= 0.0)


def test_isotonic_calibrator_is_monotonic():
    z = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    calibration = fit_isotonic_calibrator(z, y)
    probs = apply_probability_calibration(z, calibration)
    assert calibration["method"] in {"isotonic", "constant"}
    assert np.all(probs > 0.0)
    assert np.all(probs < 1.0)
    assert np.all(np.diff(probs) >= -1e-12)

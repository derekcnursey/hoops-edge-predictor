"""Unit tests for simple post-hoc sigma transforms."""

import numpy as np
import pytest

from src.sigma_calibration import apply_sigma_transform


def test_apply_sigma_transform_raw_is_identity():
    sigma = np.array([1.0, 2.0, 3.0], dtype=float)
    out = apply_sigma_transform(sigma, mode="raw")
    np.testing.assert_allclose(out, sigma)


def test_apply_sigma_transform_cap():
    sigma = np.array([10.0, 14.5, 18.0], dtype=float)
    out = apply_sigma_transform(sigma, mode="cap", cap_max=14.0)
    np.testing.assert_allclose(out, np.array([10.0, 14.0, 14.0]))


def test_apply_sigma_transform_affine_and_bounds():
    sigma = np.array([1.0, 2.0, 3.0], dtype=float)
    out = apply_sigma_transform(sigma, mode="affine", affine_a=1.0, affine_b=2.0)
    np.testing.assert_allclose(out, np.array([3.0, 5.0, 7.0]))


def test_apply_sigma_transform_shrink():
    sigma = np.array([10.0, 14.0], dtype=float)
    out = apply_sigma_transform(
        sigma,
        mode="shrink",
        shrink_alpha=0.75,
        shrink_target=12.0,
    )
    np.testing.assert_allclose(out, np.array([10.5, 13.5]))


def test_apply_sigma_transform_requires_params():
    with pytest.raises(ValueError):
        apply_sigma_transform(np.array([1.0]), mode="cap")

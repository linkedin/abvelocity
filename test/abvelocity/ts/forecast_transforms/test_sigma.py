# Original author: Reza Hosseini
"""Unit tests for ``forecast_transforms.sigma``."""

from __future__ import annotations

import math

import pandas as pd

from abvelocity.ts.forecast_transforms.sigma import (
    propagate_sigma_indep_sum,
    propagate_sigma_share,
    propagate_sigma_share_delta,
)


def test_indep_sum_three_equal_sigmas():
    sigmas = pd.Series([3.0, 4.0, 0.0])  # σ_total = sqrt(9+16+0) = 5
    assert math.isclose(propagate_sigma_indep_sum(sigmas=sigmas), 5.0)


def test_indep_sum_with_some_nan_skips_them():
    sigmas = pd.Series([3.0, float("nan"), 4.0])
    assert math.isclose(propagate_sigma_indep_sum(sigmas=sigmas), 5.0)


def test_indep_sum_all_nan_returns_nan():
    sigmas = pd.Series([float("nan"), float("nan")])
    assert math.isnan(propagate_sigma_indep_sum(sigmas=sigmas))


def test_share_constant_denom_basic():
    sigma_x = pd.Series([2.0, 4.0, 6.0])
    out = propagate_sigma_share(sigma_x=sigma_x, value_y=10.0)
    expected = pd.Series([0.2, 0.4, 0.6])
    pd.testing.assert_series_equal(left=out, right=expected, check_names=False)


def test_share_constant_denom_handles_zero_denominator():
    sigma_x = pd.Series([2.0, 4.0])
    out = propagate_sigma_share(sigma_x=sigma_x, value_y=0.0)
    assert out.isna().all()


def test_share_constant_denom_handles_nan_denominator():
    sigma_x = pd.Series([2.0, 4.0])
    out = propagate_sigma_share(sigma_x=sigma_x, value_y=float("nan"))
    assert out.isna().all()


def test_share_constant_denom_uses_absolute_value_for_denom():
    sigma_x = pd.Series([2.0, 4.0])
    out_pos = propagate_sigma_share(sigma_x=sigma_x, value_y=10.0)
    out_neg = propagate_sigma_share(sigma_x=sigma_x, value_y=-10.0)
    pd.testing.assert_series_equal(left=out_pos, right=out_neg, check_names=False)


def test_share_delta_strictly_larger_than_constant():
    """Delta method adds a w²·σ_Y² term so the sigma is always ≥ constant version."""
    sigma_x = pd.Series([2.0, 4.0, 6.0])
    value_x = pd.Series([10.0, 20.0, 30.0])
    value_y = 60.0
    sigma_y = 5.0

    constant = propagate_sigma_share(sigma_x=sigma_x, value_y=value_y)
    delta = propagate_sigma_share_delta(
        sigma_x=sigma_x,
        value_x=value_x,
        sigma_y=sigma_y,
        value_y=value_y,
    )
    assert (delta.to_numpy() >= constant.to_numpy()).all()


def test_share_delta_collapses_to_constant_when_denom_sigma_zero():
    """When σ_Y = 0, delta method matches the constant-denominator formula."""
    sigma_x = pd.Series([2.0, 4.0])
    value_x = pd.Series([10.0, 20.0])
    value_y = 30.0

    constant = propagate_sigma_share(sigma_x=sigma_x, value_y=value_y)
    delta = propagate_sigma_share_delta(
        sigma_x=sigma_x,
        value_x=value_x,
        sigma_y=0.0,
        value_y=value_y,
    )
    pd.testing.assert_series_equal(left=delta, right=constant, check_names=False)


def test_share_delta_handles_zero_denominator():
    sigma_x = pd.Series([2.0, 4.0])
    value_x = pd.Series([1.0, 2.0])
    out = propagate_sigma_share_delta(
        sigma_x=sigma_x,
        value_x=value_x,
        sigma_y=1.0,
        value_y=0.0,
    )
    assert out.isna().all()


def test_share_delta_formula_matches_closed_form():
    """σ(w) = sqrt(σ_X² + w² σ_Y²) / |Y|."""
    sigma_x = pd.Series([3.0])
    value_x = pd.Series([20.0])
    sigma_y = 4.0
    value_y = 100.0
    out = propagate_sigma_share_delta(
        sigma_x=sigma_x,
        value_x=value_x,
        sigma_y=sigma_y,
        value_y=value_y,
    )
    weight = 20.0 / 100.0
    expected = math.sqrt(3.0**2 + weight**2 * 4.0**2) / 100.0
    assert math.isclose(out.iloc[0], expected, rel_tol=1e-9)

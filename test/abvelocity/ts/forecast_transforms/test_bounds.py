# Original author: Reza Hosseini
"""Unit tests for ``forecast_transforms.bounds``."""

from __future__ import annotations

import math

import pandas as pd

from abvelocity.ts.forecast_transforms.bounds import (
    multiplier_from_coverage,
    recompute_bounds,
)


def test_multiplier_80_percent():
    assert math.isclose(multiplier_from_coverage(ci_coverage=0.80), 1.2816, abs_tol=1e-3)


def test_multiplier_95_percent():
    assert math.isclose(multiplier_from_coverage(ci_coverage=0.95), 1.9600, abs_tol=1e-3)


def test_multiplier_99_percent():
    assert math.isclose(multiplier_from_coverage(ci_coverage=0.99), 2.5758, abs_tol=1e-3)


def test_multiplier_50_percent():
    assert math.isclose(multiplier_from_coverage(ci_coverage=0.50), 0.6745, abs_tol=1e-3)


def test_recompute_bounds_basic_80():
    forecast = pd.Series([100.0, 200.0])
    sigma = pd.Series([10.0, 20.0])
    lower, upper = recompute_bounds(forecast=forecast, sigma=sigma, ci_coverage=0.80)
    # 80% CI: bounds = forecast ± 1.2816 · σ
    pd.testing.assert_series_equal(
        left=lower,
        right=pd.Series([100.0 - 1.2816 * 10.0, 200.0 - 1.2816 * 20.0]),
        check_exact=False,
        atol=1e-3,
    )
    pd.testing.assert_series_equal(
        left=upper,
        right=pd.Series([100.0 + 1.2816 * 10.0, 200.0 + 1.2816 * 20.0]),
        check_exact=False,
        atol=1e-3,
    )


def test_recompute_bounds_zero_sigma_collapses_to_forecast():
    forecast = pd.Series([100.0, 200.0])
    sigma = pd.Series([0.0, 0.0])
    lower, upper = recompute_bounds(forecast=forecast, sigma=sigma, ci_coverage=0.95)
    pd.testing.assert_series_equal(left=lower, right=forecast, check_exact=False, atol=1e-9)
    pd.testing.assert_series_equal(left=upper, right=forecast, check_exact=False, atol=1e-9)


def test_recompute_bounds_propagates_nan_sigma():
    forecast = pd.Series([100.0, 200.0])
    sigma = pd.Series([float("nan"), 20.0])
    lower, upper = recompute_bounds(forecast=forecast, sigma=sigma, ci_coverage=0.80)
    assert pd.isna(lower.iloc[0])
    assert pd.isna(upper.iloc[0])
    assert not pd.isna(lower.iloc[1])
    assert not pd.isna(upper.iloc[1])


def test_recompute_bounds_wider_at_higher_coverage():
    """Higher coverage → wider intervals."""
    forecast = pd.Series([100.0])
    sigma = pd.Series([10.0])
    _, upper_80 = recompute_bounds(forecast=forecast, sigma=sigma, ci_coverage=0.80)
    _, upper_95 = recompute_bounds(forecast=forecast, sigma=sigma, ci_coverage=0.95)
    _, upper_99 = recompute_bounds(forecast=forecast, sigma=sigma, ci_coverage=0.99)
    assert upper_80.iloc[0] < upper_95.iloc[0] < upper_99.iloc[0]

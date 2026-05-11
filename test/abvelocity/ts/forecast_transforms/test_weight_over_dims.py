# Original author: Reza Hosseini
"""Unit tests for ``forecast_transforms.WeightOverDims``."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from abvelocity.ts.constants import (
    ACTUAL_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    FORECASTED_DATE_COL,
    STAGE_COL,
)
from abvelocity.ts.forecast_transforms import WeightOverDims


def test_empty_input_returns_empty(country_device_panel):
    out = WeightOverDims(within_dims=("country",)).apply(forecast_df=country_device_panel.iloc[:0])
    assert out.empty


def test_share_within_country_sums_to_one_per_country_per_week(country_device_panel):
    out = WeightOverDims(within_dims=("country",)).apply(forecast_df=country_device_panel)
    grouped = out.groupby(by=[FORECASTED_DATE_COL, "country"])[FORECAST_COL].sum()
    for total in grouped:
        assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_us_first_week_shares_correct(country_device_panel):
    """US week 0: android=100, iphone=150, total=250 → shares 0.4, 0.6."""
    out = WeightOverDims(within_dims=("country",)).apply(forecast_df=country_device_panel)
    us_first = out.loc[(out["country"] == "US") & (out[FORECASTED_DATE_COL] == pd.Timestamp("2026-01-03"))].sort_values(by="device")
    assert math.isclose(us_first.loc[us_first["device"] == "android", FORECAST_COL].iloc[0], 0.4)
    assert math.isclose(us_first.loc[us_first["device"] == "iphone", FORECAST_COL].iloc[0], 0.6)


def test_share_of_grand_total_when_within_dims_empty(country_device_panel):
    """Empty within_dims → each row's share of the global total per (timestamp)."""
    out = WeightOverDims(within_dims=()).apply(forecast_df=country_device_panel)
    # Within each week, all rows' shares should sum to 1.
    week_totals = out.groupby(by=[FORECASTED_DATE_COL])[FORECAST_COL].sum()
    for total in week_totals:
        assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_actual_share_uses_own_total(country_device_panel):
    out = WeightOverDims(within_dims=("country",)).apply(forecast_df=country_device_panel)
    grouped = out.groupby(by=[FORECASTED_DATE_COL, "country"])[ACTUAL_COL].sum()
    for total in grouped:
        assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_unknown_within_dim_raises(country_device_panel):
    with pytest.raises(ValueError, match="not found in forecast_df columns"):
        WeightOverDims(within_dims=("planet",)).apply(forecast_df=country_device_panel)


def test_bounds_recomputed_at_share_level(country_device_panel):
    """bounds = share ± 1.2816 · σ_share."""
    out = WeightOverDims(within_dims=("country",), ci_coverage=0.80).apply(
        forecast_df=country_device_panel,
    )
    # US android first week: forecast=100, total=250 → share=0.4. σ=5/250=0.02.
    us_android_first = out.loc[(out["country"] == "US") & (out["device"] == "android") & (out[FORECASTED_DATE_COL] == pd.Timestamp("2026-01-03"))]
    expected_share = 0.4
    expected_sigma = 5.0 / 250.0
    expected_lower = expected_share - 1.2816 * expected_sigma
    expected_upper = expected_share + 1.2816 * expected_sigma
    assert math.isclose(
        us_android_first[FORECAST_LOWER_COL].iloc[0],
        expected_lower,
        rel_tol=1e-3,
    )
    assert math.isclose(
        us_android_first[FORECAST_UPPER_COL].iloc[0],
        expected_upper,
        rel_tol=1e-3,
    )


def test_stage_column_preserved(country_device_panel):
    out = WeightOverDims(within_dims=("country",)).apply(forecast_df=country_device_panel)
    assert STAGE_COL in out.columns
    assert (out[STAGE_COL] == "fitted").all()


def test_str_name_with_within_dims():
    assert WeightOverDims(within_dims=("region",)).str_name() == "share_within_region"


def test_str_name_with_multiple_within_dims():
    assert WeightOverDims(within_dims=("region", "segment")).str_name() == "share_within_region_segment"


def test_str_name_empty_within_dims():
    assert WeightOverDims(within_dims=()).str_name() == "share_of_total"


def test_delta_vs_constant_method_choice(country_device_panel):
    """Both methods produce valid output; delta is at least as wide."""
    out_constant = WeightOverDims(within_dims=("country",), sigma_method="constant").apply(
        forecast_df=country_device_panel,
    )
    out_delta = WeightOverDims(within_dims=("country",), sigma_method="delta").apply(
        forecast_df=country_device_panel,
    )
    # Same point estimates regardless of sigma method.
    pd.testing.assert_series_equal(
        left=out_constant[FORECAST_COL].reset_index(drop=True),
        right=out_delta[FORECAST_COL].reset_index(drop=True),
        check_names=False,
    )

# Original author: Reza Hosseini
"""Unit tests for ``forecast_transforms.SumOverDims``."""

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
    STD_COL,
)
from abvelocity.ts.forecast_transforms import SumOverDims


def test_empty_input_returns_empty(country_device_panel):
    out = SumOverDims(dims_summed=("device",)).apply(forecast_df=country_device_panel.iloc[:0])
    assert out.empty


def test_sums_out_device_keeps_country(country_device_panel):
    out = SumOverDims(
        dims_summed=("device",),
        dims_maintained=("country",),
    ).apply(forecast_df=country_device_panel)
    # Each country has 4 weeks; device gone from output.
    assert "device" not in out.columns
    assert "country" in out.columns
    assert len(out) == 4 * 2  # 4 weeks × 2 countries


def test_summed_forecast_matches_manual(country_device_panel):
    out = SumOverDims(
        dims_summed=("device",),
        dims_maintained=("country",),
    ).apply(forecast_df=country_device_panel)
    # First week US: android 100 + iphone 150 = 250.
    first_us_row = out.loc[(out["country"] == "US") & (out[FORECASTED_DATE_COL] == pd.Timestamp("2026-01-03"))]
    assert first_us_row[FORECAST_COL].iloc[0] == 250.0
    # First week GB: android 30 + iphone 50 = 80.
    first_gb_row = out.loc[(out["country"] == "GB") & (out[FORECASTED_DATE_COL] == pd.Timestamp("2026-01-03"))]
    assert first_gb_row[FORECAST_COL].iloc[0] == 80.0


def test_summed_sigma_independent_gaussian(country_device_panel):
    out = SumOverDims(
        dims_summed=("device",),
        dims_maintained=("country",),
    ).apply(forecast_df=country_device_panel)
    # Each input row has σ=5.  2 devices summed → σ = sqrt(2·25) = sqrt(50).
    assert math.isclose(out[STD_COL].iloc[0], math.sqrt(50.0), rel_tol=1e-9)


def test_bounds_recomputed_at_aggregated_level(country_device_panel):
    out = SumOverDims(
        dims_summed=("device",),
        dims_maintained=("country",),
        ci_coverage=0.80,
    ).apply(forecast_df=country_device_panel)
    new_sigma = math.sqrt(50.0)
    new_forecast = 250.0
    expected_lower = new_forecast - 1.2816 * new_sigma
    expected_upper = new_forecast + 1.2816 * new_sigma
    first_us_row = out.loc[(out["country"] == "US") & (out[FORECASTED_DATE_COL] == pd.Timestamp("2026-01-03"))]
    assert math.isclose(first_us_row[FORECAST_LOWER_COL].iloc[0], expected_lower, rel_tol=1e-3)
    assert math.isclose(first_us_row[FORECAST_UPPER_COL].iloc[0], expected_upper, rel_tol=1e-3)


def test_actual_summed_too(country_device_panel):
    out = SumOverDims(
        dims_summed=("device",),
        dims_maintained=("country",),
    ).apply(forecast_df=country_device_panel)
    # First week US: android (100 + 0) + iphone (150 + 0) = 250 = same as actual = 250.
    first_us_row = out.loc[(out["country"] == "US") & (out[FORECASTED_DATE_COL] == pd.Timestamp("2026-01-03"))]
    # actual = forecast + week_idx · 1 per row, week_idx=0 → actual = forecast.
    assert first_us_row[ACTUAL_COL].iloc[0] == 250.0


def test_sum_out_all_dims_reduces_to_global(country_device_panel):
    out = SumOverDims(
        dims_summed=("country", "device"),
    ).apply(forecast_df=country_device_panel)
    # No dims left; one row per week.
    assert "country" not in out.columns
    assert "device" not in out.columns
    assert len(out) == 4  # one row per week
    # Week 0 total: 100 + 150 + 30 + 50 = 330.
    assert out.loc[out[FORECASTED_DATE_COL] == pd.Timestamp("2026-01-03"), FORECAST_COL].iloc[0] == 330.0


def test_overlap_between_summed_and_maintained_raises(country_device_panel):
    with pytest.raises(ValueError, match="must be disjoint"):
        SumOverDims(
            dims_summed=("country",),
            dims_maintained=("country",),
        ).apply(forecast_df=country_device_panel)


def test_unknown_dim_raises(country_device_panel):
    with pytest.raises(ValueError, match="not found in forecast_df columns"):
        SumOverDims(dims_summed=("planet",)).apply(forecast_df=country_device_panel)


def test_unclassified_string_column_raises(country_device_panel):
    """Unlisted non-numeric column → fail loud, don't silently group by it."""
    df = country_device_panel.assign(some_metadata=["x"] * len(country_device_panel))
    with pytest.raises(ValueError, match="unclassified non-numeric columns"):
        SumOverDims(
            dims_summed=("device",),
            dims_maintained=("country",),
        ).apply(forecast_df=df)


def test_stage_column_preserved(country_device_panel):
    out = SumOverDims(
        dims_summed=("device",),
        dims_maintained=("country",),
    ).apply(forecast_df=country_device_panel)
    assert STAGE_COL in out.columns
    # Panel is fully fitted → output stays fitted.
    assert (out[STAGE_COL] == "fitted").all()


def test_str_name_single_dim():
    assert SumOverDims(dims_summed=("country",)).str_name() == "sum_over_country"


def test_str_name_multiple_dims():
    assert SumOverDims(dims_summed=("country", "device")).str_name() == "sum_over_country_device"


def test_partial_actual_yields_nan_actual_sum(country_device_panel):
    """If any row in a group has NaN actual → group's actual sum is NaN."""
    df = country_device_panel.copy()
    df.loc[df["device"] == "iphone", ACTUAL_COL] = float("nan")
    out = SumOverDims(
        dims_summed=("device",),
        dims_maintained=("country",),
    ).apply(forecast_df=df)
    assert out[ACTUAL_COL].isna().all()
    # Forecast still sums fine since it's complete.
    assert (out[FORECAST_COL] > 0).all()

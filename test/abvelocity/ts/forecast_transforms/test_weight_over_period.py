# Original author: Reza Hosseini
"""Unit tests for ``forecast_transforms.WeightOverPeriod``."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from abvelocity.ts.constants import (
    ACTUAL_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    FORECASTED_DATE_COL,
    LONGTERM_GROWTH_COL,
    STAGE_COL,
    STD_COL,
    WEEKLY_SEASONALITY_COL,
)
from abvelocity.ts.forecast_transforms import WeightOverPeriod


def test_empty_input_returns_empty(daily_complete_two_weeks):
    out = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks.iloc[:0])
    assert out.empty


def test_complete_period_shares_sum_to_one(daily_complete_two_weeks):
    out = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    # 14 input rows → 14 output rows (rows aren't reduced).
    assert len(out) == 14
    week_groups = out.groupby(by=out[FORECASTED_DATE_COL].dt.to_period("W-SAT"))[FORECAST_COL].sum()
    for total in week_groups:
        assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_uniform_input_yields_equal_shares(daily_complete_two_weeks):
    """All forecast=10 → each day is 1/7 of the week."""
    out = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    assert (out[FORECAST_COL].round(decimals=4) == round(1.0 / 7.0, 4)).all()


def test_actual_uses_own_denominator(daily_complete_two_weeks):
    """actual_share = actual / sum(actual) — independent of forecast totals."""
    out = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    week_actual_shares = out.groupby(by=out[FORECASTED_DATE_COL].dt.to_period("W-SAT"))[ACTUAL_COL].sum()
    for total in week_actual_shares:
        assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_decomposition_uses_forecast_total_as_denominator(daily_complete_two_weeks):
    """Σ component_share == forecast_share at row level → component sums to forecast share."""
    out = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    # longterm_growth=8, forecast=10 → longterm_growth/total_forecast = 8/70 per row.
    expected_longterm_share = 8.0 / 70.0
    assert (out[LONGTERM_GROWTH_COL].round(decimals=6) == round(expected_longterm_share, 6)).all()


def test_sigma_is_constant_denom_share(daily_complete_two_weeks):
    """std/total_forecast — denom treated as known constant."""
    out = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    expected_sigma_share = 1.0 / 70.0
    assert (out[STD_COL].round(decimals=6) == round(expected_sigma_share, 6)).all()


def test_bounds_recomputed_from_share_forecast_and_share_sigma(daily_complete_two_weeks):
    out = WeightOverPeriod(period="W-SAT", ci_coverage=0.80).apply(forecast_df=daily_complete_two_weeks)
    forecast_share = 1.0 / 7.0
    sigma_share = 1.0 / 70.0
    expected_lower = forecast_share - 1.2816 * sigma_share
    expected_upper = forecast_share + 1.2816 * sigma_share
    assert math.isclose(out[FORECAST_LOWER_COL].iloc[0], expected_lower, rel_tol=1e-3)
    assert math.isclose(out[FORECAST_UPPER_COL].iloc[0], expected_upper, rel_tol=1e-3)


def test_incomplete_period_rows_get_nan(daily_partial_week):
    """3 daily rows of a 7-day week → all rows NaN'd in output."""
    out = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_partial_week)
    assert len(out) == 3  # rows preserved
    assert out[FORECAST_COL].isna().all()
    assert out[ACTUAL_COL].isna().all()
    assert out[STD_COL].isna().all()
    assert out[FORECAST_LOWER_COL].isna().all()
    assert out[FORECAST_UPPER_COL].isna().all()


def test_partial_actual_in_complete_period_yields_nan_actual_share(daily_complete_two_weeks):
    """Period has all 7 rows but actual NaN on a few days → actual_share NaN
    for the whole week (denom would be partial → bogus shares otherwise).

    Mirrors the situation at the train/forecast boundary: the W-SAT week
    spanning that cutoff has all rows present (forecast extends past the
    cutoff), but the last few days' actuals are NaN because they're in
    the forecast horizon.
    """
    df = daily_complete_two_weeks.copy()
    # NaN the last 3 actuals of the second W-SAT week (Sun 2026-01-04 +
    # 7 = Sun 2026-01-11; rows 11..13 are Thu/Fri/Sat of week 2).
    df.loc[df.index[-3:], ACTUAL_COL] = float("nan")
    out = WeightOverPeriod(period="W-SAT").apply(forecast_df=df)
    second_week_mask = out[FORECASTED_DATE_COL] >= pd.Timestamp("2026-01-11")
    assert out.loc[second_week_mask, ACTUAL_COL].isna().all()
    # Forecast side is fully populated → its shares should still resolve.
    assert out.loc[second_week_mask, FORECAST_COL].notna().all()
    # First week (fully-populated actuals) is unaffected.
    first_week_mask = out[FORECASTED_DATE_COL] < pd.Timestamp("2026-01-11")
    assert out.loc[first_week_mask, ACTUAL_COL].notna().all()


def test_stage_column_preserved(daily_complete_two_weeks):
    """Rows preserved + stage flows through untouched."""
    out = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    assert STAGE_COL in out.columns
    assert (out[STAGE_COL] == "fitted").all()


def test_delta_method_yields_larger_sigma_than_constant():
    """Same data, sigma_method='delta' should produce ≥ the constant version's σ_share."""
    dates = pd.date_range(start="2026-01-04", periods=7, freq="D")
    forecast_values = np.array([10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 13.0])
    df = pd.DataFrame(
        {
            FORECASTED_DATE_COL: dates,
            "metric_id": "x:daily",
            "metric_name": "X (daily)",
            STAGE_COL: "fitted",
            ACTUAL_COL: forecast_values,
            FORECAST_COL: forecast_values,
            STD_COL: np.full(shape=7, fill_value=1.0),
            FORECAST_LOWER_COL: forecast_values - 1.2816,
            FORECAST_UPPER_COL: forecast_values + 1.2816,
            LONGTERM_GROWTH_COL: 8.0,
            WEEKLY_SEASONALITY_COL: 0.0,
        }
    )
    out_constant = WeightOverPeriod(period="W-SAT", sigma_method="constant").apply(forecast_df=df.copy())
    out_delta = WeightOverPeriod(period="W-SAT", sigma_method="delta").apply(forecast_df=df.copy())
    assert (out_delta[STD_COL].to_numpy() >= out_constant[STD_COL].to_numpy() - 1e-9).all()


def test_str_name_known_aliases():
    assert WeightOverPeriod(period="W").str_name() == "share_of_week"
    assert WeightOverPeriod(period="D").str_name() == "share_of_day"
    assert WeightOverPeriod(period="MS").str_name() == "share_of_month"


def test_str_name_unknown_alias():
    assert WeightOverPeriod(period="W-SAT").str_name() == "share_of_w-sat"

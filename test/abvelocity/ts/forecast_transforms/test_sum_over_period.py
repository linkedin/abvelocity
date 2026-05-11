# Original author: Reza Hosseini
"""Unit tests for ``forecast_transforms.SumOverPeriod``."""

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
from abvelocity.ts.forecast_transforms import SumOverPeriod


def test_empty_input_returns_empty(daily_complete_two_weeks):
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks.iloc[:0])
    assert out.empty


def test_two_complete_weeks_produce_two_rows(daily_complete_two_weeks):
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    assert len(out) == 2


def test_period_anchor_is_period_start(daily_complete_two_weeks):
    """Output ``forecasted_date`` carries the period's lower bound (Sunday for W-SAT)."""
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    assert out[FORECASTED_DATE_COL].iloc[0] == pd.Timestamp("2026-01-04")
    assert out[FORECASTED_DATE_COL].iloc[1] == pd.Timestamp("2026-01-11")


def test_forecast_sums_correctly(daily_complete_two_weeks):
    """Each row had forecast=10, 7 days/week → weekly forecast = 70."""
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    assert (out[FORECAST_COL] == 70.0).all()


def test_actual_sums_correctly(daily_complete_two_weeks):
    """Each row had actual=11, 7 days/week → weekly actual = 77."""
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    assert (out[ACTUAL_COL] == 77.0).all()


def test_sigma_independent_gaussian(daily_complete_two_weeks):
    """7 rows of σ=1 → weekly σ = sqrt(7) ≈ 2.6458."""
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    expected_sigma = math.sqrt(7.0)
    assert math.isclose(out[STD_COL].iloc[0], expected_sigma, rel_tol=1e-9)


def test_bounds_recomputed_from_new_forecast_and_sigma(daily_complete_two_weeks):
    """new_lower/upper = new_forecast ± 1.2816 · new_sigma."""
    out = SumOverPeriod(period="W-SAT", ci_coverage=0.80).apply(forecast_df=daily_complete_two_weeks)
    expected_sigma = math.sqrt(7.0)
    assert math.isclose(out[FORECAST_LOWER_COL].iloc[0], 70.0 - 1.2816 * expected_sigma, rel_tol=1e-3)
    assert math.isclose(out[FORECAST_UPPER_COL].iloc[0], 70.0 + 1.2816 * expected_sigma, rel_tol=1e-3)


def test_breakdown_columns_sum(daily_complete_two_weeks):
    """LONGTERM_GROWTH=8 per row, 7 rows → 56 per week."""
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    assert (out[LONGTERM_GROWTH_COL] == 56.0).all()


def test_incomplete_period_dropped(daily_partial_week):
    """3 rows of a 7-row week → period is dropped."""
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_partial_week)
    assert out.empty


def test_stage_column_preserved_as_fitted(daily_complete_two_weeks):
    """All 14 input rows are stage='fitted' → output weeks all 'fitted'."""
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_complete_two_weeks)
    assert STAGE_COL in out.columns
    assert (out[STAGE_COL] == "fitted").all()


def test_stage_column_any_forecast_wins(daily_complete_two_weeks):
    """A week with at least one forecast-stage row → aggregated stage is 'forecast'."""
    df = daily_complete_two_weeks.copy()
    df.loc[df.index[-1], STAGE_COL] = "forecast"  # last row of week 2 → forecast
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=df)
    # 2 weeks: week 1 fully fitted, week 2 has one forecast row → marked forecast.
    assert sorted(out[STAGE_COL]) == ["fitted", "forecast"]


def test_actual_with_some_nan_yields_nan_actual_sum():
    """Straddling cutoff: 3 fitted + 4 forecast → actual sum is NaN (sum_strict)."""
    dates = pd.date_range(start="2026-01-04", periods=7, freq="D")
    forecast_values = np.full(shape=7, fill_value=10.0)
    actual_values = np.array([11.0, 12.0, 13.0, np.nan, np.nan, np.nan, np.nan])
    df = pd.DataFrame(
        {
            FORECASTED_DATE_COL: dates,
            "metric_id": "x:daily",
            "metric_name": "X (daily)",
            STAGE_COL: ["fitted"] * 3 + ["forecast"] * 4,
            ACTUAL_COL: actual_values,
            FORECAST_COL: forecast_values,
            STD_COL: np.full(shape=7, fill_value=1.0),
            FORECAST_LOWER_COL: forecast_values - 1.2816,
            FORECAST_UPPER_COL: forecast_values + 1.2816,
            LONGTERM_GROWTH_COL: np.full(shape=7, fill_value=8.0),
            WEEKLY_SEASONALITY_COL: np.zeros(7),
        }
    )
    out = SumOverPeriod(period="W-SAT").apply(forecast_df=df)
    assert len(out) == 1
    assert pd.isna(out[ACTUAL_COL].iloc[0])
    assert out[FORECAST_COL].iloc[0] == 70.0


def test_str_name_known_aliases():
    assert SumOverPeriod(period="W-SAT").str_name() == "w-sat"
    assert SumOverPeriod(period="W").str_name() == "weekly"
    assert SumOverPeriod(period="MS").str_name() == "monthly"
    assert SumOverPeriod(period="YS").str_name() == "annual"
    assert SumOverPeriod(period="D").str_name() == "daily"


def test_str_name_quarterly():
    assert SumOverPeriod(period="QS").str_name() == "quarterly"


def test_str_name_unknown_alias_falls_back_to_lowercase():
    # Use a non-mapped alias so the str_name fallback path is exercised.
    assert SumOverPeriod(period="2D").str_name() == "2d"


def test_monthly_aggregation_handles_variable_length():
    """Feb has 28 days, March has 31 — both should aggregate cleanly."""
    dates = pd.date_range(start="2025-02-01", end="2025-03-31", freq="D")  # 28 + 31 = 59 days
    forecast_values = np.full(shape=len(dates), fill_value=1.0)
    df = pd.DataFrame(
        {
            FORECASTED_DATE_COL: dates,
            "metric_id": "x:daily",
            "metric_name": "X (daily)",
            STAGE_COL: "fitted",
            ACTUAL_COL: forecast_values,
            FORECAST_COL: forecast_values,
            STD_COL: np.full(shape=len(dates), fill_value=0.1),
            FORECAST_LOWER_COL: forecast_values - 0.1,
            FORECAST_UPPER_COL: forecast_values + 0.1,
            LONGTERM_GROWTH_COL: 0.8,
            WEEKLY_SEASONALITY_COL: 0.0,
        }
    )
    out = SumOverPeriod(period="MS").apply(forecast_df=df)
    assert len(out) == 2
    feb_total = out.loc[out[FORECASTED_DATE_COL] == pd.Timestamp("2025-02-01"), FORECAST_COL].iloc[0]
    march_total = out.loc[out[FORECASTED_DATE_COL] == pd.Timestamp("2025-03-01"), FORECAST_COL].iloc[0]
    assert feb_total == 28.0
    assert march_total == 31.0

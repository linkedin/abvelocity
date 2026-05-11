# Original author: Reza Hosseini
"""Shared fixtures for ``forecast_transforms`` unit tests.

Each fixture returns a small, hand-crafted forecast frame matching
:data:`abvelocity.ts.constants.FORECAST_TABLE_COLUMNS` minus
the columns we don't exercise (extras, run_id, run_date, …).  Tests
import these fixtures via the standard pytest mechanism — no fixture
needs to be re-declared per test file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abvelocity.ts.constants import (
    ACTUAL_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    FORECASTED_DATE_COL,
    LONGTERM_GROWTH_COL,
    METRIC_ID_COL,
    METRIC_NAME_COL,
    STAGE_COL,
    STD_COL,
    WEEKLY_SEASONALITY_COL,
)


@pytest.fixture
def fitted_daily_two_weeks() -> pd.DataFrame:
    """14 daily fitted rows for a single metric, no dim columns.

    Both ``actual`` and ``forecast`` populated.  The 80% CI is encoded
    via std=10 and z=1.2816 so ``upper - forecast = 12.816``.
    Decomposition: longterm_growth=80, weekly_seasonality varies.
    """
    dates = pd.date_range(start="2026-01-04", periods=14, freq="D")  # Sunday start
    forecast_values = np.linspace(start=100.0, stop=113.0, num=14)
    actual_values = forecast_values + np.array([1, -1, 0, 2, -2, 1, 0] * 2)
    std_values = np.full(shape=14, fill_value=10.0)
    weekly_pattern = np.array([5, -2, 0, 1, -3, 4, -5] * 2, dtype=float)
    return pd.DataFrame(
        {
            FORECASTED_DATE_COL: dates,
            METRIC_ID_COL: "x:daily",
            METRIC_NAME_COL: "X (daily)",
            STAGE_COL: "fitted",
            ACTUAL_COL: actual_values,
            FORECAST_COL: forecast_values,
            STD_COL: std_values,
            FORECAST_LOWER_COL: forecast_values - 1.2816 * std_values,
            FORECAST_UPPER_COL: forecast_values + 1.2816 * std_values,
            LONGTERM_GROWTH_COL: 80.0,
            WEEKLY_SEASONALITY_COL: weekly_pattern,
        }
    )


@pytest.fixture
def fitted_plus_forecast_two_weeks() -> pd.DataFrame:
    """7 fitted days + 7 forecast days for one metric.

    Forecast-stage rows have ``actual`` NaN; everything else populated.
    """
    dates = pd.date_range(start="2026-01-04", periods=14, freq="D")
    forecast_values = np.linspace(start=100.0, stop=113.0, num=14)
    actual_values = np.concatenate([forecast_values[:7] + np.array([1, -1, 0, 2, -2, 1, 0]), np.full(shape=7, fill_value=np.nan)])
    stages = np.array(["fitted"] * 7 + ["forecast"] * 7)
    std_values = np.full(shape=14, fill_value=10.0)
    return pd.DataFrame(
        {
            FORECASTED_DATE_COL: dates,
            METRIC_ID_COL: "x:daily",
            METRIC_NAME_COL: "X (daily)",
            STAGE_COL: stages,
            ACTUAL_COL: actual_values,
            FORECAST_COL: forecast_values,
            STD_COL: std_values,
            FORECAST_LOWER_COL: forecast_values - 1.2816 * std_values,
            FORECAST_UPPER_COL: forecast_values + 1.2816 * std_values,
            LONGTERM_GROWTH_COL: 80.0,
            WEEKLY_SEASONALITY_COL: 0.0,
        }
    )


@pytest.fixture
def country_device_panel() -> pd.DataFrame:
    """Fully-fitted weekly-anchored panel: 4 weeks × 2 countries × 2 devices.

    Used for SumOverDims / WeightOverDims unit tests.  Forecast values
    differ across (country, device) so aggregation isn't trivial.
    """
    dates = pd.date_range(start="2026-01-03", periods=4, freq="W-SAT")
    rows = []
    base_per_segment = {
        ("US", "android"): 100.0,
        ("US", "iphone"): 150.0,
        ("GB", "android"): 30.0,
        ("GB", "iphone"): 50.0,
    }
    for (country, device), base in base_per_segment.items():
        for week_idx, date in enumerate(dates):
            forecast_value = base + week_idx * 5.0
            rows.append(
                {
                    FORECASTED_DATE_COL: date,
                    METRIC_ID_COL: "signups:weekly",
                    METRIC_NAME_COL: "Signups (weekly)",
                    "country": country,
                    "device": device,
                    STAGE_COL: "fitted",
                    ACTUAL_COL: forecast_value + week_idx,
                    FORECAST_COL: forecast_value,
                    STD_COL: 5.0,
                    FORECAST_LOWER_COL: forecast_value - 1.2816 * 5.0,
                    FORECAST_UPPER_COL: forecast_value + 1.2816 * 5.0,
                    LONGTERM_GROWTH_COL: base * 0.8,
                    WEEKLY_SEASONALITY_COL: 0.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def daily_complete_two_weeks() -> pd.DataFrame:
    """14 daily rows starting on a Sunday, fully populated, no dims.

    Two complete W-SAT weeks; convenient for SumOverPeriod completeness
    checks.
    """
    dates = pd.date_range(start="2026-01-04", periods=14, freq="D")
    forecast_values = np.full(shape=14, fill_value=10.0)
    return pd.DataFrame(
        {
            FORECASTED_DATE_COL: dates,
            METRIC_ID_COL: "x:daily",
            METRIC_NAME_COL: "X (daily)",
            STAGE_COL: "fitted",
            ACTUAL_COL: forecast_values + 1.0,
            FORECAST_COL: forecast_values,
            STD_COL: 1.0,
            FORECAST_LOWER_COL: forecast_values - 1.2816,
            FORECAST_UPPER_COL: forecast_values + 1.2816,
            LONGTERM_GROWTH_COL: 8.0,
            WEEKLY_SEASONALITY_COL: 0.0,
        }
    )


@pytest.fixture
def daily_partial_week() -> pd.DataFrame:
    """3 daily rows starting on Sunday — an incomplete W-SAT week."""
    dates = pd.date_range(start="2026-01-04", periods=3, freq="D")
    forecast_values = np.full(shape=3, fill_value=10.0)
    return pd.DataFrame(
        {
            FORECASTED_DATE_COL: dates,
            METRIC_ID_COL: "x:daily",
            METRIC_NAME_COL: "X (daily)",
            STAGE_COL: "fitted",
            ACTUAL_COL: forecast_values + 1.0,
            FORECAST_COL: forecast_values,
            STD_COL: 1.0,
            FORECAST_LOWER_COL: forecast_values - 1.2816,
            FORECAST_UPPER_COL: forecast_values + 1.2816,
            LONGTERM_GROWTH_COL: 8.0,
            WEEKLY_SEASONALITY_COL: 0.0,
        }
    )

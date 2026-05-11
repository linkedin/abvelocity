# Original author: Reza Hosseini
"""Unit tests for ``forecast_transforms.period``."""

from __future__ import annotations

import pandas as pd
import pytest

from abvelocity.ts.constants import FORECASTED_DATE_COL
from abvelocity.ts.forecast_transforms.period import (
    expected_count_in_period,
    infer_input_freq,
    normalize_to_period_alias,
)


def test_expected_count_daily_in_week():
    start = pd.Timestamp("2026-01-04")  # Sunday
    end = pd.Timestamp("2026-01-10")  # Saturday
    assert expected_count_in_period(period_start=start, period_end=end, input_freq="D") == 7


def test_expected_count_daily_in_february_non_leap():
    start = pd.Timestamp("2025-02-01")
    end = pd.Timestamp("2025-02-28")
    assert expected_count_in_period(period_start=start, period_end=end, input_freq="D") == 28


def test_expected_count_daily_in_february_leap():
    start = pd.Timestamp("2024-02-01")
    end = pd.Timestamp("2024-02-29")
    assert expected_count_in_period(period_start=start, period_end=end, input_freq="D") == 29


def test_expected_count_daily_in_year_non_leap():
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-12-31")
    assert expected_count_in_period(period_start=start, period_end=end, input_freq="D") == 365


def test_expected_count_daily_in_year_leap():
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-12-31")
    assert expected_count_in_period(period_start=start, period_end=end, input_freq="D") == 366


def test_expected_count_hourly_in_day():
    start = pd.Timestamp("2026-01-04 00:00:00")
    end = pd.Timestamp("2026-01-04 23:00:00")
    assert expected_count_in_period(period_start=start, period_end=end, input_freq="h") == 24


def test_infer_input_freq_daily():
    df = pd.DataFrame({FORECASTED_DATE_COL: pd.date_range(start="2026-01-01", periods=10, freq="D")})
    assert infer_input_freq(forecast_df=df) == "D"


def test_infer_input_freq_weekly():
    df = pd.DataFrame({FORECASTED_DATE_COL: pd.date_range(start="2026-01-04", periods=10, freq="W-SAT")})
    assert infer_input_freq(forecast_df=df) == "W-SAT"


def test_infer_input_freq_irregular_raises():
    df = pd.DataFrame({FORECASTED_DATE_COL: pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-09"])})
    with pytest.raises(ValueError, match="Could not infer input freq"):
        infer_input_freq(forecast_df=df)


def test_infer_input_freq_dedupes_before_inferring():
    """Duplicate timestamps shouldn't break inference — they're collapsed first."""
    dates = pd.date_range(start="2026-01-01", periods=10, freq="D").repeat(2)
    df = pd.DataFrame({FORECASTED_DATE_COL: dates})
    assert infer_input_freq(forecast_df=df) == "D"


def test_normalize_to_period_alias_passes_through_anchored_weekly():
    """``W-SAT`` is already a valid pandas period alias — must NOT be touched."""
    assert normalize_to_period_alias(offset_or_period_alias="W-SAT") == "W-SAT"
    assert normalize_to_period_alias(offset_or_period_alias="W-SUN") == "W-SUN"
    assert normalize_to_period_alias(offset_or_period_alias="W-FRI") == "W-FRI"


def test_normalize_to_period_alias_passes_through_period_aliases():
    """``W``, ``D``, ``M``, ``Q``, ``Y`` are valid period aliases — pass through."""
    for alias in ["W", "D", "M", "Q", "Y", "h", "min", "s"]:
        assert normalize_to_period_alias(offset_or_period_alias=alias) == alias


def test_normalize_to_period_alias_collapses_offset_variants():
    """``MS``/``ME`` → ``M``, ``QS``/``QE`` → ``Q``, ``YS``/``YE``/``A`` → ``Y``."""
    assert normalize_to_period_alias(offset_or_period_alias="MS") == "M"
    assert normalize_to_period_alias(offset_or_period_alias="ME") == "M"
    assert normalize_to_period_alias(offset_or_period_alias="QS") == "Q"
    assert normalize_to_period_alias(offset_or_period_alias="QE") == "Q"
    assert normalize_to_period_alias(offset_or_period_alias="YS") == "Y"
    assert normalize_to_period_alias(offset_or_period_alias="YE") == "Y"
    assert normalize_to_period_alias(offset_or_period_alias="A") == "Y"


def test_normalize_to_period_alias_round_trips_through_pandas_to_period():
    """Every alias that maps through normalize must work as a pd.Series.dt.to_period freq."""
    series = pd.Series(pd.date_range(start="2025-01-01", periods=400, freq="D"))
    for alias in ["W-SAT", "W-SUN", "W", "MS", "ME", "M", "QS", "Q", "YS", "Y", "D"]:
        normalized = normalize_to_period_alias(offset_or_period_alias=alias)
        # Must not raise; period objects should have valid bounds.
        period_obj = series.dt.to_period(normalized)
        assert period_obj.dt.start_time.notna().all()
        assert period_obj.dt.end_time.notna().all()

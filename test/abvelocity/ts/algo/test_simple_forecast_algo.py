# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini
"""Tests for SimpleForecastAlgo and BackfillRunner integration."""

import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401 — self-registers
import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.algo.base import ALGO_REGISTRY
from abvelocity.ts.algo.simple_forecast_algo import SimpleForecastAlgo
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.backfill.runner import BackfillRunner
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import (
    ACTUAL_COL,
    CUTOFF_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    HORIZON_STEP_COL,
    METRIC_ID_COL,
    TIME_COL,
)
from abvelocity.ts.result.forecast_result import ForecastResult
from abvelocity.ts.runner import TSRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_series():
    """Daily series with values 1..21 (3 weeks), useful for exact arithmetic."""
    dates = pd.date_range("2024-01-01", periods=21, freq="D")
    return pd.DataFrame({TIME_COL: dates, "value": np.arange(1, 22, dtype=float)})


@pytest.fixture
def two_col_series():
    """Two-column daily series over 14 days."""
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    return pd.DataFrame(
        {
            TIME_COL: dates,
            "col1": np.arange(1, 15, dtype=float),
            "col2": np.arange(101, 115, dtype=float),
        }
    )


def make_forecast_config(
    value_cols=("value",),
    forecast_horizon=7,
    period=7,
    k=3,
    coverage=0.95,
    algo_name="simple",
    agg="mean",
):
    return ForecastConfig(
        time_col=TIME_COL,
        value_cols=value_cols,
        freq="D",
        forecast_horizon=forecast_horizon,
        coverage=coverage,
        algo_name=algo_name,
        algo_params={"period": period, "k": k, "agg": agg},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_simple_algo_registered():
    assert "simple" in ALGO_REGISTRY
    assert ALGO_REGISTRY["simple"] is SimpleForecastAlgo


# ---------------------------------------------------------------------------
# get_lookback_values
# ---------------------------------------------------------------------------


def test_get_lookback_values_period_1():
    series = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    algo = SimpleForecastAlgo(algo_params={"period": 1, "k": 3})
    # pos=4 (future step 1 from n=4): lookbacks at 3, 2, 1 → values 40, 30, 20
    assert algo.get_lookback_values(series, pos=4, period=1, k=3) == [40.0, 30.0, 20.0]


def test_get_lookback_values_period_7():
    series = np.arange(1.0, 22.0)  # 21 values, 0-indexed
    algo = SimpleForecastAlgo(algo_params={"period": 7, "k": 3})
    # pos=21 (step 1 from n=21): lookbacks at 14, 7, 0 → values 15, 8, 1
    assert algo.get_lookback_values(series, pos=21, period=7, k=3) == [15.0, 8.0, 1.0]


def test_get_lookback_values_skips_out_of_bounds():
    series = np.array([10.0, 20.0, 30.0])
    algo = SimpleForecastAlgo()
    # pos=3 (step 1 from n=3), period=7, k=3: all lookbacks negative
    assert algo.get_lookback_values(series, pos=3, period=7, k=3) == []


def test_get_lookback_values_partial_lookback():
    series = np.arange(1.0, 15.0)  # 14 values
    algo = SimpleForecastAlgo()
    # pos=14 (step 1 from n=14), period=7, k=3: pos-7=7, pos-14=0, pos-21=-7 (skip)
    assert algo.get_lookback_values(series, pos=14, period=7, k=3) == [8.0, 1.0]


# ---------------------------------------------------------------------------
# compute_point_and_ci
# ---------------------------------------------------------------------------


def test_compute_point_and_ci_known_values():
    algo = SimpleForecastAlgo()
    values = [10.0, 20.0, 30.0]
    mean_val = 20.0
    std_val = float(np.std([10.0, 20.0, 30.0], ddof=0))  # population std
    z_score = 1.96
    point, lower, upper = algo.compute_point_and_ci(values, agg="mean", z=z_score)
    assert np.isclose(point, mean_val)
    assert np.isclose(lower, mean_val - z_score * std_val)
    assert np.isclose(upper, mean_val + z_score * std_val)


def test_compute_point_and_ci_empty_returns_nan():
    algo = SimpleForecastAlgo()
    point, lower, upper = algo.compute_point_and_ci([], agg="mean", z=1.96)
    assert np.isnan(point) and np.isnan(lower) and np.isnan(upper)


def test_compute_point_and_ci_single_value_zero_width():
    algo = SimpleForecastAlgo()
    point, lower, upper = algo.compute_point_and_ci([42.0], agg="mean", z=1.96)
    assert np.isclose(point, 42.0)
    assert np.isclose(lower, 42.0)
    assert np.isclose(upper, 42.0)


# ---------------------------------------------------------------------------
# predict — forecast values
# ---------------------------------------------------------------------------


def test_predict_period_7_k3_step1(linear_series):
    """Step-1 forecast: mean of values at lags 7, 14, 21 from future position."""
    config = make_forecast_config(period=7, k=3, forecast_horizon=1)
    algo = SimpleForecastAlgo(algo_params={"period": 7, "k": 3})
    algo.fit(df=linear_series, config=config)
    result = algo.predict()

    future_rows = result.result_df[result.result_df[ACTUAL_COL].isna()]
    assert len(future_rows) == 1
    # n=21, step=1, future_pos=21: lookbacks at 14, 7, 0 → values 15, 8, 1 → mean=8.0
    assert np.isclose(future_rows[FORECAST_COL].iloc[0], 8.0)


def test_predict_period_1_k3_trailing_mean(linear_series):
    """period=1 → forecast is trailing 3-step mean."""
    config = make_forecast_config(period=1, k=3, forecast_horizon=1)
    algo = SimpleForecastAlgo(algo_params={"period": 1, "k": 3})
    algo.fit(df=linear_series, config=config)
    result = algo.predict()

    future_rows = result.result_df[result.result_df[ACTUAL_COL].isna()]
    # n=21, step=1, future_pos=21: lookbacks at 20, 19, 18 → values 21, 20, 19 → mean=20.0
    assert np.isclose(future_rows[FORECAST_COL].iloc[0], 20.0)


def test_predict_ci_lower_less_than_upper(linear_series):
    config = make_forecast_config(period=7, k=3, forecast_horizon=7)
    algo = SimpleForecastAlgo(algo_params={"period": 7, "k": 3})
    algo.fit(df=linear_series, config=config)
    result = algo.predict()

    future_rows = result.result_df[result.result_df[ACTUAL_COL].isna()].dropna(subset=[FORECAST_LOWER_COL, FORECAST_UPPER_COL])
    assert (future_rows[FORECAST_LOWER_COL] <= future_rows[FORECAST_COL]).all()
    assert (future_rows[FORECAST_COL] <= future_rows[FORECAST_UPPER_COL]).all()


def test_predict_returns_forecast_result(linear_series):
    config = make_forecast_config()
    algo = SimpleForecastAlgo(algo_params={"period": 7, "k": 3})
    algo.fit(df=linear_series, config=config)
    result = algo.predict()
    assert isinstance(result, ForecastResult)


def test_predict_result_df_has_required_columns(linear_series):
    config = make_forecast_config(forecast_horizon=3)
    algo = SimpleForecastAlgo(algo_params={"period": 7, "k": 3})
    algo.fit(df=linear_series, config=config)
    result = algo.predict()
    for col in (
        TIME_COL,
        METRIC_ID_COL,
        ACTUAL_COL,
        FORECAST_COL,
        FORECAST_LOWER_COL,
        FORECAST_UPPER_COL,
    ):
        assert col in result.result_df.columns


def test_predict_future_rows_count(linear_series):
    horizon = 5
    config = make_forecast_config(forecast_horizon=horizon)
    algo = SimpleForecastAlgo(algo_params={"period": 7, "k": 3})
    algo.fit(df=linear_series, config=config)
    result = algo.predict()

    future = result.result_df[result.result_df[ACTUAL_COL].isna()]
    assert len(future) == horizon


def test_predict_multivariate(two_col_series):
    config = make_forecast_config(value_cols=("col1", "col2"), forecast_horizon=3)
    algo = SimpleForecastAlgo(algo_params={"period": 7, "k": 2})
    algo.fit(df=two_col_series, config=config)
    result = algo.predict()

    assert set(result.result_df[METRIC_ID_COL].unique()) == {"col1", "col2"}
    future = result.result_df[result.result_df[ACTUAL_COL].isna()]
    assert len(future) == 3 * 2  # 3 steps × 2 metrics


def test_predict_before_fit_raises():
    algo = SimpleForecastAlgo()
    with pytest.raises(ValueError, match="fit"):
        algo.predict()


def test_predict_prediction_window_filter(linear_series):
    config = make_forecast_config(forecast_horizon=7)
    algo = SimpleForecastAlgo(algo_params={"period": 7, "k": 3})
    algo.fit(df=linear_series, config=config)
    window_start = "2024-01-22"
    window_end = "2024-01-24"
    result = algo.predict(prediction_window=(window_start, window_end))
    dates = result.result_df[TIME_COL]
    assert (dates >= pd.Timestamp(window_start)).all()
    assert (dates <= pd.Timestamp(window_end)).all()


# ---------------------------------------------------------------------------
# TSRunner integration
# ---------------------------------------------------------------------------


def test_ts_runner_with_simple_algo(linear_series):
    config = make_forecast_config(forecast_horizon=7)
    result = TSRunner(config).run(df=linear_series)
    assert result.result_df is not None
    future = result.result_df[result.result_df[ACTUAL_COL].isna()]
    assert len(future) == 7


# ---------------------------------------------------------------------------
# BackfillRunner integration — end-to-end with deterministic algo
# ---------------------------------------------------------------------------


@pytest.fixture
def backfill_series():
    """30 days of values 1..30 for exact backfill arithmetic."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame({TIME_COL: dates, "value": np.arange(1.0, 31.0)})


def test_backfill_runner_with_simple_algo_produces_result(backfill_series):
    forecast_config = make_forecast_config(value_cols=("value",), forecast_horizon=7, period=7, k=2)
    backfill_config = BackfillConfig(forecast_config=forecast_config, initial_train_size=14, horizon=7, step=7)
    result = BackfillRunner(backfill_config).run(df=backfill_series)
    assert result.result_df is not None


def test_backfill_runner_result_has_expected_columns(backfill_series):
    forecast_config = make_forecast_config(value_cols=("value",), forecast_horizon=7, period=7, k=2)
    backfill_config = BackfillConfig(forecast_config=forecast_config, initial_train_size=14, horizon=7, step=7)
    result = BackfillRunner(backfill_config).run(df=backfill_series)
    rdf = result.result_df
    for col in (TIME_COL, METRIC_ID_COL, ACTUAL_COL, FORECAST_COL, CUTOFF_COL, HORIZON_STEP_COL):
        assert col in rdf.columns


def test_backfill_runner_actuals_filled_from_prepped_df(backfill_series):
    forecast_config = make_forecast_config(value_cols=("value",), forecast_horizon=7, period=7, k=2)
    backfill_config = BackfillConfig(forecast_config=forecast_config, initial_train_size=14, horizon=7, step=7)
    result = BackfillRunner(backfill_config).run(df=backfill_series)
    assert result.result_df[ACTUAL_COL].notna().all()


def test_backfill_runner_horizon_step_range(backfill_series):
    horizon = 5
    forecast_config = make_forecast_config(value_cols=("value",), forecast_horizon=7, period=7, k=2, algo_name="simple")
    backfill_config = BackfillConfig(forecast_config=forecast_config, initial_train_size=14, horizon=horizon, step=7)
    result = BackfillRunner(backfill_config).run(df=backfill_series)
    assert set(result.result_df[HORIZON_STEP_COL].unique()) == set(range(1, horizon + 1))


def test_backfill_runner_forecast_correctness(backfill_series):
    """Verify step-1 forecast at the first cutoff (row 14, values 1..14).

    period=7, k=2: future_pos=14, lookbacks at positions 7 and 0 → values 8.0, 1.0 → mean=4.5.
    The first cutoff timestamp is 2024-01-14 (iloc[13]).
    """
    forecast_config = make_forecast_config(value_cols=("value",), forecast_horizon=7, period=7, k=2)
    backfill_config = BackfillConfig(forecast_config=forecast_config, initial_train_size=14, horizon=7, step=7)
    result = BackfillRunner(backfill_config).run(df=backfill_series)
    first_cutoff_ts = backfill_series[TIME_COL].iloc[13]  # last row of first training window
    step1 = result.result_df[(result.result_df[HORIZON_STEP_COL] == 1) & (result.result_df[CUTOFF_COL] == first_cutoff_ts)]
    assert np.isclose(step1[FORECAST_COL].iloc[0], 4.5)


def test_predict_agg_median(linear_series):
    """Median agg returns median of lookback values, not mean."""
    config = make_forecast_config(period=7, k=3, forecast_horizon=1)
    algo = SimpleForecastAlgo(algo_params={"period": 7, "k": 3, "agg": "median"})
    algo.fit(df=linear_series, config=config)
    result = algo.predict()

    future_rows = result.result_df[result.result_df[ACTUAL_COL].isna()]
    # n=21, step=1, future_pos=21: lookbacks at 14, 7, 0 → values 15, 8, 1 → median=8.0
    assert np.isclose(future_rows[FORECAST_COL].iloc[0], 8.0)


def test_predict_agg_median_differs_from_mean():
    """When lookback values are skewed, median and mean differ."""
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    # Values: 1, 1, 1, 1, 1, 1, 100 — mean != median
    series = pd.DataFrame({TIME_COL: dates, "value": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]})

    config_mean = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("value",),
        freq="D",
        forecast_horizon=1,
        algo_name="simple",
        algo_params={"period": 1, "k": 6, "agg": "mean"},
    )
    config_median = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("value",),
        freq="D",
        forecast_horizon=1,
        algo_name="simple",
        algo_params={"period": 1, "k": 6, "agg": "median"},
    )
    # Train on first 6 rows (values 1,1,1,1,1,1), forecast step 1 at pos 6 looks at 5,4,3,2,1,0
    train = series.iloc[:6].reset_index(drop=True)

    algo_mean = SimpleForecastAlgo(algo_params={"period": 1, "k": 6, "agg": "mean"})
    algo_mean.fit(df=train, config=config_mean)
    fc_mean = algo_mean.predict().result_df
    mean_forecast = fc_mean[fc_mean[ACTUAL_COL].isna()][FORECAST_COL].iloc[0]

    algo_median = SimpleForecastAlgo(algo_params={"period": 1, "k": 6, "agg": "median"})
    algo_median.fit(df=train, config=config_median)
    fc_median = algo_median.predict().result_df
    median_forecast = fc_median[fc_median[ACTUAL_COL].isna()][FORECAST_COL].iloc[0]

    # All training values are 1, so mean == median == 1 here; use skewed future lookback test above
    assert np.isclose(mean_forecast, 1.0)
    assert np.isclose(median_forecast, 1.0)


def test_predict_invalid_agg_raises(linear_series):
    config = make_forecast_config(period=7, k=3, forecast_horizon=1, agg="mode")
    algo = SimpleForecastAlgo()
    algo.fit(df=linear_series, config=config)
    with pytest.raises(ValueError, match="agg"):
        algo.predict()

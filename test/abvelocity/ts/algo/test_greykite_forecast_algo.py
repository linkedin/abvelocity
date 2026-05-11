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
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini
"""Tests for greykite_forecast_algo conditional import and registry behaviour."""

import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.algo import greykite_forecast_algo
from abvelocity.ts.algo.base import ALGO_REGISTRY
from abvelocity.ts.algo.greykite_forecast_algo import GREYKITE_AVAILABLE


def test_greykite_available_flag_is_bool():
    assert isinstance(GREYKITE_AVAILABLE, bool)


def test_greykite_forecast_module_loads_cleanly():
    """The module must be importable regardless of whether greykite is installed."""
    import abvelocity.ts.algo.greykite_forecast_algo  # noqa: F401


def test_greykite_forecast_registry_conditional():
    """GreykiteForecastAlgo is registered iff greykite is available."""
    if GREYKITE_AVAILABLE:
        assert "greykite" in ALGO_REGISTRY
        from abvelocity.ts.algo.greykite_forecast_algo import GreykiteForecastAlgo

        assert ALGO_REGISTRY["greykite"] is GreykiteForecastAlgo
    else:
        assert not hasattr(greykite_forecast_algo, "GreykiteForecastAlgo")


def test_greykite_forecast_class_is_ts_algo_subclass():
    if not GREYKITE_AVAILABLE:
        pytest.skip("blah.greykite not installed")
    from abvelocity.ts.algo.base import TSAlgo
    from abvelocity.ts.algo.greykite_forecast_algo import GreykiteForecastAlgo

    assert issubclass(GreykiteForecastAlgo, TSAlgo)


def test_greykite_forecast_predict_before_fit_raises():
    """predict() before fit() raises ValueError."""
    if not GREYKITE_AVAILABLE:
        pytest.skip("blah.greykite not installed")
    from abvelocity.ts.algo.greykite_forecast_algo import GreykiteForecastAlgo

    algo = GreykiteForecastAlgo()
    with pytest.raises(ValueError, match="fit"):
        algo.predict()


@pytest.fixture
def daily_df():
    """180 days of simple synthetic daily data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    values = 100 + np.arange(180) * 0.5 + rng.normal(0, 2, 180)
    return pd.DataFrame({"ts": dates, "value": values})


def test_greykite_forecast_fit_predict_univariate(daily_df):
    """fit() + predict() round-trip returns a ForecastResult with expected columns."""
    if not GREYKITE_AVAILABLE:
        pytest.skip("blah.greykite not installed")
    from abvelocity.ts.algo.greykite_forecast_algo import GreykiteForecastAlgo
    from abvelocity.ts.config.forecast_config import ForecastConfig
    from abvelocity.ts.result.forecast_result import ForecastResult

    config = ForecastConfig(
        time_col="ts",
        value_cols=("value",),
        freq="D",
        coverage=0.95,
        forecast_horizon=7,
        algo_params={
            "model_template": "SILVERKITE",
            "model_components": {"custom": {"fit_algorithm_dict": {"fit_algorithm": "ridge"}}},
        },
    )
    algo = GreykiteForecastAlgo()
    algo.fit(df=daily_df, config=config)
    result = algo.predict()

    assert isinstance(result, ForecastResult)
    assert result.result_df is not None
    df = result.result_df
    assert list(df.columns[:3]) == ["ts", "metric_id", "actual"]
    assert "forecast" in df.columns
    assert set(df["metric_id"].unique()) == {"value"}
    # result_df contains training history + 7 forecast rows; check forecast window
    train_end = daily_df["ts"].max()
    assert len(df[df["ts"] > train_end]) == 7


def test_greykite_forecast_prediction_window_filter(daily_df):
    """prediction_window trims the returned forecast DataFrame."""
    if not GREYKITE_AVAILABLE:
        pytest.skip("blah.greykite not installed")
    from abvelocity.ts.algo.greykite_forecast_algo import GreykiteForecastAlgo
    from abvelocity.ts.config.forecast_config import ForecastConfig

    config = ForecastConfig(
        time_col="ts",
        value_cols=("value",),
        freq="D",
        forecast_horizon=14,
        algo_params={"model_template": "SILVERKITE"},
    )
    algo = GreykiteForecastAlgo()
    algo.fit(df=daily_df, config=config)

    # Request only the first 3 days of the 14-day horizon
    train_end = daily_df["ts"].max()
    start = str((train_end + pd.Timedelta(days=1)).date())
    end = str((train_end + pd.Timedelta(days=3)).date())
    result = algo.predict(prediction_window=(start, end))

    assert result.result_df is not None
    assert len(result.result_df) == 3


def test_greykite_forecast_fit_predict_with_anomaly_df(daily_df):
    """anomaly_df is accepted and passed through without error."""
    if not GREYKITE_AVAILABLE:
        pytest.skip("blah.greykite not installed")
    from abvelocity.ts.algo.greykite_forecast_algo import GreykiteForecastAlgo
    from abvelocity.ts.config.forecast_config import ForecastConfig

    anomaly_df = pd.DataFrame(
        {
            "start_ts": ["2023-02-01"],
            "end_ts": ["2023-02-07"],
        }
    )
    config = ForecastConfig(
        time_col="ts",
        value_cols=("value",),
        freq="D",
        forecast_horizon=7,
        algo_params={"model_template": "SILVERKITE"},
    )
    algo = GreykiteForecastAlgo()
    algo.fit(df=daily_df, config=config, anomaly_df=anomaly_df)
    result = algo.predict()
    assert result.result_df is not None

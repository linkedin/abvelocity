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
"""Tests for ForecastRunner."""

from dataclasses import dataclass

import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401 — registers "simple"
import pandas as pd
import pytest
from abvelocity.ts.algo.base import ALGO_REGISTRY, TSAlgo
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import ALGO_NAME_COL, METRIC_ID_COL, LAST_TRAINING_DATE_COL
from abvelocity.ts.forecast_runner import ForecastRunner
from abvelocity.ts.result.detect_result import DetectResult
from abvelocity.ts.result.forecast_result import ForecastResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_df():
    return pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=21, freq="D"),
            "value": range(21),
        }
    )


@dataclass
class DetectReturningAlgo(TSAlgo):
    """Always returns a DetectResult — used to test the type guard."""

    def fit(self, df, config, anomaly_df=None):
        return self

    def predict(self, df=None, prediction_window=None):
        return DetectResult()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forecast_runner_returns_forecast_result(daily_df):
    config = ForecastConfig(algo_name="simple", value_cols=("value",), freq="D", forecast_horizon=3)
    result = ForecastRunner(config).run(daily_df)
    assert isinstance(result, ForecastResult)


def test_forecast_runner_result_df_has_metric_col(daily_df):
    config = ForecastConfig(algo_name="simple", value_cols=("value",), freq="D", forecast_horizon=3)
    result = ForecastRunner(config).run(daily_df)
    assert result.result_df is not None
    assert METRIC_ID_COL in result.result_df.columns


def test_forecast_runner_stamps_algo_ver_and_train_end_date(daily_df):
    config = ForecastConfig(
        algo_name="simple",
        value_cols=("value",),
        freq="D",
        forecast_horizon=3,
        train_end_date="2024-01-21",
    )
    result = ForecastRunner(config).run(daily_df)
    assert result.result_df is not None
    assert (result.result_df[ALGO_NAME_COL] == "simple").all()
    assert (result.result_df[LAST_TRAINING_DATE_COL] == "2024-01-21").all()


def test_forecast_runner_prediction_window_filters_rows(daily_df):
    config = ForecastConfig(algo_name="simple", value_cols=("value",), freq="D", forecast_horizon=7)
    result = ForecastRunner(config).run(daily_df, prediction_window=("2024-01-10", "2024-01-12"))
    assert result.result_df is not None
    ts_col = result.result_df["ts"]
    assert ts_col.min() >= pd.Timestamp("2024-01-10")
    assert ts_col.max() <= pd.Timestamp("2024-01-12")


def test_forecast_runner_wrong_result_type_raises_type_error(monkeypatch, daily_df):
    monkeypatch.setitem(ALGO_REGISTRY, "_detect_returning", DetectReturningAlgo)
    config = ForecastConfig(algo_name="_detect_returning", value_cols=("value",), freq="D")
    with pytest.raises(TypeError, match="DetectResult"):
        ForecastRunner(config).run(daily_df)


def test_forecast_runner_type_error_message_mentions_detect_runner(monkeypatch, daily_df):
    monkeypatch.setitem(ALGO_REGISTRY, "_detect_returning2", DetectReturningAlgo)
    config = ForecastConfig(algo_name="_detect_returning2", value_cols=("value",), freq="D")
    with pytest.raises(TypeError, match="AnomalyDetectRunner"):
        ForecastRunner(config).run(daily_df)


def test_forecast_runner_passes_anomaly_df(monkeypatch, daily_df):
    received = {}

    @dataclass
    class CapturingAlgo(TSAlgo):
        def fit(self, df, config, anomaly_df=None):
            received["anomaly_df"] = anomaly_df
            return self

        def predict(self, df=None, prediction_window=None):
            return ForecastResult()

    monkeypatch.setitem(ALGO_REGISTRY, "_capturing_fc", CapturingAlgo)
    config = ForecastConfig(algo_name="_capturing_fc", value_cols=("value",), freq="D")
    anomaly_df = pd.DataFrame({"start_ts": ["2024-01-05"], "end_ts": ["2024-01-06"]})
    ForecastRunner(config).run(daily_df, anomaly_df=anomaly_df)
    assert received["anomaly_df"] is anomaly_df

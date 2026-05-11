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
"""Tests for AnomalyDetectRunner."""

from dataclasses import dataclass

import pandas as pd
import pytest
from abvelocity.ts.algo.base import ALGO_REGISTRY, TSAlgo
from abvelocity.ts.config.detect_config import DetectConfig
from abvelocity.ts.detect_runner import AnomalyDetectRunner
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
class MinimalDetectAlgo(TSAlgo):
    """Minimal detect algo: always returns a DetectResult."""

    def fit(self, df, config, anomaly_df=None):
        return self

    def predict(self, df=None, prediction_window=None):
        return DetectResult()


@dataclass
class ForecastReturningAlgo(TSAlgo):
    """Always returns a ForecastResult — used to test the type guard."""

    def fit(self, df, config, anomaly_df=None):
        return self

    def predict(self, df=None, prediction_window=None):
        return ForecastResult()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_runner_returns_detect_result(monkeypatch, daily_df):
    monkeypatch.setitem(ALGO_REGISTRY, "_minimal_detect", MinimalDetectAlgo)
    config = DetectConfig(algo_name="_minimal_detect", value_cols=("value",), freq="D")
    result = AnomalyDetectRunner(config).run(daily_df)
    assert isinstance(result, DetectResult)


def test_detect_runner_wrong_result_type_raises_type_error(monkeypatch, daily_df):
    monkeypatch.setitem(ALGO_REGISTRY, "_fc_returning", ForecastReturningAlgo)
    config = DetectConfig(algo_name="_fc_returning", value_cols=("value",), freq="D")
    with pytest.raises(TypeError, match="ForecastResult"):
        AnomalyDetectRunner(config).run(daily_df)


def test_detect_runner_type_error_message_mentions_forecast_runner(monkeypatch, daily_df):
    monkeypatch.setitem(ALGO_REGISTRY, "_fc_returning2", ForecastReturningAlgo)
    config = DetectConfig(algo_name="_fc_returning2", value_cols=("value",), freq="D")
    with pytest.raises(TypeError, match="ForecastRunner"):
        AnomalyDetectRunner(config).run(daily_df)


def test_detect_runner_passes_anomaly_df(monkeypatch, daily_df):
    received = {}

    @dataclass
    class CapturingDetectAlgo(TSAlgo):
        def fit(self, df, config, anomaly_df=None):
            received["anomaly_df"] = anomaly_df
            return self

        def predict(self, df=None, prediction_window=None):
            return DetectResult()

    monkeypatch.setitem(ALGO_REGISTRY, "_capturing_detect", CapturingDetectAlgo)
    config = DetectConfig(algo_name="_capturing_detect", value_cols=("value",), freq="D")
    anomaly_df = pd.DataFrame({"start_ts": ["2024-01-05"], "end_ts": ["2024-01-06"]})
    AnomalyDetectRunner(config).run(daily_df, anomaly_df=anomaly_df)
    assert received["anomaly_df"] is anomaly_df


def test_detect_runner_passes_prediction_window(monkeypatch, daily_df):
    received = {}

    @dataclass
    class WindowCapturingAlgo(TSAlgo):
        def fit(self, df, config, anomaly_df=None):
            return self

        def predict(self, df=None, prediction_window=None):
            received["prediction_window"] = prediction_window
            return DetectResult()

    monkeypatch.setitem(ALGO_REGISTRY, "_window_detect", WindowCapturingAlgo)
    config = DetectConfig(algo_name="_window_detect", value_cols=("value",), freq="D")
    window = ("2024-01-10", "2024-01-20")
    AnomalyDetectRunner(config).run(daily_df, prediction_window=window)
    assert received["prediction_window"] == window

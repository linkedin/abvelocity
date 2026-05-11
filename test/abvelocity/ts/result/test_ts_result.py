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
"""Tests for TSResult, ForecastResult, and DetectResult."""

import pandas as pd
from abvelocity.ts.result.detect_result import DetectResult
from abvelocity.ts.result.forecast_result import ForecastResult
from abvelocity.ts.result.ts_result import TSResult


def test_ts_result_defaults():
    result = TSResult()
    assert result.result_df is None
    assert result.fit_info is None


def test_ts_result_with_fields():
    df = pd.DataFrame(
        {
            "ts": ["2024-01-01", "2024-01-02"],
            "metric": ["value", "value"],
            "actual": [1.0, 2.0],
            "forecast": [1.1, 2.1],
        }
    )
    result = TSResult(result_df=df, fit_info={"rmse": 0.5})
    assert result.result_df is not None
    assert list(result.result_df.columns) == ["ts", "metric", "actual", "forecast"]
    assert result.fit_info == {"rmse": 0.5}


def test_ts_result_json_round_trip_no_df():
    result = TSResult(fit_info={"metric": "mae", "value": 1.23})
    restored = TSResult.from_json(result.to_json())
    assert restored.result_df is None
    assert restored.fit_info == {"metric": "mae", "value": 1.23}


def test_ts_result_json_round_trip_with_df():
    df = pd.DataFrame(
        {
            "ts": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "metric": ["value", "value", "value"],
            "actual": [1.0, 2.0, 3.0],
            "forecast": [1.1, 2.1, 3.1],
            "forecast_lower": [0.9, 1.9, 2.9],
            "forecast_upper": [1.3, 2.3, 3.3],
        }
    )
    result = TSResult(result_df=df, fit_info={"key": "val"})
    restored = TSResult.from_json(result.to_json())
    assert restored.result_df is not None
    assert list(restored.result_df.columns) == list(df.columns)
    assert len(restored.result_df) == 3
    assert restored.fit_info == {"key": "val"}


def test_forecast_result_defaults():
    result = ForecastResult()
    assert result.result_df is None
    assert result.fit_info is None


def test_forecast_result_is_ts_result_subclass():
    result = ForecastResult()
    assert isinstance(result, TSResult)


def test_forecast_result_json_round_trip():
    df = pd.DataFrame({"ts": ["2024-01-01"], "value": [5.0]})
    result = ForecastResult(result_df=df, fit_info={"horizon": 1})
    restored = ForecastResult.from_json(result.to_json())
    assert restored.result_df is not None
    assert restored.fit_info == {"horizon": 1}


def test_detect_result_defaults():
    result = DetectResult()
    assert result.result_df is None
    assert result.anomalies_df is None
    assert result.fit_info is None


def test_detect_result_is_ts_result_subclass():
    result = DetectResult()
    assert isinstance(result, TSResult)


def test_detect_result_with_both_dfs():
    result_df = pd.DataFrame(
        {
            "ts": ["2024-01-01", "2024-01-02"],
            "metric": ["value", "value"],
            "actual": [1.0, 2.0],
            "forecast": [1.05, 2.05],
            "anomaly": [0, 1],
            "anomaly_score": [0.1, 0.9],
        }
    )
    anomalies_df = pd.DataFrame({"metric": ["value"], "start_ts": ["2024-01-02"], "end_ts": ["2024-01-02"]})
    result = DetectResult(result_df=result_df, anomalies_df=anomalies_df)
    assert result.result_df is not None
    assert result.anomalies_df is not None
    assert "metric" in result.anomalies_df.columns
    assert "start_ts" in result.anomalies_df.columns
    assert "end_ts" in result.anomalies_df.columns


def test_detect_result_json_round_trip():
    result_df = pd.DataFrame(
        {
            "ts": ["2024-01-01"],
            "metric": ["value"],
            "actual": [1.0],
            "forecast": [1.05],
        }
    )
    anomalies_df = pd.DataFrame({"metric": ["value"], "start_ts": ["2024-01-01"], "end_ts": ["2024-01-01"]})
    result = DetectResult(result_df=result_df, anomalies_df=anomalies_df, fit_info={"n": 1})
    restored = DetectResult.from_json(result.to_json())
    assert restored.result_df is not None
    assert restored.anomalies_df is not None
    assert list(restored.anomalies_df.columns) == ["metric", "start_ts", "end_ts"]
    assert restored.fit_info == {"n": 1}

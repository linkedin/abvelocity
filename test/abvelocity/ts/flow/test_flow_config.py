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
"""Tests for TSFlowConfig."""

import pytest
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.flow.flow_config import TSFlowConfig
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig


def make_ts_metrics_config() -> TSMetricsConfig:
    return TSMetricsConfig(time_col="event_ts", freq="D", dialect="duckdb")


def make_forecast_config() -> ForecastConfig:
    return ForecastConfig(algo_name="simple", forecast_horizon=7)


def make_backfill_config() -> BackfillConfig:
    return BackfillConfig(
        forecast_config=make_forecast_config(),
        initial_train_size=30,
        horizon=7,
    )


def test_forecast_mode_defaults():
    cfg = TSFlowConfig(
        ts_metrics_config=make_ts_metrics_config(),
        ts_model_config=make_forecast_config(),
    )
    assert cfg.mode == "forecast"
    assert cfg.backfill_config is None
    assert cfg.prediction_window is None
    assert cfg.eval is None


def test_detect_mode_valid():
    from abvelocity.ts.config.detect_config import DetectConfig

    detect_cfg = DetectConfig(forecast_config=make_forecast_config(), algo_name="")
    cfg = TSFlowConfig(
        ts_metrics_config=make_ts_metrics_config(),
        ts_model_config=detect_cfg,
        mode="detect",
    )
    assert cfg.mode == "detect"


def test_backfill_mode_valid():
    cfg = TSFlowConfig(
        ts_metrics_config=make_ts_metrics_config(),
        backfill_config=make_backfill_config(),
        mode="backfill",
    )
    assert cfg.mode == "backfill"
    assert cfg.ts_model_config is None


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        TSFlowConfig(
            ts_metrics_config=make_ts_metrics_config(),
            ts_model_config=make_forecast_config(),
            mode="unknown",
        )


def test_backfill_without_backfill_config_raises():
    with pytest.raises(ValueError, match="backfill_config must be set"):
        TSFlowConfig(
            ts_metrics_config=make_ts_metrics_config(),
            mode="backfill",
        )


def test_forecast_without_ts_model_config_raises():
    with pytest.raises(ValueError, match="ts_model_config must be set"):
        TSFlowConfig(
            ts_metrics_config=make_ts_metrics_config(),
            mode="forecast",
        )


def test_detect_without_ts_model_config_raises():
    with pytest.raises(ValueError, match="ts_model_config must be set"):
        TSFlowConfig(
            ts_metrics_config=make_ts_metrics_config(),
            mode="detect",
        )


def test_eval_and_window_stored():
    from abvelocity.ts.eval import ForecastEval

    evaluator = ForecastEval(metrics=("mae", "rmse"), group_by=("metric_id",))
    cfg = TSFlowConfig(
        ts_metrics_config=make_ts_metrics_config(),
        ts_model_config=make_forecast_config(),
        prediction_window=("2024-01-01", "2024-03-31"),
        eval=evaluator,
    )
    assert cfg.prediction_window == ("2024-01-01", "2024-03-31")
    assert cfg.eval is evaluator
    assert cfg.eval.metrics == ("mae", "rmse")
    assert cfg.eval.group_by == ("metric_id",)

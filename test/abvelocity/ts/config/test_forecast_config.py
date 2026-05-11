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
"""Tests for ForecastConfig."""

import pytest
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.config.ts_model_config import TSModelConfig
from abvelocity.ts.constants import TIME_COL


def test_forecast_config_defaults():
    cfg = ForecastConfig()
    assert cfg.forecast_horizon == 1
    # Inherited TSModelConfig defaults
    assert cfg.time_col == TIME_COL
    assert cfg.value_cols is None
    assert cfg.regressor_cols == ()
    assert cfg.coverage == 0.95
    assert cfg.algo_name == ""


def test_forecast_config_is_ts_config_subclass():
    cfg = ForecastConfig()
    assert isinstance(cfg, TSModelConfig)


def test_forecast_config_custom_horizon():
    cfg = ForecastConfig(forecast_horizon=14)
    assert cfg.forecast_horizon == 14


def test_forecast_config_inherits_ts_config_fields():
    cfg = ForecastConfig(
        value_cols=("y",),
        freq="H",
        train_end_date="2024-03-01",
        forecast_horizon=7,
        coverage=0.8,
    )
    assert cfg.value_cols == ("y",)
    assert cfg.freq == "H"
    assert cfg.train_end_date == "2024-03-01"
    assert cfg.forecast_horizon == 7
    assert cfg.coverage == 0.8


def test_forecast_config_json_round_trip():
    cfg = ForecastConfig(
        value_cols=("y1", "y2"),
        freq="D",
        forecast_horizon=14,
        coverage=0.9,
        algo_params={"key": "value"},
    )
    restored = ForecastConfig.from_json(cfg.to_json())
    assert restored == cfg


def test_forecast_config_invalid_coverage():
    with pytest.raises(ValueError, match="coverage"):
        ForecastConfig(coverage=2.0)


def test_forecast_config_empty_value_cols():
    with pytest.raises(ValueError, match="value_cols"):
        ForecastConfig(value_cols=())

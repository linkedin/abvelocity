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
"""Tests for DetectConfig."""

import pytest
from abvelocity.ts.config.detect_config import DetectConfig
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.config.ts_model_config import TSModelConfig


def test_detect_config_defaults():
    cfg = DetectConfig()
    assert cfg.forecast_config is None
    assert cfg.algo_name == ""
    assert cfg.coverage == 0.95


def test_detect_config_is_ts_config_subclass():
    cfg = DetectConfig()
    assert isinstance(cfg, TSModelConfig)


def test_detect_config_with_forecast_config():
    fc = ForecastConfig(forecast_horizon=7, freq="D")
    cfg = DetectConfig(forecast_config=fc, algo_name="greykite_detect")
    assert cfg.forecast_config is not None
    assert cfg.forecast_config.forecast_horizon == 7
    assert cfg.forecast_config.freq == "D"
    assert cfg.algo_name == "greykite_detect"


def test_detect_config_forecast_config_independence():
    """algo_name on DetectConfig controls detection; forecast_config.algo_name controls forecast."""
    fc = ForecastConfig(algo_name="greykite", forecast_horizon=3)
    cfg = DetectConfig(algo_name="greykite_detect", forecast_config=fc)
    assert cfg.algo_name == "greykite_detect"
    assert cfg.forecast_config.algo_name == "greykite"


def test_detect_config_json_round_trip_no_forecast():
    cfg = DetectConfig(algo_name="greykite_detect", coverage=0.9)
    restored = DetectConfig.from_json(cfg.to_json())
    assert restored == cfg
    assert restored.forecast_config is None


def test_detect_config_json_round_trip_with_forecast():
    fc = ForecastConfig(
        forecast_horizon=3,
        coverage=0.9,
        freq="H",
        algo_params={"model_template": "SILVERKITE"},
    )
    cfg = DetectConfig(
        forecast_config=fc,
        algo_name="greykite_detect",
        coverage=0.95,
        value_cols=("metric_a",),
    )
    restored = DetectConfig.from_json(cfg.to_json())
    assert restored == cfg
    assert restored.forecast_config is not None
    assert restored.forecast_config.forecast_horizon == 3
    assert restored.forecast_config.algo_params == {"model_template": "SILVERKITE"}


def test_detect_config_invalid_coverage():
    with pytest.raises(ValueError, match="coverage"):
        DetectConfig(coverage=1.1)

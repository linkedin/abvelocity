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
"""Tests for BackfillConfig."""

import pytest
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.config.forecast_config import ForecastConfig


@pytest.fixture
def base_fc():
    return ForecastConfig(value_cols=("y",), freq="D", forecast_horizon=7)


def test_backfill_config_defaults(base_fc):
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=30, horizon=7)
    assert cfg.step == 1
    assert cfg.window_type == "expanding"
    assert cfg.window_size is None
    assert cfg.n_windows is None


def test_backfill_config_rolling(base_fc):
    cfg = BackfillConfig(
        forecast_config=base_fc,
        initial_train_size=30,
        horizon=7,
        window_type="rolling",
        window_size=60,
    )
    assert cfg.window_type == "rolling"
    assert cfg.window_size == 60


def test_backfill_config_n_windows(base_fc):
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=30, horizon=7, n_windows=12)
    assert cfg.n_windows == 12


def test_backfill_config_json_round_trip(base_fc):
    cfg = BackfillConfig(
        forecast_config=base_fc,
        initial_train_size=30,
        horizon=7,
        step=3,
        window_type="rolling",
        window_size=60,
        n_windows=10,
    )
    restored = BackfillConfig.from_json(cfg.to_json())
    assert restored == cfg


def test_backfill_config_invalid_window_type(base_fc):
    with pytest.raises(ValueError, match="window_type"):
        BackfillConfig(forecast_config=base_fc, initial_train_size=30, horizon=7, window_type="bad")


def test_backfill_config_rolling_missing_window_size(base_fc):
    with pytest.raises(ValueError, match="window_size"):
        BackfillConfig(forecast_config=base_fc, initial_train_size=30, horizon=7, window_type="rolling")


def test_backfill_config_invalid_initial_train_size(base_fc):
    with pytest.raises(ValueError, match="initial_train_size"):
        BackfillConfig(forecast_config=base_fc, initial_train_size=0, horizon=7)


def test_backfill_config_invalid_horizon(base_fc):
    with pytest.raises(ValueError, match="horizon"):
        BackfillConfig(forecast_config=base_fc, initial_train_size=30, horizon=0)


def test_backfill_config_invalid_step(base_fc):
    with pytest.raises(ValueError, match="step"):
        BackfillConfig(forecast_config=base_fc, initial_train_size=30, horizon=7, step=0)


def test_backfill_config_invalid_n_windows(base_fc):
    with pytest.raises(ValueError, match="n_windows"):
        BackfillConfig(forecast_config=base_fc, initial_train_size=30, horizon=7, n_windows=0)


def test_backfill_config_horizon_exceeds_forecast_horizon():
    fc = ForecastConfig(value_cols=("y",), freq="D", forecast_horizon=5)
    with pytest.raises(ValueError, match="forecast_horizon"):
        BackfillConfig(forecast_config=fc, initial_train_size=30, horizon=7)


def test_backfill_config_invalid_window_size(base_fc):
    with pytest.raises(ValueError, match="window_size"):
        BackfillConfig(
            forecast_config=base_fc,
            initial_train_size=30,
            horizon=7,
            window_type="rolling",
            window_size=0,
        )

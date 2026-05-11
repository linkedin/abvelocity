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
"""Tests for TSModelConfig."""

import pytest
from abvelocity.ts.config.ts_model_config import TSModelConfig
from abvelocity.ts.constants import TIME_COL


def test_ts_config_defaults():
    cfg = TSModelConfig()
    assert cfg.time_col == TIME_COL
    assert cfg.value_cols is None
    assert cfg.regressor_cols == ()
    assert cfg.freq is None
    assert cfg.train_end_date is None
    assert cfg.coverage == 0.95
    assert cfg.algo_name == ""
    assert cfg.algo_params is None


def test_ts_config_custom_fields():
    cfg = TSModelConfig(
        time_col="timestamp",
        value_cols=("metric1", "metric2"),
        regressor_cols=("reg1", "reg2"),
        freq="D",
        train_end_date="2024-06-30",
        coverage=0.9,
        algo_name="my_algo",
        algo_params={"param_a": 1, "param_b": "v"},
    )
    assert cfg.time_col == "timestamp"
    assert cfg.value_cols == ("metric1", "metric2")
    assert cfg.regressor_cols == ("reg1", "reg2")
    assert cfg.freq == "D"
    assert cfg.train_end_date == "2024-06-30"
    assert cfg.coverage == 0.9
    assert cfg.algo_name == "my_algo"
    assert cfg.algo_params == {"param_a": 1, "param_b": "v"}


def test_ts_config_json_round_trip():
    cfg = TSModelConfig(
        value_cols=("metric1", "metric2"),
        regressor_cols=("reg1",),
        freq="D",
        train_end_date="2024-01-01",
        coverage=0.9,
        algo_name="my_algo",
        algo_params={"key": "val", "num": 42},
    )
    restored = TSModelConfig.from_json(cfg.to_json())
    assert restored == cfg


def test_ts_config_json_round_trip_defaults():
    cfg = TSModelConfig()
    restored = TSModelConfig.from_json(cfg.to_json())
    assert restored == cfg


def test_ts_config_invalid_coverage_above_one():
    with pytest.raises(ValueError, match="coverage"):
        TSModelConfig(coverage=1.5)


def test_ts_config_invalid_coverage_equals_one():
    with pytest.raises(ValueError, match="coverage"):
        TSModelConfig(coverage=1.0)


def test_ts_config_invalid_coverage_zero():
    with pytest.raises(ValueError, match="coverage"):
        TSModelConfig(coverage=0.0)


def test_ts_config_invalid_coverage_negative():
    with pytest.raises(ValueError, match="coverage"):
        TSModelConfig(coverage=-0.1)


def test_ts_config_empty_value_cols():
    with pytest.raises(ValueError, match="value_cols"):
        TSModelConfig(value_cols=())


# ---------------------------------------------------------------------------
# get_algo_params
# ---------------------------------------------------------------------------


def test_get_algo_params_returns_empty_dict_when_nothing_set():
    cfg = TSModelConfig()
    assert cfg.get_algo_params("any_metric") == {}


def test_get_algo_params_returns_algo_params_when_no_override():
    cfg = TSModelConfig(algo_params={"k": 3, "period": 7})
    assert cfg.get_algo_params("dau") == {"k": 3, "period": 7}


def test_get_algo_params_merges_override_into_base():
    # Only k is overridden for revenue; period stays from the common base.
    cfg = TSModelConfig(
        algo_params={"k": 3, "period": 7},
        algo_params_by_metric={"revenue": {"k": 1}},
    )
    assert cfg.get_algo_params("revenue") == {"k": 1, "period": 7}


def test_get_algo_params_falls_back_for_unspecified_metric():
    cfg = TSModelConfig(
        algo_params={"k": 3, "period": 7},
        algo_params_by_metric={"revenue": {"k": 1}},
    )
    assert cfg.get_algo_params("dau") == {"k": 3, "period": 7}


def test_get_algo_params_does_not_mutate_base():
    # Merging should return a new dict, not modify algo_params in place.
    cfg = TSModelConfig(
        algo_params={"k": 3, "period": 7},
        algo_params_by_metric={"revenue": {"k": 1}},
    )
    cfg.get_algo_params("revenue")
    assert cfg.algo_params == {"k": 3, "period": 7}


def test_get_algo_params_override_without_global_algo_params():
    cfg = TSModelConfig(algo_params_by_metric={"dau": {"period": 7}})
    assert cfg.get_algo_params("dau") == {"period": 7}
    assert cfg.get_algo_params("revenue") == {}


def test_ts_config_json_round_trip_with_algo_params_by_metric():
    cfg = TSModelConfig(
        algo_params={"k": 3},
        algo_params_by_metric={"dau": {"k": 7, "period": 7}, "revenue": {"k": 4}},
    )
    restored = TSModelConfig.from_json(cfg.to_json())
    assert restored == cfg
    assert restored.algo_params_by_metric == {"dau": {"k": 7, "period": 7}, "revenue": {"k": 4}}

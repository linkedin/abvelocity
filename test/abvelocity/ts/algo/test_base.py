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
"""Tests for TSAlgo abstract base class and ALGO_REGISTRY."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from abvelocity.ts.algo.base import ALGO_REGISTRY, TSAlgo
from abvelocity.ts.config.ts_model_config import TSModelConfig
from abvelocity.ts.result.ts_result import TSResult


@dataclass
class ConcreteAlgo(TSAlgo):
    """Minimal concrete TSAlgo implementation for testing."""

    def fit(
        self,
        df: pd.DataFrame,
        config: TSModelConfig,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> "ConcreteAlgo":
        return self

    def predict(
        self,
        df: Optional[pd.DataFrame] = None,
        prediction_window: Optional[tuple[str, str]] = None,
    ) -> TSResult:
        return TSResult(fit_info={"predicted": True})


def test_concrete_algo_instantiation_no_params():
    algo = ConcreteAlgo()
    assert algo.algo_params == {}


def test_concrete_algo_instantiation_with_params():
    algo = ConcreteAlgo(algo_params={"param1": "value1", "num": 42})
    assert algo.algo_params == {"param1": "value1", "num": 42}


def test_concrete_algo_none_params_normalised_to_empty_dict():
    algo = ConcreteAlgo(algo_params=None)
    assert algo.algo_params == {}


def test_concrete_algo_is_ts_algo_instance():
    algo = ConcreteAlgo()
    assert isinstance(algo, TSAlgo)


def test_algo_registry_register_and_lookup(monkeypatch):
    monkeypatch.setitem(ALGO_REGISTRY, "_test_concrete", ConcreteAlgo)
    cls = ALGO_REGISTRY.get("_test_concrete")
    assert cls is ConcreteAlgo


def test_algo_registry_missing_key_returns_none():
    result = ALGO_REGISTRY.get("__nonexistent_algo__")
    assert result is None


def test_algo_registry_instantiate_from_registry(monkeypatch):
    monkeypatch.setitem(ALGO_REGISTRY, "_test_concrete2", ConcreteAlgo)
    cls = ALGO_REGISTRY["_test_concrete2"]
    instance = cls(algo_params={"x": 1})
    assert isinstance(instance, TSAlgo)
    assert instance.algo_params == {"x": 1}


def test_concrete_algo_fit_returns_self():
    algo = ConcreteAlgo()
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3, freq="D"), "value": [1.0, 2.0, 3.0]})
    config = TSModelConfig()
    result = algo.fit(df=df, config=config)
    assert result is algo


def test_concrete_algo_predict_returns_ts_result():
    algo = ConcreteAlgo()
    result = algo.predict()
    assert isinstance(result, TSResult)
    assert result.fit_info == {"predicted": True}


def test_concrete_algo_fit_with_anomaly_df():
    algo = ConcreteAlgo()
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, freq="D"), "value": range(5)})
    anomaly_df = pd.DataFrame({"start_ts": ["2024-01-02"], "end_ts": ["2024-01-03"]})
    config = TSModelConfig()
    result = algo.fit(df=df, config=config, anomaly_df=anomaly_df)
    assert result is algo

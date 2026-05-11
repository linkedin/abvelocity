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
"""Tests for greykite_detect_algo conditional import and registry behaviour."""

import pytest
from abvelocity.ts.algo import greykite_detect_algo
from abvelocity.ts.algo.base import ALGO_REGISTRY
from abvelocity.ts.algo.greykite_detect_algo import GREYKITE_AVAILABLE


def test_greykite_available_flag_is_bool():
    assert isinstance(GREYKITE_AVAILABLE, bool)


def test_greykite_detect_module_loads_cleanly():
    """The module must be importable regardless of whether greykite is installed."""
    import abvelocity.ts.algo.greykite_detect_algo  # noqa: F401


def test_greykite_detect_registry_conditional():
    """GreykiteDetectAlgo is registered iff greykite is available."""
    if GREYKITE_AVAILABLE:
        assert "greykite_detect" in ALGO_REGISTRY
        from abvelocity.ts.algo.greykite_detect_algo import GreykiteDetectAlgo

        assert ALGO_REGISTRY["greykite_detect"] is GreykiteDetectAlgo
    else:
        assert not hasattr(greykite_detect_algo, "GreykiteDetectAlgo")


def test_greykite_detect_class_is_ts_algo_subclass():
    """When available, GreykiteDetectAlgo must be a TSAlgo subclass."""
    if not GREYKITE_AVAILABLE:
        pytest.skip("greykite not installed")
    from abvelocity.ts.algo.base import TSAlgo
    from abvelocity.ts.algo.greykite_detect_algo import GreykiteDetectAlgo

    assert issubclass(GreykiteDetectAlgo, TSAlgo)


def test_greykite_detect_fit_raises_not_implemented():
    """fit() raises NotImplementedError (stub behaviour)."""
    if not GREYKITE_AVAILABLE:
        pytest.skip("greykite not installed")
    import pandas as pd
    from abvelocity.ts.algo.greykite_detect_algo import GreykiteDetectAlgo
    from abvelocity.ts.config.ts_model_config import TSModelConfig

    algo = GreykiteDetectAlgo()
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3, freq="D"), "value": [1, 2, 3]})
    with pytest.raises(NotImplementedError):
        algo.fit(df=df, config=TSModelConfig())


def test_greykite_detect_predict_raises_not_implemented():
    """predict() raises NotImplementedError (stub behaviour)."""
    if not GREYKITE_AVAILABLE:
        pytest.skip("greykite not installed")
    from abvelocity.ts.algo.greykite_detect_algo import GreykiteDetectAlgo

    algo = GreykiteDetectAlgo()
    with pytest.raises(NotImplementedError):
        algo.predict()

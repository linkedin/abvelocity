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
"""Unit tests for ``GreykiteParamConverter.convert`` — the flat-to-nested adapter.

These tests pin the exact nested-dict output for each recognised
search-space key in isolation, plus the merged shape for combined
inputs. The whole class is gated behind
:data:`GREYKITE_AVAILABLE` because the converter only exists when
``blah.greykite`` is importable.
"""

import pytest
from abvelocity.ts.algo.greykite_forecast_algo import GREYKITE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not GREYKITE_AVAILABLE,
    reason="blah.greykite not installed; GreykiteParamConverter is only defined when it is.",
)


def _converter():
    """Return a fresh ``GreykiteParamConverter`` instance.

    Imported lazily so the module collects on Python without greykite —
    the ``pytestmark`` skip kicks in before the import is needed.
    """
    from abvelocity.ts.algo.greykite_forecast_algo import GreykiteParamConverter
    return GreykiteParamConverter()


def test_empty_params_returns_empty_dict():
    assert _converter().convert({}) == {}


def test_model_template_passes_through_at_top_level():
    assert _converter().convert({"model_template": "SILVERKITE_EMPTY"}) == {
        "model_template": "SILVERKITE_EMPTY"
    }


def test_breakdown_origin_passes_through_at_top_level():
    assert _converter().convert({"breakdown_origin": "2024-01-01"}) == {
        "breakdown_origin": "2024-01-01"
    }


def test_fit_algorithm_lands_under_custom_fit_algorithm_dict():
    out = _converter().convert({"fit_algorithm": "ridge"})
    assert out == {
        "model_components": {
            "custom": {"fit_algorithm_dict": {"fit_algorithm": "ridge"}},
        }
    }


def test_regression_weight_col_lands_under_custom():
    out = _converter().convert({"regression_weight_col": "ct1"})
    assert out == {
        "model_components": {"custom": {"regression_weight_col": "ct1"}}
    }


def test_regression_weight_col_none_is_emitted_verbatim():
    """Explicit ``None`` is a real choice (turn weighting off); converter
    should preserve it rather than dropping the key."""
    out = _converter().convert({"regression_weight_col": None})
    assert out == {
        "model_components": {"custom": {"regression_weight_col": None}}
    }


@pytest.mark.parametrize("key", [
    "yearly_seasonality",
    "weekly_seasonality",
    "daily_seasonality",
    "quarterly_seasonality",
    "monthly_seasonality",
])
def test_each_seasonality_key_lands_under_seasonality(key):
    out = _converter().convert({key: 12})
    assert out == {"model_components": {"seasonality": {key: 12}}}


def test_changepoint_reg_lands_under_changepoints_with_method_auto():
    out = _converter().convert({"changepoint_reg": 0.6})
    assert out == {
        "model_components": {
            "changepoints": {
                "changepoints_dict": {
                    "method": "auto",
                    "regularization_strength": 0.6,
                }
            }
        }
    }


def test_unknown_keys_forward_to_top_level_unchanged():
    out = _converter().convert({"some_future_knob": 42})
    assert out == {"some_future_knob": 42}


def test_combined_recognised_keys_merge_into_one_model_components():
    out = _converter().convert({
        "fit_algorithm": "ridge",
        "regression_weight_col": "ct2",
        "yearly_seasonality": 6,
        "changepoint_reg": 0.8,
    })
    assert out == {
        "model_components": {
            "custom": {
                "fit_algorithm_dict": {"fit_algorithm": "ridge"},
                "regression_weight_col": "ct2",
            },
            "seasonality": {"yearly_seasonality": 6},
            "changepoints": {
                "changepoints_dict": {
                    "method": "auto",
                    "regularization_strength": 0.8,
                }
            },
        }
    }


def test_combined_top_level_and_nested_coexist():
    out = _converter().convert({
        "model_template": "SILVERKITE_EMPTY",
        "breakdown_origin": "2024-01-01",
        "yearly_seasonality": 12,
        "x_unknown": "x",
    })
    assert out == {
        "model_template": "SILVERKITE_EMPTY",
        "breakdown_origin": "2024-01-01",
        "x_unknown": "x",
        "model_components": {"seasonality": {"yearly_seasonality": 12}},
    }


def test_converter_is_callable_alias_for_convert():
    converter = _converter()
    payload = {"changepoint_reg": 0.5}
    assert converter(payload) == converter.convert(payload)


def test_recognised_keys_constant_covers_all_handled_keys():
    """Pin the recognised-keys set so future edits don't accidentally
    drop a key from the 'known' list (which would silently start
    forwarding it as unknown)."""
    from abvelocity.ts.algo.greykite_forecast_algo import GreykiteParamConverter

    expected = {
        "model_template",
        "breakdown_origin",
        "fit_algorithm",
        "regression_weight_col",
        "changepoint_reg",
        "yearly_seasonality",
        "weekly_seasonality",
        "daily_seasonality",
        "quarterly_seasonality",
        "monthly_seasonality",
    }
    assert set(GreykiteParamConverter.RECOGNISED_KEYS) == expected

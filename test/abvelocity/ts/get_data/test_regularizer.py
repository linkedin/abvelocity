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
"""Tests for :mod:`abvelocity.ts.get_data.regularizer` —
``regularize_timeseries`` (the function) and ``Regularize`` (the
transform that wraps it)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from abvelocity.ts.get_data.regularizer import Regularize, regularize_timeseries
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig


@dataclass
class StubMetric:
    name: str


@dataclass
class StubMetricInfo:
    dims: Optional[list[str]] = None
    metrics: Optional[list[StubMetric]] = None


def make_ts_config(freq: str = "D") -> TSMetricsConfig:
    return TSMetricsConfig(time_col="event_ts", freq=freq, time_alias="ts")


# ---------------------------------------------------------------------------
# regularize_timeseries
# ---------------------------------------------------------------------------


def test_regularize_timeseries_pads_missing_dates():
    """Single missing day in the middle is padded with NaN."""
    df = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")],
            "signups": [10.0, 20.0, 40.0],
        }
    )
    out = regularize_timeseries(df, time_col="ts", freq="D")
    assert len(out) == 4
    assert pd.isna(out.loc[out["ts"] == pd.Timestamp("2024-01-03"), "signups"].iloc[0])


def test_regularize_timeseries_drops_duplicate_timestamps():
    """Duplicate timestamps → drop second; warns."""
    df = pd.DataFrame(
        {
            "ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ],
            "signups": [10.0, 99.0, 20.0, 30.0],
        }
    )
    with pytest.warns(UserWarning, match="duplicate"):
        out_df = regularize_timeseries(df, time_col="ts", freq="D")
    assert len(out_df) == 3
    assert 99.0 not in list(out_df["signups"])  # second-of-pair was dropped


def test_regularize_timeseries_replaces_inf_with_nan():
    """+inf and -inf land as NaN in numeric columns."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=4, freq="D"),
            "signups": [1.0, np.inf, -np.inf, 4.0],
        }
    )
    out = regularize_timeseries(df, time_col="ts", freq="D")
    assert pd.isna(out["signups"].iloc[1])
    assert pd.isna(out["signups"].iloc[2])
    assert out["signups"].iloc[0] == 1.0
    assert out["signups"].iloc[3] == 4.0


def test_regularize_timeseries_sorts_ascending():
    """Out-of-order rows come back ascending."""
    df = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "signups": [3.0, 1.0, 2.0],
        }
    )
    out = regularize_timeseries(df, time_col="ts", freq="D")
    assert list(out["signups"]) == [1.0, 2.0, 3.0]


def test_regularize_timeseries_coerces_string_dates_to_datetime64():
    """Time column comes in as strings, returns as datetime64[ns]."""
    df = pd.DataFrame(
        {
            "ts": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "signups": [1.0, 2.0, 3.0],
        }
    )
    out = regularize_timeseries(df, time_col="ts", freq="D")
    assert pd.api.types.is_datetime64_dtype(out["ts"])


def test_regularize_timeseries_warns_when_freq_disagrees_with_inference():
    """Provided freq differs from inferred → warn but use the provided freq."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=5, freq="D"),
            "signups": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    with pytest.warns(UserWarning, match="does not match inferred"):
        regularize_timeseries(df, time_col="ts", freq="W")


def test_regularize_timeseries_empty_df_returns_empty():
    """Empty input → empty output (no errors, no warnings)."""
    df = pd.DataFrame({"ts": [], "signups": []})
    out = regularize_timeseries(df, time_col="ts", freq="D")
    assert len(out) == 0


def test_regularize_timeseries_value_cols_filter_limits_inf_replacement():
    """When ``value_cols`` is set, only those columns get inf-NaN replacement."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3, freq="D"),
            "signups": [1.0, np.inf, 3.0],
            "other": [10.0, np.inf, 30.0],
        }
    )
    out = regularize_timeseries(df, time_col="ts", freq="D", value_cols=["signups"])
    assert pd.isna(out["signups"].iloc[1])
    # 'other' was not in value_cols, so its inf survives.
    assert np.isinf(out["other"].iloc[1])


# ---------------------------------------------------------------------------
# Regularize transform
# ---------------------------------------------------------------------------


def test_regularize_transform_pads_gaps_at_ts_config_freq():
    """Transform reads freq from ts_config when its own freq is None."""
    df = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")],
            "signups": [1.0, 3.0],
        }
    )
    info = StubMetricInfo(metrics=[StubMetric(name="signups")])
    out = Regularize().apply(df, make_ts_config("D"), info)
    assert len(out) == 3
    assert pd.isna(out.loc[out["ts"] == pd.Timestamp("2024-01-02"), "signups"].iloc[0])


def test_regularize_transform_per_dim_independent_ranges():
    """Each segment is filled against its own time range."""
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")],
                    "country": ["US", "US"],
                    "signups": [1.0, 3.0],
                }
            ),
            pd.DataFrame(
                {
                    "ts": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")],
                    "country": ["GB", "GB"],
                    "signups": [10.0, 40.0],
                }
            ),
        ],
        ignore_index=True,
    )
    info = StubMetricInfo(dims=["country"], metrics=[StubMetric(name="signups")])
    out = Regularize().apply(df, make_ts_config("D"), info)

    us = out[out["country"] == "US"].sort_values("ts")
    gb = out[out["country"] == "GB"].sort_values("ts")
    assert len(us) == 3 and len(gb) == 3
    # The US segment's middle day is the padded one — value NaN, country still labeled.
    assert pd.isna(us.loc[us["ts"] == pd.Timestamp("2024-01-02"), "signups"].iloc[0])
    assert (us["country"] == "US").all()


def test_regularize_transform_explicit_freq_overrides_ts_config():
    """``Regularize(freq="W")`` takes precedence over ts_config.freq="D"."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3, freq="W"),
            "signups": [1.0, 2.0, 3.0],
        }
    )
    info = StubMetricInfo(metrics=[StubMetric(name="signups")])
    out = Regularize(freq="W").apply(df, make_ts_config("D"), info)
    assert len(out) == 3  # already weekly; no padding needed at W


def test_regularize_transform_empty_df_returns_empty():
    """Empty input is a no-op."""
    df = pd.DataFrame({"ts": [], "signups": []})
    info = StubMetricInfo(metrics=[StubMetric(name="signups")])
    out = Regularize().apply(df, make_ts_config("D"), info)
    assert len(out) == 0


def test_regularize_transform_str_name_is_empty_string():
    """Regularize doesn't change the metric semantics → empty suffix
    (so JobConfig auto-naming skips it)."""
    assert Regularize().str_name() == ""

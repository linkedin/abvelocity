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
"""Integration tests for the post-fetch transform chain in
:meth:`abvelocity.ts.get_data.ts_metrics_query.TSMetricsQuery.get_df`.

The SQL construction path is exercised exhaustively in
``test_ts_metrics_query.py``.  This file isolates the post-fetch
transform chaining behavior: each transform's ``apply`` runs in order,
gets the right ``ts_config`` / ``metric_info`` arguments, and an empty
chain leaves the data untouched.

Uses a stub cursor that returns a pre-built DataFrame — the goal is to
verify the framework's wiring, not to re-test SQL execution.
"""

from dataclasses import dataclass

import pandas as pd
import pytest

from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.ts.get_data.regularizer import Regularize
from abvelocity.ts.get_data.transforms import Coarsen, Diff
from abvelocity.ts.get_data.ts_transform import TSTransform
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig
from abvelocity.ts.get_data.ts_metrics_query import TSMetricsQuery


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class StubCursorResult:
    df: pd.DataFrame


class StubCursor:
    """Minimal cursor — returns the canned frame regardless of query."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_df(self, query: str) -> StubCursorResult:
        return StubCursorResult(df=self._df.copy())


def make_signups_metric() -> Metric:
    return Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        name="signups",
    )


def make_metric_info(dims=None) -> MetricInfo:
    return MetricInfo(
        metric_family=MetricFamily(
            name="test_family",
            u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
            metric_join_unit_col="user_id",
        ),
        metrics=[make_signups_metric()],
        dims=dims,
        start_date="2024-01-01",
        end_date="2024-01-14",
    )


# ---------------------------------------------------------------------------
# Regularize transform — gap fill + dtype coerce when wired into the chain
# ---------------------------------------------------------------------------


def test_get_df_with_regularize_pads_missing_dates_at_freq():
    """Cursor returns a 13-row daily series with one missing day; with a
    Regularize transform in the chain → output has 14 rows."""
    raw = pd.DataFrame(
        {
            "ts": list(pd.date_range("2024-01-01", periods=14, freq="D")),
            "signups": list(range(1, 15)),
            "sample_count": [100] * 14,
        }
    )
    raw = raw[raw["ts"] != pd.Timestamp("2024-01-08")]  # drop one mid-series day

    ts_cfg = TSMetricsConfig(
        time_col="event_ts",
        freq="D",
        post_fetch_transforms=(Regularize(),),
    )
    q = TSMetricsQuery(metric_info=make_metric_info(), ts_config=ts_cfg)
    out = q.get_df(StubCursor(raw))

    assert len(out) == 14
    assert pd.isna(out.loc[out["ts"] == pd.Timestamp("2024-01-08"), "signups"].iloc[0])


def test_get_df_with_regularize_pads_per_dim_independently():
    """Two segments with different gaps → each segment padded to its own
    full range when Regularize is in the chain."""
    rows_us = pd.DataFrame(
        {
            "ts": list(pd.date_range("2024-01-01", periods=7, freq="D")),
            "country": ["US"] * 7,
            "signups": [1.0] * 7,
            "sample_count": [10] * 7,
        }
    )
    rows_gb = pd.DataFrame(
        {
            "ts": list(pd.date_range("2024-01-01", periods=7, freq="D")),
            "country": ["GB"] * 7,
            "signups": [2.0] * 7,
            "sample_count": [20] * 7,
        }
    )
    raw = pd.concat([rows_us[rows_us["ts"] != "2024-01-03"], rows_gb[rows_gb["ts"] != "2024-01-05"]], ignore_index=True)

    ts_cfg = TSMetricsConfig(
        time_col="event_ts",
        freq="D",
        post_fetch_transforms=(Regularize(),),
    )
    q = TSMetricsQuery(metric_info=make_metric_info(dims=["country"]), ts_config=ts_cfg)
    out = q.get_df(StubCursor(raw))

    us = out[out["country"] == "US"]
    gb = out[out["country"] == "GB"]
    assert len(us) == 7 and len(gb) == 7
    assert us["country"].notna().all() and gb["country"].notna().all()
    assert pd.isna(us.loc[us["ts"] == "2024-01-03", "signups"].iloc[0])
    assert pd.isna(gb.loc[gb["ts"] == "2024-01-05", "signups"].iloc[0])


# ---------------------------------------------------------------------------
# Post-fetch transform chain ordering
# ---------------------------------------------------------------------------


def test_get_df_applies_post_fetch_transforms_in_order():
    """Transforms run sequentially: Coarsen first → output has weekly buckets,
    then Diff WoW reduces each weekly value to its difference vs prior week."""
    raw = pd.DataFrame(
        {
            "ts": list(pd.date_range("2024-01-01", periods=21, freq="D")),
            "signups": [1.0] * 7 + [2.0] * 7 + [3.0] * 7,
            "sample_count": [10] * 21,
        }
    )
    ts_cfg = TSMetricsConfig(
        time_col="event_ts",
        freq="D",
        post_fetch_transforms=(Coarsen(freq="W", agg="sum"), Diff(lag_period="W", n_lag_periods=1)),
    )
    q = TSMetricsQuery(metric_info=make_metric_info(), ts_config=ts_cfg)
    out = q.get_df(StubCursor(raw))

    # After Coarsen: 3 weekly rows with sums 7, 14, 21.
    # After Diff(WoW): row 0 = NaN (no prior week), rows 1, 2 = 7, 7.
    assert len(out) == 3
    assert pd.isna(out["signups"].iloc[0])
    assert out["signups"].iloc[1] == pytest.approx(7.0)
    assert out["signups"].iloc[2] == pytest.approx(7.0)


def test_get_df_no_transforms_returns_filled_only():
    """Empty post_fetch_transforms tuple → output equals the gap-filled frame."""
    raw = pd.DataFrame(
        {
            "ts": list(pd.date_range("2024-01-01", periods=5, freq="D")),
            "signups": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sample_count": [10] * 5,
        }
    )
    ts_cfg = TSMetricsConfig(time_col="event_ts", freq="D")
    q = TSMetricsQuery(metric_info=make_metric_info(), ts_config=ts_cfg)
    out = q.get_df(StubCursor(raw))

    assert list(out["signups"]) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_get_df_passes_ts_config_and_metric_info_to_transforms():
    """Each transform's ``apply`` receives the correct ts_config and metric_info."""

    captured: dict = {}

    class Spy(TSTransform):
        def apply(self, df, ts_config, metric_info):
            captured["ts_config"] = ts_config
            captured["metric_info"] = metric_info
            return df

    raw = pd.DataFrame(
        {
            "ts": list(pd.date_range("2024-01-01", periods=3, freq="D")),
            "signups": [1.0, 2.0, 3.0],
            "sample_count": [10] * 3,
        }
    )
    ts_cfg = TSMetricsConfig(
        time_col="event_ts",
        freq="D",
        post_fetch_transforms=(Spy(),),
    )
    info = make_metric_info()
    q = TSMetricsQuery(metric_info=info, ts_config=ts_cfg)
    q.get_df(StubCursor(raw))

    assert captured["ts_config"] is ts_cfg
    assert captured["metric_info"] is info

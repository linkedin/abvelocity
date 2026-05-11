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
"""End-to-end forecast tests via DuckDB — fetch through TSMetricsQuery,
transform through ``post_fetch_transforms``, fit + forecast via
:class:`SimpleForecastAlgo`.  Uses the lightweight ``simple`` algo so the
suite runs in seconds (vs. the minutes-per-fit cost of greykite); the
goal is to exercise the **wiring**, not the modeling quality.

Each test stresses one transform combination — daily, weekly (Coarsen),
WoW diff, DoW weight — to catch interface regressions like the
``Coarsen("W-SAT")`` start-vs-end-anchor bug or the anchored-alias path
through ``rows_per_period``.
"""

import numpy as np
import pandas as pd
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.metric import SUM, Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo

# Register simple algo at import time.
import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import (
    ACTUAL_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    METRIC_ID_COL,
    TIME_COL,
)
from abvelocity.ts.flow.flow import TSFlow
from abvelocity.ts.flow.flow_config import TSFlowConfig
from abvelocity.ts.get_data.regularizer import Regularize
from abvelocity.ts.get_data.transforms import Coarsen, Diff, WeightWithinPeriod
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig

# ---------------------------------------------------------------------------
# Synthetic data — daily signups for 6 months, 1 user (single-row-per-day)
# ---------------------------------------------------------------------------

START_DATE = "2024-01-01"
END_DATE = "2024-06-30"  # 182 days
FORECAST_HORIZON = 14


def make_signups_events_df(seed: int = 17) -> pd.DataFrame:
    """One row per (member, day) covering the full window.

    Daily signup volume has a mild weekly seasonality + weak upward
    trend; that's enough for ``SimpleForecastAlgo`` (period=7 mean) to
    produce non-trivial forecasts.
    """
    rng = np.random.default_rng(seed)
    n_days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1
    n_users = 30

    rows = []
    for day_idx in range(n_days):
        date = pd.Timestamp(START_DATE) + pd.Timedelta(days=day_idx)
        dow_mult = 1.0 + 0.20 * np.sin(2 * np.pi * day_idx / 7)
        trend_mult = 1.0 + 0.001 * day_idx
        for uid in range(n_users):
            mean = 1.5 * dow_mult * trend_mult
            signups = int(rng.poisson(mean))
            rows.append({"member_id": uid, "event_ts": date, "signup": signups})
    return pd.DataFrame(rows)


def setup_duckdb_with_events(events_df: pd.DataFrame) -> DuckDBCursor:  # noqa: ARG001
    """Load ``events_df`` into a fresh DuckDB cursor.

    The DuckDB cursor inspects the caller's frame for variables named
    after the SQL table — so ``events_df`` must be a local in the
    *caller's* scope when the query runs.  Each test holds it as a
    local before calling ``setup_duckdb_with_events`` + ``flow.run``.
    """
    cursor = DuckDBCursor()
    cursor._db_connection.execute("CREATE TABLE events AS SELECT * FROM events_df")
    return cursor


def make_signup_metric_info() -> MetricInfo:
    return MetricInfo(
        metric_family=MetricFamily(
            name="signup_events",
            u_metrics_query=UMetricsQuery(table_name="events", date_col="event_ts"),
            metric_join_unit_col="member_id",
        ),
        metrics=[
            Metric(
                numerator=UMetric(col="signup", agg=SUM, name="signup"),
                numerator_agg=SUM,
                name="signups",
            )
        ],
        start_date=START_DATE,
        end_date=END_DATE,
    )


def make_simple_forecast_config(value_col: str, freq: str) -> ForecastConfig:
    """SimpleForecastAlgo with period=7 (weekly seasonality at daily freq)
    or period=1 (no seasonality) — the test only cares that the algo
    produces a non-empty result_df."""
    return ForecastConfig(
        algo_name="simple",
        value_cols=(value_col,),
        freq=freq,
        forecast_horizon=FORECAST_HORIZON,
        coverage=0.80,
        algo_params={"period": 7 if freq == "D" else 1, "k": 3, "agg": "mean"},
        train_end_date=END_DATE,
        metric_id_template=value_col,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_e2e_daily_forecast_via_duckdb():
    """Bare daily forecast: fetch + Regularize + simple algo.  Asserts the
    pipeline produces both training rows (with actuals) and the requested
    forecast horizon."""
    events_df = make_signups_events_df()  # noqa: F841 — read by DuckDB via frame inspection
    cursor = setup_duckdb_with_events(events_df)

    metric_info = make_signup_metric_info()
    flow_cfg = TSFlowConfig(
        ts_metrics_config=TSMetricsConfig(
            time_col="event_ts",
            freq="D",
            dialect="duckdb",
            post_fetch_transforms=(Regularize(),),
        ),
        ts_model_config=make_simple_forecast_config("signups", freq="D"),
        mode="forecast",
    )
    flow = TSFlow(flow_config=flow_cfg, io_param=IOParam(cursor=cursor))

    result = flow.run(metric_info)
    assert result.result_df is not None
    rdf = result.result_df

    # Forecast rows = horizon, beyond train_end.
    # Last FORECAST_HORIZON timestamps = the forecast horizon.  Filtering
    # by ``stage == "forecast"`` over-matches because the runner stamps
    # ``stage`` from ``actual.isna()`` — and training rows can have NaN
    # actuals when WeightWithinPeriod / Diff sit in the chain.
    future_rows = rdf.sort_values(TIME_COL).tail(FORECAST_HORIZON)
    assert len(future_rows) == FORECAST_HORIZON
    # All future values populated for plain daily (no NaN-injecting transforms).
    assert future_rows[ACTUAL_COL].isna().all()
    assert future_rows[FORECAST_COL].notna().all()
    assert future_rows[FORECAST_LOWER_COL].notna().all()
    assert future_rows[FORECAST_UPPER_COL].notna().all()
    # PI is well-ordered.
    assert (future_rows[FORECAST_LOWER_COL] <= future_rows[FORECAST_COL]).all()
    assert (future_rows[FORECAST_COL] <= future_rows[FORECAST_UPPER_COL]).all()


def test_e2e_weekly_forecast_via_duckdb_with_W_SAT_anchor():
    """Coarsen("W-SAT") into a weekly forecast.  Regression for the bug
    where Sun-anchored output collided with the W-SAT model freq config
    and made greykite drop every row.  ``simple`` algo is freq-agnostic
    but we still set ``freq="W-SAT"`` to lock the contract end-to-end."""
    events_df = make_signups_events_df()  # noqa: F841
    cursor = setup_duckdb_with_events(events_df)

    metric_info = make_signup_metric_info()
    flow_cfg = TSFlowConfig(
        ts_metrics_config=TSMetricsConfig(
            time_col="event_ts",
            freq="D",
            dialect="duckdb",
            post_fetch_transforms=(Regularize(), Coarsen(freq="W-SAT", agg="sum")),
        ),
        ts_model_config=make_simple_forecast_config("signups", freq="W-SAT"),
        mode="forecast",
    )
    flow = TSFlow(flow_config=flow_cfg, io_param=IOParam(cursor=cursor))

    result = flow.run(metric_info)
    rdf = result.result_df
    assert rdf is not None and not rdf.empty

    # All training rows must be Saturday-anchored (the W-SAT period end).
    training = rdf[rdf[ACTUAL_COL].notna()]
    assert (pd.to_datetime(training[TIME_COL]).dt.weekday == 5).all()
    # Forecast rows = horizon, beyond train_end.
    # Last FORECAST_HORIZON timestamps = the forecast horizon.  Filtering
    # by ``stage == "forecast"`` over-matches because the runner stamps
    # ``stage`` from ``actual.isna()`` — and training rows can have NaN
    # actuals when WeightWithinPeriod / Diff sit in the chain.
    future_rows = rdf.sort_values(TIME_COL).tail(FORECAST_HORIZON)
    assert len(future_rows) == FORECAST_HORIZON
    # And forecast timestamps continue the Saturday cadence.
    assert (pd.to_datetime(future_rows[TIME_COL]).dt.weekday == 5).all()


def test_e2e_wow_diff_forecast_via_duckdb():
    """WoW diff transform: each row replaced with (today − same-day-last-week).
    Asserts the pipeline survives a value column dominated by NaNs (first
    7 rows of every series have no week-prior baseline)."""
    events_df = make_signups_events_df()  # noqa: F841
    cursor = setup_duckdb_with_events(events_df)

    metric_info = make_signup_metric_info()
    flow_cfg = TSFlowConfig(
        ts_metrics_config=TSMetricsConfig(
            time_col="event_ts",
            freq="D",
            dialect="duckdb",
            post_fetch_transforms=(Regularize(), Diff(lag_period="W", n_lag_periods=1)),
        ),
        ts_model_config=make_simple_forecast_config("signups", freq="D"),
        mode="forecast",
    )
    flow = TSFlow(flow_config=flow_cfg, io_param=IOParam(cursor=cursor))

    result = flow.run(metric_info)
    rdf = result.result_df
    assert rdf is not None and not rdf.empty

    # First 7 rows of training have no diff baseline → NaN actuals.
    training = rdf[rdf[METRIC_ID_COL] == "signups"].sort_values(TIME_COL).reset_index(drop=True)
    assert training[ACTUAL_COL].iloc[:7].isna().all()
    # Later rows have valid diffs.
    assert training[ACTUAL_COL].iloc[7:200].notna().any()
    # Forecast horizon emitted with non-null forecast values.
    # Last FORECAST_HORIZON timestamps = the forecast horizon.  Filtering
    # by ``stage == "forecast"`` over-matches because the runner stamps
    # ``stage`` from ``actual.isna()`` — and training rows can have NaN
    # actuals when WeightWithinPeriod / Diff sit in the chain.
    future_rows = rdf.sort_values(TIME_COL).tail(FORECAST_HORIZON)
    assert len(future_rows) == FORECAST_HORIZON
    assert future_rows[FORECAST_COL].notna().all()


def test_e2e_dow_weight_forecast_via_duckdb_with_W_SAT_anchor():
    """WeightWithinPeriod("W-SAT") on daily data → weights summing to 1
    per Sun-Sat week.  Regression for the ``rows_per_period('D', 'W-SAT')``
    KeyError plus the broader anchored-alias support contract."""
    events_df = make_signups_events_df()  # noqa: F841
    cursor = setup_duckdb_with_events(events_df)

    metric_info = make_signup_metric_info()
    flow_cfg = TSFlowConfig(
        ts_metrics_config=TSMetricsConfig(
            time_col="event_ts",
            freq="D",
            dialect="duckdb",
            post_fetch_transforms=(Regularize(), WeightWithinPeriod(period="W-SAT")),
        ),
        ts_model_config=make_simple_forecast_config("signups", freq="D"),
        mode="forecast",
    )
    flow = TSFlow(flow_config=flow_cfg, io_param=IOParam(cursor=cursor))

    result = flow.run(metric_info)
    rdf = result.result_df
    assert rdf is not None and not rdf.empty

    # Every complete Sun–Sat week's weights sum to 1 (within float tol).
    training = rdf[rdf[ACTUAL_COL].notna()].copy()
    training[TIME_COL] = pd.to_datetime(training[TIME_COL])
    # Floor each timestamp to its enclosing Sun-Sat week start (Sunday) for grouping.
    training["week_anchor"] = training[TIME_COL].dt.to_period("W-SAT").dt.start_time
    weekly_sums = training.groupby("week_anchor")[ACTUAL_COL].sum()
    # Some weeks may have <7 rows from gap-fill at the boundary — only
    # complete-week buckets land non-NaN; here we assert at least some
    # weeks summed to 1.
    assert ((weekly_sums - 1.0).abs() < 1e-6).any(), f"No Sun–Sat week summed to 1.0 — weekly_sums:\n{weekly_sums}"
    # Forecast horizon emitted: 14 rows beyond train_end.  Filter by
    # timestamp (NOT ``actual.isna()``) — WeightWithinPeriod NaNs the
    # partial-week training tail too, so ``actual.isna()`` over-counts.
    # Some forecast values may also be NaN: when the partial trailing
    # week NaN'd the last few training rows, ``simple`` algo's 7-day
    # lookback can't fill the very end of the horizon.  Assert structure
    # (14 future rows) plus that the *majority* are populated.
    # Last FORECAST_HORIZON timestamps = the forecast horizon.  Filtering
    # by ``stage == "forecast"`` over-matches because the runner stamps
    # ``stage`` from ``actual.isna()`` — and training rows can have NaN
    # actuals when WeightWithinPeriod / Diff sit in the chain.
    future_rows = rdf.sort_values(TIME_COL).tail(FORECAST_HORIZON)
    assert len(future_rows) == FORECAST_HORIZON
    assert future_rows[FORECAST_COL].notna().sum() >= FORECAST_HORIZON - 2

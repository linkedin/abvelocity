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
"""TSFlow visual examples — full pipeline: DuckDB fetch → SimpleForecastAlgo → plot.

Each test runs the complete TSFlow pipeline (data fetch from DuckDB,
SimpleForecastAlgo, optional eval) on synthetic event data and saves an HTML
plot to ``docs/static/test-results/timeseries/``.

Scenarios
---------
1. **Single metric via TSFlow** — daily signups fetched from DuckDB,
   14-day seasonal mean forecast.  Shows the fetch → algo → result chain.
2. **Two metrics via TSFlow** — impressions + signups fetched from DuckDB,
   subplots side-by-side.  Shows multi-metric handling.
"""

import os
from pathlib import Path

import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401
import numpy as np
import pandas as pd
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.metric import SUM, Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import ACTUAL_COL, METRIC_ID_COL, TIME_COL
from abvelocity.ts.flow.flow import TSFlow
from abvelocity.ts.flow.flow_config import TSFlowConfig
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig
from abvelocity.ts.viz import plot_forecast

# Output directory — parents: flow/ timeseries/ abvelocity/ blah/ test/ pkg/ repo/
WRITE_PATH = Path(__file__).parents[4] / "docs" / "static" / "test-results" / "timeseries"

START_DATE = "2024-01-01"
END_DATE = "2024-03-31"


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────


def make_events_df(seed: int = 42) -> pd.DataFrame:
    """Synthetic ad-events table: one row per (member, day).

    Columns: ``member_id``, ``event_ts``, ``impression``, ``signup``.
    Impressions follow Poisson with mild weekly seasonality; signups are
    Binomial(impressions, signup_rate).
    """
    rng = np.random.default_rng(seed)
    n_days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1
    n_users = 40
    rows = []
    for d in range(n_days):
        date = pd.Timestamp(START_DATE) + pd.Timedelta(days=d)
        day_mult = 1.0 + 0.2 * np.sin(2 * np.pi * d / 7)
        for uid in range(n_users):
            impressions = int(rng.poisson(12 * day_mult))
            signups = int(rng.binomial(impressions, 0.08)) if impressions > 0 else 0
            rows.append(
                {
                    "member_id": uid,
                    "event_ts": date,
                    "impression": impressions,
                    "signup": signups,
                }
            )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_family() -> MetricFamily:
    return MetricFamily(
        name="ad_events",
        u_metrics_query=UMetricsQuery(table_name="events", date_col="event_ts"),
        metric_join_unit_col="member_id",
    )


def make_cursor(events_df: pd.DataFrame) -> DuckDBCursor:
    """Create an in-memory DuckDB cursor with events registered as a table."""
    cursor = DuckDBCursor()
    cursor._db_connection.execute("CREATE TABLE events AS SELECT * FROM events_df")
    return cursor


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — single metric (signups), 14-day horizon
# ─────────────────────────────────────────────────────────────────────────────


def test_flow_single_metric_duckdb():
    """TSFlow: DuckDB fetch → SimpleForecastAlgo → 14-day signup forecast."""
    events_df = make_events_df()  # noqa: F841 — DuckDB replacement scan
    cursor = make_cursor(events_df)

    u_signup = UMetric(col="signup", agg=SUM, name="signup")
    signup_metric = Metric(numerator=u_signup, numerator_agg=SUM, name="total_signups")
    metric_info = MetricInfo(
        metric_family=make_family(),
        metrics=[signup_metric],
        start_date=START_DATE,
        end_date=END_DATE,
    )

    fc = ForecastConfig(
        algo_name="simple",
        forecast_horizon=14,
        value_cols=("total_signups",),
        freq="D",
        algo_params={"period": 7, "k": 4},
    )
    flow_cfg = TSFlowConfig(
        ts_metrics_config=TSMetricsConfig(time_col="event_ts", freq="D", dialect="duckdb"),
        ts_model_config=fc,
        mode="forecast",
    )

    flow = TSFlow(flow_config=flow_cfg, io_param=IOParam(cursor=cursor))
    result = flow.run(metric_info)

    assert result.result_df is not None
    n_train = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1
    assert len(result.result_df) == n_train + 14
    future = result.result_df[result.result_df[ACTUAL_COL].isna()]
    assert len(future) == 14

    train_end = pd.Timestamp(END_DATE)
    fig = plot_forecast(
        result_df=result.result_df,
        time_col=TIME_COL,
        train_end_date=train_end,
        title="Daily Signups — 14-day Forecast via TSFlow + DuckDB (period=7, k=4)",
    )
    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "flow_forecast_single_metric_duckdb.html"
    fig.write_html(str(path))
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — two metrics (impressions + signups), 14-day horizon
# ─────────────────────────────────────────────────────────────────────────────


def test_flow_two_metrics_duckdb():
    """TSFlow: DuckDB fetch → SimpleForecastAlgo → impressions + signups forecast."""
    events_df = make_events_df()  # noqa: F841 — DuckDB replacement scan
    cursor = make_cursor(events_df)

    u_impression = UMetric(col="impression", agg=SUM, name="impression")
    u_signup = UMetric(col="signup", agg=SUM, name="signup")
    impressions_metric = Metric(numerator=u_impression, numerator_agg=SUM, name="impressions")
    signups_metric = Metric(numerator=u_signup, numerator_agg=SUM, name="signups")

    metric_info = MetricInfo(
        metric_family=make_family(),
        metrics=[impressions_metric, signups_metric],
        start_date=START_DATE,
        end_date=END_DATE,
    )

    fc = ForecastConfig(
        algo_name="simple",
        forecast_horizon=14,
        value_cols=("impressions", "signups"),
        freq="D",
        algo_params={"period": 7, "k": 4},
    )
    flow_cfg = TSFlowConfig(
        ts_metrics_config=TSMetricsConfig(time_col="event_ts", freq="D", dialect="duckdb"),
        ts_model_config=fc,
        mode="forecast",
    )

    flow = TSFlow(flow_config=flow_cfg, io_param=IOParam(cursor=cursor))
    result = flow.run(metric_info)

    assert result.result_df is not None
    assert set(result.result_df[METRIC_ID_COL].unique()) == {"impressions", "signups"}

    train_end = pd.Timestamp(END_DATE)
    fig = plot_forecast(
        result_df=result.result_df,
        time_col=TIME_COL,
        train_end_date=train_end,
        title="Impressions & Signups — 14-day Forecast via TSFlow + DuckDB",
        subplots=True,
    )
    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "flow_forecast_two_metrics_duckdb.html"
    fig.write_html(str(path))
    assert path.exists()

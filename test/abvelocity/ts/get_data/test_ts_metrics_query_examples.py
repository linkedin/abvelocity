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
"""TSMetricsQuery integration examples — two scenarios on synthetic event data.

Each test generates a synthetic ad-events table (impressions + clicks per
member per day), executes ``TSMetricsQuery`` against DuckDB, asserts the
output shape, and saves an HTML plot to
``docs/static/test-results/timeseries/``.

Scenarios
---------
1. **No dims** — aggregate impressions, clicks, CTR over 90 days.
2. **With dims** — same metrics broken out by ``country`` × ``device``.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.metric import SUM, Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig
from abvelocity.ts.get_data.ts_metrics_query import TSMetricsQuery
from plotly.subplots import make_subplots

# Output directory — parents: [get_data/] [timeseries/] [abvelocity/] [blah/] [test/] [pkg] [repo root]
WRITE_PATH = Path(__file__).parents[4] / "docs" / "static" / "test-results" / "timeseries"

START_DATE = "2024-01-01"
END_DATE = "2024-03-31"

# Metric column names produced by TSMetricsQuery
IMPRESSIONS_COL = "impressions"
CLICKS_COL = "clicks"
CTR_COL = "ctr"


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────


def make_events_df(n_days: int = 91, n_users: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic ad-events table.

    One row per (member, day). Impressions are Poisson-distributed.
    Clicks are Binomial(impressions, ctr) where CTR varies by country and
    device to produce distinct per-slice time series.

    Args:
        n_days: Number of calendar days.
        n_users: Number of simulated members.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
        ``member_id``, ``event_ts``, ``impression``, ``click``,
        ``country``, ``device``.
    """
    rng = np.random.default_rng(seed)
    countries = ["US", "UK", "CA"]
    devices = ["mobile", "desktop"]
    base_ctrs = {"US": 0.05, "UK": 0.04, "CA": 0.06}
    device_mults = {"mobile": 1.2, "desktop": 0.9}

    start = pd.Timestamp(START_DATE)
    rows = []
    for day in range(n_days):
        date = start + pd.Timedelta(days=day)
        # Add a mild weekly seasonality to impression volume.
        day_of_week_mult = 1.0 + 0.15 * np.sin(2 * np.pi * day / 7)
        for uid in range(n_users):
            country = countries[uid % len(countries)]
            device = devices[uid % len(devices)]
            ctr = base_ctrs[country] * device_mults[device]
            mean_impressions = 10 * day_of_week_mult
            impressions = int(rng.poisson(mean_impressions))
            clicks = int(rng.binomial(impressions, ctr)) if impressions > 0 else 0
            rows.append(
                {
                    "member_id": uid,
                    "event_ts": date,
                    "impression": impressions,
                    "click": clicks,
                    "country": country,
                    "device": device,
                }
            )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Shared metric / family definitions
# ─────────────────────────────────────────────────────────────────────────────


def make_metric_defs():
    """Return (u_impression, u_click, impressions_metric, clicks_metric, ctr_metric)."""
    u_impression = UMetric(col="impression", agg=SUM, name="impression")
    u_click = UMetric(col="click", agg=SUM, name="click")

    impressions_metric = Metric(numerator=u_impression, numerator_agg=SUM, name=IMPRESSIONS_COL)
    clicks_metric = Metric(numerator=u_click, numerator_agg=SUM, name=CLICKS_COL)
    ctr_metric = Metric(
        numerator=u_click,
        denominator=u_impression,
        numerator_agg=SUM,
        denominator_agg=SUM,
        name=CTR_COL,
    )
    return impressions_metric, clicks_metric, ctr_metric


def make_family() -> MetricFamily:
    return MetricFamily(
        name="ad_events",
        u_metrics_query=UMetricsQuery(table_name="events", date_col="event_ts"),
        metric_join_unit_col="member_id",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────


def plot_ts_metrics_no_dims(df: pd.DataFrame, title: str) -> go.Figure:
    """Three-row subplot: one panel per metric (impressions, clicks, ctr)."""
    metrics = [IMPRESSIONS_COL, CLICKS_COL, CTR_COL]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=metrics,
        vertical_spacing=0.08,
    )
    for row, metric in enumerate(metrics, start=1):
        fig.add_trace(
            go.Scatter(x=df["ts"], y=df[metric], mode="lines", name=metric, showlegend=False),
            row=row,
            col=1,
        )
    fig.update_layout(title_text=title, title_x=0.5, hovermode="x unified", height=700)
    return fig


def plot_ts_metrics_with_dims(df: pd.DataFrame, title: str) -> go.Figure:
    """Three-row subplot: one panel per metric, one line per (country, device) slice."""
    metrics = [IMPRESSIONS_COL, CLICKS_COL, CTR_COL]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=metrics,
        vertical_spacing=0.08,
    )
    slices = df[["country", "device"]].drop_duplicates().sort_values(["country", "device"])
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    for row, metric in enumerate(metrics, start=1):
        for i, (_, row_slice) in enumerate(slices.iterrows()):
            country, device = row_slice["country"], row_slice["device"]
            mask = (df["country"] == country) & (df["device"] == device)
            slice_df = df[mask].sort_values("ts")
            label = f"{country}/{device}"
            fig.add_trace(
                go.Scatter(
                    x=slice_df["ts"],
                    y=slice_df[metric],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=(row == 1),
                    line={"color": colors[i % len(colors)]},
                ),
                row=row,
                col=1,
            )
    fig.update_layout(title_text=title, title_x=0.5, hovermode="x unified", height=800)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — no dims
# ─────────────────────────────────────────────────────────────────────────────


def test_ts_metrics_query_no_dims():
    """Aggregate impressions, clicks, CTR over 90 days — no dimensional split."""
    events_df = make_events_df()  # noqa: F841 — used by DuckDB via frame inspection
    impressions_metric, clicks_metric, ctr_metric = make_metric_defs()

    metric_info = MetricInfo(
        metric_family=make_family(),
        metrics=[impressions_metric, clicks_metric, ctr_metric],
        start_date=START_DATE,
        end_date=END_DATE,
    )
    ts_config = TSMetricsConfig(time_col="event_ts", freq="D", dialect="duckdb")

    cursor = DuckDBCursor()
    cursor._db_connection.execute("CREATE TABLE events AS SELECT * FROM events_df")

    q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_config)
    df = q.get_df(cursor)

    # Basic shape checks
    assert df is not None
    assert len(df) == 91, f"Expected 91 daily rows (Q1 2024 inclusive), got {len(df)}"
    assert "ts" in df.columns
    for col in [IMPRESSIONS_COL, CLICKS_COL, CTR_COL]:
        assert col in df.columns, f"Missing column: {col}"

    # Sanity: CTR should be between 0 and 1
    assert df[CTR_COL].between(0, 1).all(), "CTR values out of [0, 1] range"

    # Sanity: clicks ≤ impressions every day
    assert (df[CLICKS_COL] <= df[IMPRESSIONS_COL]).all(), "clicks > impressions on some day"

    # Date range
    # DuckDB returns datetime.date objects; str() gives "YYYY-MM-DD" directly.
    assert str(df["ts"].min()).startswith(START_DATE)
    assert str(df["ts"].max()).startswith(END_DATE)

    print(f"\n*** No-dims result shape: {df.shape}")
    print(df.head())

    fig = plot_ts_metrics_no_dims(
        df=df,
        title="Ad Metrics (impressions / clicks / CTR) — No dims",
    )
    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "ts_metrics_no_dims.html"
    fig.write_html(str(path))
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — with dims: country × device
# ─────────────────────────────────────────────────────────────────────────────


def test_ts_metrics_query_with_dims():
    """Impressions, clicks, CTR broken out by country (US/UK/CA) × device (mobile/desktop)."""
    events_df = make_events_df()  # noqa: F841 — used by DuckDB via frame inspection
    impressions_metric, clicks_metric, ctr_metric = make_metric_defs()

    metric_info = MetricInfo(
        metric_family=make_family(),
        metrics=[impressions_metric, clicks_metric, ctr_metric],
        dims=["country", "device"],
        start_date=START_DATE,
        end_date=END_DATE,
    )
    ts_config = TSMetricsConfig(time_col="event_ts", freq="D", dialect="duckdb")

    cursor = DuckDBCursor()
    cursor._db_connection.execute("CREATE TABLE events AS SELECT * FROM events_df")

    q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_config)
    df = q.get_df(cursor)

    n_countries = 3
    n_devices = 2
    n_slices = n_countries * n_devices  # 6

    assert df is not None
    assert len(df) == 91 * n_slices, f"Expected {91 * n_slices} rows, got {len(df)}"
    assert set(df["country"].unique()) == {"US", "UK", "CA"}
    assert set(df["device"].unique()) == {"mobile", "desktop"}

    for col in [IMPRESSIONS_COL, CLICKS_COL, CTR_COL]:
        assert col in df.columns, f"Missing column: {col}"

    assert df[CTR_COL].between(0, 1).all(), "CTR values out of [0, 1] range"
    assert (df[CLICKS_COL] <= df[IMPRESSIONS_COL]).all(), "clicks > impressions in some slice"

    # CTR should differ meaningfully across slices (our generator makes them distinct)
    slice_ctrs = df.groupby(["country", "device"])[CTR_COL].mean()
    assert slice_ctrs.max() > slice_ctrs.min() * 1.1, "Slice CTRs unexpectedly uniform"

    print(f"\n*** With-dims result shape: {df.shape}")
    print(df[["ts", "country", "device", IMPRESSIONS_COL, CLICKS_COL, CTR_COL]].head(12))

    fig = plot_ts_metrics_with_dims(
        df=df,
        title="Ad Metrics by Country × Device (impressions / clicks / CTR)",
    )
    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "ts_metrics_with_dims.html"
    fig.write_html(str(path))
    assert path.exists()

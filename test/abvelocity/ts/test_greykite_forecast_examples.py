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
"""Greykite forecast examples — three scenarios on synthetic data.

Each test generates a forecast with :class:`TSRunner` + ``greykite`` and
saves an HTML plot to ``docs/static/test-results/timeseries/``.  All tests
are skipped when ``blah.greykite`` is not installed.

Scenarios
---------
1. **Single metric** — daily sessions, 30-day horizon.
2. **Multivariate** — pageviews + signups, 14-day horizon.
3. **Anomaly masking** — effect of masking a traffic spike on the forecast.
"""

import os
from pathlib import Path

# Trigger self-registration of GreykiteForecastAlgo into ALGO_REGISTRY.
import abvelocity.ts.algo.greykite_forecast_algo  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.algo.greykite_forecast_algo import GREYKITE_AVAILABLE
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import METRIC_ID_COL, TIME_COL
from abvelocity.ts.runner import TSRunner
from abvelocity.ts.testing_utils import make_daily_series
from abvelocity.ts.viz import add_multi_vrects, plot_forecast

# Output directory — relative to repo root.
# parents: [timeseries/] [abvelocity/] [blah/] [test/] [abvelocity pkg] [repo root]
WRITE_PATH = Path(__file__).parents[3] / "docs" / "static" / "test-results" / "timeseries"

# Shared algo_params: linear fit, no CV (these are demo tests, not accuracy benchmarks).
ALGO_PARAMS = {
    "model_template": "SILVERKITE",
    "model_components": {
        "custom": {"fit_algorithm_dict": {"fit_algorithm": "ridge"}},
        "uncertainty": {"uncertainty_dict": "auto"},
    },
    "evaluation_period": {"cv_max_splits": 0},
    "computation": {"verbose": 0},
}

SKIP = pytest.mark.skipif(not GREYKITE_AVAILABLE, reason="blah.greykite not installed")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — single metric, 30-day horizon
# ─────────────────────────────────────────────────────────────────────────────


@SKIP
def test_scenario_single_metric():
    """Daily sessions — 30-day forecast saved to forecast_single_metric.html."""
    df = make_daily_series(n_days=365, col="sessions", base=1000, trend=0.6, seed=1)
    train_end = pd.Timestamp("2023-11-30")

    config = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("sessions",),
        freq="D",
        train_end_date=str(train_end.date()),
        coverage=0.95,
        forecast_horizon=30,
        algo_name="greykite",
        algo_params=ALGO_PARAMS,
    )
    result = TSRunner(config).run(df=df)

    assert result.result_df is not None
    fc = result.result_df
    assert set(fc[METRIC_ID_COL].unique()) == {"sessions"}
    # result_df contains training history + forecast; check forecast period rows
    fc_future = fc[fc[TIME_COL] > train_end]
    assert len(fc_future) == 30

    fig = plot_forecast(
        result_df=fc,
        time_col=TIME_COL,
        train_end_date=train_end,
        title="Daily Sessions — 30-day forecast (SILVERKITE)",
    )
    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "forecast_single_metric.html"
    fig.write_html(str(path))
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — multivariate: pageviews + signups, 14-day horizon
# ─────────────────────────────────────────────────────────────────────────────


@SKIP
def test_scenario_multivariate():
    """Pageviews + signups — 14-day forecast saved to forecast_multivariate.html."""
    rng = np.random.default_rng(7)
    df = make_daily_series(n_days=300, col="pageviews", base=8000, trend=3.0, seed=7)
    df["signups"] = (df["pageviews"] * 0.018 + rng.normal(0, 8, 300)).clip(lower=0)

    train_end = pd.Timestamp("2023-09-30")
    config = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("pageviews", "signups"),
        freq="D",
        train_end_date=str(train_end.date()),
        coverage=0.95,
        forecast_horizon=14,
        algo_name="greykite",
        algo_params=ALGO_PARAMS,
    )
    result = TSRunner(config).run(df=df)

    assert result.result_df is not None
    fc = result.result_df
    assert set(fc[METRIC_ID_COL].unique()) == {"pageviews", "signups"}
    # result_df contains training history + forecast; check forecast period rows
    fc_future = fc[fc[TIME_COL] > train_end]
    assert len(fc_future) == 14 * 2  # 14 forecast days × 2 metrics

    fig = plot_forecast(
        result_df=fc,
        time_col=TIME_COL,
        train_end_date=train_end,
        title="Daily Pageviews & Signups — 14-day forecast (SILVERKITE)",
        subplots=True,
    )
    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "forecast_multivariate.html"
    fig.write_html(str(path))
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — anomaly masking: with vs without masking a traffic spike
# ─────────────────────────────────────────────────────────────────────────────


@SKIP
def test_scenario_anomaly_masking():
    """Anomaly masking comparison saved to forecast_anomaly_masking.html."""
    rng = np.random.default_rng(99)
    n = 270
    t = np.arange(n)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    base = 800 + 0.4 * t + 55 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 18, n)

    spike_start, spike_end = 90, 111
    spiked = base.copy()
    spiked[spike_start:spike_end] += 600

    df = pd.DataFrame({TIME_COL: dates, "requests": spiked.clip(min=0)})
    anomaly_df = pd.DataFrame(
        {
            "start_ts": [dates[spike_start]],
            "end_ts": [dates[spike_end - 1]],
        }
    )

    train_end = pd.Timestamp("2023-08-31")
    config = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("requests",),
        freq="D",
        train_end_date=str(train_end.date()),
        coverage=0.95,
        forecast_horizon=21,
        algo_name="greykite",
        algo_params=ALGO_PARAMS,
    )
    runner = TSRunner(config)
    result_raw = runner.run(df=df)
    result_masked = runner.run(df=df, anomaly_df=anomaly_df)

    assert result_raw.result_df is not None
    assert result_masked.result_df is not None

    from plotly.subplots import make_subplots

    titles = ["Without anomaly masking", "With anomaly masking (spike masked to NaN)"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.10)

    for i, result in enumerate([result_raw, result_masked], start=1):
        fc = result.result_df[result.result_df[METRIC_ID_COL] == "requests"].reset_index(drop=True)
        sub_fig = plot_forecast(
            result_df=fc,
            time_col=TIME_COL,
            train_end_date=train_end,
            title=None,
            subplots=False,
            showlegend=(i == 1),
        )
        for trace in sub_fig.data:
            fig.add_trace(trace, row=i, col=1)

        # Spike highlight
        add_multi_vrects(
            fig=fig,
            periods_df=anomaly_df,
            start_time_col="start_ts",
            end_time_col="end_ts",
            opacity=0.08,
            grouping_color_dict={"metric_id": "rgba(214, 39, 40, 1.0)"},
        )

    fig.update_layout(
        title_text="Effect of Anomaly Masking on Forecast (SILVERKITE)",
        title_x=0.5,
        hovermode="x unified",
        height=650,
    )

    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "forecast_anomaly_masking.html"
    fig.write_html(str(path))
    assert path.exists()

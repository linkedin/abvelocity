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
"""Simple-forecast visual examples — three scenarios on synthetic data.

Each test generates a forecast with :class:`SimpleForecastAlgo` and saves an
HTML plot to ``docs/static/test-results/timeseries/``. No optional
dependencies — these always run.

Scenarios
---------
1. **Weekly-seasonal** — single metric, 28-day horizon, ``period=7, k=8``.
   Good for eyeballing whether the algo picks up the day-of-week pattern.
2. **Two-metric** — pageviews + signups with weekly signal, 14-day horizon.
3. **Backfill spaghetti** — multiple rolling cutoffs overlaid on actuals,
   one forecast line per cutoff so you can see how the algo tracks the series.
"""

import os
from pathlib import Path

# Trigger self-registration of SimpleForecastAlgo into ALGO_REGISTRY.
import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.backfill.runner import BackfillRunner
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import ACTUAL_COL, CUTOFF_COL, FORECAST_COL, FORECAST_LOWER_COL, FORECAST_UPPER_COL, METRIC_ID_COL, TIME_COL
from abvelocity.ts.runner import TSRunner
from abvelocity.ts.testing_utils import make_daily_series
from abvelocity.ts.viz import plot_forecast

# Output directory — parents: timeseries/ abvelocity/ blah/ test/ pkg/ repo/
WRITE_PATH = Path(__file__).parents[3] / "docs" / "static" / "test-results" / "timeseries"

# Tab-10 palette with opacity, one entry per cutoff (8 should be plenty).
CUTOFF_COLORS = [
    "rgba(31,119,180,0.85)",
    "rgba(255,127,14,0.85)",
    "rgba(44,160,44,0.85)",
    "rgba(214,39,40,0.85)",
    "rgba(148,103,189,0.85)",
    "rgba(140,86,75,0.85)",
    "rgba(227,119,194,0.85)",
    "rgba(127,127,127,0.85)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — single metric, weekly-seasonal, 28-day horizon
# ─────────────────────────────────────────────────────────────────────────────


def test_scenario_simple_weekly_seasonal():
    """Period=7 seasonal mean on 1 year of daily data — saved to simple_forecast_weekly_seasonal.html."""
    full_df = make_daily_series(n_days=365, col="sessions", base=1000, trend=0.8, weekly_amp=120, noise=25, seed=1)
    train_end = pd.Timestamp("2023-11-01")
    train_df = full_df[full_df[TIME_COL] <= train_end].copy()

    config = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("sessions",),
        freq="D",
        train_end_date=str(train_end.date()),
        coverage=0.95,
        forecast_horizon=28,
        algo_name="simple",
        algo_params={"period": 7, "k": 8},
    )
    result = TSRunner(config).run(df=train_df)

    assert result.result_df is not None
    future_rows = result.result_df[result.result_df[ACTUAL_COL].isna()]
    assert len(future_rows) == 28

    fig = plot_forecast(
        result_df=result.result_df,
        time_col=TIME_COL,
        train_end_date=train_end,
        title="Sessions — 28-day Seasonal Mean Forecast (period=7, k=8)",
    )
    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "simple_forecast_weekly_seasonal.html"
    fig.write_html(str(path))
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — two metrics, 14-day horizon
# ─────────────────────────────────────────────────────────────────────────────


def test_scenario_simple_multivariate():
    """Pageviews + correlated signups — saved to simple_forecast_multivariate.html."""
    rng = np.random.default_rng(7)
    full_df = make_daily_series(n_days=300, col="pageviews", base=8000, trend=3.0, weekly_amp=600, noise=80, seed=7)
    full_df["signups"] = (full_df["pageviews"] * 0.018 + rng.normal(0, 12, 300)).clip(lower=0)

    train_end = pd.Timestamp("2023-09-15")
    train_df = full_df[full_df[TIME_COL] <= train_end].copy()

    config = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("pageviews", "signups"),
        freq="D",
        train_end_date=str(train_end.date()),
        coverage=0.95,
        forecast_horizon=14,
        algo_name="simple",
        algo_params={"period": 7, "k": 4},
    )
    result = TSRunner(config).run(df=train_df)

    assert result.result_df is not None
    assert set(result.result_df[METRIC_ID_COL].unique()) == {"pageviews", "signups"}

    fig = plot_forecast(
        result_df=result.result_df,
        time_col=TIME_COL,
        train_end_date=train_end,
        title="Pageviews & Signups — 14-day Seasonal Mean Forecast (period=7, k=4)",
        subplots=True,
    )
    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "simple_forecast_multivariate.html"
    fig.write_html(str(path))
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — backfill spaghetti: rolling cutoffs overlaid on actuals
# ─────────────────────────────────────────────────────────────────────────────


def test_scenario_simple_backfill_spaghetti():
    """BackfillRunner with 6 cutoffs (step=45d) — saved to simple_forecast_backfill_spaghetti.html.

    Each forecast window is drawn as a separate coloured line against the full
    actual series in grey, giving a quick visual for how well the algo tracks.
    """
    full_df = make_daily_series(n_days=365, col="sessions", base=1000, trend=0.8, weekly_amp=120, noise=25, seed=1)

    forecast_config = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("sessions",),
        freq="D",
        forecast_horizon=28,
        coverage=0.95,
        algo_name="simple",
        algo_params={"period": 7, "k": 6},
    )
    backfill_config = BackfillConfig(
        forecast_config=forecast_config,
        initial_train_size=90,
        horizon=14,
        step=45,
    )
    result = BackfillRunner(backfill_config).run(df=full_df)
    bdf = result.result_df
    assert bdf is not None

    cutoffs = sorted(bdf[CUTOFF_COL].unique())

    fig = go.Figure()

    # Full actual series as grey backdrop.
    fig.add_trace(
        go.Scatter(
            name="Actual",
            x=full_df[TIME_COL],
            y=full_df["sessions"],
            mode="lines",
            line=dict(color="rgba(60,60,60,0.45)", width=1.5),
        )
    )

    # One forecast line + CI band per cutoff.
    for idx, cutoff_ts in enumerate(cutoffs):
        color = CUTOFF_COLORS[idx % len(CUTOFF_COLORS)]
        band_color = color.replace("0.85", "0.12")
        label = str(cutoff_ts.date())
        window = bdf[bdf[CUTOFF_COL] == cutoff_ts].sort_values(TIME_COL)

        fig.add_trace(
            go.Scatter(
                name=f"Cutoff {label}",
                x=window[TIME_COL],
                y=window[FORECAST_COL],
                mode="lines",
                line=dict(color=color, width=2),
            )
        )

        ci_window = window.dropna(subset=[FORECAST_LOWER_COL, FORECAST_UPPER_COL])
        if not ci_window.empty:
            fig.add_trace(
                go.Scatter(
                    x=ci_window[TIME_COL],
                    y=ci_window[FORECAST_LOWER_COL],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ci_window[TIME_COL],
                    y=ci_window[FORECAST_UPPER_COL],
                    mode="lines",
                    fill="tonexty",
                    fillcolor=band_color,
                    line=dict(width=0),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title_text="Sessions — BackfillRunner spaghetti (period=7, k=6, step=45d, horizon=14d)",
        title_x=0.5,
        xaxis_title=TIME_COL,
        hovermode="x unified",
        height=500,
    )

    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "simple_forecast_backfill_spaghetti.html"
    fig.write_html(str(path))
    assert path.exists()

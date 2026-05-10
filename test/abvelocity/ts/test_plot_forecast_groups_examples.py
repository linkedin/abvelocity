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
"""Visual examples for :func:`plot_forecast_groups_vs_actual` on synthetic data.

Each test runs ``BackfillRunner`` with :class:`SimpleForecastAlgo` (pure
seasonal-naive — period=7, k=4 to keep the trend-lag bias under ~1 %) on
synthetic daily data, then renders one HTML under
``docs/static/test-results/timeseries/`` so reviewers can eyeball the
result without re-running.

Why ``k=4`` rather than ``k=8``: simple seasonal-naive averages the last
``k`` same-DOW values; on a series with non-trivial trend (``0.8/day``
here), a high ``k`` anchors the forecast multiple weeks behind the
current level. ``k=4`` keeps the mean lag ≈17 days, small enough that
the forecasts visually track the actuals while still having seasonal
structure.

Scenarios
---------
1. **Rolling H-day-ahead overlay** — three horizons (1, 7, 14) over a
   year of synthetic signups; CI band per horizon shares its line's color.
2. **Per-cutoff trajectory overlay** — eight rolling training cutoffs,
   each with its own forecast trajectory; same generic function, different
   ``group_col``.
"""

import os
from pathlib import Path
from typing import Tuple

# Trigger self-registration of SimpleForecastAlgo into ALGO_REGISTRY.
import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401
import pandas as pd
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.backfill.runner import BackfillRunner
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import (
    ACTUAL_COL,
    CUTOFF_COL,
    HORIZON_STEP_COL,
    TIME_COL,
)
from abvelocity.ts.testing_utils import make_daily_series
from abvelocity.ts.viz import plot_forecast_groups_vs_actual

# Output directory — parents: ts/ abvelocity/ blah/ test/ abvelocity/ abvelocity/.
WRITE_PATH = Path(__file__).parents[3] / "docs" / "static" / "test-results" / "timeseries"


def make_backfill_panel(
    n_days: int = 365,
    metric_col: str = "signups",
    base: float = 1000.0,
    trend: float = 0.8,
    weekly_amp: float = 120.0,
    annual_amp: float = 80.0,
    noise: float = 18.0,
    initial_train_size: int = 90,
    horizon: int = 21,
    step: int = 1,
    period: int = 7,
    k: int = 4,
    seed: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesize daily actuals, run BackfillRunner, return (panel, actuals).

    The panel has one row per (cutoff, forecasted-date) with
    ``horizon_step = (ts - cutoff).dt.days`` added and rows filtered to
    ``horizon_step >= 1`` (forecast rows only). It is **only** the
    forecast-window rows — sparse cutoffs leave gaps in the calendar.

    The returned ``actuals_overlay`` DataFrame is the gapless ground-truth
    series (one row per calendar day in the synthetic window). Pass it
    to :func:`plot_forecast_groups_vs_actual` via ``actuals_df=...`` so
    the actuals trace is continuous regardless of cutoff sparsity.
    """
    actuals_df = make_daily_series(
        n_days=n_days,
        col=metric_col,
        base=base,
        trend=trend,
        weekly_amp=weekly_amp,
        annual_amp=annual_amp,
        noise=noise,
        seed=seed,
    )

    forecast_config = ForecastConfig(
        time_col=TIME_COL,
        value_cols=(metric_col,),
        freq="D",
        forecast_horizon=horizon,
        coverage=0.95,
        algo_name="simple",
        algo_params={"period": period, "k": k},
    )
    backfill_config = BackfillConfig(
        forecast_config=forecast_config,
        initial_train_size=initial_train_size,
        horizon=horizon,
        step=step,
    )
    panel = BackfillRunner(backfill_config).run(df=actuals_df).result_df.copy()

    actuals_overlay = actuals_df.rename(columns={metric_col: ACTUAL_COL})
    panel[ACTUAL_COL] = panel[TIME_COL].map(actuals_overlay.set_index(TIME_COL)[ACTUAL_COL])
    panel[HORIZON_STEP_COL] = (panel[TIME_COL] - panel[CUTOFF_COL]).dt.days
    panel = panel[panel[HORIZON_STEP_COL] >= 1].reset_index(drop=True)
    return panel, actuals_overlay


def test_scenario_groups_by_horizon_overlay():
    """3 fixed horizons {1, 7, 14} as rolling forecast curves overlaid on actuals.

    Saved to ``forecast_groups_by_horizon.html``. Each horizon's CI band
    shares its line's color so the band visually belongs to the line.

    Uses **daily** cutoffs (step=1) so each horizon-step line has one
    forecast per calendar day — that's what makes within-week
    seasonality visible in the line shape. With weekly cutoffs (step=7),
    every H=1 forecast would land on the same DOW and the line would
    sample only one weekday — masking the weekly pattern.

    Simple-seasonal-naive (k=4) under-forecasts by ~1 % on average due
    to trend lag — that's a real artifact of a seasonal-only model on
    a trended series, not a plotting bug.
    """
    panel, actuals_overlay = make_backfill_panel()

    fig = plot_forecast_groups_vs_actual(
        df=panel,
        group_col=HORIZON_STEP_COL,
        group_values=(1, 7, 14),
        group_label_template="H={value}d",
        actuals_df=actuals_overlay,
        title="Signups — rolling H-day-ahead forecasts vs actual (simple seasonal-naive, k=4)",
        ylabel="signups",
    )

    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "forecast_groups_by_horizon.html"
    fig.write_html(str(path))
    assert path.exists()


def test_scenario_groups_by_cutoff_overlay():
    """Multiple training cutoffs as forecast trajectories, each in its own color.

    Saved to ``forecast_groups_by_cutoff.html``. Demonstrates the same
    function works when the grouping dimension is the training cutoff
    (not the horizon) — same code path, different ``group_col``.
    Sparse cutoffs (step=45) leave gaps between forecast windows; the
    actuals trace is rendered from the gapless ``actuals_df`` kwarg
    rather than deduping from ``panel`` so the actual line stays
    continuous across the gaps.
    """
    panel, actuals_overlay = make_backfill_panel(
        step=45,
        horizon=28,
        initial_train_size=90,
        seed=1,
        metric_col="sessions",
    )

    # Use a string-formatted cutoff label so legend entries are readable
    # ("cutoff 2023-04-15" rather than the full Timestamp repr).
    panel = panel.copy()
    panel["cutoff_label"] = panel[CUTOFF_COL].dt.strftime("%Y-%m-%d")
    cutoff_labels = sorted(panel["cutoff_label"].unique())

    fig = plot_forecast_groups_vs_actual(
        df=panel,
        group_col="cutoff_label",
        group_values=cutoff_labels,
        group_label_template="cutoff {value}",
        actuals_df=actuals_overlay,
        title="Sessions — per-cutoff forecast trajectories vs actual (simple seasonal-naive, k=4)",
        ylabel="sessions",
    )

    os.makedirs(WRITE_PATH, exist_ok=True)
    path = WRITE_PATH / "forecast_groups_by_cutoff.html"
    fig.write_html(str(path))
    assert path.exists()

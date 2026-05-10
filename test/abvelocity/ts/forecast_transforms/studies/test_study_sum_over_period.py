# Original author: Reza Hosseini
"""End-to-end study: synthetic daily series → SumOverPeriod (W-SAT, MS).

Generates a 2-year daily synthetic series, fits Silverkite once, then
writes one report directory per period scenario:

  - ``sum_over_period_weekly``  (W-SAT)
  - ``sum_over_period_monthly`` (MS)

Gated under :data:`gk_test_gate` since it pulls greykite.  Run with
``RUN_ALL_GK_TESTS=1 pytest -s …``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.forecast_runner import ForecastRunner
from abvelocity.ts.forecast_transforms import SumOverPeriod
from abvelocity.ts.viz import make_input_and_transform_plots_sections, write_index_html
from abvelocity.ts.gk_test_gate import gk_test_gate

from .synthetic_data import daily_series

pytestmark = gk_test_gate

REPORTS_ROOT = Path(__file__).resolve().parents[5] / "docs" / "static" / "test-results-local" / "forecast_transforms_studies"


def test_study_sum_over_period() -> None:
    raw_df = daily_series(n_days=365 * 2, seed=7)
    daily_df = ForecastRunner(
        config=ForecastConfig(
            algo_name="greykite",
            value_cols=("y",),
            freq="D",
            forecast_horizon=60,
            coverage=0.80,
            train_end_date=str(raw_df["ts"].max().date()),
            algo_params={"breakdown_origin": "first_value"},
            metric_id_template="signups:daily",
            metric_name_template="Signups (daily)",
        ),
    ).run(df=raw_df).result_df

    weekly_df = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_df)
    monthly_df = SumOverPeriod(period="MS").apply(forecast_df=daily_df)
    quarterly_df = SumOverPeriod(period="QS").apply(forecast_df=daily_df)

    assert not weekly_df.empty
    assert not monthly_df.empty
    assert not quarterly_df.empty
    assert (weekly_df["forecast"] > 0).all()
    assert (monthly_df["forecast"] > 0).all()
    assert (quarterly_df["forecast"] > 0).all()

    write_index_html(
        output_dir=REPORTS_ROOT / "sum_over_period_weekly",
        study_name="SumOverPeriod study — daily → weekly (synthetic)",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption="Silverkite fit on a 2-year synthetic daily series with weekly + annual seasonality.",
            transform_df=weekly_df,
            transform_title="Weekly aggregation — SumOverPeriod(\"W-SAT\")",
            transform_caption=f"Output: {len(weekly_df)} weeks; forecast values sum to {weekly_df['forecast'].sum():.1f}.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "sum_over_period_monthly",
        study_name="SumOverPeriod study — daily → monthly (synthetic)",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption="Silverkite fit on a 2-year synthetic daily series with weekly + annual seasonality.",
            transform_df=monthly_df,
            transform_title="Monthly aggregation — SumOverPeriod(\"MS\")",
            transform_caption=f"Output: {len(monthly_df)} months; forecast values sum to {monthly_df['forecast'].sum():.1f}.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "sum_over_period_quarterly",
        study_name="SumOverPeriod study — daily → quarterly (synthetic)",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption="Silverkite fit on a 2-year synthetic daily series with weekly + annual seasonality.",
            transform_df=quarterly_df,
            transform_title="Quarterly aggregation — SumOverPeriod(\"QS\")",
            transform_caption=f"Output: {len(quarterly_df)} quarters; forecast values sum to {quarterly_df['forecast'].sum():.1f}.",
        ),
    )

    daily_total = daily_df["forecast"].sum()
    weekly_total = weekly_df["forecast"].sum()
    assert weekly_total <= daily_total
    assert weekly_total > 0.5 * daily_total

    assert pd.api.types.is_datetime64_any_dtype(weekly_df["forecasted_date"])
    assert pd.api.types.is_datetime64_any_dtype(monthly_df["forecasted_date"])

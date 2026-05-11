# Original author: Reza Hosseini
"""End-to-end study: DC bike sharing daily counts → period transforms.

Fits Silverkite once on the daily total bike-rental count, then writes
one report directory per transform scenario:

  - ``bikesharing_raw_daily_fit``
  - ``bikesharing_sum_over_period_weekly``  (W-SAT)
  - ``bikesharing_sum_over_period_monthly`` (MS)
  - ``bikesharing_weight_over_period_weekly``
  - ``bikesharing_weight_over_period_monthly``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from abvelocity.ts.forecast_transforms import SumOverPeriod, WeightOverPeriod
from abvelocity.ts.gk_test_gate import gk_test_gate
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.forecast_runner import ForecastRunner
from abvelocity.ts.viz import (
    make_forecast_plots_section,
    make_input_and_transform_plots_sections,
    write_index_html,
)

from .real_datasets import load_bikesharing_daily

pytestmark = gk_test_gate

REPORTS_ROOT = Path(__file__).resolve().parents[5] / "docs" / "static" / "test-results-local" / "forecast_transforms_studies"


def test_study_bikesharing_period_variants() -> None:
    raw_df = load_bikesharing_daily()
    raw_df_short = raw_df[["ts", "y"]]
    daily_df = ForecastRunner(
        config=ForecastConfig(
            algo_name="greykite",
            value_cols=("y",),
            freq="D",
            forecast_horizon=60,
            coverage=0.80,
            train_end_date=str(raw_df_short["ts"].max().date()),
            algo_params={"breakdown_origin": "first_value"},
            metric_id_template="bike_rentals:daily",
            metric_name_template="DC bike rentals (daily)",
        ),
    ).run(df=raw_df_short).result_df

    weekly_sum_df = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_df)
    monthly_sum_df = SumOverPeriod(period="MS").apply(forecast_df=daily_df)
    quarterly_sum_df = SumOverPeriod(period="QS").apply(forecast_df=daily_df)
    weekly_share_df = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_df)
    monthly_share_df = WeightOverPeriod(period="MS").apply(forecast_df=daily_df)

    assert not weekly_sum_df.empty
    assert not monthly_sum_df.empty
    assert (weekly_sum_df["forecast"] > 0).all()
    assert (monthly_sum_df["forecast"] > 0).all()
    week_sums = (
        weekly_share_df.dropna(subset=["forecast"])
        .groupby(by=weekly_share_df.dropna(subset=["forecast"])["forecasted_date"].dt.to_period("W-SAT"))["forecast"]
        .sum()
    )
    assert (abs(week_sums - 1.0) < 1e-6).all()

    raw_caption = (
        f"DC Capital Bikeshare daily totals "
        f"({raw_df['ts'].min().date()} → {raw_df['ts'].max().date()}, "
        f"{len(raw_df)} days). 60-day forecast horizon."
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "bikesharing_raw_daily_fit",
        study_name="Bikesharing — daily Silverkite fit (no transform)",
        sections=[make_forecast_plots_section(daily_df, "Daily bike rentals — Silverkite fit", raw_caption)],
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "bikesharing_sum_over_period_weekly",
        study_name="Bikesharing — SumOverPeriod(\"W-SAT\")",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption=raw_caption,
            transform_df=weekly_sum_df,
            transform_title="Weekly totals (sum_over_period W-SAT)",
            transform_caption=f"Output: {len(weekly_sum_df)} weeks. Forecasts and components summed across each Sun–Sat week.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "bikesharing_sum_over_period_monthly",
        study_name="Bikesharing — SumOverPeriod(\"MS\")",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption=raw_caption,
            transform_df=monthly_sum_df,
            transform_title="Monthly totals (sum_over_period MS)",
            transform_caption=f"Output: {len(monthly_sum_df)} months. Daily forecast summed into calendar-month buckets.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "bikesharing_sum_over_period_quarterly",
        study_name="Bikesharing — SumOverPeriod(\"QS\")",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption=raw_caption,
            transform_df=quarterly_sum_df,
            transform_title="Quarterly totals (sum_over_period QS)",
            transform_caption=f"Output: {len(quarterly_sum_df)} quarters. Daily forecast summed into calendar-quarter buckets — useful for visualizing growth.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "bikesharing_weight_over_period_weekly",
        study_name="Bikesharing — WeightOverPeriod(\"W-SAT\")",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption=raw_caption,
            transform_df=weekly_share_df,
            transform_title="Day-of-week shares within each W-SAT week",
            transform_caption=f"Each daily row → its share of the Sun–Sat week's total. Complete weeks: {len(week_sums)}.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "bikesharing_weight_over_period_monthly",
        study_name="Bikesharing — WeightOverPeriod(\"MS\")",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption=raw_caption,
            transform_df=monthly_share_df,
            transform_title="Day-of-month shares within each calendar month",
            transform_caption="Each daily row → its share of its calendar-month total.",
        ),
    )

    assert pd.api.types.is_datetime64_any_dtype(weekly_sum_df["forecasted_date"])

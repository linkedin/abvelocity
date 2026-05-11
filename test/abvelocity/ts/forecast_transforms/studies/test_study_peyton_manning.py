# Original author: Reza Hosseini
"""End-to-end study: Peyton Manning Wikipedia views with US holidays.

The classic Prophet/Silverkite demo dataset.  Strong yearly seasonality,
weekly seasonality, and pronounced holiday spikes aligned with NFL
playoff weekends.

The Silverkite fit explicitly enables US holidays so the ``Event``
breakdown group ends up non-trivial; that group maps to
``holiday_impact`` in the assembled forecast frame, surfaced as a
component subplot in every report.

One report directory per scenario:

  - ``peyton_manning_raw_daily_fit``
  - ``peyton_manning_sum_over_period_weekly``  (W-SAT)
  - ``peyton_manning_sum_over_period_monthly`` (MS)
  - ``peyton_manning_weight_over_period_weekly``
  - ``peyton_manning_weight_over_period_monthly``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.forecast_runner import ForecastRunner
from abvelocity.ts.forecast_transforms import SumOverPeriod, WeightOverPeriod
from abvelocity.ts.viz import (
    make_forecast_plots_section,
    make_input_and_transform_plots_sections,
    write_index_html,
)
from abvelocity.ts.gk_test_gate import gk_test_gate

from .real_datasets import load_peyton_manning_window

pytestmark = gk_test_gate

REPORTS_ROOT = Path(__file__).resolve().parents[5] / "docs" / "static" / "test-results-local" / "forecast_transforms_studies"


def test_study_peyton_manning_holidays() -> None:
    raw_df = load_peyton_manning_window()

    raw_df_short = raw_df[["ts", "y"]]
    daily_df = ForecastRunner(
        config=ForecastConfig(
            algo_name="greykite",
            value_cols=("y",),
            freq="D",
            forecast_horizon=90,
            coverage=0.80,
            train_end_date=str(raw_df_short["ts"].max().date()),
            algo_params={
                "model_components": {
                    "events": {
                        "holiday_lookup_countries": ["UnitedStates"],
                        "holidays_to_model_separately": "auto",
                        "holiday_pre_num_days": 2,
                        "holiday_post_num_days": 2,
                    },
                },
                "breakdown_origin": "first_value",
            },
            metric_id_template="peyton_manning:log_views",
            metric_name_template="Peyton Manning Wikipedia (log views)",
        ),
    ).run(df=raw_df_short).result_df

    weekly_sum_df = SumOverPeriod(period="W-SAT").apply(forecast_df=daily_df)
    monthly_sum_df = SumOverPeriod(period="MS").apply(forecast_df=daily_df)
    quarterly_sum_df = SumOverPeriod(period="QS").apply(forecast_df=daily_df)
    weekly_share_df = WeightOverPeriod(period="W-SAT").apply(forecast_df=daily_df)
    monthly_share_df = WeightOverPeriod(period="MS").apply(forecast_df=daily_df)

    assert not weekly_sum_df.empty
    assert not monthly_sum_df.empty
    assert not quarterly_sum_df.empty

    raw_caption = (
        f"Wikipedia daily page views (log) "
        f"{raw_df['ts'].min().date()} → {raw_df['ts'].max().date()}, "
        f"{len(raw_df)} days; 90-day forecast horizon. "
        f"Holiday impacts spike around Thanksgiving / playoff weekends."
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "peyton_manning_raw_daily_fit",
        study_name="Peyton Manning — daily Silverkite fit with US holidays",
        sections=[make_forecast_plots_section(daily_df, "Daily Peyton Manning views — Silverkite + US holidays", raw_caption)],
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "peyton_manning_sum_over_period_weekly",
        study_name="Peyton Manning — SumOverPeriod(\"W-SAT\")",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption=raw_caption,
            transform_df=weekly_sum_df,
            transform_title="Weekly totals (sum_over_period W-SAT)",
            transform_caption=f"Output: {len(weekly_sum_df)} weeks. Component breakdown is summed across the week.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "peyton_manning_sum_over_period_monthly",
        study_name="Peyton Manning — SumOverPeriod(\"MS\")",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption=raw_caption,
            transform_df=monthly_sum_df,
            transform_title="Monthly totals (sum_over_period MS)",
            transform_caption=f"Output: {len(monthly_sum_df)} months.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "peyton_manning_sum_over_period_quarterly",
        study_name="Peyton Manning — SumOverPeriod(\"QS\")",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption=raw_caption,
            transform_df=quarterly_sum_df,
            transform_title="Quarterly totals (sum_over_period QS)",
            transform_caption=f"Output: {len(quarterly_sum_df)} quarters — coarse buckets show the underlying growth shape.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "peyton_manning_weight_over_period_weekly",
        study_name="Peyton Manning — WeightOverPeriod(\"W-SAT\")",
        sections=make_input_and_transform_plots_sections(
            input_df=daily_df,
            input_title="Daily Silverkite fit (input)",
            input_caption=raw_caption,
            transform_df=weekly_share_df,
            transform_title="Day-of-week shares within each W-SAT week",
            transform_caption="Each daily row → its share of its Sun–Sat week's total.",
        ),
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "peyton_manning_weight_over_period_monthly",
        study_name="Peyton Manning — WeightOverPeriod(\"MS\")",
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

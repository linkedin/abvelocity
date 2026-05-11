# Original author: Reza Hosseini
"""End-to-end study: daily Silverkite fit → WeightOverPeriod (W-SAT).

Each daily row becomes its share of its enclosing W-SAT week's total.
Writes an HTML report with the daily fit + the share series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abvelocity.ts.forecast_transforms import WeightOverPeriod
from abvelocity.ts.gk_test_gate import gk_test_gate
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.forecast_runner import ForecastRunner
from abvelocity.ts.viz import plot_breakdown, plot_forecast, write_index_html

from .synthetic_data import daily_series

pytestmark = gk_test_gate

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "docs" / "static" / "test-results-local" / "forecast_transforms_studies" / "weight_over_period"


def test_study_weight_over_period() -> None:
    raw_df = daily_series(n_days=365 * 2, weekly_amp=20.0, seed=11)
    forecast_df = ForecastRunner(
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

    share_df = WeightOverPeriod(period="W-SAT").apply(forecast_df=forecast_df)

    # Within each complete week, forecast shares sum to ~1.  Incomplete
    # weeks (partial first/last) come back NaN-masked; drop those rows
    # before the weekly groupby so the sum isn't a phantom 0.
    share_df = share_df.copy()
    share_df["_week_start"] = share_df["forecasted_date"].dt.to_period("W-SAT").dt.start_time
    complete_share_df = share_df.dropna(subset=["forecast"])
    week_sums = complete_share_df.groupby("_week_start")["forecast"].sum()
    assert (np.isclose(week_sums, 1.0, atol=1e-6)).all()
    assert (complete_share_df["forecast"] > 0).all()

    share_no_helper = share_df.drop(columns=["_week_start"])
    daily_section = [
        plot_forecast(result_df=forecast_df, title="Daily Silverkite forecast"),
        plot_breakdown(result_df=forecast_df, title="Daily — breakdown components"),
    ]
    share_section = [
        plot_forecast(
            result_df=share_no_helper,
            title="Daily share within W-SAT week (forecast / week-total)",
        ),
        plot_breakdown(
            result_df=share_no_helper,
            title="Daily share — breakdown components (each / week-total)",
        ),
    ]

    out_path = write_index_html(
        output_dir=OUTPUT_DIR,
        study_name="WeightOverPeriod study — daily share within W-SAT week",
        sections=[
            (
                "Input forecast (daily)",
                "Silverkite fit, 2-year daily synthetic series.",
                [fig for fig in daily_section if fig is not None],
            ),
            (
                "Daily share within week",
                f"Each daily row replaced by its share of its W-SAT week's total. "
                f"Output rows: {len(share_df)}, complete weeks: {len(week_sums)}.",
                [fig for fig in share_section if fig is not None],
            ),
        ],
    )
    assert out_path.exists()
    print(f"\nstudy report written: {out_path}")

    # Output dtype sanity.
    assert pd.api.types.is_datetime64_any_dtype(share_df["forecasted_date"])

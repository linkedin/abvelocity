# Original author: Reza Hosseini
"""End-to-end study: country × device daily fits → WeightOverDims (within country).

Fits Silverkite per (country, device), stacks results, then runs
``WeightOverDims(within_dims=("country",))`` so each row becomes the
device's share of its country-day total.  Writes an HTML report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.forecast_runner import ForecastRunner
from abvelocity.ts.forecast_transforms import WeightOverDims
from abvelocity.ts.viz import plot_breakdown, plot_forecast, write_index_html
from abvelocity.ts.gk_test_gate import gk_test_gate

from .synthetic_data import country_device_panel

pytestmark = gk_test_gate

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "docs" / "static" / "test-results-local" / "forecast_transforms_studies" / "weight_over_dims"


def test_study_weight_over_dims() -> None:
    panel_df = country_device_panel(n_days=365, seed=11)

    forecast_df = ForecastRunner(
        config=ForecastConfig(
            algo_name="greykite",
            value_cols=("y",),
            dim_cols=("country", "device"),
            freq="D",
            forecast_horizon=30,
            coverage=0.80,
            train_end_date=str(panel_df["ts"].max().date()),
            algo_params={"breakdown_origin": "first_value"},
            metric_id_template="signups:daily",
            metric_name_template="Signups (daily)",
        ),
    ).run(df=panel_df).result_df

    share_df = WeightOverDims(within_dims=("country",)).apply(forecast_df=forecast_df)

    # Per (country, date) the device-shares should sum to ~1.
    country_day_sum = share_df.groupby(by=["country", "forecasted_date"])["forecast"].sum()
    assert (np.isclose(country_day_sum.dropna(), 1.0, atol=1e-6)).all()

    segments_section = [
        plot_forecast(
            result_df=forecast_df,
            title="Per-segment forecasts (country × device)",
            groupby=("country", "device"),
        ),
        plot_breakdown(
            result_df=forecast_df,
            title="Per-segment — breakdown components",
            groupby=("country", "device"),
        ),
    ]
    share_section = [
        plot_forecast(
            result_df=share_df,
            title="Device share within country (forecast / country-day total)",
            groupby=("country", "device"),
        ),
        plot_breakdown(
            result_df=share_df,
            title="Device-share — breakdown components (each / country-day total)",
            groupby=("country", "device"),
        ),
    ]

    out_path = write_index_html(
        output_dir=OUTPUT_DIR,
        study_name="WeightOverDims study — device share within country",
        sections=[
            (
                "Per-segment forecasts",
                "Silverkite fit per (country, device) segment.",
                [fig for fig in segments_section if fig is not None],
            ),
            (
                "Device share within country",
                f"Output rows: {len(share_df)}; "
                f"shares sum to 1 within each (country, day) group.",
                [fig for fig in share_section if fig is not None],
            ),
        ],
    )
    assert out_path.exists()
    print(f"\nstudy report written: {out_path}")

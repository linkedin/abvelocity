# Original author: Reza Hosseini
"""End-to-end study: country × device daily fits → SumOverDims (drop device).

Fits Silverkite per (country, device) segment, stacks results, then runs
``SumOverDims(dims_summed=("device",), dims_maintained=("country",))`` to
get country-level forecasts.  Writes an HTML report comparing per-segment
fits with the aggregated country-level series.
"""

from __future__ import annotations

from pathlib import Path

from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.forecast_runner import ForecastRunner
from abvelocity.ts.forecast_transforms import SumOverDims
from abvelocity.ts.viz import plot_breakdown, plot_forecast, write_index_html
from abvelocity.ts.gk_test_gate import gk_test_gate

from .synthetic_data import country_device_panel

pytestmark = gk_test_gate

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "docs" / "static" / "test-results-local" / "forecast_transforms_studies" / "sum_over_dims"


def test_study_sum_over_dims() -> None:
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

    country_df = SumOverDims(
        dims_summed=("device",),
        dims_maintained=("country",),
    ).apply(forecast_df=forecast_df)

    assert "device" not in country_df.columns
    assert "country" in country_df.columns
    assert (country_df["forecast"] > 0).all()

    # Sanity: country forecast == sum of (country, device) device forecasts at the same date.
    sample_date = forecast_df["forecasted_date"].iloc[0]
    us_devices_total = forecast_df.loc[
        (forecast_df["country"] == "US") & (forecast_df["forecasted_date"] == sample_date),
        "forecast",
    ].sum()
    us_country = country_df.loc[
        (country_df["country"] == "US") & (country_df["forecasted_date"] == sample_date),
        "forecast",
    ].iloc[0]
    assert abs(us_country - us_devices_total) < 1e-6

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
    country_section = [
        plot_forecast(
            result_df=country_df,
            title="Country-level forecasts (device summed out)",
            groupby=("country",),
        ),
        plot_breakdown(
            result_df=country_df,
            title="Country-level — breakdown components (summed across devices)",
            groupby=("country",),
        ),
    ]

    out_path = write_index_html(
        output_dir=OUTPUT_DIR,
        study_name="SumOverDims study — country × device → country",
        sections=[
            (
                "Per-segment forecasts",
                "Silverkite fit per (country, device) segment.",
                [fig for fig in segments_section if fig is not None],
            ),
            (
                "Country-level after sum_over_dims",
                f"Output rows: {len(country_df)}; countries: "
                f"{sorted(country_df['country'].unique())}.",
                [fig for fig in country_section if fig is not None],
            ),
        ],
    )
    assert out_path.exists()
    print(f"\nstudy report written: {out_path}")

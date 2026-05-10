# Original author: Reza Hosseini
"""End-to-end study: Birmingham parking occupancy panel → dim transforms.

Fits Silverkite once per (station), then writes one report directory per
scenario:

  - ``parking_raw_per_station_fits``
  - ``parking_sum_over_dims``  (sum across stations → total)
  - ``parking_weight_over_dims``  (each station's share of total)
"""

from __future__ import annotations

from pathlib import Path

from abvelocity.ts.forecast_transforms import SumOverDims, WeightOverDims
from abvelocity.ts.gk_test_gate import gk_test_gate
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.forecast_runner import ForecastRunner
from abvelocity.ts.viz import plot_breakdown, plot_forecast, write_index_html

from .real_datasets import PARKING_STATIONS, load_parking_panel

pytestmark = gk_test_gate

REPORTS_ROOT = Path(__file__).resolve().parents[5] / "docs" / "static" / "test-results-local" / "forecast_transforms_studies"


def test_study_parking_dim_variants() -> None:
    panel_df = load_parking_panel()
    forecast_df = ForecastRunner(
        config=ForecastConfig(
            algo_name="greykite",
            value_cols=("y",),
            dim_cols=("station",),
            freq="D",
            forecast_horizon=14,
            coverage=0.80,
            train_end_date=str(panel_df["ts"].max().date()),
            algo_params={"breakdown_origin": "first_value"},
            metric_id_template="parking:occupancy_ratio",
            metric_name_template="Mean daily occupancy ratio",
        ),
    ).run(df=panel_df).result_df

    total_df = SumOverDims(dims_summed=("station",)).apply(forecast_df=forecast_df)
    share_df = WeightOverDims().apply(forecast_df=forecast_df)

    assert "station" not in total_df.columns
    assert "station" in share_df.columns
    day_sums = share_df.groupby(by="forecasted_date")["forecast"].sum()
    assert (abs(day_sums.dropna() - 1.0) < 1e-6).all()

    raw_caption = (
        f"Birmingham car-park dataset; daily mean occupancy ratio per station. "
        f"Stations: {list(PARKING_STATIONS)}."
    )

    def per_station_section(title):
        return (
            title,
            raw_caption,
            [
                fig
                for fig in (
                    plot_forecast(
                        result_df=forecast_df,
                        title=f"{title} — forecast",
                        groupby=("station",),
                    ),
                    plot_breakdown(
                        result_df=forecast_df,
                        title=f"{title} — breakdown components",
                        groupby=("station",),
                    ),
                )
                if fig is not None
            ],
        )

    write_index_html(
        output_dir=REPORTS_ROOT / "parking_raw_per_station_fits",
        study_name="Parking — per-station Silverkite fits (no transform)",
        sections=[per_station_section(title="Per-station fits (input)")],
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "parking_sum_over_dims",
        study_name="Parking — SumOverDims((\"station\",))",
        sections=[
            per_station_section(title="Per-station fits (input)"),
            (
                "Total occupancy after summing across stations",
                f"Output: {len(total_df)} daily rows; station dim collapsed.",
                [
                    fig
                    for fig in (
                        plot_forecast(result_df=total_df, title="Total daily occupancy — forecast"),
                        plot_breakdown(result_df=total_df, title="Total — breakdown components"),
                    )
                    if fig is not None
                ],
            ),
        ],
    )
    write_index_html(
        output_dir=REPORTS_ROOT / "parking_weight_over_dims",
        study_name="Parking — WeightOverDims()",
        sections=[
            per_station_section(title="Per-station fits (input)"),
            (
                "Each station's share of total daily occupancy",
                f"Output: {len(share_df)} rows. Shares sum to 1 within every day "
                f"across the {len(PARKING_STATIONS)} stations.",
                [
                    fig
                    for fig in (
                        plot_forecast(
                            result_df=share_df,
                            title="Per-station share of daily total — forecast",
                            groupby=("station",),
                        ),
                        plot_breakdown(
                            result_df=share_df,
                            title="Per-station share — breakdown components",
                            groupby=("station",),
                        ),
                    )
                    if fig is not None
                ],
            ),
        ],
    )

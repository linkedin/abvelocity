# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

"""
Test dimension support with timeseries data using DuckDB cursor.

This test demonstrates the full pipeline:
1. Generate simulated experiment data
2. Add random dates to create timeseries
3. Load data into DuckDB
4. Query with dimensions (date, variant, experiment_status)
5. Plot timeseries results
"""

from pathlib import Path

import pandas as pd
from abvelocity.core.get_data.agg_metrics_query import get_agg_metrics_query_from_info
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.sim.examples import simulate_data_uni1
from abvelocity.core.utils.gen_combined_figs_html import gen_combined_figs_html
from abvelocity.core.utils.plot_lines_markers import plot_long_df

# Path for saving HTML plots
PLOT_WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/timeseries").resolve()


def test_timeseries_with_dims_and_duckdb():
    """
    End-to-end test: Generate sim data, add dates, query with dims, and plot timeseries.
    """
    # =========================================================================
    # Step 1: Simulate data and add dates
    # =========================================================================
    sim = simulate_data_uni1(population_size=5000, population_seed=42)
    sim.add_dates(
        start_date="2024-01-01",
        end_date="2024-01-28",
        date_unit="D",
        date_col="date",
        date_seed=123,
    )
    df_with_dates = sim.expt_metric_df

    # Add experiment_status dimension: first 2 weeks ramping, last 2 weeks active
    df_with_dates["experiment_status"] = df_with_dates["date"].apply(lambda x: "ramping" if x < "2024-01-15" else "active")

    # =========================================================================
    # Step 2: Load into DuckDB
    # =========================================================================
    cursor = DuckDBCursor(max_retries=1)
    cursor._db_connection.register("metrics_temp", df_with_dates)
    cursor._db_connection.execute("CREATE TABLE metrics AS SELECT * FROM metrics_temp;")
    cursor._db_connection.unregister("metrics_temp")

    # =========================================================================
    # Step 3: Define metrics and family
    # =========================================================================
    metric1 = Metric(
        name="total_metric1",
        numerator=UMetric(col="metric1", agg="MAX", fill_na=0, name="metric1"),
        numerator_agg="SUM",
    )
    metric2 = Metric(
        name="total_metric2",
        numerator=UMetric(col="metric2", agg="MAX", fill_na=0, name="metric2"),
        numerator_agg="SUM",
    )
    metric_family = MetricFamily(
        name="test_metrics",
        u_metrics_query=UMetricsQuery(table_name="metrics", date_col="date"),
        metric_join_unit_col="id",
    )
    metric_cols = ["total_metric1", "total_metric2"]

    # =========================================================================
    # Step 4: Run queries for each dimension combination
    # =========================================================================
    def run_query(dims):
        info = MetricInfo(metric_family=metric_family, metrics=[metric1, metric2], dims=dims)
        query = get_agg_metrics_query_from_info(metric_info=info, start_date="2024-01-01", end_date="2024-01-28")
        result_df = cursor.get_df(query).df
        result_df["date"] = pd.to_datetime(result_df["date"])
        return result_df

    daily_df = run_query(dims=["date"]).sort_values("date")

    daily_variant_df = run_query(dims=["date", "variant"])
    daily_variant_df["variant"] = daily_variant_df["variant"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x))
    daily_variant_df = daily_variant_df.sort_values(["variant", "date"])

    daily_variant_status_df = run_query(dims=["date", "variant", "experiment_status"])
    daily_variant_status_df["variant"] = daily_variant_status_df["variant"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x))
    daily_variant_status_df = daily_variant_status_df.sort_values(["variant", "experiment_status", "date"])

    # =========================================================================
    # Step 5: Plot timeseries — one file per (query slice, metric)
    # =========================================================================
    queries = [
        (daily_df, None, "daily"),
        (daily_variant_df, ["variant"], "by_variant"),
        (daily_variant_status_df, ["variant", "experiment_status"], "by_variant_status"),
    ]

    fig_dict = {
        f"{label} | {metric_col}": plot_long_df(df=df_q, x_col="date", y_col=metric_col, group_by_cols=group_cols)["fig"]
        for df_q, group_cols, label in queries
        for metric_col in metric_cols
    }

    PLOT_WRITE_PATH.mkdir(parents=True, exist_ok=True)
    gen_combined_figs_html(
        fig_dict=fig_dict,
        html_file_name=str(PLOT_WRITE_PATH / "test_timeseries_dims_duckdb.html"),
        n_cols=len(metric_cols),
    )

    # =========================================================================
    # Step 6: Assertions
    # =========================================================================
    assert len(daily_df) > 0
    assert daily_df["date"].nunique() > 1
    assert daily_variant_df["variant"].nunique() >= 2
    assert daily_variant_status_df["experiment_status"].nunique() == 2

    for date in daily_df["date"].unique():
        overall = daily_df[daily_df["date"] == date]["total_metric1"].values[0]
        variant_sum = daily_variant_df[daily_variant_df["date"] == date]["total_metric1"].sum()
        assert abs(overall - variant_sum) < 0.01, f"Totals don't match for {date}"

    cursor.close()

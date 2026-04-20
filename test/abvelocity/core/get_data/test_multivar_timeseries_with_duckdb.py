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
Test timeseries analysis with two overlapping experiments (multi-variant data).

Pipeline:
1. Simulate two overlapping experiments (variant_1: control/v1/v2, variant_2: control/enabled)
2. Add random dates via sim.add_dates()
3. Load into DuckDB
4. Query timeseries broken down by variant_1, variant_2, and combined variant tuple
5. Plot results
"""

from pathlib import Path

import pandas as pd
from abvelocity.core.get_data.agg_metrics_query import get_agg_metrics_query_from_info
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.sim.examples import simulate_data_multi1
from abvelocity.core.utils.gen_combined_figs_html import gen_combined_figs_html
from abvelocity.core.utils.plot_lines_markers import plot_long_df

PLOT_WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/timeseries").resolve()


def test_multivar_timeseries_with_duckdb():
    """
    End-to-end test: two overlapping experiments, timeseries per experiment variant.
    """
    # =========================================================================
    # Step 1: Simulate two overlapping experiments and add dates
    # =========================================================================
    sim = simulate_data_multi1(population_size=5000, population_seed=42)
    sim.add_dates(start_date="2024-01-01", end_date="2024-01-28", date_unit="D", date_col="date", date_seed=42)
    df = sim.expt_metric_df

    # =========================================================================
    # Step 2: Load into DuckDB
    # =========================================================================
    cursor = DuckDBCursor(max_retries=1)
    cursor._db_connection.register("metrics_temp", df)
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

    def run_query(dims):
        info = MetricInfo(metric_family=metric_family, metrics=[metric1, metric2], dims=dims)
        query = get_agg_metrics_query_from_info(metric_info=info, start_date="2024-01-01", end_date="2024-01-28")
        result_df = cursor.get_df(query).df
        result_df["date"] = pd.to_datetime(result_df["date"])
        return result_df

    # =========================================================================
    # Step 4: Run queries for each dimension combination
    # =========================================================================
    daily_df = run_query(dims=["date"]).sort_values("date")
    daily_v1_df = run_query(dims=["date", "variant_1"]).sort_values(["variant_1", "date"])
    daily_v2_df = run_query(dims=["date", "variant_2"]).sort_values(["variant_2", "date"])
    daily_both_df = run_query(dims=["date", "variant_1", "variant_2"]).sort_values(["variant_1", "variant_2", "date"])

    # =========================================================================
    # Step 5: Plot timeseries — one file per (query slice, metric)
    # =========================================================================
    queries = [
        (daily_df, None, "daily"),
        (daily_v1_df, ["variant_1"], "by_variant_1"),
        (daily_v2_df, ["variant_2"], "by_variant_2"),
        (daily_both_df, ["variant_1", "variant_2"], "by_variant_combo"),
    ]

    fig_dict = {
        f"{label} | {metric_col}": plot_long_df(df=df_q, x_col="date", y_col=metric_col, group_by_cols=group_cols)["fig"]
        for df_q, group_cols, label in queries
        for metric_col in metric_cols
    }

    PLOT_WRITE_PATH.mkdir(parents=True, exist_ok=True)
    gen_combined_figs_html(
        fig_dict=fig_dict,
        html_file_name=str(PLOT_WRITE_PATH / "test_multivar_timeseries_duckdb.html"),
        n_cols=len(metric_cols),
    )

    # =========================================================================
    # Step 6: Assertions
    # =========================================================================
    assert len(daily_df) == 28
    assert daily_v1_df["variant_1"].nunique() >= 3
    assert daily_v2_df["variant_2"].nunique() >= 2

    for date in daily_df["date"].unique():
        overall = daily_df[daily_df["date"] == date]["total_metric1"].values[0]
        v1_sum = daily_v1_df[daily_v1_df["date"] == date]["total_metric1"].sum()
        assert abs(overall - v1_sum) < 0.01, f"variant_1 totals don't match for {date}"

    cursor.close()


if __name__ == "__main__":
    test_multivar_timeseries_with_duckdb()

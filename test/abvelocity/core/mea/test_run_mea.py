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

import os
from pathlib import Path

import numpy as np
import pytest
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.mea.mea import MEA
from abvelocity.core.mea.run_mea import run_mea
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.sim.add_random_dates_to_df import add_random_dates_to_df
from abvelocity.core.sim.examples import simulate_data_multi1
from abvelocity.core.sim.sim import EXPT_UNIT_COL

SAVE_PATH = str(Path(__file__).parents[4].joinpath("docs/static/test-results/test-run-mea-locally/").resolve())
os.makedirs(SAVE_PATH, exist_ok=True)

START_DATE = "2024-01-01"
END_DATE = "2024-01-31"
DATE_COL = "date"
METRIC_TABLE = "sim_metrics"


@pytest.fixture
def sim_data():
    """Simulated multi-experiment data with a date column added to the metric slice."""
    sim = simulate_data_multi1()

    # Add a date column to the metric slice so UMetricsQuery WHERE clause works.
    metric_df = add_random_dates_to_df(
        df=sim.expt_metric_df[["id", "metric1", "metric2"]],
        start_date=START_DATE,
        end_date=END_DATE,
        date_col=DATE_COL,
        seed=42,
    )
    return sim, metric_df


@pytest.fixture
def duckdb_cursor(sim_data):
    """DuckDB cursor with the metric table registered."""
    _, metric_df = sim_data
    cursor = DuckDBCursor(max_retries=1)
    cursor._db_connection.register("sim_metrics_view", metric_df)
    cursor._db_connection.execute(f"CREATE TABLE {METRIC_TABLE} AS SELECT * FROM sim_metrics_view")
    cursor._db_connection.unregister("sim_metrics_view")
    yield cursor
    cursor.close()


@pytest.fixture
def analysis_info():
    """AnalysisInfo with dates set manually (no SQL expt join needed since dc is passed)."""
    expt1 = ExptInfo(name="expt1", start_date=START_DATE, end_date=END_DATE)
    expt2 = ExptInfo(name="expt2", start_date=START_DATE, end_date=END_DATE)

    metrics = [
        Metric(numerator=UMetric(col="metric1", agg="SUM", name="metric1"), name="metric1"),
        Metric(numerator=UMetric(col="metric2", agg="SUM", name="metric2"), name="metric2"),
    ]
    metric_family = MetricFamily(
        name=METRIC_TABLE,
        u_metrics_query=UMetricsQuery(table_name=METRIC_TABLE, date_col=DATE_COL),
        metric_join_unit_col=EXPT_UNIT_COL,
        metrics=metrics,
    )

    return AnalysisInfo(
        multi_expt_info=MultiExptInfo(
            expt_info_list=[expt1, expt2],
            merge_method="cross",
            expt_unit_col=EXPT_UNIT_COL,
        ),
        metric_info_list=[MetricInfo(metric_family=metric_family, metrics=metrics)],
        start_date=START_DATE,
        end_date=END_DATE,
    )


@pytest.fixture
def duckdb_cursor_with_expt(sim_data):
    """DuckDB cursor with both the metric table and the two per-experiment tables registered."""
    sim, metric_df = sim_data
    expt1_df = sim.expt_df[["id", "variant_1"]].rename(columns={"variant_1": "variant"})
    expt2_df = sim.expt_df[["id", "variant_2"]].rename(columns={"variant_2": "variant"})

    cursor = DuckDBCursor(max_retries=1)
    for table_name, df in [
        (METRIC_TABLE, metric_df),
        ("sim_expt1", expt1_df),
        ("sim_expt2", expt2_df),
    ]:
        view = f"{table_name}_view"
        cursor._db_connection.register(view, df)
        cursor._db_connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {view}")
        cursor._db_connection.unregister(view)

    yield cursor
    cursor.close()


@pytest.fixture
def analysis_info_with_expt_queries():
    """AnalysisInfo where each ExptInfo carries a SQL query — get_mea_data will run fully."""
    expt1 = ExptInfo(
        name="expt1",
        start_date=START_DATE,
        end_date=END_DATE,
        query="SELECT id, variant FROM sim_expt1",
        expt_unit_col=EXPT_UNIT_COL,
    )
    expt2 = ExptInfo(
        name="expt2",
        start_date=START_DATE,
        end_date=END_DATE,
        query="SELECT id, variant FROM sim_expt2",
        expt_unit_col=EXPT_UNIT_COL,
    )

    metrics = [
        Metric(numerator=UMetric(col="metric1", agg="SUM", name="metric1"), name="metric1"),
        Metric(numerator=UMetric(col="metric2", agg="SUM", name="metric2"), name="metric2"),
    ]
    metric_family = MetricFamily(
        name=METRIC_TABLE,
        u_metrics_query=UMetricsQuery(table_name=METRIC_TABLE, date_col=DATE_COL),
        metric_join_unit_col=EXPT_UNIT_COL,
        metrics=metrics,
    )

    return AnalysisInfo(
        multi_expt_info=MultiExptInfo(
            expt_info_list=[expt1, expt2],
            merge_method="cross",
            expt_unit_col=EXPT_UNIT_COL,
        ),
        metric_info_list=[MetricInfo(metric_family=metric_family, metrics=metrics)],
    )


def test_run_mea_with_sim_and_local_cursor(sim_data, duckdb_cursor, analysis_info):
    """Runs run_mea with simulated data passed as dc (bypassing SQL expt join) and a local
    DuckDB cursor for the metric table. Verifies agg_metrics is computed and the
    full report is written to docs/static/test-results/mea-local."""
    sim, _ = sim_data
    io_param = IOParam(cursor=duckdb_cursor)

    # Pass the pre-joined sim df directly so run_mea skips SQL expt data fetching.
    dc = DataContainer(pandas_df=sim.expt_metric_df)

    mea_result = run_mea(
        io_param=io_param,
        analysis_info=analysis_info,
        method="simple",
        dc=dc,
    )

    assert mea_result is not None
    assert mea_result.agg_metrics is not None
    agg_df = mea_result.agg_metrics
    assert "metric_family" in agg_df.columns
    assert "metric" in agg_df.columns
    assert "numer" in agg_df.columns

    write_path = f"{SAVE_PATH}/run_mea_sim/"
    mea = MEA()
    mea_report = mea.publish(
        mea_result=mea_result,
        analysis_info=analysis_info,
        write_path=write_path,
        add_timestamp_to_path=False,
        html_file_name="mea_report.html",
        markdown_file_name="mea_report.md",
        end_user_report=False,
    )

    assert mea_report is not None
    assert mea_report["html_str"] != "", "First test report html_str should not be empty"


def test_run_mea_via_get_mea_data(sim_data, duckdb_cursor_with_expt, analysis_info_with_expt_queries):
    """Runs run_mea without passing dc — get_mea_data fetches and joins expt + metric data
    via SQL using the local DuckDB cursor. This exercises the full data pipeline."""
    io_param = IOParam(cursor=duckdb_cursor_with_expt)

    mea_result = run_mea(
        io_param=io_param,
        analysis_info=analysis_info_with_expt_queries,
        method="simple",
    )

    assert mea_result is not None
    assert mea_result.agg_metrics is not None
    agg_df = mea_result.agg_metrics
    assert "metric_family" in agg_df.columns
    assert "metric" in agg_df.columns
    assert "numer" in agg_df.columns

    write_path = f"{SAVE_PATH}/run_mea_sim_via_get_mea_data/"
    mea = MEA()
    mea_report = mea.publish(
        mea_result=mea_result,
        analysis_info=analysis_info_with_expt_queries,
        write_path=write_path,
        add_timestamp_to_path=False,
        html_file_name="mea_report.html",
        markdown_file_name="mea_report.md",
        end_user_report=False,
    )

    assert mea_report is not None
    assert mea_report["html_str"] != ""


def test_run_mea_ratio_metric(duckdb_cursor_with_expt):
    """Runs run_mea with a single ratio metric (metric1 / metric2) via the full SQL pipeline."""
    ratio_metric = Metric(
        numerator=UMetric(col="metric1", agg="SUM", name="metric1"),
        numerator_agg="SUM",
        denominator=UMetric(col="metric2", agg="SUM", name="metric2"),
        denominator_agg="SUM",
        name="metric1_per_metric2",
    )
    metric_family = MetricFamily(
        name=METRIC_TABLE,
        u_metrics_query=UMetricsQuery(table_name=METRIC_TABLE, date_col=DATE_COL),
        metric_join_unit_col=EXPT_UNIT_COL,
        metrics=[ratio_metric],
    )
    expt1 = ExptInfo(
        name="expt1",
        start_date=START_DATE,
        end_date=END_DATE,
        query="SELECT id, variant FROM sim_expt1",
        expt_unit_col=EXPT_UNIT_COL,
    )
    expt2 = ExptInfo(
        name="expt2",
        start_date=START_DATE,
        end_date=END_DATE,
        query="SELECT id, variant FROM sim_expt2",
        expt_unit_col=EXPT_UNIT_COL,
    )
    ratio_analysis_info = AnalysisInfo(
        multi_expt_info=MultiExptInfo(
            expt_info_list=[expt1, expt2],
            merge_method="cross",
            expt_unit_col=EXPT_UNIT_COL,
        ),
        metric_info_list=[MetricInfo(metric_family=metric_family, metrics=[ratio_metric])],
    )

    mea_result = run_mea(
        io_param=IOParam(cursor=duckdb_cursor_with_expt),
        analysis_info=ratio_analysis_info,
        method="simple",
    )

    assert mea_result is not None
    assert mea_result.agg_metrics is not None
    agg_df = mea_result.agg_metrics
    assert "metric_family" in agg_df.columns
    assert "metric" in agg_df.columns
    assert "numer" in agg_df.columns
    assert "denom" in agg_df.columns
    assert "metric1_per_metric2" in agg_df["metric"].values

    write_path = f"{SAVE_PATH}/run_mea_ratio_metric/"
    mea = MEA()
    mea_report = mea.publish(
        mea_result=mea_result,
        analysis_info=ratio_analysis_info,
        write_path=write_path,
        add_timestamp_to_path=False,
        html_file_name="mea_report.html",
        end_user_report=False,
    )

    assert mea_report is not None
    assert mea_report["html_str"] != ""


def test_run_mea_mixed_metrics_agg_table(sim_data, duckdb_cursor_with_expt):
    """Runs run_mea with two simple metrics and one ratio metric in a single MetricInfo,
    then validates that agg_metrics is a clean long-format table with exactly
    one row per metric and correct numer / denom / value entries."""
    _, metric_df = sim_data

    # Build three metrics sharing the same metric family / table.
    metric1 = Metric(
        numerator=UMetric(col="metric1", agg="SUM", name="metric1"),
        name="metric1",
    )
    metric2 = Metric(
        numerator=UMetric(col="metric2", agg="SUM", name="metric2"),
        name="metric2",
    )
    ratio_metric = Metric(
        numerator=UMetric(col="metric1", agg="SUM", name="metric1"),
        numerator_agg="SUM",
        denominator=UMetric(col="metric2", agg="SUM", name="metric2"),
        denominator_agg="SUM",
        name="metric1_per_metric2",
    )
    metrics = [metric1, metric2, ratio_metric]
    metric_family = MetricFamily(
        name=METRIC_TABLE,
        u_metrics_query=UMetricsQuery(table_name=METRIC_TABLE, date_col=DATE_COL),
        metric_join_unit_col=EXPT_UNIT_COL,
        metrics=metrics,
    )
    expt1 = ExptInfo(
        name="expt1",
        start_date=START_DATE,
        end_date=END_DATE,
        query="SELECT id, variant FROM sim_expt1",
        expt_unit_col=EXPT_UNIT_COL,
    )
    expt2 = ExptInfo(
        name="expt2",
        start_date=START_DATE,
        end_date=END_DATE,
        query="SELECT id, variant FROM sim_expt2",
        expt_unit_col=EXPT_UNIT_COL,
    )
    mixed_analysis_info = AnalysisInfo(
        multi_expt_info=MultiExptInfo(
            expt_info_list=[expt1, expt2],
            merge_method="cross",
            expt_unit_col=EXPT_UNIT_COL,
        ),
        metric_info_list=[MetricInfo(metric_family=metric_family, metrics=metrics)],
    )

    mea_result = run_mea(
        io_param=IOParam(cursor=duckdb_cursor_with_expt),
        analysis_info=mixed_analysis_info,
        method="simple",
    )

    assert mea_result is not None
    agg_df = mea_result.agg_metrics
    assert agg_df is not None

    # Schema check.
    for col in ("metric_family", "metric", "numer", "denom", "value", "sample_count"):
        assert col in agg_df.columns, f"Expected column '{col}' missing from agg_metrics"

    # One row per metric.
    assert len(agg_df) == 3
    assert set(agg_df["metric"].values) == {"metric1", "metric2", "metric1_per_metric2"}

    # Simple metrics have no denom.
    for metric_name in ("metric1", "metric2"):
        row = agg_df.loc[agg_df["metric"] == metric_name].iloc[0]
        assert row["denom"] is None or (isinstance(row["denom"], float) and np.isnan(row["denom"])), f"Simple metric '{metric_name}' should have NaN denom"

    # Ratio metric has a denom and value == numer / denom.
    ratio_row = agg_df.loc[agg_df["metric"] == "metric1_per_metric2"].iloc[0]
    assert not np.isnan(ratio_row["denom"]), "Ratio metric should have a non-NaN denom"
    assert np.isclose(ratio_row["value"], ratio_row["numer"] / ratio_row["denom"])

    # Numer values must match raw sums from the full metric_df (no experiment join).
    expected_numer_m1 = metric_df["metric1"].sum()
    expected_numer_m2 = metric_df["metric2"].sum()
    assert np.isclose(agg_df.loc[agg_df["metric"] == "metric1", "numer"].iloc[0], expected_numer_m1)
    assert np.isclose(agg_df.loc[agg_df["metric"] == "metric2", "numer"].iloc[0], expected_numer_m2)
    assert np.isclose(agg_df.loc[agg_df["metric"] == "metric1_per_metric2", "numer"].iloc[0], expected_numer_m1)
    assert np.isclose(agg_df.loc[agg_df["metric"] == "metric1_per_metric2", "denom"].iloc[0], expected_numer_m2)

    write_path = f"{SAVE_PATH}/run_mea_mixed_metrics/"
    mea = MEA()
    mea_report = mea.publish(
        mea_result=mea_result,
        analysis_info=mixed_analysis_info,
        write_path=write_path,
        add_timestamp_to_path=False,
        html_file_name="mea_report.html",
        end_user_report=False,
    )
    assert mea_report is not None
    assert mea_report["html_str"] != ""

# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
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
Combined query-structure and live DuckDB tests for SwiAggMetricsQueries.

Dataset: 10 units (user_id / member_id 1–10).
  user_metrics:  sessions  = user_id * 10
  ratio_events:  clicks    = user_id * 10,  eligible_days = 2 (per user)

Experiment assignments:
  expt1: member_ids {1, 2, 3, 4}
  expt2: member_ids {3, 4, 5, 6}
  impacted (UNION ALL + DISTINCT): {1, 2, 3, 4, 5, 6}  → 6 units
  complement:                       {7, 8, 9, 10}       → 4 units

Expected values — avg_sessions:
  raw        = mean(10..100)  = 55.0   (n=10)
  impacted   = mean(10..60)   = 35.0   (n=6)
  complement = mean(70..100)  = 85.0   (n=4)

Expected values — clicks_per_day (SUM clicks / SUM eligible_days):
  raw        = 550 / 20 = 27.5   (n=10)
  impacted   = 210 / 12 = 17.5   (n=6)
  complement = 340 /  8 = 42.5   (n=4)

Note: expt_unit_col="member_id" (assignment) ≠ metric_join_unit_col="user_id" (metrics).
"""

import pandas as pd
import pytest
from abvelocity.core.get_data.agg_metrics_query import get_agg_metrics_query_from_info
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.get_data.impacted_units_query import ImpactedUnitsQuery
from abvelocity.core.get_data.swi_agg_metrics_queries import SEGMENT_COMPLEMENT, SEGMENT_IMPACTED, SEGMENT_RAW, SwiAggMetricsQueries
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal

START_DATE = "2024-01-01"
END_DATE = "2024-01-31"

AVG_SESSIONS_METRIC = Metric(
    numerator=UMetric(col="sessions", agg="SUM", fill_na=0, name="sessions"),
    numerator_agg="AVG",
    name="avg_sessions",
)

CLICKS_PER_DAY_METRIC = Metric(
    numerator=UMetric(col="clicks", agg="SUM", fill_na=0, name="clicks"),
    numerator_agg="SUM",
    denominator=UMetric(col="eligible_days", agg="SUM", fill_na=0, name="eligible_days"),
    denominator_agg="SUM",
    name="clicks_per_day",
)

USER_METRICS_FAMILY = MetricFamily(
    name="user_metrics",
    u_metrics_query=UMetricsQuery(table_name="user_metrics", date_col="event_date"),
    metric_join_unit_col="user_id",
)

RATIO_METRICS_FAMILY = MetricFamily(
    name="ratio_metrics",
    u_metrics_query=UMetricsQuery(table_name="ratio_events", date_col="event_date"),
    metric_join_unit_col="user_id",
)

USER_METRIC_INFO = MetricInfo(metric_family=USER_METRICS_FAMILY, metrics=[AVG_SESSIONS_METRIC])
RATIO_METRIC_INFO = MetricInfo(metric_family=RATIO_METRICS_FAMILY, metrics=[CLICKS_PER_DAY_METRIC])


@pytest.fixture
def duckdb_cursor():
    cursor = DuckDBCursor(max_retries=1)

    user_metrics_df = pd.DataFrame(
        {
            "user_id": range(1, 11),
            "sessions": [uid * 10 for uid in range(1, 11)],
            "event_date": ["2024-01-15"] * 10,
        }
    )
    ratio_events_df = pd.DataFrame(
        {
            "user_id": range(1, 11),
            "clicks": [uid * 10 for uid in range(1, 11)],
            "eligible_days": [2] * 10,
            "event_date": ["2024-01-15"] * 10,
        }
    )
    expt1_df = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4],
            "variant": ["enabled", "enabled", "control", "control"],
        }
    )
    expt2_df = pd.DataFrame(
        {
            "member_id": [3, 4, 5, 6],
            "variant": ["enabled", "enabled", "control", "control"],
        }
    )

    for df, table in [
        (user_metrics_df, "user_metrics"),
        (ratio_events_df, "ratio_events"),
        (expt1_df, "expt1_asgmt"),
        (expt2_df, "expt2_asgmt"),
    ]:
        cursor._db_connection.register(f"{table}_view", df)
        cursor._db_connection.execute(f"CREATE TABLE {table} AS SELECT * FROM {table}_view")
        cursor._db_connection.unregister(f"{table}_view")

    yield cursor
    cursor.close()


@pytest.fixture
def two_expt_analysis_info():
    expt1 = ExptInfo(expt_unit_col="member_id", query="SELECT member_id, variant FROM expt1_asgmt")
    expt2 = ExptInfo(expt_unit_col="member_id", query="SELECT member_id, variant FROM expt2_asgmt")
    return AnalysisInfo(
        multi_expt_info=MultiExptInfo(expt_info_list=[expt1, expt2]),
        metric_info_list=[USER_METRIC_INFO, RATIO_METRIC_INFO],
        start_date=START_DATE,
        end_date=END_DATE,
    )


def test_construct_stores_queries_and_df_has_correct_structure(duckdb_cursor, two_expt_analysis_info):
    swi = SwiAggMetricsQueries(two_expt_analysis_info)
    assert swi.queries is None

    df = swi.get_pandas_df(duckdb_cursor)

    # queries dict: two families, each with three segments
    assert set(swi.queries.keys()) == {"user_metrics", "ratio_metrics"}
    for family_queries in swi.queries.values():
        assert set(family_queries.keys()) == {SEGMENT_RAW, SEGMENT_IMPACTED, SEGMENT_COMPLEMENT}

    # normalized df: 6 rows (2 families × 3 segments), correct columns and label sets
    assert len(df) == 6
    assert {
        "metric_family",
        "metric",
        "numer",
        "denom",
        "value",
        "sample_count",
        "segment",
    }.issubset(df.columns)
    assert set(df["segment"].unique()) == {SEGMENT_RAW, SEGMENT_IMPACTED, SEGMENT_COMPLEMENT}
    assert set(df["metric_family"].unique()) == {"user_metrics", "ratio_metrics"}
    assert set(df["metric"].unique()) == {"avg_sessions", "clicks_per_day"}


def test_raw_segment(duckdb_cursor, two_expt_analysis_info):
    swi = SwiAggMetricsQueries(two_expt_analysis_info)
    df = swi.get_pandas_df(duckdb_cursor)

    # SQL: raw query has no population filter
    assert_query_is_equal(
        swi.queries["user_metrics"][SEGMENT_RAW],
        get_agg_metrics_query_from_info(USER_METRIC_INFO, START_DATE, END_DATE),
    )
    assert_query_is_equal(
        swi.queries["ratio_metrics"][SEGMENT_RAW],
        get_agg_metrics_query_from_info(RATIO_METRIC_INFO, START_DATE, END_DATE),
    )

    # DuckDB: raw segment — both metrics, all 10 units
    pd.testing.assert_frame_equal(
        df[df["segment"] == SEGMENT_RAW].sort_values("metric_family").reset_index(drop=True),
        pd.DataFrame(
            {
                "metric_family": ["ratio_metrics", "user_metrics"],
                "metric": ["clicks_per_day", "avg_sessions"],
                "numer": [550.0, 55.0],
                "denom": [20.0, None],
                "value": [27.5, 55.0],
                "sample_count": [10, 10],
                "segment": [SEGMENT_RAW, SEGMENT_RAW],
            }
        ),
        check_dtype=False,
    )


def test_impacted_and_complement_segments_single_experiment(duckdb_cursor):
    """Single experiment (expt1 only): impacted = {1,2,3,4}, complement = {5..10}."""
    expt1 = ExptInfo(expt_unit_col="member_id", query="SELECT member_id, variant FROM expt1_asgmt")
    multi_expt_info = MultiExptInfo(expt_info_list=[expt1])
    analysis_info = AnalysisInfo(
        multi_expt_info=multi_expt_info,
        metric_info_list=[USER_METRIC_INFO],
        start_date=START_DATE,
        end_date=END_DATE,
    )

    swi = SwiAggMetricsQueries(analysis_info)
    df = swi.get_pandas_df(duckdb_cursor)

    # SQL: IN / NOT IN with single-experiment subquery
    impacted_subq = ImpactedUnitsQuery(multi_expt_info).construct()
    assert_query_is_equal(
        swi.queries["user_metrics"][SEGMENT_IMPACTED],
        f"""
            SELECT
                AVG(sessions) AS avg_sessions_numer,
                AVG(sessions) AS avg_sessions,
                COUNT(*) AS sample_count
            FROM (
                SELECT user_id, SUM(sessions) AS sessions
                FROM user_metrics
                WHERE event_date BETWEEN '{START_DATE}' AND '{END_DATE}'
                    AND user_id IN ({impacted_subq})
                GROUP BY 1
            ) AS u_metrics
        """,
    )
    assert_query_is_equal(
        swi.queries["user_metrics"][SEGMENT_COMPLEMENT],
        f"""
            SELECT
                AVG(sessions) AS avg_sessions_numer,
                AVG(sessions) AS avg_sessions,
                COUNT(*) AS sample_count
            FROM (
                SELECT user_id, SUM(sessions) AS sessions
                FROM user_metrics
                WHERE event_date BETWEEN '{START_DATE}' AND '{END_DATE}'
                    AND user_id NOT IN ({impacted_subq})
                GROUP BY 1
            ) AS u_metrics
        """,
    )

    # DuckDB: impacted = mean(10..40) = 25.0 (n=4), complement = mean(50..100) = 75.0 (n=6)
    pd.testing.assert_frame_equal(
        df[df["segment"] == SEGMENT_IMPACTED].reset_index(drop=True),
        pd.DataFrame(
            {
                "metric_family": ["user_metrics"],
                "metric": ["avg_sessions"],
                "numer": [25.0],
                "denom": [None],
                "value": [25.0],
                "sample_count": [4],
                "segment": [SEGMENT_IMPACTED],
            }
        ),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        df[df["segment"] == SEGMENT_COMPLEMENT].reset_index(drop=True),
        pd.DataFrame(
            {
                "metric_family": ["user_metrics"],
                "metric": ["avg_sessions"],
                "numer": [75.0],
                "denom": [None],
                "value": [75.0],
                "sample_count": [6],
                "segment": [SEGMENT_COMPLEMENT],
            }
        ),
        check_dtype=False,
    )


def test_impacted_and_complement_segments_two_experiments(duckdb_cursor, two_expt_analysis_info):
    """Two overlapping experiments and two metric families — assert the full normalized df."""
    swi = SwiAggMetricsQueries(two_expt_analysis_info)
    df = swi.get_pandas_df(duckdb_cursor)

    # SQL: UNION ALL in IN subquery; user_id (metric) vs member_id (expt) handled transparently
    impacted_subq = ImpactedUnitsQuery(two_expt_analysis_info.multi_expt_info).construct()
    assert_query_is_equal(
        swi.queries["user_metrics"][SEGMENT_IMPACTED],
        f"""
            SELECT
                AVG(sessions) AS avg_sessions_numer,
                AVG(sessions) AS avg_sessions,
                COUNT(*) AS sample_count
            FROM (
                SELECT user_id, SUM(sessions) AS sessions
                FROM user_metrics
                WHERE event_date BETWEEN '{START_DATE}' AND '{END_DATE}'
                    AND user_id IN ({impacted_subq})
                GROUP BY 1
            ) AS u_metrics
        """,
    )
    assert "UNION ALL" in impacted_subq
    assert "user_id IN" in swi.queries["user_metrics"][SEGMENT_IMPACTED].replace("\n", " ")
    assert "member_id" in swi.queries["user_metrics"][SEGMENT_IMPACTED]

    # DuckDB: full normalized df — 6 rows sorted by (segment, metric_family)
    pd.testing.assert_frame_equal(
        df.sort_values(["segment", "metric_family"]).reset_index(drop=True),
        pd.DataFrame(
            {
                "metric_family": [
                    "ratio_metrics",
                    "user_metrics",  # complement
                    "ratio_metrics",
                    "user_metrics",  # impacted
                    "ratio_metrics",
                    "user_metrics",  # raw
                ],
                "metric": [
                    "clicks_per_day",
                    "avg_sessions",
                    "clicks_per_day",
                    "avg_sessions",
                    "clicks_per_day",
                    "avg_sessions",
                ],
                "numer": [340.0, 85.0, 210.0, 35.0, 550.0, 55.0],
                "denom": [8.0, None, 12.0, None, 20.0, None],
                "value": [42.5, 85.0, 17.5, 35.0, 27.5, 55.0],
                "sample_count": [4, 4, 6, 6, 10, 10],
                "segment": [
                    SEGMENT_COMPLEMENT,
                    SEGMENT_COMPLEMENT,
                    SEGMENT_IMPACTED,
                    SEGMENT_IMPACTED,
                    SEGMENT_RAW,
                    SEGMENT_RAW,
                ],
            }
        ),
        check_dtype=False,
    )


def test_get_pandas_df_dict(duckdb_cursor, two_expt_analysis_info):
    """get_pandas_df_dict returns wide-format raw dfs keyed by family and segment."""
    swi = SwiAggMetricsQueries(two_expt_analysis_info)
    df_dict = swi.get_pandas_df_dict(duckdb_cursor)

    assert set(df_dict.keys()) == {"user_metrics", "ratio_metrics"}
    for family_dfs in df_dict.values():
        assert set(family_dfs.keys()) == {SEGMENT_RAW, SEGMENT_IMPACTED, SEGMENT_COMPLEMENT}

    # user_metrics wide format — columns: avg_sessions_numer, avg_sessions, sample_count
    pd.testing.assert_frame_equal(
        df_dict["user_metrics"][SEGMENT_RAW].reset_index(drop=True),
        pd.DataFrame({"avg_sessions_numer": [55.0], "avg_sessions": [55.0], "sample_count": [10]}),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        df_dict["user_metrics"][SEGMENT_IMPACTED].reset_index(drop=True),
        pd.DataFrame({"avg_sessions_numer": [35.0], "avg_sessions": [35.0], "sample_count": [6]}),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        df_dict["user_metrics"][SEGMENT_COMPLEMENT].reset_index(drop=True),
        pd.DataFrame({"avg_sessions_numer": [85.0], "avg_sessions": [85.0], "sample_count": [4]}),
        check_dtype=False,
    )

    # ratio_metrics wide format — columns: clicks_per_day_numer, clicks_per_day_denom, clicks_per_day, sample_count
    pd.testing.assert_frame_equal(
        df_dict["ratio_metrics"][SEGMENT_RAW].reset_index(drop=True),
        pd.DataFrame(
            {
                "clicks_per_day_numer": [550.0],
                "clicks_per_day_denom": [20.0],
                "clicks_per_day": [27.5],
                "sample_count": [10],
            }
        ),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        df_dict["ratio_metrics"][SEGMENT_IMPACTED].reset_index(drop=True),
        pd.DataFrame(
            {
                "clicks_per_day_numer": [210.0],
                "clicks_per_day_denom": [12.0],
                "clicks_per_day": [17.5],
                "sample_count": [6],
            }
        ),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        df_dict["ratio_metrics"][SEGMENT_COMPLEMENT].reset_index(drop=True),
        pd.DataFrame(
            {
                "clicks_per_day_numer": [340.0],
                "clicks_per_day_denom": [8.0],
                "clicks_per_day": [42.5],
                "sample_count": [4],
            }
        ),
        check_dtype=False,
    )


def test_sample_counts_partition_raw(duckdb_cursor, two_expt_analysis_info):
    """For each metric family, impacted + complement sample counts sum to raw."""
    df = SwiAggMetricsQueries(two_expt_analysis_info).get_pandas_df(duckdb_cursor)
    for family in ("user_metrics", "ratio_metrics"):
        fdf = df[df["metric_family"] == family]
        raw_n = fdf[fdf["segment"] == SEGMENT_RAW].iloc[0]["sample_count"]
        imp_n = fdf[fdf["segment"] == SEGMENT_IMPACTED].iloc[0]["sample_count"]
        comp_n = fdf[fdf["segment"] == SEGMENT_COMPLEMENT].iloc[0]["sample_count"]
        assert imp_n + comp_n == raw_n


def test_multiple_metric_families_each_get_three_segments(duckdb_cursor):
    """Two metric families using the same table → both in queries dict and DataFrame."""
    signup_family = MetricFamily(
        name="signup_metrics",
        u_metrics_query=UMetricsQuery(table_name="user_metrics", date_col="event_date"),
        metric_join_unit_col="user_id",
    )
    expt1 = ExptInfo(expt_unit_col="member_id", query="SELECT member_id, variant FROM expt1_asgmt")
    analysis_info = AnalysisInfo(
        multi_expt_info=MultiExptInfo(expt_info_list=[expt1]),
        metric_info_list=[
            USER_METRIC_INFO,
            MetricInfo(metric_family=signup_family, metrics=[AVG_SESSIONS_METRIC]),
        ],
        start_date=START_DATE,
        end_date=END_DATE,
    )

    swi = SwiAggMetricsQueries(analysis_info)
    df = swi.get_pandas_df(duckdb_cursor)

    assert set(swi.queries.keys()) == {"user_metrics", "signup_metrics"}
    for family_queries in swi.queries.values():
        assert set(family_queries.keys()) == {SEGMENT_RAW, SEGMENT_IMPACTED, SEGMENT_COMPLEMENT}
    assert set(df["metric_family"].unique()) == {"user_metrics", "signup_metrics"}
    assert len(df) == 6  # 3 segments × 2 families × 1 metric each


def test_empty_metric_info_list_raises():
    analysis_info = AnalysisInfo(
        multi_expt_info=MultiExptInfo(expt_info_list=[ExptInfo(expt_unit_col="member_id", query="SELECT member_id FROM t")]),
        metric_info_list=[],
        start_date=START_DATE,
        end_date=END_DATE,
    )
    with pytest.raises(ValueError, match="metric_info_list"):
        SwiAggMetricsQueries(analysis_info).construct()

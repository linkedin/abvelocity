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

from copy import deepcopy
from typing import Dict

from abvelocity.core.journey.event.gen_event_query import EventTable, MultiEventTable
from abvelocity.core.journey.seq.seq import Seq
from abvelocity.core.journey.seq.seq_info import FULLY_DEDUPED, UNDEDUPED, SeqInfo
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.join_query import JoinQuery
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal


class JourneySeq(Seq):
    """Specific Seq implementation inheriting methods from the imported Seq class."""

    def gen_event_queries(self) -> Dict[str, str]:
        # Assumes gen_event_queries_via_multi_event_tables exists on the imported Seq class
        return self.gen_event_queries_via_multi_event_tables(multi_event_tables=self.multi_event_table)


JOURNEY_EVENTS = MultiEventTable(
    event_tables=[
        EventTable(table_name="ExternalData.PageView", event_label="page_view"),
        EventTable(table_name="ExternalData.PageClick", event_label="page_click"),
    ],
    common_info=EventTable(),
)
JOURNEY_SEQ_INFO_LIST = [
    SeqInfo(deduping_method=UNDEDUPED, max_seq_index=12),
    SeqInfo(deduping_method=FULLY_DEDUPED, max_seq_index=9),
]


POST_JOINS = [
    JoinQuery(
        right_table="user_table",
        join_type="LEFT",
        on=[("user", "session")],
        select_right_columns=[
            "country",
            "age",
        ],
        right_conditions=None,
        right_date_col="date",
    ),
]


# CREATE THE SINGLE GLOBAL INSTANCE CONSTANT
JOURNEY_SEQ = JourneySeq(
    create_table_prefix="rezas",
    start_date="2025-05-21-00",
    end_date="2025-05-28-00",
    # Use deepcopy to ensure the class's internal pristine copy is isolated
    seq_info_list=deepcopy(JOURNEY_SEQ_INFO_LIST),
    multi_event_table=JOURNEY_EVENTS,
    io_param=IOParam(cursor=None, print_to_html=None, save_path="", file_name_suffix=""),
    materialization=None,
    # Minimal arguments for Seq constructor
    partition_by_cols=["user"],
    conditions=[],
    post_joins=POST_JOINS,
)


def test_journey_seq_with_reset(capfd):
    """
    Tests idempotency by explicitly calling seq.reset_seq_info_list() on the
    shared JOURNEY_SEQ instance.
    """

    seq = JOURNEY_SEQ
    base_prefix = seq.create_table_prefix

    # FIRST RUN EXECUTION (Mutates seq.seq_info_list)
    seq.gen_event_queries()
    seq.gen_seq_queries()
    seq.gen_join_queries()

    # FIRST RUN ASSERTIONS
    expected_base_name = f"{base_prefix}_{UNDEDUPED}_seq"
    expected_final_name = f"{expected_base_name}_joined"

    final_table_name_run1 = seq.seq_info_list[0].output_table_name
    assert final_table_name_run1 == expected_final_name

    # SECOND RUN SETUP & EXECUTION
    seq.reset_seq_info_list()  # Fix applied here

    seq.gen_event_queries()
    seq.gen_seq_queries()
    seq.gen_join_queries()

    capfd.readouterr()

    # SECOND RUN ASSERTIONS
    final_table_name_run2 = seq.seq_info_list[0].output_table_name
    assert final_table_name_run2 == expected_final_name


def test_journey_seq_with_gen_all(capfd):
    """
    Tests idempotency using the gen_all_queries wrapper on the shared JOURNEY_SEQ instance.
    (Relies on gen_all_queries internally calling reset_seq_info_list).
    """

    seq = JOURNEY_SEQ
    base_prefix = seq.create_table_prefix
    expected_final_name = f"{base_prefix}_{UNDEDUPED}_seq_joined"

    # FIRST RUN EXECUTION
    seq.gen_all_queries()

    # FIRST RUN ASSERTIONS
    final_table_name_run1 = seq.seq_info_list[0].output_table_name
    assert final_table_name_run1 == expected_final_name

    # SECOND RUN EXECUTION
    seq.gen_all_queries()

    capfd.readouterr()

    # SECOND RUN ASSERTIONS
    final_table_name_run2 = seq.seq_info_list[0].output_table_name
    assert final_table_name_run2 == expected_final_name


def test_aug_agg_time_pushes_conditions_to_inner_query():
    """Test that aug_agg_time injects conditions inside the TransformTimeQuery subquery."""
    seq = deepcopy(JOURNEY_SEQ)
    seq.time_col_format = "unix_ms"
    seq.time_unit = "day"
    seq.gen_all_queries()

    table_name = seq.get_seq_info(FULLY_DEDUPED).output_table_name
    conditions = ["etl_date >= '2026-03-01-00'", "country_code = 'US'"]

    result_table, _, group_by_cols_w_time = seq.aug_agg_time(table_name=table_name, group_by_cols=[], conditions=conditions)

    expected_subquery = (
        "(SELECT *,\n"
        "       DATE_TRUNC('DAY', FROM_UNIXTIME(seq_start_time / 1000.0)) AS agg_time\n"
        "FROM (rezas_fully_deduped_seq_joined)\n"
        "WHERE etl_date >= '2026-03-01-00' AND country_code = 'US')"
    )
    assert result_table == expected_subquery
    assert group_by_cols_w_time == ["agg_time"]


def test_aug_agg_time_no_conditions_no_where():
    """Test that aug_agg_time without conditions produces no WHERE clause."""
    seq = deepcopy(JOURNEY_SEQ)
    seq.time_col_format = "unix_ms"
    seq.time_unit = "day"
    seq.gen_all_queries()

    table_name = seq.get_seq_info(FULLY_DEDUPED).output_table_name

    result_table, _, group_by_cols_w_time = seq.aug_agg_time(table_name=table_name, group_by_cols=[])

    expected_subquery = "(SELECT *,\n" "       DATE_TRUNC('DAY', FROM_UNIXTIME(seq_start_time / 1000.0)) AS agg_time\n" "FROM (rezas_fully_deduped_seq_joined))"
    assert result_table == expected_subquery
    assert group_by_cols_w_time == ["agg_time"]


def test_gen_seq_summary_query_pushes_conditions_with_group_by_time():
    """Test that gen_seq_summary_query pushes conditions to the inner subquery
    when group_by_time=True, so Trino can push predicates to the table scan."""
    seq = deepcopy(JOURNEY_SEQ)
    seq.time_col_format = "unix_ms"
    seq.time_unit = "day"
    seq.delta_time_unit = "minute"
    seq.gen_all_queries()

    conditions = ["etl_date >= '2026-03-01-00'", "pm_pagekey = 'desktop'"]

    calc_res = seq.gen_seq_summary_query(
        deduping_method=FULLY_DEDUPED,
        count_distinct_col="memberid",
        group_by_time=True,
        conditions=conditions,
    )

    expected_query = """
        SELECT agg_time,
               COUNT(DISTINCT memberid) AS seq_count,
               AVG(seq_length) AS avg_seq_length,
               AVG(DATE_DIFF('MINUTE', FROM_UNIXTIME(seq_start_time / 1000.0),
                   FROM_UNIXTIME(seq_end_time / 1000.0))) AS avg_seq_time
        FROM (SELECT *,
                     DATE_TRUNC('DAY', FROM_UNIXTIME(seq_start_time / 1000.0)) AS agg_time
              FROM (rezas_fully_deduped_seq_joined)
              WHERE etl_date >= '2026-03-01-00' AND pm_pagekey = 'desktop')
        GROUP BY agg_time
    """
    assert_query_is_equal(calc_res.query, expected_query)


def test_gen_seq_summary_query_conditions_at_outer_when_no_group_by_time():
    """Test that when group_by_time=False, conditions stay in the outer WHERE clause."""
    seq = deepcopy(JOURNEY_SEQ)
    seq.time_col_format = "unix_ms"
    seq.time_unit = "day"
    seq.delta_time_unit = "minute"
    seq.gen_all_queries()

    conditions = ["etl_date >= '2026-03-01-00'"]

    calc_res = seq.gen_seq_summary_query(
        deduping_method=FULLY_DEDUPED,
        count_distinct_col="memberid",
        group_by_time=False,
        conditions=conditions,
    )

    expected_query = """
        SELECT COUNT(DISTINCT memberid) AS seq_count,
               AVG(seq_length) AS avg_seq_length,
               AVG(DATE_DIFF('MINUTE', FROM_UNIXTIME(seq_start_time / 1000.0),
                   FROM_UNIXTIME(seq_end_time / 1000.0))) AS avg_seq_time
        FROM rezas_fully_deduped_seq_joined
        WHERE etl_date >= '2026-03-01-00'
    """
    assert_query_is_equal(calc_res.query, expected_query)


def test_calc_conversion_pushes_conditions_with_group_by_time():
    """Test that calc_conversion pushes conditions into the inner subquery
    when group_by_time=True, so Trino can push predicates to the table scan."""
    seq = deepcopy(JOURNEY_SEQ)
    seq.time_col_format = "unix_ms"
    seq.time_unit = "day"
    seq.io_param = None
    seq.gen_all_queries()

    conditions = ["etl_date >= '2026-03-01-00'", "country_code = 'US'"]

    calc_res = seq.calc_conversion(
        numerator_list=["page_view", "page_click"],
        denominator_list=["page_view"],
        require_all_numerator=True,
        count_distinct_col="user",
        conditions=conditions,
        group_by_cols=[],
        group_by_time=True,
        deduping_method=FULLY_DEDUPED,
    )

    expected_query = """
        SELECT agg_time,
               COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_seq,
                   ARRAY['page_view', 'page_click'])) = 2
                   THEN user ELSE NULL END) AS numer_count,
               COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_seq,
                   ARRAY['page_view'])) > 0
                   THEN user ELSE NULL END) AS denom_count,
               CAST(COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_seq,
                   ARRAY['page_view', 'page_click'])) = 2
                   THEN user ELSE NULL END) AS DOUBLE)
               / NULLIF(COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_seq,
                   ARRAY['page_view'])) > 0
                   THEN user ELSE NULL END), 0) AS conversion_rate
        FROM (SELECT *,
                     DATE_TRUNC('DAY', FROM_UNIXTIME(seq_start_time / 1000.0)) AS agg_time
              FROM (rezas_fully_deduped_seq_joined)
              WHERE etl_date >= '2026-03-01-00' AND country_code = 'US')
        GROUP BY agg_time
    """
    assert_query_is_equal(calc_res.query, expected_query)


def test_calc_conversion_conditions_at_outer_when_no_group_by_time():
    """Test that when group_by_time=False, conditions stay in the outer WHERE clause."""
    seq = deepcopy(JOURNEY_SEQ)
    seq.time_col_format = "unix_ms"
    seq.time_unit = "day"
    seq.io_param = None
    seq.gen_all_queries()

    conditions = ["etl_date >= '2026-03-01-00'"]

    calc_res = seq.calc_conversion(
        numerator_list=["page_view"],
        denominator_list=["page_view"],
        count_distinct_col="user",
        conditions=conditions,
        group_by_cols=[],
        group_by_time=False,
        deduping_method=FULLY_DEDUPED,
    )

    expected_query = """
        SELECT COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_seq,
                   ARRAY['page_view'])) > 0
                   THEN user ELSE NULL END) AS numer_count,
               COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_seq,
                   ARRAY['page_view'])) > 0
                   THEN user ELSE NULL END) AS denom_count,
               CAST(COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_seq,
                   ARRAY['page_view'])) > 0
                   THEN user ELSE NULL END) AS DOUBLE)
               / NULLIF(COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_seq,
                   ARRAY['page_view'])) > 0
                   THEN user ELSE NULL END), 0) AS conversion_rate
        FROM rezas_fully_deduped_seq_joined
        WHERE etl_date >= '2026-03-01-00'
    """
    assert_query_is_equal(calc_res.query, expected_query)

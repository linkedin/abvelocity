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

from abvelocity.core.journey.event.gen_event_query import EventTable, MultiEventTable
from abvelocity.core.journey.event.gen_event_tables_query import gen_event_tables_query, union_tables_query
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal


def test_gen_event_tables_query():
    """
    Tests `gen_event_tables_query` using MultiEventTable,
    ensuring common_info propagation correctly affects generated queries.
    """
    # These dates are passed to gen_event_tables_query, but will be overridden
    # by common_event_info's dates if EventTable.start_date/end_date are None.
    start_date_param = "2024-10-06-00"
    end_date_param = "2024-10-13-00"
    create_table_prefix = "u_test_schema.temp_foo"

    # Common info that should propagate to fields that are None in individual EventTables
    # These dates will be used because event1 and event2 don't specify them.
    common_start_date = "2024-01-01"
    common_end_date = "2024-01-31"

    common_event_info = EventTable(
        select_cols=["common_id", "common_timestamp"],
        conditions=["common_condition = TRUE"],
        date_col="common_date_col",  # This will now propagate as EventTable's default is None
        table_name="common_base_table",
        table_query="SELECT c.col1, c.col2 FROM common_view c",
        start_date=common_start_date,
        end_date=common_end_date,
    )

    event1 = EventTable(
        table_name="SPECIFIC_IMPRESSION_TABLE",
        event_label="impression",
        conditions=[
            "specific_impression_condition IS NOT NULL",
            "MOD(impression_id, 100) IN (1)",
        ],
    )
    event2 = EventTable(
        table_name="SPECIFIC_CLICK_TABLE",
        event_label="click",
        select_cols=["click_id", "click_time"],
        conditions=None,
        table_query="SELECT x, y FROM another_view",
    )

    multi_event_table = MultiEventTable(event_tables=[event1, event2], common_info=common_event_info)

    obtained_queries = gen_event_tables_query(
        multi_event_table=multi_event_table,
        start_date=start_date_param,
        end_date=end_date_param,
        create_table_prefix=create_table_prefix,
    )
    # Expected table names
    expected_table_names = [
        "u_test_schema.temp_foo_specific_impression_table_impression",
        "u_test_schema.temp_foo_specific_click_table_click",
    ]

    expected_queries = [
        # Expected query for event1 (has its own conditions, so common conditions NOT added)
        # date_col is now common_date_col
        f"""
        DROP TABLE IF EXISTS u_test_schema.temp_foo_specific_impression_table_impression;
        CREATE TABLE IF NOT EXISTS u_test_schema.temp_foo_specific_impression_table_impression AS
        SELECT
            common_id, common_timestamp,
            'impression' AS event
        FROM (SELECT c.col1, c.col2 FROM common_view c)
        WHERE TRUE
            AND common_date_col BETWEEN '{common_start_date}' AND '{common_end_date}'
            AND specific_impression_condition IS NOT NULL
            AND MOD(impression_id, 100) IN (1)
        """,
        # Expected query for event2 (mix of specific and common propagated values)
        # date_col is now common_date_col
        f"""
        DROP TABLE IF EXISTS u_test_schema.temp_foo_specific_click_table_click;
        CREATE TABLE IF NOT EXISTS u_test_schema.temp_foo_specific_click_table_click AS
        SELECT
            click_id, click_time,
            'click' AS event
        FROM (SELECT x, y FROM another_view)
        WHERE TRUE
            AND common_date_col BETWEEN '{common_start_date}' AND '{common_end_date}'
            AND common_condition = TRUE
        """,
    ]

    for obtained, exp_name, exp_query in zip(obtained_queries, expected_table_names, expected_queries):
        assert obtained.table_name == exp_name
        # Expected queries include CREATE TABLE IF NOT EXISTS, so pass flag
        assert_query_is_equal(obtained.gen_rebuild_query(if_not_exists=True), exp_query)


def test_union_tables_query():
    """
    Tests `union_tables_query` using MultiEventTable.
    Manually sets `output_table_name` for EventTable instances to simulate
    prior execution of `gen_event_tables_query`.
    """
    create_table_prefix = "user_activity"
    order_by = "timestamp DESC"

    # For union_tables_query, the EventTable instances *must* already have output_table_name set.
    # We simulate this here by setting them manually.
    event_tables = [
        EventTable(
            table_name="signup_events",
            event_label="signup",
            select_cols=["user_id", "timestamp"],
            output_table_name="user_activity_signup_events",
        ),
        EventTable(
            table_name="purchase_events",
            event_label="purchase",
            select_cols=["user_id", "timestamp"],
            output_table_name="user_activity_purchase_events",
        ),
        EventTable(
            table_name="login_events",
            event_label="login",
            select_cols=["user_id", "timestamp"],
            output_table_name="user_activity_login_events",
        ),
    ]

    # Create MultiEventTable (common_info is not strictly necessary for this test,
    # as its propagation effect would have already influenced output_table_name
    # if this were part of a larger workflow).
    multi_event_table = MultiEventTable(event_tables=event_tables)

    expected_query = """
        DROP TABLE IF EXISTS user_activity_union;
        CREATE TABLE IF NOT EXISTS user_activity_union AS
            SELECT *
            FROM user_activity_signup_events
        UNION ALL
            SELECT *
            FROM user_activity_purchase_events
        UNION ALL
            SELECT *
            FROM user_activity_login_events
        ORDER BY timestamp DESC
    """

    obtained_tq = union_tables_query(multi_event_table, create_table_prefix, order_by)
    # Expected union query includes IF NOT EXISTS
    assert_query_is_equal(obtained_tq.gen_rebuild_query(if_not_exists=True), expected_query)

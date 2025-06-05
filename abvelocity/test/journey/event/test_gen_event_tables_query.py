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


from abvelocity.journey.event.gen_event_query import EventTable
from abvelocity.journey.event.gen_event_tables_query import (
    gen_event_tables_query,
    union_tables_query,
)
from abvelocity.testing.assert_query_is_equal import assert_query_is_equal


def test_gen_event_tables_query():
    """Tests `gen_event_tables_query`"""
    start_date = "2024-10-06-00"
    end_date = "2024-10-13-00"
    create_table_prefix = "u_owner.temp_reza"

    event_tables = [
        EventTable(
            table_name="TRACKING.randomProductUpsellImpressionEvent",
            event_label="impression",
            date_col="datepartition",
            select_cols=["datepartition"],
            conditions=[
                "RandomProductfunnelcommonheader.referenceid IS NOT NULL",
                "MOD(header.memberid, 10000) IN (13)",
            ],
            output_table_name="u_owner.temp_reza_tracking_RandomProduct_upsell_impression_event",
        ),
        EventTable(
            table_name="TRACKING.randomProductUpsellClickEvent",
            event_label="click",
            date_col="datepartition",
            select_cols=["datepartition"],
            conditions=["header.memberid IS NOT NULL"],
            output_table_name="u_owner.temp_reza_tracking_RandomProduct_upsell_click_event",
        ),
    ]

    obtained_queries = gen_event_tables_query(
        event_tables=event_tables,
        start_date=start_date,
        end_date=end_date,
        create_table_prefix=create_table_prefix,
    )

    expected_queries = [
        """
        DROP TABLE IF EXISTS u_owner.temp_reza_tracking_RandomProduct_upsell_impression_event_impression;
        CREATE TABLE IF NOT EXISTS u_owner.temp_reza_tracking_RandomProduct_upsell_impression_event_impression AS
        SELECT
            datepartition,
            'impression' AS event
        FROM TRACKING.randomProductUpsellImpressionEvent
        WHERE TRUE
            AND datepartition BETWEEN '2024-10-06-00' AND '2024-10-13-00'
            AND RandomProductfunnelcommonheader.referenceid IS NOT NULL
            AND MOD(header.memberid, 10000) IN (13)
        """,
        """
        DROP TABLE IF EXISTS u_owner.temp_reza_tracking_RandomProduct_upsell_click_event_click;
        CREATE TABLE IF NOT EXISTS u_owner.temp_reza_tracking_RandomProduct_upsell_click_event_click AS
        SELECT
            datepartition,
            'click' AS event
        FROM TRACKING.randomProductUpsellClickEvent
        WHERE TRUE
            AND datepartition BETWEEN '2024-10-06-00' AND '2024-10-13-00'
            AND header.memberid IS NOT NULL
        """,
    ]

    for obtained, expected in zip(obtained_queries, expected_queries):
        assert_query_is_equal(obtained, expected)


def test_union_tables_query():
    # Test data setup with select_cols ["user_id", "timestamp"] but query should use SELECT *
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
    create_table_prefix = "user_activity"
    order_by = "timestamp DESC"

    # Expected query output with SELECT * (not columns listed explicitly)
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

    # Call the function with test data
    obtained_query = union_tables_query(event_tables, create_table_prefix, order_by)

    # Use the custom assertion to compare the queries
    assert_query_is_equal(obtained_query, expected_query)

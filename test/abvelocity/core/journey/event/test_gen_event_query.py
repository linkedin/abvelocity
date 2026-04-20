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

import pytest
from abvelocity.core.journey.event.gen_event_query import DATE_COL_DEFAULT, EventTable, MultiEventTable, convert_to_snake_case, gen_event_query
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal


def test_convert_to_snake_case():
    # Test case 1
    result = convert_to_snake_case("random_owner.unit_journey_raw_TRACKING.SomeTableWithData")
    assert result == "random_owner_unit_journey_raw_tracking_some_table_with_data"

    # Test case 2: already in snake case
    result = convert_to_snake_case("already_snake_case_string")
    assert result == "already_snake_case_string"

    # Test case 3: camel case
    result = convert_to_snake_case("CamelCaseExample")
    assert result == "camel_case_example"

    # Test case 4: only periods
    result = convert_to_snake_case("string.with.periods")
    assert result == "string_with_periods"

    # Test case 5: empty string
    result = convert_to_snake_case("")
    assert result == ""


def test_gen_event_query():
    """Tests `gen_event_query`"""
    start_date = "2024-10-06-00"
    end_date = "2024-10-13-00"
    create_table_prefix = "random_owner.temp_reza"

    event_table = EventTable(
        table_name="TRACKING.SomeInterestingEvent",
        event_label="impression",
        select_cols=[
            "datepartition",
            "header.unitid",
            "header.randomVisit",
            "header.time AS time",
            "requestHeader.pageKey",
            "header.pageInstance.pageId",
            "productheader.glueid",
        ],
        date_col="datepartition",
        conditions=[
            "productheader.glueid IS NOT NULL",
            "MOD(header.unitid, 10000) IN (13)",
        ],
    )

    obtained_tq = gen_event_query(
        event_table=event_table,
        start_date=start_date,
        end_date=end_date,
        create_table_prefix=create_table_prefix,
    )
    # Validate table name
    assert obtained_tq.table_name == "random_owner.temp_reza_tracking_some_interesting_event_impression"

    # Expected body-only query (no DROP/CREATE)
    expected_body_query = """
    SELECT
        datepartition,
        header.unitid,
        header.randomVisit,
        header.time AS time,
        requestHeader.pageKey,
        header.pageInstance.pageId,
        productheader.glueid,
        'impression' AS event
    FROM TRACKING.SomeInterestingEvent
    WHERE TRUE
        AND datepartition BETWEEN '2024-10-06-00' AND '2024-10-13-00'
        AND productheader.glueid IS NOT NULL
        AND MOD(header.unitid, 10000) IN (13)
    """

    assert_query_is_equal(obtained_tq.main_query, expected_body_query)


def test_multi_event_table_unique_labels_success():
    """Tests that MultiEventTable allows unique event_labels."""
    event1 = EventTable(event_label="login", table_name="users.logins")
    event2 = EventTable(event_label="signup", table_name="users.signups")
    event3 = EventTable(event_label="purchase", table_name="users.purchases")

    # Should not raise an error
    multi_event = MultiEventTable(event_tables=[event1, event2, event3])
    assert len(multi_event.event_tables) == 3


def test_multi_event_table_unique_labels_failure():
    """Tests that MultiEventTable raises ValueError for duplicate event_labels."""
    event1 = EventTable(event_label="view", table_name="data.page_views")
    event2 = EventTable(event_label="click", table_name="data.clicks")
    event3 = EventTable(event_label="view", table_name="data.old_views")  # Duplicate label

    with pytest.raises(ValueError) as excinfo:
        MultiEventTable(event_tables=[event1, event2, event3])

    assert "All event_labels within 'event_tables' must be unique." in str(excinfo.value)
    assert "'view'" in str(excinfo.value)  # Check if the duplicate label is mentioned


def test_multi_event_table_propagate_common_info():
    """Tests the propagation of common_info fields to event_tables."""
    common_info = EventTable(
        select_cols=["id", "timestamp"],
        conditions=["is_active = TRUE"],
        date_col="event_date",
        table_name="default_table",
        table_query="SELECT * FROM default_source WHERE type = 'data'",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    event1 = EventTable(event_label="eventA")
    event2 = EventTable(event_label="eventB", table_name="specific_table", conditions=["specific = 'yes'"])
    event3 = EventTable(event_label="eventC", select_cols=["custom_col"], date_col="custom_date_col")

    multi_event = MultiEventTable(event_tables=[event1, event2, event3], common_info=common_info)

    # Before propagation, check initial state. `date_col` now defaults to None.
    assert event1.select_cols is None
    assert event1.conditions is None
    assert event1.date_col is None  # Changed from `== DATE_COL_DEFAULT`
    assert event1.table_name is None
    assert event1.table_query is None
    assert event1.start_date is None

    # Perform propagation
    multi_event.propogate_common_info()

    # Assert propagation for event1 (initially None, so should get common_info)
    assert event1.select_cols == ["id", "timestamp"]
    assert event1.conditions == ["is_active = TRUE"]
    # Changed from `== DATE_COL_DEFAULT` to reflect the propagation from common_info.
    assert event1.date_col == "event_date"
    assert event1.table_name == "default_table"
    assert event1.table_query == "SELECT * FROM default_source WHERE type = 'data'"
    assert event1.start_date == "2024-01-01"
    assert event1.end_date == "2024-01-31"

    # Assert propagation for event2 (had existing conditions, so common should NOT override)
    assert event2.select_cols == ["id", "timestamp"]
    assert event2.conditions == ["specific = 'yes'"]
    assert event2.date_col == "event_date"
    assert event2.table_name == "specific_table"  # Should retain its specific value
    assert event2.table_query == "SELECT * FROM default_source WHERE type = 'data'"
    assert event2.start_date == "2024-01-01"
    assert event2.end_date == "2024-01-31"

    # Assert propagation for event3 (initially None for conditions, so should get common_info)
    assert event3.select_cols == ["custom_col"]  # Should retain its specific value
    assert event3.conditions == ["is_active = TRUE"]
    assert event3.date_col == "custom_date_col"  # Should retain its specific value
    assert event3.table_name == "default_table"
    assert event3.table_query == "SELECT * FROM default_source WHERE type = 'data'"
    assert event3.start_date == "2024-01-01"
    assert event3.end_date == "2024-01-31"

    # Test case: common_info is None. This tests the fallback to DATE_COL_DEFAULT.
    # Resetting event1 for this sub-test, as it was modified above.
    event1_reset = EventTable(event_label="eventA_reset")
    multi_event_no_common = MultiEventTable(event_tables=[event1_reset])

    # Assert initial state is None
    assert event1_reset.date_col is None

    multi_event_no_common.propogate_common_info()

    # The propagation method will now assign the default date_col if common_info is None
    assert event1_reset.select_cols is None
    assert event1_reset.conditions is None
    # Now, `date_col` is assigned the default value when common_info is None
    assert event1_reset.date_col == DATE_COL_DEFAULT

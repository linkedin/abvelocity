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


from abvelocity.journey.event.gen_event_query import (
    EventTable,
    convert_to_snake_case,
    gen_event_query,
)
from abvelocity.testing.assert_query_is_equal import assert_query_is_equal


def test_convert_to_snake_case():
    # Test case 1
    result = convert_to_snake_case(
        "u_owner.member_journey_raw_TRACKING.randomProductChooserPlanActionEvent"
    )
    assert result == "u_owner_member_journey_raw_tracking_RandomProduct_chooser_plan_action_event"

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
    create_table_prefix = "u_owner.temp_reza"

    event_table = EventTable(
        table_name="TRACKING.randomProductUpsellImpressionEvent",
        event_label="impression",
        select_cols=[
            "datepartition",
            "header.memberid",
            "header.sessionUrn",
            "header.time AS time",
            "requestHeader.pageKey",
            "header.pageInstance.pageUrn",
            "RandomProductfunnelcommonheader.referenceid",
        ],
        date_col="datepartition",
        conditions=[
            "RandomProductfunnelcommonheader.referenceid IS NOT NULL",
            "MOD(header.memberid, 10000) IN (13)",
        ],
    )

    obtained_query = gen_event_query(
        event_table=event_table,
        start_date=start_date,
        end_date=end_date,
        create_table_prefix=create_table_prefix,
    )

    expected_query = """
    DROP TABLE IF EXISTS u_owner.temp_reza_tracking_RandomProduct_upsell_impression_event_impression;
    CREATE TABLE IF NOT EXISTS u_owner.temp_reza_tracking_RandomProduct_upsell_impression_event_impression AS
    SELECT
        datepartition,
        header.memberid,
        header.sessionUrn,
        header.time AS time,
        requestHeader.pageKey,
        header.pageInstance.pageUrn,
        RandomProductfunnelcommonheader.referenceid,
        'impression' AS event
    FROM TRACKING.randomProductUpsellImpressionEvent
    WHERE TRUE
        AND datepartition BETWEEN '2024-10-06-00' AND '2024-10-13-00'
        AND RandomProductfunnelcommonheader.referenceid IS NOT NULL
        AND MOD(header.memberid, 10000) IN (13)
    """

    assert_query_is_equal(obtained_query, expected_query)

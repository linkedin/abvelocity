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

from abvelocity.core.journey.seq.gen_event_count_map_query import gen_event_count_map_query
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal


def test_gen_event_count_map_query_no_time():
    """Test the function without providing the optional time_col."""
    table_name = "user_activity"
    partition_by_cols = ["user_id", "session_id"]
    event_col = "action"

    # Expected query when time_col is None
    expected_query = """
    WITH EventCountsAndTime AS (
        SELECT
            user_id, session_id, action,
            COUNT(*) AS event_count
        FROM user_activity
        GROUP BY user_id, session_id, action
    )
    SELECT
        user_id, session_id,
        MAP_AGG(action, event_count) AS event_seq,
        SUM(EVENT_COUNT) AS SEQ_LENGTH
    FROM EventCountsAndTime
    GROUP BY user_id, session_id
    """

    # Generate the query using the function
    generated_query = gen_event_count_map_query(
        table_name=table_name,
        partition_by_cols=partition_by_cols,
        event_col=event_col,
        # time_col is omitted
    )

    # Assert that the generated query matches the expected one
    assert_query_is_equal(generated_query, expected_query)


def test_gen_event_count_map_query_with_time():
    """Test the function when providing the optional time_col."""
    table_name = "user_activity"
    partition_by_cols = ["user_id", "session_id"]
    event_col = "action"
    time_col = "event_timestamp"

    # Expected query when time_col is provided
    expected_query = """
    WITH EventCountsAndTime AS (
        SELECT
            user_id, session_id, action, MIN(event_timestamp) AS min_t, MAX(event_timestamp) AS max_t,
            COUNT(*) AS event_count
        FROM user_activity
        GROUP BY user_id, session_id, action
    )
    SELECT
        user_id, session_id,
        MAP_AGG(action, event_count) AS event_seq,
        SUM(EVENT_COUNT) AS SEQ_LENGTH,
        MIN(min_t) AS start_seq_time,
        MAX(max_t) AS end_seq_time
    FROM EventCountsAndTime
    GROUP BY user_id, session_id
    """

    # Generate the query using the function
    generated_query = gen_event_count_map_query(
        table_name=table_name,
        partition_by_cols=partition_by_cols,
        event_col=event_col,
        time_col=time_col,  # time_col is provided
    )

    # Assert that the generated query matches the expected one
    assert_query_is_equal(generated_query, expected_query)

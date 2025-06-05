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


from abvelocity.journey.seq.gen_undeduped_seq_query import gen_undeduped_seq_query
from abvelocity.testing.assert_query_is_equal import assert_query_is_equal


def test_gen_undeduped_seq_query_default_order():
    """Test the function with default time ordering (no order_list)."""
    table_name = "activity_log"
    partition_by_cols = ["user_id"]
    time_col = "ts"
    event_col = "action"

    # Expected query when order_list is None
    expected_query = """
    WITH AggregatedSequences AS (
        SELECT
            user_id,
            ARRAY_AGG(action ORDER BY ts) AS event_seq,
            MIN(ts) AS seq_start_time,
            MAX(ts) AS seq_end_time
        FROM activity_log
        GROUP BY user_id
    )
    SELECT
        user_id,
        event_seq,
        seq_start_time,
        seq_end_time,
        cardinality(event_seq) AS seq_length
    FROM AggregatedSequences
    """

    # Generate the query using the function
    generated_query = gen_undeduped_seq_query(
        table_name=table_name,
        partition_by_cols=partition_by_cols,
        time_col=time_col,
        event_col=event_col,
        # order_list is omitted
    )

    # Assert that the generated query matches the expected one
    assert_query_is_equal(generated_query, expected_query)


def test_gen_undeduped_seq_query_with_order_list():
    """Test the function with a custom order_list."""
    table_name = "activity_log"
    partition_by_cols = ["user_id", "day"]
    time_col = "ts"
    event_col = "action"
    order_list = ["view", "click", "purchase"]  # Custom order

    sort_array_str = (
        """ARRAY_AGG(action ORDER BY """
        + """CASE WHEN action = 'view' THEN 0 WHEN action = 'click' THEN 1 WHEN action = 'purchase' THEN 2 ELSE 3 END, ts)"""
        + """ AS event_seq"""
    )

    # Expected query when order_list is provided
    expected_query = f"""
    WITH AggregatedSequences AS (
        SELECT
            user_id, day,
            {sort_array_str},
            MIN(ts) AS seq_start_time,
            MAX(ts) AS seq_end_time
        FROM activity_log
        GROUP BY user_id, day
    )
    SELECT
        user_id, day,
        event_seq,
        seq_start_time,
        seq_end_time,
        cardinality(event_seq) AS seq_length
    FROM AggregatedSequences
    """

    # Generate the query using the function
    generated_query = gen_undeduped_seq_query(
        table_name=table_name,
        partition_by_cols=partition_by_cols,
        time_col=time_col,
        event_col=event_col,
        order_list=order_list,  # order_list is provided
    )

    # Assert that the generated query matches the expected one
    assert_query_is_equal(generated_query, expected_query)


def test_gen_undeduped_seq_query_with_partial_order_list():
    """Test the function with a partial custom order_list."""
    table_name = "activity_log"
    partition_by_cols = ["user_id", "day"]
    time_col = "ts"
    event_col = "action"
    order_list = ["purchase", "view"]  # Partial custom order

    # Expected query when a partial order_list is provided
    # 'purchase' gets 0, 'view' gets 1, others get 2, then ordered by time
    expected_query = """
    WITH AggregatedSequences AS (
        SELECT
            user_id, day,
            ARRAY_AGG(action ORDER BY CASE WHEN action = 'purchase' THEN 0 WHEN action = 'view' THEN 1 ELSE 2 END, ts) AS event_seq,
            MIN(ts) AS seq_start_time,
            MAX(ts) AS seq_end_time
        FROM activity_log
        GROUP BY user_id, day
    )
    SELECT
        user_id, day,
        event_seq,
        seq_start_time,
        seq_end_time,
        cardinality(event_seq) AS seq_length
    FROM AggregatedSequences
    """

    # Generate the query using the function
    generated_query = gen_undeduped_seq_query(
        table_name=table_name,
        partition_by_cols=partition_by_cols,
        time_col=time_col,
        event_col=event_col,
        order_list=order_list,  # Partial order_list is provided
    )

    # Assert that the generated query matches the expected one
    assert_query_is_equal(generated_query, expected_query)

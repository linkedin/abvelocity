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
from abvelocity.core.journey.seq.gen_deduped_seq_query import gen_deduped_seq_query
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal


def test_gen_deduped_seq_query():

    table_name = "events_table"
    partition_by_cols = ["user_id", "session_id"]

    expected_query = """
    WITH RankedEvents AS (
        SELECT
            user_id, session_id,
            time,
            event,
            LAG(event) OVER (PARTITION BY user_id, session_id ORDER BY time) AS previous_event
        FROM events_table
    ),
    DeduplicatedEvents AS (
        SELECT
            user_id, session_id,
            time,
            event
        FROM RankedEvents
        WHERE event != previous_event OR previous_event IS NULL
    ),
    ArrayedEvents AS (
        SELECT
            user_id, session_id,
            ARRAY_AGG(event ORDER BY time) AS event_seq,
            MIN(time) AS seq_start_time,
            MAX(time) AS seq_end_time
        FROM DeduplicatedEvents
        GROUP BY user_id, session_id
    )
    SELECT
        user_id, session_id,
        event_seq,
        seq_start_time,
        seq_end_time,
        cardinality(event_seq) AS seq_length
    FROM ArrayedEvents
    """.strip()

    generated_query = gen_deduped_seq_query(table_name, partition_by_cols)

    assert_query_is_equal(generated_query, expected_query)


def test_gen_deduped_seq_query_invalid_partition_by_cols():
    table_name = "events_table"
    with pytest.raises(ValueError, match="partition_by_cols must be a list of column names."):
        gen_deduped_seq_query(table_name, "invalid_partition_col")

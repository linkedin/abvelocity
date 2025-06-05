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

from abvelocity.journey.seq.seq_query import SeqQuery
from abvelocity.testing.assert_query_is_equal import assert_query_is_equal


def test_gen_consecutive_deduping():
    # Test for consecutive deduping method
    query = SeqQuery(
        event_table_name="events",
        time_col="event_time",
        event_col="event_name",
        output_table_name=None,
        partition_by_cols=["user_id", "session_id"],
        deduping_method="consecutive_deduped",
    )

    expected_query = """
    WITH RankedEvents AS (
        SELECT
            user_id, session_id,
            event_time,
            event_name,
            LAG(event_name) OVER (PARTITION BY user_id, session_id ORDER BY event_time) AS previous_event
        FROM events
    ),
    DeduplicatedEvents AS (
        SELECT
            user_id, session_id,
            event_time,
            event_name
        FROM RankedEvents
        WHERE event_name != previous_event OR previous_event IS NULL
    ),
    ArrayedEvents AS (
        SELECT
            user_id, session_id,
            ARRAY_AGG(event_name ORDER BY event_time) AS event_seq,
            MIN(event_time) AS seq_start_time,
            MAX(event_time) AS seq_end_time
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

    generated_query = query.gen()
    assert_query_is_equal(generated_query, expected_query)


def test_gen_fully_deduping():
    # Test for fully deduping method
    query = SeqQuery(
        event_table_name="events",
        time_col="event_time",
        event_col="event_name",
        output_table_name=None,
        partition_by_cols=["user_id", "session_id"],
        deduping_method="fully_deduped",
    )

    expected_query = """
    WITH RankedEvents AS (
        SELECT
            user_id, session_id,
            event_time,
            event_name,
            ROW_NUMBER() OVER (PARTITION BY user_id, session_id, event_name ORDER BY event_time) AS event_rank
        FROM events
    ),
    DeduplicatedEvents AS (
        SELECT
            user_id, session_id,
            event_time,
            event_name
        FROM RankedEvents
        WHERE event_rank = 1
    ),
    ArrayedEvents AS (
        SELECT
            user_id, session_id,
            ARRAY_AGG(event_name ORDER BY event_time) AS event_seq,
            MIN(event_time) AS seq_start_time,
            MAX(event_time) AS seq_end_time
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

    generated_query = query.gen()
    assert_query_is_equal(generated_query, expected_query)


def test_gen_with_max_seq_index():
    # Test for max_seq_index
    query = SeqQuery(
        event_table_name="events",
        time_col="event_time",
        event_col="event_name",
        output_table_name=None,
        partition_by_cols=["user_id", "session_id"],
        deduping_method="consecutive_deduped",
        max_seq_index=3,
    )

    expected_query = """
    WITH BaseData AS (WITH RankedEvents AS (
        SELECT user_id, session_id, event_time, event_name,
               LAG(event_name) OVER (PARTITION BY user_id, session_id ORDER BY event_time) AS previous_event
        FROM events
    ),
    DeduplicatedEvents AS (
        SELECT user_id, session_id, event_time, event_name
        FROM RankedEvents
        WHERE event_name != previous_event OR previous_event IS NULL
    ),
    ArrayedEvents AS (
        SELECT user_id, session_id, ARRAY_AGG(event_name ORDER BY event_time) AS event_seq,
            MIN(event_time) AS seq_start_time,
            MAX(event_time) AS seq_end_time
        FROM DeduplicatedEvents
        GROUP BY user_id, session_id
    )
    SELECT user_id, session_id, event_seq, seq_start_time, seq_end_time, cardinality(event_seq) AS seq_length
    FROM ArrayedEvents)
    SELECT user_id, session_id, event_seq, seq_start_time, seq_end_time, seq_length,
        element_at(event_seq, 1) AS s1, element_at(event_seq, 2) AS s2, element_at(event_seq, 3) AS s3
    FROM BaseData

    """.strip()

    generated_query = query.gen()
    assert_query_is_equal(generated_query, expected_query)


def test_gen_invalid_deduping_method():
    # Test for invalid deduping_method
    with pytest.raises(ValueError, match="deduping_method must be"):
        SeqQuery(
            event_table_name="events",
            time_col="event_time",
            event_col="event_name",
            output_table_name="output",
            partition_by_cols=["user_id", "session_id"],
            deduping_method="invalid_method",
        ).gen()

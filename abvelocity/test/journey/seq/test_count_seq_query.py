# BSD 2-CLAUSE LICENSE
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

from abvelocity.journey.seq.count_seq_query import CountSeqQuery
from abvelocity.testing.assert_query_is_equal import assert_query_is_equal


def test_gen_count_seq_query_default():
    # Test case where no conditions or extra group by columns are provided.
    query = CountSeqQuery(table_name="my_event_table", max_seq_index=5)

    expected_query = """
    SELECT
        s1, s2, s3, s4, s5, COUNT(*) AS count, COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS percent
    FROM my_event_table
    GROUP BY s1, s2, s3, s4, s5
    """

    # Using the custom assert function to check if the queries are equal
    assert_query_is_equal(query.gen(), expected_query)


def test_count_seq_query_with_conditions_and_groupby():
    # Test case where conditions and extra group by columns are provided.
    query = CountSeqQuery(
        table_name="my_event_table",
        max_seq_index=5,
        count_distinct_col="memberid",
        extra_groupby_cols=["date", "location"],
        conditions=["status = 'active'", "country = 'US'"],
    )

    expected_query = """
    SELECT
        s1, s2, s3, s4, s5, COUNT(DISTINCT memberid) AS count,
        COUNT(DISTINCT memberid) * 100.0 / SUM(COUNT(DISTINCT memberid)) OVER () AS percent,
        date,
        location
    FROM my_event_table
    WHERE status = 'active' AND country = 'US'
    GROUP BY s1, s2, s3, s4, s5, date, location
    """

    # Using the custom assert function to check if the queries are equal
    assert_query_is_equal(query.gen(), expected_query)


def test_gen_count_seq_query_with_percent():
    # Test case for generating percentages alongside counts
    query = CountSeqQuery(
        table_name="my_event_table",
        max_seq_index=3,
        count_distinct_col="userid",
    )

    expected_query = """
    SELECT
        s1, s2, s3, COUNT(DISTINCT userid) AS count, COUNT(DISTINCT userid) * 100.0 / SUM(COUNT(DISTINCT userid)) OVER () AS percent
    FROM my_event_table
    GROUP BY s1, s2, s3
    """

    assert_query_is_equal(query.gen(), expected_query)


def test_percent_with_conditions():
    # Test for percentage with conditions in the WHERE clause
    query = CountSeqQuery(
        table_name="my_event_table",
        max_seq_index=2,
        count_distinct_col="memberid",
        conditions=["status = 'complete'"],
    )

    expected_query = """
    SELECT
        s1, s2, COUNT(DISTINCT memberid) AS count, COUNT(DISTINCT memberid) * 100.0 / SUM(COUNT(DISTINCT memberid)) OVER () AS percent
    FROM my_event_table
    WHERE status = 'complete'
    GROUP BY s1, s2
    """

    assert_query_is_equal(query.gen(), expected_query)

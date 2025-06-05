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


from abvelocity.journey.seq.add_seq_elements_as_cols import add_seq_elements_as_cols
from abvelocity.testing.assert_query_is_equal import assert_query_is_equal


def test_add_seq_elements_as_cols_with_extra_cols():
    base_query = "SELECT * FROM seq_table"
    partition_by_cols = ["user_id", "session_id"]
    max_seq_index = 3
    seq_col = "event_seq"
    extra_cols = ["created_at", "updated_at"]

    expected_query = """
    WITH BaseData AS (SELECT * FROM seq_table)
    SELECT
        user_id, session_id,
        event_seq,
        created_at, updated_at,
        element_at(event_seq, 1) AS s1, element_at(event_seq, 2) AS s2, element_at(event_seq, 3) AS s3
    FROM BaseData
    """.strip()

    generated_query = add_seq_elements_as_cols(
        base_query, partition_by_cols, max_seq_index, seq_col, extra_cols
    ).strip()

    print("generated_query")
    print("expected_query")

    assert_query_is_equal(generated_query, expected_query)


def test_add_seq_elements_as_cols_without_extra_cols():
    base_query = "SELECT * FROM seq_table"
    partition_by_cols = ["user_id", "session_id"]
    max_seq_index = 2
    seq_col = "event_seq"
    extra_cols = None

    expected_query = """
    WITH BaseData AS (SELECT * FROM seq_table)
    SELECT
        user_id, session_id,
        event_seq,
        element_at(event_seq, 1) AS s1, element_at(event_seq, 2) AS s2
    FROM BaseData
    """.strip()

    generated_query = add_seq_elements_as_cols(
        base_query, partition_by_cols, max_seq_index, seq_col, extra_cols
    ).strip()

    assert_query_is_equal(generated_query, expected_query)

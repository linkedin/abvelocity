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
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal


def test_assert_query_is_equal():
    # Test with identical queries
    query1 = "SELECT * FROM users WHERE id = 1"
    query2 = "SELECT * FROM users WHERE id = 1"
    assert_query_is_equal(query1, query2)

    # Test with different spacing and line breaks
    query3 = "SELECT *  FROM users   WHERE id = 1"
    query4 = "SELECT * FROM users WHERE id = 1"
    assert_query_is_equal(query3, query4)

    # Test with extra line breaks
    query5 = "SELECT *\nFROM users\nWHERE id = 1"
    query6 = "SELECT * FROM users WHERE id = 1"
    assert_query_is_equal(query5, query6)


def test_assert_query_is_not_equal():
    query1 = "SELECT * FROM users WHERE id = 1"
    query2 = "SELECT * FROM users WHERE id = 2"
    with pytest.raises(AssertionError):
        assert_query_is_equal(query1, query2)

    query3 = "SELECT name FROM users WHERE id = 1"
    query4 = "SELECT * FROM users WHERE id = 1"
    with pytest.raises(AssertionError):
        assert_query_is_equal(query3, query4)


def test_equal_queries():
    query1 = "SELECT * FROM MyTable WHERE column1 = 'Value1' AND column2 = 'value2'"
    query2 = "select * from mytable where COLUMN1 = 'Value1' and COLUMN2 = 'value2'"
    assert assert_query_is_equal(query1, query2), "Test failed: Equal queries"


def test_different_string_literals():
    query1 = "SELECT * FROM MyTable WHERE column1 = 'Value1' AND column2 = 'value2'"
    query2 = "select * from mytable where COLUMN1 = 'value1' and COLUMN2 = 'value2'"
    with pytest.raises(AssertionError):  # Use pytest.raises
        assert_query_is_equal(query1, query2)


def test_different_keywords():
    query1 = "SELECT * FROM MyTable WHERE column1 = 'Value1' AND column2 = 'value2'"
    query2 = "select * from mytable where COLUMN1 = 'Value1' OR COLUMN2 = 'value2'"
    with pytest.raises(AssertionError):
        assert_query_is_equal(query1, query2)


def test_different_table_names():
    query1 = "SELECT * FROM MyTable WHERE column1 = 'Value1'"
    query2 = "select * from othertable where COLUMN1 = 'Value1'"
    with pytest.raises(AssertionError):  # Use pytest.raises here
        assert_query_is_equal(query1, query2)


def test_different_column_names():
    query1 = "SELECT col1 FROM MyTable WHERE col2 = 'Value1'"
    query2 = "select column1 from mytable where column2 = 'Value1'"
    with pytest.raises(AssertionError):
        assert_query_is_equal(query1, query2), "Test failed: Different column names"


def test_with_comments():
    query1 = """
    SELECT * -- This is a comment
    FROM MyTable /* Another
    multi-line comment */
    WHERE column1 = 'Value1'
    """
    query2 = "select * from mytable where column1 = 'Value1'"
    assert assert_query_is_equal(query1, query2), "Test failed: Comments not handled correctly"

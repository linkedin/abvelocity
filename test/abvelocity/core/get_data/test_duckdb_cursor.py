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


import duckdb
import pandas as pd
import pytest
from abvelocity.core.get_data.cursor import SqlQueryResult
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor

# --- Pytest Fixture ---


@pytest.fixture(scope="function")
def duckdb_cursor():
    """Provides a fresh DuckDBCursor connected to an in-memory database."""

    # Instantiates the cursor using max_retries=1 to prevent the test from waiting
    cursor = DuckDBCursor(max_retries=1)

    # 1. Setup Mock Data: Use DuckDB's register function
    df_users = pd.DataFrame({"user_id": [101, 102, 103], "username": ["alice", "bob", "charlie"], "age": [30, 45, 22]})

    # Register the DataFrame as a temporary view
    cursor._db_connection.register("users_mock_view", df_users)

    # Use the registered view to create a persistent table visible to the cursor
    cursor._db_connection.execute("CREATE TABLE users_mock AS SELECT * FROM users_mock_view;")

    # Now unregister the temporary view
    cursor._db_connection.unregister("users_mock_view")

    yield cursor

    # 2. Teardown (Clean up the connection)
    cursor.close()


# --- Test Cases ---


def test_get_df_select_data(duckdb_cursor: DuckDBCursor):
    """Tests retrieval of data using get_df() and checks the resulting DataFrame structure."""
    query = "SELECT user_id, age FROM users_mock WHERE age > 25 ORDER BY user_id;"
    result = duckdb_cursor.get_df(query)

    # Assertions for data retrieval and structure
    assert isinstance(result, SqlQueryResult)
    assert isinstance(result.df, pd.DataFrame)
    assert result.df.shape == (2, 2)
    assert list(result.df["user_id"]) == [101, 102]
    assert result.query_time_taken > 0


def test_execute_multi_create_and_drop_table(duckdb_cursor: DuckDBCursor):
    """Tests DDL operations (CREATE and DROP) using execute_multi()."""

    # 1. CREATE TABLE
    create_query = "CREATE TABLE new_table_test (id INTEGER, name VARCHAR);"
    duckdb_cursor.execute_multi(create_query)

    # Verify the table exists by trying to query it (a clean SELECT)
    check_query = "SELECT * FROM new_table_test LIMIT 0;"
    result = duckdb_cursor.get_df(check_query)
    assert result.df is not None
    assert list(result.df.columns) == ["id", "name"]

    # 2. DROP TABLE
    drop_query = "DROP TABLE new_table_test;"
    duckdb_cursor.execute_multi(drop_query)

    # 3. Verify it was dropped by expecting a DuckDB error
    with pytest.raises(duckdb.CatalogException, match="Table with name new_table_test does not exist"):
        # We use the raw execute method here. Since max_retries=1 in the fixture,
        # the test fails fast on the first attempt, confirming the table is gone.
        duckdb_cursor.execute("SELECT * FROM new_table_test;")


def test_execute_multi_dml_operations(duckdb_cursor: DuckDBCursor):
    """Tests DML operations (INSERT and UPDATE) and verifies the final state."""

    # 1. Create a table and insert initial data
    setup_query = """
    CREATE TABLE balances (user_id INTEGER, amount DOUBLE);
    INSERT INTO balances VALUES (500, 100.0);
    """
    duckdb_cursor.execute_multi(setup_query)

    # 2. Execute an UPDATE operation using execute_multi
    update_query = "UPDATE balances SET amount = amount + 50.0 WHERE user_id = 500;"
    duckdb_cursor.execute_multi(update_query)

    # 3. Verify the final data state using get_df()
    result = duckdb_cursor.get_df("SELECT amount FROM balances WHERE user_id = 500;")

    assert result.df.shape == (1, 1)
    assert result.df.iloc[0]["amount"] == 150.0


def test_execute_multi_multiple_statements_in_one_call(duckdb_cursor: DuckDBCursor):
    """Tests that execute_multi can correctly handle both DDL and DML in one string."""
    combined_query = """
    CREATE TABLE multi_test (key VARCHAR, count INTEGER);
    INSERT INTO multi_test VALUES ('A', 5);
    INSERT INTO multi_test VALUES ('B', 10);
    """
    duckdb_cursor.execute_multi(combined_query)

    # Verify both inserts happened
    result = duckdb_cursor.get_df("SELECT SUM(count) AS total FROM multi_test;")
    assert result.df.iloc[0]["total"] == 15

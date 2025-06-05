import sqlite3

import pandas as pd
import pytest

from abvelocity.get_data.data_container import DataContainer
from abvelocity.get_data.sqlite_cursor import SqliteCursor


# Test fixtures
@pytest.fixture
def sqlite_conn():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def cursor(sqlite_conn):
    return SqliteCursor(sqlite_conn)


@pytest.fixture
def sample_dfs():
    df1 = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    df2 = pd.DataFrame({"id": [2, 3], "category": ["x", "y"]})
    return df1, df2


@pytest.fixture
def sample_sql_tables(sqlite_conn):
    df1 = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    df2 = pd.DataFrame({"id": [2, 3], "category": ["x", "y"]})
    df1.to_sql("table1", sqlite_conn, if_exists="replace", index=False)
    df2.to_sql("table2", sqlite_conn, if_exists="replace", index=False)
    return DataContainer(table_name="table1", is_sql_table=True), DataContainer(
        table_name="table2", is_sql_table=True
    )


# Tests
def test_dataframe_join(sample_dfs):
    df1, df2 = sample_dfs
    dc1 = DataContainer(df=df1, is_df=True)
    dc2 = DataContainer(df=df2, is_df=True)
    result = dc1.join(dc2, join_type="INNER", on="a.id = b.id")
    assert result.is_df
    assert not result.is_sql_table
    assert result.df.equals(pd.DataFrame({"id": [2], "value": ["b"], "category": ["x"]}))


def test_sql_join_no_materialize(cursor, sample_sql_tables):  # cursor added
    dc1, dc2 = sample_sql_tables
    result = dc1.join(dc2, join_type="INNER", on="a.id = b.id", cursor=cursor)  # cursor added
    assert not result.is_df
    assert not result.is_sql_table
    # Update assert to match the new query
    expected_query = "SELECT a.*, b.category FROM table1 AS a INNER JOIN table2 AS b ON a.id = b.id"
    assert result.query == expected_query


def test_sql_join_materialize(cursor, sample_sql_tables):
    dc1, dc2 = sample_sql_tables
    result = dc1.join(
        dc2,
        join_type="INNER",
        on="a.id = b.id",
        table_name="joined_table",
        cursor=cursor,
        materialize_to_sql=True,
    )
    assert not result.is_df
    assert result.is_sql_table
    assert result.table_name == "joined_table"
    # Verify table exists and has correct data
    check_df = pd.read_sql_query("SELECT * FROM joined_table", cursor.data_source)
    expected_df = pd.DataFrame({"id": [2], "value": ["b"], "category": ["x"]})
    check_df = check_df[expected_df.columns]  # reorder the columns.
    assert check_df.equals(expected_df)


def test_sql_join_materialize_drop_existing(cursor, sample_sql_tables):
    dc1, dc2 = sample_sql_tables
    # Create the table first with different data
    pd.DataFrame({"id": [999], "value": ["old"], "category": ["old"]}).to_sql(
        "joined_table", cursor.data_source, if_exists="replace", index=False
    )
    result = dc1.join(
        dc2,
        join_type="INNER",
        on="a.id = b.id",
        table_name="joined_table",
        cursor=cursor,
        materialize_to_sql=True,
    )
    assert result.is_sql_table
    # Verify old data is gone, new data is correct
    check_df = pd.read_sql_query("SELECT * FROM joined_table", cursor.data_source)
    expected_df = pd.DataFrame({"id": [2], "value": ["b"], "category": ["x"]})
    check_df = check_df[expected_df.columns]  # reorder the columns.
    assert check_df.equals(expected_df)


def test_materialize_to_df(cursor, sample_sql_tables):  # cursor added
    dc1, dc2 = sample_sql_tables
    result = dc1.join(dc2, join_type="INNER", on="a.id = b.id", cursor=cursor)  # cursor added
    result.materialize(cursor, to_df=True)
    assert result.is_df
    assert not result.is_sql_table
    expected_df = pd.DataFrame({"id": [2], "value": ["b"], "category": ["x"]})
    assert result.df.equals(expected_df)

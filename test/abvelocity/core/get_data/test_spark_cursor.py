# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
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
"""Tests for SparkCursor.write_pandas_df and write_spark_df."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from abvelocity.core.get_data.spark_cursor import ConnArgs, SparkCursor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cursor_with_mock_session():
    """Return a SparkCursor backed by a MagicMock SparkSession."""
    mock_session = MagicMock()
    conn_args = ConnArgs(spark_session=mock_session)
    cursor = SparkCursor(conn_args=conn_args)
    return cursor, mock_session


def _spark_df_mock(mock_session, columns=None):
    """Return the Spark DataFrame mock produced by createDataFrame, optionally with set columns."""
    mock_df = mock_session.createDataFrame.return_value
    if columns is not None:
        mock_df.columns = list(columns)
    return mock_df


def _writer(mock_session):
    """Return the DataFrameWriter mock chained from createDataFrame."""
    return mock_session.createDataFrame.return_value.write


# ---------------------------------------------------------------------------
# Tests: argument validation
# ---------------------------------------------------------------------------


def test_partition_col_without_value_raises():
    cursor, _ = make_cursor_with_mock_session()
    df = pd.DataFrame({"ts": ["2024-01-01"], "value": [1.0]})
    with pytest.raises(ValueError, match="partition_value is required"):
        cursor.write_pandas_df(
            df=df,
            table_name="my_table",
            partition_col="datepartition",
        )


# ---------------------------------------------------------------------------
# Tests: basic write (no partitioning)
# ---------------------------------------------------------------------------


def test_write_basic_calls_save_as_table():
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"ts": ["2024-01-01", "2024-01-02"], "value": [1.0, 2.0]})

    cursor.write_pandas_df(df=df, table_name="my_schema.my_table")

    mock_session.createDataFrame.assert_called_once_with(df)
    writer = _writer(mock_session)
    writer.mode.assert_called_once_with("append")
    writer.mode.return_value.saveAsTable.assert_called_once_with("my_schema.my_table")


def test_write_default_mode_is_append():
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"value": [1.0]})
    cursor.write_pandas_df(df=df, table_name="t")
    _writer(mock_session).mode.assert_called_once_with("append")


def test_write_overwrite_mode_passed_through():
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"value": [1.0]})
    cursor.write_pandas_df(df=df, table_name="t", mode="overwrite")
    _writer(mock_session).mode.assert_called_once_with("overwrite")


# ---------------------------------------------------------------------------
# Tests: data_format
# ---------------------------------------------------------------------------


def test_write_with_data_format_calls_format():
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"value": [1.0]})
    cursor.write_pandas_df(df=df, table_name="t", data_format="orc")

    writer = _writer(mock_session)
    mode_writer = writer.mode.return_value
    mode_writer.format.assert_called_once_with("orc")


def test_write_without_data_format_skips_format():
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"value": [1.0]})
    cursor.write_pandas_df(df=df, table_name="t")

    writer = _writer(mock_session)
    mode_writer = writer.mode.return_value
    mode_writer.format.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: partitioning
# ---------------------------------------------------------------------------


def test_write_with_partition_calls_partition_by():
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"ts": ["2024-01-01"], "value": [1.0]})
    _spark_df_mock(mock_session, columns=["ts", "value"])  # no datepartition

    with patch("abvelocity.core.get_data.spark_cursor.lit", MagicMock(return_value="__lit__")):
        cursor.write_pandas_df(
            df=df,
            table_name="t",
            partition_col="datepartition",
            partition_value="2024-01-01-00",
        )

    spark_df_mock = mock_session.createDataFrame.return_value
    writer = spark_df_mock.withColumn.return_value.write
    mode_chain = writer.mode.return_value
    mode_chain.partitionBy.assert_called_once_with("datepartition")


def test_write_partition_col_not_in_df_adds_literal():
    """When partition_col is absent from the Spark DataFrame, withColumn is called to add it."""
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"ts": ["2024-01-01"], "value": [1.0]})
    _spark_df_mock(mock_session, columns=["ts", "value"])  # no datepartition

    with patch("abvelocity.core.get_data.spark_cursor.lit", MagicMock(return_value="__lit__")):
        cursor.write_pandas_df(
            df=df,
            table_name="t",
            partition_col="datepartition",
            partition_value="2024-01-01-00",
        )

    spark_df_mock = mock_session.createDataFrame.return_value
    spark_df_mock.withColumn.assert_called_once()
    col_name_called = spark_df_mock.withColumn.call_args[0][0]
    assert col_name_called == "datepartition"


def test_write_partition_col_already_in_df_skips_with_column():
    """When partition_col is already in the Spark DataFrame columns, withColumn is NOT called."""
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"ts": ["2024-01-01"], "value": [1.0], "datepartition": ["2024-01-01-00"]})
    _spark_df_mock(mock_session, columns=["ts", "value", "datepartition"])

    cursor.write_pandas_df(
        df=df,
        table_name="t",
        partition_col="datepartition",
        partition_value="2024-01-01-00",
    )

    spark_df_mock = mock_session.createDataFrame.return_value
    spark_df_mock.withColumn.assert_not_called()


def test_write_without_partition_skips_partition_by():
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"value": [1.0]})
    cursor.write_pandas_df(df=df, table_name="t")

    spark_df_mock = mock_session.createDataFrame.return_value
    mode_chain = spark_df_mock.write.mode.return_value
    mode_chain.partitionBy.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: write_spark_df
# ---------------------------------------------------------------------------


def make_spark_df_mock(columns=("ts", "value")):
    """Return a MagicMock that looks like a Spark DataFrame."""
    mock_df = MagicMock()
    mock_df.columns = list(columns)
    return mock_df


def test_write_spark_df_basic_calls_save_as_table():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock()

    cursor.write_spark_df(df=spark_df, table_name="my_schema.my_table")

    spark_df.write.mode.assert_called_once_with("append")
    spark_df.write.mode.return_value.saveAsTable.assert_called_once_with("my_schema.my_table")


def test_write_spark_df_no_createDataFrame_called():
    cursor, mock_session = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock()

    cursor.write_spark_df(df=spark_df, table_name="t")

    mock_session.createDataFrame.assert_not_called()


def test_write_spark_df_default_mode_is_append():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock()
    cursor.write_spark_df(df=spark_df, table_name="t")
    spark_df.write.mode.assert_called_once_with("append")


def test_write_spark_df_overwrite_mode_passed_through():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock()
    cursor.write_spark_df(df=spark_df, table_name="t", mode="overwrite")
    spark_df.write.mode.assert_called_once_with("overwrite")


def test_write_spark_df_with_data_format_calls_format():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock()
    cursor.write_spark_df(df=spark_df, table_name="t", data_format="parquet")
    spark_df.write.mode.return_value.format.assert_called_once_with("parquet")


def test_write_spark_df_without_data_format_skips_format():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock()
    cursor.write_spark_df(df=spark_df, table_name="t")
    spark_df.write.mode.return_value.format.assert_not_called()


def test_write_spark_df_partition_col_without_value_raises():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock()
    with pytest.raises(ValueError, match="partition_value is required"):
        cursor.write_spark_df(df=spark_df, table_name="t", partition_col="datepartition")


def test_write_spark_df_partition_col_not_in_df_adds_literal():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock(columns=("ts", "value"))  # no datepartition

    with patch("abvelocity.core.get_data.spark_cursor.lit", MagicMock(return_value="__lit__")):
        cursor.write_spark_df(
            df=spark_df,
            table_name="t",
            partition_col="datepartition",
            partition_value="2024-01-01-00",
        )

    spark_df.withColumn.assert_called_once()
    assert spark_df.withColumn.call_args[0][0] == "datepartition"


def test_write_spark_df_partition_col_already_in_df_skips_with_column():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock(columns=("ts", "value", "datepartition"))

    cursor.write_spark_df(
        df=spark_df,
        table_name="t",
        partition_col="datepartition",
        partition_value="2024-01-01-00",
    )

    spark_df.withColumn.assert_not_called()


def test_write_spark_df_with_partition_calls_partition_by():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock(columns=("ts", "value"))

    with patch("abvelocity.core.get_data.spark_cursor.lit", MagicMock(return_value="__lit__")):
        cursor.write_spark_df(
            df=spark_df,
            table_name="t",
            partition_col="datepartition",
            partition_value="2024-01-01-00",
        )

    writer = spark_df.withColumn.return_value.write
    writer.mode.return_value.partitionBy.assert_called_once_with("datepartition")


def test_write_spark_df_without_partition_skips_partition_by():
    cursor, _ = make_cursor_with_mock_session()
    spark_df = make_spark_df_mock()
    cursor.write_spark_df(df=spark_df, table_name="t")
    spark_df.write.mode.return_value.partitionBy.assert_not_called()

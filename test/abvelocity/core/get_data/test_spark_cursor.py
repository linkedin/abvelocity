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
"""Tests for SparkCursor.write_pandas_df, write_spark_df, and spark_df_to_pandas."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Skip this whole module when pyspark isn't installed. Every test in here
# either calls ``SparkCursor.write_pandas_df`` (which invokes
# ``spark_schema_from_pandas_dtypes`` and dereferences pyspark types at call
# time), exercises ``spark_schema_from_pandas_dtypes`` directly, or tests a
# pyspark-pandas compat shim — none of them are useful without pyspark
# importable. The pre-merge ``project`` satellite ships without pyspark, so
# an unguarded module-level import here would crash the whole file's tests
# with ``NameError``/``ImportError`` instead of cleanly skipping.
pytest.importorskip("pyspark")

from abvelocity.core.get_data.spark_cursor import (  # noqa: E402
    PANDAS_DTYPE_TO_SPARK_TYPE,
    SPARK_TEMPORAL_DTYPES,
    ConnArgs,
    SparkCursor,
    spark_df_to_pandas,
    spark_schema_from_pandas_dtypes,
)
from pyspark.sql.types import (  # noqa: E402
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructType,
    TimestampType,
)

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

    # createDataFrame is now schema-aware — pdf passed positionally, schema as kwarg.
    assert mock_session.createDataFrame.call_count == 1
    args, kwargs = mock_session.createDataFrame.call_args
    assert args == (df,)
    assert "schema" in kwargs and kwargs["schema"] is not None
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
# Tests: schema inference from pandas dtypes
# ---------------------------------------------------------------------------


def test_spark_schema_from_pandas_dtypes_maps_common_dtypes():
    """Object → String, int64 → Long, float64 → Double, bool → Boolean, datetime64[ns] → Timestamp."""
    df = pd.DataFrame(
        {
            "name": pd.Series(["a", "b"], dtype="object"),
            "count": pd.Series([1, 2], dtype="int64"),
            "amount": pd.Series([1.0, 2.0], dtype="float64"),
            "active": pd.Series([True, False], dtype="bool"),
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    schema = spark_schema_from_pandas_dtypes(df=df)
    by_name = {field.name: type(field.dataType) for field in schema.fields}
    assert by_name == {
        "name": StringType,
        "count": LongType,
        "amount": DoubleType,
        "active": BooleanType,
        "ts": TimestampType,
    }


def test_spark_schema_from_pandas_dtypes_all_fields_nullable():
    """Every StructField must be nullable=True so callers don't have to clean nulls upstream."""
    df = pd.DataFrame({"a": [1], "b": [1.0], "c": ["x"]})
    schema = spark_schema_from_pandas_dtypes(df=df)
    assert all(field.nullable is True for field in schema.fields)


def test_spark_schema_from_pandas_dtypes_all_null_float_col_still_double():
    """Regression: an all-NaN float column still maps to DoubleType, not StringType.

    This is the exact prod failure mode from production on 2026-05-05 — pyspark's
    value-inference path raised ``ValueError: Some of types cannot be determined
    after inferring`` on the all-None ``actual`` column. Pandas keeps the dtype
    as ``float64`` even when every value is ``NaN``, so we should too.
    """
    df = pd.DataFrame({"actual": pd.Series([float("nan"), float("nan")], dtype="float64")})
    schema = spark_schema_from_pandas_dtypes(df=df)
    assert len(schema.fields) == 1
    assert isinstance(schema.fields[0].dataType, DoubleType)


def test_spark_schema_from_pandas_dtypes_unknown_dtype_falls_back_to_string():
    """A dtype string not in the map (e.g. ``category``) falls back to StringType."""
    df = pd.DataFrame({"c": pd.Categorical(["a", "b"])})
    assert str(df["c"].dtype) not in PANDAS_DTYPE_TO_SPARK_TYPE
    schema = spark_schema_from_pandas_dtypes(df=df)
    assert isinstance(schema.fields[0].dataType, StringType)


def test_spark_schema_from_pandas_dtypes_int32_and_float32_distinct():
    """int32 → IntegerType (not LongType); float32 → FloatType (not DoubleType)."""
    df = pd.DataFrame(
        {
            "i32": pd.Series([1, 2], dtype="int32"),
            "f32": pd.Series([1.0, 2.0], dtype="float32"),
        }
    )
    schema = spark_schema_from_pandas_dtypes(df=df)
    by_name = {field.name: type(field.dataType) for field in schema.fields}
    assert by_name == {"i32": IntegerType, "f32": FloatType}


def test_write_pandas_df_no_schema_builds_one_from_dtypes():
    """No ``schema=`` kwarg → write_pandas_df builds a StructType from the dtypes
    and passes it explicitly to createDataFrame (no value-inference)."""
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame(
        {"actual": pd.Series([float("nan"), float("nan")], dtype="float64"),
         "metric_id": ["a", "b"]}
    )
    cursor.write_pandas_df(df=df, table_name="t")
    assert mock_session.createDataFrame.call_count == 1
    args, kwargs = mock_session.createDataFrame.call_args
    assert args == (df,)
    assert isinstance(kwargs["schema"], StructType)
    by_name = {f.name: type(f.dataType) for f in kwargs["schema"].fields}
    assert by_name["actual"] is DoubleType
    assert by_name["metric_id"] is StringType


def test_write_pandas_df_explicit_schema_passes_through_verbatim():
    """When the caller hands in a ``schema=``, write_pandas_df forwards it as-is —
    no re-derivation from dtypes, no overrides."""
    cursor, mock_session = make_cursor_with_mock_session()
    df = pd.DataFrame({"x": [1, 2]})
    explicit_schema = MagicMock()  # opaque token; identity check is enough
    cursor.write_pandas_df(df=df, table_name="t", schema=explicit_schema)
    args, kwargs = mock_session.createDataFrame.call_args
    assert kwargs["schema"] is explicit_schema


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


# ---------------------------------------------------------------------------
# Tests: spark_df_to_pandas
#
# Workaround for pyspark 3.1.1 (LI shaded) ↔ pandas 2.2+ skew on the
# holdem grid — see the docstring on spark_df_to_pandas for the full story.
# ---------------------------------------------------------------------------


def make_temporal_spark_df_mock(toPandas_result: pd.DataFrame, dtypes):
    """Build a Spark-DataFrame mock that round-trips through the cast→toPandas→to_datetime path.

    The shape of the mock matches what spark_df_to_pandas exercises:
      - ``.dtypes`` is a list of ``(col_name, dtype_str)`` pairs.
      - ``df[col]`` returns a column mock with ``.cast("string")``.
      - ``.withColumn(name, col)`` returns the same df mock (chainable).
      - ``.toPandas()`` returns the supplied pandas frame.
    """
    mock_df = MagicMock()
    mock_df.dtypes = list(dtypes)
    mock_df.withColumn.return_value = mock_df
    mock_df.toPandas.return_value = toPandas_result
    # Column subscript: any column name → MagicMock with .cast()
    mock_df.__getitem__ = lambda self, key: MagicMock()
    return mock_df


def test_spark_df_to_pandas_casts_timestamp_columns_to_string_in_spark():
    """TIMESTAMP columns get a Spark withColumn cast — bypassing pyspark's broken astype path."""
    pandas_after_cast = pd.DataFrame(
        {"ts": ["2024-01-01 12:00:00", "2024-01-02 13:00:00"], "value": [1.0, 2.0]}
    )
    spark_df = make_temporal_spark_df_mock(
        toPandas_result=pandas_after_cast,
        dtypes=[("ts", "timestamp"), ("value", "double")],
    )

    spark_df_to_pandas(spark_df)

    # Exactly one withColumn — only the timestamp column needed casting.
    spark_df.withColumn.assert_called_once()
    assert spark_df.withColumn.call_args[0][0] == "ts"


def test_spark_df_to_pandas_casts_date_columns_too():
    """DATE columns also need the workaround — same broken astype path."""
    pandas_after_cast = pd.DataFrame({"d": ["2024-01-01", "2024-01-02"], "value": [1.0, 2.0]})
    spark_df = make_temporal_spark_df_mock(
        toPandas_result=pandas_after_cast,
        dtypes=[("d", "date"), ("value", "double")],
    )

    spark_df_to_pandas(spark_df)

    spark_df.withColumn.assert_called_once()
    assert spark_df.withColumn.call_args[0][0] == "d"


def test_spark_df_to_pandas_casts_both_temporal_kinds():
    """Mixed TIMESTAMP + DATE → both columns get cast, non-temporal stays untouched."""
    pandas_after_cast = pd.DataFrame(
        {"ts": ["2024-01-01 12:00:00"], "d": ["2024-01-01"], "value": [1.0]}
    )
    spark_df = make_temporal_spark_df_mock(
        toPandas_result=pandas_after_cast,
        dtypes=[("ts", "timestamp"), ("d", "date"), ("value", "double")],
    )

    spark_df_to_pandas(spark_df)

    # Two withColumn calls, one per temporal column. Order follows dtypes order.
    cast_cols = [call.args[0] for call in spark_df.withColumn.call_args_list]
    assert cast_cols == ["ts", "d"]


def test_spark_df_to_pandas_skips_cast_when_no_temporal_columns():
    """No TIMESTAMP / DATE columns → withColumn is never called (workaround is a no-op)."""
    pandas_after_cast = pd.DataFrame({"value": [1.0, 2.0], "label": ["a", "b"]})
    spark_df = make_temporal_spark_df_mock(
        toPandas_result=pandas_after_cast,
        dtypes=[("value", "double"), ("label", "string")],
    )

    spark_df_to_pandas(spark_df)

    spark_df.withColumn.assert_not_called()


def test_spark_df_to_pandas_restores_datetime64_dtype_on_temporal_columns():
    """After toPandas, the temporal columns come back as proper datetime64[ns] — not strings."""
    pandas_after_cast = pd.DataFrame(
        {"ts": ["2024-01-01 12:00:00", "2024-01-02 13:00:00"], "value": [1.0, 2.0]}
    )
    spark_df = make_temporal_spark_df_mock(
        toPandas_result=pandas_after_cast,
        dtypes=[("ts", "timestamp"), ("value", "double")],
    )

    df = spark_df_to_pandas(spark_df)

    # ts column should now hold real timestamps, not strings.
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])
    assert df["ts"].iloc[0] == pd.Timestamp("2024-01-01 12:00:00")
    # Non-temporal column stays as it was.
    assert df["value"].iloc[0] == 1.0


def test_spark_df_to_pandas_returns_empty_frame_when_spark_df_is_empty():
    """Empty result still passes through — the workaround must not blow up on 0 rows."""
    pandas_after_cast = pd.DataFrame({"ts": pd.Series([], dtype="object"), "value": pd.Series([], dtype="float64")})
    spark_df = make_temporal_spark_df_mock(
        toPandas_result=pandas_after_cast,
        dtypes=[("ts", "timestamp"), ("value", "double")],
    )

    df = spark_df_to_pandas(spark_df)

    assert df.empty
    # Even an empty column should be retyped as datetime — pd.to_datetime on
    # an empty object series yields a datetime64[ns] series.
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])


def test_spark_temporal_dtypes_constant_is_what_get_df_uses():
    """SPARK_TEMPORAL_DTYPES is the published constant — keep it in sync with the workaround logic."""
    assert SPARK_TEMPORAL_DTYPES == ("timestamp", "date")


# ---------------------------------------------------------------------------
# Tests: SparkCursor.get_df integration with the workaround
# ---------------------------------------------------------------------------


def test_get_df_routes_through_spark_df_to_pandas():
    """``get_df`` uses ``spark_df_to_pandas`` — so the workaround applies on every read."""
    cursor, _ = make_cursor_with_mock_session()
    pandas_after_cast = pd.DataFrame({"ts": ["2024-01-01 12:00:00"], "value": [1.0]})
    cursor._spark_df = make_temporal_spark_df_mock(
        toPandas_result=pandas_after_cast,
        dtypes=[("ts", "timestamp"), ("value", "double")],
    )
    # Bypass the real execute() — the test is about post-execute materialization.
    with patch.object(cursor, "execute"):
        result = cursor.get_df("SELECT * FROM does_not_matter")

    assert pd.api.types.is_datetime64_any_dtype(result.df["ts"])
    assert result.df["ts"].iloc[0] == pd.Timestamp("2024-01-01 12:00:00")


# ---------------------------------------------------------------------------
# Tests: pyspark 3.1.1 ↔ pandas 2.x iteritems compat shim (top of spark_cursor.py)
# ---------------------------------------------------------------------------


def test_iteritems_shim_active_after_module_import():
    """Importing ``spark_cursor`` runs the module-top shim that aliases
    ``pd.DataFrame.iteritems`` to ``pd.DataFrame.items``. By the time any
    test in this file runs, ``iteritems`` must be present and callable."""
    assert hasattr(pd.DataFrame, "iteritems")
    assert pd.DataFrame.iteritems is pd.DataFrame.items


def test_iteritems_yields_same_pairs_as_items():
    """The shim makes ``df.iteritems()`` functionally identical to
    ``df.items()`` — same ``(column_name, Series)`` pairs in the same
    order. This is the contract pyspark's
    ``_convert_from_pandas`` no-schema branch relies on."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})

    items_pairs = [(name, list(s)) for name, s in df.items()]
    iteritems_pairs = [(name, list(s)) for name, s in df.iteritems()]

    assert items_pairs == iteritems_pairs
    assert items_pairs == [("a", [1, 2, 3]), ("b", [4.0, 5.0, 6.0]), ("c", ["x", "y", "z"])]


@pytest.mark.parametrize("nrows", [0, 1, 10])
def test_iteritems_works_on_empty_and_nonempty_dfs(nrows):
    """The shim must work on 0-row DataFrames too — that's what an empty
    forecast result feeds ``createDataFrame``, which would otherwise hit
    the broken pyspark code path during teardown."""
    df = pd.DataFrame({"a": list(range(nrows)), "b": ["x"] * nrows})
    pairs = list(df.iteritems())
    assert [name for name, _ in pairs] == ["a", "b"]
    assert all(len(series) == nrows for _, series in pairs)


def test_pyspark_no_schema_branch_walk_succeeds():
    """Mirror the pyspark 3.1.1 ``_convert_from_pandas`` no-schema branch:
    ``for column, series in pdf.iteritems():`` — without the shim this
    raises ``AttributeError`` on pandas 2.x. With the shim it walks
    every column."""
    df = pd.DataFrame({
        "metric_name":     ["signups", "signups"],
        "forecast":        [1.1, 2.2],
        "forecasted_date": pd.to_datetime(["2026-05-05", "2026-05-06"]),
    })
    columns_seen = []
    for column, series in df.iteritems():
        columns_seen.append(column)
        assert isinstance(series, pd.Series)
    assert columns_seen == ["metric_name", "forecast", "forecasted_date"]

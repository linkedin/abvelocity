# Original author: Reza Hosseini
"""Tests for :mod:`abvelocity.core.get_data.sql_inline_writer`."""

import datetime

import pandas as pd
import pytest

from abvelocity.core.get_data.sql_inline_writer import (
    build_insert_values_sql,
    chunked,
    format_sql_value,
    render_values_clauses,
    upsert_pandas_df,
    write_pandas_df,
)


class DummyCursor:
    """Minimal cursor double — records every ``execute(sql)`` for assertions."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, sql: str) -> None:
        self.executed.append(sql)


# ──────────────────────────────────────────────────────────────────────────
# format_sql_value
# ──────────────────────────────────────────────────────────────────────────


def test_format_sql_value_none_renders_null():
    assert format_sql_value(value=None) == "NULL"


def test_format_sql_value_nan_renders_null():
    assert format_sql_value(value=float("nan")) == "NULL"


def test_format_sql_value_string_doubles_single_quotes():
    """Embedded ``'`` in strings must be escaped to ``''`` to keep SQL valid."""
    assert format_sql_value(value="O'Brien") == "'O''Brien'"


def test_format_sql_value_int_and_float_render_as_numeric_literals():
    assert format_sql_value(value=3) == "3"
    assert format_sql_value(value=1.5) == "1.5"


def test_format_sql_value_bool_renders_as_trino_keyword():
    assert format_sql_value(value=True) == "TRUE"
    assert format_sql_value(value=False) == "FALSE"


def test_format_sql_value_date_typed_column_renders_date_literal():
    """Listed in ``date_columns`` → DATE literal, not a quoted string."""
    out = format_sql_value(
        value=datetime.date(2024, 1, 3),
        column_name="forecasted_date",
        date_columns=frozenset({"forecasted_date"}),
    )
    assert out == "DATE '2024-01-03'"


def test_format_sql_value_date_typed_accepts_string_input():
    """ISO strings into a DATE-typed column must still render as DATE literals."""
    out = format_sql_value(
        value="2024-01-03",
        column_name="forecasted_date",
        date_columns=frozenset({"forecasted_date"}),
    )
    assert out == "DATE '2024-01-03'"


def test_format_sql_value_date_value_into_non_date_column_renders_quoted_string():
    """A datetime value going into a non-DATE column should NOT become a DATE literal."""
    out = format_sql_value(
        value=datetime.date(2024, 1, 3),
        column_name="some_string_col",
        date_columns=frozenset({"forecasted_date"}),
    )
    assert out == "'2024-01-03'"


# ──────────────────────────────────────────────────────────────────────────
# render_values_clauses / chunked / build_insert_values_sql
# ──────────────────────────────────────────────────────────────────────────


def test_render_values_clauses_emits_one_tuple_per_row():
    df = pd.DataFrame({"metric_id": ["a", "b"], "forecast": [1.5, float("nan")]})
    out = render_values_clauses(df=df, columns=["metric_id", "forecast"])
    assert out == ["('a', 1.5)", "('b', NULL)"]


def test_render_values_clauses_orders_by_columns_argument():
    """Tuple positions must follow the ``columns`` arg, not df column order."""
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    out = render_values_clauses(df=df, columns=["b", "a"])
    assert out == ["('x', 1)"]


def test_render_values_clauses_date_columns_propagates():
    df = pd.DataFrame({"ds": [datetime.date(2024, 1, 3)], "v": [9]})
    out = render_values_clauses(
        df=df,
        columns=["ds", "v"],
        date_columns=frozenset({"ds"}),
    )
    assert out == ["(DATE '2024-01-03', 9)"]


def test_chunked_yields_full_then_remainder():
    assert list(chunked(seq=[1, 2, 3, 4, 5], size=2)) == [[1, 2], [3, 4], [5]]


def test_chunked_empty_input_yields_nothing():
    assert list(chunked(seq=[], size=10)) == []


def test_build_insert_values_sql_assembles_full_statement():
    sql = build_insert_values_sql(
        table_name="db.t",
        columns=["a", "b"],
        values_clauses=["(1, 'x')", "(2, 'y')"],
    )
    assert sql == "INSERT INTO db.t (\"a\", \"b\") VALUES\n(1, 'x'),\n(2, 'y')"


def test_build_insert_values_sql_quotes_column_names():
    """Reserved words / dotted columns must be quoted to survive Trino parsing."""
    sql = build_insert_values_sql(
        table_name="db.t",
        columns=["last_training_date"],
        values_clauses=["('2024-01-03')"],
    )
    assert '"last_training_date"' in sql


# ──────────────────────────────────────────────────────────────────────────
# write_pandas_df
# ──────────────────────────────────────────────────────────────────────────


def test_write_pandas_df_append_mode_emits_only_inserts():
    cursor = DummyCursor()
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    inserted = write_pandas_df(
        cursor=cursor,
        df=df,
        table_name="db.t",
        mode="append",
    )
    assert inserted == 2
    assert len(cursor.executed) == 1
    assert cursor.executed[0].startswith("INSERT INTO db.t")
    assert "(1, 'x')" in cursor.executed[0]
    assert "(2, 'y')" in cursor.executed[0]


def test_write_pandas_df_overwrite_partition_emits_delete_then_insert():
    cursor = DummyCursor()
    df = pd.DataFrame({"part": ["2024-01-03", "2024-01-03"], "v": [1, 2]})
    inserted = write_pandas_df(
        cursor=cursor,
        df=df,
        table_name="db.t",
        mode="overwrite_partition",
        partition_col="part",
        partition_value="2024-01-03",
    )
    assert inserted == 2
    assert len(cursor.executed) == 2
    assert cursor.executed[0] == "DELETE FROM db.t WHERE \"part\" = '2024-01-03'"
    assert cursor.executed[1].startswith("INSERT INTO db.t")


def test_write_pandas_df_chunks_inserts_when_rows_exceed_chunk_size():
    cursor = DummyCursor()
    df = pd.DataFrame({"a": list(range(5)), "b": list(range(5))})
    inserted = write_pandas_df(
        cursor=cursor,
        df=df,
        table_name="db.t",
        mode="append",
        chunk_size=2,
    )
    assert inserted == 5
    # ceil(5/2) = 3 INSERT statements, no DELETE in append mode.
    assert len(cursor.executed) == 3
    assert all(stmt.startswith("INSERT INTO") for stmt in cursor.executed)


def test_write_pandas_df_overwrite_partition_requires_partition_args():
    cursor = DummyCursor()
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="partition_col"):
        write_pandas_df(
            cursor=cursor,
            df=df,
            table_name="db.t",
            mode="overwrite_partition",
        )


def test_write_pandas_df_rejects_unknown_mode():
    cursor = DummyCursor()
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="mode"):
        write_pandas_df(cursor=cursor, df=df, table_name="db.t", mode="merge")


def test_write_pandas_df_rejects_invalid_chunk_size():
    cursor = DummyCursor()
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="chunk_size"):
        write_pandas_df(
            cursor=cursor,
            df=df,
            table_name="db.t",
            chunk_size=0,
        )


# ──────────────────────────────────────────────────────────────────────────
# upsert_pandas_df
# ──────────────────────────────────────────────────────────────────────────


def test_upsert_pandas_df_emits_multi_key_delete_then_insert():
    cursor = DummyCursor()
    df = pd.DataFrame(
        {
            "k1": ["a", "a", "b"],
            "k2": [1, 1, 2],
            "v": [10, 11, 12],
        }
    )
    inserted = upsert_pandas_df(
        cursor=cursor,
        df=df,
        table_name="db.t",
        key_cols=["k1", "k2"],
    )
    assert inserted == 3
    delete_sql = cursor.executed[0]
    assert delete_sql.startswith("DELETE FROM db.t")
    # Distinct (k1, k2) tuples: ('a', 1) and ('b', 2). Both must appear in the IN.
    assert '("k1", "k2") IN' in delete_sql
    assert "('a', 1)" in delete_sql
    assert "('b', 2)" in delete_sql
    assert cursor.executed[1].startswith("INSERT INTO db.t")


def test_upsert_pandas_df_dedups_keys_so_delete_is_compact():
    """3 input rows but only 2 distinct (k1) tuples → DELETE IN has 2 entries, not 3."""
    cursor = DummyCursor()
    df = pd.DataFrame({"k1": ["a", "a", "b"], "v": [1, 2, 3]})
    upsert_pandas_df(
        cursor=cursor,
        df=df,
        table_name="db.t",
        key_cols=["k1"],
    )
    delete_sql = cursor.executed[0]
    # 'a' appears once (deduped), 'b' once.
    assert delete_sql.count("'a'") == 1
    assert delete_sql.count("'b'") == 1


def test_upsert_pandas_df_rejects_empty_df():
    cursor = DummyCursor()
    with pytest.raises(ValueError, match="empty"):
        upsert_pandas_df(
            cursor=cursor,
            df=pd.DataFrame({"k1": [], "v": []}),
            table_name="db.t",
            key_cols=["k1"],
        )


def test_upsert_pandas_df_rejects_missing_key_cols():
    cursor = DummyCursor()
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="missing key_cols"):
        upsert_pandas_df(
            cursor=cursor,
            df=df,
            table_name="db.t",
            key_cols=["a", "b"],
        )

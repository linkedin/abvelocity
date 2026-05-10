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
"""Generic Python→Trino inline-VALUES writer.

The Trino DB-API client doesn't expose ``executemany`` and there is
no ``COPY``-from-pandas path. For laptop / dev INSERTs of a few
thousand rows, the cheapest portable approach is to render the
DataFrame as inline SQL ``VALUES`` literals and execute as one (or
a chunked few) INSERT statements.

This module provides the schema-agnostic primitives:

  - :func:`format_sql_value` — Python value → Trino SQL literal.
  - :func:`render_values_clauses` — DataFrame rows → ``(v1, v2, …)``
    literal strings.
  - :func:`chunked` — slice helper.
  - :func:`build_insert_values_sql` — assemble the INSERT statement.
  - :func:`write_pandas_df` — full writer with ``append`` /
    ``overwrite_partition`` modes; takes any cursor-like object with
    an ``execute(sql)`` method.
  - :func:`upsert_pandas_df` — multi-key DELETE-then-INSERT (rebuilds
    rows whose key tuples appear in ``df``).

These functions are cursor-agnostic — pass any object exposing
``execute(sql)``. ``PrestoCursor`` wraps them in instance methods of
the same names so callers can do ``cursor.write_pandas_df(df,
table)``.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, FrozenSet, Iterable, List, Optional, Sequence

import pandas as pd

# Per-INSERT row cap. Keeps any single statement well under Trino's
# max statement size (~a few MB). 500 rows × ~22 cols × ~30 chars ≈
# 330KB per chunk — comfortably under the limit.
DEFAULT_INSERT_CHUNK_SIZE = 500

EMPTY_DATE_COLUMNS: FrozenSet[str] = frozenset()


def format_sql_value(
    value: Any,
    column_name: str = "",
    date_columns: FrozenSet[str] = EMPTY_DATE_COLUMNS,
) -> str:
    """Render one Python value as a Trino SQL literal.

    Type handling:
      - ``None`` / ``NaN`` → ``NULL``
      - ``bool`` → ``TRUE`` / ``FALSE``
      - ``int`` / ``float`` → numeric literal
      - ``datetime.date`` / ``datetime.datetime`` / ``pd.Timestamp`` →
        ``DATE 'YYYY-MM-DD'`` if the column is in ``date_columns``,
        otherwise a single-quoted string. (Time component is dropped.)
      - ``str`` going into a ``date_columns``-listed column →
        ``DATE '<string>'`` (caller must pre-validate the format).
      - Other strings → ``'<escaped>'`` (single quotes doubled).

    Args:
        value: Python value to render.
        column_name: Target column name; used only to look up
            membership in ``date_columns``. Pass empty when the
            target type doesn't matter.
        date_columns: Columns whose target SQL type is ``DATE``.
            Membership triggers DATE-literal rendering.

    Returns:
        Trino SQL literal as a string.

    Raises:
        TypeError: If ``column_name`` is in ``date_columns`` and
            ``value`` is not a recognized date / datetime / string.
    """
    if value is None:
        return "NULL"
    if isinstance(value, float) and math.isnan(value):
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if column_name in date_columns:
        if isinstance(value, (datetime.date, datetime.datetime)):
            return f"DATE '{value.isoformat()[:10]}'"
        if isinstance(value, pd.Timestamp):
            return f"DATE '{value.date().isoformat()}'"
        if isinstance(value, str):
            return f"DATE '{value[:10]}'"
        raise TypeError(f"Unsupported value type for DATE column {column_name!r}: {type(value).__name__}")
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return f"'{value.isoformat()[:10]}'"
    if isinstance(value, pd.Timestamp):
        return f"'{value.date().isoformat()}'"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def render_values_clauses(
    df: pd.DataFrame,
    columns: Sequence[str],
    date_columns: FrozenSet[str] = EMPTY_DATE_COLUMNS,
) -> List[str]:
    """Render one ``(v1, v2, …)`` tuple per ``df`` row as a SQL literal string.

    Args:
        df: Source DataFrame.
        columns: Column order — the rendered tuple positions follow this.
        date_columns: Columns to render as ``DATE 'YYYY-MM-DD'`` literals.

    Returns:
        One string per row of ``df``, each shaped ``(literal1, literal2, …)``.
    """
    cols = list(columns)
    sub = df[cols]
    return [
        "(" + ", ".join(format_sql_value(value=row[i], column_name=cols[i], date_columns=date_columns) for i in range(len(cols))) + ")"
        for row in sub.itertuples(index=False, name=None)
    ]


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield successive ``size``-length slices of ``seq``."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_insert_values_sql(
    table_name: str,
    columns: Sequence[str],
    values_clauses: Sequence[str],
) -> str:
    """Assemble a single ``INSERT INTO ... VALUES (...), (...), ...`` statement.

    Args:
        table_name: Fully-qualified target table.
        columns: Column names; rendered as quoted identifiers in the
            INSERT column list.
        values_clauses: Output of :func:`render_values_clauses`.

    Returns:
        A single executable INSERT SQL string (no trailing semicolon).
    """
    quoted_cols = ", ".join(f'"{col}"' for col in columns)
    values_sql = ",\n".join(values_clauses)
    return f"INSERT INTO {table_name} ({quoted_cols}) VALUES\n{values_sql}"


def write_pandas_df(
    cursor: Any,
    df: pd.DataFrame,
    table_name: str,
    mode: str = "append",
    partition_col: Optional[str] = None,
    partition_value: Optional[str] = None,
    date_columns: FrozenSet[str] = EMPTY_DATE_COLUMNS,
    chunk_size: int = DEFAULT_INSERT_CHUNK_SIZE,
) -> int:
    """Write a DataFrame to a Trino table via inline-VALUES INSERT.

    Modes:
      ``"append"``: INSERT only — adds rows to whatever's already
        there. ``partition_col`` / ``partition_value`` are ignored.
      ``"overwrite_partition"``: ``DELETE FROM table WHERE
        partition_col = partition_value``, then INSERT.
        ``partition_col`` and ``partition_value`` are required.

    Args:
        cursor: Any object with an ``execute(sql)`` method
            (e.g. :class:`PrestoCursor` or its inner cursor).
        df: DataFrame whose columns are the target column list.
            ``NaN`` becomes SQL ``NULL``.
        table_name: Fully-qualified target table.
        mode: ``"append"`` or ``"overwrite_partition"``.
        partition_col: Required for ``"overwrite_partition"``.
        partition_value: Required for ``"overwrite_partition"``.
        date_columns: Columns to render as DATE literals.
        chunk_size: Max rows per INSERT statement.

    Returns:
        Number of rows inserted.

    Raises:
        ValueError: If args don't match the chosen ``mode``.
    """
    if mode not in {"append", "overwrite_partition"}:
        raise ValueError(f"Unsupported mode {mode!r}; expected 'append' or 'overwrite_partition'.")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    if mode == "overwrite_partition":
        if not partition_col or not partition_value:
            raise ValueError("mode='overwrite_partition' requires partition_col and partition_value.")
        sanitized = partition_value.replace("'", "''")
        cursor.execute(f"DELETE FROM {table_name} WHERE \"{partition_col}\" = '{sanitized}'")

    columns = list(df.columns)
    values_clauses = render_values_clauses(df=df, columns=columns, date_columns=date_columns)
    for chunk in chunked(seq=values_clauses, size=chunk_size):
        sql = build_insert_values_sql(
            table_name=table_name,
            columns=columns,
            values_clauses=chunk,
        )
        cursor.execute(sql)
    return len(values_clauses)


def upsert_pandas_df(
    cursor: Any,
    df: pd.DataFrame,
    table_name: str,
    key_cols: Sequence[str],
    date_columns: FrozenSet[str] = EMPTY_DATE_COLUMNS,
    chunk_size: int = DEFAULT_INSERT_CHUNK_SIZE,
) -> int:
    """Multi-key delete-then-insert. Rebuilds rows whose key tuples appear in ``df``.

    Computes the distinct ``key_cols`` tuples present in ``df``,
    issues a single ``DELETE FROM table WHERE (k1, k2, ...) IN
    ((v1a, v2a, ...), (v1b, v2b, ...), ...)``, then INSERTs ``df`` in
    chunks. Idempotent: running with the same ``df`` twice leaves the
    table in the same state.

    Use case: write forecasts where the unique key spans
    ``(last_training_date, metric_id)`` — a per-partition INSERT
    OVERWRITE would clobber sibling metric rows in the same
    partition; this delete-by-tuple keeps siblings.

    Args:
        cursor: Object with ``execute(sql)``.
        df: DataFrame to write. Must contain every column in
            ``key_cols`` plus the rest of the target schema.
        table_name: Fully-qualified target table.
        key_cols: Columns whose distinct values define the rows to
            delete before inserting. Order matters for the ``IN``
            tuple comparison (Trino requires positional alignment).
        date_columns: Columns to render as DATE literals (used both
            in the DELETE filter for any date-typed key column AND in
            the INSERT VALUES).
        chunk_size: Max rows per INSERT statement.

    Returns:
        Number of rows inserted.

    Raises:
        ValueError: If ``df`` is empty (cannot infer keys to delete).
    """
    if df.empty:
        raise ValueError("upsert_pandas_df cannot infer keys to delete from an empty df.")
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing key_cols: {missing}")

    distinct_keys = df[list(key_cols)].drop_duplicates().reset_index(drop=True)
    key_tuples = render_values_clauses(
        df=distinct_keys,
        columns=key_cols,
        date_columns=date_columns,
    )
    quoted_keys = ", ".join(f'"{c}"' for c in key_cols)
    cursor.execute(f"DELETE FROM {table_name} WHERE ({quoted_keys}) IN ({', '.join(key_tuples)})")

    columns = list(df.columns)
    values_clauses = render_values_clauses(df=df, columns=columns, date_columns=date_columns)
    for chunk in chunked(seq=values_clauses, size=chunk_size):
        sql = build_insert_values_sql(
            table_name=table_name,
            columns=columns,
            values_clauses=chunk,
        )
        cursor.execute(sql)
    return len(values_clauses)

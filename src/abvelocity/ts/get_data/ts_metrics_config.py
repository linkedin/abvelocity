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
"""Configuration dataclass for time-series metric queries."""

from dataclasses import dataclass, field
from typing import Optional, Sequence

from abvelocity.ts.constants import TIME_COL
from abvelocity.ts.get_data.ts_transform import TSTransform

FREQ_TO_SQL_UNIT: dict[str, str] = {
    # Second
    "s": "second",
    "S": "second",
    # Minute — "T" is the legacy pandas alias, "min" is pandas ≥ 2.2
    "T": "minute",
    "min": "minute",
    # Hour — "H" is legacy, "h" is pandas ≥ 2.2
    "H": "hour",
    "h": "hour",
    # Day
    "D": "day",
    "d": "day",
    # Week
    "W": "week",
    # Month — "M" is legacy, "MS"/"ME" are start/end variants
    "M": "month",
    "MS": "month",
    "ME": "month",
    # Quarter — "Q" is legacy, "QS"/"QE" are start/end variants
    "Q": "quarter",
    "QS": "quarter",
    "QE": "quarter",
    # Year — "Y"/"A" are legacy, "YS"/"YE" are pandas ≥ 2.2
    "Y": "year",
    "A": "year",
    "YS": "year",
    "YE": "year",
}
"""Maps pandas-style freq aliases to SQL DATE_TRUNC unit strings.

Covers both legacy pandas < 2.2 aliases (``"H"``, ``"T"``, ``"M"``,
``"Q"``, ``"A"``) and the renamed aliases from pandas ≥ 2.2 (``"h"``,
``"min"``, ``"ME"``, ``"QE"``, ``"YE"``).  Month/quarter/year start and
end variants all map to the same truncation unit.  Offset-qualified forms
like ``"QS-JAN"`` are handled by stripping the suffix before lookup.
"""

SUPPORTED_DIALECTS = ("trino", "spark", "mysql", "duckdb")

# Python strftime → Java SimpleDateFormat, used for Spark TO_TIMESTAMP.
STRFTIME_TO_JAVA: dict[str, str] = {
    "%Y": "yyyy",
    "%m": "MM",
    "%d": "dd",
    "%H": "HH",
    "%M": "mm",
    "%S": "ss",
    "%f": "SSSSSS",
}


def parse_col_expr(time_col: str, time_format: str, dialect: str) -> str:
    """Wrap ``time_col`` in a dialect-appropriate string-to-timestamp call.

    Args:
        time_col: Column name holding the string-encoded timestamp.
        time_format: Python strftime format describing the string encoding,
            e.g. ``"%Y-%m-%d:%H"`` for ``"2024-01-01:00"``.
        dialect: ``"trino"`` or ``"spark"``.

    Returns:
        SQL parse expression (all use strftime format except Spark):

        - Trino:  ``DATE_PARSE(col, '%Y-%m-%d:%H')``
        - MySQL:  ``STR_TO_DATE(col, '%Y-%m-%d:%H')``
        - DuckDB: ``STRPTIME(col, '%Y-%m-%d:%H')``  *(testing only)*
        - Spark:  ``TO_TIMESTAMP(col, 'yyyy-MM-dd:HH')``
    """
    if dialect in ("trino", "mysql", "duckdb"):
        # All three use strftime syntax, same as Python.
        # trino: DATE_PARSE, mysql: STR_TO_DATE, duckdb: STRPTIME
        func = {"trino": "DATE_PARSE", "mysql": "STR_TO_DATE", "duckdb": "STRPTIME"}[dialect]
        return f"{func}({time_col}, '{time_format}')"
    # spark: TO_TIMESTAMP uses Java SimpleDateFormat syntax.
    java_fmt = time_format
    for py_token, java_token in STRFTIME_TO_JAVA.items():
        java_fmt = java_fmt.replace(py_token, java_token)
    return f"TO_TIMESTAMP({time_col}, '{java_fmt}')"


def build_time_expr(
    time_col: str,
    freq: str,
    dialect: str,
    time_format: Optional[str] = None,
) -> str:
    """Builds a SQL DATE_TRUNC expression for the given freq and dialect.

    Args:
        time_col: Column name in the source table (timestamp or string type).
        freq: Pandas-style frequency alias.  Both legacy (``"H"``, ``"T"``,
            ``"M"``, ``"Q"``, ``"A"``) and pandas ≥ 2.2 aliases (``"h"``,
            ``"min"``, ``"ME"``, ``"QE"``, ``"YE"``) are accepted.
            Offset-qualified forms like ``"QS-JAN"`` are also handled.
        dialect: SQL dialect — ``"trino"`` or ``"spark"``.
        time_format: Optional Python strftime format string for when
            ``time_col`` is a string column rather than a native timestamp
            (e.g. ``"%Y-%m-%d:%H"`` for values like ``"2024-01-01:00"``).
            When ``None`` the column is passed directly to ``DATE_TRUNC``.
            String-to-timestamp parsing is delegated to :func:`parse_col_expr`.

    Returns:
        SQL expression string, e.g. ``"DATE_TRUNC('day', event_ts)"``.

    Raises:
        ValueError: If ``freq`` or ``dialect`` is not supported.
    """
    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError(f"Unsupported dialect {dialect!r}. Supported: {SUPPORTED_DIALECTS}")
    # Strip offset qualifier, e.g. "QS-JAN" → "QS".
    freq_key = freq.split("-")[0].strip()
    unit = FREQ_TO_SQL_UNIT.get(freq_key)
    if unit is None:
        raise ValueError(f"Unsupported freq {freq!r}. " f"Supported aliases: {sorted(FREQ_TO_SQL_UNIT)}")
    col_expr = parse_col_expr(time_col, time_format, dialect) if time_format else time_col
    return f"DATE_TRUNC('{unit}', {col_expr})"


@dataclass
class TSMetricsConfig:
    """Configuration for building a time-bucketed aggregated metric query.

    Specifies how to extract and bucket a timestamp column so that the
    resulting DataFrame has one row per ``(time_bucket × dims)`` and can
    be passed directly to
    :class:`~abvelocity.ts.runner.TSRunner`.

    Attributes:
        time_col: Raw timestamp (or string) column name in the source table.
        freq: Pandas-style frequency alias — see :data:`FREQ_TO_SQL_UNIT` for
            all supported values.  Both legacy aliases (``"H"``, ``"T"``,
            ``"M"``, ``"Q"``, ``"A"``) and pandas ≥ 2.2 aliases (``"h"``,
            ``"min"``, ``"ME"``, ``"QE"``, ``"YE"``) are accepted.
        dialect: SQL dialect for query generation — ``"trino"`` (default)
            or ``"spark"``.
        time_format: Optional Python strftime format for string-typed time
            columns (e.g. ``"%Y-%m-%d:%H"``).  When set the column is wrapped
            in a parse call before ``DATE_TRUNC``.
        time_expr: Custom SQL expression for time bucketing. Auto-built
            from the other fields when ``None``.  Use this for non-standard
            cases, e.g. ``"TO_DATE(CAST(date_int AS VARCHAR), 'YYYYMMDD')"``.
        time_alias: Output column name for the bucketed timestamp.
            Defaults to :data:`~abvelocity.ts.constants.TIME_COL`
            (``"ts"``) so the result is directly compatible with
            :class:`~abvelocity.ts.config.ts_model_config.TSModelConfig`.
    """

    time_col: str
    """Raw timestamp (or string) column name in the source table."""

    freq: str
    """Pandas-style frequency alias, e.g. ``"h"``, ``"D"``, ``"MS"``, ``"QS-JAN"``, ``"YE"``."""

    dialect: str = "trino"
    """SQL dialect — ``"trino"`` or ``"spark"``."""

    time_format: Optional[str] = None
    """Python strftime format for string-typed time columns, e.g. ``"%Y-%m-%d:%H"``
    for values like ``"2024-01-01:00"``.  When set, ``time_col`` is wrapped in
    ``DATE_PARSE`` (Trino) or ``TO_TIMESTAMP`` (Spark) before ``DATE_TRUNC``.
    Ignored when ``time_expr`` is provided explicitly."""

    time_expr: Optional[str] = None
    """Custom SQL truncation expression; auto-built from ``time_col`` + ``freq``
    + ``dialect`` (+ ``time_format`` if set) when ``None``."""

    time_alias: str = field(default=TIME_COL)
    """Output column name for the bucketed timestamp (default ``"ts"``)."""

    post_fetch_transforms: Sequence[TSTransform] = field(default_factory=tuple)
    """Transformations applied to the wide-format DataFrame after fetch +
    ``DimCollapser``.  Run in order.  ``TSTransform`` lives in its own
    module (``ts_transform.py``) so this annotation can name the element
    type without importing the concrete-transforms module."""

    def __post_init__(self) -> None:
        if self.time_expr is None:
            self.time_expr = build_time_expr(self.time_col, self.freq, self.dialect, self.time_format)

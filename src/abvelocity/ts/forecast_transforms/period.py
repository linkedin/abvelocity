# Original author: Reza Hosseini
"""Time-period helpers — input-freq inference and per-period
expected-row-count computation.

Both helpers are date-aware: the row-count check uses the actual
period bounds (so monthly aggregations correctly expect 28 / 29 / 30 /
31 days depending on the calendar month, and yearly expects 365 or
366 by leap year).  No hard-coded lookup table.
"""

from __future__ import annotations

import pandas as pd

from abvelocity.ts.constants import FORECASTED_DATE_COL


def expected_count_in_period(
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    input_freq: str,
) -> int:
    """How many ``input_freq``-aligned timestamps fall within
    ``[period_start, period_end]`` inclusive.

    Args:
        period_start: Start of the target period (inclusive).
        period_end: End of the target period (inclusive).
        input_freq: Pandas freq alias of the input rows (``"D"``,
            ``"h"``, ``"min"``, ...).

    Returns:
        Count of input-freq rows that should fall in the period when
        the input data is gap-free and continuous.
    """
    return len(pd.date_range(start=period_start, end=period_end, freq=input_freq))


def normalize_to_period_alias(offset_or_period_alias: str) -> str:
    """Convert offset aliases (``"MS"``/``"ME"``, ``"QS"``/``"QE"``,
    ``"YS"``/``"YE"``/``"A"``) to the period alias pandas' ``Period`` /
    ``dt.to_period`` accepts (``"M"`` / ``"Q"`` / ``"Y"``).

    Anchored weekly aliases like ``"W-SAT"`` pass through unchanged —
    period accepts those.  Non-monthly/quarterly/yearly aliases pass
    through unchanged.

    Args:
        offset_or_period_alias: Pandas freq alias (offset-style or
            period-style).

    Returns:
        Period-style alias suitable for ``Series.dt.to_period(...)``.
    """
    base = offset_or_period_alias.split("-", 1)[0]
    offset_to_period = {
        "MS": "M",
        "ME": "M",
        "QS": "Q",
        "QE": "Q",
        "YS": "Y",
        "YE": "Y",
        "A": "Y",
    }
    return offset_to_period.get(base, offset_or_period_alias)


def infer_input_freq(forecast_df: pd.DataFrame, time_col: str = FORECASTED_DATE_COL) -> str:
    """Infer the freq of the forecast frame's timestamp column.

    Uses :func:`pandas.infer_freq` over the deduplicated, sorted unique
    timestamps in ``time_col``.  Mixed-stage frames are fine — fitted
    and forecast rows share the same freq grid.

    Args:
        forecast_df: The forecast frame.
        time_col: Timestamp column name.  Defaults to ``forecasted_date``.

    Returns:
        Pandas freq alias inferred from the data.

    Raises:
        ValueError: When the freq can't be inferred (too few rows,
            irregular gaps).  Caller should fail fast rather than guess.
    """
    timestamps = pd.to_datetime(forecast_df[time_col]).drop_duplicates().sort_values()
    freq = pd.infer_freq(timestamps)
    if freq is None:
        raise ValueError(f"Could not infer input freq from {time_col!r} — got " f"{len(timestamps)} unique timestamps with irregular gaps.")
    return freq

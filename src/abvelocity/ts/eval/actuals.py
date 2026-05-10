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
"""Per-(metric, forecasted_date) actuals deduplication, polymorphic over pandas / SQL.

A canonical forecast OH table holds one row per (training cutoff, forecasted
date), so a single (metric, forecasted_date) value appears in many partitions
— one per cutoff that has it as a fitted row. They *should* all agree on
``actual``; in practice late-arriving corrections, upstream drift, or
stale-code partitions cause occasional disagreement.

This module provides three layered functions:

* :func:`warn_on_partition_inconsistency` — pure pandas. Takes a frame of
  per-(metric, date) aggregates (mean / std / min / max / count) and (a)
  warns on any (metric, date) whose coefficient of variation exceeds
  ``relative_tolerance``, (b) returns a clean ``[metric_id, forecasted_date,
  actual]`` frame using the mean as the representative value.
* :func:`dedupe_actuals_from_partitions_df` — pandas core. Takes a long-format
  DataFrame of partition rows, computes the per-(metric, date) aggregates in
  pandas, then delegates the warn + cleanup to the helper above.
* :func:`fetch_actuals_from_forecast_table` — :class:`DataContainer` /
  :class:`IOParam` adapter. Routes to the pandas core when ``dc.is_pandas_df``,
  or runs the aggregation in SQL via ``io_param.cursor`` when ``dc.is_sql_table``,
  and pipes the result through the same warn + cleanup helper.

The mean (rather than max / min) is the representative value because
old buggy partitions could be either higher or lower than the truth, so
extremes aren't defensible.
"""

import warnings
from typing import Optional

import pandas as pd
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.param.io_param import IOParam

# SQL aggregation query used by :func:`fetch_actuals_from_forecast_table` in
# SQL mode. Aliases are deliberately verbose (``actual_mean``, ``actual_stddev``,
# etc.) so downstream variable names match what the column actually holds.
ACTUALS_AGG_QUERY = """
SELECT
    metric_id,
    forecasted_date,
    AVG(actual)    AS actual_mean,
    STDDEV(actual) AS actual_stddev,
    MIN(actual)    AS actual_min,
    MAX(actual)    AS actual_max,
    COUNT(actual)  AS actual_partition_count
FROM {table}
WHERE stage = 'fitted'
  AND actual IS NOT NULL
GROUP BY metric_id, forecasted_date
"""

AGGREGATE_OUTPUT_COLS = ["metric_id", "forecasted_date", "actual"]
AGGREGATE_DETAIL_COLS = [
    "metric_id",
    "forecasted_date",
    "actual_mean",
    "actual_stddev",
    "actual_cv",
    "actual_min",
    "actual_max",
    "actual_partition_count",
]


def warn_on_partition_inconsistency(
    aggregates: pd.DataFrame,
    relative_tolerance: float = 1e-3,
) -> pd.DataFrame:
    """Apply the cross-partition CV check, warn on disagreements, return clean actuals.

    Args:
        aggregates: Per-(metric_id, forecasted_date) aggregates with columns
            ``actual_mean``, ``actual_stddev``, ``actual_min``, ``actual_max``,
            ``actual_partition_count``. Typically the output of either
            :data:`ACTUALS_AGG_QUERY` (SQL path) or the pandas equivalent
            inside :func:`dedupe_actuals_from_partitions_df` (df path).
        relative_tolerance: Coefficient-of-variation (``stddev / |mean|``)
            threshold above which a (metric, date) disagreement triggers a
            warning. Default ``1e-3`` (0.1%): ignore tiny float / late-arrival
            noise, flag anything bigger.

    Returns:
        DataFrame with columns ``metric_id``, ``forecasted_date``, ``actual``
        (= the mean across partitions). Empty input → empty result with the
        same column shape.
    """
    if aggregates.empty:
        return pd.DataFrame(columns=AGGREGATE_OUTPUT_COLS)

    aggregates = aggregates.copy()
    # CV = stddev / |mean|. When |mean| ≈ 0 the division would explode; clip
    # to a tiny floor only for the CV math (the returned ``actual`` column
    # is NOT clipped — zero values stay zero). With this guard, mean=0
    # stddev≠0 still produces a huge CV → warns correctly; mean=0 stddev=0
    # produces CV=0 → no warning, also correct.
    cv_div_by_zero_floor = 1e-12
    abs_actual_mean = aggregates["actual_mean"].abs().clip(lower=cv_div_by_zero_floor)
    aggregates["actual_cv"] = aggregates["actual_stddev"].fillna(0) / abs_actual_mean
    inconsistent = aggregates[aggregates["actual_cv"] > relative_tolerance]
    if not inconsistent.empty:
        sample = inconsistent.head(5)[AGGREGATE_DETAIL_COLS]
        warnings.warn(
            f"{len(inconsistent)} (metric_id, forecasted_date) pairs disagree across partitions "
            f"(CV > {relative_tolerance:.1%}). Likely upstream drift or stale-code partitions. "
            f"First {len(sample)} examples:\n"
            f"{sample.to_string(index=False)}",
            stacklevel=2,
        )
    return aggregates.rename(columns={"actual_mean": "actual"})[AGGREGATE_OUTPUT_COLS]


def dedupe_actuals_from_partitions_df(
    df: pd.DataFrame,
    relative_tolerance: float = 1e-3,
    metric_col: str = "metric_id",
    time_col: str = "forecasted_date",
    actual_col: str = "actual",
    stage_col: str = "stage",
) -> pd.DataFrame:
    """Pandas core: aggregate partition rows in pandas, warn on inconsistency, return clean actuals.

    Filters ``df`` to ``stage_col == 'fitted'`` and ``actual_col`` non-null,
    groups by (``metric_col``, ``time_col``), computes mean / std / min / max
    / count, and pipes through :func:`warn_on_partition_inconsistency`.

    Args:
        df: Long-format partition rows (one row per cutoff × forecasted-date,
            with the same (metric, date) appearing many times). Must contain
            ``metric_col``, ``time_col``, ``actual_col``, ``stage_col``.
        relative_tolerance: Forwarded to :func:`warn_on_partition_inconsistency`.
        metric_col: Metric-identifier column name.
        time_col: Forecasted-date column name.
        actual_col: Actual-value column name.
        stage_col: Partition-stage column name (rows with ``"fitted"`` are kept).

    Returns:
        DataFrame with columns ``[metric_col, time_col, actual_col]`` — one
        row per (metric, date), holding the mean across partitions.
    """
    if df.empty:
        return pd.DataFrame(columns=[metric_col, time_col, actual_col])
    required = {metric_col, time_col, actual_col, stage_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}.")

    fitted = df[(df[stage_col] == "fitted") & df[actual_col].notna()]
    if fitted.empty:
        return pd.DataFrame(columns=[metric_col, time_col, actual_col])

    aggregates = (
        fitted.groupby([metric_col, time_col])[actual_col]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                metric_col: "metric_id",
                time_col: "forecasted_date",
                "mean": "actual_mean",
                "std": "actual_stddev",
                "min": "actual_min",
                "max": "actual_max",
                "count": "actual_partition_count",
            }
        )
    )
    aggregates["forecasted_date"] = pd.to_datetime(aggregates["forecasted_date"])
    cleaned = warn_on_partition_inconsistency(aggregates=aggregates, relative_tolerance=relative_tolerance)
    return cleaned.rename(columns={"metric_id": metric_col, "forecasted_date": time_col, "actual": actual_col})


def fetch_actuals_from_forecast_table(
    dc: DataContainer,
    io_param: Optional[IOParam] = None,
    relative_tolerance: float = 1e-3,
) -> pd.DataFrame:
    """DataContainer adapter — pandas mode → df core; SQL mode → cursor + SQL.

    The polymorphism lets a single call site work in both ``mint``-driven prod
    pipelines (``DataContainer(is_sql_table=True, table_name=...)``) and local
    / CloudNotebook notebook flows (``DataContainer(is_pandas_df=True, pandas_df=...)``).

    Args:
        dc: Source of the partition rows. Must be either pandas-mode
            (``is_pandas_df=True`` and ``pandas_df`` populated) or SQL-mode
            (``is_sql_table=True`` and ``table_name`` populated).
        io_param: Required when ``dc.is_sql_table`` — the SQL aggregation runs
            via ``io_param.cursor.get_df(...)``. Ignored in pandas mode.
        relative_tolerance: Forwarded to :func:`warn_on_partition_inconsistency`.

    Returns:
        DataFrame with columns ``metric_id``, ``forecasted_date``, ``actual``
        — one row per (metric, date), holding the mean across partitions.

    Raises:
        ValueError: if ``dc`` is neither pandas- nor SQL-mode; if pandas-mode
            ``pandas_df`` is None; if SQL-mode ``table_name`` is None or
            ``io_param`` / ``io_param.cursor`` is None.
    """
    if dc.is_pandas_df:
        if dc.pandas_df is None:
            raise ValueError("DataContainer.is_pandas_df=True but pandas_df is None.")
        return dedupe_actuals_from_partitions_df(df=dc.pandas_df, relative_tolerance=relative_tolerance)
    if dc.is_sql_table:
        if dc.table_name is None:
            raise ValueError("DataContainer.is_sql_table=True but table_name is None.")
        if io_param is None or io_param.cursor is None:
            raise ValueError("io_param.cursor is required for a SQL-mode DataContainer.")
        sql_result = io_param.cursor.get_df(ACTUALS_AGG_QUERY.format(table=dc.table_name))
        aggregates = sql_result.df
        if aggregates is None or aggregates.empty:
            return pd.DataFrame(columns=AGGREGATE_OUTPUT_COLS)
        aggregates = aggregates.copy()
        aggregates["forecasted_date"] = pd.to_datetime(aggregates["forecasted_date"])
        return warn_on_partition_inconsistency(aggregates=aggregates, relative_tolerance=relative_tolerance)
    raise ValueError(
        "DataContainer must be is_pandas_df=True (with pandas_df set) or "
        "is_sql_table=True (with table_name set)."
    )

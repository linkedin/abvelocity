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

from typing import Optional

import pandas as pd
from abvelocity.core.get_data.cursor import Cursor
from abvelocity.core.get_data.get_u_metrics_query_from_info import get_u_metrics_query_from_info
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.constants import METRIC_NAME_COL
from abvelocity.core.param.metric_info import MetricInfo

METRIC_FAMILY_COL = "metric_family"


def get_agg_metrics_query_from_info(metric_info: MetricInfo, start_date: str, end_date: str, condition: Optional[str] = None) -> str:
    """Constructs a query that computes aggregated metrics (across units).

    Builds on top of get_u_metrics_query_from_info by wrapping it to compute
    cross-unit aggregations based on metric definitions.

    For each metric, the query returns:
    - {metric_name}_numer: Aggregated numerator value
    - {metric_name}_denom: Aggregated denominator value (if denominator exists)
    - {metric_name}: Final metric value (numer/denom for ratios, or just numer for simple metrics)
    - {metric_name}_sample_count: Metric-specific sample size (if sample_count is specified)
    - sample_count: Overall sample size (COUNT(*) of all units)

    If dimensions are specified in the metric_family they are included in SELECT and GROUP BY.

    Args:
        metric_info: Contains the metrics, family, and dimension definitions.
        start_date: Start of the metric data window.
        end_date: End of the metric data window.
        condition: Optional SQL condition injected into the WHERE clause.

    Returns:
        SQL query string producing one row per dim combination (or one row when no dims).
    """
    u_metrics_query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date=start_date,
        end_date=end_date,
        condition=condition,
    )

    dims = metric_info.dims
    select_parts = []

    if dims is not None:
        select_parts.extend(dims)

    for metric in metric_info.metrics:
        metric_name = metric.name
        numer_col = metric.numerator.name
        numer_agg = metric.numerator_agg
        select_parts.append(f"{numer_agg}({numer_col}) AS {metric_name}_numer")

        if metric.denominator is not None:
            denom_col = metric.denominator.name
            denom_agg = metric.denominator_agg
            select_parts.append(f"{denom_agg}({denom_col}) AS {metric_name}_denom")
            select_parts.append(f"{numer_agg}({numer_col}) / NULLIF({denom_agg}({denom_col}), 0) AS {metric_name}")
        else:
            select_parts.append(f"{numer_agg}({numer_col}) AS {metric_name}")

        if metric.sample_count is not None:
            sample_count_col = metric.sample_count.name
            select_parts.append(f"SUM({sample_count_col}) AS {metric_name}_sample_count")

    select_parts.append("COUNT(*) AS sample_count")

    select_clause = ",\n    ".join(select_parts)

    group_by_clause = ""
    if dims is not None and len(dims) > 0:
        group_by_indices = [str(i + 1) for i in range(len(dims))]
        group_by_clause = f"\nGROUP BY {', '.join(group_by_indices)}"

    final_query = f"""
        SELECT
        {select_clause}
        FROM (
        {u_metrics_query}
        ) AS u_metrics{group_by_clause}
    """

    print(f"\n*** Agg metrics query constructed with {len(metric_info.metrics)} metrics")
    if dims:
        print(f"\n*** with dims: {dims}")
    return final_query


def normalize_family_df(
    raw_df: pd.DataFrame,
    family_name: str,
    metric_info: MetricInfo,
) -> pd.DataFrame:
    """Converts a wide-format query result into long-format with fixed columns.

    Each metric in ``metric_info.metrics`` becomes one row per raw_df row
    (one row per dim combination, or a single row when no dims are defined).

    Args:
        raw_df: Wide-format DataFrame from ``get_agg_metrics_query_from_info``.
            Columns follow the ``{metric}_numer``, ``{metric}_denom``, ``{metric}``,
            ``{metric}_sample_count`` (optional), ``sample_count`` convention.
        family_name: Written into the ``metric_family`` column.
        metric_info: Provides the list of metrics and dim columns to extract.

    Returns:
        Long-format DataFrame with columns:
        ``metric_family``, ``metric`` (METRIC_NAME_COL), ``numer``, ``denom``,
        ``value``, ``sample_count``, plus one column per dim (if dims are defined).
        ``denom`` is ``None`` for simple (non-ratio) metrics.
    """
    dims = metric_info.dims or []
    rows = []

    for _, raw_row in raw_df.iterrows():
        for metric in metric_info.metrics:
            metric_name = metric.name
            numer = raw_row.get(f"{metric_name}_numer")
            denom = raw_row.get(f"{metric_name}_denom")
            value = raw_row.get(metric_name)

            metric_sample_count_col = f"{metric_name}_sample_count"
            if metric_sample_count_col in raw_row.index:
                sample_count = raw_row[metric_sample_count_col]
            else:
                sample_count = raw_row.get("sample_count")

            row = {
                METRIC_FAMILY_COL: family_name,
                METRIC_NAME_COL: metric_name,
                "numer": numer,
                "denom": denom,
                "value": value,
                "sample_count": sample_count,
            }
            for dim in dims:
                row[dim] = raw_row.get(dim)

            rows.append(row)

    return pd.DataFrame(rows)


class AggMetricsQuery:
    """Builds and executes aggregated metric queries for each MetricInfo in an AnalysisInfo.

    Executes one query per MetricInfo (no experiment join — raw metrics over all data
    in the analysis window, or filtered by an optional ``condition``).

    Queries are stored in ``self.queries`` as ``{metric_family_name: sql_string}``.

    Usage::

        q = AggMetricsQuery(analysis_info)
        queries    = q.construct()             # builds and stores SQL strings
        df_dict    = q.get_pandas_df_dict(cursor)  # wide-format per family
        df         = q.get_pandas_df(cursor)       # normalized long-format
    """

    def __init__(self, analysis_info: AnalysisInfo, condition: Optional[str] = None):
        self.analysis_info = analysis_info
        self.condition = condition
        self.queries: Optional[dict] = None

    def construct(self) -> dict:
        """Builds one SQL query per MetricInfo and stores them in ``self.queries``.

        Returns:
            ``{metric_family_name: sql_string}``.
        """
        self.queries = {}
        for metric_info in self.analysis_info.metric_info_list or []:
            if not metric_info.metrics:
                continue
            family_name = metric_info.metric_family.name
            self.queries[family_name] = get_agg_metrics_query_from_info(
                metric_info=metric_info,
                start_date=self.analysis_info.start_date,
                end_date=self.analysis_info.end_date,
                condition=self.condition,
            )
        return self.queries

    def get_pandas_df_dict(self, cursor: Cursor) -> dict:
        """Executes all queries; returns ``{family_name: raw_wide_df}``.

        Calls ``construct()`` automatically if not already called.
        """
        if self.queries is None:
            self.construct()
        return {family_name: cursor.get_df(query).df for family_name, query in self.queries.items()}

    def get_pandas_df(self, cursor: Cursor) -> pd.DataFrame:
        """Executes all queries and returns a normalized long-format DataFrame.

        Calls ``get_pandas_df_dict`` then normalizes each wide-format result via
        ``normalize_family_df``.

        Returns:
            Long-format DataFrame with columns:
            ``metric_family``, ``metric``, ``numer``, ``denom``, ``value``,
            ``sample_count`` (plus any dim columns).
            ``denom`` is ``None`` for simple (non-ratio) metrics.
            Returns an empty DataFrame if no metrics are defined.
        """
        df_dict = self.get_pandas_df_dict(cursor)

        metric_info_by_family = {mi.metric_family.name: mi for mi in self.analysis_info.metric_info_list or [] if mi.metrics}
        normalized_dfs = [normalize_family_df(raw_df, family_name, metric_info_by_family[family_name]) for family_name, raw_df in df_dict.items()]
        if not normalized_dfs:
            return pd.DataFrame()
        return pd.concat(normalized_dfs, ignore_index=True)

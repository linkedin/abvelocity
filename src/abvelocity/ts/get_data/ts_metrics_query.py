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
"""Query builder for time-bucketed aggregated metric queries."""

from typing import Optional

import pandas as pd
from abvelocity.core.get_data.cursor import Cursor
from abvelocity.core.param.metric import get_u_metrics
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.utils.dim_collapser import DimCollapser
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig


class TSMetricsQuery:
    """Builds and executes a time-bucketed aggregated metric query.

    Wraps the two-layer SQL pattern used throughout ``get_data``:

    - **Inner query** (per-unit aggregation): generated via the
      ``MetricFamily.u_metrics_query`` mechanism, with ``time_col``
      injected as an extra grouping column so it is available in the
      subquery output.
    - **Outer query** (cross-unit aggregation per time bucket): applies
      ``DATE_TRUNC`` to bucket the timestamp, then aggregates numerator /
      denominator to produce the final metric value (including ratios) per
      ``(time_bucket × dims)``.

    The result is **wide-format** — one row per ``(time_bucket × dims)``,
    one column per metric value — directly usable as input to
    :class:`~abvelocity.ts.runner.TSRunner` by setting
    ``TSModelConfig.value_cols`` to the metric name columns.

    Usage::

        ts_cfg = TSMetricsConfig(time_col="event_ts", freq="D")
        q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_cfg)
        sql = q.construct()          # inspect or log the SQL
        df  = q.get_df(cursor)       # execute and return wide-format DataFrame

    Attributes:
        metric_info: Metric definitions, table info, dims, and date window.
        ts_config: Time-bucketing configuration (column, freq, dialect,
            optional custom expression, output alias).
        condition: Optional extra SQL ``WHERE`` condition appended to
            whatever ``metric_info.condition`` already specifies.
        query: SQL string populated by :meth:`construct`; ``None`` until
            :meth:`construct` is called.
    """

    def __init__(
        self,
        metric_info: MetricInfo,
        ts_config: TSMetricsConfig,
        condition: Optional[str] = None,
        dim_collapser: Optional[DimCollapser] = None,
    ) -> None:
        self.metric_info = metric_info
        self.ts_config = ts_config
        self.condition = condition
        self.dim_collapser = dim_collapser
        self.query: Optional[str] = None

    def construct(self) -> str:
        """Build the two-layer time-bucketed SQL query.

        Stores the result in ``self.query`` and returns it.

        The inner subquery is generated via ``MetricFamily.u_metrics_query``
        with ``time_col`` injected as an extra dim so it is available to the
        outer query.  The outer query applies ``DATE_TRUNC`` and computes
        final metric values (including ratio metrics) grouped by
        ``(time_bucket × dims)``.

        Returns:
            SQL query string.

        Raises:
            ValueError: If ``metric_info.start_date`` or ``metric_info.end_date``
                is ``None``, or if ``metric_info.metrics`` is empty.
        """
        metric_info = self.metric_info
        ts_config = self.ts_config

        if not metric_info.start_date or not metric_info.end_date:
            raise ValueError("metric_info.start_date and metric_info.end_date must be set.")
        if not metric_info.metrics:
            raise ValueError("metric_info.metrics must be non-empty.")

        metric_family = metric_info.metric_family
        outer_dims = metric_info.dims or []

        # ------------------------------------------------------------------ #
        # Merge conditions (caller condition + metric_info.condition)
        # ------------------------------------------------------------------ #
        condition = self.condition
        if metric_info.condition:
            condition = f"{condition} AND {metric_info.condition}" if condition else metric_info.condition

        # ------------------------------------------------------------------ #
        # Inner query: per-unit aggregation
        # time_col is injected as the first extra dim so the outer query can
        # reference it for DATE_TRUNC.  Existing dims follow after.
        # ------------------------------------------------------------------ #
        inner_dims = [ts_config.time_col] + outer_dims
        u_metrics = get_u_metrics(metric_info.metrics)

        inner_query = metric_family.u_metrics_query.construct(
            start_date=metric_info.start_date,
            end_date=metric_info.end_date,
            metric_join_unit_col=metric_family.metric_join_unit_col,
            u_metrics=u_metrics,
            condition=condition,
            dims=inner_dims,
            **metric_family.u_metrics_query_params,
        )

        # ------------------------------------------------------------------ #
        # Outer query: cross-unit aggregation per time bucket
        # ------------------------------------------------------------------ #
        # SELECT: time bucket first, then dims, then metric aggregations
        select_parts = [f"{ts_config.time_expr} AS {ts_config.time_alias}"]
        select_parts.extend(outer_dims)

        for metric in metric_info.metrics:
            metric_name = metric.name
            numer_col = metric.numerator.name
            numer_agg = metric.numerator_agg
            select_parts.append(f"{numer_agg}({numer_col}) AS {metric_name}_numer")

            if metric.denominator is not None:
                denom_col = metric.denominator.name
                denom_agg = metric.denominator_agg
                select_parts.append(f"{denom_agg}({denom_col}) AS {metric_name}_denom")
                select_parts.append(f"{numer_agg}({numer_col}) / NULLIF({denom_agg}({denom_col}), 0)" f" AS {metric_name}")
            else:
                select_parts.append(f"{numer_agg}({numer_col}) AS {metric_name}")

            if metric.sample_count is not None:
                select_parts.append(f"SUM({metric.sample_count.name}) AS {metric_name}_sample_count")

        select_parts.append("COUNT(*) AS sample_count")

        select_clause = ",\n    ".join(select_parts)

        # GROUP BY: time bucket (pos 1) + outer dims (pos 2, 3, ...)
        n_group_by = 1 + len(outer_dims)
        group_by_clause = ", ".join(str(i + 1) for i in range(n_group_by))

        self.query = f"""SELECT
    {select_clause}
FROM (
{inner_query}
) AS u_metrics
GROUP BY {group_by_clause}
ORDER BY 1"""

        return self.query

    def get_df(self, cursor: Cursor) -> pd.DataFrame:
        """Execute the query and return a wide-format DataFrame.

        Pipeline:

        1. Build SQL via :meth:`construct` (lazy).
        2. Execute via ``cursor.get_df``.
        3. Apply ``dim_collapser`` (if provided).
        4. Apply each ``ts_config.post_fetch_transforms`` in order.

        Regularization (parse dates, drop dups, fill missings, replace
        ±inf) is itself just a transform —
        :class:`~abvelocity.ts.get_data.transforms.Regularize` —
        and is the conventional first entry in
        ``post_fetch_transforms``.  ``TSMetricsQuery`` does not force it,
        so callers can opt out, swap, or chain a different regularizer.

        Args:
            cursor: Any :class:`~abvelocity.get_data.cursor.Cursor`
                subclass (Trino, Spark, DuckDB, etc.).

        Returns:
            Wide-format DataFrame with columns:
            ``{time_alias}``, ``{dims}``,
            ``{metric}_numer``, ``{metric}_denom`` (ratio metrics only),
            ``{metric}`` (final value), ``{metric}_sample_count`` (if defined),
            ``sample_count``.
            One row per ``(time_bucket × dims)``, ordered by time.
        """
        if self.query is None:
            self.construct()
        result = cursor.get_df(self.query)
        df = result.df
        if self.dim_collapser is not None:
            df = self.dim_collapser.apply(df)
        for transform in self.ts_config.post_fetch_transforms:
            df = transform.apply(df, self.ts_config, self.metric_info)
        return df

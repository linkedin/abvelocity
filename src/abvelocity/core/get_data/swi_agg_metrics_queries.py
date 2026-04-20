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
from abvelocity.core.get_data.agg_metrics_query import get_agg_metrics_query_from_info, normalize_family_df
from abvelocity.core.get_data.cursor import Cursor
from abvelocity.core.get_data.impacted_units_query import ImpactedUnitsQuery
from abvelocity.core.param.analysis_info import AnalysisInfo

SEGMENT_RAW = "raw"
SEGMENT_IMPACTED = "impacted"
SEGMENT_COMPLEMENT = "complement"


class SwiAggMetricsQueries:
    """Builds and executes three aggregated metric queries per MetricInfo for SWI.

    The impacted population is the UNION ALL of all units triggered by any experiment
    in ``analysis_info.multi_expt_info``. The impacted units query is built once and
    reused for all MetricInfos. Three queries are constructed per MetricInfo:

    - **raw** (``SEGMENT_RAW``): full metric table — no population filter.
    - **impacted** (``SEGMENT_IMPACTED``): units triggered by at least one experiment.
    - **complement** (``SEGMENT_COMPLEMENT``): units not triggered by any experiment.

    Queries are stored in ``self.queries`` as a nested dict keyed by metric family name
    and then by segment label::

        {metric_family_name: {SEGMENT_RAW: sql, SEGMENT_IMPACTED: sql, SEGMENT_COMPLEMENT: sql}}

    Usage::

        q = SwiAggMetricsQueries(analysis_info)
        queries = q.construct()       # builds and stores all SQL strings
        df = q.get_pandas_df(cursor)  # executes all queries, returns a single DataFrame
    """

    def __init__(self, analysis_info: AnalysisInfo):
        self.analysis_info = analysis_info
        self.queries: Optional[dict] = None
        self._metric_info_by_family: dict = {}

    def construct(self) -> dict:
        """Builds all segment queries and stores them in ``self.queries``.

        Returns:
            Nested dict ``{metric_family_name: {segment: sql_string}}``.

        Raises:
            ValueError: If ``analysis_info.metric_info_list`` is empty or None.
        """
        if not self.analysis_info.metric_info_list:
            raise ValueError("analysis_info.metric_info_list must not be empty.")

        start_date = self.analysis_info.start_date
        end_date = self.analysis_info.end_date

        # Built once; reused for all MetricInfos.
        impacted_units_query = ImpactedUnitsQuery(self.analysis_info.multi_expt_info).construct()

        self.queries = {}
        self._metric_info_by_family = {}
        for metric_info in self.analysis_info.metric_info_list:
            family_name = metric_info.metric_family.name
            self._metric_info_by_family[family_name] = metric_info
            metric_join_unit_col = metric_info.metric_family.metric_join_unit_col

            impacted_condition = f"{metric_join_unit_col} IN ({impacted_units_query})"
            complement_condition = f"{metric_join_unit_col} NOT IN ({impacted_units_query})"

            self.queries[family_name] = {
                SEGMENT_RAW: get_agg_metrics_query_from_info(metric_info, start_date, end_date),
                SEGMENT_IMPACTED: get_agg_metrics_query_from_info(metric_info, start_date, end_date, condition=impacted_condition),
                SEGMENT_COMPLEMENT: get_agg_metrics_query_from_info(metric_info, start_date, end_date, condition=complement_condition),
            }

        return self.queries

    def get_pandas_df_dict(self, cursor: Cursor) -> dict:
        """Executes all queries; returns raw wide-format results as a nested dict.

        Calls ``construct()`` automatically if not already called.

        Args:
            cursor: Any ``Cursor`` subclass (DuckDBCursor, PrestoCursor, etc.).

        Returns:
            ``{metric_family_name: {segment: raw_df}}`` where each ``raw_df`` is the
            wide-format query result (columns like ``{metric}_numer``, ``{metric}``,
            ``sample_count``).
        """
        if self.queries is None:
            self.construct()

        result: dict = {}
        for family_name, segment_queries in self.queries.items():
            result[family_name] = {}
            for segment, query in segment_queries.items():
                result[family_name][segment] = cursor.get_df(query).df

        return result

    def get_pandas_df(self, cursor: Cursor) -> pd.DataFrame:
        """Executes all queries and returns a single normalized long-format DataFrame.

        Calls ``get_pandas_df_dict`` then normalizes each wide-format result via
        ``_normalize_family_df``, appending a ``segment`` column before concatenating.

        Args:
            cursor: Any ``Cursor`` subclass (DuckDBCursor, PrestoCursor, etc.).

        Returns:
            Long-format DataFrame with columns:
            ``metric_family``, ``metric``, ``numer``, ``denom``, ``value``,
            ``sample_count``, ``segment`` (plus any dim columns).
            ``denom`` is ``None`` for simple (non-ratio) metrics.
        """
        df_dict = self.get_pandas_df_dict(cursor)

        normalized_dfs = []
        for family_name, segment_dfs in df_dict.items():
            metric_info = self._metric_info_by_family[family_name]
            for segment, raw_df in segment_dfs.items():
                norm_df = normalize_family_df(raw_df, family_name, metric_info)
                norm_df["segment"] = segment
                normalized_dfs.append(norm_df)

        return pd.concat(normalized_dfs, ignore_index=True)

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
from abvelocity.core.param.expt_info import MultiExptInfo


class ImpactedUnitsQuery:
    """Builds and executes a SQL query for the distinct unit IDs triggered by any experiment.

    Takes the UNION ALL of each experiment's assignment query in
    ``multi_expt_info.expt_info_list`` and selects the distinct unit IDs. A unit is
    considered impacted if it appears in the assignment data for any of the experiments.

    NULL unit IDs are filtered in each branch so that ``NOT IN`` conditions built on
    top of this query are safe.

    Note: ``expt_unit_col`` (from assignment data) and ``metric_join_unit_col`` (from
    metric data) may differ in column name but represent the same entity. When this
    query is embedded in an IN / NOT IN condition on the metric side, SQL compares
    values, so the name difference is handled transparently.

    Usage::

        q = ImpactedUnitsQuery(multi_expt_info)
        sql = q.construct()          # builds and stores the SQL string
        df = q.get_pandas_df(cursor) # executes and returns a DataFrame of unit IDs
    """

    def __init__(self, multi_expt_info: MultiExptInfo):
        self.multi_expt_info = multi_expt_info
        self.query: Optional[str] = None

    def construct(self) -> str:
        """Builds the UNION ALL query and stores it in ``self.query``.

        Returns:
            SQL query string with a single output column (``expt_unit_col``) containing
            distinct triggered unit IDs.

        Raises:
            ValueError: If ``expt_info_list`` is empty or ``expt_unit_col`` cannot be
                resolved consistently across all experiments.
        """
        expt_info_list = self.multi_expt_info.expt_info_list
        if not expt_info_list:
            raise ValueError("multi_expt_info.expt_info_list must not be empty.")

        expt_unit_col = _resolve_expt_unit_col(self.multi_expt_info)

        union_parts = []
        for i, expt_info in enumerate(expt_info_list):
            if expt_info.query is None:
                expt_info.gen_query()
            # Filter NULL unit IDs so NOT IN conditions on this subquery are safe.
            union_parts.append(f"SELECT {expt_unit_col}\nFROM (\n{expt_info.query}\n) AS expt_{i}" f"\nWHERE {expt_unit_col} IS NOT NULL")

        # UNION ALL is cheaper than UNION (no intermediate per-branch dedup).
        # The single outer SELECT DISTINCT handles deduplication across all experiments.
        union_body = "\nUNION ALL\n".join(union_parts)
        self.query = f"SELECT DISTINCT {expt_unit_col}\nFROM (\n{union_body}\n) AS impacted_units"
        return self.query

    def get_pandas_df(self, cursor: Cursor) -> pd.DataFrame:
        """Executes the query and returns a DataFrame of distinct impacted unit IDs.

        Calls ``construct()`` automatically if not already called.
        """
        if self.query is None:
            self.construct()
        return cursor.get_df(self.query).df


def _resolve_expt_unit_col(multi_expt_info: MultiExptInfo) -> str:
    """Returns the common ``expt_unit_col`` across all experiments in ``multi_expt_info``.

    Checks ``multi_expt_info.expt_unit_col`` first (which ``MultiExptInfo.__post_init__``
    propagates to individual ExptInfos). Falls back to checking that all individual
    ExptInfos share the same non-None value.

    Raises:
        ValueError: If a single consistent column name cannot be resolved.
    """
    if multi_expt_info.expt_unit_col is not None:
        return multi_expt_info.expt_unit_col

    unit_cols = {expt.expt_unit_col for expt in multi_expt_info.expt_info_list}
    unit_cols.discard(None)
    if len(unit_cols) != 1:
        raise ValueError(
            "All ExptInfo objects must share the same expt_unit_col for the impacted-population "
            f"union. Set it on MultiExptInfo or ensure all ExptInfos agree. Got: {unit_cols}"
        )
    return unit_cols.pop()

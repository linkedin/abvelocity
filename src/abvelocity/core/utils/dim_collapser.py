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
"""DimCollapser: keep top-K dim combinations by a metric, collapse the rest."""

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class DimCollapser:
    """Keep the top-K dimension combinations by total metric volume; collapse the rest.

    Works for any number of dimensions. All dim columns in a low-volume combination
    are replaced with ``fallback``, then rows are re-aggregated (summed) so the
    result has at most K+1 distinct dim combinations: the top K plus one "other" bucket.

    Args:
        dims: Column names to consider as a combined key (e.g. ``["country"]`` or
            ``["product", "platform"]``). Must match the dim columns in the DataFrame.
        k: Number of top combinations to keep.
        rank_by: Numeric column used to rank combinations (e.g. a metric value column).
        fallback: Label assigned to all dim columns for non-top-K rows. Defaults to ``"other"``.
        group_by: Additional columns to preserve during re-aggregation (e.g. a time column).
            Rows are grouped by ``group_by + dims`` when summing after collapse.

    Example::

        collapser = DimCollapser(dims=["region"], k=5, rank_by="signups", group_by=["date"])
        collapsed_df = collapser.apply(df)
    """

    dims: List[str]
    k: int
    rank_by: str
    fallback: str = "other"
    group_by: List[str] = field(default_factory=list)

    def top_k_combinations(self, df: pd.DataFrame) -> set:
        """Return the set of top-K dim combinations ranked by total ``rank_by``."""
        totals = df.groupby(self.dims)[self.rank_by].sum()
        top = totals.nlargest(self.k)
        # For a single dim, index is scalar values; for multiple dims, tuples.
        return set(top.index.tolist())

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse non-top-K dim combinations into ``fallback`` and re-aggregate.

        Args:
            df: Input DataFrame. Must contain all columns in ``dims``, ``rank_by``,
                and ``group_by``.

        Returns:
            DataFrame with the same numeric columns summed, dim combinations reduced
            to at most K+1 distinct values, ordered by ``group_by + dims``.
        """
        top_k = self.top_k_combinations(df)
        df = df.copy()

        if len(self.dims) == 1:
            dim = self.dims[0]
            mask = df[dim].isin(top_k)
            df.loc[~mask, dim] = self.fallback
        else:
            mask = pd.Series(
                [tuple(row) in top_k for row in df[self.dims].itertuples(index=False)],
                index=df.index,
            )
            df.loc[~mask, self.dims] = self.fallback

        group_cols = self.group_by + self.dims
        return df.groupby(group_cols, as_index=False).sum(numeric_only=True).sort_values(group_cols).reset_index(drop=True)

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
"""Base time-series result dataclass."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from abvelocity.core.utils.serialization import DataFrameConfig
from abvelocity.ts.result.ts_result_row import TSResultRow
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class TSResult(DataClassJSONMixin):
    """Base result from a time-series fit/predict run.

    The model object lives on the :class:`~abvelocity.ts.algo.base.TSAlgo`
    instance; this result contains only data artifacts.

    Two complementary views of the same output are exposed — pandas
    (``result_df``, the runtime-primary container) and typed rows
    (``rows``, useful for JSON / gRPC / test fixtures). Algorithms
    typically populate ``result_df``; callers needing a typed view can
    populate ``rows`` or build one ad-hoc via
    :meth:`TSResultRow.from_df`.

    Attributes:
        result_df: Long-format output DataFrame; the canonical schema is
            defined by :class:`~abvelocity.ts.result.ts_result_row.TSResultRow`.
            One row per (timestamp × metric_id × dim-combination).
            :class:`~abvelocity.ts.runner.TSRunner` stamps the
            full set of scheduled-pipeline columns after the algo returns.
        rows: Optional typed view of the same data. Not auto-synced with
            ``result_df`` — callers set whichever representation they
            produced; conversion via :meth:`TSResultRow.from_df` /
            :meth:`TSResultRow.to_df` is explicit when needed.
        fit_info: JSON-serializable diagnostics dictionary produced during the
            fit step.
    """

    result_df: Optional[pd.DataFrame] = None
    """Output DataFrame whose schema matches :class:`TSResultRow`."""

    rows: Optional[List[TSResultRow]] = None
    """Optional typed row view (for JSON / gRPC / tests)."""

    fit_info: Optional[Dict[str, Any]] = None
    """JSON-serializable diagnostics from the fit step."""

    class Config(DataFrameConfig):
        pass

    # -------------------------------------------------------------------
    # Views / constructors — both views of the same data reachable from
    # either input. Not auto-synced on mutation; convert on demand.
    # -------------------------------------------------------------------

    def to_df(self) -> Optional[pd.DataFrame]:
        """Return a pandas view of the result.

        If ``result_df`` is populated, returns it directly (no copy).
        Otherwise builds one from ``rows``. Returns ``None`` when
        neither is populated.
        """
        if self.result_df is not None:
            return self.result_df
        if self.rows is not None:
            return _rows_to_df(self.rows)
        return None

    def to_rows(self) -> List[TSResultRow]:
        """Return a typed-row view of the result.

        If ``rows`` is populated, returns it directly. Otherwise builds
        one from ``result_df``. Returns an empty list when neither is
        populated.
        """
        if self.rows is not None:
            return self.rows
        if self.result_df is not None and not self.result_df.empty:
            return _df_to_rows(self.result_df)
        return []

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        fit_info: Optional[Dict[str, Any]] = None,
    ) -> "TSResult":
        """Construct a :class:`TSResult` whose ``result_df`` is the
        given DataFrame."""
        return cls(result_df=df, fit_info=fit_info)

    @classmethod
    def from_rows(
        cls,
        rows: List[TSResultRow],
        fit_info: Optional[Dict[str, Any]] = None,
    ) -> "TSResult":
        """Construct a :class:`TSResult` whose ``rows`` is the given
        list (and ``result_df`` is left ``None`` — call :meth:`to_df`
        to materialize it on demand)."""
        return cls(rows=rows, fit_info=fit_info)


# ------------------------------------------------------------------------
# Private collection-level converters.
# Not public API — callers go through TSResult.to_df / to_rows.
# ------------------------------------------------------------------------


def _rows_to_df(rows: List[TSResultRow]) -> pd.DataFrame:
    """Build a DataFrame (canonical column order + dtypes) from a list
    of :class:`TSResultRow`. Returns an empty DataFrame with the right
    columns + dtypes when ``rows`` is empty."""
    columns = list(TSResultRow.columns())
    dtypes = TSResultRow.dtypes()
    if not rows:
        empty = pd.DataFrame({c: pd.Series([], dtype=dtypes[c]) for c in columns})
        return empty
    df = pd.DataFrame([asdict(r) for r in rows])
    return df[columns].astype(dtypes)


def _df_to_rows(df: pd.DataFrame) -> List[TSResultRow]:
    """Build a list of :class:`TSResultRow` from a DataFrame. Iterates
    per row; meant for JSON / gRPC / test-fixture boundaries, not for
    hot-path bulk transforms (use the DataFrame directly for those)."""
    subset = df[list(TSResultRow.columns())]
    return [TSResultRow.from_series(row) for _, row in subset.iterrows()]

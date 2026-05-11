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
"""Time-series data hygiene — both the function form
(:func:`regularize_timeseries`) and the transform form
(:class:`Regularize`) live here.

The hygiene pipeline mirrors greykite's ``get_canonical_data`` minus
modeling-layer concerns (anomaly adjustment, regressor / lagged-regressor
handling, ``fit_df`` construction, ``train_end_date`` derivation,
timezone localization).

Lives in its own module — separate from the lookup-flavored helpers in
:mod:`time_properties` (``describe_timeseries``, ``find_missing_dates``,
``infer_freq``) and from the algebraic transforms in
:mod:`transforms` (``Coarsen``, ``Diff``, ``WeightWithinPeriod``).
Regularization is the conventional FIRST entry in a transform chain;
its own module keeps that role legible.

Note
----
Function modeled on
:func:`abvelocity.ts.common.time_properties.get_canonical_data`,
slimmed of anomaly / regressor / fit_df / train_end_date logic.
"""

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
import pandas as pd

from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.ts.constants import TIME_COL
from abvelocity.ts.common.time_properties import fill_missing_dates, infer_freq
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig
from abvelocity.ts.get_data.ts_transform import TSTransform


def regularize_timeseries(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    value_cols: list = None,
    freq: str = None,
) -> pd.DataFrame:
    """Return a regularized version of ``df``: parsed timestamps, no
    duplicates, no ±inf, sorted, and gaps filled at the inferred (or
    given) freq.

    Steps (mirror greykite's ``get_canonical_data`` minus the modeling
    bits):

    1. Coerce ``time_col`` to ``datetime64[ns]`` via ``pd.to_datetime``.
    2. Drop duplicate timestamps (keep first); warn if any were dropped.
    3. Sort ascending by ``time_col``.
    4. Infer freq when not provided; warn if a provided freq disagrees
       with the inferred one.
    5. Pad missing time-buckets via :func:`fill_missing_dates`.
    6. Replace ±inf in ``value_cols`` (or every numeric column when
       ``value_cols`` is ``None``) with ``NaN``.

    Args:
        df: Input DataFrame.
        time_col: Time column name.
        value_cols: Numeric columns whose ±inf should be NaN-replaced.
            Pass ``None`` to apply to every numeric column.
        freq: pandas DateOffset alias (``"D"``, ``"W"``, ``"h"``, ...).
            Inferred from data when ``None``.

    Returns:
        Regularized DataFrame.  Length differs from the input when
        duplicates were dropped or gaps were filled.
    """
    if df.empty:
        return df.copy()

    out_df = df.copy()
    out_df[time_col] = pd.to_datetime(out_df[time_col])

    n_before = len(out_df)
    out_df = out_df.drop_duplicates(subset=[time_col], keep="first")
    if len(out_df) < n_before:
        warnings.warn(f"Dropped {n_before - len(out_df)} duplicate timestamps.", UserWarning)

    out_df = out_df.sort_values(by=time_col).reset_index(drop=True)

    inferred_freq = infer_freq(out_df, time_col)
    if freq is None:
        freq = inferred_freq
    elif inferred_freq is not None and freq != inferred_freq:
        warnings.warn(
            f"Provided frequency {freq!r} does not match inferred frequency " f"{inferred_freq!r}. Using {freq!r}.",
            UserWarning,
        )

    out_df, _, _ = fill_missing_dates(out_df, time_col=time_col, freq=freq)

    cols_to_clean = value_cols if value_cols is not None else [c for c in out_df.columns if c != time_col and pd.api.types.is_numeric_dtype(out_df[c])]
    for col in cols_to_clean:
        if col in out_df.columns:
            out_df[col] = out_df[col].replace([np.inf, -np.inf], np.nan)

    return out_df


@dataclass(frozen=True)
class Regularize(TSTransform):
    """Apply :func:`regularize_timeseries` over a wide-format metrics
    frame, optionally per-dim.

    The conventional first entry in a transform chain: every downstream
    transform (Coarsen, WeightWithinPeriod, Diff) assumes a continuous
    time axis with parsed timestamps and no ±inf.  With dims, each
    segment is regularized against its own observed time range.

    Args:
        freq: Pandas DateOffset alias for gap-filling.  Defaults to
            ``ts_config.freq`` when ``None``.
    """

    freq: Optional[str] = None

    def apply(
        self,
        df: pd.DataFrame,
        ts_config: TSMetricsConfig,
        metric_info: MetricInfo,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        time_col = ts_config.time_alias
        freq = self.freq or ts_config.freq
        value_cols = [m.name for m in (metric_info.metrics or []) if m.name in df.columns] or None

        dims = [d for d in (metric_info.dims or []) if d in df.columns]
        if not dims:
            return regularize_timeseries(df, time_col=time_col, value_cols=value_cols, freq=freq)

        pieces = []
        for _, group in df.groupby(dims, dropna=False, sort=False):
            cleaned = regularize_timeseries(group, time_col=time_col, value_cols=value_cols, freq=freq)
            for d in dims:
                # Forward-fill dim labels onto the newly-padded rows so
                # downstream groupbys still see the right segment.
                cleaned[d] = group[d].iloc[0]
            pieces.append(cleaned)
        return pd.concat(pieces, ignore_index=True)

    def str_name(self) -> str:
        return ""  # Regularize doesn't change the metric semantics.

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
"""Concrete post-fetch transforms (``Coarsen``, ``WeightWithinPeriod``,
``Diff``) and the lag-math helpers they share.

The :class:`~abvelocity.ts.get_data.ts_transform.TSTransform` ABC
lives in :mod:`ts_transform`; data-hygiene regularization
(:class:`~abvelocity.ts.get_data.regularizer.Regularize`) lives
in :mod:`regularizer` — both kept separate so the dependency graph stays
a DAG.

Why pandas instead of SQL window functions? Diffing across dialects and
stacking multiple transforms in a single SQL emit gets ugly fast.  At
our scale data-over-wire is not the bottleneck.

The helpers in this module (``PERIOD_DIVISOR``, ``rows_per_period``,
``lag_orders_to_rows``, ``compute_lag_values``) are shared across the
concrete transforms and intentionally public so other consumers can
reuse them outside the forecasting context.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig
from abvelocity.ts.get_data.ts_transform import TSTransform

ALLOWED_AGGS = ("mean", "median", "max", "min", "sum")
"""Closed enum of aggregation names accepted by transforms.

Keep this as strings — callable aggs have not earned their complexity."""


PERIOD_DIVISOR: dict[tuple[str, str], int] = {
    ("D", "W"): 7,
    ("h", "D"): 24,
    ("h", "W"): 24 * 7,
    ("min", "h"): 60,
    ("min", "D"): 60 * 24,
    ("s", "min"): 60,
    ("s", "h"): 3600,
}
"""How many ``source_freq`` rows fit in one ``target_period``.

Keys are ``(source_freq, target_period)`` pairs.  Months / quarters are
intentionally excluded — variable length means a different completeness
rule (count expected days in the actual month).  Add when needed.
"""


def rows_per_period(source_freq: str, target_period: str) -> int:
    """Return rows-per-period for a (source, target) pair, or raise.

    Raises ``ValueError`` for unsupported / variable-length pairs (M, Q)
    so callers don't silently get wrong completeness checks.

    Anchored week aliases (``"W-SAT"``, ``"W-SUN"``, ...) are normalized
    to their base (``"W"``) before lookup — anchor controls the period
    *boundary*, not the row count (a week is always 7 days).
    """
    source_base = source_freq.split("-", 1)[0]
    target_base = target_period.split("-", 1)[0]
    if source_base == target_base:
        return 1
    try:
        return PERIOD_DIVISOR[(source_base, target_base)]
    except KeyError:
        raise ValueError(
            f"Unsupported (source_freq, target_period) pair: " f"({source_freq!r}, {target_period!r}). Supported pairs: " f"{sorted(PERIOD_DIVISOR.keys())}."
        )


def lag_orders_to_rows(
    lag_period_orders: Sequence[int],
    source_freq: str,
    lag_period: str,
) -> list[int]:
    """Translate orders measured in ``lag_period`` units into row counts at ``source_freq``.

    Example: at daily freq, weekly orders ``[1, 2, 3]`` translate to row
    offsets ``[7, 14, 21]``.  Useful when a transform needs to work on
    row positions (e.g., greykite-style row-shift) rather than dates.
    """
    multiplier = rows_per_period(source_freq, lag_period)
    return [k * multiplier for k in lag_period_orders]


FIXED_LAG_DELTAS = {
    "D": ("days", 1),
    "W": ("weeks", 1),
    "h": ("hours", 1),
    "H": ("hours", 1),
    "min": ("minutes", 1),
    "T": ("minutes", 1),
    "s": ("seconds", 1),
    "S": ("seconds", 1),
}
"""Lag periods with fixed durations — translated to ``pd.Timedelta``.

These can be added to ``DatetimeIndex`` values without ambiguity.
"""

VARIABLE_LAG_DELTAS = {
    "M": ("months", 1),
    "MS": ("months", 1),
    "ME": ("months", 1),
    "Q": ("months", 3),
    "QS": ("months", 3),
    "QE": ("months", 3),
    "Y": ("years", 1),
    "YS": ("years", 1),
    "YE": ("years", 1),
    "A": ("years", 1),
}
"""Lag periods with variable real-time duration — translated to
``pd.DateOffset``, which preserves calendar semantics (``2024-03-31 -
DateOffset(months=1) == 2024-02-29``).
"""


def to_lag_delta(lag_period: str):
    """Return a ``Timedelta`` (fixed) or ``DateOffset`` (calendar) for one
    unit of ``lag_period``.

    Multiply by an integer to step ``k`` periods.
    """
    if lag_period in FIXED_LAG_DELTAS:
        kw, n = FIXED_LAG_DELTAS[lag_period]
        return pd.Timedelta(**{kw: n})
    if lag_period in VARIABLE_LAG_DELTAS:
        kw, n = VARIABLE_LAG_DELTAS[lag_period]
        return pd.DateOffset(**{kw: n})
    raise ValueError(f"Unsupported lag_period {lag_period!r}. " f"Supported: {sorted(set(FIXED_LAG_DELTAS) | set(VARIABLE_LAG_DELTAS))}.")


def compute_lag_values(
    series: pd.Series,
    lag_period: str,
    lag_period_orders: Sequence[int],
    agg: Optional[str] = None,
) -> pd.Series:
    """Date-aware lookup of past values, optionally aggregated across multiple lags.

    For each timestamp ``t`` in ``series.index``, the row's lag-``k`` value is
    ``series.loc[t - k * delta]`` (or NaN when that timestamp is absent).
    Multiple orders combine via ``agg``.

    The lookup uses an explicit timestamp difference (not ``Series.shift(freq=…)``,
    which snaps daily indices to period boundaries — wrong here).

    Args:
        series: Value series indexed by a ``DatetimeIndex`` (need not be sorted).
        lag_period: Period unit; one of the keys of :data:`FIXED_LAG_DELTAS`
            (``"D"``, ``"W"``, ``"h"``, ``"min"``, ``"s"``) or
            :data:`VARIABLE_LAG_DELTAS` (``"M"``/``"MS"``/``"ME"``, ``"Q"``,
            ``"Y"``).
        lag_period_orders: Multipliers applied to the unit delta.  Must be
            non-empty, all-positive.
        agg: One of :data:`ALLOWED_AGGS`.  Required when more than one
            order is supplied; ignored otherwise.

    Returns:
        Series aligned to ``series.index``.

    Raises:
        ValueError: Empty / non-positive orders, unknown ``agg`` or
            ``lag_period``, or multi-order call without ``agg``.
        TypeError: ``series`` not indexed by a ``DatetimeIndex``.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must be indexed by a DatetimeIndex.")
    if not lag_period_orders:
        raise ValueError("lag_period_orders must be non-empty.")
    if any(k <= 0 for k in lag_period_orders):
        raise ValueError(f"All lag_period_orders must be positive; got {list(lag_period_orders)!r}.")
    if len(lag_period_orders) > 1 and agg is None:
        raise ValueError("agg is required when len(lag_period_orders) > 1.")
    if agg is not None and agg not in ALLOWED_AGGS:
        raise ValueError(f"agg must be one of {ALLOWED_AGGS}; got {agg!r}.")

    delta = to_lag_delta(lag_period)
    lagged_cols: dict[str, pd.Series] = {}
    for k in lag_period_orders:
        lookup_dates = series.index - k * delta
        # `series.reindex(lookup_dates)` returns values keyed by lookup_dates.
        # We want those values aligned to ``series.index`` positions, so reuse
        # the array directly.
        looked_up = series.reindex(lookup_dates).to_numpy()
        lagged_cols[f"lag_{k}"] = pd.Series(looked_up, index=series.index)

    lagged = pd.DataFrame(lagged_cols, index=series.index)
    if len(lag_period_orders) == 1:
        return lagged.iloc[:, 0]
    # Strict: if ANY lag lookup is NaN the baseline is NaN.  Otherwise
    # "mean of last 3 weeks" with only 1 week available silently degrades
    # to "mean of 1 week" — a misleading baseline.
    aggregated = lagged.agg(agg, axis=1)
    aggregated[lagged.isna().any(axis=1)] = float("nan")
    return aggregated


# ---------------------------------------------------------------------------
# Transform framework
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coarsen(TSTransform):
    """Aggregate a fine-grained series into coarser buckets (e.g., D → W).

    Groups by ``time_alias`` truncated to ``freq`` (plus any dims) and
    applies ``agg`` to all non-group columns.  All metric columns receive
    the *same* aggregation — pass one Coarsen per agg style if a pipeline
    needs different rules per metric.

    Args:
        freq: Target frequency alias (``"W"``, ``"MS"``, ``"QS"``, ...).
        agg: One of :data:`ALLOWED_AGGS`.  Default ``"sum"``.
    """

    freq: str
    agg: str = "sum"

    def __post_init__(self) -> None:
        if self.agg not in ALLOWED_AGGS:
            raise ValueError(f"agg must be one of {ALLOWED_AGGS}; got {self.agg!r}.")

    def apply(
        self,
        df: pd.DataFrame,
        ts_config: TSMetricsConfig,
        metric_info: MetricInfo,
    ) -> pd.DataFrame:
        time_col = ts_config.time_alias
        dims = list(metric_info.dims or [])

        out_df = df.copy()
        # End-of-period anchor (e.g. the Saturday that closes a Sun-Sat
        # week).  Picked over ``start_time`` because pandas' freq-inference
        # on a Sunday-anchored series returns ``"W-SUN"`` (week ending
        # Sun), which collides with a ``"W-SAT"`` model config — greykite
        # then drops every row when re-regularizing.  ``.dt.normalize()``
        # snaps end_time's 23:59:59.999 to 00:00:00 of the same day.
        out_df[time_col] = pd.to_datetime(out_df[time_col]).dt.to_period(self.freq).dt.end_time.dt.normalize()

        group_cols = [time_col] + [d for d in dims if d in out_df.columns]
        agg_cols = [c for c in out_df.columns if c not in group_cols]
        if not agg_cols:
            return out_df.drop_duplicates(subset=group_cols).reset_index(drop=True)

        # Numeric columns aggregate; non-numeric (string dims that snuck
        # in as values) take "first" — same row should have identical
        # values once group_cols are correct.
        numeric_cols = [c for c in agg_cols if pd.api.types.is_numeric_dtype(out_df[c])]
        non_numeric = [c for c in agg_cols if c not in numeric_cols]

        agg_map: dict[str, str] = {c: self.agg for c in numeric_cols}
        agg_map.update({c: "first" for c in non_numeric})

        grouped_df = out_df.groupby(group_cols, dropna=False, as_index=False).agg(agg_map)
        return grouped_df.sort_values(group_cols).reset_index(drop=True)

    def str_name(self) -> str:
        # Common abbreviations get readable suffixes; everything else falls
        # back to the literal freq alias.
        freq_to_word = {"W": "weekly", "D": "daily", "MS": "monthly", "ME": "monthly", "M": "monthly"}
        word = freq_to_word.get(self.freq, self.freq.lower())
        return word if self.agg == "sum" else f"{word}_{self.agg}"


@dataclass(frozen=True)
class WeightWithinPeriod(TSTransform):
    """Replace each metric value with its share of the enclosing period's total.

    For a daily series with ``period="W"``, each Mon..Sun row becomes
    ``daily_value / weekly_total``.  Within a complete period the weights
    sum to 1.

    Rows in **incomplete** periods (fewer rows than expected for the
    source freq) are marked NaN so downstream forecasting doesn't train on
    partial-week distributions.

    Args:
        period: Coarser period alias — must form a supported pair with
            the source freq in :data:`PERIOD_DIVISOR`.
    """

    period: str

    def apply(
        self,
        df: pd.DataFrame,
        ts_config: TSMetricsConfig,
        metric_info: MetricInfo,
    ) -> pd.DataFrame:
        time_col = ts_config.time_alias
        expected_rows = rows_per_period(ts_config.freq, self.period)

        out_df = df.copy()
        out_df[time_col] = pd.to_datetime(out_df[time_col])

        dims = [d for d in (metric_info.dims or []) if d in out_df.columns]
        bucket_col = "__weight_bucket__"
        out_df[bucket_col] = out_df[time_col].dt.to_period(self.period).dt.start_time
        group_cols = [bucket_col] + dims

        value_cols = [m.name for m in (metric_info.metrics or []) if m.name in out_df.columns and pd.api.types.is_numeric_dtype(out_df[m.name])]

        gb = out_df.groupby(group_cols, dropna=False)
        for col in value_cols:
            sums = gb[col].transform("sum")
            counts = gb[col].transform("size")
            complete = counts == expected_rows
            denom = sums.where(sums != 0)  # divide-by-zero → NaN
            out_df[col] = (out_df[col] / denom).where(complete)

        return out_df.drop(columns=[bucket_col])

    def str_name(self) -> str:
        period_word = {"W": "week", "D": "day", "MS": "month", "M": "month", "ME": "month", "h": "hour"}
        word = period_word.get(self.period, self.period.lower())
        return f"within_{word}_weight"


LAG_PERIOD_SHORT = {
    "D": "d",
    "W": "w",
    "h": "h",
    "H": "h",
    "min": "min",
    "T": "min",
    "s": "s",
    "S": "s",
    "M": "m",
    "MS": "m",
    "ME": "m",
    "Q": "q",
    "QS": "q",
    "QE": "q",
    "Y": "y",
    "YS": "y",
    "YE": "y",
    "A": "y",
}
"""One-letter lag-period codes used in suffix() (``W → "w" → "wow_diff"``)."""


@dataclass(frozen=True)
class Diff(TSTransform):
    """Subtract a date-aware baseline from each value in the metric columns.

    The baseline is :func:`compute_lag_values` applied per value column;
    rows whose lookup target falls outside the data become NaN diffs by
    construction.

    Patterns::

        Diff(lag_period="D", n_lag_periods=1)                          # day-over-day
        Diff(lag_period="W", n_lag_periods=1)                          # week-over-week
        Diff(lag_period="W", n_lag_periods=3, agg="median")            # value − median(weeks 1, 2, 3)
        Diff(lag_period="W", lag_period_orders=[1, 2, 4], agg="mean")  # explicit orders
        Diff(lag_period="W", n_lag_periods=3, agg="mean", relative=True)
                                                                       # (value − base) / base

    Args:
        lag_period: Period unit; must be a key of
            :data:`FIXED_LAG_DELTAS` or :data:`VARIABLE_LAG_DELTAS`.
        n_lag_periods: Use orders ``[1..n_lag_periods]``.  Mutually
            exclusive with ``lag_period_orders``.
        lag_period_orders: Explicit list of period offsets to look up.
        agg: One of :data:`ALLOWED_AGGS`.  Required when the resolved
            order count > 1.
        relative: When ``True``, return ``(value − baseline) / baseline``
            instead of the raw difference (NaN where baseline == 0).
    """

    lag_period: str
    n_lag_periods: Optional[int] = None
    lag_period_orders: Optional[Sequence[int]] = None
    agg: Optional[str] = None
    relative: bool = False

    def __post_init__(self) -> None:
        if (self.n_lag_periods is None) == (self.lag_period_orders is None):
            raise ValueError(
                "Exactly one of n_lag_periods / lag_period_orders must be set "
                f"(got n_lag_periods={self.n_lag_periods!r}, "
                f"lag_period_orders={self.lag_period_orders!r})."
            )
        if self.n_lag_periods is not None and self.n_lag_periods < 1:
            raise ValueError(f"n_lag_periods must be >= 1; got {self.n_lag_periods!r}.")

        orders = self.resolved_orders()
        if len(orders) > 1 and self.agg is None:
            raise ValueError("agg is required when there is more than one lag.")
        if self.agg is not None and self.agg not in ALLOWED_AGGS:
            raise ValueError(f"agg must be one of {ALLOWED_AGGS}; got {self.agg!r}.")

    def resolved_orders(self) -> list[int]:
        if self.n_lag_periods is not None:
            return list(range(1, self.n_lag_periods + 1))
        return list(self.lag_period_orders)

    def apply(
        self,
        df: pd.DataFrame,
        ts_config: TSMetricsConfig,
        metric_info: MetricInfo,
    ) -> pd.DataFrame:
        time_col = ts_config.time_alias
        out = df.copy()
        out[time_col] = pd.to_datetime(out[time_col])

        dims = [d for d in (metric_info.dims or []) if d in out.columns]
        value_cols = [m.name for m in (metric_info.metrics or []) if m.name in out.columns and pd.api.types.is_numeric_dtype(out[m.name])]

        orders = self.resolved_orders()

        # Compute per-(dim) so each segment's baseline uses its own history.
        if dims:
            for col in value_cols:
                pieces = []
                for _, group in out.groupby(dims, dropna=False, sort=False):
                    indexed = pd.Series(group[col].to_numpy(dtype=float), index=group[time_col])
                    baseline = compute_lag_values(indexed, self.lag_period, orders, agg=self.agg)
                    pieces.append(self.subtract_baseline(indexed, baseline, group.index))
                out[col] = pd.concat(pieces).reindex(out.index)
        else:
            for col in value_cols:
                indexed = pd.Series(out[col].to_numpy(dtype=float), index=out[time_col])
                baseline = compute_lag_values(indexed, self.lag_period, orders, agg=self.agg)
                out[col] = self.subtract_baseline(indexed, baseline, out.index).reindex(out.index)

        return out

    def subtract_baseline(
        self,
        values: pd.Series,
        baseline: pd.Series,
        out_index: pd.Index,
    ) -> pd.Series:
        """Return ``values - baseline`` (or the relative form ``(v - b) / b``)
        as a Series indexed by ``out_index``.

        Zero-baseline rows yield NaN in the relative form (not ±inf).
        """
        v = values.to_numpy(dtype=float)
        b = baseline.to_numpy(dtype=float)
        diff = v - b
        if self.relative:
            safe = np.where(b == 0, np.nan, b)
            return pd.Series(diff / safe, index=out_index)
        return pd.Series(diff, index=out_index)

    def str_name(self) -> str:
        short = LAG_PERIOD_SHORT.get(self.lag_period, self.lag_period.lower())
        rel = "rel_" if self.relative else ""

        if self.n_lag_periods is not None:
            n = self.n_lag_periods
            if n == 1:
                return f"{rel}{short}o{short}_diff"
            return f"{rel}{self.agg}_{n}{short}_diff"

        # explicit orders path
        orders = list(self.lag_period_orders or [])
        if len(orders) == 1:
            return f"{rel}{short}o{short}_diff"
        return f"{rel}{self.agg}_{len(orders)}{short}_diff"

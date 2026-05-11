# Original author: Reza Hosseini
"""Aggregation primitives — small, generic helpers used by the
two-pass + join structure of every transform's ``apply()``.

Two flavors:

- **Sum side**: aggregate forecast-bearing columns
  (forecast / sigma / bounds / breakdown) on one pass and the
  ``actual`` column on a separate pass, then join the two reduced
  frames on the group key.
- **Share side**: rows aren't reduced; reweight forecast-bearing
  columns and ``actual`` independently in place.

Both sides treat ``actual`` minimally — it has no variance, no bounds,
no breakdown.  Keeping the actual path skinny is the whole point
of the split.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from abvelocity.ts.constants import (
    ACTUAL_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    STAGE_COL,
    STD_COL,
)
from abvelocity.ts.forecast_transforms.bounds import recompute_bounds
from abvelocity.ts.forecast_transforms.column_classes import (
    BREAKDOWN_COLS,
    METADATA_COLS,
    POINT_SUMMABLE_COLS,
    SIGMA_COLS,
)
from abvelocity.ts.forecast_transforms.sigma import (
    propagate_sigma_indep_sum,
    propagate_sigma_share,
    propagate_sigma_share_delta,
)


SIGMA_METHODS = ("constant", "delta")
"""Allowed values for the share-side ``sigma_method`` parameter.

- ``"constant"``: denominator treated as known constant.  σ(w) = σ_X / |Y|.
- ``"delta"``: full delta method with denominator uncertainty
  propagated.  σ(w) = √(σ_X² + w² σ_Y²) / |Y|.
"""


def sum_strict(values: pd.Series) -> float:
    """Sum ``values``, returning NaN when *any* entry is NaN.

    All-or-nothing semantics: a period bucket either has every input
    row populated (real sum) or it doesn't (NaN).  No partial sums —
    a Mon-Wed slice of actuals from a Sun-Sat week shouldn't add up
    as if it were a full week's actual.

    Differs from both:

    - Pandas' default ``Series.sum()`` (which returns ``0`` for
      all-NaN, and ignores NaNs giving partial sums for mixed input).
    - ``Series.sum(min_count=1)`` (which returns NaN only when every
      entry is NaN; still gives partial sums for mixed input).

    Args:
        values: Numeric Series to sum.

    Returns:
        The sum, or ``NaN`` when any entry is NaN.
    """
    if values.isna().any():
        return float("nan")
    return float(values.sum())


def grouping_columns(forecast_df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Identity / non-numeric columns that act as the natural group key
    for aggregation.

    Returns every non-numeric column that isn't in ``exclude`` *or* in
    :data:`~abvelocity.ts.forecast_transforms.column_classes.METADATA_COLS`
    (TSRunner-stamped metadata like ``ts``, ``run_id``, etc., which are
    never grouping keys — see that constant for rationale).

    Args:
        forecast_df: The forecast frame.
        exclude: Additional columns to leave OUT of the group key
            (typically the timestamp column when the caller has already
            replaced it with a period anchor).

    Returns:
        Stable-ordered list of grouping column names.
    """
    excluded = set(exclude) | set(METADATA_COLS)
    return [c for c in forecast_df.columns if c not in excluded and not pd.api.types.is_numeric_dtype(forecast_df[c])]


# ---------------------------------------------------------------------------
# Sum side — reduce rows.  Two passes, then join.
# ---------------------------------------------------------------------------


def aggregate_forecast_side_sum(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    ci_coverage: float,
) -> pd.DataFrame:
    """Sum-aggregate the forecast-bearing columns by ``group_cols``.

    Columns processed:

    - **Decomposition + ``forecast``**: ``sum_strict``.
    - **``std``**: independent-Gaussian propagation
      (:func:`propagate_sigma_indep_sum`).
    - **Bounds (``forecast_lower`` / ``forecast_upper``)**: dropped
      from the agg, recomputed downstream from new forecast ± z·new_σ.

    The ``actual`` column is NOT touched — the actual side is its own
    pass.

    Args:
        df: Input frame (already prepped: stage dropped, period columns
            attached, anything else the caller wants in ``group_cols``).
        group_cols: Columns to group by (timestamp + identity + dims).
        ci_coverage: Two-sided coverage for bound recomputation.

    Returns:
        Reduced frame keyed by ``group_cols`` with forecast-side columns
        aggregated and bounds recomputed.  No ``actual`` column.
    """
    forecast_cols = [col for col in (*POINT_SUMMABLE_COLS, *SIGMA_COLS) if col != ACTUAL_COL and col in df.columns]
    if not forecast_cols:
        return df.loc[:, list(group_cols)].drop_duplicates().reset_index(drop=True)

    agg_map: dict = {}
    for col in forecast_cols:
        if col in SIGMA_COLS:
            agg_map[col] = propagate_sigma_indep_sum
        else:
            agg_map[col] = sum_strict

    agg_df = df.groupby(by=list(group_cols), dropna=False, as_index=False).agg(agg_map)

    # Recompute bounds from new forecast ± z·new_sigma.
    if FORECAST_COL in agg_df.columns and STD_COL in agg_df.columns:
        lower, upper = recompute_bounds(
            forecast=agg_df[FORECAST_COL],
            sigma=agg_df[STD_COL],
            ci_coverage=ci_coverage,
        )
        if FORECAST_LOWER_COL in df.columns:
            agg_df[FORECAST_LOWER_COL] = lower
        if FORECAST_UPPER_COL in df.columns:
            agg_df[FORECAST_UPPER_COL] = upper

    return agg_df


def aggregate_stage_side(
    df: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Aggregate the ``stage`` column by ``group_cols``.

    A group is ``"forecast"`` if any underlying row is ``"forecast"``,
    otherwise ``"fitted"``.  This keeps the train/forecast cutoff
    inferable from post-transform frames (downstream plots use ``stage``
    to draw the cutoff vline; would otherwise have to be threaded
    through manually).

    Args:
        df: Input frame.
        group_cols: Columns to group by.

    Returns:
        Reduced frame with ``group_cols`` plus a ``stage`` column when
        present in the input; just ``group_cols`` otherwise.
    """
    if STAGE_COL not in df.columns:
        return df.loc[:, list(group_cols)].drop_duplicates().reset_index(drop=True)
    return df.groupby(by=list(group_cols), dropna=False, as_index=False).agg(
        {STAGE_COL: lambda series: "forecast" if (series == "forecast").any() else "fitted"}
    )


def aggregate_actual_side_sum(
    df: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Sum-aggregate just the ``actual`` column by ``group_cols``.

    Returns a reduced two-column frame (group_cols + actual) ready to
    merge onto the forecast-side aggregate.  Empty actual column or
    missing column → returns just the group_cols (caller's merge will
    leave ``actual`` absent in the output, which is acceptable).

    Args:
        df: Input frame (already prepped, same shape as the forecast
            side's input).
        group_cols: Columns to group by.

    Returns:
        Reduced frame with ``group_cols`` plus an ``actual`` column
        (when present in the input).
    """
    if ACTUAL_COL not in df.columns:
        return df.loc[:, list(group_cols)].drop_duplicates().reset_index(drop=True)
    return df.groupby(by=list(group_cols), dropna=False, as_index=False).agg({ACTUAL_COL: sum_strict})


# ---------------------------------------------------------------------------
# Share side — rows preserved.  Two in-place passes.
# ---------------------------------------------------------------------------


def apply_forecast_side_share(
    df: pd.DataFrame,
    denom_group_cols: Sequence[str],
    ci_coverage: float,
    sigma_method: str = "constant",
) -> pd.DataFrame:
    """Reweight forecast-bearing columns as shares of their
    ``denom_group_cols``-grouped totals.

    Math by column class:

    - **``forecast``**: divide by group total of ``forecast``.
    - **Decomposition**: divide by group total of ``forecast`` (universal
      denominator → ``Σ component_share = forecast_share`` at the row
      level).
    - **``std``**: dispatched on ``sigma_method`` —
      ``"constant"`` uses :func:`propagate_sigma_share` (denominator as
      known constant), ``"delta"`` uses
      :func:`propagate_sigma_share_delta` (full delta method with
      denominator uncertainty).
    - **Bounds**: recomputed from new forecast ± z·new_σ.

    The ``actual`` column is NOT touched — it has its own share pass.

    Args:
        df: Input frame (modified in place semantically; copy if you
            need the original).
        denom_group_cols: Columns whose levels define the group whose
            total is the share denominator.
        ci_coverage: Two-sided coverage for bound recomputation.
        sigma_method: One of :data:`SIGMA_METHODS`.

    Returns:
        The (mutated) input frame.

    Raises:
        ValueError: When ``sigma_method`` is not in :data:`SIGMA_METHODS`.
    """
    if sigma_method not in SIGMA_METHODS:
        raise ValueError(f"sigma_method must be one of {SIGMA_METHODS}; got {sigma_method!r}.")
    if FORECAST_COL not in df.columns:
        return df

    forecast_total = df.groupby(by=list(denom_group_cols), dropna=False)[FORECAST_COL].transform("sum")
    safe_forecast_total = forecast_total.where(forecast_total != 0)

    # Forecast: own-column share (= safe_forecast_total in this case).
    df[FORECAST_COL] = df[FORECAST_COL] / safe_forecast_total

    # Decomposition: same forecast-total denominator.
    for col in BREAKDOWN_COLS:
        if col in df.columns:
            df[col] = df[col] / safe_forecast_total

    # Sigma share — dispatch on method.  We need value_y per group
    # for both methods; "delta" additionally needs sigma_y per group.
    if STD_COL in df.columns:
        # Pre-compute the per-group denominator sigma (independent-sum
        # of std) only when we'll use it.
        if sigma_method == "delta":
            sigma_y_per_row = df.groupby(by=list(denom_group_cols), dropna=False)[STD_COL].transform(propagate_sigma_indep_sum)
        else:
            sigma_y_per_row = None

        # X (numerator) is the *original* forecast value, but at this
        # point df[FORECAST_COL] has already been divided.  Reconstruct
        # the original by multiplying back by the group total — same
        # value as before the share pass.
        original_forecast = df[FORECAST_COL] * safe_forecast_total

        new_sigma = pd.Series(np.nan, index=df.index, dtype=float)
        for _, group_idx in df.groupby(by=list(denom_group_cols), dropna=False).groups.items():
            rows = df.loc[group_idx]
            value_y = safe_forecast_total.loc[group_idx].iloc[0] if len(group_idx) else float("nan")
            if sigma_method == "delta":
                sigma_y = sigma_y_per_row.loc[group_idx].iloc[0] if len(group_idx) else float("nan")
                new_sigma.loc[group_idx] = propagate_sigma_share_delta(
                    sigma_x=rows[STD_COL],
                    value_x=original_forecast.loc[group_idx],
                    sigma_y=sigma_y,
                    value_y=value_y,
                ).values
            else:
                new_sigma.loc[group_idx] = propagate_sigma_share(
                    sigma_x=rows[STD_COL],
                    value_y=value_y,
                ).values
        df[STD_COL] = new_sigma

    # Recompute bounds.
    if STD_COL in df.columns:
        lower, upper = recompute_bounds(
            forecast=df[FORECAST_COL],
            sigma=df[STD_COL],
            ci_coverage=ci_coverage,
        )
        if FORECAST_LOWER_COL in df.columns:
            df[FORECAST_LOWER_COL] = lower
        if FORECAST_UPPER_COL in df.columns:
            df[FORECAST_UPPER_COL] = upper

    return df


def apply_actual_side_share(
    df: pd.DataFrame,
    denom_group_cols: Sequence[str],
) -> pd.DataFrame:
    """Reweight just the ``actual`` column as its share of its
    ``denom_group_cols``-grouped own total.

    ``actual_share = actual / total_actual`` — own-column denominator,
    independent of the forecast side.  NaN where the group's actual sum
    is zero, or where *any* actual in the group is NaN (all-or-nothing
    semantics matching :func:`sum_strict` on the sum side: a partially-
    populated period total isn't a meaningful denominator).

    Args:
        df: Input frame (modified in place semantically).
        denom_group_cols: Columns defining the share group.

    Returns:
        The (mutated) input frame.
    """
    if ACTUAL_COL not in df.columns:
        return df
    actual_grouped = df.groupby(by=list(denom_group_cols), dropna=False)[ACTUAL_COL]
    # NaN-out the denominator for any group that has even one missing
    # actual — partial-period totals would otherwise produce bogus shares
    # for rows like the last few days of fitted data when the rest of
    # their period is in the forecast horizon.
    group_has_nan = actual_grouped.transform(lambda series: series.isna().any())
    actual_total = actual_grouped.transform("sum")
    safe_actual_total = actual_total.where(~group_has_nan).where(actual_total != 0)
    df[ACTUAL_COL] = df[ACTUAL_COL] / safe_actual_total
    return df

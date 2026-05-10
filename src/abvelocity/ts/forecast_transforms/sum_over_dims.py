# Original author: Reza Hosseini
"""``SumOverDims`` — collapse one or more dimension columns by summing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from abvelocity.ts.constants import (
    FORECASTED_DATE_COL,
    METRIC_ID_COL,
    METRIC_NAME_COL,
    STAGE_COL,
)
from abvelocity.ts.forecast_transforms.aggregation import (
    aggregate_actual_side_sum,
    aggregate_forecast_side_sum,
    aggregate_stage_side,
)
from abvelocity.ts.forecast_transforms.column_classes import METADATA_COLS
from abvelocity.ts.forecast_transforms.base import ForecastTransform


# Columns that always belong in the group key when present — identity
# and timestamp.  Caller doesn't list these in ``dims_maintained``.
_ALWAYS_GROUPED = (METRIC_ID_COL, METRIC_NAME_COL, FORECASTED_DATE_COL)


@dataclass(frozen=True)
class SumOverDims(ForecastTransform):
    """Sum forecasts across the given dimension columns.

    Country-level signups +
    ``SumOverDims(dims_summed=("country",), dims_maintained=("region",))``
    → region-level signups (country drops out, region stays as a row
    dimension in the output).

    The two dim sets are explicit so the API doesn't silently group by
    a string column the caller didn't intend to keep.  Identity
    columns (``metric_id``, ``metric_name``) and the timestamp column
    are always part of the group key — caller doesn't list them.

    No completeness check on dim aggregation — there's no canonical
    "all dim levels expected" concept.  Caller validates upstream.

    Implementation: two-pass + join, identical to :class:`SumOverPeriod`
    minus the time-anchor + completeness machinery.

    Args:
        dims_summed: Dim column names to sum across and drop from the
            output.
        dims_maintained: Dim column names to keep in the group key
            (preserved as row dimensions in the output).  Empty tuple
            means "no extra dims" — the output groups by identity +
            timestamp only.
        ci_coverage: Two-sided coverage for the recomputed
            ``forecast_lower`` / ``forecast_upper`` bounds; default
            0.80 matches the JobConfig default.

    Raises:
        ValueError: When ``dims_summed`` and ``dims_maintained`` overlap,
            when any listed column isn't in the input frame, or when
            the input has a non-numeric column not classified into
            either set (or identity / timestamp / stage).
    """

    dims_summed: Tuple[str, ...]
    dims_maintained: Tuple[str, ...] = ()
    ci_coverage: float = 0.80

    def apply(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if forecast_df.empty:
            return forecast_df.copy()

        overlap = set(self.dims_summed) & set(self.dims_maintained)
        if overlap:
            raise ValueError(f"dims_summed and dims_maintained must be disjoint; " f"overlap: {sorted(overlap)!r}.")

        all_listed = list(self.dims_summed) + list(self.dims_maintained)
        missing = [col for col in all_listed if col not in forecast_df.columns]
        if missing:
            raise ValueError(f"dims {missing!r} not found in forecast_df columns " f"{list(forecast_df.columns)!r}.")

        always_grouped = [col for col in _ALWAYS_GROUPED if col in forecast_df.columns]
        accounted = set(always_grouped) | set(all_listed) | {STAGE_COL} | set(METADATA_COLS)
        unclassified = sorted(col for col in forecast_df.columns if col not in accounted and not pd.api.types.is_numeric_dtype(forecast_df[col]))
        if unclassified:
            raise ValueError(f"unclassified non-numeric columns {unclassified!r} — " f"add each to dims_summed or dims_maintained.")

        out_df = forecast_df.drop(columns=list(self.dims_summed), errors="ignore")
        group_cols = always_grouped + list(self.dims_maintained)

        forecast_part = aggregate_forecast_side_sum(
            df=out_df,
            group_cols=group_cols,
            ci_coverage=self.ci_coverage,
        )
        actual_part = aggregate_actual_side_sum(df=out_df, group_cols=group_cols)
        stage_part = aggregate_stage_side(df=out_df, group_cols=group_cols)
        merged = forecast_part.merge(right=actual_part, on=group_cols, how="left").merge(right=stage_part, on=group_cols, how="left")

        return merged.sort_values(by=group_cols).reset_index(drop=True)

    def str_name(self) -> str:
        return f"sum_over_{'_'.join(self.dims_summed)}"

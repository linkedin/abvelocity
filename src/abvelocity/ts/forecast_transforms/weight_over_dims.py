# Original author: Reza Hosseini
"""``WeightOverDims`` — each row's share of its dim-grouped total."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from abvelocity.ts.constants import (
    FORECASTED_DATE_COL,
    METRIC_ID_COL,
    METRIC_NAME_COL,
)
from abvelocity.ts.forecast_transforms.aggregation import (
    apply_actual_side_share,
    apply_forecast_side_share,
)
from abvelocity.ts.forecast_transforms.base import ForecastTransform


@dataclass(frozen=True)
class WeightOverDims(ForecastTransform):
    """Replace each forecast value with its share of its
    ``within_dims``-grouped total.

    Country-level signups + ``WeightOverDims(within_dims=("region",))``
    → each country becomes its share of its region's total.  Empty
    ``within_dims`` = share of the grand total per (timestamp,
    identity) group.

    Implementation: rows aren't reduced; two in-place share passes
    (forecast-side + actual-side), no join needed.

    Args:
        within_dims: Dim columns whose levels define the group within
            which weights sum to 1.  Empty tuple means "share of grand
            total" within each (timestamp, identity) bucket.
        sigma_method: ``"constant"`` (default) treats the group
            forecast total as a known denominator; ``"delta"`` uses
            the full delta method with denominator uncertainty
            propagated.  See
            :data:`abvelocity.ts.forecast_transforms.aggregation.SIGMA_METHODS`.
        ci_coverage: Two-sided coverage for the recomputed
            ``forecast_lower`` / ``forecast_upper`` bounds; default
            0.80 matches the JobConfig default.

    Raises:
        ValueError: When any of ``within_dims`` isn't a column of the
            input frame.
    """

    within_dims: Tuple[str, ...] = ()
    sigma_method: str = "constant"
    ci_coverage: float = 0.80

    def apply(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if forecast_df.empty:
            return forecast_df.copy()

        missing = [dim for dim in self.within_dims if dim not in forecast_df.columns]
        if missing:
            raise ValueError(f"within_dims {missing!r} not found in forecast_df columns " f"{list(forecast_df.columns)!r}.")

        out_df = forecast_df.copy()

        identity_cols = [col for col in (METRIC_ID_COL, METRIC_NAME_COL) if col in out_df.columns]
        time_cols = [FORECASTED_DATE_COL] if FORECASTED_DATE_COL in out_df.columns else []
        denom_group = time_cols + identity_cols + list(self.within_dims)

        apply_forecast_side_share(
            df=out_df,
            denom_group_cols=denom_group,
            ci_coverage=self.ci_coverage,
            sigma_method=self.sigma_method,
        )
        apply_actual_side_share(df=out_df, denom_group_cols=denom_group)

        return out_df.reset_index(drop=True)

    def str_name(self) -> str:
        if not self.within_dims:
            return "share_of_total"
        return f"share_within_{'_'.join(self.within_dims)}"

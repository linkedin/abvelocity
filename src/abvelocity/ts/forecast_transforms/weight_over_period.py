# Original author: Reza Hosseini
"""``WeightOverPeriod`` — each row's share of the enclosing time
period's total."""

from __future__ import annotations

from dataclasses import dataclass

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
from abvelocity.ts.forecast_transforms.period import (
    expected_count_in_period,
    infer_input_freq,
    normalize_to_period_alias,
)


@dataclass(frozen=True)
class WeightOverPeriod(ForecastTransform):
    """Replace each forecast value with its share of the enclosing
    period's total.

    Daily forecast + ``WeightOverPeriod("W-SAT")`` → each Sun..Sat row
    becomes ``daily_forecast / sum_of_week_forecast``.  Within a complete
    period the forecast shares sum to 1.

    Rows in **incomplete** periods come back NaN — partial-period
    shares aren't meaningful.

    Implementation: rows aren't reduced (one share per input row), so
    no join is needed.  Instead, two in-place passes:

      1. Reweight the forecast side (forecast / std / bounds /
         breakdown).
      2. Reweight the actual side (just ``actual / total_actual``).

    Args:
        period: Pandas freq alias for the enclosing period (``"W-SAT"``,
            ``"MS"``, ``"YS"``, ``"D"``, ...).
        sigma_method: ``"constant"`` (default) treats the period
            forecast total as a known denominator; ``"delta"`` uses
            the full delta method with denominator uncertainty
            propagated.  See
            :data:`abvelocity.ts.forecast_transforms.aggregation.SIGMA_METHODS`.
        ci_coverage: Two-sided coverage for the recomputed
            ``forecast_lower`` / ``forecast_upper`` bounds; default
            0.80 matches the JobConfig default.
    """

    period: str
    sigma_method: str = "constant"
    ci_coverage: float = 0.80

    def apply(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if forecast_df.empty:
            return forecast_df.copy()

        input_freq = infer_input_freq(forecast_df=forecast_df)

        out_df = forecast_df.copy()
        out_df[FORECASTED_DATE_COL] = pd.to_datetime(out_df[FORECASTED_DATE_COL])

        period_alias = normalize_to_period_alias(offset_or_period_alias=self.period)
        period_obj = out_df[FORECASTED_DATE_COL].dt.to_period(period_alias)
        out_df["_period_start"] = period_obj.dt.start_time.dt.normalize()
        out_df["_period_end"] = period_obj.dt.end_time.dt.normalize()

        identity_cols = [col for col in (METRIC_ID_COL, METRIC_NAME_COL) if col in out_df.columns]
        denom_group = identity_cols + ["_period_start"]

        # Completeness mask — applied at the end so partial-period rows
        # come back NaN'd uniformly across all value columns.
        actual_count = out_df.groupby(by=identity_cols + ["_period_start", "_period_end"], dropna=False)[FORECASTED_DATE_COL].transform("size")
        expected_count = pd.Series(
            data=[
                expected_count_in_period(period_start=start, period_end=end, input_freq=input_freq)
                for start, end in zip(out_df["_period_start"], out_df["_period_end"])
            ],
            index=out_df.index,
        )
        is_complete = actual_count >= expected_count

        # Two share passes.
        apply_forecast_side_share(
            df=out_df,
            denom_group_cols=denom_group,
            ci_coverage=self.ci_coverage,
            sigma_method=self.sigma_method,
        )
        apply_actual_side_share(df=out_df, denom_group_cols=denom_group)

        # NaN-mask incomplete periods on every value-bearing column.
        for col in out_df.columns:
            if pd.api.types.is_numeric_dtype(out_df[col]):
                out_df[col] = out_df[col].where(is_complete)

        return out_df.drop(columns=["_period_start", "_period_end"]).reset_index(drop=True)

    def str_name(self) -> str:
        period_word = {
            "W": "week",
            "D": "day",
            "MS": "month",
            "M": "month",
            "ME": "month",
            "h": "hour",
        }
        word = period_word.get(self.period, self.period.lower())
        return f"share_of_{word}"

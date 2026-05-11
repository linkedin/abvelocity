# Original author: Reza Hosseini
"""``SumOverPeriod`` — daily → weekly / monthly / annual time aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from abvelocity.ts.constants import FORECASTED_DATE_COL, STAGE_COL
from abvelocity.ts.forecast_transforms.aggregation import (
    aggregate_actual_side_sum,
    aggregate_forecast_side_sum,
    aggregate_stage_side,
    grouping_columns,
)
from abvelocity.ts.forecast_transforms.base import ForecastTransform
from abvelocity.ts.forecast_transforms.period import (
    expected_count_in_period,
    infer_input_freq,
    normalize_to_period_alias,
)


@dataclass(frozen=True)
class SumOverPeriod(ForecastTransform):
    """Sum forecasts into coarser period-start-anchored time buckets.

    Daily → weekly with ``SumOverPeriod("W-SAT")``, daily → monthly with
    ``SumOverPeriod("MS")``, daily → annual with ``SumOverPeriod("YS")``,
    hourly → daily with ``SumOverPeriod("D")``.  Input freq is inferred
    from the frame's timestamp column.

    Output's ``forecasted_date`` carries the **period start** (lower
    bound).  Incomplete periods are dropped.

    Implementation: two-pass + join.

      1. Aggregate the forecast side (forecast / std / bounds /
         breakdown).
      2. Aggregate the actual side (just ``actual``).
      3. Join by group key.
      4. Drop incomplete periods via :func:`expected_count_in_period`.

    Args:
        period: Target pandas freq alias (``"W-SAT"``, ``"MS"``,
            ``"YS"``, ``"D"``, ...).  Same field name as
            :class:`WeightOverPeriod` for consistency.
        ci_coverage: Two-sided coverage for the recomputed
            ``forecast_lower`` / ``forecast_upper`` bounds; default
            0.80 matches the JobConfig default.
    """

    period: str
    ci_coverage: float = 0.80

    def apply(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if forecast_df.empty:
            return forecast_df.copy()

        input_freq = infer_input_freq(forecast_df=forecast_df)

        out_df = forecast_df.copy()
        out_df[FORECASTED_DATE_COL] = pd.to_datetime(out_df[FORECASTED_DATE_COL])

        # Anchor each row to its enclosing period's START (lower bound).
        period_alias = normalize_to_period_alias(offset_or_period_alias=self.period)
        period_obj = out_df[FORECASTED_DATE_COL].dt.to_period(period_alias)
        out_df[FORECASTED_DATE_COL] = period_obj.dt.start_time.dt.normalize()
        out_df["_period_end"] = period_obj.dt.end_time.dt.normalize()

        # Stage gets aggregated post-merge (any-forecast-wins) so it
        # propagates through; exclude it from the group key.
        group_cols = [FORECASTED_DATE_COL, "_period_end"] + grouping_columns(
            forecast_df=out_df,
            exclude=[FORECASTED_DATE_COL, "_period_end", STAGE_COL],
        )

        forecast_part = aggregate_forecast_side_sum(
            df=out_df,
            group_cols=group_cols,
            ci_coverage=self.ci_coverage,
        )
        actual_part = aggregate_actual_side_sum(df=out_df, group_cols=group_cols)
        stage_part = aggregate_stage_side(df=out_df, group_cols=group_cols)
        merged = forecast_part.merge(right=actual_part, on=group_cols, how="left").merge(right=stage_part, on=group_cols, how="left")

        # Completeness: drop rows where actual_count < expected.
        actual_count = out_df.groupby(by=group_cols, dropna=False, as_index=False).size().rename(columns={"size": "_actual_count"})
        merged = merged.merge(right=actual_count, on=group_cols, how="left")
        merged["_expected_count"] = [
            expected_count_in_period(period_start=start, period_end=end, input_freq=input_freq)
            for start, end in zip(merged[FORECASTED_DATE_COL], merged["_period_end"])
        ]
        merged = merged.loc[merged["_actual_count"] >= merged["_expected_count"]].copy()

        return (
            merged.drop(columns=["_period_end", "_actual_count", "_expected_count"])
            .sort_values(by=[FORECASTED_DATE_COL] + [col for col in group_cols if col not in (FORECASTED_DATE_COL, "_period_end")])
            .reset_index(drop=True)
        )

    def str_name(self) -> str:
        period_to_word = {
            "W": "weekly",
            "D": "daily",
            "MS": "monthly",
            "ME": "monthly",
            "M": "monthly",
            "QS": "quarterly",
            "QE": "quarterly",
            "Q": "quarterly",
            "YS": "annual",
            "YE": "annual",
            "Y": "annual",
        }
        return period_to_word.get(self.period, self.period.lower())

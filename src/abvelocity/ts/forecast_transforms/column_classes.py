# Original author: Reza Hosseini
"""Column classifications for the forecast frame.

The forecast frame produced by ``TSFlow.run`` has three kinds of
numeric columns; each behaves differently under aggregation.
Classifying them up front lets the transforms drive math by column
class instead of by hard-coded names sprinkled through ``apply()``.
"""

from __future__ import annotations

from typing import Tuple

from abvelocity.ts.constants import (
    ACTUAL_COL,
    ALGO_NAME_COL,
    ALGO_VERSION_COL,
    ANNUAL_SEASONALITY_COL,
    DAILY_SEASONALITY_COL,
    EXTRAS_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    HOLIDAY_IMPACT_COL,
    LAST_TRAINING_DATE_COL,
    LONGTERM_GROWTH_COL,
    RESIDUAL_COL,
    RUN_DATE_COL,
    RUN_ID_COL,
    SHORTTERM_GROWTH_COL,
    STD_COL,
    TIME_COL,
    WEEKLY_SEASONALITY_COL,
)


POINT_SUMMABLE_COLS: Tuple[str, ...] = (
    ACTUAL_COL,
    FORECAST_COL,
    LONGTERM_GROWTH_COL,
    SHORTTERM_GROWTH_COL,
    DAILY_SEASONALITY_COL,
    WEEKLY_SEASONALITY_COL,
    ANNUAL_SEASONALITY_COL,
    HOLIDAY_IMPACT_COL,
    RESIDUAL_COL,
)
"""Numeric columns that combine additively under sum aggregation.

Decomposition columns (``longterm_growth``, ``*_seasonality``,
``holiday_impact``, ``residual``) are linear contributions to
``forecast``, so summing them across a period or dim group gives the
period/group's total contribution from each component.
"""

POINT_SHAREABLE_COLS: Tuple[str, ...] = (
    ACTUAL_COL,
    FORECAST_COL,
)
"""Point columns that have a meaningful share interpretation.

A "share of weekly forecast" is well-defined; a "share of weekly
seasonality contribution" is not (breakdown components don't form
a budget that sums to a recognizable whole).  Breakdown columns
are reweighted using the ``forecast`` group total as the universal
denominator instead — that preserves
``Σ component_share = forecast_share`` at the row level.
"""

SIGMA_COLS: Tuple[str, ...] = (STD_COL,)
"""Standard-deviation columns; need variance-aware propagation."""

BOUND_COLS: Tuple[str, ...] = (FORECAST_LOWER_COL, FORECAST_UPPER_COL)
"""Prediction-interval bound columns; recomputed from new forecast
± z·new_sigma after any aggregation.
"""

BREAKDOWN_COLS: Tuple[str, ...] = tuple(c for c in POINT_SUMMABLE_COLS if c not in POINT_SHAREABLE_COLS)
"""Breakdown / component columns — POINT_SUMMABLE minus
POINT_SHAREABLE.  Reweighted against the forecast period total so the
sum of component shares equals the forecast share at each row.
"""

METADATA_COLS: Tuple[str, ...] = (
    TIME_COL,
    LAST_TRAINING_DATE_COL,
    ALGO_NAME_COL,
    ALGO_VERSION_COL,
    EXTRAS_COL,
    RUN_ID_COL,
    RUN_DATE_COL,
)
"""TSRunner-stamped metadata columns that are *not* grouping keys.

These coexist with the canonical schema on every frame produced via
:class:`~abvelocity.ts.forecast_runner.ForecastRunner`, but they
are either:

- duplicate forms of the time axis (``ts`` mirrors ``forecasted_date``),
- per-row config scalars (``algo_name``, ``algo_version``,
  ``last_training_date``), or
- run identity / extra-payload columns (``run_id``, ``run_date``,
  ``extras``).

Including them in a groupby would either split every row into its own
bucket (the time-axis case) or be meaningless.  The transforms exclude
them from group-key derivation and pass them through aggregation
unchanged."""

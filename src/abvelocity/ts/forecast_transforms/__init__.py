# Original author: Reza Hosseini
"""Post-forecast transforms — derive a forecast from another forecast.

See :doc:`architecture` for the design (column classes, two-pass +
join structure, period-anchor convention, completeness rule).

Public API:

  - :class:`ForecastTransform` — abstract base class.
  - :class:`SumOverPeriod` — time-axis aggregation
    (D → W / MS / YS / D).
  - :class:`WeightOverPeriod` — time-axis share-within-period.
  - :class:`SumOverDims` — dim-axis aggregation.
  - :class:`WeightOverDims` — dim-axis share-within-group.

Utilities are exposed at the submodule level (``period``, ``sigma``,
``bounds``, ``aggregation``, ``column_classes``) for callers who need
the building blocks directly.
"""

from abvelocity.ts.forecast_transforms.base import ForecastTransform
from abvelocity.ts.forecast_transforms.sum_over_dims import SumOverDims
from abvelocity.ts.forecast_transforms.sum_over_period import SumOverPeriod
from abvelocity.ts.forecast_transforms.weight_over_dims import WeightOverDims
from abvelocity.ts.forecast_transforms.weight_over_period import WeightOverPeriod

__all__ = [
    "ForecastTransform",
    "SumOverDims",
    "SumOverPeriod",
    "WeightOverDims",
    "WeightOverPeriod",
]

# Original author: Reza Hosseini
"""Prediction-interval bound recomputation.

After any aggregation that produces a new forecast and a new sigma,
the original ``forecast_lower`` / ``forecast_upper`` columns no longer
correspond to the new center.  Recompute via
``forecast ± multiplier·σ`` where ``multiplier`` is the two-sided
Gaussian quantile for the desired ``ci_coverage``.

No multiplier is hardcoded; the caller supplies ``ci_coverage`` and
the helpers convert via :func:`multiplier_from_coverage`.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from scipy.stats import norm


def multiplier_from_coverage(ci_coverage: float) -> float:
    """Two-sided Gaussian quantile for ``ci_coverage``.

    ``ci_coverage = 0.80`` → 1.2816, ``0.95`` → 1.9600, ``0.99`` →
    2.5758, etc.

    Args:
        ci_coverage: Coverage in (0, 1).

    Returns:
        Multiplier such that ``forecast ± multiplier·σ`` brackets
        ``ci_coverage`` of the Gaussian distribution.
    """
    return float(norm.ppf((1.0 + ci_coverage) / 2.0))


def recompute_bounds(
    forecast: pd.Series,
    sigma: pd.Series,
    ci_coverage: float,
) -> Tuple[pd.Series, pd.Series]:
    """Lower / upper bounds = ``forecast ∓ multiplier·σ``.

    Args:
        forecast: New point forecast values.
        sigma: New sigma values.
        ci_coverage: Two-sided coverage in (0, 1).  Required — caller
            supplies it (no default to discourage silent assumptions).

    Returns:
        ``(lower, upper)`` tuple of Series.
    """
    multiplier = multiplier_from_coverage(ci_coverage=ci_coverage)
    lower = forecast - multiplier * sigma
    upper = forecast + multiplier * sigma
    return lower, upper

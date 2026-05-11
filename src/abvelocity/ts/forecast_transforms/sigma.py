# Original author: Reza Hosseini
"""Sigma propagation utilities for sums and shares.

Two regimes:

- **Independent-Gaussian sum** — variance of a sum of independent
  random variables is the sum of variances; sigma is the square root.
- **Share with constant-denominator approximation** — for ``w = X/Y``
  where ``Y`` is the *observed* group sum, treat ``Y`` as a known
  constant.  Drops the denominator-uncertainty term of the delta
  method.  Justified when the consumer reports the observed sum as a
  fact rather than estimating its sampling distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def propagate_sigma_indep_sum(sigmas: pd.Series) -> float:
    """Sigma for an independent-Gaussian sum: ``sqrt(Σ σ²)``.

    Variance of a sum of independent random variables is the sum of
    their variances.  Returns NaN if every input is NaN; ignores NaN
    entries otherwise (treats them as zero-variance — caller is
    responsible for ensuring that's appropriate).

    Args:
        sigmas: Per-row sigmas to combine.

    Returns:
        Aggregated sigma, or ``NaN`` when no finite input remains.
    """
    finite = sigmas.dropna()
    if finite.empty:
        return float("nan")
    return float(np.sqrt((finite**2).sum()))


def propagate_sigma_share(sigma_x: pd.Series, value_y: float) -> pd.Series:
    """Sigma for ``share = X / Y`` treating ``Y`` as a known constant.

    Drops the denominator-uncertainty term of the delta method:

        σ(w) = σ_X / |Y|

    Justified when the denominator is the *observed* group total — we
    take the observed sum as a fact rather than propagating its
    sampling uncertainty.  Understates the share's true sigma relative
    to :func:`propagate_sigma_share_delta`, but matches how shares are
    reported in practice and avoids the σ_Y plumbing.

    Args:
        sigma_x: Per-row sigma of the numerator.
        value_y: Value of the denominator (the observed group sum).

    Returns:
        Per-row sigma of the share, ``NaN`` where ``value_y == 0``.
    """
    if value_y == 0 or pd.isna(value_y):
        return pd.Series(np.nan, index=sigma_x.index)
    return sigma_x / abs(value_y)


def propagate_sigma_share_delta(
    sigma_x: pd.Series,
    value_x: pd.Series,
    sigma_y: float,
    value_y: float,
) -> pd.Series:
    """Sigma for ``share = X / Y`` via the delta method, assuming
    ``X ⊥ Y``.

    Full first-order Taylor expansion of ``w = X/Y``:

        Var(w) ≈ (∂w/∂X)² · Var(X) + (∂w/∂Y)² · Var(Y)
               = Var(X)/Y² + (X/Y)² · Var(Y)/Y²

    Taking square roots and grouping ``|Y|`` out:

        σ(w) = √(σ_X² + w² · σ_Y²) / |Y|     where w = X/Y

    Numerator-denominator independence isn't strictly true when
    ``Y = Σ X_i`` (the focal X is a term in the sum), so dropping the
    covariance is an approximation; the resulting overstatement is
    small when each share is small and the count is moderate.  The
    Taylor approximation itself is standard for ratio uncertainty.

    Use this when you want the denominator's sampling uncertainty
    propagated; use :func:`propagate_sigma_share` when you want the
    denominator treated as a known constant (the simpler default).

    Args:
        sigma_x: Per-row sigma of the numerator.
        value_x: Per-row value of the numerator.
        sigma_y: Sigma of the denominator (typically the
            independent-sum sigma of the X_i's, via
            :func:`propagate_sigma_indep_sum`).
        value_y: Value of the denominator.

    Returns:
        Per-row sigma of the share, ``NaN`` where ``value_y == 0``.
    """
    if value_y == 0 or pd.isna(value_y):
        return pd.Series(np.nan, index=sigma_x.index)
    w = value_x / value_y
    var_share = (sigma_x**2 + (w**2) * (sigma_y**2)) / (value_y**2)
    return np.sqrt(var_share)

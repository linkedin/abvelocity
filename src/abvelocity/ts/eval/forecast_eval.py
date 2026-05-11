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
"""Forecast evaluation: compute accuracy metrics on any result_df with actual + forecast columns.

Operates on any long-format DataFrame that has ``actual`` and ``forecast``
columns — a regular ``TSResult.result_df``, a ``BackfillResult.result_df``,
or any subset thereof. The caller controls which metrics to compute and how
to group the output.

Supported metrics
-----------------
``mae``
    Mean Absolute Error: ``mean(|actual - prediction|)``.
``rmse``
    Root Mean Squared Error: ``sqrt(mean((actual - prediction)²))``.
``mape``
    Mean Absolute Percentage Error:
    ``mean(|actual - prediction| / |actual|) × 100``.
    Rows where ``actual == 0`` are excluded from the mean.
``smape``
    Symmetric MAPE:
    ``mean(|actual - prediction| / ((|actual| + |prediction|) / 2)) × 100``.
    Rows where ``|actual| + |prediction| == 0`` are excluded.
``r2``
    Coefficient of determination:
    ``1 - SS_res / SS_tot`` where ``SS_tot = sum((actual - mean(actual))²)``.
    Returns ``NaN`` when ``SS_tot == 0`` (constant actual series).
``medae``
    Median Absolute Error: ``median(|actual - prediction|)``.
``coverage``
    Prediction interval coverage: fraction of rows where
    ``forecast_lower <= actual <= forecast_upper``. Returns ``NaN``
    when ``forecast_lower`` / ``forecast_upper`` are not in the DataFrame.

The following metrics operate on the **signed residual** ``e = actual - forecast``
(positive = underprediction). They are most meaningful when grouped by
``(metric, horizon_step)`` so that the error distribution is estimated
separately per horizon step.

``bias``
    Mean signed residual: ``mean(actual - forecast)``.
``sigma``
    Sample standard deviation of the signed residual: ``std(e, ddof=1)``.
    Returns ``NaN`` when fewer than 2 valid rows are available.
``q25``
    25th percentile of the signed residual.
``q50``
    Median signed residual (50th percentile).
``q75``
    75th percentile of the signed residual.
``iqr``
    Interquartile range of the signed residual: ``q75 - q25``.
    Captures the spread of the middle 50 % of errors, robust to outliers.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from abvelocity.ts.constants import ACTUAL_COL, FORECAST_COL, FORECAST_LOWER_COL, FORECAST_UPPER_COL, HORIZON_STEP_COL, METRIC_ID_COL

SUPPORTED_METRICS = {
    "mae",
    "rmse",
    "mape",
    "smape",
    "medape",
    "r2",
    "medae",
    "coverage",
    "bias",
    "sigma",
    "q25",
    "q50",
    "q75",
    "iqr",
}


DEFAULT_TRIM = 0.01
"""Default fraction of largest-|error| rows trimmed before computing
point-error metrics (MAE / RMSE / MAPE / sMAPE / MedAPE / MedAE). 1%
catches genuine anomalies the user may not have flagged ahead of time
without distorting the metric for normal data. Set ``trim=0.0`` to
disable. Skipped for R² / coverage / residual-distribution metrics —
trimming would meaningfully distort those (R² inflates; coverage uses
intervals not point errors; bias/sigma/quantiles measure distribution
shape itself)."""


def _trim_by_abs_error(
    actual: np.ndarray,
    prediction: np.ndarray,
    trim: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop the top ``trim`` fraction of rows by ``|actual - prediction|``.

    ``trim=0.01`` on 1000 rows drops the 10 worst-error rows. A no-op
    when ``trim <= 0`` or when ``actual`` is empty. Falls back to no-op
    if the implied drop count would consume the whole array.

    Args:
        actual: Actual values (1-D float array).
        prediction: Predicted values (1-D float array, same length).
        trim: Fraction in ``[0, 0.5)`` — fraction of rows to drop.

    Returns:
        ``(trimmed_actual, trimmed_prediction)`` — same dtype, possibly
        shorter.

    Raises:
        ValueError: If ``trim`` is outside ``[0, 0.5)``.
    """
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim!r}.")
    if trim <= 0 or len(actual) == 0:
        return actual, prediction
    n = len(actual)
    # Floor (truncating int conversion) — NOT ceil — so small groups
    # don't get systematically over-trimmed. With ceil + the default
    # ``trim=0.01``, any group with <100 rows always drops at least
    # 1 row, which makes the effective trim fraction 1/n (8.3% on the
    # typical n=12 per-horizon group) instead of the configured 1%.
    # Floor makes the effective trim converge to 0 for small groups
    # — matching the user's intent that 1% on a tiny group means
    # "leave it alone".
    n_drop = int(n * trim)
    if n_drop <= 0 or n_drop >= n:
        return actual, prediction
    abs_err = np.abs(actual - prediction)
    keep_idx = np.argsort(abs_err)[: n - n_drop]
    return actual[keep_idx], prediction[keep_idx]


def compute_eval(
    df: pd.DataFrame,
    metrics: Tuple[str, ...] = ("mae", "rmse", "mape", "smape", "r2"),
    group_by: Tuple[str, ...] = (METRIC_ID_COL, HORIZON_STEP_COL),
    trim: float = DEFAULT_TRIM,
) -> pd.DataFrame:
    """Compute prediction accuracy metrics grouped by the requested columns.

    Args:
        df: Long-format DataFrame containing at least ``actual`` and
            ``forecast`` columns. Typically a ``BackfillResult.result_df``
            or ``TSResult.result_df``. Rows where either column is ``NaN``
            are excluded from every metric computation.
        metrics: Tuple of metric names to compute. Supported values:
            ``"mae"``, ``"rmse"``, ``"mape"``, ``"smape"``, ``"r2"``,
            ``"medae"``, ``"coverage"``, ``"bias"``, ``"sigma"``,
            ``"q25"``, ``"q50"``, ``"q75"``, ``"iqr"``.
            Raises ``ValueError`` for unknown names.
        group_by: Columns to group by before computing metrics. Defaults to
            ``(METRIC_ID_COL, HORIZON_STEP_COL)`` which yields a per-metric
            horizon-degradation curve — the most common use case for
            backfill eval. Pass ``(METRIC_ID_COL,)`` for aggregate accuracy,
            or ``(METRIC_ID_COL, CUTOFF_COL)`` for per-cutoff stability.
        trim: Fraction of largest-|error| rows trimmed before computing
            point-error metrics (mae, rmse, mape, smape, medae). Default
            :data:`DEFAULT_TRIM` (1%). Skipped for r2 / coverage /
            residual-distribution metrics. Set to ``0.0`` to disable
            trimming entirely.

    Returns:
        DataFrame with one row per group. Columns are the ``group_by``
        columns plus ``n`` (number of valid rows in that group) and one
        column per requested metric.

    Raises:
        ValueError: If any element of ``metrics`` is not in
            :data:`SUPPORTED_METRICS`, or if ``trim`` is outside
            ``[0, 0.5)``.
    """
    unknown = set(metrics) - SUPPORTED_METRICS
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}. Supported: {sorted(SUPPORTED_METRICS)}")
    for required in (ACTUAL_COL, FORECAST_COL):
        if required not in df.columns:
            raise ValueError(f"Column {required!r} is required but not found in df.")

    rows: List[dict] = []

    for group_keys, group_df in df.groupby(list(group_by)):
        if len(group_by) == 1:
            group_keys = (group_keys,)

        row: dict = dict(zip(group_by, group_keys))

        valid = group_df[[ACTUAL_COL, FORECAST_COL]].dropna()
        actual = valid[ACTUAL_COL].to_numpy(dtype=float)
        prediction = valid[FORECAST_COL].to_numpy(dtype=float)
        row["n"] = len(valid)

        # Trim once and reuse for every point-error metric. R² / coverage /
        # residual-distribution metrics use the un-trimmed pair below.
        actual_t, prediction_t = _trim_by_abs_error(actual, prediction, trim)

        if "mae" in metrics:
            row["mae"] = float(np.mean(np.abs(actual_t - prediction_t))) if len(actual_t) else float("nan")

        if "rmse" in metrics:
            row["rmse"] = float(np.sqrt(np.mean((actual_t - prediction_t) ** 2))) if len(actual_t) else float("nan")

        if "mape" in metrics:
            row["mape"] = compute_mape(actual_t, prediction_t)

        if "smape" in metrics:
            row["smape"] = compute_smape(actual_t, prediction_t)

        if "medape" in metrics:
            row["medape"] = compute_medape(actual_t, prediction_t)

        if "r2" in metrics:
            # R² intentionally uses the un-trimmed series — it's a ratio of
            # sums of squares that becomes meaningless if the worst-fit
            # rows are dropped (the residual SS shrinks artificially while
            # the total SS stays close to its full-data value).
            row["r2"] = compute_r2(actual, prediction)

        if "medae" in metrics:
            row["medae"] = float(np.median(np.abs(actual_t - prediction_t))) if len(actual_t) else float("nan")

        if "coverage" in metrics:
            row["coverage"] = compute_coverage(group_df)

        residual_metrics = {"bias", "sigma", "q25", "q50", "q75", "iqr"}
        if residual_metrics & set(metrics):
            # Residual-distribution metrics use the un-trimmed errors —
            # trimming top-|error| asymmetrically alters mean / std /
            # quantiles, which defeats the purpose of these metrics.
            errors = actual - prediction
            if "bias" in metrics:
                row["bias"] = float(np.mean(errors)) if len(errors) else float("nan")
            if "sigma" in metrics:
                row["sigma"] = float(np.std(errors, ddof=1)) if len(errors) >= 2 else float("nan")
            if residual_metrics & {"q25", "q50", "q75", "iqr"} & set(metrics):
                q25, q50, q75 = (
                    (
                        float(np.percentile(errors, 25)),
                        float(np.percentile(errors, 50)),
                        float(np.percentile(errors, 75)),
                    )
                    if len(errors)
                    else (float("nan"), float("nan"), float("nan"))
                )
                if "q25" in metrics:
                    row["q25"] = q25
                if "q50" in metrics:
                    row["q50"] = q50
                if "q75" in metrics:
                    row["q75"] = q75
                if "iqr" in metrics:
                    row["iqr"] = q75 - q25 if len(errors) else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def compute_mape(
    actual: np.ndarray,
    prediction: np.ndarray,
    trim: float = 0.0,
) -> float:
    """MAPE: ``mean(|actual - prediction| / |actual|) × 100``.

    Rows where ``actual == 0`` are excluded. Returns ``NaN`` when no
    valid rows remain.

    Args:
        actual: Actual values.
        prediction: Predicted values.
        trim: Fraction of largest-|error| rows to drop before computing.
            Default ``0.0`` (raw MAPE). Set non-zero to mitigate impact
            of unflagged anomalies.
    """
    actual, prediction = _trim_by_abs_error(actual, prediction, trim)
    mask = np.abs(actual) > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - prediction[mask]) / np.abs(actual[mask])) * 100)


def compute_smape(
    actual: np.ndarray,
    prediction: np.ndarray,
    trim: float = 0.0,
) -> float:
    """sMAPE: ``mean(|actual - prediction| / ((|actual| + |prediction|) / 2)) × 100``.

    The denominator is the average of the two absolute values. Rows where
    ``|actual| + |prediction| == 0`` are excluded. Returns ``NaN`` when no
    valid rows remain.

    Args:
        actual: Actual values.
        prediction: Predicted values.
        trim: Fraction of largest-|error| rows to drop before computing.
            Default ``0.0`` (raw sMAPE).
    """
    actual, prediction = _trim_by_abs_error(actual, prediction, trim)
    denom = (np.abs(actual) + np.abs(prediction)) / 2.0
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - prediction[mask]) / denom[mask]) * 100)


def compute_medape(
    actual: np.ndarray,
    prediction: np.ndarray,
    trim: float = 0.0,
) -> float:
    """MedAPE: ``median(|actual - prediction| / |actual|) × 100``.

    Median variant of MAPE — robust to outliers. Rows where ``actual == 0``
    are excluded. Returns ``NaN`` when no valid rows remain.

    Args:
        actual: Actual values.
        prediction: Predicted values.
        trim: Fraction of largest-|error| rows to drop before computing.
            Default ``0.0`` (raw MedAPE). Note: MedAPE is already
            outlier-robust by construction; trim is supported for symmetry
            with the other point-error metrics but is rarely needed.
    """
    actual, prediction = _trim_by_abs_error(actual, prediction, trim)
    mask = np.abs(actual) > 0
    if not mask.any():
        return float("nan")
    return float(np.median(np.abs(actual[mask] - prediction[mask]) / np.abs(actual[mask])) * 100)


def compute_r2(actual: np.ndarray, prediction: np.ndarray) -> float:
    """R²: ``1 - SS_res / SS_tot``.

    Returns ``NaN`` when ``SS_tot == 0`` (constant actual series) or when
    there are no valid rows.
    """
    if len(actual) == 0:
        return float("nan")
    ss_res = float(np.sum((actual - prediction) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_coverage(group_df: pd.DataFrame) -> float:
    """Prediction interval coverage: fraction where ``lower <= actual <= upper``.

    Returns ``NaN`` when all CI values in this group are ``NaN``.

    Raises:
        ValueError: If ``forecast_lower`` or ``forecast_upper`` are not in ``group_df``.
    """
    for col in (FORECAST_LOWER_COL, FORECAST_UPPER_COL):
        if col not in group_df.columns:
            raise ValueError(f"Column {col!r} is required for coverage but not found in df. " "Ensure the prediction algo produces prediction intervals.")
    ci_df = group_df[[ACTUAL_COL, FORECAST_LOWER_COL, FORECAST_UPPER_COL]].dropna()
    if ci_df.empty:
        return float("nan")
    inside = (ci_df[ACTUAL_COL] >= ci_df[FORECAST_LOWER_COL]) & (ci_df[ACTUAL_COL] <= ci_df[FORECAST_UPPER_COL])
    return float(inside.mean())


DEFAULT_FORECAST_METRICS: Tuple[str, ...] = ("mae", "rmse", "mape", "smape", "r2")


@dataclass
class ForecastEval:
    """High-level forecast evaluator.

    Wraps :func:`compute_eval` with a parameterized configuration. Pass
    an instance to :attr:`TSFlowConfig.eval` to have
    :meth:`TSFlow.compute_eval` invoke :meth:`run` automatically.

    Attributes:
        metrics: Metrics to compute — subset of :data:`SUPPORTED_METRICS`.
        group_by: Columns to group by. Defaults to
            ``(METRIC_ID_COL, HORIZON_STEP_COL)``.
    """

    metrics: Tuple[str, ...] = DEFAULT_FORECAST_METRICS
    group_by: Tuple[str, ...] = (METRIC_ID_COL, HORIZON_STEP_COL)
    trim: float = DEFAULT_TRIM
    """Fraction of largest-|error| rows trimmed from point-error metrics
    (mae/rmse/mape/smape/medae). See :data:`DEFAULT_TRIM`."""

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the configured forecast metrics on ``df``."""
        return compute_eval(df, metrics=self.metrics, group_by=self.group_by, trim=self.trim)

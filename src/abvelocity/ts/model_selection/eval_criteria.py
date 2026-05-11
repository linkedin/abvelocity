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
"""Eval criteria for ranking model-selection candidates.

An :class:`EvalCriteria` reduces a candidate's persisted backtest output
(long-format DataFrame with ``actual`` and ``forecast`` columns) into:

* a per-group eval-metrics frame — one row per ``group_by`` combination,
  columns for every eval metric in :attr:`eval_metrics`. Built directly
  via :func:`~abvelocity.ts.eval.forecast_eval.compute_eval`.
* a single primary score for ranking — :attr:`primary_eval_metric`
  reduced across the per-group rows by :attr:`primary_eval_reduction`.

The criteria object is pure post-prediction logic — it knows nothing about
algos, configs, or backtest mechanics. The same criteria can be applied to
any DataFrame produced by any predictor.

Terminology note
----------------
The bare word **metric** in this codebase defaults to the *response
metric* — the response variable being forecast (signups, bookings,
DAU, …). The ``metric_id`` column on every result frame carries it,
and where extra clarity helps we write "metric (response metric)" in
prose.

The accuracy statistics computed from ``(actual, forecast)`` pairs
(MAPE, sMAPE, MedAE, RMSE, …) are always **eval metrics** — written
with the qualifier in docstrings, error messages, and field names
(``eval_metrics``, ``primary_eval_metric``, etc.). Never use bare
"metric" for an accuracy stat.
"""

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from abvelocity.ts.constants import HORIZON_STEP_COL, METRIC_ID_COL
from abvelocity.ts.eval.forecast_eval import DEFAULT_TRIM, SUPPORTED_METRICS, compute_eval

VALID_REDUCTIONS = {"mean", "median", "max", "min", "sum"}


@dataclass
class EvalCriteria:
    """Configuration for reducing a candidate's eval frame to a primary score.

    Naming note: throughout this package the word "metric" is used in two
    distinct senses, which we disambiguate at the field level:

    * **forecast-target metric** — what is being forecast (signups,
      bookings, DAU). Identified by the ``metric_id`` column on every
      result frame. Outside this class.
    * **eval metric** — an accuracy statistic computed from
      ``(actual, forecast)`` pairs (``"mape"``, ``"smape"``, …). The
      fields below all refer to *eval* metrics, hence the ``eval_*``
      prefix.

    Attributes:
        eval_metrics: Eval metrics to compute on every persisted
            prediction frame. Must be a subset of
            :data:`~abvelocity.ts.eval.forecast_eval.SUPPORTED_METRICS`.
            Default ``("mape", "smape", "medae", "mae", "rmse")``.
        group_by: Columns to group by when computing eval metrics. Default
            ``(METRIC_ID_COL, HORIZON_STEP_COL)`` yields per-forecast-target
            horizon-degradation curves. Pass ``(METRIC_ID_COL,)`` to
            collapse across horizon, or ``(METRIC_ID_COL, "cutoff")`` for
            per-cutoff stability.
        primary_eval_metric: Name of the eval metric used to rank
            candidates. Must appear in :attr:`eval_metrics`. Default
            ``"mape"``.
        primary_eval_reduction: How to reduce :attr:`primary_eval_metric`
            across the per-group rows to one scalar. One of ``"mean"``,
            ``"median"``, ``"max"``, ``"min"``, ``"sum"``. Default
            ``"mean"``.
        lower_is_better: ``True`` (default) ranks ascending by score.
            Set to ``False`` for eval metrics where larger is better
            (e.g. ``"r2"``).
        trim: Fraction of largest-|error| rows trimmed before computing
            point-error metrics (mae / rmse / mape / smape / medae).
            Default
            :data:`~abvelocity.ts.eval.forecast_eval.DEFAULT_TRIM`
            (1%) — catches genuine anomalies the user may not have
            flagged ahead of time without distorting the metric for
            normal data. Skipped for r2 / coverage / residual-
            distribution metrics. Set to ``0.0`` to disable.
    """

    eval_metrics: Tuple[str, ...] = ("mape", "smape", "medae", "mae", "rmse")
    """Eval metrics to compute on every persisted prediction frame."""

    group_by: Tuple[str, ...] = (METRIC_ID_COL, HORIZON_STEP_COL)
    """Columns to group by when computing per-group eval metrics."""

    primary_eval_metric: str = "mape"
    """Eval metric used to rank candidates; must appear in :attr:`eval_metrics`."""

    primary_eval_reduction: str = "mean"
    """Reduction across per-group rows to one scalar (``"mean"``/``"median"``/``"max"``/``"min"``/``"sum"``)."""

    lower_is_better: bool = True
    """If True (default), candidates are ranked ascending by ``primary_score``."""

    trim: float = DEFAULT_TRIM
    """Trim fraction for point-error metrics; see class docstring."""

    def __post_init__(self) -> None:
        if not self.eval_metrics:
            raise ValueError("EvalCriteria.eval_metrics must be non-empty.")
        unknown = set(self.eval_metrics) - SUPPORTED_METRICS
        if unknown:
            raise ValueError(
                f"Unknown eval metrics: {sorted(unknown)}. "
                f"Supported: {sorted(SUPPORTED_METRICS)}."
            )
        if self.primary_eval_metric not in self.eval_metrics:
            raise ValueError(
                f"primary_eval_metric={self.primary_eval_metric!r} must be in "
                f"eval_metrics={list(self.eval_metrics)}."
            )
        if self.primary_eval_reduction not in VALID_REDUCTIONS:
            raise ValueError(
                f"primary_eval_reduction={self.primary_eval_reduction!r} must be one of {sorted(VALID_REDUCTIONS)}."
            )
        if not self.group_by:
            raise ValueError("EvalCriteria.group_by must be non-empty.")
        if not 0.0 <= self.trim < 0.5:
            raise ValueError(f"EvalCriteria.trim must be in [0, 0.5), got {self.trim!r}.")

    def worst_score(self) -> float:
        """Return the sentinel score used for failed candidates.

        Returns:
            ``float('inf')`` when :attr:`lower_is_better`, ``float('-inf')``
            otherwise — so failed candidates always sort last regardless
            of optimisation direction.
        """
        return float("inf") if self.lower_is_better else float("-inf")

    def evaluate_metrics(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the per-group metrics frame for one candidate.

        Args:
            prediction_df: Long-format DataFrame containing at least
                ``actual`` and ``forecast`` columns plus every column in
                :attr:`group_by`. Typically the ``result_df`` of a
                :class:`~abvelocity.ts.backfill.result.BackfillResult`.

        Returns:
            DataFrame with one row per :attr:`group_by` combination and
            one column per eval metric in :attr:`eval_metrics`. Plus the
            bookkeeping ``n`` column (number of rows in the group)
            emitted by :func:`compute_eval`.

        Raises:
            ValueError: If ``actual`` / ``forecast`` columns are missing
                or if any ``group_by`` column is absent.
        """
        missing = [c for c in self.group_by if c not in prediction_df.columns]
        if missing:
            raise ValueError(
                f"prediction_df is missing group_by columns: {missing}. "
                f"Available columns: {list(prediction_df.columns)}."
            )
        return compute_eval(
            prediction_df,
            metrics=self.eval_metrics,
            group_by=self.group_by,
            trim=self.trim,
        )

    def primary_score(self, metrics_df: pd.DataFrame) -> float:
        """Reduce the per-group metrics frame to a single ranking scalar.

        Args:
            metrics_df: Output of :meth:`evaluate_metrics`.

        Returns:
            Single scalar: :attr:`primary_eval_metric` aggregated across
            the per-group rows by :attr:`primary_eval_reduction`.
            :meth:`worst_score` if the column is empty or fully ``NaN``
            (so failed candidates always sort last).
        """
        if self.primary_eval_metric not in metrics_df.columns:
            raise ValueError(
                f"primary_eval_metric={self.primary_eval_metric!r} not found in metrics_df columns "
                f"{list(metrics_df.columns)}."
            )
        col = metrics_df[self.primary_eval_metric]
        if col.dropna().empty:
            return self.worst_score()
        if self.primary_eval_reduction == "mean":
            return float(col.mean())
        if self.primary_eval_reduction == "median":
            return float(col.median())
        if self.primary_eval_reduction == "max":
            return float(col.max())
        if self.primary_eval_reduction == "min":
            return float(col.min())
        if self.primary_eval_reduction == "sum":
            return float(col.sum())
        raise ValueError(f"Unhandled primary_eval_reduction={self.primary_eval_reduction!r}.")

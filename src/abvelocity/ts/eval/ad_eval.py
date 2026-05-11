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
"""Anomaly-detection evaluation — ADEval (class) + compute_ad_eval (function).

Classification metrics for AD runs: precision / recall / F1 / accuracy
using sklearn, plus soft variants parameterized by a tolerance window
(``soft_window > 0`` allows matches within ±window time-steps).

Soft scoring follows the same semantics as
``ts/gk/detection/common/ad_evaluation.py`` but exposed here as a single
scalar per metric for the positive (True = anomaly) class.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from abvelocity.ts.constants import METRIC_ID_COL
from sklearn import metrics as sk_metrics

VALID_AD_METRICS = frozenset({"precision", "recall", "f1", "accuracy", "soft_precision", "soft_recall", "soft_f1"})

DEFAULT_AD_METRICS: Tuple[str, ...] = ("precision", "recall", "f1")


def _expand_with_window(arr: np.ndarray, window: int) -> np.ndarray:
    """Return a boolean array where position i is True if any of
    ``arr[i-window..i+window]`` is True. Used for soft AD scoring.

    Edge effects: shifts past the array boundary contribute ``False``
    (matches the ``ffill``/``bfill`` behavior of the gk soft scorer only
    in the interior — we keep edges unfilled to avoid double-counting).
    """
    if window <= 0:
        return arr.astype(bool)
    s = pd.Series(arr)
    expanded = np.zeros(len(arr), dtype=bool)
    for k in range(-window, window + 1):
        expanded |= s.shift(k).fillna(False).astype(bool).to_numpy()
    return expanded


def compute_ad_eval(
    df: pd.DataFrame,
    metric_names: Tuple[str, ...] = DEFAULT_AD_METRICS,
    group_by: Optional[Tuple[str, ...]] = None,
    true_col: str = "is_anomaly",
    pred_col: str = "is_anomaly_predicted",
    soft_window: int = 0,
) -> pd.DataFrame:
    """Compute AD classification metrics on a TSResult ``result_df``.

    Args:
        df: Long-format DataFrame with ground-truth + predicted anomaly
            columns. Rows where either column is ``NaN`` are excluded.
        metric_names: Tuple of metric names. Supported:
            ``"precision"``, ``"recall"``, ``"f1"``, ``"accuracy"``,
            ``"soft_precision"``, ``"soft_recall"``, ``"soft_f1"``.
            Raises ``ValueError`` for unknown names.
        group_by: Columns to group by. ``None`` → single-row result.
        true_col: Name of the ground-truth boolean column.
        pred_col: Name of the predicted boolean column.
        soft_window: Tolerance window for soft_* metrics (time-steps).
            0 → strict. Ignored for non-soft metrics.

    Returns:
        DataFrame with one row per group (or one row total if ``group_by``
        is ``None``) and one column per requested metric. Reports the
        positive (``True``) class.
    """
    unknown = set(metric_names) - VALID_AD_METRICS
    if unknown:
        raise ValueError(f"Unknown AD eval metrics: {sorted(unknown)}. " f"Supported: {sorted(VALID_AD_METRICS)}.")
    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"{true_col!r} and {pred_col!r} must both be present in df.columns={list(df.columns)}")

    needs_soft = any(m.startswith("soft_") for m in metric_names)

    def _compute(sub: pd.DataFrame) -> dict:
        s = sub.dropna(subset=[true_col, pred_col])
        if s.empty:
            return {m: float("nan") for m in metric_names}
        y_true = s[true_col].astype(bool).to_numpy()
        y_pred = s[pred_col].astype(bool).to_numpy()
        out = {}
        if "precision" in metric_names:
            out["precision"] = float(sk_metrics.precision_score(y_true, y_pred, zero_division=0))
        if "recall" in metric_names:
            out["recall"] = float(sk_metrics.recall_score(y_true, y_pred, zero_division=0))
        if "f1" in metric_names:
            out["f1"] = float(sk_metrics.f1_score(y_true, y_pred, zero_division=0))
        if "accuracy" in metric_names:
            out["accuracy"] = float(sk_metrics.accuracy_score(y_true, y_pred))
        if needs_soft:
            y_true_soft = _expand_with_window(y_true, soft_window)
            y_pred_soft = _expand_with_window(y_pred, soft_window)
            if "soft_precision" in metric_names:
                out["soft_precision"] = float(sk_metrics.precision_score(y_true_soft, y_pred, zero_division=0))
            if "soft_recall" in metric_names:
                out["soft_recall"] = float(sk_metrics.recall_score(y_true, y_pred_soft, zero_division=0))
            if "soft_f1" in metric_names:
                sp = float(sk_metrics.precision_score(y_true_soft, y_pred, zero_division=0))
                sr = float(sk_metrics.recall_score(y_true, y_pred_soft, zero_division=0))
                out["soft_f1"] = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0.0
        return out

    if not group_by:
        return pd.DataFrame([_compute(df)])

    rows = []
    for key, sub in df.groupby(list(group_by), dropna=False):
        row = _compute(sub)
        if not isinstance(key, tuple):
            key = (key,)
        for col, val in zip(group_by, key):
            row[col] = val
        rows.append(row)
    out_df = pd.DataFrame(rows)
    ordered = list(group_by) + [c for c in out_df.columns if c not in group_by]
    return out_df[ordered]


@dataclass
class ADEval:
    """High-level anomaly-detection evaluator.

    Wraps :func:`compute_ad_eval` with a parameterized configuration.
    Pass an instance to :attr:`TSFlowConfig.eval` to have
    :meth:`TSFlow.compute_eval` invoke :meth:`run` automatically.

    Attributes:
        metrics: Metrics to compute — subset of :data:`VALID_AD_METRICS`.
        group_by: Columns to group by. ``None`` → single-row result.
        true_col: Ground-truth boolean column name.
        pred_col: Predicted boolean column name.
        soft_window: Tolerance window for soft_* metrics (time-steps).
    """

    metrics: Tuple[str, ...] = DEFAULT_AD_METRICS
    group_by: Optional[Tuple[str, ...]] = (METRIC_ID_COL,)
    true_col: str = "is_anomaly"
    pred_col: str = "is_anomaly_predicted"
    soft_window: int = 0

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the configured AD metrics on ``df``."""
        return compute_ad_eval(
            df,
            metric_names=self.metrics,
            group_by=self.group_by,
            true_col=self.true_col,
            pred_col=self.pred_col,
            soft_window=self.soft_window,
        )

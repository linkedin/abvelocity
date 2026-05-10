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
"""Tests for ADEval and compute_ad_eval."""

import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.eval.ad_eval import ADEval, _expand_with_window, compute_ad_eval


@pytest.fixture
def single_group_df() -> pd.DataFrame:
    """6-row perfect-recall case, precision = 3/4 = 0.75, recall = 3/3 = 1.0."""
    return pd.DataFrame(
        {
            "metric_id": ["m"] * 6,
            "is_anomaly": [True, False, True, False, True, False],
            "is_anomaly_predicted": [True, True, True, False, True, False],
        }
    )


@pytest.fixture
def two_group_df() -> pd.DataFrame:
    """Two groups with different per-group metrics."""
    return pd.DataFrame(
        {
            "metric_id": ["a", "a", "a", "b", "b", "b"],
            "is_anomaly": [True, False, True, True, False, False],
            "is_anomaly_predicted": [True, False, True, False, True, False],
        }
    )


# ---------------------------------------------------------------------------
# compute_ad_eval — functional API
# ---------------------------------------------------------------------------


def test_compute_ad_eval_basic_precision_recall_f1(single_group_df):
    out = compute_ad_eval(single_group_df, metric_names=("precision", "recall", "f1"))
    assert out.shape == (1, 3)
    assert out["recall"].iloc[0] == 1.0
    assert out["precision"].iloc[0] == pytest.approx(0.75)
    assert out["f1"].iloc[0] == pytest.approx(2 * 0.75 * 1.0 / (0.75 + 1.0))


def test_compute_ad_eval_accuracy(single_group_df):
    out = compute_ad_eval(single_group_df, metric_names=("accuracy",))
    # 5 of 6 correct (row index 1 is False predicted True).
    assert out["accuracy"].iloc[0] == pytest.approx(5 / 6)


def test_compute_ad_eval_group_by(two_group_df):
    out = compute_ad_eval(two_group_df, metric_names=("precision", "recall"), group_by=("metric_id",))
    assert len(out) == 2
    row_a = out[out["metric_id"] == "a"].iloc[0]
    row_b = out[out["metric_id"] == "b"].iloc[0]
    # Group a: 2 TP / 2 predicted positives → precision=1; 2 TP / 2 actual → recall=1
    assert row_a["precision"] == 1.0
    assert row_a["recall"] == 1.0
    # Group b: 1 predicted positive, 0 TP → precision=0; 0 TP / 1 actual → recall=0
    assert row_b["precision"] == 0.0
    assert row_b["recall"] == 0.0


def test_compute_ad_eval_custom_col_names():
    df = pd.DataFrame({"y_true": [True, False], "y_hat": [True, True]})
    out = compute_ad_eval(df, metric_names=("precision",), true_col="y_true", pred_col="y_hat")
    assert out["precision"].iloc[0] == pytest.approx(0.5)


def test_compute_ad_eval_drops_nan_rows():
    df = pd.DataFrame(
        {
            "is_anomaly": [True, True, None, True],
            "is_anomaly_predicted": [True, False, True, True],
        }
    )
    out = compute_ad_eval(df, metric_names=("recall",))
    # Valid rows: (T,T), (T,F), (T,T) → recall = 2/3
    assert out["recall"].iloc[0] == pytest.approx(2 / 3)


def test_compute_ad_eval_empty_group_returns_nan():
    df = pd.DataFrame({"is_anomaly": [None, None], "is_anomaly_predicted": [True, False]})
    out = compute_ad_eval(df, metric_names=("precision", "recall"))
    assert np.isnan(out["precision"].iloc[0])
    assert np.isnan(out["recall"].iloc[0])


def test_compute_ad_eval_rejects_unknown_metric():
    df = pd.DataFrame({"is_anomaly": [True], "is_anomaly_predicted": [True]})
    with pytest.raises(ValueError, match="Unknown AD eval metrics"):
        compute_ad_eval(df, metric_names=("fake_metric",))


def test_compute_ad_eval_missing_cols_raises():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="must both be present"):
        compute_ad_eval(df)


# ---------------------------------------------------------------------------
# Soft metrics + _expand_with_window helper
# ---------------------------------------------------------------------------


def test_expand_with_window_zero_is_noop():
    arr = np.array([True, False, True])
    out = _expand_with_window(arr, window=0)
    assert (out == arr).all()


def test_expand_with_window_expands_neighbors():
    arr = np.array([False, True, False, False, True, False])
    # Window=1 → each True extends ±1 step.
    out = _expand_with_window(arr, window=1)
    assert list(out) == [True, True, True, True, True, True]


def test_compute_ad_eval_soft_recall_recovers_late_prediction():
    # Prediction shifted +1 step: strict recall=0, soft_recall (window=1) = 1.
    df = pd.DataFrame(
        {
            "is_anomaly": [False, True, False, False, True, False],
            "is_anomaly_predicted": [False, False, True, False, False, True],
        }
    )
    out_strict = compute_ad_eval(df, metric_names=("recall",))
    assert out_strict["recall"].iloc[0] == 0.0
    out_soft = compute_ad_eval(df, metric_names=("soft_recall",), soft_window=1)
    assert out_soft["soft_recall"].iloc[0] == 1.0


def test_compute_ad_eval_soft_f1_matches_harmonic_mean():
    df = pd.DataFrame(
        {
            "is_anomaly": [True, False, False, True, False],
            "is_anomaly_predicted": [False, True, False, True, False],
        }
    )
    out = compute_ad_eval(df, metric_names=("soft_precision", "soft_recall", "soft_f1"), soft_window=1)
    sp = out["soft_precision"].iloc[0]
    sr = out["soft_recall"].iloc[0]
    expected_f1 = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0.0
    assert out["soft_f1"].iloc[0] == pytest.approx(expected_f1)


# ---------------------------------------------------------------------------
# ADEval — class wrapper
# ---------------------------------------------------------------------------


def test_ad_eval_defaults(single_group_df):
    evaluator = ADEval()
    out = evaluator.run(single_group_df)
    assert {"precision", "recall", "f1"}.issubset(out.columns)
    assert "metric_id" in out.columns  # default group_by


def test_ad_eval_with_custom_metrics_and_group_by(two_group_df):
    evaluator = ADEval(metrics=("precision",), group_by=("metric_id",))
    out = evaluator.run(two_group_df)
    assert set(out.columns) == {"metric_id", "precision"}
    assert len(out) == 2


def test_ad_eval_with_soft_window():
    df = pd.DataFrame(
        {
            "metric_id": ["m"] * 6,
            "is_anomaly": [False, True, False, False, True, False],
            "is_anomaly_predicted": [False, False, True, False, False, True],
        }
    )
    evaluator = ADEval(metrics=("soft_recall",), soft_window=1)
    out = evaluator.run(df)
    assert out["soft_recall"].iloc[0] == 1.0


def test_ad_eval_with_custom_cols():
    df = pd.DataFrame({"y_true": [True, True], "y_hat": [True, False]})
    evaluator = ADEval(metrics=("recall",), group_by=None, true_col="y_true", pred_col="y_hat")
    out = evaluator.run(df)
    assert out["recall"].iloc[0] == pytest.approx(0.5)

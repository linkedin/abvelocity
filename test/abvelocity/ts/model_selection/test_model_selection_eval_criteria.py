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
"""Tests for EvalCriteria."""

import math

import pandas as pd
import pytest
from abvelocity.ts.constants import ACTUAL_COL, FORECAST_COL, HORIZON_STEP_COL, METRIC_ID_COL
from abvelocity.ts.model_selection.eval_criteria import EvalCriteria


def make_prediction_df(actuals, forecasts, metric_id="y"):
    n = len(actuals)
    return pd.DataFrame(
        {
            METRIC_ID_COL: [metric_id] * n,
            HORIZON_STEP_COL: list(range(1, n + 1)),
            ACTUAL_COL: actuals,
            FORECAST_COL: forecasts,
        }
    )


def test_default_criteria_construction():
    criteria = EvalCriteria()
    assert "mape" in criteria.eval_metrics
    assert criteria.primary_eval_metric == "mape"
    assert criteria.primary_eval_reduction == "mean"
    assert criteria.lower_is_better is True


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown eval metrics"):
        EvalCriteria(eval_metrics=("not_a_metric",), primary_eval_metric="not_a_metric")


def test_primary_metric_must_be_in_metrics():
    with pytest.raises(ValueError, match="must be in"):
        EvalCriteria(eval_metrics=("mape",), primary_eval_metric="rmse")


def test_invalid_reduction_raises():
    with pytest.raises(ValueError, match="primary_eval_reduction"):
        EvalCriteria(primary_eval_reduction="geometric_mean")


def test_evaluate_metrics_returns_per_group_columns():
    # Build two horizon steps; actual and forecast deviate by 10 each.
    df = make_prediction_df(actuals=[100.0, 200.0], forecasts=[90.0, 220.0])
    criteria = EvalCriteria(eval_metrics=("mae", "mape"), primary_eval_metric="mape")
    metrics_df = criteria.evaluate_metrics(df)
    assert {"mae", "mape", METRIC_ID_COL, HORIZON_STEP_COL} <= set(metrics_df.columns)
    # Two horizon steps -> two rows.
    assert len(metrics_df) == 2
    # MAE per row is |10|, |20|.
    assert sorted(metrics_df["mae"].tolist()) == [10.0, 20.0]


def test_primary_score_mean_reduction():
    df = make_prediction_df(actuals=[100.0, 100.0], forecasts=[90.0, 110.0])
    criteria = EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape", primary_eval_reduction="mean")
    metrics_df = criteria.evaluate_metrics(df)
    score = criteria.primary_score(metrics_df)
    # Both rows have MAPE=10%; mean = 10.
    assert math.isclose(score, 10.0, rel_tol=1e-9)


def test_primary_score_max_reduction():
    df = make_prediction_df(actuals=[100.0, 100.0], forecasts=[80.0, 95.0])
    criteria = EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape", primary_eval_reduction="max")
    metrics_df = criteria.evaluate_metrics(df)
    score = criteria.primary_score(metrics_df)
    # Row 0: MAPE=20%; row 1: 5%; max = 20.
    assert math.isclose(score, 20.0, rel_tol=1e-9)


def test_primary_score_returns_inf_when_all_nan():
    df = pd.DataFrame(
        {
            METRIC_ID_COL: ["y"],
            HORIZON_STEP_COL: [1],
            ACTUAL_COL: [0.0],   # MAPE undefined when actual=0; gets dropped.
            FORECAST_COL: [10.0],
        }
    )
    criteria = EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape")
    metrics_df = criteria.evaluate_metrics(df)
    score = criteria.primary_score(metrics_df)
    assert score == float("inf")


def test_worst_score_inverts_with_lower_is_better():
    asc = EvalCriteria(lower_is_better=True)
    desc = EvalCriteria(eval_metrics=("r2",), primary_eval_metric="r2", lower_is_better=False)
    assert asc.worst_score() == float("inf")
    assert desc.worst_score() == float("-inf")


def test_evaluate_metrics_missing_group_by_column_raises():
    df = make_prediction_df([100.0], [110.0])
    df = df.drop(columns=[HORIZON_STEP_COL])
    criteria = EvalCriteria()
    with pytest.raises(ValueError, match="missing group_by columns"):
        criteria.evaluate_metrics(df)


def test_eval_criteria_trim_default_is_one_percent():
    from abvelocity.ts.eval.forecast_eval import DEFAULT_TRIM

    assert EvalCriteria().trim == DEFAULT_TRIM
    assert DEFAULT_TRIM == 0.01


def test_eval_criteria_trim_invalid_raises():
    with pytest.raises(ValueError, match=r"trim must be in \[0, 0.5\)"):
        EvalCriteria(trim=0.5)
    with pytest.raises(ValueError, match=r"trim must be in \[0, 0.5\)"):
        EvalCriteria(trim=-0.01)

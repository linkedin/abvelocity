import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.common.constants import TIME_COL, VALUE_COL
from abvelocity.ts.common.evaluation import mean_absolute_percent_error
from abvelocity.ts.gk.sklearn.estimator.null_model import DummyEstimator
from abvelocity.ts.gk.sklearn.sklearn_scorer import _PredictScorerDF, make_scorer_df
from sklearn.metrics import mean_absolute_error


def test_predict_score_df():
    """Tests _PredictScorerDF by checking whether it can
    properly score a DummyEstimator
    """
    periods = 20
    model = DummyEstimator()
    X = pd.DataFrame(
        {TIME_COL: pd.date_range("2018-01-01", periods=periods, freq="D"), VALUE_COL: np.arange(periods)}  # the first value is 0, so MAPE will divide by 0
    )
    model.fit(X)

    def method_caller(estimator, method, *args, **kwargs):
        """Call estimator with method and args and kwargs."""
        return getattr(estimator, method)(*args, **kwargs)

    with pytest.warns(Warning, match="Score is undefined for this split, setting to `np.nan`."):
        scorer = _PredictScorerDF(mean_absolute_percent_error, 1, {})
        score = scorer._score(method_caller, model, X, X[VALUE_COL])
        assert np.isnan(score)

        scorer = _PredictScorerDF(mean_absolute_percent_error, -1, {})
        score = scorer._score(method_caller, model, X, X[VALUE_COL])
        assert np.isnan(score)

    scorer = _PredictScorerDF(mean_absolute_error, -1, {})
    score = scorer._score(method_caller, model, X, X[VALUE_COL])
    model.predict(X)
    assert score == -5.0  # MAE of 9.5 vs [0, 1, 2, ..., 19]


def test_make_scorer_df():
    """Tests make_scorer_df"""
    scorer = make_scorer_df("r2", greater_is_better=True, some_kwarg=True)
    assert isinstance(scorer, _PredictScorerDF)
    assert scorer._sign == 1
    assert scorer._kwargs == {"some_kwarg": True}

    kwargs = {"kwarg1": True, "kwarg2": "val"}
    scorer = make_scorer_df(mean_absolute_percent_error, greater_is_better=False, **kwargs)
    assert isinstance(scorer, _PredictScorerDF)
    assert scorer._sign == -1
    assert scorer._kwargs == kwargs

import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.common import constants as cst
from abvelocity.ts.common.constants import FRACTION_OUTSIDE_TOLERANCE, LOGGER_NAME
from abvelocity.ts.common.evaluation import EvaluationMetricEnum, add_preaggregation_to_scorer, fraction_outside_tolerance
from abvelocity.ts.common.python_utils import assert_equal
from abvelocity.ts.common.testing_utils import assert_eval_function_equal, generate_df_with_reg_for_tests
from abvelocity.ts.gk.framework.constants import CUSTOM_SCORE_FUNC_NAME, CV_REPORT_METRICS_ALL, FRACTION_OUTSIDE_TOLERANCE_NAME
from abvelocity.ts.gk.framework.pipeline.utils import (
    get_basic_pipeline,
    get_best_index,
    get_default_time_parameters,
    get_forecast,
    get_hyperparameter_searcher,
    get_score_func_with_aggregation,
    get_scoring_and_refit,
)
from abvelocity.ts.gk.framework.utils.framework_testing_utils import assert_proper_grid_search, assert_refit, assert_scoring
from abvelocity.ts.gk.sklearn.cross_validation import RollingTimeSeriesSplit
from abvelocity.ts.gk.sklearn.estimator.null_model import DummyEstimator
from abvelocity.ts.gk.sklearn.estimator.silverkite_estimator import SilverkiteEstimator
from scipy.stats import randint as sp_randint
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.pipeline import Pipeline
from testfixtures import LogCapture

from abvelocity.ts.gk_test_gate import gk_test_gate

pytestmark = gk_test_gate


def test_get_best_index():
    """Tests `get_best_index`"""
    results = {
        "mean_test_score": np.array([1.0, 3.0, 2.0]),
        "mean_test_MSE": np.array([3.0, 1.0, -2.0]),
    }
    assert get_best_index(results) == 0
    assert get_best_index(results=results, metric="MSE", greater_is_better=True) == 0
    assert get_best_index(results=results, metric="MSE", greater_is_better=False) == 2


def test_get_default_time_parameters():
    """Tests get_default_time_parameters function"""
    # enough data to support forecast_horizon=test_horizon=cv_horizon
    one_hour = 3600.0
    num_observations = 100
    time_params = get_default_time_parameters(
        period=one_hour,
        num_observations=num_observations,
        forecast_horizon=None,
        test_horizon=None,
        periods_between_train_test=None,
        cv_horizon=None,
        cv_min_train_periods=None,
        cv_expanding_window=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=0,
        cv_max_splits=3,
    )
    assert time_params == {
        "forecast_horizon": 24,
        "test_horizon": 24,
        "periods_between_train_test": 0,
        "cv_horizon": 24,
        "cv_min_train_periods": None,
        "cv_periods_between_train_test": 0,
    }

    # default CV split will be used
    num_observations = 100
    time_params = get_default_time_parameters(
        period=24 * one_hour,
        num_observations=num_observations,
        forecast_horizon=None,
        test_horizon=None,
        periods_between_train_test=None,
        cv_horizon=None,
        cv_min_train_periods=None,
        cv_expanding_window=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=0,
        cv_max_splits=3,
    )
    assert time_params == {
        "forecast_horizon": 30,
        "test_horizon": 30,
        "periods_between_train_test": 0,
        "cv_horizon": 30,
        "cv_min_train_periods": None,
        "cv_periods_between_train_test": 0,
    }

    # default CV split will be used
    num_observations = 20
    time_params = get_default_time_parameters(
        period=24 * one_hour,
        num_observations=num_observations,
        forecast_horizon=None,
        test_horizon=None,
        cv_horizon=None,
        periods_between_train_test=None,
        cv_min_train_periods=4,
        cv_expanding_window=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=1,
        cv_max_splits=3,
    )
    assert time_params == {
        "forecast_horizon": 10,
        "test_horizon": 10,
        "periods_between_train_test": 0,
        "cv_horizon": 10,
        "cv_min_train_periods": 4,
        "cv_periods_between_train_test": 1,
    }

    # default CV split will be used
    num_observations = 20
    time_params = get_default_time_parameters(
        period=24 * one_hour,
        num_observations=num_observations,
        forecast_horizon=None,
        test_horizon=None,
        periods_between_train_test=10,
        cv_horizon=None,
        cv_min_train_periods=10,
        cv_expanding_window=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=21,
        cv_max_splits=3,
    )
    assert time_params == {
        "forecast_horizon": 10,
        "test_horizon": 10,
        "periods_between_train_test": 10,
        "cv_horizon": 10,
        "cv_min_train_periods": 10,
        "cv_periods_between_train_test": 21,
    }

    # default CV split will be used
    num_observations = 20
    time_params = get_default_time_parameters(
        period=24 * one_hour,
        num_observations=num_observations,
        forecast_horizon=None,
        test_horizon=None,
        periods_between_train_test=5,
        cv_horizon=None,
        cv_min_train_periods=8,
        cv_expanding_window=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=None,
        cv_max_splits=3,
    )
    assert time_params == {
        "forecast_horizon": 10,
        "test_horizon": 10,
        "periods_between_train_test": 5,
        "cv_horizon": 10,
        "cv_min_train_periods": 8,
        "cv_periods_between_train_test": 5,
    }


def test_get_basic_pipeline_custom():
    """Tests get_basic_pipeline with custom estimator"""
    pipeline = get_basic_pipeline(
        estimator=SilverkiteEstimator(),
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        score_func_greater_is_better=False,
        agg_periods=10,
        agg_func=np.sum,
        relative_error_tolerance=None,
        coverage=None,
        null_model_params={"strategy": "mean"},
    )

    expected_score_func, _, _ = get_score_func_with_aggregation(
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_func(), agg_periods=10, agg_func=np.sum, greater_is_better=False
    )

    # checks estimator parameters
    assert isinstance(pipeline.steps[-1][-1], SilverkiteEstimator)
    assert pipeline.steps[-1][-1].fit_algorithm_dict is None
    assert pipeline.steps[-1][-1].extra_pred_cols is None
    assert pipeline.steps[-1][-1].coverage is None
    assert pipeline.steps[-1][-1].null_model_params["strategy"] == "mean"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_eval_function_equal(pipeline.steps[-1][-1].score_func, expected_score_func)


def test_get_basic_pipeline_apply_reg():
    """Tests get_basic_pipeline fit and predict methods on
    a dataset with regressors, and checks if pipeline parameters
    can be set.
    """
    df = generate_df_with_reg_for_tests("D", 50)
    # adds degenerate columns
    df["train_df"]["cst1"] = "constant"
    df["train_df"]["cst2"] = 1.0
    df["test_df"]["cst1"] = "constant"
    df["test_df"]["cst2"] = 1.0
    pipeline = get_basic_pipeline(
        estimator=SilverkiteEstimator(),
        score_func=EvaluationMetricEnum.MeanSquaredError.name,
        score_func_greater_is_better=False,
        agg_periods=None,
        agg_func=None,
        relative_error_tolerance=None,
        coverage=0.95,
        null_model_params=None,
        regressor_cols=["regressor1", "regressor2", "regressor3", "regressor_bool", "regressor_categ", "cst1", "cst2"],
    )
    pipeline.fit(df["train_df"])
    assert pipeline.named_steps["degenerate"].drop_cols == []
    pipeline.predict(df["test_df"])

    # drops degenerate columns, normalizes
    pipeline.set_params(
        degenerate__drop_degenerate=True,
        input__regressors_numeric__normalize__normalize_algorithm="PowerTransformer",
    )
    pipeline.fit(df["train_df"])
    # (column order is swapped by column selectors and feature union)
    assert pipeline.named_steps["degenerate"].drop_cols == ["cst2", "cst1"]
    predictions = pipeline.predict(df["test_df"])
    assert predictions.shape[0] == df["test_df"].shape[0]

    with pytest.raises(ValueError, match="Invalid parameter"):
        pipeline.set_params(
            degenerate__drop_degenerate=True,
            input__regressors_numeric__normalize__unknown_param="PowerTransformer",
        )


def test_get_score_func_with_aggregation():
    """Tests get_score_func_with_aggregation function"""
    # tests callable score function
    score_func = mean_absolute_error
    greater_is_better = False
    score_func, greater_is_better, short_name = get_score_func_with_aggregation(
        score_func, greater_is_better=greater_is_better, agg_periods=None, agg_func=None
    )
    assert_eval_function_equal(score_func, mean_absolute_error)
    assert greater_is_better is False
    assert short_name == CUSTOM_SCORE_FUNC_NAME

    # tests `EvaluationMetricEnum` string lookup
    score_func = "MedianAbsoluteError"
    greater_is_better = True  # should be overridden
    score_func, greater_is_better, short_name = get_score_func_with_aggregation(
        score_func, greater_is_better=greater_is_better, agg_periods=None, agg_func=None
    )
    assert_eval_function_equal(score_func, median_absolute_error)
    assert greater_is_better is False
    assert short_name == EvaluationMetricEnum.MedianAbsoluteError.get_metric_name()

    # tests `FRACTION_OUTSIDE_TOLERANCE_NAME` lookup
    score_func = FRACTION_OUTSIDE_TOLERANCE
    greater_is_better = True  # should be overridden
    score_func, greater_is_better, short_name = get_score_func_with_aggregation(
        score_func, greater_is_better=greater_is_better, agg_periods=None, agg_func=None, relative_error_tolerance=0.02
    )
    assert_eval_function_equal(score_func, partial(fraction_outside_tolerance, rtol=0.02))
    assert greater_is_better is False
    assert short_name == FRACTION_OUTSIDE_TOLERANCE_NAME

    # tests exception
    with pytest.raises(NotImplementedError, match=r"Evaluation metric.*not available"):
        get_score_func_with_aggregation("unknown_estimator")

    with pytest.raises(ValueError, match="Must specify `relative_error_tolerance` to request " "FRACTION_OUTSIDE_TOLERANCE as a metric."):
        get_score_func_with_aggregation(score_func=FRACTION_OUTSIDE_TOLERANCE)
    with pytest.raises(ValueError, match="`score_func` must be an `EvaluationMetricEnum` member name, " "FRACTION_OUTSIDE_TOLERANCE, or callable."):
        get_score_func_with_aggregation(score_func=["wrong_type"])

    # tests preaggregation on score function
    with pytest.warns(UserWarning) as record:
        score_func, greater_is_better, short_name = get_score_func_with_aggregation(
            "MeanAbsoluteError", greater_is_better=False, agg_periods=3, agg_func=np.sum
        )
        assert_eval_function_equal(score_func, add_preaggregation_to_scorer(mean_absolute_error, agg_periods=3, agg_func=np.sum))
        assert greater_is_better is False
        assert short_name == EvaluationMetricEnum.MeanAbsoluteError.get_metric_name()

        y_true = pd.Series([3, 1, np.nan, 3, np.inf])  # np.nan and np.inf are ignored
        y_pred = pd.Series([1, 4, 100, 2, -2])
        assert score_func(y_true, y_pred) == 0.0  # 7 vs 7
        assert "Requested agg_periods=3, but there are only 1. Using all for aggregation" in record[0].message.args[0]
        assert "2 value(s) in y_true were NA or infinite and are omitted in error calc." in record[4].message.args[0]


def test_get_scoring_and_refit():
    """Tests `get_scoring_and_refit`"""
    enum = EvaluationMetricEnum.MeanAbsolutePercentError
    scoring, refit = get_scoring_and_refit()
    assert_refit(refit, expected_metric=enum.get_metric_name(), expected_greater_is_better=enum.get_metric_greater_is_better())
    expected_keys = {enum.get_metric_name()}
    assert_scoring(scoring=scoring, expected_keys=expected_keys)

    # Tests all parameters where `score_func_greater_is_better=True`,
    # `score_func` is contained in `cv_report_metrics`,
    # and `cv_report_metrics=CV_REPORT_METRICS_ALL`.
    enum = EvaluationMetricEnum.Correlation
    agg_periods = 7
    agg_func = np.sum
    relative_error_tolerance = 0.025
    scoring, refit = get_scoring_and_refit(
        score_func=enum.name,
        score_func_greater_is_better=enum.get_metric_greater_is_better(),
        cv_report_metrics=CV_REPORT_METRICS_ALL,
        agg_periods=agg_periods,
        agg_func=agg_func,
        relative_error_tolerance=relative_error_tolerance,
    )
    assert_refit(refit, expected_metric=enum.get_metric_name(), expected_greater_is_better=enum.get_metric_greater_is_better())
    enum_names = set(enum.get_metric_name() for enum in EvaluationMetricEnum)
    assert_scoring(
        scoring=scoring,
        expected_keys=enum_names | {FRACTION_OUTSIDE_TOLERANCE_NAME},
        agg_periods=agg_periods,
        agg_func=agg_func,
        relative_error_tolerance=relative_error_tolerance,
    )

    # score_func is a callable,
    # `cv_report_metrics=CV_REPORT_METRICS_ALL`,
    # and `relative_error_tolerance=None`
    relative_error_tolerance = None
    scoring, refit = get_scoring_and_refit(
        score_func=mean_absolute_error,
        score_func_greater_is_better=False,
        cv_report_metrics=CV_REPORT_METRICS_ALL,
        agg_periods=None,
        agg_func=None,
        relative_error_tolerance=relative_error_tolerance,
    )
    assert_refit(refit, expected_metric=CUSTOM_SCORE_FUNC_NAME, expected_greater_is_better=False)  # custom name for callable
    assert_scoring(
        scoring=scoring,
        expected_keys=enum_names | {CUSTOM_SCORE_FUNC_NAME},  # does not include `FRACTION_OUTSIDE_TOLERANCE_NAME`
        agg_periods=None,
        agg_func=None,
        relative_error_tolerance=relative_error_tolerance,
    )
    assert_eval_function_equal(scoring[CUSTOM_SCORE_FUNC_NAME]._score_func, mean_absolute_error)

    # `score_func=FRACTION_OUTSIDE_TOLERANCE`, cv_report_metrics is a list
    relative_error_tolerance = 0.025
    cv_report_metrics = [EvaluationMetricEnum.MeanAbsolutePercentError.name, EvaluationMetricEnum.MeanSquaredError.name]
    scoring, refit = get_scoring_and_refit(
        score_func=FRACTION_OUTSIDE_TOLERANCE,
        score_func_greater_is_better=False,
        cv_report_metrics=cv_report_metrics,
        agg_periods=None,
        agg_func=None,
        relative_error_tolerance=relative_error_tolerance,
    )
    assert_refit(refit, expected_metric=FRACTION_OUTSIDE_TOLERANCE_NAME, expected_greater_is_better=False)
    assert_scoring(
        scoring=scoring,
        expected_keys={
            EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_name(),
            EvaluationMetricEnum.MeanSquaredError.get_metric_name(),
            FRACTION_OUTSIDE_TOLERANCE_NAME,
        },
        agg_periods=None,
        agg_func=None,
        relative_error_tolerance=relative_error_tolerance,
    )


def test_get_hyperparameter_searcher():
    """Tests get_hyperparameter_searcher"""
    model = DummyEstimator()
    with LogCapture(LOGGER_NAME) as log_capture:
        hyperparameter_grid = {
            "strategy": ["mean", "median", "quantile", "constant"],
            "constant": [20.0],
            "quantile": [0.8],
        }
        grid_search = get_hyperparameter_searcher(hyperparameter_grid=hyperparameter_grid, model=model)
        assert grid_search.n_iter == 4
        scoring, refit = get_scoring_and_refit()
        assert_scoring(scoring=grid_search.scoring, expected_keys=scoring.keys())
        assert grid_search.n_jobs == 1
        assert_refit(grid_search.refit, expected_metric=EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_name(), expected_greater_is_better=False)
        assert grid_search.cv is None
        assert grid_search.verbose == 1
        assert grid_search.pre_dispatch == "2*n_jobs"
        assert grid_search.return_train_score
        log_capture.check((LOGGER_NAME, "DEBUG", "Setting hyperparameter_budget to 4 for full grid search."))

    with LogCapture(LOGGER_NAME) as log_capture:
        # specifies `get_scoring_and_refit` kwargs, uses a distribution
        hyperparameter_grid = [
            {
                "strategy": ["mean", "median", "quantile", "constant"],
                "constant": [20.0],
                "quantile": [0.8],
            },
            {"strategy": ["constant"], "constant": sp_randint(1, 3, 4)},
        ]
        grid_search = get_hyperparameter_searcher(
            hyperparameter_grid=hyperparameter_grid,
            model=model,
            cv=4,
            hyperparameter_budget=None,
            n_jobs=4,
            verbose=2,
            score_func=EvaluationMetricEnum.Quantile95.name,
            cv_report_metrics=CV_REPORT_METRICS_ALL,
        )

        assert grid_search.n_iter == 10
        enum_names = set(enum.get_metric_name() for enum in EvaluationMetricEnum)
        assert_scoring(scoring=grid_search.scoring, expected_keys=enum_names)
        assert grid_search.n_jobs == 4
        assert_refit(grid_search.refit, expected_metric=EvaluationMetricEnum.Quantile95.get_metric_name(), expected_greater_is_better=False)
        assert grid_search.cv == 4
        assert grid_search.verbose == 2
        assert grid_search.pre_dispatch == "2*n_jobs"
        assert grid_search.return_train_score
        log_capture.check((LOGGER_NAME, "WARNING", "Setting hyperparameter_budget to 10 to sample from " "provided distributions (and lists)."))

    with LogCapture(LOGGER_NAME) as log_capture:
        # specifies RollingTimeSeriesSplit `cv`, no logging messages
        hyperparameter_grid = [
            {
                "strategy": ["mean", "median", "quantile", "constant"],
                "constant": [20.0],
                "quantile": [0.8],
            },
            {"strategy": ["constant"], "constant": sp_randint(1, 30)},
        ]
        hyperparameter_budget = 3
        cv = RollingTimeSeriesSplit(forecast_horizon=3)
        grid_search = get_hyperparameter_searcher(
            hyperparameter_grid=hyperparameter_grid, model=model, cv=cv, hyperparameter_budget=hyperparameter_budget, n_jobs=4, verbose=2
        )

        assert grid_search.n_iter == hyperparameter_budget
        assert grid_search.n_jobs == 4
        assert isinstance(grid_search.cv, RollingTimeSeriesSplit)
        assert grid_search.verbose == 2
        assert grid_search.pre_dispatch == "2*n_jobs"
        assert grid_search.return_train_score
        log_capture.check()


def run_dummy_grid_search(hyperparameter_grid, n_jobs=1, **kwargs):
    """Runs a pandas.DataFrame through hyperparameter_grid search
    with custom CV splits on a simple dataset to show that
    all the pieces fit together.

    Parameters
    ----------
    hyperparameter_grid : `dict` or `list` [`dict`]
        Passed to ``get_hyperparameter_searcher``.
        Should be compatible with DummyEstimator
    n_jobs : `int` or None, default=-1
        Passed to ``get_hyperparameter_searcher``
    kwargs : additional parameters
        Passed to ``get_hyperparameter_searcher``

    Returns
    -------
    grid_search : `~sklearn.model_selection.RandomizedSearchCV`
        Grid search output (fitted RandomizedSearchCV object).
    """
    # dummy dataset, model, and CV splitter
    periods = 10
    X = pd.DataFrame({cst.TIME_COL: pd.date_range("2018-01-01", periods=periods, freq="D"), cst.VALUE_COL: np.arange(1, periods + 1)})
    model = DummyEstimator()
    cv = RollingTimeSeriesSplit(forecast_horizon=3)  # 1 CV split

    # requested grid searcher
    grid_search = get_hyperparameter_searcher(hyperparameter_grid=hyperparameter_grid, model=model, cv=cv, n_jobs=n_jobs, **kwargs)

    grid_search.fit(X, X[cst.VALUE_COL])  # need to pass in y to evaluate score() function
    return grid_search


def test_run_hyperparameter_searcher():
    """Tests running hyperparameter_grid search using
    `get_hyperparameter_searcher` output
    """
    # Grid search with explicit hyperparameter_grid and standard `score_func`
    hyperparameter_grid = [
        {
            "strategy": ["mean", "median", "quantile"],
            "quantile": [0.8],
        },
        {"strategy": ["constant"], "constant": [1, 3, 4, 5, 10]},
    ]
    with pytest.warns(UserWarning) as record:
        # full hyperparameter_grid search
        metric = EvaluationMetricEnum.MeanAbsolutePercentError
        grid_search = run_dummy_grid_search(hyperparameter_grid, score_func=metric.name)
        assert_proper_grid_search(
            grid_search,
            expected_grid_size=8,
            lower_bound=0.0,  # MAPE is reported with original sign
            score_func=metric.name,
            greater_is_better=metric.get_metric_greater_is_better(),
        )
        # limited hyperparameter_grid search
        grid_search = run_dummy_grid_search(hyperparameter_grid, score_func=metric.name, hyperparameter_budget=1)
        assert_proper_grid_search(
            grid_search,
            expected_grid_size=1,
            lower_bound=0.0,  # MAPE is repored with original sign
            score_func=metric.name,
            greater_is_better=metric.get_metric_greater_is_better(),
        )
        all_warning_msg = " ".join([record[i].message.args[0] for i in range(len(record))])
        assert "There is only one CV split" in all_warning_msg

    # Grid search with random hyperparameter_grid,
    # custom `score_func`, and additional report metrics
    random_hyperparameter_grid = [
        {
            "strategy": ["mean", "median", "quantile"],
            "quantile": [0.8],
        },
        {"strategy": ["constant"], "constant": sp_randint(1, 30)},
    ]
    with pytest.warns(UserWarning, match="There is only one CV split"):
        # full hyperparameter_grid search, default 10
        grid_search = run_dummy_grid_search(
            random_hyperparameter_grid,
            score_func=mean_squared_error,
            score_func_greater_is_better=False,
            agg_periods=7,
            agg_func=np.sum,
            cv_report_metrics=CV_REPORT_METRICS_ALL,
            relative_error_tolerance=0.05,
            n_jobs=-1,
        )  # test parallelism
        expected_names = [enum.get_metric_name() for enum in EvaluationMetricEnum] + [CUSTOM_SCORE_FUNC_NAME, FRACTION_OUTSIDE_TOLERANCE_NAME]
        assert_proper_grid_search(
            grid_search, expected_grid_size=10, lower_bound=0.0, score_func=mean_squared_error, greater_is_better=False, cv_report_metrics_names=expected_names
        )
        # checks if custom metrics are properly calculated
        assert_equal(
            grid_search.cv_results_[f"mean_test_{FRACTION_OUTSIDE_TOLERANCE_NAME}"],
            grid_search.cv_results_[f"mean_test_{EvaluationMetricEnum.FractionOutsideTolerance5.get_metric_name()}"],
        )
        assert_equal(
            grid_search.cv_results_[f"mean_test_{CUSTOM_SCORE_FUNC_NAME}"],
            grid_search.cv_results_[f"mean_test_{EvaluationMetricEnum.MeanSquaredError.get_metric_name()}"],
        )

        # limited hyperparameter_grid search
        grid_search = run_dummy_grid_search(
            random_hyperparameter_grid,
            hyperparameter_budget=2,
            score_func=mean_squared_error,
            agg_periods=7,
            agg_func=np.sum,
            cv_report_metrics=CV_REPORT_METRICS_ALL,
            relative_error_tolerance=0.05,
            n_jobs=-1,
        )
        assert_proper_grid_search(
            grid_search, expected_grid_size=2, lower_bound=0.0, score_func=mean_squared_error, greater_is_better=False, cv_report_metrics_names=expected_names
        )


def test_get_forecast():
    """Tests get_forecast function"""
    X = pd.DataFrame({cst.TIME_COL: pd.date_range("2018-01-01", periods=10, freq="D"), cst.VALUE_COL: np.arange(10)})
    # coverage is sufficient to request uncertainty interval,
    # even with ``uncertainty_dict=None``
    coverage = 0.95

    # test forecast with bands
    trained_model = Pipeline([("estimator", SilverkiteEstimator(coverage=coverage))])
    trained_model.fit(X, X[cst.VALUE_COL])

    with pytest.warns(UserWarning) as record:
        forecast = get_forecast(X, trained_model, relative_error_tolerance=0.01)
        assert forecast.df.shape == (X.shape[0], 5)
        assert forecast.time_col == cst.TIME_COL
        assert forecast.actual_col == cst.ACTUAL_COL
        assert forecast.predicted_col == cst.PREDICTED_COL
        assert forecast.predicted_lower_col == cst.PREDICTED_LOWER_COL
        assert forecast.predicted_upper_col == cst.PREDICTED_UPPER_COL
        assert forecast.null_model_predicted_col is None  # there is no null model by default
        assert forecast.ylabel == cst.VALUE_COL
        assert forecast.train_end_date == X[cst.TIME_COL].max()
        assert forecast.forecast_horizon is None
        assert forecast.coverage == coverage
        assert forecast.r2_loss_function == mean_squared_error
        assert forecast.estimator
        assert forecast.relative_error_tolerance == 0.01
        assert "y_true contains 0. MAPE is undefined." in record[0].message.args[0]
        assert "y_true contains 0. MedAPE is undefined." in record[1].message.args[0]

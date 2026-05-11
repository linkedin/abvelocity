import datetime

import abvelocity.ts.common.constants as cst
import pytest
from abvelocity.ts.common.testing_utils import generate_df_for_tests
from abvelocity.ts.gk.sklearn.estimator.one_by_one_estimator import OneByOneEstimator
from abvelocity.ts.gk.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator


@pytest.fixture
def daily_data():
    return generate_df_for_tests(freq="D", periods=730, train_start_date=datetime.datetime(2018, 1, 1), conti_year_origin=2018)


def test_no_coverage(daily_data):
    """Tests no coverage is provided."""
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=[1, 2],
        coverage=None,
        estimator_params={"autoreg_dict": "auto", "yearly_seasonality": 2, "feature_sets_enabled": False},
    )
    train_df = daily_data["train_df"]
    model.fit(train_df, time_col=cst.TIME_COL, value_col=cst.VALUE_COL)
    test_df = daily_data["test_df"]
    predict = model.predict(test_df.iloc[:3])
    assert cst.PREDICTED_LOWER_COL not in predict.columns
    assert cst.PREDICTED_UPPER_COL not in predict.columns


def test_forecast_one_by_one_not_activated(daily_data):
    """Tests forecast one-by-one is not activated when no parameter
    depends on forecast horizon.
    """
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=[1, 2],
        estimator_params={"autoreg_dict": None, "yearly_seasonality": 2, "feature_sets_enabled": False},
    )
    train_df = daily_data["train_df"]
    model.fit(train_df, time_col=cst.TIME_COL, value_col=cst.VALUE_COL)
    assert len(model.estimators) == 1
    assert model.estimator_map_list == [3]
    assert model.pred_indices is None


def test_set_params():
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=1,
        estimator_params={"autoreg_dict": "auto", "yearly_seasonality": 2, "feature_sets_enabled": False},
    )
    assert model.estimator_map == 1
    assert model.estimator_params["yearly_seasonality"] == 2
    assert model.estimator_params.get("daily_seasonality") is None

    model.set_params(**{"estimator_map": 2, "yearly_seasonality": 4, "daily_seasonality": 2})
    assert model.estimator_map == 2
    assert model.estimator_params["yearly_seasonality"] == 4
    assert model.estimator_params["daily_seasonality"] == 2

    assert set(model.estimator_param_names) == set(SimpleSilverkiteEstimator().get_params().keys())

    with pytest.raises(
        ValueError,
        match=r"Invalid parameter some_param for estimator OneByOneEstimator. "
        r"Check the list of available parameters with "
        r"`estimator.get\_params\(\).keys\(\)`.",
    ):
        model.set_params(**{"some_param": 5})


def test_errors(daily_data):
    """Tests errors."""
    model = OneByOneEstimator(
        estimator="SimpleSilverkiteEstimator",
        forecast_horizon=3,
        estimator_map=[1, 1],
        estimator_params={"autoreg_dict": "auto", "yearly_seasonality": 2, "feature_sets_enabled": False},
    )
    train_df = daily_data["train_df"]
    with pytest.raises(ValueError, match="Sum of forecast one by one estimator map must equal to forecast horizon."):
        model.fit(train_df, time_col=cst.TIME_COL, value_col=cst.VALUE_COL)

    model = OneByOneEstimator(
        estimator="SomeEstimator",
        forecast_horizon=3,
        estimator_map=[1, 1],
        estimator_params={"autoreg_dict": "auto", "yearly_seasonality": 2, "feature_sets_enabled": False},
    )
    train_df = daily_data["train_df"]
    with pytest.raises(ValueError, match="Estimator SomeEstimator does not support forecast one-by-one."):
        model.fit(train_df, time_col=cst.TIME_COL, value_col=cst.VALUE_COL)

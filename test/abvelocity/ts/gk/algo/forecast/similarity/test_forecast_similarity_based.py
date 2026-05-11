import plotly.graph_objects as go
from abvelocity.ts.gk.algo.forecast.similarity.forecast_similarity_based import forecast_similarity_based
from abvelocity.ts.common.evaluation import EvaluationMetricEnum, calc_pred_err
from abvelocity.ts.common.testing_utils import generate_df_for_tests


from abvelocity.ts.gk_test_gate import gk_test_gate

pytestmark = gk_test_gate


def test_forecast_similarity_based():
    """Testing the function: forecast_similarity_based in various examples"""
    data = generate_df_for_tests(freq="D", periods=30 * 8)  # 8 months
    df = data["df"]
    train_df = data["train_df"]
    test_df = data["test_df"]

    df["z"] = df["y"] + 1
    train_df["z"] = train_df["y"] + 1
    test_df["z"] = test_df["y"] + 1

    train_df = train_df[["ts", "y", "z"]]
    test_df = test_df[["ts", "y", "z"]]

    res = forecast_similarity_based(df=train_df, time_col="ts", value_cols=["y", "z"], agg_method="median", agg_func=None, match_cols=["dow"])

    # forecast using predict
    fdf_median = res["predict"](test_df)
    assert (fdf_median["z"] - fdf_median["y"] - 1.0).abs().max().round(2) == 0.0, "forecast for z must be forecast for y + 1 at each timestamp"
    err = calc_pred_err(test_df["y"], fdf_median["y"])
    enum = EvaluationMetricEnum.Correlation
    assert err[enum.get_metric_name()] > 0.3

    # forecast using predict_n
    fdf_median = res["predict_n"](test_df.shape[0])
    err = calc_pred_err(test_df["y"], fdf_median["y"])
    assert err[enum.get_metric_name()] > 0.3

    res = forecast_similarity_based(df=train_df, time_col="ts", value_cols=["y", "z"], agg_method="mean", agg_func=None, match_cols=["dow"])

    # forecast using the mean of all similar times
    fdf_mean = res["predict"](test_df)
    err = calc_pred_err(test_df["y"], fdf_mean["y"])
    assert err[enum.get_metric_name()] > 0.3

    res = forecast_similarity_based(df=train_df, time_col="ts", value_cols=["y", "z"], agg_method="most_recent", agg_func=None, match_cols=["dow"], recent_k=3)

    # forecast using the mean of 3 recent times similar to the given time
    fdf_recent3_mean = res["predict"](test_df)
    err = calc_pred_err(test_df["y"], fdf_recent3_mean["y"])
    assert err[enum.get_metric_name()] > 0.3

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"].dt.strftime("%Y-%m-%d"), y=df["y"], name="true", opacity=0.5, mode="lines"))
    fig.add_trace(go.Scatter(x=fdf_median["ts"].dt.strftime("%Y-%m-%d"), y=fdf_median["y"], opacity=0.5, name="median pred wrt dow", mode="lines"))
    fig.add_trace(go.Scatter(x=fdf_mean["ts"].dt.strftime("%Y-%m-%d"), y=fdf_mean["y"], opacity=0.5, name="mean pred wrt dow", mode="lines"))
    fig.add_trace(
        go.Scatter(x=fdf_recent3_mean["ts"].dt.strftime("%Y-%m-%d"), y=fdf_recent3_mean["y"], opacity=0.5, name="mean recent 3 wrt dow", mode="lines")
    )

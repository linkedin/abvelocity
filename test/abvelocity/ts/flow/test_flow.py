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
"""Tests for TSFlow."""

from unittest.mock import MagicMock, patch

# Register simple algo so ForecastRunner can resolve it.
import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401
import pandas as pd
import pytest
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import FORECAST_COL, METRIC_ID_COL, TIME_COL
from abvelocity.ts.eval import ADEval, ForecastEval
from abvelocity.ts.flow.flow import TSFlow
from abvelocity.ts.flow.flow_config import TSFlowConfig
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig
from abvelocity.ts.result.forecast_result import ForecastResult
from abvelocity.ts.testing_utils import make_daily_series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ts_metrics_config() -> TSMetricsConfig:
    return TSMetricsConfig(time_col="event_ts", freq="D", dialect="duckdb")


def make_forecast_config(horizon: int = 7) -> ForecastConfig:
    return ForecastConfig(
        algo_name="simple",
        forecast_horizon=horizon,
        value_cols=("dau",),
        algo_params={"period": 7, "k": 3},
    )


def make_flow_config(mode: str = "forecast", **kwargs) -> TSFlowConfig:
    if mode == "backfill":
        bc = BackfillConfig(
            forecast_config=make_forecast_config(),
            initial_train_size=30,
            horizon=7,
        )
        return TSFlowConfig(
            ts_metrics_config=make_ts_metrics_config(),
            mode="backfill",
            backfill_config=bc,
            **kwargs,
        )
    return TSFlowConfig(
        ts_metrics_config=make_ts_metrics_config(),
        ts_model_config=make_forecast_config(),
        mode=mode,
        **kwargs,
    )


def make_wide_df(n_days: int = 60) -> pd.DataFrame:
    return make_daily_series(n_days=n_days, col="dau")


def make_metric_info(name: str = "dau_family") -> MagicMock:
    mi = MagicMock()
    mi.metric_family.name = name
    mi.metrics = None
    return mi


# ---------------------------------------------------------------------------
# fetch_data raises without cursor
# ---------------------------------------------------------------------------


def test_fetch_data_raises_without_io_param():
    flow = TSFlow(flow_config=make_flow_config())
    with pytest.raises(ValueError, match="io_param with a cursor"):
        flow.fetch_data(make_metric_info())


def test_fetch_data_raises_without_cursor():
    from abvelocity.core.param.io_param import IOParam

    flow = TSFlow(flow_config=make_flow_config(), io_param=IOParam(cursor=None))
    with pytest.raises(ValueError, match="io_param with a cursor"):
        flow.fetch_data(make_metric_info())


# ---------------------------------------------------------------------------
# run_algo dispatch
# ---------------------------------------------------------------------------


def test_run_algo_forecast_returns_forecast_result():
    df = make_wide_df(60)
    flow = TSFlow(flow_config=make_flow_config(mode="forecast"))
    result = flow.run_algo(df)
    assert isinstance(result, ForecastResult)
    assert result.result_df is not None
    assert FORECAST_COL in result.result_df.columns


def test_run_algo_backfill_returns_backfill_result():
    from abvelocity.ts.backfill.result import BackfillResult

    df = make_wide_df(60)
    flow = TSFlow(flow_config=make_flow_config(mode="backfill"))
    result = flow.run_algo(df)
    assert isinstance(result, BackfillResult)
    assert result.result_df is not None


def test_run_algo_detect_dispatches_to_anomaly_runner():
    from abvelocity.ts.config.detect_config import DetectConfig
    from abvelocity.ts.result.detect_result import DetectResult

    detect_cfg = DetectConfig(forecast_config=make_forecast_config(), algo_name="")
    flow_cfg = TSFlowConfig(
        ts_metrics_config=make_ts_metrics_config(),
        ts_model_config=detect_cfg,
        mode="detect",
    )
    flow = TSFlow(flow_config=flow_cfg)
    df = make_wide_df(60)

    fake_result = DetectResult(result_df=pd.DataFrame(), anomalies_df=pd.DataFrame())
    with patch(
        "abvelocity.ts.flow.flow.AnomalyDetectRunner.run",
        return_value=fake_result,
    ):
        result = flow.run_algo(df)
    assert isinstance(result, DetectResult)


# ---------------------------------------------------------------------------
# compute_eval
# ---------------------------------------------------------------------------


def test_compute_eval_returns_none_when_no_eval():
    flow = TSFlow(flow_config=make_flow_config())
    assert flow.compute_eval(pd.DataFrame()) is None


def test_compute_eval_returns_none_when_result_df_is_none():
    flow = TSFlow(flow_config=make_flow_config(eval=ForecastEval(metrics=("mae",))))
    assert flow.compute_eval(None) is None


def test_compute_eval_returns_df_for_valid_result():
    df = make_wide_df(60)
    flow = TSFlow(flow_config=make_flow_config(mode="backfill", eval=ForecastEval(metrics=("mae", "rmse"))))
    backfill_result = flow.run_algo(df)
    eval_df = flow.compute_eval(backfill_result.result_df)
    assert eval_df is not None
    assert "mae" in eval_df.columns
    assert "rmse" in eval_df.columns


def test_compute_eval_dispatches_ad_metrics_in_detect_mode():
    """Passing ADEval on flow_config.eval computes precision/recall/f1."""
    flow = TSFlow(flow_config=make_flow_config(mode="detect", eval=ADEval(metrics=("precision", "recall", "f1"))))
    detect_df = pd.DataFrame(
        {
            "metric_id": ["m"] * 6,
            "is_anomaly": [True, False, True, False, True, False],
            "is_anomaly_predicted": [True, False, False, False, True, True],
        }
    )
    eval_df = flow.compute_eval(detect_df)
    assert eval_df is not None
    assert {"precision", "recall", "f1"}.issubset(eval_df.columns)


def test_compute_eval_ad_soft_metrics():
    """Soft AD metrics computed with soft_window > 0."""
    flow = TSFlow(
        flow_config=make_flow_config(
            mode="detect",
            eval=ADEval(metrics=("soft_precision", "soft_recall", "soft_f1"), soft_window=1),
        )
    )
    # Prediction is 1 step late; soft_window=1 recovers full recall.
    detect_df = pd.DataFrame(
        {
            "metric_id": ["m"] * 6,
            "is_anomaly": [False, True, False, False, True, False],
            "is_anomaly_predicted": [False, False, True, False, False, True],
        }
    )
    eval_df = flow.compute_eval(detect_df)
    assert eval_df is not None
    assert float(eval_df["soft_recall"].iloc[0]) == 1.0


# ---------------------------------------------------------------------------
# run (end-to-end with pre-fetched df)
# ---------------------------------------------------------------------------


def test_run_with_pre_fetched_df():
    df = make_wide_df(60)
    flow = TSFlow(flow_config=make_flow_config(mode="forecast"))
    result = flow.run(make_metric_info(), df=df)
    assert result.result_df is not None
    assert FORECAST_COL in result.result_df.columns
    assert result.eval_df is None  # eval not set


def test_run_includes_eval_when_requested():
    df = make_wide_df(60)
    flow = TSFlow(flow_config=make_flow_config(mode="backfill", eval=ForecastEval(metrics=("mae",))))
    result = flow.run(make_metric_info(), df=df)
    assert result.result_df is not None
    assert result.eval_df is not None
    assert "mae" in result.eval_df.columns


def test_run_prediction_window_filters_rows():
    df = make_wide_df(90)
    flow = TSFlow(
        flow_config=make_flow_config(
            mode="forecast",
            prediction_window=("2023-03-01", "2023-03-31"),
        )
    )
    result = flow.run(make_metric_info(), df=df)
    assert result.result_df is not None
    ts = result.result_df[TIME_COL]
    assert (ts >= pd.Timestamp("2023-03-01")).all()
    assert (ts <= pd.Timestamp("2023-03-31")).all()


def test_run_fetches_data_when_df_not_provided():
    """When df=None, run() should call fetch_data (which raises here — no cursor)."""
    flow = TSFlow(flow_config=make_flow_config(mode="forecast"))
    with pytest.raises(ValueError, match="io_param with a cursor"):
        flow.run(make_metric_info())


# ---------------------------------------------------------------------------
# fetch_data + full end-to-end with DuckDB
# ---------------------------------------------------------------------------


def test_fetch_data_and_run_with_duckdb():
    """Full pipeline: DuckDB fetch → SimpleForecastAlgo → ForecastResult."""
    import numpy as np
    from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
    from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
    from abvelocity.core.param.io_param import IOParam
    from abvelocity.core.param.metric import Metric, UMetric
    from abvelocity.core.param.metric_family import MetricFamily
    from abvelocity.core.param.metric_info import MetricInfo

    # --- synthetic event data: one row per (member, day) ---
    rng = np.random.default_rng(0)
    n_days, n_users = 60, 20
    start = pd.Timestamp("2024-01-01")
    rows = []
    for d in range(n_days):
        date = start + pd.Timedelta(days=d)
        for uid in range(n_users):
            rows.append(
                {
                    "member_id": uid,
                    "event_ts": date,
                    "signup": int(rng.binomial(1, 0.3)),
                }
            )
    events_df = pd.DataFrame(rows)  # noqa: F841 — referenced by DuckDB replacement scan

    # --- DuckDB setup ---
    cursor = DuckDBCursor()
    cursor._db_connection.execute("CREATE TABLE signups AS SELECT * FROM events_df")

    # --- metric definitions ---
    u_signup = UMetric(col="signup", agg="SUM", name="signup")
    signup_metric = Metric(numerator=u_signup, numerator_agg="SUM", name="total_signups")
    family = MetricFamily(
        name="signup_family",
        u_metrics_query=UMetricsQuery(table_name="signups", date_col="event_ts"),
        metric_join_unit_col="member_id",
    )
    metric_info = MetricInfo(
        metric_family=family,
        metrics=[signup_metric],
        start_date="2024-01-01",
        end_date="2024-02-29",
    )

    # --- flow config ---
    fc = ForecastConfig(
        algo_name="simple",
        forecast_horizon=7,
        value_cols=("total_signups",),
        freq="D",
        algo_params={"period": 7, "k": 3},
    )
    flow_cfg = TSFlowConfig(
        ts_metrics_config=TSMetricsConfig(time_col="event_ts", freq="D", dialect="duckdb"),
        ts_model_config=fc,
        mode="forecast",
    )

    # --- run ---
    flow = TSFlow(flow_config=flow_cfg, io_param=IOParam(cursor=cursor))
    result = flow.run(metric_info)

    assert result.result_df is not None
    assert FORECAST_COL in result.result_df.columns
    assert "total_signups" in result.result_df[METRIC_ID_COL].values
    # training rows + 7 forecast rows
    assert len(result.result_df) == n_days + 7

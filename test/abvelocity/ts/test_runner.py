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
"""Tests for TSRunner."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pytest
from abvelocity.ts.algo.base import ALGO_REGISTRY, TSAlgo
from abvelocity.ts.config.ts_model_config import TSModelConfig
from abvelocity.ts.constants import ALGO_NAME_COL, LAST_TRAINING_DATE_COL
from abvelocity.ts.result.ts_result import TSResult
from abvelocity.ts.runner import TSRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SENTINEL_RESULT = TSResult(fit_info={"sentinel": True})


@dataclass
class TrackingAlgo(TSAlgo):
    """Records the arguments received by fit() and predict()."""

    last_fit_df: Optional[pd.DataFrame] = None
    last_fit_config: Optional[TSModelConfig] = None
    last_fit_anomaly_df: Optional[pd.DataFrame] = None
    last_predict_prediction_window: Optional[tuple[str, str]] = None

    def fit(
        self,
        df: pd.DataFrame,
        config: TSModelConfig,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> "TrackingAlgo":
        self.last_fit_df = df
        self.last_fit_config = config
        self.last_fit_anomaly_df = anomaly_df
        return self

    def predict(
        self,
        df: Optional[pd.DataFrame] = None,
        prediction_window: Optional[tuple[str, str]] = None,
    ) -> TSResult:
        self.last_predict_prediction_window = prediction_window
        return SENTINEL_RESULT


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_unknown_algo_raises_value_error():
    cfg = TSModelConfig(algo_name="__nonexistent_algo_xyz__")
    runner = TSRunner(config=cfg)
    df = pd.DataFrame({"ts": [], "value": []})
    with pytest.raises(ValueError, match="__nonexistent_algo_xyz__"):
        runner.run(df=df)


def test_runner_error_message_lists_available_algos(monkeypatch):
    """The ValueError message must include names of registered algos."""
    monkeypatch.setitem(ALGO_REGISTRY, "_test_known_runner", TrackingAlgo)
    cfg = TSModelConfig(algo_name="__unknown_algo_abc__")
    runner = TSRunner(config=cfg)
    df = pd.DataFrame({"ts": [], "value": []})
    with pytest.raises(ValueError, match="_test_known_runner"):
        runner.run(df=df)


def test_runner_calls_fit_then_predict(monkeypatch):
    """run() must call fit() then predict() and return predict()'s result."""
    monkeypatch.setitem(ALGO_REGISTRY, "_tracking", TrackingAlgo)
    cfg = TSModelConfig(algo_name="_tracking")
    runner = TSRunner(config=cfg)

    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3, freq="D"), "value": [1.0, 2.0, 3.0]})
    result = runner.run(df=df)

    assert result is SENTINEL_RESULT


def test_runner_passes_prediction_window(monkeypatch):
    monkeypatch.setitem(ALGO_REGISTRY, "_tracking2", TrackingAlgo)
    cfg = TSModelConfig(algo_name="_tracking2")
    runner = TSRunner(config=cfg)

    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3, freq="D"), "value": range(3)})
    window = ("2024-01-04", "2024-01-07")

    # We need to capture the algo instance; patch _get_algo to intercept
    algo_instance = TrackingAlgo()
    monkeypatch.setattr(runner, "get_algo", lambda: algo_instance)

    runner.run(df=df, prediction_window=window)
    assert algo_instance.last_predict_prediction_window == window


def test_runner_passes_anomaly_df(monkeypatch):
    cfg = TSModelConfig(algo_name="_tracking3")
    runner = TSRunner(config=cfg)

    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, freq="D"), "value": range(5)})
    anomaly_df = pd.DataFrame({"start_ts": ["2024-01-02"], "end_ts": ["2024-01-03"]})

    algo_instance = TrackingAlgo()
    monkeypatch.setattr(runner, "get_algo", lambda: algo_instance)

    runner.run(df=df, anomaly_df=anomaly_df)
    assert algo_instance.last_fit_anomaly_df is anomaly_df


def test_runner_passes_config_to_fit(monkeypatch):
    cfg = TSModelConfig(algo_name="_tracking4", value_cols=("y",), freq="H")
    runner = TSRunner(config=cfg)

    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3, freq="h"), "y": [1.0, 2.0, 3.0]})

    algo_instance = TrackingAlgo()
    monkeypatch.setattr(runner, "get_algo", lambda: algo_instance)

    runner.run(df=df)
    assert algo_instance.last_fit_config is cfg


def test_runner_stamps_model_name_and_last_training_date(monkeypatch):
    """run() stamps model_name and last_training_date onto result_df."""

    @dataclass
    class ResultAlgo(TSAlgo):
        def fit(self, df, config, anomaly_df=None):
            return self

        def predict(self, df=None, prediction_window=None):
            return TSResult(result_df=pd.DataFrame({"ts": ["2024-01-01"], "metric_id": ["v"], "actual": [1.0]}))

    monkeypatch.setitem(ALGO_REGISTRY, "_stamping", ResultAlgo)
    cfg = TSModelConfig(algo_name="_stamping", train_end_date="2024-01-01")
    runner = TSRunner(config=cfg)
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=1, freq="D"), "value": [1.0]})
    result = runner.run(df=df)
    assert result.result_df is not None
    assert (result.result_df[ALGO_NAME_COL] == "_stamping").all()
    assert (result.result_df[LAST_TRAINING_DATE_COL] == "2024-01-01").all()


def test_runner_stamps_full_pipeline_columns(monkeypatch):
    """run() adds the full scheduled-pipeline column set to result_df.

    Metadata (metric_id via template, metric_name via template, algo_name,
    algo_version) is pulled from ForecastConfig; derived columns (stage,
    forecasted_date, last_training_date) are computed from existing
    fields; components + std + extras are NaN/None when the algo didn't
    populate them.
    """
    # Imports inside the test to keep this test self-contained — the rest
    # of the file's imports don't need these constants.
    from abvelocity.ts.config.forecast_config import ForecastConfig
    from abvelocity.ts.constants import (
        ALGO_VERSION_COL,
        ANNUAL_SEASONALITY_COL,
        DAILY_SEASONALITY_COL,
        EXTRAS_COL,
        FORECASTED_DATE_COL,
        HOLIDAY_IMPACT_COL,
        LONGTERM_GROWTH_COL,
        METRIC_ID_COL,
        METRIC_NAME_COL,
        RESIDUAL_COL,
        RUN_DATE_COL,
        RUN_ID_COL,
        SHORTTERM_GROWTH_COL,
        STAGE_COL,
        STAGE_FITTED,
        STAGE_FORECAST,
        STD_COL,
        WEEKLY_SEASONALITY_COL,
    )

    @dataclass
    class TinyAlgo(TSAlgo):
        def fit(self, df, config, anomaly_df=None):
            return self

        def predict(self, df=None, prediction_window=None):
            return TSResult(
                result_df=pd.DataFrame(
                    {
                        "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                        "metric_id": ["signups", "signups", "signups"],
                        "actual": [10.0, 11.0, None],  # last row is future
                        "point_forecast": [10.1, 11.1, 12.5],
                        "ci_low": [9.0, 10.0, 11.0],
                        "ci_high": [11.0, 12.0, 14.0],
                    }
                )
            )

    monkeypatch.setitem(ALGO_REGISTRY, "_pipeline", TinyAlgo)
    cfg = ForecastConfig(
        algo_name="_pipeline",
        algo_version="v1",
        train_end_date="2024-01-02",
        forecast_horizon=1,
        metric_id_template="randomProduct_signups_daily:country=US",  # scalar (no placeholders)
        metric_name_template="RandomProduct Daily Signups (US)",
    )
    runner = TSRunner(config=cfg)
    result = runner.run(df=pd.DataFrame({"ts": [], "value": []}))
    assert result.result_df is not None
    rdf = result.result_df

    # metric_id / metric_name rendered from scalar templates (no placeholders).
    assert (rdf[METRIC_ID_COL] == "randomProduct_signups_daily:country=US").all()
    assert (rdf[METRIC_NAME_COL] == "RandomProduct Daily Signups (US)").all()
    # algo_name / algo_version stamped as scalars from config.
    assert (rdf[ALGO_NAME_COL] == "_pipeline").all()
    assert (rdf[ALGO_VERSION_COL] == "v1").all()
    assert (rdf[LAST_TRAINING_DATE_COL] == "2024-01-02").all()

    # Derived columns.
    assert (
        rdf[FORECASTED_DATE_COL]
        == [
            pd.Timestamp("2024-01-01").date(),
            pd.Timestamp("2024-01-02").date(),
            pd.Timestamp("2024-01-03").date(),
        ]
    ).all()
    assert rdf[STAGE_COL].tolist() == [STAGE_FITTED, STAGE_FITTED, STAGE_FORECAST]

    # Null-populated columns (algo didn't produce them).
    for col in (
        STD_COL,
        LONGTERM_GROWTH_COL,
        SHORTTERM_GROWTH_COL,
        DAILY_SEASONALITY_COL,
        WEEKLY_SEASONALITY_COL,
        ANNUAL_SEASONALITY_COL,
        HOLIDAY_IMPACT_COL,
        RESIDUAL_COL,
    ):
        assert rdf[col].isna().all(), f"{col} should be NaN by default"
    assert (rdf[EXTRAS_COL].isna() | (rdf[EXTRAS_COL] == {}) | rdf[EXTRAS_COL].isnull()).all()

    # Run-identity columns are caller-set; the runner leaves them empty.
    assert rdf[RUN_ID_COL].isna().all()
    assert rdf[RUN_DATE_COL].isna().all()


def test_runner_metric_id_template_expansion(monkeypatch):
    """metric_id_template with {value_col} + dim placeholders expands per row."""
    from abvelocity.ts.config.forecast_config import ForecastConfig
    from abvelocity.ts.constants import METRIC_ID_COL, METRIC_NAME_COL

    @dataclass
    class MultiAlgo(TSAlgo):
        def fit(self, df, config, anomaly_df=None):
            return self

        def predict(self, df=None, prediction_window=None):
            return TSResult(
                result_df=pd.DataFrame(
                    {
                        "ts": pd.to_datetime(["2024-01-01"] * 4),
                        "metric_id": ["signups", "signups", "revenue", "revenue"],
                        "country": ["US", "CA", "US", "CA"],
                        "actual": [1.0, 2.0, 3.0, 4.0],
                    }
                )
            )

    monkeypatch.setitem(ALGO_REGISTRY, "_multi", MultiAlgo)
    cfg = ForecastConfig(
        algo_name="_multi",
        train_end_date="2024-01-01",
        forecast_horizon=1,
        dim_cols=("country",),
        metric_id_template="randomProduct_daily:m={value_col}|country={country}",
        metric_name_template="RandomProduct Daily — {value_col} ({country})",
    )
    rdf = TSRunner(config=cfg).run(df=pd.DataFrame({"ts": [], "value": []})).result_df
    assert rdf[METRIC_ID_COL].tolist() == [
        "randomProduct_daily:m=signups|country=US",
        "randomProduct_daily:m=signups|country=CA",
        "randomProduct_daily:m=revenue|country=US",
        "randomProduct_daily:m=revenue|country=CA",
    ]
    assert rdf[METRIC_NAME_COL].tolist() == [
        "RandomProduct Daily — signups (US)",
        "RandomProduct Daily — signups (CA)",
        "RandomProduct Daily — revenue (US)",
        "RandomProduct Daily — revenue (CA)",
    ]


def test_forecast_config_rejects_unknown_placeholders():
    """__post_init__ validation surfaces config errors before run()."""
    from abvelocity.ts.config.forecast_config import ForecastConfig

    with pytest.raises(ValueError, match="unknown placeholders.*device"):
        ForecastConfig(
            algo_name="noop",
            dim_cols=("country",),
            metric_id_template="x:country={country}|device={device}",  # device not in dim_cols
        )


def test_runner_algo_name_default_when_no_metric_id_template(monkeypatch):
    """When ForecastConfig.metric_id_template is None the algo's stamp is preserved;
    algo_name in result_df always matches config.algo_name (no fallback needed)."""
    from abvelocity.ts.config.forecast_config import ForecastConfig

    @dataclass
    class NoopAlgo(TSAlgo):
        def fit(self, df, config, anomaly_df=None):
            return self

        def predict(self, df=None, prediction_window=None):
            return TSResult(result_df=pd.DataFrame({"ts": ["2024-01-01"], "metric_id": ["v"], "actual": [1.0]}))

    monkeypatch.setitem(ALGO_REGISTRY, "_noop", NoopAlgo)
    cfg = ForecastConfig(algo_name="_noop", train_end_date="2024-01-01")
    result = TSRunner(config=cfg).run(df=pd.DataFrame({"ts": [], "value": []}))
    from abvelocity.ts.constants import METRIC_ID_COL

    # algo stamped metric_id="v"; no template → preserved.
    assert (result.result_df[METRIC_ID_COL] == "v").all()
    assert (result.result_df[ALGO_NAME_COL] == "_noop").all()


def test_runner_get_algo_passes_algo_params(monkeypatch):
    """_get_algo() passes algo_params from config to the constructor."""
    monkeypatch.setitem(ALGO_REGISTRY, "_tracking5", TrackingAlgo)
    params = {"model_template": "SILVERKITE"}
    cfg = TSModelConfig(algo_name="_tracking5", algo_params=params)
    runner = TSRunner(config=cfg)
    algo = runner.get_algo()
    assert algo.algo_params == params

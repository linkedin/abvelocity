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
"""TSFlow: class-based timeseries pipeline (fetch → algo → eval).

Each :meth:`TSFlow.run` call chains three steps for a single metric group:

1. **Fetch** (:meth:`fetch_data`): builds a
   :class:`~abvelocity.ts.get_data.ts_metrics_query.TSMetricsQuery`
   from ``metric_info`` and ``flow_config.ts_metrics_config``, then executes
   it via ``io_param.cursor`` to obtain a wide-format DataFrame.
   Skipped when a pre-fetched ``df`` is supplied directly.

2. **Algo** (:meth:`run_algo`): dispatches to the appropriate typed runner
   based on ``flow_config.mode``:

   - ``"forecast"`` → :class:`~abvelocity.ts.forecast_runner.ForecastRunner`
   - ``"detect"``   → :class:`~abvelocity.ts.detect_runner.AnomalyDetectRunner`
   - ``"backfill"`` → :class:`~abvelocity.ts.backfill.runner.BackfillRunner`

3. **Eval** (:meth:`compute_eval`): optionally calls
   :func:`~abvelocity.ts.eval.compute_eval` on the result
   when ``flow_config.eval_metrics`` is set.

Spark scaling note
------------------
``TSFlow`` is stateless after construction — each ``run()`` call is
independent and safe to parallelize across metric groups.
"""

import dataclasses
from typing import Optional, Union

import pandas as pd
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.ts.backfill.runner import BackfillRunner
from abvelocity.ts.detect_runner import AnomalyDetectRunner
from abvelocity.ts.flow.flow_config import TSFlowConfig
from abvelocity.ts.flow.flow_result import TSFlowResult
from abvelocity.ts.forecast_runner import ForecastRunner
from abvelocity.ts.get_data.ts_metrics_query import TSMetricsQuery
from abvelocity.ts.result.detect_result import DetectResult
from abvelocity.ts.result.forecast_result import ForecastResult


class TSFlow:
    """Timeseries pipeline: fetch → algo → eval for a single metric group.

    Args:
        flow_config: Full flow configuration (algo config, time-bucketing
            config, mode, and optional eval settings).
        io_param: IO/DB parameters containing the cursor used for data
            fetching.  May be ``None`` when passing a pre-fetched ``df``
            directly to :meth:`run`.
    """

    def __init__(self, flow_config: TSFlowConfig, io_param: Optional[IOParam] = None) -> None:
        self.flow_config = flow_config
        self.io_param = io_param

    # ------------------------------------------------------------------
    # Step 1 – data fetch
    # ------------------------------------------------------------------

    def fetch_data(self, metric_info: MetricInfo) -> pd.DataFrame:
        """Fetch time-bucketed metric data for one metric group.

        Args:
            metric_info: Metric group definition (table, metrics, dims,
                dates).

        Returns:
            Wide-format DataFrame — one row per ``(time_bucket × dims)``.

        Raises:
            ValueError: If ``io_param`` or ``io_param.cursor`` is ``None``.
        """
        if self.io_param is None or self.io_param.cursor is None:
            raise ValueError("io_param with a cursor is required for fetch_data. " "Pass df directly to run() to skip data fetching.")
        q = TSMetricsQuery(metric_info, self.flow_config.ts_metrics_config)
        return q.get_df(self.io_param.cursor)

    # ------------------------------------------------------------------
    # Step 2 – algo
    # ------------------------------------------------------------------

    def run_algo(
        self,
        df: pd.DataFrame,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> Union[ForecastResult, DetectResult]:
        """Dispatch to the appropriate runner based on ``flow_config.mode``.

        Args:
            df: Wide-format training/input DataFrame.
            anomaly_df: Optional known-anomaly intervals.

        Returns:
            :class:`ForecastResult`, :class:`DetectResult`, or
            :class:`~abvelocity.ts.backfill.result.BackfillResult`.
        """
        cfg = self.flow_config
        if cfg.mode == "forecast":
            return ForecastRunner(cfg.ts_model_config).run(
                df=df,
                prediction_window=cfg.prediction_window,
                anomaly_df=anomaly_df,
            )
        if cfg.mode == "detect":
            return AnomalyDetectRunner(cfg.ts_model_config).run(
                df=df,
                prediction_window=cfg.prediction_window,
                anomaly_df=anomaly_df,
            )
        # mode == "backfill"
        return BackfillRunner(cfg.backfill_config).run(df=df, anomaly_df=anomaly_df)

    # ------------------------------------------------------------------
    # Step 3 – eval
    # ------------------------------------------------------------------

    def compute_eval(self, result_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Compute eval metrics by delegating to ``flow_config.eval.run``.

        Returns ``None`` when ``flow_config.eval`` is not set or
        ``result_df`` is ``None``. The caller parameterizes the evaluator
        (metrics, group_by, thresholds, soft window) by constructing a
        :class:`~abvelocity.ts.eval.forecast_eval.ForecastEval`
        or :class:`~abvelocity.ts.eval.ad_eval.ADEval` and
        passing it on :attr:`TSFlowConfig.eval`.
        """
        if self.flow_config.eval is None or result_df is None:
            return None
        return self.flow_config.eval.run(result_df)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        metric_info: MetricInfo,
        df: Optional[pd.DataFrame] = None,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> TSFlowResult:
        """Run the full pipeline for a single metric group.

        Args:
            metric_info: Metric group definition.
            df: Pre-fetched wide-format DataFrame.  When provided, the data
                fetch step is skipped — useful for testing or when the data
                is already in memory.
            anomaly_df: Optional known-anomaly intervals forwarded to the
                algo step.

        Returns:
            :class:`TSFlowResult` with ``result_df``, optional ``eval_df``,
            and optional ``fit_info``.
        """
        if df is None:
            df = self.fetch_data(metric_info)

        # Derive value_cols from metric_info when not explicitly set.
        cfg = self.flow_config
        if cfg.ts_model_config is not None and cfg.ts_model_config.value_cols is None:
            derived = tuple(m.name for m in (metric_info.metrics or []))
            if not derived:
                raise ValueError("ts_model_config.value_cols is None and metric_info.metrics is empty — " "cannot derive value_cols automatically.")
            self.flow_config = dataclasses.replace(cfg, ts_model_config=dataclasses.replace(cfg.ts_model_config, value_cols=derived))

        result = self.run_algo(df, anomaly_df=anomaly_df)
        result_df = getattr(result, "result_df", None)

        return TSFlowResult(
            result_df=result_df,
            eval_df=self.compute_eval(result_df),
            fit_info=getattr(result, "fit_info", None),
        )

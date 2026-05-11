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
"""TSFlowConfig: configuration for a full timeseries flow run."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.config.ts_model_config import TSModelConfig
from abvelocity.ts.eval import ADEval, ForecastEval
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig

VALID_FLOW_MODES = ("forecast", "detect", "backfill")


@dataclass
class TSFlowConfig:
    """Configuration for a full timeseries flow run (data fetch + algo + eval).

    A ``TSFlowConfig`` bundles everything ``TSFlow`` needs to operate on one
    or more ``MetricInfo`` groups:

    - **ts_metrics_config**: describes how to bucket the raw timestamp column
      and which SQL dialect to use when fetching data via
      :class:`~abvelocity.ts.get_data.ts_metrics_query.TSMetricsQuery`.
    - **ts_model_config / backfill_config**: the algorithm configuration, depending
      on ``mode``.
    - **mode**: selects which typed runner is invoked.
    - **prediction_window / eval_metrics**: optional post-processing controls.

    Attributes:
        ts_metrics_config: How to bucket the time axis when fetching data from
            the DB via ``TSMetricsQuery``.
        ts_model_config: Algo configuration passed to the typed runner.
            Must be a
            :class:`~abvelocity.ts.config.forecast_config.ForecastConfig`
            for ``mode="forecast"`` or a
            :class:`~abvelocity.ts.config.detect_config.DetectConfig`
            for ``mode="detect"``.  Unused when ``mode="backfill"``
            (the forecast config lives inside ``backfill_config`` in that case).
        mode: Run mode — ``"forecast"``, ``"detect"``, or ``"backfill"``.
        backfill_config: Required when ``mode="backfill"``.  Controls the
            sliding-cutoff logic (initial window, horizon, step, etc.).
        prediction_window: Optional ``(start_date, end_date)`` ISO strings
            to restrict the returned rows after forecasting / detection.
            Ignored for backfill mode.
        eval: Optional parameterized evaluator.  Pass a
            :class:`~abvelocity.ts.eval.forecast_eval.ForecastEval`
            for forecast / backfill modes (regression metrics), or a
            :class:`~abvelocity.ts.eval.ad_eval.ADEval` for detect
            mode (classification metrics incl. soft_*).  ``None`` → skip
            eval.  Defaults to ``None``.
    """

    ts_metrics_config: TSMetricsConfig
    """Time-bucketing config used by TSMetricsQuery to fetch data from DB."""

    ts_model_config: Optional[TSModelConfig] = None
    """Algo config; used for forecast and detect modes."""

    mode: str = "forecast"
    """Run mode: 'forecast', 'detect', or 'backfill'."""

    backfill_config: Optional[BackfillConfig] = None
    """Required when mode='backfill'."""

    prediction_window: Optional[Tuple[str, str]] = None
    """Optional (start_date, end_date) filter applied after prediction."""

    eval: Optional[Union[ForecastEval, ADEval]] = None
    """Optional evaluator — ``ForecastEval`` or ``ADEval``. ``None`` → skip eval."""

    def __post_init__(self) -> None:
        if self.mode not in VALID_FLOW_MODES:
            raise ValueError(f"mode must be one of {VALID_FLOW_MODES!r}, got {self.mode!r}.")
        if self.mode == "backfill" and self.backfill_config is None:
            raise ValueError("backfill_config must be set when mode='backfill'.")
        if self.mode != "backfill" and self.ts_model_config is None:
            raise ValueError(f"ts_model_config must be set when mode={self.mode!r}.")

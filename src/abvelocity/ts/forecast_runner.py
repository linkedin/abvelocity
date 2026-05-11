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
"""Typed runner for time-series forecasting.

:class:`ForecastRunner` is a thin, type-safe facade over
:class:`~abvelocity.ts.runner.TSRunner` that:

- Enforces :class:`~abvelocity.ts.config.forecast_config.ForecastConfig`
  as the config type (so ``forecast_horizon`` is always available to the algo).
- Guarantees the return type is
  :class:`~abvelocity.ts.result.forecast_result.ForecastResult`,
  raising :exc:`TypeError` when the registered algo returns something else.

Use :class:`~abvelocity.ts.detect_runner.AnomalyDetectRunner`
for anomaly detection runs.
"""

from typing import Optional

import pandas as pd
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.result.forecast_result import ForecastResult
from abvelocity.ts.runner import TSRunner

# Eagerly import the known algos so they self-register in
# ``ALGO_REGISTRY`` the moment ``ForecastRunner`` becomes available.
# Without this, callers had to remember to ``import …greykite_forecast_algo``
# themselves just for the side effect.  Each import is wrapped in
# ``ImportError`` so optional algos (greykite has its own pinned deps)
# don't break a base ForecastRunner import for environments without
# them — the algo will simply not be in the registry, and a config
# referencing it will fail with the usual "Unknown algo_name" message.
try:
    import abvelocity.ts.algo.greykite_forecast_algo  # noqa: F401
except ImportError:
    pass
try:
    import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401
except ImportError:
    pass


class ForecastRunner:
    """Typed facade over :class:`TSRunner` for forecasting runs.

    Attributes:
        config: Forecasting configuration; ``forecast_horizon`` is required.
    """

    def __init__(self, config: ForecastConfig) -> None:
        self.config = config

    def run(
        self,
        df: pd.DataFrame,
        prediction_window: Optional[tuple[str, str]] = None,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> ForecastResult:
        """Fit the configured forecast algorithm and return predictions.

        Delegates to :meth:`TSRunner.run` then asserts the result is a
        :class:`ForecastResult`.

        Args:
            df: Training DataFrame.
            prediction_window: Optional ``(start_date, end_date)`` ISO strings
                restricting the prediction output range.
            anomaly_df: Optional known-anomaly intervals used to mask anomalous
                periods during training.

        Returns:
            :class:`ForecastResult` with ``result_df`` in long format.

        Raises:
            TypeError: If the registered algo returns a result that is not a
                :class:`ForecastResult` (e.g. a detection algo was registered
                under ``config.algo_name``).
        """
        result = TSRunner(self.config).run(
            df=df,
            prediction_window=prediction_window,
            anomaly_df=anomaly_df,
        )
        if not isinstance(result, ForecastResult):
            raise TypeError(
                f"algo_name={self.config.algo_name!r} returned "
                f"{type(result).__name__}, expected ForecastResult. "
                "Use AnomalyDetectRunner for detection algos."
            )
        return result

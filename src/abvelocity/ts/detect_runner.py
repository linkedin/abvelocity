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
"""Typed runner for time-series anomaly detection.

:class:`AnomalyDetectRunner` is a thin, type-safe facade over
:class:`~abvelocity.ts.runner.TSRunner` that:

- Enforces :class:`~abvelocity.ts.config.detect_config.DetectConfig`
  as the config type (so the nested ``forecast_config`` is always available
  to the detection algo).
- Guarantees the return type is
  :class:`~abvelocity.ts.result.detect_result.DetectResult`,
  raising :exc:`TypeError` when the registered algo returns something else.

Use :class:`~abvelocity.ts.forecast_runner.ForecastRunner`
for forecasting runs.
"""

from typing import Optional

import pandas as pd
from abvelocity.ts.config.detect_config import DetectConfig
from abvelocity.ts.result.detect_result import DetectResult
from abvelocity.ts.runner import TSRunner


class AnomalyDetectRunner:
    """Typed facade over :class:`TSRunner` for anomaly detection runs.

    Attributes:
        config: Detection configuration; may include a nested
            ``forecast_config`` for the underlying forecast step.
    """

    def __init__(self, config: DetectConfig) -> None:
        self.config = config

    def run(
        self,
        df: pd.DataFrame,
        prediction_window: Optional[tuple[str, str]] = None,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> DetectResult:
        """Fit the configured detection algorithm and return results.

        Delegates to :meth:`TSRunner.run` then asserts the result is a
        :class:`DetectResult`.

        Args:
            df: Training DataFrame.
            prediction_window: Optional ``(start_date, end_date)`` ISO strings
                restricting the detection output range.
            anomaly_df: Optional known-anomaly intervals used to mask anomalous
                periods during training.

        Returns:
            :class:`DetectResult` with ``result_df`` and ``anomalies_df``
            in long format.

        Raises:
            TypeError: If the registered algo returns a result that is not a
                :class:`DetectResult` (e.g. a forecast algo was registered
                under ``config.algo_name``).
        """
        result = TSRunner(self.config).run(
            df=df,
            prediction_window=prediction_window,
            anomaly_df=anomaly_df,
        )
        if not isinstance(result, DetectResult):
            raise TypeError(
                f"algo_name={self.config.algo_name!r} returned " f"{type(result).__name__}, expected DetectResult. " "Use ForecastRunner for forecast algos."
            )
        return result

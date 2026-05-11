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
"""Anomaly-detection-specific configuration dataclass."""

import warnings
from dataclasses import dataclass
from typing import Optional

from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.config.ts_model_config import TSModelConfig

SHARED_DATA_FIELDS = ("time_col", "value_cols", "freq", "train_end_date", "coverage")


@dataclass
class DetectConfig(TSModelConfig):
    """Configuration for time-series anomaly detection.

    The ``algo_name`` / ``algo_params`` fields inherited from :class:`TSModelConfig`
    control the *detection* algorithm. The nested ``forecast_config`` controls
    the *forecast* algorithm used inside the detection step (mirrors the
    greykite-detection design where ``ForecastConfig`` and ``ADConfig`` are
    sibling configs passed to ``GreykiteDetector``).

    ``detection_window`` (a timestamp range) is NOT stored here — it is passed
    directly to :meth:`~abvelocity.ts.runner.TSRunner.run` as
    a ``prediction_window`` argument, mirroring the oi-schemas
    ``PredictionRequest`` / ``AlgoConfig`` split.

    Attributes:
        forecast_config: Forecasting algo config for the underlying forecast
            step inside detection. ``forecast_config.algo_name`` and
            ``forecast_config.algo_params`` control the forecast algo;
            ``DetectConfig.algo_name`` / ``algo_params`` control the detection
            algo itself.
    """

    forecast_config: Optional[ForecastConfig] = None
    """Forecasting algo config for the underlying forecast step inside detection."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.forecast_config is None:
            return
        mismatches = [
            f"  {field}: DetectConfig={getattr(self, field)!r} vs " f"forecast_config={getattr(self.forecast_config, field)!r}"
            for field in SHARED_DATA_FIELDS
            if getattr(self, field) != getattr(self.forecast_config, field)
        ]
        if mismatches:
            warnings.warn(
                "DetectConfig and its nested forecast_config have differing "
                "data fields (note: algo_name/algo_params differences are "
                "intentional — detect and forecast algos can differ):\n" + "\n".join(mismatches),
                UserWarning,
                stacklevel=2,
            )

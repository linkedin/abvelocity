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
"""Abstract base class and algorithm registry for time-series algorithms."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from abvelocity.ts.config.ts_model_config import TSModelConfig
from abvelocity.ts.result.ts_result import TSResult

ALGO_REGISTRY: dict[str, type["TSAlgo"]] = {}
"""Registry mapping algo name strings to :class:`TSAlgo` subclasses.

Concrete algorithm modules register themselves here at import time::

    from abvelocity.ts.algo.base import ALGO_REGISTRY

    ALGO_REGISTRY["my_algo"] = MyAlgo

:class:`~abvelocity.ts.runner.TSRunner` uses this registry
to instantiate the algorithm selected by
:attr:`~abvelocity.ts.config.ts_model_config.TSModelConfig.algo_name`.
"""


@dataclass
class TSAlgo:
    """Abstract base class for time-series algorithms.

    Subclasses must implement :meth:`fit` and :meth:`predict`, then register
    themselves in :data:`ALGO_REGISTRY` so that
    :class:`~abvelocity.ts.runner.TSRunner` can look them up
    by name.

    Uses ``@dataclass`` + ``@abstractmethod`` (no ``ABC`` base), consistent
    with the :class:`~abvelocity.stats.estimator.Estimator` pattern.

    Attributes:
        algo_params: Algorithm-specific parameters passed from
            :class:`~abvelocity.ts.config.ts_model_config.TSModelConfig`.
            Defaults to an empty dict after ``__post_init__``.
    """

    algo_params: Optional[Dict[str, Any]] = None
    """Algorithm-specific parameters; normalised to ``{}`` in ``__post_init__``."""

    def __post_init__(self) -> None:
        self.algo_params = self.algo_params or {}

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        config: TSModelConfig,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> "TSAlgo":
        """Fit the algorithm on historical data.

        Args:
            df: Training DataFrame; must contain ``config.time_col`` and all
                ``config.value_cols`` (plus ``config.regressor_cols`` if any).
            config: Time-series configuration driving this fit.
            anomaly_df: Optional known-anomaly intervals used to mask anomalous
                periods during training (mirrors ``greykite_api.py`` pattern).

        Returns:
            Self, to allow method chaining.
        """
        ...

    @abstractmethod
    def predict(
        self,
        df: Optional[pd.DataFrame] = None,
        prediction_window: Optional[tuple[str, str]] = None,
    ) -> TSResult:
        """Generate predictions (forecast or anomaly scores).

        Args:
            df: Optional holdout DataFrame for in-sample evaluation.
            prediction_window: Optional ``(start_date, end_date)`` ISO strings
                restricting the prediction output range (mirrors oi-schemas
                ``PredictionRequest``).

        Returns:
            A :class:`~abvelocity.ts.result.ts_result.TSResult`
            (or subclass) with the prediction output.

        Note:
            The returned ``result_df`` will be missing ``algo_ver`` and
            ``train_end_date`` columns until
            :meth:`~abvelocity.ts.runner.TSRunner.run` stamps
            them. Call the algo directly only when you do not need the full
            fixed output schema.
        """
        ...

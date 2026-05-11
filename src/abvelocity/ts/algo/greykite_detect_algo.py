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
"""Greykite-based anomaly detection algorithm stub.

Conditionally imports ``blah.greykite.detection`` (same Blah fork used
by ``greykite_forecast_algo``). If the package is not installed the module
still loads cleanly and ``GreykiteDetectAlgo`` is simply not registered.
"""

try:
    from abvelocity.ts.gk.detection.detector.config import ADConfig  # noqa: F401
    from abvelocity.ts.gk.detection.detector.greykite import GreykiteDetector  # noqa: F401

    GREYKITE_AVAILABLE = True
except ImportError:
    GREYKITE_AVAILABLE = False

if GREYKITE_AVAILABLE:
    from dataclasses import dataclass
    from typing import Optional

    import pandas as pd
    from abvelocity.ts.algo.base import ALGO_REGISTRY, TSAlgo
    from abvelocity.ts.config.ts_model_config import TSModelConfig
    from abvelocity.ts.result.detect_result import DetectResult

    @dataclass
    class GreykiteDetectAlgo(TSAlgo):
        """Greykite-based anomaly detection algorithm.

        Uses ``blah.greykite.detection`` (Blah fork). Registered as
        ``"greykite_detect"`` in :data:`ALGO_REGISTRY` when the package is
        importable.

        Both :meth:`fit` and :meth:`predict` raise
        :exc:`NotImplementedError`; full implementation is deferred to
        a future iteration.
        """

        def fit(
            self,
            df: pd.DataFrame,
            config: TSModelConfig,
            anomaly_df: Optional[pd.DataFrame] = None,
        ) -> "GreykiteDetectAlgo":
            raise NotImplementedError("GreykiteDetectAlgo.fit not yet implemented")

        def predict(
            self,
            df: Optional[pd.DataFrame] = None,
            prediction_window: Optional[tuple[str, str]] = None,
        ) -> DetectResult:
            raise NotImplementedError("GreykiteDetectAlgo.predict not yet implemented")

    ALGO_REGISTRY["greykite_detect"] = GreykiteDetectAlgo

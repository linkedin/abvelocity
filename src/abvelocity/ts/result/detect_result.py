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
"""Anomaly detection result dataclass."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from abvelocity.core.utils.serialization import DataFrameConfig
from abvelocity.ts.result.ts_result import TSResult


@dataclass
class DetectResult(TSResult):
    """Result from a time-series anomaly detection run.

    Extends :class:`TSResult` with an ``anomalies_df`` holding interval-level
    anomaly spans. The inherited ``result_df`` additionally contains
    ``anomaly`` (binary flag) and ``anomaly_score`` (continuous score)
    columns alongside the standard fixed-schema forecast columns.

    Attributes:
        anomalies_df: Interval-level anomaly DataFrame with fixed columns
            ``metric``, ``start_ts``, ``end_ts``.
    """

    anomalies_df: Optional[pd.DataFrame] = None
    """Interval-level anomaly spans with columns ``metric``, ``start_ts``, ``end_ts``."""

    class Config(DataFrameConfig):
        pass

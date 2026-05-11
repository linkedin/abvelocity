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
"""Row-level schema dataclass for :class:`TSResult` output DataFrames.

:class:`TSResultRow` is the canonical schema for one row of
:attr:`~abvelocity.ts.result.ts_result.TSResult.result_df`. It
is the single source of truth for column names, Python types, pandas
dtypes, and (future) gRPC/protobuf serialization — no parallel schema
in constants files.

The same class covers both forecast and anomaly-detection outputs:

* **Shared fields** (``ts``, ``metric_id``, ``stage``, ``actual``,
  ``algo_name``, ``run_id``, …) always populated.
* **Forecast-only fields** (``forecast``, ``forecast_lower``,
  ``forecast_upper``, ``std``, the four decomposition components) are
  ``Optional`` — populated for forecast runs, ``NaN`` for detection.
* **Detection-only fields** (``is_anomaly``, ``is_anomaly_predicted``,
  ``anomaly_score``, ``anomaly_severity``) are ``Optional`` —
  populated for detection runs, ``NaN`` for forecasts.

pandas remains the runtime container inside abvelocity for
performance (vectorized ops, Spark interop). Schema introspection lives
on this class:

* :meth:`TSResultRow.columns`     — tuple of column names in write order
* :meth:`TSResultRow.dtypes`      — dict mapping column → pandas dtype
* :meth:`TSResultRow.from_series` — one row of a DataFrame → dataclass

Collection-level conversion (DataFrame ↔ ``list[TSResultRow]``) lives
on :class:`~abvelocity.ts.result.ts_result.TSResult` via
``TSResult.to_df`` / ``to_rows`` / ``from_df`` / ``from_rows`` — that's
the container that owns both views of the same data.

mashumaro's :class:`DataClassJSONMixin` gives free JSON round-trip
today; protobuf generation (whenever a gRPC service is added) can be
driven off the same dataclass via ``betterproto`` / ``proto-plus`` /
direct ``.proto`` generation without creating a second schema.
"""

from dataclasses import dataclass, fields
from datetime import date, datetime
from typing import Any, Dict, Optional

import pandas as pd
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class TSResultRow(DataClassJSONMixin):
    """One row of a time-series result DataFrame (forecast or detection)."""

    # --- identity (shared forecast/detection) ---
    ts: datetime
    metric_id: str
    metric_name: Optional[str]
    stage: str  # "fitted" | "forecast" (detection rows are always "fitted")
    forecasted_date: date
    last_training_date: Optional[str]

    # --- observed value (shared) ---
    actual: Optional[float]

    # --- forecast-only values ---
    forecast: Optional[float]
    forecast_lower: Optional[float]
    forecast_upper: Optional[float]
    std: Optional[float]

    # --- forecast-only additive decomposition components ---
    longterm_growth: Optional[float]
    shortterm_growth: Optional[float]
    daily_seasonality: Optional[float]
    weekly_seasonality: Optional[float]
    annual_seasonality: Optional[float]
    holiday_impact: Optional[float]
    residual: Optional[float]

    # --- detection-only values ---
    is_anomaly: Optional[bool]
    """Ground-truth anomaly label when known (from labeled data or an
    oracle)."""

    is_anomaly_predicted: Optional[bool]
    """Model-predicted anomaly flag."""

    anomaly_score: Optional[float]
    """Continuous anomaly score; higher = more anomalous."""

    anomaly_severity: Optional[str]
    """Severity bucket, e.g. ``"low"`` / ``"medium"`` / ``"high"``."""

    # --- open-ended extras bag (shared) ---
    extras: Optional[Dict[str, float]]

    # --- run identity (shared) ---
    algo_name: str
    algo_version: Optional[str]
    run_id: Optional[str]
    run_date: Optional[date]

    # -------------------------------------------------------------------
    # Schema introspection — classmethods that expose column names,
    # dtypes, and a single-row constructor. Collection-level conversion
    # lives on :class:`TSResult`.
    # -------------------------------------------------------------------

    @classmethod
    def columns(cls) -> tuple[str, ...]:
        """Column names, in canonical write order."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def dtypes(cls) -> Dict[str, Any]:
        """Mapping of column name → pandas dtype."""
        return {f.name: _PANDAS_DTYPE[f.type] for f in fields(cls)}

    @classmethod
    def from_series(cls, s: pd.Series) -> "TSResultRow":
        """Build a :class:`TSResultRow` from one DataFrame row."""
        return cls(**{n: s[n] for n in cls.columns()})


# Mapping from Python field type to pandas dtype used by ``dtypes()``.
# Kept next to the dataclass so the schema lives in one file.
_PANDAS_DTYPE: Dict[Any, Any] = {
    datetime: "datetime64[ns]",
    date: "datetime64[ns]",
    str: pd.StringDtype(),
    Optional[str]: pd.StringDtype(),
    Optional[date]: "datetime64[ns]",
    float: "float64",
    Optional[float]: pd.Float64Dtype(),
    Optional[bool]: pd.BooleanDtype(),
    Optional[Dict[str, float]]: "object",
}

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
"""Simple seasonal mean forecasting algorithm.

Registered as ``"simple"`` in :data:`~abvelocity.ts.algo.base.ALGO_REGISTRY`.

Design
------
For each future step ``h`` (1-indexed), the forecast is the mean of the ``k``
training values spaced ``period`` apart, looking back from the future position:

    positions: n+h-1 - period, n+h-1 - 2*period, …, n+h-1 - k*period

where ``n`` is the length of the training series and positions are 0-indexed.

- ``period=1`` → trailing k-step mean (e.g. average of last 3 values).
- ``period=7, k=3`` → average of the same weekday over the last 3 weeks
  (lags 7, 14, 21).

Lookback positions that fall outside the training window are silently skipped,
so the algo works even when fewer than ``k`` full periods are available.

Prediction interval
-------------------
CI = mean ± z × std, where:

- ``z`` is the normal quantile at ``(1 + coverage) / 2``.
- ``std`` is the **population** standard deviation of the available lookback
  values (``ddof=0``). With ``k=1`` the std is 0 and the CI degenerates to
  a point — the caller should interpret this as high uncertainty, not
  high confidence.

``algo_params`` keys
--------------------
``period`` : int, default ``1``
    Seasonal period in rows.
``k`` : int, default ``3``
    Number of past periods to average.
``agg`` : ``"mean"`` | ``"median"``, default ``"mean"``
    Aggregation function applied to the k lookback values.
    ``"median"`` is more robust to outlier periods.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from abvelocity.ts.algo.base import ALGO_REGISTRY, TSAlgo
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.config.ts_model_config import TSModelConfig
from abvelocity.ts.constants import ACTUAL_COL, FORECAST_COL, FORECAST_LOWER_COL, FORECAST_UPPER_COL, METRIC_ID_COL
from abvelocity.ts.result.forecast_result import ForecastResult
from scipy.stats import norm


@dataclass
class SimpleForecastAlgo(TSAlgo):
    """Seasonal mean forecaster with normal-distribution CI.

    See module docstring for the full algorithm description.

    Attributes:
        algo_params: Accepts ``period`` (int) and ``k`` (int).
        fitted_df: Training DataFrame stored during ``fit()``.
        fitted_config: Config stored during ``fit()``.
    """

    fitted_df: Optional[pd.DataFrame] = None
    """Training DataFrame; populated by ``fit()``."""

    fitted_config: Optional[TSModelConfig] = None
    """Config used during fitting; populated by ``fit()``."""

    def fit(
        self,
        df: pd.DataFrame,
        config: TSModelConfig,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> "SimpleForecastAlgo":
        """Store the training data and config.

        Args:
            df: Training DataFrame with ``config.time_col`` and all
                ``config.value_cols``.
            config: Time-series configuration.
            anomaly_df: Unused; accepted for interface compatibility.

        Returns:
            Self.
        """
        self.fitted_df = df.copy().reset_index(drop=True)
        self.fitted_config = config
        return self

    def predict(
        self,
        df: Optional[pd.DataFrame] = None,
        prediction_window: Optional[Tuple[str, str]] = None,
    ) -> ForecastResult:
        """Generate seasonal mean forecasts with CI for all value columns.

        Returns a long-format :class:`ForecastResult` covering both the
        training period (with in-sample predictions) and the forecast horizon
        (with ``actual=NaN``).

        Args:
            df: Unused; accepted for interface compatibility.
            prediction_window: Optional ``(start_date, end_date)`` ISO strings
                to restrict the returned rows.

        Returns:
            :class:`ForecastResult` with ``result_df`` in long format.

        Raises:
            ValueError: If called before ``fit()``.
        """
        if self.fitted_df is None or self.fitted_config is None:
            raise ValueError("Must call fit() before predict().")

        config = self.fitted_config
        time_col = config.time_col
        value_cols = list(config.value_cols)

        # horizon and future_dates are shared across all metrics (same time axis).
        default_params = config.get_algo_params("")
        horizon = config.forecast_horizon if isinstance(config, ForecastConfig) else int(default_params.get("forecast_horizon", 1))

        freq = config.freq or pd.infer_freq(self.fitted_df[time_col])
        last_ts = self.fitted_df[time_col].iloc[-1]
        future_dates = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]

        z = float(norm.ppf((1.0 + config.coverage) / 2.0))

        per_col_dfs = []
        for col in value_cols:
            # Resolve params per metric so algo_params_by_metric overrides apply.
            params = config.get_algo_params(col)
            period = int(params.get("period", 1))
            k = int(params.get("k", 3))
            agg = str(params.get("agg", "mean"))
            if agg not in {"mean", "median"}:
                raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}.")
            per_col_dfs.append(
                self.build_col_df(
                    col=col,
                    time_col=time_col,
                    series=self.fitted_df[col].to_numpy(dtype=float),
                    train_dates=self.fitted_df[time_col].to_numpy(),
                    future_dates=future_dates,
                    horizon=horizon,
                    period=period,
                    k=k,
                    agg=agg,
                    z=z,
                )
            )

        result_df = pd.concat(per_col_dfs, ignore_index=True)

        if prediction_window is not None:
            start, end = prediction_window
            ts_col = result_df[time_col]
            mask = (ts_col >= pd.Timestamp(start)) & (ts_col <= pd.Timestamp(end))
            result_df = result_df.loc[mask].reset_index(drop=True)

        return ForecastResult(result_df=result_df)

    def build_col_df(
        self,
        col: str,
        time_col: str,
        series: np.ndarray,
        train_dates: np.ndarray,
        future_dates: pd.DatetimeIndex,
        horizon: int,
        period: int,
        k: int,
        agg: str,
        z: float,
    ) -> pd.DataFrame:
        """Build the result DataFrame for a single value column.

        Includes training rows (with actuals and in-sample predictions) and
        future rows (with forecasts and ``actual=NaN``).
        """
        n = len(series)
        rows = []

        for pos in range(n):
            fc, lower, upper = self.compute_point_and_ci(self.get_lookback_values(series, pos, period, k), agg, z)
            rows.append(
                {
                    time_col: train_dates[pos],
                    METRIC_ID_COL: col,
                    ACTUAL_COL: series[pos],
                    FORECAST_COL: fc,
                    FORECAST_LOWER_COL: lower,
                    FORECAST_UPPER_COL: upper,
                }
            )

        for step in range(1, horizon + 1):
            future_pos = n + step - 1
            fc, lower, upper = self.compute_point_and_ci(self.get_lookback_values(series, future_pos, period, k), agg, z)
            rows.append(
                {
                    time_col: future_dates[step - 1],
                    METRIC_ID_COL: col,
                    ACTUAL_COL: float("nan"),
                    FORECAST_COL: fc,
                    FORECAST_LOWER_COL: lower,
                    FORECAST_UPPER_COL: upper,
                }
            )

        return pd.DataFrame(rows)

    def get_lookback_values(self, series: np.ndarray, pos: int, period: int, k: int) -> List[float]:
        """Return up to k lookback values at positions pos - period, pos - 2*period, …

        Positions outside [0, len(series)) are silently skipped.
        """
        values = []
        for j in range(1, k + 1):
            lookback_pos = pos - j * period
            if 0 <= lookback_pos < len(series):
                values.append(float(series[lookback_pos]))
        return values

    def compute_point_and_ci(self, lookback_vals: List[float], agg: str, z: float) -> Tuple[float, float, float]:
        """Compute point forecast and symmetric CI from lookback values.

        The point forecast is ``mean`` or ``median`` of the lookback values.
        The CI is ``point ± z × std`` where ``std`` is the population standard
        deviation of the lookback values regardless of the chosen ``agg``.

        Args:
            lookback_vals: Available historical values to aggregate.
            agg: ``"mean"`` or ``"median"``.
            z: Normal quantile for the requested coverage.

        Returns:
            ``(forecast, lower, upper)`` — all ``NaN`` when no values available.
        """
        if not lookback_vals:
            nan = float("nan")
            return nan, nan, nan
        arr = np.array(lookback_vals, dtype=float)
        point = float(np.median(arr) if agg == "median" else np.mean(arr))
        std_val = float(np.std(arr, ddof=0))
        return point, point - z * std_val, point + z * std_val


ALGO_REGISTRY["simple"] = SimpleForecastAlgo

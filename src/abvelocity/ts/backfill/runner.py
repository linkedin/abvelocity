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
"""BackfillRunner: slides a training cutoff forward and collects historical forecasts."""

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional

import pandas as pd
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.backfill.result import BackfillResult
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import ACTUAL_COL, CUTOFF_COL, FORECAST_LOWER_COL, FORECAST_UPPER_COL, HORIZON_STEP_COL, METRIC_ID_COL
from abvelocity.ts.forecast_runner import ForecastRunner

logger = logging.getLogger(__name__)


class BackfillRunner:
    """Slides a training cutoff forward through a prepped DataFrame and collects forecasts.

    For each cutoff the algo is re-fitted on the available training window and
    ``horizon`` steps of forecasts are generated. Actuals for those steps are
    pulled directly from the prepped input DataFrame (which contains observed
    values for all rows). The result is a long-format DataFrame of past
    forecasts tagged by ``cutoff`` (last training timestamp) and
    ``horizon_step`` (1 … h), suitable for eval or backfilling a forecast
    store.

    Attributes:
        config: Backfill configuration controlling cutoff generation,
            window type, and the underlying forecast algo.
    """

    def __init__(self, config: BackfillConfig) -> None:
        self.config = config

    def run(
        self,
        df: pd.DataFrame,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> BackfillResult:
        """Run backfill across all cutoffs and return collected forecasts.

        Args:
            df: Prepped DataFrame in wide format containing ``time_col`` and
                all ``value_cols`` for the full period (training + evaluation).
                Actuals must be present for all rows including the forecast
                horizon — the runner pulls them directly from this DataFrame.
            anomaly_df: Optional known-anomaly intervals forwarded to each
                per-cutoff ``TSRunner.run()`` call.

        Returns:
            :class:`BackfillResult` with ``result_df`` covering all cutoffs.

        Raises:
            ValueError: If no valid cutoffs can be generated from ``df``.
        """
        cutoff_indices = self.resolve_cutoff_indices(df)
        if not cutoff_indices:
            raise ValueError(
                f"No valid cutoffs found. DataFrame has {len(df)} rows but "
                f"initial_train_size={self.config.initial_train_size} and "
                f"horizon={self.config.horizon} leave no room for any cutoff."
            )

        fc = self.config.forecast_config
        time_col = fc.time_col
        value_cols = list(fc.value_cols)

        # Pre-build long-format actual lookup from the full prepped df.
        actual_lookup = self.build_actual_lookup(df, time_col, value_cols)

        all_dfs: List[pd.DataFrame] = []
        all_fit_info: Dict[str, Any] = {}

        for cutoff_idx in cutoff_indices:
            train_df = self.get_train_df(df, cutoff_idx)
            cutoff_ts = df[time_col].iloc[cutoff_idx - 1]

            cutoff_fc = self.build_cutoff_forecast_config(fc, cutoff_ts)
            result = ForecastRunner(cutoff_fc).run(df=train_df, anomaly_df=anomaly_df)

            if result.result_df is None:
                continue

            window_df = self.extract_horizon_window(result.result_df, cutoff_ts, time_col)
            if window_df.empty:
                continue

            window_df = self.fill_actuals(window_df, actual_lookup, time_col)
            window_df[CUTOFF_COL] = cutoff_ts
            all_dfs.append(window_df)

            if result.fit_info:
                all_fit_info[str(cutoff_ts)] = result.fit_info

        result_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else None
        return BackfillResult(
            result_df=result_df,
            fit_info=all_fit_info if all_fit_info else None,
        )

    def resolve_cutoff_indices(self, df: pd.DataFrame) -> List[int]:
        """Return the cutoff indices to run, honoring explicit ``config.cutoffs`` if set.

        When :attr:`BackfillConfig.cutoffs` is non-empty, each date is
        looked up in ``df[time_col]`` and converted to an exclusive-end
        index (``idx_of_date + 1``) so the training window ends at and
        includes that date. Otherwise this falls through to
        :meth:`generate_cutoff_indices`'s algorithmic spec.

        The lookup keys on ``pd.Timestamp(t).date()`` (time-of-day
        normalised away). Caveats:

        * **Sub-daily granularities** (hourly / minutely): multiple
          timestamps would collapse onto the same date — use the
          algorithmic spec for those instead of explicit dates.
        * **Long-format DataFrames** (one row per ``(metric_id, date)``
          combination): the dict-by-date keeps only the last index per
          repeated date, which is wrong here. The algorithmic path has
          the same row-vs-date ambiguity by design (``len(df)`` is row
          count, not unique-date count).

        Args:
            df: Prepped input DataFrame.

        Returns:
            List of cutoff indices in ascending order.

        Raises:
            ValueError: If ``config.cutoffs`` is set and any date is not
                found in ``df[time_col]``.
        """
        cfg = self.config
        if cfg.cutoffs:
            time_col = cfg.forecast_config.time_col
            ts_index = {pd.Timestamp(t).date(): i for i, t in enumerate(pd.to_datetime(df[time_col]))}
            indices: List[int] = []
            n = len(df)
            short_horizon: List[str] = []
            for date_str in cfg.cutoffs:
                target = pd.Timestamp(date_str).date()
                if target not in ts_index:
                    # Hard raise: this is almost certainly a typo / wrong date,
                    # not a "didn't fit the horizon" judgement call.
                    raise ValueError(
                        f"cutoff date {date_str!r} not found in df['{time_col}']; "
                        f"available range is {pd.to_datetime(df[time_col]).min().date()} → "
                        f"{pd.to_datetime(df[time_col]).max().date()}."
                    )
                cutoff_idx = ts_index[target] + 1
                if cutoff_idx + cfg.horizon > n:
                    # Cutoff sits within ``horizon`` rows of the end of df, so
                    # eval metrics for this cutoff will be computed over a
                    # partial horizon (the missing-actual rows drop out of
                    # MAPE/RMSE/etc.). We honour the user's explicit choice
                    # rather than dropping or raising — controlling cutoff
                    # dates is annoying and forcing the user to get the
                    # horizon-math right per-date is overkill — but warn so
                    # they know which cutoffs will produce partial-horizon
                    # numbers.
                    short_horizon.append(date_str)
                indices.append(cutoff_idx)
            if short_horizon:
                logger.warning(
                    "BackfillRunner: %d explicit cutoff(s) sit within horizon=%d of the end of df; "
                    "their eval metrics will be computed over a partial horizon: %s.",
                    len(short_horizon),
                    cfg.horizon,
                    short_horizon,
                )
            return sorted(indices)
        return self.generate_cutoff_indices(len(df))

    def resolve_cutoff_dates(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """Return the actual cutoff timestamps for this run.

        Useful for logging / persisting (e.g.
        :class:`~abvelocity.ts.model_selection.base.ModelSelection`
        writes these to ``cutoffs.json``) so callers don't have to mentally
        convert from indices.

        Args:
            df: Prepped input DataFrame.

        Returns:
            Sorted list of :class:`pandas.Timestamp` — one per cutoff,
            each being the *last training timestamp* (i.e. ``cutoff_idx - 1``).
        """
        time_col = self.config.forecast_config.time_col
        return [df[time_col].iloc[idx - 1] for idx in self.resolve_cutoff_indices(df)]

    def generate_cutoff_indices(self, n: int) -> List[int]:
        """Return list of cutoff row counts (exclusive training end indices).

        The cutoff index ``m`` means: train on rows ``0 … m-1``, forecast
        rows ``m … m + horizon - 1``. The last valid cutoff is
        ``n - horizon`` so the full horizon fits within ``df``.

        When ``n_windows`` is set, the most recent ``n_windows`` cutoffs are
        returned (matching greykite's ``use_most_recent_splits=True`` and
        StatsForecast's ``n_windows`` semantics).

        Args:
            n: Total number of rows in the input DataFrame.

        Returns:
            List of cutoff indices in ascending order.
        """
        cfg = self.config
        # last valid cutoff: train on [:m], forecast [:m+horizon] stays within df
        all_cutoffs = list(range(cfg.initial_train_size, n - cfg.horizon + 1, cfg.step))
        if cfg.n_windows is not None:
            n_windows = cfg.n_windows
            all_cutoffs = all_cutoffs[-n_windows:]
        return all_cutoffs

    def get_train_df(self, df: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
        """Slice the training DataFrame for the given cutoff index.

        Args:
            df: Full prepped DataFrame.
            cutoff_idx: Exclusive end index; rows ``0 … cutoff_idx-1`` are
                used for training (expanding) or the last ``window_size``
                rows up to ``cutoff_idx`` (rolling).

        Returns:
            Training slice, reset index.
        """
        if self.config.window_type == "rolling":
            start = max(0, cutoff_idx - self.config.window_size)  # type: ignore[operator]
            return df.iloc[start:cutoff_idx].reset_index(drop=True)
        return df.iloc[:cutoff_idx].reset_index(drop=True)

    def build_cutoff_forecast_config(self, fc: ForecastConfig, cutoff_ts: pd.Timestamp) -> ForecastConfig:
        """Return a copy of ``fc`` with ``train_end_date`` set to ``cutoff_ts``.

        Args:
            fc: Base forecast config.
            cutoff_ts: Last training timestamp for this cutoff.

        Returns:
            New :class:`ForecastConfig` with updated ``train_end_date``.
        """
        return replace(fc, train_end_date=str(cutoff_ts.date()))

    def build_actual_lookup(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_cols: List[str],
    ) -> pd.DataFrame:
        """Melt the prepped wide DataFrame to a long-format actual lookup.

        Args:
            df: Full prepped DataFrame (wide format).
            time_col: Timestamp column name.
            value_cols: Value columns to melt (the forecast targets).

        Returns:
            Long-format DataFrame with columns ``(time_col, metric, actual)``.
        """
        return df[[time_col] + value_cols].melt(
            id_vars=[time_col],
            var_name=METRIC_ID_COL,
            value_name=ACTUAL_COL,
        )

    def extract_horizon_window(
        self,
        result_df: pd.DataFrame,
        cutoff_ts: pd.Timestamp,
        time_col: str,
    ) -> pd.DataFrame:
        """Filter result_df to the forecast horizon window and add ``horizon_step``.

        Rows with ``ts > cutoff_ts`` are the forecast period. Steps are
        numbered 1 … h per metric in chronological order.

        Args:
            result_df: Long-format result from ``TSRunner.run()``.
            cutoff_ts: Last training timestamp for this cutoff.
            time_col: Timestamp column name.

        Returns:
            Filtered and annotated DataFrame.
        """
        window = result_df[result_df[time_col] > cutoff_ts].copy()
        window = window.sort_values([METRIC_ID_COL, time_col]).reset_index(drop=True)
        window[HORIZON_STEP_COL] = window.groupby(METRIC_ID_COL).cumcount() + 1
        window = window[window[HORIZON_STEP_COL] <= self.config.horizon]

        # Drop algo-filled NaN actuals; real actuals are injected by fill_actuals.
        cols_to_keep = [c for c in window.columns if c != ACTUAL_COL or c not in window.columns]
        # Keep all columns but drop ACTUAL_COL to avoid stale NaN values from TSRunner.
        if ACTUAL_COL in window.columns:
            window = window.drop(columns=[ACTUAL_COL])
        # Also drop CI cols if all-NaN (some algos omit them); keep if present.
        for ci_col in [FORECAST_LOWER_COL, FORECAST_UPPER_COL]:
            if ci_col in window.columns and window[ci_col].isna().all():
                window = window.drop(columns=[ci_col])
        _ = cols_to_keep  # unused, suppresses F841
        return window

    def fill_actuals(
        self,
        window_df: pd.DataFrame,
        actual_lookup: pd.DataFrame,
        time_col: str,
    ) -> pd.DataFrame:
        """Join actual values from the prepped DataFrame into ``window_df``.

        Args:
            window_df: Forecast window DataFrame (no ``actual`` column).
            actual_lookup: Long-format lookup with ``(time_col, metric, actual)``
                built from the full prepped DataFrame.
            time_col: Timestamp column name.

        Returns:
            ``window_df`` with ``actual`` column added.
        """
        return window_df.merge(actual_lookup, on=[time_col, METRIC_ID_COL], how="left")

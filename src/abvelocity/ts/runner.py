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
"""TSRunner: orchestrates the fit/predict lifecycle for time-series algorithms."""

from typing import Optional

import numpy as np
import pandas as pd
from abvelocity.ts.algo.base import ALGO_REGISTRY, TSAlgo
from abvelocity.ts.common.time_properties import validate_per_dim_timestamps
from abvelocity.ts.config.ts_model_config import TSModelConfig
from abvelocity.ts.constants import (
    ACTUAL_COL,
    ALGO_NAME_COL,
    ALGO_VERSION_COL,
    ANNUAL_SEASONALITY_COL,
    DAILY_SEASONALITY_COL,
    EXTRAS_COL,
    FORECASTED_DATE_COL,
    HOLIDAY_IMPACT_COL,
    LAST_TRAINING_DATE_COL,
    LONGTERM_GROWTH_COL,
    METRIC_ID_COL,
    METRIC_NAME_COL,
    RESIDUAL_COL,
    RUN_DATE_COL,
    RUN_ID_COL,
    SHORTTERM_GROWTH_COL,
    STAGE_COL,
    STAGE_FITTED,
    STAGE_FORECAST,
    STD_COL,
    TIME_COL,
    WEEKLY_SEASONALITY_COL,
)
from abvelocity.ts.result.ts_result import TSResult


class TSRunner:
    """Orchestrates the fit/predict lifecycle for a time-series algorithm.

    Looks up the algorithm by :attr:`config.algo_name` in :data:`ALGO_REGISTRY`,
    calls :meth:`~abvelocity.ts.algo.base.TSAlgo.fit`, then
    :meth:`~abvelocity.ts.algo.base.TSAlgo.predict`.

    The ``prediction_window`` argument mirrors the oi-schemas
    ``PredictionRequest`` / ``AlgoConfig`` split: the window is passed at
    *run time*, not baked into the config.

    After the algorithm returns, the runner stamps a set of
    scheduled-pipeline columns onto ``result_df`` so the output is
    directly consumable by the downstream persistence pipeline.
    Columns the algorithm didn't populate (e.g. forecast components for
    non-decomposing algos) are filled with ``NaN`` / ``None`` so the
    column set is uniform regardless of algo.

    Attributes:
        config: Time-series configuration used for both fit and predict.
    """

    def __init__(self, config: TSModelConfig) -> None:
        self.config = config

    def run(
        self,
        df: pd.DataFrame,
        prediction_window: Optional[tuple[str, str]] = None,
        anomaly_df: Optional[pd.DataFrame] = None,
    ) -> TSResult:
        """Fit the configured algorithm and return predictions.

        Pre-flights the input frame via
        :func:`~abvelocity.ts.common.time_properties.validate_per_dim_timestamps`
        so a panel with missing ``dim_cols`` (collapsed segments → repeated
        timestamps) or a single series with gaps fails loud here rather
        than producing a silently corrupted fit.

        Args:
            df: Training DataFrame.
            prediction_window: Optional ``(start_date, end_date)`` ISO strings
                restricting the prediction output range.
            anomaly_df: Optional known-anomaly intervals used to mask anomalous
                periods during training.

        Returns:
            A :class:`~abvelocity.ts.result.ts_result.TSResult`
            (or subclass) instance with prediction output, with the
            scheduled-pipeline columns stamped on ``result_df``.

        Raises:
            ValueError: When per-dim timestamps are duplicated or
                irregular.  See
                :func:`~abvelocity.ts.common.time_properties.validate_per_dim_timestamps`.
        """
        validate_per_dim_timestamps(
            df=df,
            time_col=self.config.time_col,
            dim_cols=tuple(self.config.dim_cols or ()),
        )

        algo = self.get_algo()
        algo.fit(df=df, config=self.config, anomaly_df=anomaly_df)
        result = algo.predict(prediction_window=prediction_window)
        if result.result_df is not None:
            self._stamp_pipeline_columns(result.result_df)
        return result

    def _stamp_pipeline_columns(self, result_df: pd.DataFrame) -> None:
        """Add the scheduled-pipeline columns in place.

        Algo-produced columns (``ts``, ``metric_id``, ``actual``,
        ``point_forecast``, ``ci_low``, ``ci_high``) are left untouched
        when already present. Metadata columns (``metric_id`` via
        template, ``metric_name`` via template, ``algo_name``,
        ``algo_version``, ``last_training_date``, ``forecasted_date``,
        ``stage``) are derived from the config and/or other columns.
        Components, ``std``, and ``extras`` are added as ``NaN`` / ``None``
        when the algorithm didn't populate them, so the output column
        set is uniform regardless of algo. Run identity (``run_id``,
        ``run_date``) is stamped by the caller, not by the runner.
        """
        # DATE projection of the timestamp column.
        if TIME_COL in result_df.columns:
            ts_col = pd.to_datetime(result_df[TIME_COL], errors="coerce")
            result_df[FORECASTED_DATE_COL] = ts_col.dt.date

        result_df[LAST_TRAINING_DATE_COL] = self.config.train_end_date

        # Stage: "fitted" where actual is observed, "forecast" otherwise.
        if ACTUAL_COL in result_df.columns:
            result_df[STAGE_COL] = np.where(
                result_df[ACTUAL_COL].notna(),
                STAGE_FITTED,
                STAGE_FORECAST,
            )
        else:
            # Defensive: no actual column means we can't distinguish; mark all
            # as forecast so the consumer schema is still complete.
            result_df[STAGE_COL] = STAGE_FORECAST

        # metric_id / metric_name — snapshot the algo-produced value-column
        # name column FIRST so both templates see the original {value_col}
        # (otherwise after metric_id_template renders, METRIC_ID_COL holds
        # the rendered id and metric_name_template would see that instead).
        if METRIC_ID_COL in result_df.columns:
            value_col_series = result_df[METRIC_ID_COL].copy()
        else:
            value_col_series = pd.Series([None] * len(result_df), index=result_df.index)

        metric_id_template = getattr(self.config, "metric_id_template", None)
        if metric_id_template is not None:
            result_df[METRIC_ID_COL] = _render_template(metric_id_template, result_df, value_col_series)
        elif METRIC_ID_COL not in result_df.columns:
            # Algo didn't stamp it AND no template → leave NaN so the column
            # always exists. Rare edge case (algos generally stamp).
            result_df[METRIC_ID_COL] = None

        metric_name_template = getattr(self.config, "metric_name_template", None)
        if metric_name_template is not None:
            result_df[METRIC_NAME_COL] = _render_template(metric_name_template, result_df, value_col_series)
        else:
            result_df[METRIC_NAME_COL] = None

        # algo_name / algo_version — config-level scalars, not templates.
        result_df[ALGO_NAME_COL] = self.config.algo_name
        result_df[ALGO_VERSION_COL] = getattr(self.config, "algo_version", None)

        # Components + std + extras: algos can populate these before
        # returning. Default to NaN / None so the column set is uniform.
        for col in (
            STD_COL,
            LONGTERM_GROWTH_COL,
            SHORTTERM_GROWTH_COL,
            DAILY_SEASONALITY_COL,
            WEEKLY_SEASONALITY_COL,
            ANNUAL_SEASONALITY_COL,
            HOLIDAY_IMPACT_COL,
            RESIDUAL_COL,
        ):
            if col not in result_df.columns:
                result_df[col] = np.nan
        if EXTRAS_COL not in result_df.columns:
            # Object column; each cell is `None` or a `dict[str, float]`.
            result_df[EXTRAS_COL] = None

        # Run identity — stamped by the caller (scheduled-pipeline entry)
        # when known. Leaving NaN here is fine; the caller sets them
        # before persisting.
        if RUN_ID_COL not in result_df.columns:
            result_df[RUN_ID_COL] = None
        if RUN_DATE_COL not in result_df.columns:
            result_df[RUN_DATE_COL] = pd.NaT

    def get_algo(self) -> TSAlgo:
        """Instantiate the algorithm registered under ``config.algo_name``.

        Raises:
            ValueError: If ``config.algo_name`` is not found in
                :data:`ALGO_REGISTRY`.
        """
        cls = ALGO_REGISTRY.get(self.config.algo_name)
        if cls is None:
            raise ValueError(f"Unknown algo_name: {self.config.algo_name!r}. " f"Available: {list(ALGO_REGISTRY.keys())}")
        return cls(algo_params=self.config.algo_params)


def _render_template(
    template: str,
    result_df: pd.DataFrame,
    value_col_series: pd.Series,
) -> pd.Series:
    """Render a format-string template per row of ``result_df``.

    Placeholders are matched against row values: ``{value_col}`` uses the
    corresponding entry of ``value_col_series`` (usually a snapshot of
    the algo-stamped ``METRIC_ID_COL``), any other placeholder is matched
    against a column of the same name (e.g. the dim columns).

    A template with no placeholders returns a constant Series — fast path.
    """
    # Fast path: no placeholders → scalar broadcast.
    if "{" not in template:
        return pd.Series([template] * len(result_df), index=result_df.index)

    def _fmt(row: pd.Series) -> str:
        ctx = {"value_col": value_col_series.loc[row.name]}
        # Expose every column as a potential placeholder so {country}, etc.,
        # resolve. Config-time validation already rejected unknown ones.
        for col, val in row.items():
            ctx.setdefault(col, val)
        return template.format(**ctx)

    return result_df.apply(_fmt, axis=1)

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
"""Live forecast evaluation orchestrator: DataContainer fetchers + ``run_live_forecast_eval``.

The pure-pandas evaluation core (``ForecastEvalReport``,
``evaluate_forecasts_vs_actuals``, the rendering helpers, ``DEFAULT_LIVE_EVAL_METRICS``)
lives in :mod:`abvelocity.ts.eval.forecast_eval_report`. This module
adds the I/O layer:

* SQL templates (:data:`LIVE_EVAL_FORECASTS_QUERY`,
  :data:`LIVE_EVAL_CUTOFFS_QUERY`).
* :func:`list_training_cutoffs` and :func:`fetch_forecasts_at_training_cutoffs`
  — :class:`DataContainer`-polymorphic fetchers (pandas-mode locals or
  SQL-mode ``io_param.cursor`` queries).
* :func:`run_live_forecast_eval` — top-level orchestrator that composes
  the fetchers, the actuals dedupe (from
  :mod:`abvelocity.ts.eval.actuals`), an optional ``transform``
  hook, and the pure-pandas core into a single call.

Two layers, one entry point. Tests use the pure-pandas core directly with
hand-crafted fixtures; production / CloudNotebook notebooks pass a
``DataContainer`` to :func:`run_live_forecast_eval` and let it do the
fetching.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.param.io_param import IOParam
from abvelocity.ts.constants import (
    ACTUAL_COL,
    HORIZON_STEP_COL,
    LAST_TRAINING_DATE_COL,
    TIME_COL,
)
from abvelocity.ts.eval.actuals import fetch_actuals_from_forecast_table
from abvelocity.ts.eval.forecast_eval_report import (
    DEFAULT_LIVE_EVAL_METRICS,
    FigureMaker,
    ForecastEvalReport,
    evaluate_forecasts_vs_actuals,
)

# SQL pulled forecasts: stage='forecast' rows from the listed cutoffs only,
# restricted to days-ahead in [1, max_days_ahead]. Mirrors the script-layer
# template in ``forecast_table_io.FORECASTS_FROM_CUTOFFS_QUERY`` but lives
# in the library so :func:`run_live_forecast_eval` works without depending
# on the dev_testing scripts package.
LIVE_EVAL_FORECASTS_QUERY = """
SELECT
    last_training_date,
    metric_id,
    metric_name,
    forecasted_date,
    forecast,
    forecast_lower,
    forecast_upper,
    DATE_DIFF('day', CAST(last_training_date AS DATE), forecasted_date) AS days_ahead
FROM {table}
WHERE last_training_date IN ({training_cutoff_in_clause})
  AND stage = 'forecast'
  AND DATE_DIFF('day', CAST(last_training_date AS DATE), forecasted_date) BETWEEN 1 AND {max_days_ahead}
ORDER BY last_training_date, metric_id, forecasted_date
"""

# Distinct cutoffs in the table (used when caller doesn't supply training_cutoffs).
LIVE_EVAL_CUTOFFS_QUERY = """
SELECT DISTINCT last_training_date
FROM {table}
WHERE CAST(last_training_date AS DATE) >= CURRENT_DATE - INTERVAL '{lookback_days}' DAY
ORDER BY last_training_date
"""


# A transform takes (forecasts_df, actuals_df) and returns transformed
# (forecast_df, actuals_df) — useful for re-expressing the eval in a
# derived space (e.g. within-week weights, log-space, ratios). Identity by
# default. Forecasts in forecast_df must keep ``last_training_date`` so per-cutoff
# trajectories survive the transform.
TransformFn = Callable[[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]


def list_training_cutoffs(
    dc: DataContainer,
    io_param: Optional[IOParam] = None,
    lookback_days: Optional[int] = None,
    training_cutoff_col: str = LAST_TRAINING_DATE_COL,
) -> List[date]:
    """Distinct training cutoffs from ``dc``, optionally restricted to recent ones.

    Pandas mode: pulls distinct values from ``dc.pandas_df[training_cutoff_col]``.
    SQL mode: runs ``LIVE_EVAL_CUTOFFS_QUERY`` via ``io_param.cursor``.

    Args:
        dc: Forecast partitions container — pandas-mode (with
            ``pandas_df`` set) or SQL-mode (with ``table_name`` set).
        io_param: Required when ``dc.is_sql_table`` — the cursor lives
            on ``io_param.cursor``. Ignored in pandas mode.
        lookback_days: Restrict to cutoffs in the last N days. ``None``
            → all distinct cutoffs. SQL-mode only honors this when
            non-``None``.
        training_cutoff_col: Column name in pandas-mode ``dc.pandas_df``.

    Returns:
        Sorted list of unique :class:`datetime.date` cutoff values.
    """
    if dc.is_pandas_df:
        if dc.pandas_df is None:
            raise ValueError("dc.is_pandas_df=True but pandas_df is None.")
        if training_cutoff_col not in dc.pandas_df.columns:
            raise ValueError(
                f"pandas_df is missing required column {training_cutoff_col!r}."
            )
        cutoffs = pd.to_datetime(dc.pandas_df[training_cutoff_col]).dt.normalize().unique()
        cutoffs_dates = sorted(pd.Timestamp(c).date() for c in cutoffs)
        if lookback_days is not None and cutoffs_dates:
            cutoff_floor = pd.Timestamp(max(cutoffs_dates)) - pd.Timedelta(days=lookback_days)
            cutoffs_dates = [c for c in cutoffs_dates if pd.Timestamp(c) >= cutoff_floor]
        return cutoffs_dates
    if dc.is_sql_table:
        if dc.table_name is None:
            raise ValueError("dc.is_sql_table=True but table_name is None.")
        if io_param is None or io_param.cursor is None:
            raise ValueError("io_param.cursor is required for SQL-mode dc.")
        if lookback_days is None:
            lookback_days = 365  # default upper bound; keeps the SQL well-formed.
        sql = LIVE_EVAL_CUTOFFS_QUERY.format(
            table=dc.table_name, lookback_days=lookback_days,
        )
        df = io_param.cursor.get_df(sql).df
        if df is None or df.empty:
            return []
        return [pd.to_datetime(v).date() for v in df[training_cutoff_col]]
    raise ValueError("dc must be pandas-mode or SQL-mode.")


def fetch_forecasts_at_training_cutoffs(
    dc: DataContainer,
    training_cutoffs: Sequence[date],
    max_days_ahead: int,
    io_param: Optional[IOParam] = None,
    *,
    stage_col: str = "stage",
    forecast_stage_value: str = "forecast",
    training_cutoff_col: str = LAST_TRAINING_DATE_COL,
    forecasted_date_col: str = "forecasted_date",
) -> pd.DataFrame:
    """Pull stage='forecast' rows at the listed cutoffs, restricted to days +1..max_days_ahead.

    Pandas-mode: filters ``dc.pandas_df`` locally.
    SQL-mode: runs ``LIVE_EVAL_FORECASTS_QUERY`` via ``io_param.cursor``.

    Returns the same column shape regardless of mode.

    Args:
        dc: Forecast partitions container.
        training_cutoffs: Cutoff dates to include.
        max_days_ahead: Max horizon (inclusive) — rows where
            ``forecasted_date - training_cutoff > max_days_ahead`` are
            dropped.
        io_param: Required for SQL mode (carries the cursor).
        stage_col: Stage column in pandas-mode ``dc.pandas_df``.
        forecast_stage_value: Stage value identifying forecast rows.
        training_cutoff_col: Cutoff column in pandas-mode.
        forecasted_date_col: Forecasted-date column in pandas-mode.

    Returns:
        Long DataFrame with ``last_training_date`` (datetime),
        ``metric_id``, ``forecasted_date`` (datetime), ``forecast``,
        ``forecast_lower``, ``forecast_upper``, ``days_ahead``. May
        also carry ``metric_name`` when present in the source.
    """
    if not training_cutoffs:
        return pd.DataFrame(
            columns=[
                training_cutoff_col, "metric_id", forecasted_date_col,
                "forecast", "forecast_lower", "forecast_upper", "days_ahead",
            ]
        )

    if dc.is_pandas_df:
        if dc.pandas_df is None:
            raise ValueError("dc.is_pandas_df=True but pandas_df is None.")
        df = dc.pandas_df
        cutoffs_ts = {pd.Timestamp(c).normalize() for c in training_cutoffs}
        local_cutoffs = pd.to_datetime(df[training_cutoff_col]).dt.normalize()
        local_dates = pd.to_datetime(df[forecasted_date_col])
        days_ahead = (local_dates - local_cutoffs).dt.days
        mask = (
            (df[stage_col] == forecast_stage_value)
            & local_cutoffs.isin(cutoffs_ts)
            & (days_ahead >= 1)
            & (days_ahead <= max_days_ahead)
        )
        out = df[mask].copy()
        out[forecasted_date_col] = pd.to_datetime(out[forecasted_date_col])
        out[training_cutoff_col] = pd.to_datetime(out[training_cutoff_col])
        out["days_ahead"] = days_ahead[mask].astype("Int64")
        return out.reset_index(drop=True)
    if dc.is_sql_table:
        if dc.table_name is None:
            raise ValueError("dc.is_sql_table=True but table_name is None.")
        if io_param is None or io_param.cursor is None:
            raise ValueError("io_param.cursor is required for SQL-mode dc.")
        in_clause = ", ".join(
            f"'{pd.Timestamp(c).date().isoformat()}'" for c in training_cutoffs
        )
        sql = LIVE_EVAL_FORECASTS_QUERY.format(
            table=dc.table_name,
            training_cutoff_in_clause=in_clause,
            max_days_ahead=max_days_ahead,
        )
        df = io_param.cursor.get_df(sql).df
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df[forecasted_date_col] = pd.to_datetime(df[forecasted_date_col])
        df[training_cutoff_col] = pd.to_datetime(df[training_cutoff_col])
        return df
    raise ValueError("dc must be pandas-mode or SQL-mode.")


def run_live_forecast_eval(
    dc: DataContainer,
    io_param: Optional[IOParam] = None,
    *,
    training_cutoffs: Optional[Sequence[date]] = None,
    lookback_days: int = 90,
    max_days_ahead: int = 14,
    transform: Optional[TransformFn] = None,
    metrics: Tuple[str, ...] = DEFAULT_LIVE_EVAL_METRICS,
    actuals_window_days: Optional[int] = 90,
    figure_makers: Optional[Sequence[FigureMaker]] = None,
    report_title: str = "Live forecast evaluation",
    report_summary: str = "",
) -> ForecastEvalReport:
    """End-to-end orchestrator: fetch → optional transform → evaluate.

    Pulls forecasts at the chosen cutoffs and the deduped actuals from
    ``dc`` (pandas or SQL mode), runs an optional ``transform`` over
    them (e.g. within-week weighting), then delegates to
    :func:`evaluate_forecasts_vs_actuals`.

    Args:
        dc: Forecast partitions container — typically ``DataContainer(
            is_sql_table=True, table_name="u_clearsight.…")`` for prod
            data or ``DataContainer(is_pandas_df=True, pandas_df=…)``
            for tests / CloudNotebook notebooks.
        io_param: Required for SQL-mode ``dc`` — carries the cursor.
        training_cutoffs: Cutoff dates to include. ``None`` →
            :func:`list_training_cutoffs` with ``lookback_days``.
        lookback_days: Recent-window for cutoff inference. Ignored when
            ``training_cutoffs`` is supplied.
        max_days_ahead: Max horizon (inclusive). Forwarded to both the
            forecast fetch and :func:`evaluate_forecasts_vs_actuals`.
        transform: Optional ``(forecast_df, actuals_df) →
            (transformed_forecast_df, transformed_actuals_df)`` callable.
            Identity by default. The transformed forecast frame must keep
            ``last_training_date`` so per-cutoff trajectories survive the
            transform.
        metrics: Forwarded to :func:`evaluate_forecasts_vs_actuals`.
        actuals_window_days: Forwarded to
            :func:`evaluate_forecasts_vs_actuals`.
        figure_makers: Forwarded to :func:`evaluate_forecasts_vs_actuals`.
            ``None`` → use :data:`DEFAULT_FIGURE_MAKERS` (per-training-cutoff
            forecast overlays + per-horizon error curve + per-horizon forecast
            overlays). Pass a custom list to swap or restrict the rendered
            sections (e.g. weight-quality reports drop the per-horizon error
            curve since the triangular cutoff×weekday distribution makes it
            wonky).
        report_title: Forwarded to :func:`evaluate_forecasts_vs_actuals`.
        report_summary: Forwarded to :func:`evaluate_forecasts_vs_actuals`.

    Returns:
        :class:`ForecastEvalReport`.
    """
    if training_cutoffs is None:
        training_cutoffs = list_training_cutoffs(
            dc=dc, io_param=io_param, lookback_days=lookback_days,
        )
        if not training_cutoffs:
            raise ValueError(
                f"No training cutoffs found in dc within last {lookback_days} days; "
                "nothing to evaluate."
            )

    forecasts = fetch_forecasts_at_training_cutoffs(
        dc=dc,
        training_cutoffs=training_cutoffs,
        max_days_ahead=max_days_ahead,
        io_param=io_param,
    )
    actuals = fetch_actuals_from_forecast_table(dc=dc, io_param=io_param)

    # Forecast rows carry their own ``actual`` column (NaN for stage='forecast'
    # in the partition layout), which would name-conflict with the deduped
    # ``actual`` joined from ``actuals``. Drop it before the merge.
    if ACTUAL_COL in forecasts.columns:
        forecasts = forecasts.drop(columns=[ACTUAL_COL])
    forecast_df = forecasts.merge(actuals, on=["metric_id", "forecasted_date"], how="left")
    forecast_df = forecast_df.rename(columns={"forecasted_date": TIME_COL, "days_ahead": HORIZON_STEP_COL})
    actuals_df = actuals.rename(columns={"forecasted_date": TIME_COL})

    if transform is not None:
        forecast_df, actuals_df = transform(forecast_df, actuals_df)

    return evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df,
        actuals_df=actuals_df,
        training_cutoffs=[pd.to_datetime(c) for c in training_cutoffs],
        max_days_ahead=max_days_ahead,
        metrics=metrics,
        actuals_window_days=actuals_window_days,
        figure_makers=figure_makers,
        report_title=report_title,
        report_summary=report_summary,
    )

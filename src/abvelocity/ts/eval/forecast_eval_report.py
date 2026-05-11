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
"""Pure-pandas forecast-evaluation core: errors → figure sections → HTML report.

This module is the pandas-only side of the live-eval stack — no
:class:`DataContainer`, no SQL, no I/O. It takes already-loaded frames and
emits a :class:`ForecastEvalReport` (errors per (metric, horizon) and per
metric, a list of :class:`FigureSection` views, self-contained HTML string).

The companion module :mod:`abvelocity.ts.eval.live_eval` adds the
DataContainer-polymorphic fetchers and the ``run_live_forecast_eval``
orchestrator that wires Trino / pandas-mode I/O into this core.

Default figure sections (each a :class:`FigureSection` produced by a
:data:`FigureMaker` callable):

* **Per-training-cutoff forecast overlays** — continuous actuals + each cutoff's
  ``+1..max_days_ahead`` forecast trajectory overlaid. Answers: "are
  individual forecasts tracking?" Capped to the ``max_cutoffs_to_plot``
  most recent cutoffs (errors are still computed across every cutoff).
* **Per-horizon error curve** — accuracy as a function of
  ``horizon_step``. Answers: "how fast does the model degrade as we
  forecast further out?"
* **Per-horizon forecast overlays** — actuals plus one continuous curve per chosen
  horizon (e.g. 1d, 7d, 14d), plotted across time. Answers: "are
  forecasts at different horizons consistent? does the forecast drift
  as the horizon grows?" Spread between the curves is the model's
  cross-horizon stability.

Public surface:

* :class:`ForecastEvalReport` — dataclass with the joined panel,
  errors, figure sections, and rendered HTML.
* :class:`FigureSection`, :data:`FigureMaker`, :class:`FigureMakerContext`
  — extension hooks for adding/replacing figure sections.
* :func:`make_per_training_cutoff_forecast_overlays`,
  :func:`make_per_horizon_error_curve`,
  :func:`make_per_horizon_forecast_overlays` — factories for the default figure
  makers, each parameterizable.
* :data:`DEFAULT_FIGURE_MAKERS` — the default ``[per-training-cutoff forecast overlays, error_curve,
  per_horizon_forecast_overlays]`` list used when callers don't pass their own.
* :func:`evaluate_forecasts_vs_actuals` — the pure-pandas orchestrator.
* :data:`DEFAULT_LIVE_EVAL_METRICS` — default metric tuple shared with
  the live orchestrator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
from abvelocity.core.utils.color_utils import get_distinct_colors
from abvelocity.ts.constants import (
    ACTUAL_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    HORIZON_STEP_COL,
    LAST_TRAINING_DATE_COL,
    METRIC_ID_COL,
    TIME_COL,
)
from abvelocity.ts.eval.forecast_eval import compute_eval
from abvelocity.ts.viz import plot_forecast_at_training_cutoffs, plot_forecast_groups_vs_actual

DEFAULT_LIVE_EVAL_METRICS: Tuple[str, ...] = (
    "mape", "smape", "mae", "rmse", "coverage",
)


@dataclass
class FigureMakerContext:
    """All inputs a :data:`FigureMaker` may need — pre-computed once per evaluation.

    Passed by :func:`evaluate_forecasts_vs_actuals` to each registered
    :data:`FigureMaker`. Carries both the joined panel and the pre-computed
    error tables so makers don't have to recompute them.

    Attributes:
        forecast_df: Long panel filtered to the chosen cutoffs and the
            ``[1, max_days_ahead]`` horizon window. One row per (metric,
            cutoff, forecasted_date).
        actuals_df: Continuous (metric_id, ts, actual) overlay or
            ``None`` when the caller didn't supply one.
        errors_per_metric_horizon: Output of :func:`compute_eval` grouped
            by (metric_id, horizon_step).
        errors_per_metric: Output of :func:`compute_eval` grouped by
            (metric_id,) only.
        training_cutoffs: Cutoff values resolved for this evaluation
            (sorted ascending). Use ``[-N:]`` slicing in your maker if
            you want the most-recent N for plotting.
        max_days_ahead: Inclusive horizon cap that was applied to
            ``forecast_df``.
        metric_id_col: Metric column name (defaults to ``"metric_id"``).
        time_col: Forecasted-date column.
        actual_col: Actual-value column.
        forecast_col: Point-forecast column.
        forecast_lower_col: Lower CI column or ``None`` when no CI.
        forecast_upper_col: Upper CI column or ``None`` when no CI.
        training_cutoff_col: Training-cutoff column.
        horizon_step_col: Horizon-step column.
        has_ci: Convenience: ``True`` iff both CI columns are present in
            ``forecast_df`` (so makers don't have to check both).
        actuals_window_days: Forwarded so makers that show actuals can
            re-window them consistently with the rest of the report.
        ylabel: Suggested per-figure y-label (usually the metric_id).
    """

    forecast_df: pd.DataFrame
    actuals_df: Optional[pd.DataFrame]
    errors_per_metric_horizon: pd.DataFrame
    errors_per_metric: pd.DataFrame
    training_cutoffs: List[Any]
    max_days_ahead: int
    metric_id_col: str
    time_col: str
    actual_col: str
    forecast_col: str
    forecast_lower_col: Optional[str]
    forecast_upper_col: Optional[str]
    training_cutoff_col: str
    horizon_step_col: str
    has_ci: bool
    actuals_window_days: Optional[int]
    ylabel: Optional[str]


@dataclass
class FigureSection:
    """One titled section of the HTML report — title, description, per-metric figures.

    Attributes:
        name: Stable machine-readable identifier (e.g.
            ``"per_training_cutoff_forecast_overlays"``). Used for HTML ``div`` ids
            and for callers who want to look the section up by name.
        title: Human-readable section heading rendered as ``<h2>`` in the
            HTML (e.g. ``"Forecast vs actual at training cutoffs"``).
        description: Short prose rendered between the heading and the
            figures. Use it to call out what the reader should look for
            (e.g. "Spread between curves = cross-horizon stability").
        figures: One Plotly figure per metric_id. Empty dict is allowed
            (the section just renders an italic placeholder).
    """

    name: str
    title: str
    description: str
    figures: Dict[str, go.Figure] = field(default_factory=dict)


# Each figure maker is a callable that takes a fully-populated context and
# returns one section of the report. Several makers can be registered;
# :func:`evaluate_forecasts_vs_actuals` calls them in order.
FigureMaker = Callable[[FigureMakerContext], FigureSection]


@dataclass
class ForecastEvalReport:
    """Result of an end-to-end forecast evaluation.

    Attributes:
        forecast_df: Joined long-format DataFrame with one row per
            (metric_id, training_cutoff, forecasted_date), holding
            forecast / actual / CI / horizon columns.
        actuals_df: Continuous (metric_id, ts, actual) series used
            for the actuals reference lines. Spans more history than
            ``forecast_df`` whenever ``actuals_window_days`` is larger
            than the forecast windows.
        training_cutoffs: List of training-cutoff values used.
        errors_per_metric_horizon: One row per (metric_id,
            horizon_step) with the configured metrics — the
            horizon-degradation curve.
        errors_per_metric: One row per metric_id with the same metrics
            aggregated across all horizons — the headline accuracy.
        figure_sections: One :class:`FigureSection` per registered figure
            maker. Defaults render three: per-training-cutoff forecast
            overlays, per-horizon error curve, and per-horizon forecast
            overlays. Each section bundles a title, description, and
            per-metric figures.
        html_str: Self-contained HTML string with summary, per-metric
            and per-(metric, horizon) error tables, and every figure
            section embedded inline. Write to disk via
            :meth:`write_html`.
    """

    forecast_df: pd.DataFrame
    actuals_df: pd.DataFrame
    training_cutoffs: List[Any]
    errors_per_metric_horizon: pd.DataFrame
    errors_per_metric: pd.DataFrame
    figure_sections: List[FigureSection] = field(default_factory=list)
    html_str: str = ""

    def write_html(self, path: Path) -> Path:
        """Write :attr:`html_str` to ``path``, creating parent dirs as needed.

        Args:
            path: Destination ``.html`` file. Missing parent directories are
                created automatically (``mkdir(parents=True)``). An existing
                file at ``path`` is overwritten.

        Returns:
            The same ``path`` (as a :class:`pathlib.Path`) — handy for chaining
            (e.g. ``open(report.write_html(...))``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html_str)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Default figure makers — factory functions returning configured FigureMakers
# ─────────────────────────────────────────────────────────────────────────────


def make_per_training_cutoff_forecast_overlays(
    *,
    max_cutoffs_to_plot: int = 8,
    name: str = "per_training_cutoff_forecast_overlays",
    title: str = "Forecast vs actual at training cutoffs",
    description: str = (
        "Continuous gray line: actual values. Each colored short trajectory: "
        "one training cutoff's <code>+1..max_days_ahead</code> forecast with its "
        "CI band. Capped to the most recent cutoffs to keep the plot readable; "
        "errors above are computed across <em>all</em> cutoffs."
    ),
) -> FigureMaker:
    """Build the per-training-cutoff per-training-cutoff forecast overlays figure maker.

    For each metric, draws a continuous actuals line plus one short forecast
    trajectory per training cutoff, each with its CI band. Most recent N
    cutoffs only — beyond ~8, the plot becomes visual noise.

    Args:
        max_cutoffs_to_plot: How many of the most recent cutoffs to draw.
            Defaults to 8. The error tables in the report always reflect
            every cutoff in ``forecast_df`` regardless of this cap.
        name: Identifier baked into the returned :class:`FigureSection`.
        title: HTML ``<h2>`` heading for the section.
        description: HTML/markdown-style snippet between the heading and
            the figures.

    Returns:
        A :data:`FigureMaker` (callable) ready to slot into
        ``evaluate_forecasts_vs_actuals(figure_makers=…)``.
    """

    def make(ctx: FigureMakerContext) -> FigureSection:
        """Render one overlay figure per metric using the configured cap.

        Args:
            ctx: The shared :class:`FigureMakerContext`.

        Returns:
            :class:`FigureSection` with one Plotly figure per metric_id.
        """
        cutoffs_to_plot = (
            list(ctx.training_cutoffs[-max_cutoffs_to_plot:])
            if len(ctx.training_cutoffs) > max_cutoffs_to_plot
            else list(ctx.training_cutoffs)
        )
        figures: Dict[str, go.Figure] = {}
        for metric_id in sorted(ctx.forecast_df[ctx.metric_id_col].unique()):
            metric_panel = ctx.forecast_df[ctx.forecast_df[ctx.metric_id_col] == metric_id]
            metric_panel = metric_panel[metric_panel[ctx.training_cutoff_col].isin(cutoffs_to_plot)]
            if metric_panel.empty:
                continue
            metric_overlay = _select_actuals_df_for_metric(
                actuals_df=ctx.actuals_df,
                metric_id=metric_id,
                metric_id_col=ctx.metric_id_col,
                time_col=ctx.time_col,
                actual_col=ctx.actual_col,
                forecast_df=metric_panel,
                actuals_window_days=ctx.actuals_window_days,
            )
            figures[metric_id] = plot_forecast_at_training_cutoffs(
                df=metric_panel,
                training_cutoffs=cutoffs_to_plot,
                max_days_ahead=ctx.max_days_ahead,
                training_cutoff_col=ctx.training_cutoff_col,
                time_col=ctx.time_col,
                actual_col=ctx.actual_col,
                forecast_col=ctx.forecast_col,
                forecast_lower_col=ctx.forecast_lower_col if ctx.has_ci else None,
                forecast_upper_col=ctx.forecast_upper_col if ctx.has_ci else None,
                actuals_df=metric_overlay,
                training_cutoff_label_template="cutoff {value}",
                title=f"{metric_id} — forecast vs actual at training cutoffs",
                xlabel="forecasted date",
                ylabel=ctx.ylabel or metric_id,
            )
        return FigureSection(name=name, title=title, description=description, figures=figures)

    return make


def make_per_horizon_error_curve(
    *,
    error_metrics: Sequence[str] = ("mape", "smape"),
    name: str = "per_horizon_error_curve",
    title: str = "Per-horizon error curve",
    description: str = (
        "Error metrics as a function of horizon step. A flat curve means the "
        "model degrades gracefully; a sharp upward curve means the further-out "
        "forecasts are unreliable."
    ),
) -> FigureMaker:
    """Build the per-horizon error-degradation curve figure maker.

    For each metric, draws one line per ``error_metric`` showing how the
    error grows as ``horizon_step`` increases.

    Args:
        error_metrics: Which error columns from ``errors_per_metric_horizon``
            to overlay. Defaults to ``("mape", "smape")``. Ones not present
            in the table (e.g. coverage was dropped because CI was missing)
            are silently skipped.
        name: Identifier baked into the returned :class:`FigureSection`.
        title: HTML ``<h2>`` heading.
        description: Section description.

    Returns:
        A :data:`FigureMaker` callable.
    """

    def make(ctx: FigureMakerContext) -> FigureSection:
        """Render one per-horizon error curve per metric.

        Args:
            ctx: The shared :class:`FigureMakerContext`.

        Returns:
            :class:`FigureSection` with one Plotly figure per metric_id.
        """
        figures: Dict[str, go.Figure] = {}
        present_metrics = [m for m in error_metrics if m in ctx.errors_per_metric_horizon.columns]
        if not present_metrics:
            return FigureSection(name=name, title=title, description=description, figures={})
        colors = get_distinct_colors(num_colors=len(present_metrics), opacity=0.9)
        for metric_id in sorted(ctx.errors_per_metric_horizon[ctx.metric_id_col].unique()):
            sub = (
                ctx.errors_per_metric_horizon[ctx.errors_per_metric_horizon[ctx.metric_id_col] == metric_id]
                .sort_values(ctx.horizon_step_col)
            )
            if sub.empty:
                continue
            fig = go.Figure()
            for idx, error_metric in enumerate(present_metrics):
                fig.add_trace(go.Scatter(
                    x=sub[ctx.horizon_step_col],
                    y=sub[error_metric],
                    mode="lines+markers",
                    name=error_metric,
                    line=dict(color=colors[idx], width=2),
                    marker=dict(size=5),
                ))
            fig.update_layout(
                title=f"{metric_id} — error vs horizon",
                xaxis_title="horizon step (days ahead)",
                yaxis_title="error",
                height=380,
                margin=dict(t=60, b=50),
            )
            figures[metric_id] = fig
        return FigureSection(name=name, title=title, description=description, figures=figures)

    return make


def make_per_horizon_forecast_overlays(
    *,
    horizons: Sequence[int] = (1, 7, 14),
    name: str = "per_horizon_forecast_overlays",
    title: str = "Per-horizon forecast overlays",
    description: str = (
        "One continuous curve per chosen horizon (e.g. 1d, 7d, 14d ahead) plus the "
        "actuals series. <strong>Spread between the curves is the model's "
        "cross-horizon stability</strong> — if 1-day-ahead and 14-day-ahead forecasts "
        "agree on the same future trajectory, the model is robust; if they fan out, "
        "the further-out horizons are drifting."
    ),
) -> FigureMaker:
    """Build the per-horizon rolling-forecast overlay figure maker.

    For each metric, picks rows at each requested ``horizon_step`` and draws
    them as a continuous time series, alongside the actuals reference. Each
    horizon gets its own colored line (and CI band if available).

    Args:
        horizons: Horizon-step values to overlay. Defaults to ``(1, 7, 14)``.
            Horizons not present in ``forecast_df`` (e.g. requested 14 with
            ``max_days_ahead=7``) are silently dropped.
        name: Identifier baked into the returned :class:`FigureSection`.
        title: HTML ``<h2>`` heading.
        description: Section description.

    Returns:
        A :data:`FigureMaker` callable.
    """

    def make(ctx: FigureMakerContext) -> FigureSection:
        """Render one per-horizon overlay per metric.

        Args:
            ctx: The shared :class:`FigureMakerContext`.

        Returns:
            :class:`FigureSection` with one Plotly figure per metric_id.
        """
        figures: Dict[str, go.Figure] = {}
        for metric_id in sorted(ctx.forecast_df[ctx.metric_id_col].unique()):
            metric_panel = ctx.forecast_df[ctx.forecast_df[ctx.metric_id_col] == metric_id]
            if metric_panel.empty:
                continue
            present_horizons = [
                h for h in horizons if h in set(metric_panel[ctx.horizon_step_col].unique())
            ]
            if not present_horizons:
                continue
            metric_overlay = _select_actuals_df_for_metric(
                actuals_df=ctx.actuals_df,
                metric_id=metric_id,
                metric_id_col=ctx.metric_id_col,
                time_col=ctx.time_col,
                actual_col=ctx.actual_col,
                forecast_df=metric_panel,
                actuals_window_days=ctx.actuals_window_days,
            )
            figures[metric_id] = plot_forecast_groups_vs_actual(
                df=metric_panel,
                group_col=ctx.horizon_step_col,
                group_values=present_horizons,
                time_col=ctx.time_col,
                actual_col=ctx.actual_col,
                forecast_col=ctx.forecast_col,
                forecast_lower_col=ctx.forecast_lower_col if ctx.has_ci else None,
                forecast_upper_col=ctx.forecast_upper_col if ctx.has_ci else None,
                group_label_template="horizon={value}d",
                title=f"{metric_id} — rolling forecasts at horizons {list(present_horizons)} vs actual",
                xlabel="forecasted date",
                ylabel=ctx.ylabel or metric_id,
                actuals_df=metric_overlay,
            )
        return FigureSection(name=name, title=title, description=description, figures=figures)

    return make


# Default figure-makers list. Order matters — sections render top-to-bottom
# in the HTML report.
DEFAULT_FIGURE_MAKERS: List[FigureMaker] = [
    make_per_training_cutoff_forecast_overlays(),
    make_per_horizon_error_curve(),
    make_per_horizon_forecast_overlays(),
]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level evaluator
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_forecasts_vs_actuals(
    forecast_df: pd.DataFrame,
    actuals_df: Optional[pd.DataFrame] = None,
    *,
    metric_id_col: str = METRIC_ID_COL,
    time_col: str = TIME_COL,
    actual_col: str = ACTUAL_COL,
    forecast_col: str = FORECAST_COL,
    forecast_lower_col: Optional[str] = FORECAST_LOWER_COL,
    forecast_upper_col: Optional[str] = FORECAST_UPPER_COL,
    training_cutoff_col: str = LAST_TRAINING_DATE_COL,
    horizon_step_col: str = HORIZON_STEP_COL,
    training_cutoffs: Optional[Sequence[Any]] = None,
    max_days_ahead: int = 14,
    metrics: Tuple[str, ...] = DEFAULT_LIVE_EVAL_METRICS,
    actuals_window_days: Optional[int] = 90,
    figure_makers: Optional[Sequence[FigureMaker]] = None,
    report_title: str = "Live forecast evaluation",
    report_summary: str = "",
    ylabel: Optional[str] = None,
) -> ForecastEvalReport:
    """End-to-end eval on a pre-joined forecast/actual frame — pure pandas, no I/O.

    Steps performed:

      1. Validate required columns are present in ``forecast_df``.
      2. Filter to ``training_cutoffs`` (or use every distinct cutoff
         present in ``forecast_df``).
      3. Add a ``horizon_step_col`` derived from
         ``time_col − training_cutoff_col`` if missing, then drop rows
         outside ``[1, max_days_ahead]``.
      4. Drop rows where actual is NaN (forecast-only rows can't
         contribute to errors).
      5. Compute per-(metric, horizon) and per-metric errors via
         :func:`compute_eval`.
      6. Run each registered :data:`FigureMaker` in order, collecting
         their :class:`FigureSection` outputs.
      7. Compose a self-contained HTML report with the figure sections
         and error tables.

    Args:
        forecast_df: Long-format DataFrame with one row per (metric, training
            cutoff, forecasted timestamp). Must contain
            ``metric_id_col``, ``time_col``, ``training_cutoff_col``,
            ``forecast_col``, ``actual_col``, optionally
            ``forecast_lower_col`` / ``forecast_upper_col``,
            ``horizon_step_col``.
        actuals_df: Optional continuous-actuals DataFrame with
            ``metric_id_col``, ``time_col``, ``actual_col``. Drives the
            gray reference lines in the figure sections that need it.
            ``None`` → makers fall back to deduping from ``forecast_df``.
        metric_id_col: Column identifying the metric (e.g.
            ``"randomProduct_signups:daily"``). Used to render one figure per
            metric per section and to group the error tables.
        time_col: Forecasted timestamp column.
        actual_col: Actual-value column.
        forecast_col: Point-forecast column.
        forecast_lower_col: Lower CI bound column; ``None`` → no CI bands
            in figures and ``coverage`` will be dropped from
            ``metrics`` if present.
        forecast_upper_col: Upper CI bound column; ``None`` → no CI bands
            in figures and ``coverage`` will be dropped from
            ``metrics`` if present.
        training_cutoff_col: Training-cutoff column.
        horizon_step_col: Horizon-step column. Computed in-place when
            absent from ``forecast_df`` (as integer days from cutoff to
            forecasted date).
        training_cutoffs: Subset of cutoff values to evaluate. ``None``
            → every distinct cutoff in ``forecast_df`` after the
            days-ahead filter.
        max_days_ahead: Drop rows whose horizon-step is < 1 or
            > ``max_days_ahead``.
        metrics: Forwarded to :func:`compute_eval` for both the
            per-(metric, horizon) and per-metric breakdowns.
        actuals_window_days: When supplied (and ``actuals_df`` is
            given), restrict figure-section actuals traces to the last
            ``actuals_window_days`` days before the latest forecast_df
            timestamp. Avoids dwarfing the forecast trajectories under
            a multi-year continuous actuals line. ``None`` → no
            restriction.
        figure_makers: Sequence of :data:`FigureMaker` callables.
            ``None`` → use :data:`DEFAULT_FIGURE_MAKERS`
            (per-training-cutoff forecast overlays + per-horizon error
            curve + per-horizon forecast overlays). Pass an empty list
            to skip figure rendering entirely.
        report_title: Title used in the HTML ``<h1>``.
        report_summary: Optional HTML/markdown-style snippet rendered
            below the title (e.g. context, link to source, caveats).
        ylabel: Default y-axis label suggested to figure makers.
            ``None`` → makers use the metric_id.

    Returns:
        :class:`ForecastEvalReport`.

    Raises:
        ValueError: if required columns are missing; if no rows survive
            the days-ahead / cutoff filter; if ``training_cutoffs``
            references values not in ``forecast_df``.
    """
    required = {metric_id_col, time_col, training_cutoff_col, forecast_col, actual_col}
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"forecast_df is missing required columns: {sorted(missing)}.")

    forecast_df = forecast_df.copy()
    forecast_df[time_col] = pd.to_datetime(forecast_df[time_col])
    forecast_df[training_cutoff_col] = pd.to_datetime(forecast_df[training_cutoff_col])

    if horizon_step_col not in forecast_df.columns:
        forecast_df[horizon_step_col] = (
            forecast_df[time_col] - forecast_df[training_cutoff_col]
        ).dt.days.astype("Int64")

    horizon_mask = (forecast_df[horizon_step_col] >= 1) & (forecast_df[horizon_step_col] <= max_days_ahead)
    forecast_df = forecast_df[horizon_mask]
    if forecast_df.empty:
        raise ValueError(
            f"No rows in forecast_df after filtering to horizon in [1, {max_days_ahead}]. "
            f"Check that {time_col} > {training_cutoff_col} for at least some rows."
        )

    if training_cutoffs is None:
        resolved_cutoffs: List[Any] = sorted(forecast_df[training_cutoff_col].unique())
    else:
        requested = [pd.to_datetime(c) for c in training_cutoffs]
        present = set(forecast_df[training_cutoff_col].unique())
        unknown = [c for c in requested if c not in present]
        if unknown:
            raise ValueError(
                f"training_cutoffs not present in forecast_df after filter: "
                f"{[str(c) for c in unknown[:5]]}{'…' if len(unknown) > 5 else ''}"
            )
        forecast_df = forecast_df[forecast_df[training_cutoff_col].isin(requested)]
        resolved_cutoffs = list(requested)

    eval_input = forecast_df.dropna(subset=[forecast_col, actual_col]).copy()
    if eval_input.empty:
        raise ValueError(
            "No rows with both forecast and actual non-null after filter — "
            "cannot compute any errors. Check that the actuals join populated."
        )
    # ``compute_eval`` accesses CI bounds only when ``coverage`` is requested.
    # Downgrade gracefully when CI cols are missing — error tables are still
    # useful without coverage.
    metrics_to_use = list(metrics)
    has_ci = (
        forecast_lower_col is not None
        and forecast_upper_col is not None
        and forecast_lower_col in eval_input.columns
        and forecast_upper_col in eval_input.columns
    )
    if "coverage" in metrics_to_use and not has_ci:
        warnings.warn(
            "coverage requested but forecast_lower / forecast_upper not in forecast_df; "
            "dropping coverage from metrics.",
            stacklevel=2,
        )
        metrics_to_use = [m for m in metrics_to_use if m != "coverage"]

    errors_per_metric_horizon = compute_eval(
        eval_input,
        metrics=tuple(metrics_to_use),
        group_by=(metric_id_col, horizon_step_col),
    )
    errors_per_metric = compute_eval(
        eval_input,
        metrics=tuple(metrics_to_use),
        group_by=(metric_id_col,),
    )

    ctx = FigureMakerContext(
        forecast_df=forecast_df,
        actuals_df=actuals_df,
        errors_per_metric_horizon=errors_per_metric_horizon,
        errors_per_metric=errors_per_metric,
        training_cutoffs=resolved_cutoffs,
        max_days_ahead=max_days_ahead,
        metric_id_col=metric_id_col,
        time_col=time_col,
        actual_col=actual_col,
        forecast_col=forecast_col,
        forecast_lower_col=forecast_lower_col,
        forecast_upper_col=forecast_upper_col,
        training_cutoff_col=training_cutoff_col,
        horizon_step_col=horizon_step_col,
        has_ci=has_ci,
        actuals_window_days=actuals_window_days,
        ylabel=ylabel,
    )
    makers = list(figure_makers) if figure_makers is not None else list(DEFAULT_FIGURE_MAKERS)
    figure_sections: List[FigureSection] = [maker(ctx) for maker in makers]

    html_str = _render_eval_html(
        report_title=report_title,
        report_summary=report_summary,
        training_cutoffs=resolved_cutoffs,
        errors_per_metric=errors_per_metric,
        errors_per_metric_horizon=errors_per_metric_horizon,
        figure_sections=figure_sections,
    )

    return ForecastEvalReport(
        forecast_df=forecast_df.reset_index(drop=True),
        actuals_df=actuals_df if actuals_df is not None else pd.DataFrame(),
        training_cutoffs=resolved_cutoffs,
        errors_per_metric_horizon=errors_per_metric_horizon,
        errors_per_metric=errors_per_metric,
        figure_sections=figure_sections,
        html_str=html_str,
    )


def _select_actuals_df_for_metric(
    actuals_df: Optional[pd.DataFrame],
    metric_id: str,
    metric_id_col: str,
    time_col: str,
    actual_col: str,
    forecast_df: pd.DataFrame,
    actuals_window_days: Optional[int],
) -> Optional[pd.DataFrame]:
    """Pick the per-metric actuals slice, optionally windowed to recent history.

    Args:
        actuals_df: Continuous-actuals frame (with ``metric_id_col``,
            ``time_col``, ``actual_col``). ``None`` / empty → caller draws
            actuals dedup'd from the forecast frame.
        metric_id: Metric whose slice we want.
        metric_id_col: Column to filter on (skipped when ``actuals_df``
            doesn't carry it — callers can pass already-filtered frames).
        time_col: Timestamp column.
        actual_col: Actual-value column kept in the returned slice.
        forecast_df: Per-metric forecast frame; only used to anchor the
            windowing horizon (``max_ts`` of its ``time_col``).
        actuals_window_days: When set, restrict to rows in
            ``[max_forecast_ts - days, max_forecast_ts]``. ``None`` → keep
            the entire history.

    Returns:
        A ``[time_col, actual_col]`` frame sorted ascending by ``time_col``
        and reset-indexed. ``None`` when nothing remains after the metric
        filter or the time window.
    """
    if actuals_df is None or actuals_df.empty:
        return None
    if metric_id_col in actuals_df.columns:
        slice_ = actuals_df[actuals_df[metric_id_col] == metric_id]
    else:
        slice_ = actuals_df
    if slice_.empty:
        return None
    slice_ = slice_[[time_col, actual_col]].copy()
    slice_[time_col] = pd.to_datetime(slice_[time_col])
    if actuals_window_days is not None and not forecast_df.empty:
        max_ts = pd.to_datetime(forecast_df[time_col]).max()
        slice_ = slice_[slice_[time_col] >= max_ts - pd.Timedelta(days=actuals_window_days)]
    return slice_.sort_values(time_col).reset_index(drop=True) if not slice_.empty else None


def _render_eval_html(
    *,
    report_title: str,
    report_summary: str,
    training_cutoffs: Sequence[Any],
    errors_per_metric: pd.DataFrame,
    errors_per_metric_horizon: pd.DataFrame,
    figure_sections: Sequence[FigureSection],
) -> str:
    """Compose the self-contained HTML report — no external CSS / JS files.

    Args:
        report_title: Rendered into the ``<h1>`` and ``<title>`` tags.
        report_summary: Optional HTML/markdown-style snippet rendered in a
            ``.note`` block below the title (e.g. caveats, source links).
            Empty string → block is omitted.
        training_cutoffs: Cutoff values, formatted as ``"YYYY-MM-DD"`` and
            shown in the metadata line. Empty → ``"no cutoffs"``.
        errors_per_metric: One row per metric_id; rendered as the headline
            accuracy table (all numeric columns ``%.4f``-formatted).
        errors_per_metric_horizon: One row per (metric_id, horizon_step);
            rendered as the per-horizon detail table.
        figure_sections: Ordered sections (title + description + per-metric
            figures) emitted by the registered figure makers. Plotly.js is
            loaded once via CDN on the very first figure across all
            sections; subsequent figures skip the script tag.

    Returns:
        Self-contained HTML string (DOCTYPE through ``</html>``). Safe to
        write directly to a ``.html`` file — no external assets.
    """
    cutoff_strs = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in training_cutoffs]
    cutoff_summary = (
        f"{len(cutoff_strs)} cutoffs ({cutoff_strs[0]} → {cutoff_strs[-1]})"
        if cutoff_strs else "no cutoffs"
    )

    per_metric_table = errors_per_metric.to_html(
        index=False, classes="data", float_format=lambda x: f"{x:.4f}",
    )
    per_horizon_table = errors_per_metric_horizon.to_html(
        index=False, classes="data", float_format=lambda x: f"{x:.4f}",
    )

    section_blocks: List[str] = []
    figure_counter = 0  # global so plotly.js is only included on the first
    for section_idx, section in enumerate(figure_sections, start=1):
        # Each section_idx is offset by the count of non-figure tables (2)
        # so the heading numbering starts at 3.
        heading_idx = section_idx + 2
        figure_blocks: List[str] = []
        for metric_id, fig in sorted(section.figures.items()):
            include_js = "cdn" if figure_counter == 0 else False
            figure_blocks.append(
                f"<h4>{metric_id}</h4>\n"
                + fig.to_html(
                    include_plotlyjs=include_js,
                    full_html=False,
                    div_id=f"fig_{section.name}_{figure_counter}",
                )
            )
            figure_counter += 1
        figures_html = "\n".join(figure_blocks) or "<p><em>No figures rendered.</em></p>"
        description_block = (
            f"<p class='section-desc'>{section.description}</p>"
            if section.description else ""
        )
        section_blocks.append(
            f"<h2>{heading_idx}. {section.title}</h2>\n"
            + description_block
            + "\n"
            + figures_html
        )

    sections_html = "\n".join(section_blocks)
    summary_block = f"<div class='note'>{report_summary}</div>" if report_summary else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{report_title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 1100px; margin: 2em auto; padding: 0 1.5em; line-height: 1.5; color: #24292e; }}
h1, h2, h3, h4 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
h4 {{ border-bottom: none; padding-bottom: 0; margin-top: 1.4em; color: #444; }}
table.data {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }}
table.data th, table.data td {{ border: 1px solid #ddd; padding: 0.4em 0.7em; text-align: right; }}
table.data th {{ background: #f6f8fa; }}
.meta {{ color: #6a737d; font-size: 0.9em; }}
.note {{ background: #fffbdd; border-left: 4px solid #ffe7a0; padding: 0.5em 1em; margin: 1em 0; }}
.section-desc {{ color: #444; font-size: 0.95em; margin: 0.4em 0 0.8em; }}
</style></head><body>

<h1>{report_title}</h1>
<p class="meta">Generated {datetime.now().isoformat(timespec='seconds')} • {cutoff_summary}.</p>
{summary_block}

<h2>1. Per-metric headline accuracy</h2>
{per_metric_table}

<h2>2. Per-(metric, horizon) accuracy</h2>
{per_horizon_table}

{sections_html}

</body></html>
"""

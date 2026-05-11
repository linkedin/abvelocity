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
"""Plotting utilities for time-series forecasting and anomaly detection.

Standalone Plotly helpers — no dependency on ``blah.greykite``.
``abvelocity.ts.common.viz.timeseries_plotting`` is the
vendored greykite copy used by ``gk/*`` code; the helpers here are the
clean abvelocity-canonical layer (different column names, simpler
config) and should be the entry point for any new abvelocity code.

Functions
---------
``plot_forecast_vs_actual``
    Plot a single-metric forecast DataFrame (wide format) with actuals,
    point forecast, and optional CI band.

``plot_forecast_groups_vs_actual``
    Overlay one forecast curve per value of any grouping column
    (horizon step, country, variant, cutoff bucket, …) on a single
    actuals series. Each group's CI band shares its line's color.

``plot_forecast_at_training_cutoffs``
    Specialised wrapper for the "spaghetti per training cutoff" view:
    continuous actuals + each training cutoff's next ``max_days_ahead``
    forecast as its own trajectory. Sugar over
    ``plot_forecast_groups_vs_actual`` plus a days-ahead filter.

``add_multi_vrects``
    Overlay shaded vertical rectangles onto an existing figure from a
    DataFrame of anomaly / event periods.

``plot_forecast``
    High-level wrapper around ``plot_forecast_vs_actual`` that accepts the
    long-format ``result_df`` produced by :class:`TSRunner` and returns one
    figure per metric (or a vertically-stacked subplot figure when
    ``subplots=True``).

``plot_breakdown``
    Plot the breakdown components (trend / weekly / yearly / holiday)
    overlaid on a single figure (or per-segment subplots when
    ``groupby`` is non-trivial).

``write_index_html``
    Self-contained HTML report writer — accepts a list of
    ``(title, description, [figures])`` sections and emits a deterministic
    ``index.html`` next to the rest of the test-results tree.

``deterministic_plotly_div_id``
    Stable div id derived from a figure's JSON content; used so a rerun
    against the same data produces a byte-identical report.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from abvelocity.core.utils.color_utils import get_distinct_colors
from abvelocity.ts.constants import (
    ACTUAL_COL,
    END_TS_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    FORECASTED_DATE_COL,
    LAST_TRAINING_DATE_COL,
    METRIC_ID_COL,
    STAGE_COL,
    START_TS_COL,
    TIME_COL,
)
from abvelocity.ts.forecast_transforms.column_classes import BREAKDOWN_COLS
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# plot_forecast_vs_actual
# Adapted from abvelocity.ts.common.viz.timeseries_plotting
# ─────────────────────────────────────────────────────────────────────────────


def plot_forecast_vs_actual(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    actual_col: str = ACTUAL_COL,
    forecast_col: str = FORECAST_COL,
    forecast_lower_col: Optional[str] = FORECAST_LOWER_COL,
    forecast_upper_col: Optional[str] = FORECAST_UPPER_COL,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    train_end_date: Optional[str] = None,
    title: Optional[str] = None,
    showlegend: bool = True,
    actual_mode: str = "lines+markers",
    actual_color: str = "rgba(250, 43, 20, 0.7)",
    actual_marker_size: float = 2.0,
    forecast_color: str = "rgba(0, 90, 181, 0.7)",
    forecast_dash: str = "solid",
    ci_band_color: str = "rgba(0, 90, 181, 0.15)",
    ci_boundary_color: str = "rgba(0, 90, 181, 0.5)",
    ci_boundary_width: float = 0.0,
    vline_color: str = "rgba(100, 100, 100, 0.9)",
    vline_width: float = 1.0,
    xaxis_hoverformat: str = "%a %Y-%m-%d",
) -> go.Figure:
    """Plot forecast with prediction intervals against actuals.

    Adapted from
    ``blah.greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual``.

    Args:
        df: DataFrame containing ``time_col``, ``actual_col``, ``forecast_col``,
            and optionally ``forecast_lower_col`` / ``forecast_upper_col``.
        time_col: Timestamp column name.
        actual_col: Actual-values column name.
        forecast_col: Point-forecast column name.
        forecast_lower_col: Lower CI bound column; set to ``None`` to omit CI band.
        forecast_upper_col: Upper CI bound column; set to ``None`` to omit CI band.
        xlabel: x-axis label. Defaults to ``time_col``.
        ylabel: y-axis label. Defaults to ``None``.
        train_end_date: Timestamp marking the train/forecast split. A vertical
            dashed line is drawn at this date when provided. Must be a value
            present in ``df[time_col]``.
        title: Plot title.
        showlegend: Whether to show the legend.
        actual_mode: Plotly scatter mode for actuals
            (``"lines"``, ``"markers"``, or ``"lines+markers"``).
        actual_color: Color for actual values.
        actual_marker_size: Marker size for actuals (used when mode includes
            ``"markers"``).
        forecast_color: Color for the forecast line.
        forecast_dash: Dash style for the forecast line.
        ci_band_color: Fill color for the prediction interval band.
        ci_boundary_color: Line color for the CI boundary traces.
        ci_boundary_width: Width of CI boundary lines (0 = hidden).
        vline_color: Color of the train-end vertical line.
        vline_width: Width of the train-end vertical line.
        xaxis_hoverformat: strftime format for the x-axis hover label.
            Defaults to ``"%a %Y-%m-%d"`` so hover reads like
            ``"Mon 2024-01-01"``.  Pass ``None`` for plotly's default.

    Returns:
        Plotly figure.
    """
    if title is None:
        title = "Forecast vs Actual"
    if xlabel is None:
        xlabel = time_col

    fill_kwargs = {"mode": "lines", "fillcolor": ci_band_color, "fill": "tonexty"}
    data = []

    # Lower CI bound (invisible boundary, used as fill baseline).
    if forecast_lower_col is not None and forecast_lower_col in df.columns:
        data.append(
            go.Scatter(
                name="Lower Bound",
                x=df[time_col],
                y=df[forecast_lower_col],
                mode="lines",
                line=dict(width=ci_boundary_width, color=ci_boundary_color),
                legendgroup="interval",
            )
        )

    # Upper CI bound (fills down to lower bound).
    if forecast_upper_col is not None and forecast_upper_col in df.columns:
        upper_fill = fill_kwargs if (forecast_lower_col is not None and forecast_lower_col in df.columns) else {}
        data.append(
            go.Scatter(
                name="Upper Bound",
                x=df[time_col],
                y=df[forecast_upper_col],
                line=dict(width=ci_boundary_width, color=ci_boundary_color),
                legendgroup="interval",
                **upper_fill,
            )
        )

    # Actual values.
    actual_params: dict = {}
    if "lines" in actual_mode:
        actual_params["line"] = dict(color=actual_color)
    if "markers" in actual_mode:
        actual_params["marker"] = dict(color=actual_color, size=actual_marker_size)
    data.append(
        go.Scatter(
            name="Actual",
            x=df[time_col],
            y=df[actual_col],
            mode=actual_mode,
            **actual_params,
        )
    )

    # Forecast line.
    data.append(
        go.Scatter(
            name="Forecast",
            x=df[time_col],
            y=df[forecast_col],
            line=dict(color=forecast_color, dash=forecast_dash),
        )
    )

    xaxis_kwargs = {"title": xlabel}
    if xaxis_hoverformat is not None:
        xaxis_kwargs["hoverformat"] = xaxis_hoverformat
    layout = go.Layout(
        xaxis=dict(**xaxis_kwargs),
        yaxis=dict(title=ylabel),
        title=title,
        title_x=0.5,
        showlegend=showlegend,
        legend={"traceorder": "reversed"},
    )
    fig = go.Figure(data=data, layout=layout)

    if train_end_date is not None:
        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=train_end_date,
                    y0=0,
                    x1=train_end_date,
                    y1=1,
                    line=dict(color=vline_color, width=vline_width),
                )
            ],
            annotations=[
                dict(
                    xref="x",
                    x=train_end_date,
                    yref="paper",
                    y=0.97,
                    text="Train End Date",
                    showarrow=True,
                    arrowhead=0,
                    ax=-60,
                    ay=0,
                )
            ],
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# plot_forecast_groups_vs_actual
# ─────────────────────────────────────────────────────────────────────────────


def plot_forecast_groups_vs_actual(
    df: pd.DataFrame,
    group_col: Union[str, Sequence[str]],
    group_values: Optional[Sequence] = None,
    time_col: str = TIME_COL,
    actual_col: str = ACTUAL_COL,
    forecast_col: str = FORECAST_COL,
    forecast_lower_col: Optional[str] = FORECAST_LOWER_COL,
    forecast_upper_col: Optional[str] = FORECAST_UPPER_COL,
    group_label_template: str = "{value}",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    showlegend: bool = True,
    actuals_df: Optional[pd.DataFrame] = None,
    actual_color: str = "rgba(50, 50, 50, 0.85)",
    actual_mode: str = "lines+markers",
    actual_marker_size: float = 3.0,
    group_line_opacity: float = 0.85,
    group_band_opacity: float = 0.12,
    group_colors: Optional[Sequence[str]] = None,
    xaxis_hoverformat: Optional[str] = "%a %Y-%m-%d",
) -> go.Figure:
    """Overlay one forecast curve per group value (single- or multi-dimensional) on a shared actuals series.

    For each value (or value-tuple) in ``group_values``, picks the rows
    where every grouping column equals the corresponding component,
    sorted by ``time_col``, and renders the forecast line plus an
    optional CI band — both in the same color so the band visually
    belongs to its line. Actuals are deduplicated across the group
    slices and drawn once on top in ``actual_color``.

    Single-dim and multi-dim grouping use the same call signature:

    * ``group_col="horizon_step", group_values=[1, 7, 14],
      group_label_template="horizon={value}d"`` — rolling H-day-ahead
      backtest comparison (single dimension).
    * ``group_col=("country", "surface"),
      group_values=[("us", "voyager"), ("us", "web")]`` — two-dim
      overlay; each ``group_values`` element is a tuple of length two.
    * ``group_col=("country", "surface"), group_values=None`` —
      auto-discover: every distinct ``(country, surface)`` combination
      present in ``df`` becomes its own curve.
    * ``group_col="horizon_step", group_values=None`` — auto-discover
      every distinct horizon present in ``df``.

    Legend labels are formatted via ``str.format`` and receive both a
    joined-string ``{value}`` plus per-column kwargs:

    * ``"{value}"`` → ``"us, voyager"`` (comma-joined; default).
    * ``"{country}/{surface}"`` → ``"us/voyager"`` (named columns).
    * ``"horizon={value}d"`` → ``"horizon=7d"`` (single-dim; ``{value}`` is the lone scalar).

    Actuals are deduplicated **by ``time_col`` only** — the function
    assumes every group shares the same actual on a given date (true
    for horizon / cutoff / model-variant overlays, where the same
    underlying series is being forecasted from different angles). For
    overlays where actuals genuinely differ per group (e.g. country),
    actuals would collapse to one arbitrary group's values; the
    intended workaround there is to call this function once per group
    or pre-aggregate to a shared actual.

    The caller assembles ``df`` however they like — Trino (e.g. via
    :class:`~abvelocity.ts.li.data.li_cursor.LiCursor`),
    CloudNotebook's ``%%sql`` magic, a pre-baked parquet, etc. — this function
    only consumes a DataFrame.

    Args:
        df: Long-format DataFrame with one row per (group combination,
            forecasted timestamp). Must contain ``time_col``,
            ``actual_col``, ``forecast_col``, every column in
            ``group_col``, and optionally ``forecast_lower_col`` /
            ``forecast_upper_col``.
        group_col: Column name (single-dim) or sequence of column names
            (multi-dim) whose distinct combinations define the overlaid
            curves.
        group_values: The combinations to overlay. ``None`` →
            auto-discover every distinct combination present in
            ``df[group_col]``. For single-dim ``group_col``, each
            element is a scalar (e.g. ``[1, 7, 14]``). For multi-dim,
            each element must be a tuple of the same length as
            ``group_col`` (e.g. ``[("us", "voyager"), ("us", "web")]``).
        time_col: Timestamp column (the forecasted date).
        actual_col: Actual-value column.
        forecast_col: Point-forecast column.
        forecast_lower_col: Lower CI bound; ``None`` to omit CI bands.
        forecast_upper_col: Upper CI bound; ``None`` to omit CI bands.
        group_label_template: ``str.format`` template for legend
            labels. Available kwargs are ``{value}`` (comma-joined for
            multi-dim) plus one kwarg per grouping column. Defaults to
            ``"{value}"``.
        xlabel: x-axis label. Defaults to ``time_col``.
        ylabel: y-axis label.
        title: Plot title.
        showlegend: Whether to render the legend.
        actuals_df: Optional separate DataFrame to source the actuals
            trace from. Must contain ``time_col`` and ``actual_col``.
            Use this when ``df`` covers only forecast-window rows of
            sparse cutoffs (so deduping from ``df`` would skip the
            calendar dates between cutoff windows). When ``None``,
            actuals are simply deduped by ``time_col`` from ``df``
            (``keep="first"``) — fine for dense-cutoff panels. Either
            way, ``NaN`` actuals are kept and plotly breaks the line
            at them; if you want a genuinely-missing day to show as
            a gap, the caller should insert a ``NaN`` row at that
            date (e.g. via reindex to a full date range). For reads
            against the prod OH forecast table,
            ``eval/forecast_table_io.fetch_actuals_from_forecast_table``
            returns clean per-date actuals with cross-partition
            consistency warnings — pipe its output here.
        actual_color: Color for the actuals trace.
        actual_mode: Plotly scatter mode for actuals.
        actual_marker_size: Marker size for actuals (used when mode
            includes ``"markers"``).
        group_line_opacity: Opacity for the forecast lines (0..1).
        group_band_opacity: Opacity for the CI bands (0..1). Lower
            than the line opacity so overlapping bands stay readable.
        group_colors: Optional explicit color per group combination
            (rgba strings). When ``None``, distinct colors come from
            :func:`get_distinct_colors` and the band uses the same
            base color at ``group_band_opacity``. When provided, the
            same color is used for both line and band — the caller
            controls any soft-band effect.
        xaxis_hoverformat: strftime format for x-axis hover; default
            ``"%a %Y-%m-%d"`` so hover reads ``"Mon 2024-01-01"`` —
            day-of-week makes weekly seasonality patterns obvious.
            Pass ``None`` for plotly's default.

    Returns:
        Plotly figure with one CI band + forecast line per group
        combination and a single actuals trace overlaid.

    Raises:
        ValueError: if auto-discovery yields no combinations; if a
            combination is not present in ``df``; if any required
            column is missing from ``df``; if a multi-dim
            ``group_values`` element has the wrong tuple length; or if
            ``group_colors`` length does not match the resolved
            combinations.
    """
    group_cols: Tuple[str, ...] = (group_col,) if isinstance(group_col, str) else tuple(group_col)
    if not group_cols:
        raise ValueError("group_col must name at least one column.")

    # ``actual_col`` is required in ``df`` only when ``actuals_df`` isn't
    # provided — otherwise the actuals trace comes entirely from the
    # external frame and ``df`` only needs the forecast / group columns.
    required = set(group_cols) | {time_col, forecast_col}
    if actuals_df is None:
        required.add(actual_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}.")

    # Resolve combinations: auto-discover from df, or normalize the
    # caller's list. Internal representation is always a list of tuples
    # of length len(group_cols), regardless of dim.
    combinations: List[Tuple] = []
    if group_values is None:
        discovered = (
            df[list(group_cols)]
            .dropna()
            .drop_duplicates()
            .sort_values(by=list(group_cols))
            .itertuples(index=False, name=None)
        )
        combinations = [tuple(combo) for combo in discovered]
        if not combinations:
            raise ValueError(
                f"Auto-discovery found no distinct combinations of {list(group_cols)} in df."
            )
    else:
        if not group_values:
            raise ValueError("group_values is empty — pass None to auto-discover, or a non-empty sequence.")
        for raw in group_values:
            if len(group_cols) == 1:
                combinations.append((raw,) if not isinstance(raw, tuple) else raw)
            else:
                if not isinstance(raw, tuple) or len(raw) != len(group_cols):
                    raise ValueError(
                        f"For multi-dim group_col {list(group_cols)}, each "
                        f"group_values element must be a tuple of length "
                        f"{len(group_cols)}; got {raw!r}."
                    )
                combinations.append(raw)

    # Validate every combination exists in df.
    for combo in combinations:
        mask = pd.Series(True, index=df.index)
        for col, val in zip(group_cols, combo):
            mask &= (df[col] == val)
        if not mask.any():
            raise ValueError(
                f"Combination {combo} for columns {list(group_cols)} "
                f"not present in df."
            )

    if group_colors is None:
        line_colors = get_distinct_colors(num_colors=len(combinations), opacity=group_line_opacity)
        band_colors = get_distinct_colors(num_colors=len(combinations), opacity=group_band_opacity)
    else:
        if len(group_colors) != len(combinations):
            raise ValueError(
                f"group_colors length ({len(group_colors)}) does not match "
                f"the resolved {len(combinations)} group combinations."
            )
        line_colors = list(group_colors)
        band_colors = list(group_colors)

    has_ci = (
        forecast_lower_col is not None and forecast_lower_col in df.columns
        and forecast_upper_col is not None and forecast_upper_col in df.columns
    )

    data: List[go.Scatter] = []
    for combo, line_color, band_color in zip(combinations, line_colors, band_colors):
        mask = pd.Series(True, index=df.index)
        for col, val in zip(group_cols, combo):
            mask &= (df[col] == val)
        slice_df = df[mask].sort_values(by=time_col)

        joined_value = ", ".join(str(component) for component in combo)
        legend_group = f"{'+'.join(group_cols)}={joined_value}"
        format_kwargs = {"value": joined_value}
        format_kwargs.update(dict(zip(group_cols, combo)))
        legend_label = group_label_template.format(**format_kwargs)

        if has_ci:
            # Lower bound: invisible boundary used as fill baseline; not in legend.
            data.append(
                go.Scatter(
                    name=f"{legend_label} lower",
                    x=slice_df[time_col],
                    y=slice_df[forecast_lower_col],
                    mode="lines",
                    line=dict(width=0, color=band_color),
                    legendgroup=legend_group,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Upper bound: fills down to lower, in the same color as the line.
            data.append(
                go.Scatter(
                    name=f"{legend_label} upper",
                    x=slice_df[time_col],
                    y=slice_df[forecast_upper_col],
                    mode="lines",
                    line=dict(width=0, color=band_color),
                    fill="tonexty",
                    fillcolor=band_color,
                    legendgroup=legend_group,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        data.append(
            go.Scatter(
                name=legend_label,
                x=slice_df[time_col],
                y=slice_df[forecast_col],
                mode="lines",
                line=dict(color=line_color),
                legendgroup=legend_group,
            )
        )

    # Actuals trace: dumb — render whatever the caller hands in, after a
    # simple dedupe on ``time_col``. NaN values are preserved so plotly's
    # default ``connectgaps=False`` breaks the line at genuinely-missing
    # dates (caller's responsibility to insert those NaN rows when they
    # want a gap visualised — e.g. via ``df.reindex(full_date_range)``).
    # No conflict resolution between duplicates, no non-NaN preference —
    # if the caller has dirty data they should clean it upstream (for OH
    # table reads, see ``fetch_actuals_from_forecast_table`` which
    # already returns one deduped row per (metric, date) with cross-
    # partition CV warnings).
    actuals_source = actuals_df if actuals_df is not None else df
    for required_col in (time_col, actual_col):
        if required_col not in actuals_source.columns:
            raise ValueError(
                f"actuals source is missing required column {required_col!r}."
            )
    actual_df = (
        actuals_source[[time_col, actual_col]]
        .drop_duplicates(subset=[time_col], keep="first")
        .sort_values(by=time_col)
    )
    actual_kwargs: dict = {}
    if "lines" in actual_mode:
        actual_kwargs["line"] = dict(color=actual_color)
    if "markers" in actual_mode:
        actual_kwargs["marker"] = dict(color=actual_color, size=actual_marker_size)
    data.append(
        go.Scatter(
            name="Actual",
            x=actual_df[time_col],
            y=actual_df[actual_col],
            mode=actual_mode,
            **actual_kwargs,
        )
    )

    if title is None:
        title = f"Forecast vs Actual by {' + '.join(group_cols)}"
    if xlabel is None:
        xlabel = time_col

    xaxis_kwargs: dict = {"title": xlabel}
    if xaxis_hoverformat is not None:
        xaxis_kwargs["hoverformat"] = xaxis_hoverformat
    layout = go.Layout(
        xaxis=dict(**xaxis_kwargs),
        yaxis=dict(title=ylabel),
        title=title,
        title_x=0.5,
        showlegend=showlegend,
        legend={"traceorder": "reversed"},
    )
    return go.Figure(data=data, layout=layout)


# ─────────────────────────────────────────────────────────────────────────────
# plot_forecast_at_training_cutoffs
# ─────────────────────────────────────────────────────────────────────────────


def plot_forecast_at_training_cutoffs(
    df: pd.DataFrame,
    training_cutoffs: Optional[Sequence] = None,
    max_days_ahead: int = 7,
    training_cutoff_col: str = LAST_TRAINING_DATE_COL,
    time_col: str = TIME_COL,
    actual_col: str = ACTUAL_COL,
    forecast_col: str = FORECAST_COL,
    forecast_lower_col: Optional[str] = FORECAST_LOWER_COL,
    forecast_upper_col: Optional[str] = FORECAST_UPPER_COL,
    actuals_df: Optional[pd.DataFrame] = None,
    training_cutoff_label_template: str = "cutoff {value}",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    showlegend: bool = True,
    actual_color: str = "rgba(50, 50, 50, 0.85)",
    actual_mode: str = "lines+markers",
    actual_marker_size: float = 3.0,
    training_cutoff_line_opacity: float = 0.85,
    training_cutoff_band_opacity: float = 0.12,
    training_cutoff_colors: Optional[Sequence[str]] = None,
    xaxis_hoverformat: Optional[str] = "%a %Y-%m-%d",
) -> go.Figure:
    """Overlay each training cutoff's next ``max_days_ahead`` forecast on a continuous actuals line.

    The "actuals continuous + per-training-cutoff short trajectories" view:
    the actuals line spans the whole observed range, and each entry in
    ``training_cutoffs`` (or every distinct training cutoff in ``df``
    when ``None``) draws one short forecast curve covering that cutoff's
    day +1 through day + ``max_days_ahead``. Each trajectory is one
    color; CI band per cutoff shares its line's color when CI columns
    are present. Standard "spaghetti vs actual" forecast plot.

    Internally a thin wrapper around
    :func:`plot_forecast_groups_vs_actual` with
    ``group_col=training_cutoff_col``, after filtering ``df`` to rows
    whose ``time_col − training_cutoff_col`` lies in
    ``[1, max_days_ahead]`` days.

    Args:
        df: Long-format DataFrame with one row per (training cutoff,
            forecasted timestamp). Must contain ``training_cutoff_col``,
            ``time_col``, ``forecast_col``, optionally ``actual_col``
            and CI columns.
        training_cutoffs: Training-cutoff values to plot. ``None``
            (default) plots every distinct training cutoff present in
            ``df`` after the days-ahead filter.
        max_days_ahead: Per-cutoff trajectory length. Each cutoff draws
            a curve from day +1 to day + ``max_days_ahead``.
        training_cutoff_col: Cutoff column name. Defaults to
            ``LAST_TRAINING_DATE_COL`` (``"last_training_date"``).
        time_col: Forecasted-date column.
        actual_col: Actual-value column.
        forecast_col: Point-forecast column.
        forecast_lower_col: Lower CI bound; ``None`` to omit CI bands.
        forecast_upper_col: Upper CI bound; ``None`` to omit CI bands.
        actuals_df: Optional separate DataFrame for the actuals trace
            (must have ``time_col`` + ``actual_col``). Useful when ``df``
            only covers forecast-window dates and you want a continuous
            actuals line spanning more history. Falls back to deduping
            from ``df`` when ``None``.
        training_cutoff_label_template: ``str.format`` template for
            legend labels. Defaults to ``"cutoff {value}"``.
        title: Plot title.
        xlabel: x-axis label. Defaults to ``time_col``.
        ylabel: y-axis label.
        showlegend: Whether to render the legend.
        actual_color: Color for the actuals trace.
        actual_mode: Plotly scatter mode for actuals.
        actual_marker_size: Marker size for actuals.
        training_cutoff_line_opacity: Opacity for per-cutoff forecast lines.
        training_cutoff_band_opacity: Opacity for per-cutoff CI bands.
        training_cutoff_colors: Optional explicit color list, one per cutoff.
        xaxis_hoverformat: strftime format for x-axis hover.

    Returns:
        Plotly figure with one CI band + forecast line per training cutoff
        and a single actuals trace overlaid.

    Raises:
        ValueError: if no rows survive the days-ahead filter; if
            ``training_cutoffs`` contains values not present in ``df``;
            or any of the validation cases inherited from
            :func:`plot_forecast_groups_vs_actual`.
    """
    if training_cutoff_col not in df.columns:
        raise ValueError(
            f"df is missing required column {training_cutoff_col!r} "
            "(the training-cutoff column)."
        )

    days_ahead = (
        pd.to_datetime(df[time_col]) - pd.to_datetime(df[training_cutoff_col])
    ).dt.days
    mask = (days_ahead >= 1) & (days_ahead <= max_days_ahead)
    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows survived the days-ahead filter "
            f"(1 <= {time_col} - {training_cutoff_col} <= {max_days_ahead}). "
            f"Check that training-cutoff and time columns are populated "
            f"and that max_days_ahead matches the data."
        )

    if training_cutoffs is None:
        training_cutoffs = sorted(filtered[training_cutoff_col].unique())

    if title is None:
        title = (
            f"Forecast vs Actual — {max_days_ahead}-day trajectory per "
            f"training cutoff"
        )

    return plot_forecast_groups_vs_actual(
        df=filtered,
        group_col=training_cutoff_col,
        group_values=training_cutoffs,
        time_col=time_col,
        actual_col=actual_col,
        forecast_col=forecast_col,
        forecast_lower_col=forecast_lower_col,
        forecast_upper_col=forecast_upper_col,
        group_label_template=training_cutoff_label_template,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        showlegend=showlegend,
        actuals_df=actuals_df,
        actual_color=actual_color,
        actual_mode=actual_mode,
        actual_marker_size=actual_marker_size,
        group_line_opacity=training_cutoff_line_opacity,
        group_band_opacity=training_cutoff_band_opacity,
        group_colors=training_cutoff_colors,
        xaxis_hoverformat=xaxis_hoverformat,
    )


# ─────────────────────────────────────────────────────────────────────────────
# add_multi_vrects
# Adapted from abvelocity.ts.common.viz.timeseries_annotate
# ─────────────────────────────────────────────────────────────────────────────


def add_multi_vrects(
    fig: go.Figure,
    periods_df: pd.DataFrame,
    grouping_col: Optional[str] = None,
    start_time_col: str = START_TS_COL,
    end_time_col: str = END_TS_COL,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    annotation_text_col: Optional[str] = None,
    annotation_position: str = "top left",
    opacity: float = 0.5,
    grouping_color_dict: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[go.Figure, Dict[str, str]]]:
    """Overlay shaded vertical rectangles onto an existing figure.

    Adapted from
    ``blah.greykite.common.viz.timeseries_annotate.add_multi_vrects``.

    Each row in ``periods_df`` defines one rectangle spanning
    ``[start_time_col, end_time_col]``.  Colors are grouped by
    ``grouping_col`` when provided.

    Args:
        fig: Existing Plotly figure to augment.
        periods_df: DataFrame with at least ``start_time_col`` and
            ``end_time_col`` columns.
        grouping_col: Column used to assign the same color to all
            rectangles in a group. When ``None``, all rectangles get a
            single color.
        start_time_col: Column name for period start.
        end_time_col: Column name for period end.
        y_min: Lower y limit of rectangles (``None`` = full height).
        y_max: Upper y limit of rectangles (``None`` = full height).
        annotation_text_col: Column whose values are used as rectangle
            annotations. ``None`` = no annotations.
        annotation_position: Annotation position string (Plotly convention).
        opacity: Fill opacity of the rectangles.
        grouping_color_dict: Pre-specified group → color mapping.
            When ``None``, colors are generated automatically.

    Returns:
        ``{"fig": <updated figure>, "grouping_color_dict": <dict>}``
    """
    periods_df = periods_df.copy()

    if start_time_col not in periods_df.columns:
        raise ValueError(f"start_time_col {start_time_col!r} not found in periods_df columns: " f"{list(periods_df.columns)}")
    if end_time_col not in periods_df.columns:
        raise ValueError(f"end_time_col {end_time_col!r} not found in periods_df columns: " f"{list(periods_df.columns)}")

    if grouping_col is None:
        grouping_col = METRIC_ID_COL
        periods_df[METRIC_ID_COL] = METRIC_ID_COL

    if grouping_col not in periods_df.columns:
        raise ValueError(f"grouping_col {grouping_col!r} not found in periods_df columns: " f"{list(periods_df.columns)}")
    if annotation_text_col is not None and annotation_text_col not in periods_df.columns:
        raise ValueError(f"annotation_text_col {annotation_text_col!r} not found in periods_df columns: " f"{list(periods_df.columns)}")

    groups = sorted(set(periods_df[grouping_col]))
    if grouping_color_dict is None:
        colors = get_distinct_colors(len(groups), opacity=1.0)
        grouping_color_dict = {g: colors[i] for i, g in enumerate(groups)}

    for group in groups:
        mask = periods_df[grouping_col] == group
        fillcolor = grouping_color_dict[group]
        for _, row in periods_df.loc[mask].iterrows():
            x0, x1 = row[start_time_col], row[end_time_col]
            annotation_text = row[annotation_text_col] if annotation_text_col else ""
            fig.add_vrect(
                x0=x0,
                x1=x1,
                y0=y_min,
                y1=y_max,
                fillcolor=fillcolor,
                opacity=opacity,
                line_width=2,
                line_color=fillcolor,
                layer="below",
                annotation_text=annotation_text,
                annotation_position=annotation_position,
            )

    return {"fig": fig, "grouping_color_dict": grouping_color_dict}


# ─────────────────────────────────────────────────────────────────────────────
# plot_forecast  — high-level wrapper for long-format result_df
# ─────────────────────────────────────────────────────────────────────────────


def plot_forecast(
    result_df: pd.DataFrame,
    time_col: str = TIME_COL,
    train_end_date: Optional[str] = None,
    title: Optional[str] = None,
    groupby: Tuple[str, ...] = (METRIC_ID_COL,),
    subplots: bool = True,
    height_per_metric: int = 350,
    **kwargs,
) -> go.Figure:
    """Plot a long-format forecast ``result_df`` returned by :class:`TSRunner`.

    Iterates over each unique combo of ``groupby`` values and calls
    :func:`plot_forecast_vs_actual` for each.  When ``subplots=True``
    (default) all combos share the x-axis in a stacked subplot figure.
    When ``subplots=False`` only the **first** combo is plotted.

    Args:
        result_df: Long-format DataFrame with the canonical abvelocity
            forecast schema.
        time_col: Timestamp column name (default ``TIME_COL = "ts"``).
        train_end_date: Optional train/forecast split timestamp; drawn as
            a vertical line on each subplot.  When ``None`` and the frame
            carries a ``stage`` column, it's auto-inferred via
            :func:`infer_train_end` (last fitted-stage date).
        title: Overall figure title. Defaults to ``"Forecast"`` when
            ``subplots=True``, or to the matched group label when
            ``subplots=False``.
        groupby: Columns whose unique combos define one panel each.
            Defaults to ``(METRIC_ID_COL,)`` — one subplot per metric,
            the canonical TSRunner output.  Pass any other tuple
            (e.g. ``("country", "device")``) to split a panel frame
            into per-segment subplots; the function synthesizes a
            transient ``metric_id`` from those columns so the loop logic
            stays the same.
        subplots: When ``True`` (default), each ``groupby`` combo gets
            its own subplot row.  When ``False``, only the first combo
            is plotted as a standalone figure.
        height_per_metric: Pixel height per subplot row.
        **kwargs: Extra keyword arguments forwarded to
            :func:`plot_forecast_vs_actual` (e.g. ``actual_color``,
            ``forecast_dash``).

    Returns:
        Plotly figure.
    """
    # Post-transform frames shed the original "ts" timestamp column
    # (transforms operate on the DATE projection ``forecasted_date``).
    # Fall back when the requested time_col isn't present so callers
    # don't have to pass it explicitly on every call.
    if time_col not in result_df.columns and FORECASTED_DATE_COL in result_df.columns:
        time_col = FORECASTED_DATE_COL

    if train_end_date is None:
        inferred = infer_train_end(result_df=result_df)
        if inferred is not None:
            train_end_date = str(inferred.date())

    if list(groupby) != [METRIC_ID_COL]:
        # Synthesize a per-combo metric_id so the loop below treats each
        # segment combo as its own panel — caller's original metric_id
        # column is overwritten in this local copy only.
        result_df = result_df.copy()
        result_df[METRIC_ID_COL] = result_df[list(groupby)].astype(str).agg(" | ".join, axis=1)

    metrics: List[str] = list(result_df[METRIC_ID_COL].unique())
    if not metrics:
        raise ValueError("result_df contains no rows — nothing to plot.")

    if not subplots or len(metrics) == 1:
        metric = metrics[0]
        df_m = result_df[result_df[METRIC_ID_COL] == metric].reset_index(drop=True)
        return plot_forecast_vs_actual(
            df=df_m,
            time_col=time_col,
            train_end_date=train_end_date,
            title=title or metric,
            **kwargs,
        )

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=True,
        subplot_titles=metrics,
        vertical_spacing=0.08,
    )

    for i, metric in enumerate(metrics, start=1):
        df_m = result_df[result_df[METRIC_ID_COL] == metric].reset_index(drop=True)
        sub_fig = plot_forecast_vs_actual(
            df=df_m,
            time_col=time_col,
            train_end_date=None,  # handled below via add_vline per row
            title=None,
            **kwargs,
        )
        # Only show legend entries for the first subplot; set showlegend on
        # each trace directly because fig.add_trace ignores the sub-figure's
        # layout-level showlegend setting.
        for trace in sub_fig.data:
            trace.showlegend = i == 1
            fig.add_trace(trace, row=i, col=1)

    if train_end_date is not None:
        fig.add_vline(
            x=str(pd.Timestamp(train_end_date).date()),
            line=dict(color="rgba(100,100,100,0.9)", width=1),
        )

    fig.update_layout(
        title_text=title or "Forecast",
        title_x=0.5,
        hovermode="x unified",
        height=height_per_metric * len(metrics),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers operating on the canonical abvelocity forecast frame
# (``forecasted_date`` time col, stage-aware cutoff inference,
# breakdown columns).  ``plot_forecast`` above takes a generic
# wide-format frame; the helpers here build on it.
# ─────────────────────────────────────────────────────────────────────────────


def deterministic_plotly_div_id(figure: go.Figure) -> str:
    """Stable div id derived from the figure's JSON content.

    Same content → same div id → byte-identical HTML across runs.  The
    canonical copy lives in
    ``abvelocity.core.utils.print_to_html``; inlined here so
    studies don't depend on a freshly-rebuilt abvelocity wheel.
    """
    return "fig_" + hashlib.sha256(figure.to_json().encode(encoding="utf-8")).hexdigest()[:8]


def infer_train_end(result_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Return the last fitted-stage date, or ``None`` if none present."""
    if STAGE_COL not in result_df.columns:
        return None
    fitted = result_df.loc[result_df[STAGE_COL] == "fitted", FORECASTED_DATE_COL]
    if fitted.empty:
        return None
    return pd.Timestamp(fitted.max())


def plot_breakdown(
    result_df: pd.DataFrame,
    title: str,
    groupby: Tuple[str, ...] = (),
    train_end_date: Optional[pd.Timestamp] = None,
) -> Optional[go.Figure]:
    """All breakdown components overlaid on one plot — matches greykite's
    native ``plot_components`` style.

    For multi-segment frames (``groupby`` non-empty), each segment
    gets its own subplot row; within each row all components are
    overlaid as separate traces.

    Args:
        result_df: Frame in canonical abvelocity forecast schema —
            breakdown columns from
            :data:`~abvelocity.ts.forecast_transforms.column_classes.BREAKDOWN_COLS`
            are the components that get plotted.
        title: Figure title.
        groupby: Dim columns whose levels split into per-segment
            subplot rows.  Empty → one figure with all components
            overlaid.
        train_end_date: Optional explicit cutoff drawn as a gray vline.
            Inferred from ``stage`` when ``None``.

    Returns:
        Plotly figure, or ``None`` when the frame has no breakdown
        columns to plot.
    """
    component_cols = [col for col in BREAKDOWN_COLS if col in result_df.columns]
    if not component_cols:
        return None

    df_for_plot = result_df.sort_values(by=FORECASTED_DATE_COL).copy()
    if train_end_date is None:
        train_end_date = infer_train_end(result_df=df_for_plot)
    train_end_str = str(pd.Timestamp(train_end_date).date()) if train_end_date is not None else None

    # Plotly's default color cycler is per-trace order, which drifts
    # across segment rows; pin one color per component name so the same
    # component shows the same color in every subplot.
    palette = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    component_color = {col: palette[i % len(palette)] for i, col in enumerate(iterable=component_cols)}

    if not groupby:
        figure = go.Figure()
        for component_col in component_cols:
            figure.add_trace(
                trace=go.Scatter(
                    x=df_for_plot[FORECASTED_DATE_COL],
                    y=df_for_plot[component_col],
                    name=component_col,
                    mode="lines",
                    line=dict(color=component_color[component_col]),
                ),
            )
        if train_end_str is not None:
            figure.add_vline(
                x=train_end_str,
                line=dict(color="rgba(100,100,100,0.9)", width=1),
            )
        figure.update_layout(
            title_text=title,
            title_x=0.5,
            height=500,
            hovermode="x unified",
            xaxis=dict(hoverformat="%a %Y-%m-%d"),
        )
        return figure

    segment_groups = list(df_for_plot.groupby(by=list(groupby), dropna=False))
    figure = make_subplots(
        rows=len(segment_groups),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[" | ".join(map(str, key)) if isinstance(key, tuple) else str(key) for key, _ in segment_groups],
        vertical_spacing=0.06,
    )
    for row_idx, (_, segment_df) in enumerate(iterable=segment_groups, start=1):
        ordered = segment_df.sort_values(by=FORECASTED_DATE_COL)
        for component_col in component_cols:
            figure.add_trace(
                trace=go.Scatter(
                    x=ordered[FORECASTED_DATE_COL],
                    y=ordered[component_col],
                    name=component_col,
                    legendgroup=component_col,
                    showlegend=(row_idx == 1),
                    mode="lines",
                    line=dict(color=component_color[component_col]),
                ),
                row=row_idx,
                col=1,
            )

    if train_end_str is not None:
        figure.add_vline(
            x=train_end_str,
            line=dict(color="rgba(100,100,100,0.9)", width=1),
        )

    # Apply day-of-week hover format to every per-segment subplot's
    # x-axis (xaxis, xaxis2, xaxis3, ...).
    for axis_idx in range(1, len(segment_groups) + 1):
        axis_key = "xaxis" if axis_idx == 1 else f"xaxis{axis_idx}"
        figure.update_layout({axis_key: dict(hoverformat="%a %Y-%m-%d")})

    figure.update_layout(
        title_text=title,
        title_x=0.5,
        height=300 * len(segment_groups),
        hovermode="x unified",
    )
    return figure


def make_forecast_plots_section(
    forecast_df: pd.DataFrame,
    title: str,
    caption: str,
) -> Tuple[str, str, List[go.Figure]]:
    """Build one ``write_index_html`` section pairing a forecast plot
    with its breakdown plot.

    The standard "study/script" layout is one section per scenario,
    each showing the forecast curve + the breakdown components stacked
    underneath.  This helper returns the
    ``(title, caption, [forecast_fig, breakdown_fig])`` tuple
    :func:`write_index_html` expects, dropping the breakdown figure
    when the frame has no breakdown columns to plot (so non-additive
    fits don't end up with empty subfigure boxes).

    Args:
        forecast_df: Frame in canonical abvelocity forecast schema —
            same shape :func:`plot_forecast` and :func:`plot_breakdown`
            both accept.
        title: Section title.  Used for both figure suffixes and the
            ``<h2>`` heading on the rendered HTML.
        caption: Free-form description placed under the section title;
            ``None`` allowed for "no caption" but here we keep it
            non-optional since every existing caller supplies one.

    Returns:
        ``(title, caption, [figs])`` ready to drop into the
        ``sections`` arg of :func:`write_index_html`.
    """
    figures = [
        plot_forecast(result_df=forecast_df, title=f"{title} — forecast"),
        plot_breakdown(result_df=forecast_df, title=f"{title} — breakdown components"),
    ]
    return (title, caption, [fig for fig in figures if fig is not None])


def make_input_and_transform_plots_sections(
    input_df: pd.DataFrame,
    input_title: str,
    input_caption: str,
    transform_df: pd.DataFrame,
    transform_title: str,
    transform_caption: str,
) -> List[Tuple[str, str, List[go.Figure]]]:
    """Two-section block: a raw fit (input) followed by one transform.

    Convenience for the typical study/script layout where every
    transform's report opens with the input section then shows the
    transform's output side-by-side.  Both sections are built via
    :func:`make_forecast_plots_section`.

    Args:
        input_df: The pre-transform forecast frame (input).
        input_title: Section title for the input — caller picks the
            wording (e.g. ``"Daily Silverkite fit (input)"`` for a
            Silverkite fit, or ``"Hourly raw fit"`` for something else).
        input_caption: Caption for the input section.
        transform_df: The post-transform forecast frame (output).
        transform_title: Section title for the transform output.
        transform_caption: Caption for the transform section.

    Returns:
        Two-element list of section tuples ready for the
        ``sections`` arg of :func:`write_index_html`.
    """
    return [
        make_forecast_plots_section(forecast_df=input_df, title=input_title, caption=input_caption),
        make_forecast_plots_section(forecast_df=transform_df, title=transform_title, caption=transform_caption),
    ]


def write_index_html(
    output_dir: Path,
    study_name: str,
    sections: List[Tuple[str, Optional[str], List[go.Figure]]],
) -> Path:
    """Write a self-contained HTML report containing all figures.

    Each section becomes an ``<h2>`` block: title, optional description
    paragraph, then the section's figures inline.  Plotly's CDN is
    pulled once at the top of the page so figures only carry their own
    div ids.

    Args:
        output_dir: Directory to write into (created if missing).
        study_name: Used in the page title.
        sections: List of ``(section_title, description, figures)``.
            ``description`` may be ``None``.

    Returns:
        Absolute path of the written ``index.html``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "index.html"

    parts: List[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{study_name}</title>",
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        "</head>",
        "<body>",
        f"<h1>{study_name}</h1>",
    ]

    for section_title, description, figures in sections:
        parts.append(f"<h2>{section_title}</h2>")
        if description:
            parts.append(f"<p>{description}</p>")
        for figure in figures:
            parts.append(
                figure.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    div_id=deterministic_plotly_div_id(figure=figure),
                )
            )

    parts.append("</body></html>")
    out_path.write_text(data="\n".join(parts))
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Event / holiday annotations
# ─────────────────────────────────────────────────────────────────────────────


DEFAULT_EVENT_MARKER_PALETTE: Tuple[str, ...] = (
    "#FFD700",  # gold
    "#FF00FF",  # magenta
    "#00CED1",  # dark turquoise
    "#39FF14",  # neon green
    "#FF6F00",  # deep orange
    "#8B0000",  # dark red
    "#000080",  # navy
    "#FFB6C1",  # light pink
)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Render ``#RRGGBB`` + alpha as a Plotly-compatible ``rgba(r,g,b,a)`` string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def add_event_traces(
    fig: go.Figure,
    events_df: pd.DataFrame,
    *,
    date_col: str = "date",
    name_col: str = "name",
    group_col: str = "country",
    y_col: str = "y",
    palette: Sequence[str] = DEFAULT_EVENT_MARKER_PALETTE,
    marker_symbol: str = "star",
    marker_size: int = 12,
    legend_prefix: str = "event",
) -> None:
    """Overlay event annotations on a time-series figure — one toggleable trace per group.

    Generic event-marker overlay: each row in ``events_df`` becomes a
    point at ``(date_col, y_col)``, colored by ``group_col``,
    rendered with the chosen marker symbol. Hover shows weekday +
    date + event name + group + y-value.

    The caller is responsible for computing y-values per event row
    (e.g. by looking the date up in an actuals or forecast series).
    Keeps this helper pure-viz: it doesn't peek at any modeling
    object and plays well with any time-series chart.

    Args:
        fig: Figure to mutate in-place.
        events_df: DataFrame with at least ``date_col``, ``name_col``,
            ``group_col``, ``y_col``. Rows with NaN in ``y_col`` are
            dropped (they have no meaningful y-position).
        date_col: Name of the datetime column. Default ``"date"``.
        name_col: Name of the human-readable label column.
            Default ``"name"``.
        group_col: Column whose distinct values become separate
            legend-toggleable traces (e.g. ``"country"`` for
            holidays). Default ``"country"``.
        y_col: Column carrying the y-position for each marker. The
            caller fills this from whatever series makes sense
            (actuals on observed days, forecast on future days, etc.).
            Default ``"y"``.
        palette: Per-group marker colors, recycled if there are more
            groups than colors. Defaults to
            :data:`DEFAULT_EVENT_MARKER_PALETTE` (vivid hues chosen
            to stand out from typical line palettes).
        marker_symbol: Plotly marker symbol — defaults to ``"star"``.
        marker_size: Marker pixel size.
        legend_prefix: Trace-name prefix; the legend entry is
            ``f"{legend_prefix} ({group_value})"``. Default
            ``"event"``.
    """
    if events_df.empty:
        return
    sub = events_df.dropna(subset=[y_col]).copy()
    if sub.empty:
        return

    hover = (
        "<b>%{x|%A %Y-%m-%d}</b><br>"
        "%{customdata[0]} (%{customdata[1]})<br>"
        "value: %{y:,.0f}"
        f"<extra>{legend_prefix} — %{{fullData.name}}</extra>"
    )
    for i, (group_value, group_df) in enumerate(iterable=sub.groupby(group_col)):
        group_df = group_df.sort_values(by=date_col)
        custom = group_df[[name_col, group_col]].to_numpy()
        fig.add_trace(go.Scatter(
            x=group_df[date_col],
            y=group_df[y_col],
            customdata=custom,
            mode="markers",
            name=f"{legend_prefix} ({group_value})",
            marker=dict(
                color=palette[i % len(palette)],
                size=marker_size,
                symbol=marker_symbol,
                line=dict(color="black", width=1),
            ),
            hovertemplate=hover,
        ))

# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Albert Chen, Sayan Patra
"""Plotting functions in plotly."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from abvelocity.ts.common import constants as cst
from abvelocity.ts.common.features.timeseries_features import build_time_features_df
from abvelocity.ts.common.logging import LoggingLevelEnum, log_message
from abvelocity.ts.common.python_utils import update_dictionary
from abvelocity.ts.common.viz.colors_utils import get_color_palette, get_distinct_colors
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots


def plot_multivariate(
    df,
    x_col,
    y_col_style_dict="plotly",
    default_color="rgba(0, 145, 202, 1.0)",
    xlabel=None,
    ylabel=cst.VALUE_COL,
    title=None,
    showlegend=True,
):
    """Plots one or more lines against the same x-axis values.

    Args:
        df: Data frame with ``x_col`` and columns named by the keys in ``y_col_style_dict``.
            ``pandas.DataFrame``.
        x_col: Which column to plot on the x-axis.  ``str``.
        y_col_style_dict: The column(s) to plot on the y-axis, and how to style them.
            ``dict`` [``str``, ``dict`` or None] or "plotly" or "auto" or "auto-fill", default "plotly".

            If a dictionary:

                - key : ``str``
                    column name in ``df``
                - value : ``dict`` or None
                    Optional styling options, passed as kwargs to `go.Scatter`.
                    If None, uses the default: line labeled by the column name.
                    See reference page for `plotly.graph_objects.Scatter` for options
                    (e.g. color, mode, width/size, opacity).
                    https://plotly.com/python/reference/#scatter.

            If a string, plots all columns in ``df`` besides ``x_col`` against ``x_col``:

                - "plotly": plot lines with default plotly styling
                - "auto": plot lines with color ``default_color``, sorted by value (ascending)
                - "auto-fill": plot lines with color ``default_color``, sorted by value (ascending), and fills between lines
        default_color: Default line color when ``y_col_style_dict`` is one of "auto", "auto-fill".
            ``str``, default "rgba(0, 145, 202, 1.0)" (blue).
        xlabel: x-axis label. If None, default is ``x_col``.  ``str`` or None, default None.
        ylabel: y-axis label.  ``str`` or None, default ``VALUE_COL``.
        title: Plot title. If None, default is based on axis labels.  ``str`` or None, default None.
        showlegend: Whether to show the legend.  ``bool``, default True.

    Returns:
        fig: Interactive plotly graph of one or more columns in ``df`` against ``x_col``.
            ``plotly.graph_objects.Figure``.

            See `~abvelocity.ts.common.viz.timeseries_plotting.plot_forecast_vs_actual`
            return value for how to plot the figure and add customization.
    """

    if xlabel is None:
        xlabel = x_col
    if title is None and ylabel is not None:
        title = f"{ylabel} vs {xlabel}"

    auto_style = {"line": {"color": default_color}}
    if y_col_style_dict == "plotly":
        # Uses plotly default style
        y_col_style_dict = {col: None for col in df.columns if col != x_col}
    elif y_col_style_dict in ["auto", "auto-fill"]:
        # Columns ordered from low to high
        means = df.drop(columns=x_col).mean()
        column_order = list(means.sort_values().index)
        if y_col_style_dict == "auto":
            # Lines with color `default_color`
            y_col_style_dict = {col: auto_style for col in column_order}
        elif y_col_style_dict == "auto-fill":
            # Lines with color `default_color`, with fill between lines
            y_col_style_dict = {column_order[0]: auto_style}
            y_col_style_dict.update({col: {"line": {"color": default_color}, "fill": "tonexty"} for col in column_order[1:]})

    data = []
    default_style = dict(mode="lines")
    for column, style_dict in y_col_style_dict.items():
        # By default, column name in ``df`` is used to label the line
        default_col_style = update_dictionary(default_style, overwrite_dict={"name": column})
        # User can overwrite any of the default values, or remove them by setting key value to None
        style_dict = update_dictionary(default_col_style, overwrite_dict=style_dict)
        line = go.Scatter(x=df[x_col], y=df[column], **style_dict)
        data.append(line)

    layout = go.Layout(
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        title=title,
        title_x=0.5,
        showlegend=showlegend,
        legend={"traceorder": "reversed"},  # Matches the order of ``y_col_style_dict`` (bottom to top)
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def plot_multivariate_grouped(
    df,
    x_col,
    y_col_style_dict,
    grouping_x_col,
    grouping_x_col_values,
    grouping_y_col_style_dict,
    colors=DEFAULT_PLOTLY_COLORS,
    xlabel=None,
    ylabel=cst.VALUE_COL,
    title=None,
    showlegend=True,
):
    """Plots multiple lines against the same x-axis values.

    The lines can partially share the x-axis values.

    See parameter descriptions for a running example.

    Args:
        df: Data frame with ``x_col`` and columns named by the keys in ``y_col_style_dict``,
            ``grouping_x_col``, ``grouping_y_col_style_dict``.  ``pandas.DataFrame``.

            For example::

                df = pd.DataFrame({
                    time: [dt(2018, 1, 1),
                            dt(2018, 1, 2),
                            dt(2018, 1, 3)],
                    "y1": [8.5, 2.0, 3.0],
                    "y2": [1.4, 2.1, 3.4],
                    "y3": [4.2, 3.1, 3.0],
                    "y4": [0, 1, 2],
                    "y5": [10, 9, 8],
                    "group": [1, 2, 1],
                })
            This will be our running example.
        x_col: Which column to plot on the x-axis. "time" in our example.  ``str``.
        y_col_style_dict: The column(s) to plot on the y-axis, and how to style them.
            These columns are plotted against the complete x-axis.
            ``dict`` [``str``, ``dict`` or None].

            - key : ``str``
                column name in ``df``
            - value : ``dict`` or None
                Optional styling options, passed as kwargs to `go.Scatter`.
                If None, uses the default: line labeled by the column name.
                If line color is not given, it is added according to ``colors``.
                See reference page for `plotly.graph_objects.Scatter` for options
                (e.g. color, mode, width/size, opacity).
                https://plotly.com/python/reference/#scatter.

            For example::

                y_col_style_dict={
                    "y1": {
                        "name": "y1_name",
                        "legendgroup": "one",
                        "mode": "markers",
                        "line": None  # Remove line params since we use mode="markers"
                    },
                    "y2": None,
                }

            The function will add a line color to "y1" and "y2" based on the ``colors`` parameter.
            It will also add a name to "y2", since none was given. The "name" of "y1" will be preserved.

            The output ``fig`` will have one line each for each of "y1" and "y2", each plot against
            the entire "time" column.
        grouping_x_col: Which column to use to group columns in ``grouping_y_col_style_dict``.
            "group" in our example.  ``str``.
        grouping_x_col_values: Which values to use for grouping. If None, uses all the unique values in
            ``df`` [``grouping_x_col``].
            In our example, specifying ``grouping_x_col_values == [1, 2]`` would plot
            separate lines corresponding to ``group==1`` and ``group==2``.
            ``list`` [``int``] or None.
        grouping_y_col_style_dict: The column(s) to plot on the y-axis, and how to style them.
            These columns are plotted against partial x-axis.
            For each ``grouping_x_col_values`` an element in this dictionary produces
            one line.  ``dict`` [``str``, ``dict`` or None].

            - key : ``str``
                column name in ``df``
            - value : ``dict`` or None
                Optional styling options, passed as kwargs to `go.Scatter`.
                If None, uses the default: line labeled by the ``grouping_x_col_values``,
                ``grouping_x_col`` and column name.
                If a name is given, it is augmented with the ``grouping_x_col_values``.
                If line color is not given, it is added according to ``colors``.
                All the lines sharing same ``grouping_x_col_values`` have the same color.
                See reference page for `plotly.graph_objects.Scatter` for options
                (e.g. color, mode, width/size, opacity).
                https://plotly.com/python/reference/#scatter.

            For example::

                grouping_y_col_style_dict={
                    "y3": {
                        "line": {
                            "color": "blue"
                        }
                    },
                    "y4": {
                        "name": "y4_name",
                        "line": {
                            "width": 2,
                            "dash": "dot"
                        }
                    },
                    "y5": None,
                }

            The function will add a line color to "y4" and "y5" based on the ``colors`` parameter.
            The line color of "y3" will be "blue" as specified. We also preserve the given line
            properties of "y4".

            The function adds a name to "y3" and "y5", since none was given. The given "name" of "y4"
            will be augmented with ``grouping_x_col_values``.

            Each element of ``grouping_y_col_style_dict`` gets one line for each ``grouping_x_col_values``.
            In our example, there will be 2 lines corresponding to "y3", named "1_y3" and "2_y3".
            "1_y3" is plotted against "time = [dt(2018, 1, 1), dt(2018, 1, 3)]", corresponding to ``group==1``.
            "2_y3" is plotted against "time = [dt(2018, 1, 2)", corresponding to ``group==2``.
        colors: Which colors to use to build a color palette for plotting.
            This can be a list of RGB colors or a ``str`` from ``PLOTLY_SCALES``.
            Required number of colors equals sum of the length of ``y_col_style_dict``
            and length of ``grouping_x_col_values``.
            See `~abvelocity.ts.common.viz.colors_utils.get_color_palette` for details.
            [``str``, ``list`` [``str``]], default ``DEFAULT_PLOTLY_COLORS``.
        xlabel: x-axis label. If None, default is ``x_col``.  ``str`` or None, default None.
        ylabel: y-axis label.  ``str`` or None, default ``VALUE_COL``.
        title: Plot title. If None, default is based on axis labels.  ``str`` or None, default None.
        showlegend: Whether to show the legend.  ``bool``, default True.

    Returns:
        fig: Interactive plotly graph of one or more columns in ``df`` against ``x_col``.
            ``plotly.graph_objects.Figure``.

            See `~abvelocity.ts.common.viz.timeseries_plotting.plot_forecast_vs_actual`
            return value for how to plot the figure and add customization.
    """

    available_grouping_x_col_values = np.unique(df[grouping_x_col])
    if grouping_x_col_values is None:
        grouping_x_col_values = available_grouping_x_col_values
    else:
        missing_grouping_x_col_values = set(grouping_x_col_values) - set(available_grouping_x_col_values)
        if len(missing_grouping_x_col_values) > 0:
            raise ValueError(f"Following 'grouping_x_col_values' are missing in '{grouping_x_col}' column: " f"{missing_grouping_x_col_values}")

    # Chooses the color palette
    n_color = len(y_col_style_dict) + len(grouping_x_col_values)
    color_palette = get_color_palette(num=n_color, colors=colors)

    # Updates colors for y_col_style_dict if it is not specified
    for color_num, (column, style_dict) in enumerate(y_col_style_dict.items()):
        if style_dict is None:
            style_dict = {}
        default_color = {"color": color_palette[color_num]}
        style_dict["line"] = update_dictionary(default_color, overwrite_dict=style_dict.get("line"))
        y_col_style_dict[column] = style_dict

    # Standardizes dataset for the next figure
    df_standardized = df.copy().drop_duplicates(subset=[x_col]).sort_values(by=x_col)

    # This figure plots the whole xaxis vs yaxis values
    fig = plot_multivariate(
        df=df_standardized,
        x_col=x_col,
        y_col_style_dict=y_col_style_dict,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        showlegend=showlegend,
    )
    data = fig.data
    layout = fig.layout

    # These figures plot the sliced xaxis vs yaxis values
    for color_num, grouping_x_col_value in enumerate(grouping_x_col_values, len(y_col_style_dict)):
        default_color = {"color": color_palette[color_num]}

        sliced_y_col_style_dict = grouping_y_col_style_dict.copy()

        for column, style_dict in sliced_y_col_style_dict.items():
            # Updates colors if it is not specified
            if style_dict is None:
                style_dict = {}
            line_dict = update_dictionary(default_color, overwrite_dict=style_dict.get("line"))

            # Augments names with grouping_x_col_value
            name = style_dict.get("name")
            if name is None:
                updated_name = f"{grouping_x_col_value}_{grouping_x_col}_{column}"
            else:
                updated_name = f"{grouping_x_col_value}_{name}"

            overwrite_dict = {"name": updated_name, "line": line_dict}
            style_dict = update_dictionary(style_dict, overwrite_dict=overwrite_dict)
            sliced_y_col_style_dict[column] = style_dict

        df_sliced = df[df[grouping_x_col] == grouping_x_col_value]
        fig = plot_multivariate(df=df_sliced, x_col=x_col, y_col_style_dict=sliced_y_col_style_dict)
        data = data + fig.data

    fig = go.Figure(data=data, layout=layout)

    return fig


def plot_univariate(
    df,
    x_col,
    y_col,
    xlabel=None,
    ylabel=None,
    title=None,
    color="rgb(32, 149, 212)",  # light blue
    showlegend=True,
):
    """Simple plot of univariate timeseries.

    Args:
        df: Data frame with ``x_col`` and ``y_col``.  ``pandas.DataFrame``.
        x_col: x-axis column name, usually the time column.  ``str``.
        y_col: y-axis column name, the value the plot.  ``str``.
        xlabel: x-axis label.  ``str`` or None, default None.
        ylabel: y-axis label.  ``str`` or None, default None.
        title: Plot title. If None, default is based on axis labels.  ``str`` or None, default None.
        color: Line color.  ``str``, default "rgb(32, 149, 212)" (light blue).
        showlegend: Whether to show the legend.  ``bool``, default True.

    Returns:
        fig: Interactive plotly graph of the value against time.  ``plotly.graph_objects.Figure``.

            See `~abvelocity.ts.common.viz.timeseries_plotting.plot_forecast_vs_actual`
            return value for how to plot the figure and add customization.

    See Also:
        `~abvelocity.ts.common.viz.timeseries_plotting.plot_multivariate`
        Provides more styling options. Also consider using plotly's `go.Scatter` and `go.Layout` directly.
    """
    # sets default x and y-axis names based on column names
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col

    y_col_style_dict = {y_col: dict(name=y_col, mode="lines", line=dict(color=color), opacity=0.8)}
    return plot_multivariate(
        df,
        x_col,
        y_col_style_dict,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        showlegend=showlegend,
    )


def plot_forecast_vs_actual(
    df,
    time_col=cst.TIME_COL,
    actual_col=cst.ACTUAL_COL,
    predicted_col=cst.PREDICTED_COL,
    predicted_lower_col=cst.PREDICTED_LOWER_COL,
    predicted_upper_col=cst.PREDICTED_UPPER_COL,
    xlabel=cst.TIME_COL,
    ylabel=cst.VALUE_COL,
    train_end_date=None,
    title=None,
    showlegend=True,
    actual_mode="lines+markers",
    actual_points_color="rgba(250, 43, 20, 0.7)",  # red
    actual_points_size=2.0,
    actual_color_opacity=1.0,
    forecast_curve_color="rgba(0, 90, 181, 0.7)",  # blue
    forecast_curve_dash="solid",
    ci_band_color="rgba(0, 90, 181, 0.15)",  # light blue
    ci_boundary_curve_color="rgba(0, 90, 181, 0.5)",  # light blue
    ci_boundary_curve_width=0.0,  # no line
    vertical_line_color="rgba(100, 100, 100, 0.9)",  # black color with opacity of 0.9
    vertical_line_width=1.0,
):
    """Plots forecast with prediction intervals, against actuals.

    Adapted from plotly user guide:
    https://plot.ly/python/v3/continuous-error-bars/#basic-continuous-error-bars

    Args:
        df: Timestamp, predicted, and actual values.  ``pandas.DataFrame``.
        time_col: Column in df with timestamp (x-axis).  ``str``,
            default `~abvelocity.ts.common.constants.TIME_COL`.
        actual_col: Column in df with actual values.  ``str``,
            default `~abvelocity.ts.common.constants.ACTUAL_COL`.
        predicted_col: Column in df with predicted values.  ``str``,
            default `~abvelocity.ts.common.constants.PREDICTED_COL`.
        predicted_lower_col: Column in df with predicted lower bound.  ``str`` or None,
            default `~abvelocity.ts.common.constants.PREDICTED_LOWER_COL`.
        predicted_upper_col: Column in df with predicted upper bound.  ``str`` or None,
            default `~abvelocity.ts.common.constants.PREDICTED_UPPER_COL`.
        xlabel: x-axis label.  ``str``, default `~abvelocity.ts.common.constants.TIME_COL`.
        ylabel: y-axis label.  ``str``, default `~abvelocity.ts.common.constants.VALUE_COL`.
        train_end_date: Train end date.
            Must be a value in ``df[time_col]``.  ``datetime.datetime`` or None, default None.
        title: Plot title.  ``str`` or None, default None.
        showlegend: Whether to show a plot legend.  ``bool``, default True.
        actual_mode: How to show the actuals.
            Options: ``markers``, ``lines``, ``lines+markers``.
            ``str``, default "lines+markers".
        actual_points_color: Color of actual line/marker.  ``str``, default "rgba(99, 114, 218, 1.0)".
        actual_points_size: Size of actual markers.
            Only used if "markers" is in ``actual_mode``.  ``float``, default 2.0.
        actual_color_opacity: Opacity of actual values points.  ``float`` or None, default 1.0.
        forecast_curve_color: Color of forecasted values.  ``str``, default "rgba(0, 145, 202, 1.0)".
        forecast_curve_dash: 'dash' property of forecast ``scatter.line``.
            One of: ``['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']``
            or a string containing a dash length list in pixels or percentages
            (e.g. ``'5px 10px 2px 2px'``, ``'5, 10, 2, 2'``, ``'10% 20% 40%'``).
            ``str``, default "solid".
        ci_band_color: Fill color of the prediction bands.  ``str``, default "rgba(0, 145, 202, 0.15)".
        ci_boundary_curve_color: Color of the prediction upper/lower lines.
            ``str``, default "rgba(0, 145, 202, 0.15)".
        ci_boundary_curve_width: Width of the prediction upper/lower lines.
            default 0.0 (hidden).  ``float``, default 0.0.
        vertical_line_color: Color of the vertical line indicating train end date.
            Default is black with opacity of 0.9.
            ``str``, default "rgba(100, 100, 100, 0.9)".
        vertical_line_width: width of the vertical line indicating train end date.
            ``float``, default 1.0.

    Returns:
        fig: Plotly figure of forecast against actuals, with prediction
            intervals if available.  ``plotly.graph_objects.Figure``.

            Can show, convert to HTML, update::

                # show figure
                fig.show()

                # get HTML string, write to file
                fig.to_html(include_plotlyjs=False, full_html=True)
                fig.write_html("figure.html", include_plotlyjs=False, full_html=True)

                # customize layout (https://plot.ly/python/v3/user-guide/)
                update_layout = dict(
                    yaxis=dict(title="new ylabel"),
                    title_text="new title",
                    title_x=0.5,
                    title_font_size=30)
                fig.update_layout(update_layout)
    """
    if title is None:
        title = "Forecast vs Actual"
    if train_end_date is not None and not all(pd.Series(train_end_date).isin(df[time_col])):
        raise ValueError(f"train_end_date {train_end_date} is not found in df['{time_col}']")

    fill_dict = {"mode": "lines", "fillcolor": ci_band_color, "fill": "tonexty"}
    data = []
    if predicted_lower_col is not None:
        lower_bound = go.Scatter(
            name="Lower Bound",
            x=df[time_col],
            y=df[predicted_lower_col],
            mode="lines",
            line=dict(width=ci_boundary_curve_width, color=ci_boundary_curve_color),
            legendgroup="interval",  # show/hide with the upper bound
        )
        data.append(lower_bound)

    # plotly fills between current and previous element in `data`.
    # Only fill if lower bound exists.
    forecast_fill_dict = fill_dict if predicted_lower_col is not None else {}
    if predicted_upper_col is not None:
        upper_bound = go.Scatter(
            name="Upper Bound",
            x=df[time_col],
            y=df[predicted_upper_col],
            line=dict(width=ci_boundary_curve_width, color=ci_boundary_curve_color),
            legendgroup="interval",  # show/hide with the lower bound
            **forecast_fill_dict,
        )
        data.append(upper_bound)

    # If `predicted_lower_col` and `predicted_upper_col`, then the full range
    # has been filled in. If only one of them, then fill in between that line
    # and forecast.
    actual_params = {}
    if "lines" in actual_mode:
        actual_params.update(line=dict(color=actual_points_color))
    if "markers" in actual_mode:
        actual_params.update(marker=dict(color=actual_points_color, size=actual_points_size))
    actual = go.Scatter(
        name="Actual",
        x=df[time_col],
        y=df[actual_col],
        mode=actual_mode,
        opacity=actual_color_opacity,
        **actual_params,
    )
    data.append(actual)

    forecast_fill_dict = fill_dict if (predicted_lower_col is None) != (predicted_upper_col is None) else {}
    forecast = go.Scatter(
        name="Forecast",
        x=df[time_col],
        y=df[predicted_col],
        line=dict(color=forecast_curve_color, dash=forecast_curve_dash),
        **forecast_fill_dict,
    )
    data.append(forecast)

    layout = go.Layout(
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        title=title,
        title_x=0.5,
        showlegend=showlegend,
        # legend order from top to bottom: Actual, Forecast, Upper Bound, Lower Bound
        legend={"traceorder": "reversed"},
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update()

    # adds a vertical line to separate training and testing phases
    if train_end_date is not None:
        new_layout = dict(
            # add vertical line
            shapes=[
                dict(
                    type="line",
                    xref="x",
                    yref="paper",  # y-reference is assigned to the plot paper [0,1]
                    x0=train_end_date,
                    y0=0,
                    x1=train_end_date,
                    y1=1,
                    line=dict(color=vertical_line_color, width=vertical_line_width),
                )
            ],
            # add text annotation
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
        fig.update_layout(new_layout)
    return fig


def split_range_into_groups(n, group_size, which_group_complete="last"):
    """Partitions `n` elements into adjacent groups, each with `group_size` elements.

    Group number starts from 0 and increments upward.
    Can be used to generate groups for sliding window aggregation.

    Args:
        n: number of elements to split into groups.  int.
        group_size: number of elements per group.  int.
        which_group_complete: If n % group_size > 0, one group will have fewer than `group_size` elements.
            if "first", the first group is full if possible, and last group may be incomplete.
            if "last", (default) the last group is full if possible,
            and first group may be incomplete.  str.

    Returns:
        np.array of length n: values correspond to the element's group number.

    Examples:
        >>> split_range_into_groups(10, 1, "last")
        array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> split_range_into_groups(10, 2, "last")
        array([0., 0., 1., 1., 2., 2., 3., 3., 4., 4.])
        >>> split_range_into_groups(10, 3, "last")
        array([0., 1., 1., 1., 2., 2., 2., 3., 3., 3.])
        >>> split_range_into_groups(10, 4, "last")
        array([0., 0., 1., 1., 1., 1., 2., 2., 2., 2.])
        >>> split_range_into_groups(10, 4, "first")
        array([0., 0., 0., 0., 1., 1., 1., 1., 2., 2.])
        >>> split_range_into_groups(10, 5, "last")
        array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
        >>> split_range_into_groups(10, 6, "last")
        array([0., 0., 0., 0., 1., 1., 1., 1., 1., 1.])
        >>> split_range_into_groups(10, 10, "last")
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        >>> split_range_into_groups(10, 12, "last")
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    if which_group_complete.lower() == "first":
        offset = 0
    else:
        offset = group_size - n % group_size
        offset = offset % group_size  # sets offset to 0 if n % group_size == 0
    return np.floor(np.arange(offset, n + offset) / group_size)


def add_groupby_column(
    df,
    time_col,
    groupby_time_feature=None,
    groupby_sliding_window_size=None,
    groupby_custom_column=None,
):
    """Extracts a column to group by from ``df``.

    Exactly one of ``groupby_time_feature``, ``groupby_sliding_window_size``,
    ``groupby_custom_column`` must be provided.

    Args:
        df: Contains the univariate time series / forecast.  ``pandas.DataFrame``.
        time_col: The name of the time column of the univariate time series / forecast.  ``str``.
        groupby_time_feature: If provided, groups by a column generated by
            `~abvelocity.ts.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.  ``str`` or None, optional.
        groupby_sliding_window_size: If provided, sequentially partitions data into groups of size
            ``groupby_sliding_window_size``.  ``int`` or None, optional.
        groupby_custom_column: If provided, groups by this column value.
            Should be same length as the ``df``.  ``pandas.Series`` or None, optional.

    Returns:
        result: Dictionary with two items.  ``dict``.

            * ``"df"`` : ``pandas.DataFrame``
                ``df`` with a grouping column added.
                The column can be used to group rows together.

            * ``"groupby_col"`` : ``str``
                The name of the groupby column added to ``df``.
                The column name depends on the grouping method:

                    - ``groupby_time_feature`` for ``groupby_time_feature``
                    - ``{cst.TIME_COL}_downsample`` for ``groupby_sliding_window_size``
                    - ``groupby_custom_column.name`` for ``groupby_custom_column``.
    """
    # Resets index to support indexing in groupby_sliding_window_size
    df = df.copy()
    dt = pd.Series(df[time_col].values)
    # Determines the groups
    is_groupby_time_feature = 1 if groupby_time_feature is not None else 0
    is_groupby_sliding_window_size = 1 if groupby_sliding_window_size is not None else 0
    is_groupby_custom_column = 1 if groupby_custom_column is not None else 0
    if is_groupby_time_feature + is_groupby_sliding_window_size + is_groupby_custom_column != 1:
        raise ValueError("Exactly one of (groupby_time_feature, groupby_rolling_window_size, groupby_custom_column)" "must be specified")
    groups = None
    if is_groupby_time_feature == 1:
        # Group by a value derived from the time column
        time_features = build_time_features_df(dt, conti_year_origin=min(dt).year)
        groups = time_features[groupby_time_feature]
        groups.name = groupby_time_feature
    elif is_groupby_sliding_window_size == 1:
        # Group by sliding window for evaluation over time
        index_dates = split_range_into_groups(
            n=df.shape[0], group_size=groupby_sliding_window_size, which_group_complete="last"
        )  # ensures the last group is complete (first group may be partial)
        groups = dt[index_dates * groupby_sliding_window_size]  # uses first date in each group as grouping value
        groups.name = f"{time_col}_downsample"
    elif is_groupby_custom_column == 1:
        # Group by custom column
        groups = groupby_custom_column

    groups_col_name = groups.name if groups.name is not None else "groups"
    df[groups_col_name] = groups.values
    if df.index.name in df.columns:
        # Removes ambiguity in case the index name is the same as the newly added column,
        # (or an existing column).
        df.index.name = None
    return {"df": df, "groupby_col": groups_col_name}


def grouping_evaluation(df, groupby_col, grouping_func, grouping_func_name):
    """Groups ``df`` and evaluates a function on each group.

    The function takes a ``pandas.DataFrame`` and returns a scalar.

    Args:
        df: Input data. For example, univariate time series, or forecast result.
            Contains ``groupby_col`` and columns to apply ``grouping_func`` on.
            ``pandas.DataFrame``.
        groupby_col: Column name in ``df`` to group by.  ``str``.
        grouping_func: Function that is applied to each group via `pandas.groupBy.apply`.
            Signature (grp: ``pandas.DataFrame``) -> aggregated value: ``float``.  ``callable``.
        grouping_func_name: What to call the output column generated by ``grouping_func``.  ``str``.

    Returns:
        grouped_df: Dataframe with ``grouping_func`` evaluated on each level of ``df[groupby_col]``.
            Contains two columns.  ``pandas.DataFrame``.

                - ``groupby_col``: The groupby value
                - ``grouping_func_name``: The output of ``grouping_func`` on the group
    """
    grouped_df = df.groupby(groupby_col).apply(grouping_func).reset_index().rename({0: grouping_func_name}, axis=1)

    return grouped_df


def flexible_grouping_evaluation(
    df,
    map_func_dict=None,
    groupby_col=None,
    agg_kwargs=None,
    extend_col_names=True,
    unpack_list=True,
    list_names_dict=None,
):
    """Flexible aggregation.

    Generates additional columns for evaluation via ``map_func_dict``, groups by ``groupby_col``,
    then aggregates according to ``agg_kwargs``.

    This function calls `pandas.DataFrame.apply` and
    `pandas.core.groupby.DataFrameGroupBy.agg` internally.

    Args:
        df: DataFrame to transform / aggregate.  ``pandas.DataFrame``.
        map_func_dict: Row-wise transformation functions to create new columns.
            If None, no new columns are added.  ``dict`` [``str``, ``callable``] or None, default None.

            key: new column name.
            value: row-wise function to apply to ``df`` to generate the column value.
                Signature (row: ``pandas.DataFrame``) -> transformed value: ``float``.

            For example::

                map_func_dict = {
                    "residual": lambda row: row["predicted"] - row["actual"],
                    "squared_error": lambda row: (row["predicted"] - row["actual"])**2
                }
        groupby_col: Which column to group by.
            Can be in ``df`` or generated by ``map_func_dict``.
            If None, no grouping or aggregation is done.  ``str`` or None, default None.
        agg_kwargs: Passed as keyword args to `pandas.core.groupby.DataFrameGroupBy.aggregate` after creating
            new columns and grouping by ``groupby_col``. Must be provided if ``groupby_col is not None``.
            To fully customize output column names, pass a dictionary as shown below.
            ``dict`` or None, default None.

            For example::

                # Example 1, named aggregation to explicitly name output columns.
                # Assume ``df`` contains ``abs_percent_err``, ``abs_err`` columns.
                # Output columns are "MedAPE", "MAPE", "MAE", etc. in a single level index.
                from functools import partial
                agg_kwargs = {
                    # output column name: (column to aggregate, aggregation function)
                    "MedAPE": pd.NamedAgg(column="abs_percent_err", aggfunc=np.nanmedian),
                    "MAPE": pd.NamedAgg(column="abs_percent_err", aggfunc=np.nanmean),
                    "MAE": pd.NamedAgg(column="abs_err", aggfunc=np.nanmean),
                    "q95_abs_err": pd.NamedAgg(column="abs_err", aggfunc=partial(np.nanquantile, q=0.95)),
                    "q05_abs_err": pd.NamedAgg(column="abs_err", aggfunc=partial(np.nanquantile, q=0.05)),
                }

                # Example 2, multi-level aggregation using `func` parameter
                # to `pandas.core.groupby.DataFrameGroupBy.aggregate`.
                # Assume ``df`` contains ``y1``, ``y2`` columns.
                agg_kwargs = {
                    "func": {
                        "y1": [np.nanmedian, np.nanmean],
                        "y2": [np.nanmedian, np.nanmax],
                    }
                }
                # `extend_col_names` controls the output column names
                extend_col_names = True  # output columns are "y1_nanmean", "y1_nanmedian", "y2_nanmean", "y2_nanmax"
                extend_col_names = False  # output columns are "nanmean", "nanmedian", "nanmean", "nanmax"
        extend_col_names: How to flatten index after aggregation.
            In some cases, the column index after aggregation is a multi-index.
            This parameter controls how to flatten an index with 2 levels to 1 level.
            ``bool`` or None, default True.

                - If None, the index is not flattened.
                - If True, column name is a composite: ``{index0}_{index1}``
                  Use this option if index1 is not unique.
                - If False, column name is simply ``{index1}``

            Ignored if the ColumnIndex after aggregation has only one level (e.g.
            if named aggregation is used in ``agg_kwargs``).
        unpack_list: Whether to unpack (flatten) columns that contain list/tuple after aggregation,
            to create one column per element of the list/tuple.
            If True, ``list_names_dict`` can be used to rename the unpacked columns.
            ``bool``, default True.
        list_names_dict: If ``unpack_list`` is True, this dictionary can optionally be
            used to rename the unpacked columns.
            ``dict`` [``str``, ``list`` [``str``]] or None, default None.

                - Key = column name after aggregation, before unpacking.
                  E.g. ``{index0}_{index1}`` or ``{index1}`` depending on ``extend_col_names``.
                - Value = list of names to use for the unpacked columns. Length must match
                  the length of the lists contained in the column.

            If a particular list/tuple column is not found in this dictionary, appends
            0, 1, 2, ..., n-1 to the original column name, where n = list length.

            For example, if the column contains a tuple of length 4 corresponding to
            quantiles 0.1, 0.25, 0.75, 0.9, then the following would be appropriate::

                aggfunc = lambda grp: partial(np.nanquantile, q=[0.1, 0.25, 0.75, 0.9])(grp).tolist()
                agg_kwargs = {
                    "value_Q": pd.NamedAgg(column="value", aggfunc=aggfunc)
                }
                list_names_dict = {
                    # the key is the name of the unpacked column
                    "value_Q" : ["Q0.10", "Q0.25", "Q0.75", "Q0.90"]
                }
                # Output columns are "Q0.10", "Q0.25", "Q0.75", "Q0.90"

                # In this example, if list_names_dict=None, the default output column names
                # would be: "value_Q0", "value_Q1", "value_Q2", "value_Q3"

    Returns:
        df_transformed: df after transformation and optional aggregation.  ``pandas.DataFrame``.

            If ``groupby_col`` is None, returns ``df`` with additional columns as the keys in ``map_func_dict``.
            Otherwise, ``df`` is grouped by ``groupby_col`` and this becomes the index. Columns
            are determined by ``agg_kwargs`` and ``extend_col_names``.
    """
    if groupby_col and not agg_kwargs:
        raise ValueError("Must specify `agg_kwargs` if grouping is requested via `groupby_col`.")
    if agg_kwargs and not groupby_col:
        log_message(
            "`agg_kwargs` is ignored because `groupby_col` is None. " "Specify `groupby_col` to allow aggregation.",
            LoggingLevelEnum.WARNING,
        )

    df = df.copy()
    if map_func_dict is not None:
        for col_name, func in map_func_dict.items():
            df[col_name] = df.apply(func, axis=1)

    if groupby_col is not None:
        groups = df.groupby(groupby_col)
        with warnings.catch_warnings():
            # Ignores pandas FutureWarning. Use NamedAgg in pandas 0.25.+
            warnings.filterwarnings("ignore", message="using a dict with renaming is deprecated", category=FutureWarning)
            df_transformed = groups.agg(**agg_kwargs)
        if extend_col_names is not None and df_transformed.columns.nlevels > 1:
            # Flattens multi-level column index
            if extend_col_names:
                # By concatenating names
                df_transformed.columns = ["_".join(col).strip("_") for col in df_transformed.columns]
            else:
                # By using level 1 names
                df_transformed.columns = list(df_transformed.columns.get_level_values(1))
                if np.any(df_transformed.columns.duplicated()):
                    warnings.warn("Column names are not unique. Use `extend_col_names=True` " "to uniquely identify every column.")
    else:
        # No grouping is requested
        df_transformed = df

    if unpack_list and df_transformed.shape[0] > 0:
        # Identifies the columns that contain list elements
        which_list_cols = df_transformed.iloc[0].apply(lambda x: isinstance(x, (list, tuple)))
        list_cols = list(which_list_cols[which_list_cols].index)
        for col in list_cols:
            if isinstance(df_transformed[col], pd.DataFrame):
                warnings.warn(
                    f"Skipping list unpacking for `{col}`. There are multiple columns "
                    f"with this name. Make sure column names are unique to enable unpacking."
                )
                continue
            # Unpacks the column, creating one column for each list entry
            list_df = pd.DataFrame(df_transformed[col].to_list())
            n_cols = list_df.shape[1]
            # Adds column names
            if list_names_dict is not None and col in list_names_dict:
                found_length = len(list_names_dict[col])
                if found_length != n_cols:
                    raise ValueError(
                        f"list_names_dict['{col}'] has length {found_length}, "
                        f"but there are {n_cols} columns to name. Example row(s):\n"
                        f"{list_df.head(2)}"
                    )
                list_df.columns = [f"{list_names_dict.get(col)[i]}" for i in range(n_cols)]
            else:
                list_df.columns = [f"{col}{i}" for i in range(n_cols)]
            # replaces original column with new ones
            list_df.index = df_transformed.index
            del df_transformed[col]
            df_transformed = pd.concat([df_transformed, list_df], axis=1)

        if list_names_dict:
            unused_names = sorted(list(set(list_names_dict.keys()) - set(list_cols)))
            if len(unused_names) > 0:
                warnings.warn(
                    "These names from `list_names_dict` are not used, because the "
                    "column (key) is not found in the dataframe after aggregation:\n"
                    f"{unused_names}.\nAvailable columns are:\n"
                    f"{list_cols}."
                )

    return df_transformed


def plot_dual_axis_figure(
    df,
    x_col,
    y_left_col,
    y_right_col,
    grouping_col=None,
    xlabel=None,
    ylabel_left=None,
    ylabel_right=None,
    title=None,
    y_left_linestyle="solid",
    y_right_linestyle="dash",
    opacity=0.9,
    axis_font_size=18,
    title_font_size=20,
    x_range=None,
    y_left_range=None,
    y_right_range=None,
    x_tick_format=None,
    y_left_tick_format=None,
    y_right_tick_format=None,
    x_hover_format=None,
    y_left_hover_format=None,
    y_right_hover_format=None,
    group_color_dict=None,
):
    """Generic function to plot a dual y-axis plot.

    The x-axis is specified by ``x_col``.
    The left and right y-axes are specified by ``y_left_col`` and ``y_right_col`` respectively.
    If ``grouping_col`` is specified, then multiple pairs of curves are drawn, one for each level in ``grouping_col``.

    Args:
        df: The input dataframe. Must contain the columns ``x_col``, ``y_left_col`` and ``y_right_col``.
            If ``grouping_col`` is not None, it must also contain the ``grouping_col`` column.
            ``pandas.DataFrame``.
            For example, the dataframe could look like this.

            +-----------+----------------+-----------------+------------------+
            | ``x_col`` | ``y_left_col`` | ``y_right_col`` | ``grouping_col`` |
            +===========+================+=================+==================+
            |   1.10    |     20.12      |      0.21       |       "A"        |
            +-----------+----------------+-----------------+------------------+
            |   1.40    |     40.31      |      0.43       |       "A"        |
            +-----------+----------------+-----------------+------------------+
            |   1.23    |     63.21      |      NaN        |       "B"        |
            +-----------+----------------+-----------------+------------------+
            |   1.54    |     10.31      |      0.12       |       "B"        |
            +-----------+----------------+-----------------+------------------+
            |    ...    |      ...       |       ...       |       ...        |
            +-----------+----------------+-----------------+------------------+
        x_col: The column name of the column in ``df`` to be used for the x-axis.  ``str``.
        y_left_col: The column name of the column in ``df`` to be used for the left y-axis.  ``str``.
        y_right_col: The column name of the column in ``df`` to be used for the right y-axis.  ``str``.
        grouping_col: Name of the grouping column in ``df`` to be used for overlaying curves for each
            level in ``grouping_col``.  ``str`` or None, default None.
        xlabel: Name for the x-axis label. If it is `None`, then it is set to be ``x_col``.
            ``str`` or None, default None.
        ylabel_left: Name for the left y-axis label. If it is `None`, then it is set to be ``y_left_col``.
            ``str`` or None, default None.
        ylabel_right: Name for the right y-axis label. If it is `None`, then it is set to be ``y_right_col``.
            ``str`` or None, default None.
        title: The title for the plot.  ``str`` or None, default None.
        y_left_linestyle: Line style for the left y-axis curve.  ``str``, default "solid".
        y_right_linestyle: Line style for the right y-axis curve.  ``str``, default "dash".
        opacity: The opacity of the colors. This has to be a number between 0 and 1.  ``float``, default 0.9.
        axis_font_size: The size of the axis fonts.  ``int``, default 18.
        title_font_size: The size of the title fonts.  ``int``, default 20.
        x_range: Range of the x-axis.  ``list`` or None, default None.
        y_left_range: Range of the left y-axis.  ``list`` or None, default None.
        y_right_range: Range of the right y-axis.  ``list`` or None, default None.
        x_tick_format: Format of the ticks on the x-axis.  ``str`` or None, default None.
        y_left_tick_format: Format of the ticks on the left y-axis.  ``str`` or None, default None.
        y_right_tick_format: Format of the ticks on the right y-axis.  ``str`` or None, default None.
        x_hover_format: Format of the values when hovering for the x-axis.  ``str`` or None, default None.
        y_left_hover_format: Format of the values when hovering for the left y-axis.
            ``str`` or None, default None.
        y_right_hover_format: Format of the values when hovering for the right y-axis.
            ``str`` or None, default None.
        group_color_dict: Dictionary with a mapping from levels within the ``grouping_col`` and a specified color.
            The keys are the levels in ``grouping_col`` and the values are a specified color.
            If ``group_color_dict`` is `None`, the colors are generated using the function
            `abvelocity.ts.common.viz.colors_utils.get_distinct_colors`.
            ``dict`` [``str``, ``str``] or None, default None.

    Returns:
        fig: Dual y-axes plot.  ``plotly.graph_objects.Figure``.
    """
    if any([col not in df.columns for col in [x_col, y_left_col, y_right_col]]):
        raise ValueError(f"`df` must contain the columns: '{x_col}', '{y_left_col}' and '{y_right_col}'!")

    # If no custom labels are given, we simply use the names of the passed columns.
    if xlabel is None:
        xlabel = x_col
    if ylabel_left is None:
        ylabel_left = y_left_col
    if ylabel_right is None:
        ylabel_right = y_right_col
    # Stores the data for the left and right curves.
    y_left_data = []
    y_right_data = []
    # Creates the curve(s).
    if grouping_col is None:  # No `grouping_col`
        # In this case, only one color is needed.
        color = get_distinct_colors(num_colors=1, opacity=opacity)[0]
        df = df.reset_index(drop=True).sort_values(x_col)
        # Left lines.
        line_left = go.Scatter(
            name=ylabel_left,
            x=df[x_col].tolist(),
            y=df[y_left_col].tolist(),
            showlegend=True,
            line=dict(dash=y_left_linestyle, color=color),
        )
        y_left_data.append(line_left)
        # Right lines.
        line_right = go.Scatter(
            name=ylabel_right,
            x=df[x_col].tolist(),
            y=df[y_right_col].tolist(),
            showlegend=True,
            line=dict(dash=y_right_linestyle, color=color),
        )
        y_right_data.append(line_right)
    else:  # `grouping_col` is not None.
        # Gets the levels for the specified `grouping_col`.
        levels = df.groupby(grouping_col).groups
        # Assigns colors to levels if not specified.
        if group_color_dict is None:
            color_list = get_distinct_colors(num_colors=len(levels), opacity=opacity)
            group_color_dict = {level: color_list[i] for i, level in enumerate(levels.keys())}
        # Generates curves for each level.
        for level, indices in levels.items():
            df_subset = df.loc[indices].reset_index(drop=True).sort_values(x_col)
            # Left lines.
            line_left = go.Scatter(
                name=ylabel_left,
                legendgroup=f"{grouping_col} = {level}",
                legendgrouptitle_text=f"{grouping_col} = {level}",
                x=df_subset[x_col].tolist(),
                y=df_subset[y_left_col].tolist(),
                showlegend=True,
                line=dict(dash=y_left_linestyle, color=group_color_dict[level]),
            )
            y_left_data.append(line_left)
            # Right lines.
            line_right = go.Scatter(
                name=ylabel_right,
                legendgroup=f"{grouping_col} = {level}",
                legendgrouptitle_text=f"{grouping_col} = {level}",
                x=df_subset[x_col].tolist(),
                y=df_subset[y_right_col].tolist(),
                showlegend=True,
                line=dict(dash=y_right_linestyle, color=group_color_dict[level]),
            )
            y_right_data.append(line_right)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for line_left, line_right in zip(y_left_data, y_right_data):
        fig.add_trace(line_left, secondary_y=False)
        fig.add_trace(line_right, secondary_y=True)
    # Updates figure layout.
    fig.update_layout(
        title_text=title,
        titlefont=dict(size=title_font_size),
        autosize=False,
        width=1000,
        height=800,
        hovermode="x",
    )
    # Updates x-axis.
    fig.update_xaxes(
        title=xlabel,
        titlefont=dict(size=axis_font_size),
        range=x_range,
        tickfont_size=axis_font_size,
        tickformat=x_tick_format,
        hoverformat=x_hover_format,
    ),
    # Updates the left y-axis.
    fig.update_yaxes(
        title_text=ylabel_left,
        secondary_y=False,
        titlefont=dict(size=axis_font_size),
        range=y_left_range,
        tickfont_size=axis_font_size,
        tickformat=y_left_tick_format,
        hoverformat=y_left_hover_format,
    )
    # Updates the right y-axis.
    fig.update_yaxes(
        title_text=ylabel_right,
        secondary_y=True,
        titlefont=dict(size=axis_font_size),
        range=y_right_range,
        tickfont_size=axis_font_size,
        tickformat=y_right_tick_format,
        hoverformat=y_right_hover_format,
    )
    return fig


def plt_compare_timeseries(
    df_dict,
    time_col,
    value_col,
    legends_dict=None,
    colors_dict=None,
    start_time=None,
    end_time=None,
    transform=lambda x: x,
    transform_name="",
    plt_title="",
    alpha=0.6,
    linewidth=2,
):
    """Compare a collection of timeseries by overlaying them on a single plotly figure.

    Args:
        df_dict: Keys are labels for each series; values are dataframes with ``time_col`` and ``value_col``.
            ``dict`` [``str``, ``pandas.DataFrame``].
        time_col: Column name for the time axis.  ``str``.
        value_col: Column name for the value axis.  ``str``.
        legends_dict: Maps df_dict keys to legend labels. If None, df_dict keys are used.
            ``dict`` [``str``, ``str``] or None, default None.
        colors_dict: Maps df_dict keys to color strings. If None, plotly default colors are used.
            ``dict`` [``str``, ``str``] or None, default None.
        start_time: If provided, filters each series to ``time_col >= start_time``.
            datetime or None, default None.
        end_time: If provided, filters each series to ``time_col <= end_time``.
            datetime or None, default None.
        transform: Function applied to the value column before plotting.  callable, default identity.
        transform_name: Name of the transformation, appended to the title.  ``str``, default "".
        plt_title: Plot title.  ``str``, default "".
        alpha: Opacity of the traces.  ``float``, default 0.6.
        linewidth: Width of the lines.  ``float``, default 2.

    Returns:
        fig: ``plotly.graph_objects.Figure``.
    """
    labels = list(df_dict.keys())
    if legends_dict is None:
        legends_dict = {label: label for label in labels}

    fig = go.Figure()
    for label in labels:
        df = df_dict[label].copy()
        if start_time is not None:
            df = df[df[time_col] >= start_time]
        if end_time is not None:
            df = df[df[time_col] <= end_time]
        color = colors_dict.get(label) if colors_dict else None
        legend = legends_dict.get(label, label)
        trace_kwargs = dict(opacity=alpha, line=dict(width=linewidth))
        if color is not None:
            trace_kwargs["line"]["color"] = color
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=transform(df[value_col]),
                name=legend,
                mode="lines",
                **trace_kwargs,
            )
        )
    title = plt_title + (f" ({transform_name})" if transform_name else "")
    fig.update_layout(title=title, xaxis_title=time_col, yaxis_title=value_col)
    return fig

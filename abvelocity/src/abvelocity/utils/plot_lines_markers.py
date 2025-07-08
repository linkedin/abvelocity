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
# Original author: Reza Hosseini

from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go

# Assuming these are imported from their respective modules
from abvelocity.utils.color_utils import get_distinct_colors
from abvelocity.utils.prep_df_for_grouped_plot import prep_df_for_grouped_plot


def plot_lines_markers(
    df: pd.DataFrame,
    x_col: str,
    line_cols: Optional[List[str]] = None,
    marker_cols: Optional[List[str]] = None,
    band_cols: Optional[
        List[str]
    ] = None,  # Corrected: Expects list of column names where each column contains tuples
    band_cols_dict: Optional[
        Dict[str, List[str]]
    ] = None,  # Expects dict where value is [lower_bound_col, upper_bound_col]
    line_colors: Optional[List[str]] = None,
    marker_colors: Optional[List[str]] = None,
    band_colors: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Creates a Plotly figure with lines, markers, and/or filled bands.

    This function provides a lightweight and easy-to-use interface to visualize
    data from a DataFrame, allowing for the plotting of multiple series with
    customizable colors and legends.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot. It must include
            `x_col` and any columns specified in `line_cols`, `marker_cols`,
            `band_cols`, or `band_cols_dict`.
        x_col (str): The name of the column to use for the x-axis.
        line_cols (Optional[List[str]]): A list of column names whose values
            will be plotted as lines/curves on the y-axis. Defaults to None.
        marker_cols (Optional[List[str]]): A list of column names whose values
            will be plotted as markers/points on the y-axis. Defaults to None.
        band_cols (Optional[List[str]]): A list of column names. Each column
            specified here is expected to contain tuples, where each tuple
            represents the (lower bound, upper bound) of a band at a given
            x-value. Defaults to None.
        band_cols_dict (Optional[Dict[str, List[str]]]): A dictionary where
            keys are the desired legend names for bands, and values are lists
            containing two column names representing the lower and upper bounds
            of a band (e.g., `{"forecast": ["forecast_lower", "forecast_upper"]}`).
            Defaults to None.
        line_colors (Optional[List[str]]): A list of color strings to use for
            `line_cols`. If provided, its length must be at least the number
            of `line_cols`. Defaults to None.
        marker_colors (Optional[List[str]]): A list of color strings to use for
            `marker_cols`. If provided, its length must be at least the number
            of `marker_cols`. Defaults to None.
        band_colors (Optional[List[str]]): A list of color strings to use for
            `band_cols` or `band_cols_dict`. These colors will be used as the
            fill color for each band. If provided, its length must be at least
            the number of bands. Defaults to None, in which case distinct colors
            are generated with 0.2 opacity.
        title (Optional[str]): The title of the plot. Defaults to None,
            resulting in no title.

    Returns:
        go.Figure: An interactive Plotly graph figure.

    Raises:
        ValueError: If `line_colors`, `marker_colors`, or `band_colors` lists
            are provided but are shorter than the corresponding column lists.
        ValueError: If none of `line_cols`, `marker_cols`, `band_cols`, or
            `band_cols_dict` are provided.
    """

    if line_colors is not None and line_cols is not None:
        if len(line_colors) < len(line_cols):
            raise ValueError(
                "If `line_colors` is passed, its length must be at least `len(line_cols)`"
            )

    if marker_colors is not None and marker_cols is not None:
        if len(marker_colors) < len(marker_cols):
            raise ValueError(
                "If `marker_colors` is passed, its length must be at least `len(marker_cols)`"
            )

    if band_colors is not None and band_cols is not None:
        if len(band_colors) < len(band_cols):
            raise ValueError(
                "If `band_colors` is passed, its length must be at least `len(band_cols)`"
            )

    if band_colors is not None and band_cols_dict is not None:
        if len(band_colors) < len(band_cols_dict):
            raise ValueError(
                "If `band_colors` is passed, its length must be at least `len(band_cols_dict)`"
            )

    if line_cols is None and marker_cols is None and band_cols is None and band_cols_dict is None:
        raise ValueError(
            "At least one of `line_cols` or `marker_cols` or `band_cols`"
            " or `band_cols_dict` must be passed as a list (not None)."
        )

    fig = go.Figure()
    # Below we count the number of figure components to assign proper labels to legends.
    count_fig_data = -1
    if line_cols is not None:
        for i, col in enumerate(line_cols):
            line_properties = (
                go.scatter.Line(color=line_colors[i])
                if line_colors is not None
                else go.scatter.Line()
            )

            fig.add_trace(
                go.Scatter(
                    x=df[x_col], y=df[col], mode="lines", line=line_properties, showlegend=True
                )
            )
            count_fig_data += 1
            fig["data"][count_fig_data]["name"] = col

    if marker_cols is not None:
        for i, col in enumerate(marker_cols):
            marker_properties = (
                go.scatter.Marker(color=marker_colors[i])
                if marker_colors is not None
                else go.scatter.Marker()
            )
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    mode="markers",
                    marker=marker_properties,
                    showlegend=True,
                )
            )
            count_fig_data += 1
            fig["data"][count_fig_data]["name"] = col

    if band_cols is not None:
        if band_colors is None:
            band_colors = get_distinct_colors(num_colors=len(band_cols), opacity=0.2)

        for i, col_name in enumerate(band_cols):  # Changed 'col_tuple' to 'col_name'
            # Ensure line properties are defined for band outlines, even if transparent
            line_properties_band = go.scatter.Line(color="rgba(0, 0, 0, 0)")

            fig.add_traces(
                [
                    go.Scatter(
                        x=df[x_col],
                        y=df[col_name].apply(lambda b: b[1]),  # Corrected to apply on column
                        mode="lines",
                        line=line_properties_band,
                        showlegend=False,  # Don't show legend for the invisible upper line
                    ),
                    go.Scatter(
                        x=df[x_col],
                        y=df[col_name].apply(lambda b: b[0]),  # Corrected to apply on column
                        mode="lines",
                        line=line_properties_band,
                        fill="tonexty",
                        fillcolor=band_colors[i],
                        showlegend=True,
                    ),
                ]
            )

            # Assign legend name to the filled band trace
            # The last added trace (index len(fig["data"]) - 1) is the filled lower bound trace
            fig["data"][len(fig["data"]) - 1]["name"] = col_name  # Use the column name as legend

    if band_cols_dict is not None:
        if band_colors is None:
            band_colors = get_distinct_colors(num_colors=len(band_cols_dict), opacity=0.2)

        for i, name in enumerate(band_cols_dict):
            col1 = band_cols_dict[name][0]
            col2 = band_cols_dict[name][1]
            # Ensure line properties are defined for band outlines, even if transparent
            line_properties_band = go.scatter.Line(color="rgba(0, 0, 0, 0)")

            fig.add_traces(
                [
                    go.Scatter(
                        x=df[x_col],
                        y=df[col2],
                        mode="lines",
                        line=line_properties_band,
                        showlegend=False,  # Don't show legend for the invisible upper line
                    ),
                    go.Scatter(
                        x=df[x_col],
                        y=df[col1],
                        mode="lines",
                        line=line_properties_band,
                        fill="tonexty",
                        fillcolor=band_colors[i],
                        showlegend=True,
                    ),
                ]
            )
            # Assign legend name to the filled band trace
            # The last added trace (index len(fig["data"]) - 1) is the filled lower bound trace
            fig["data"][len(fig["data"]) - 1]["name"] = name

    fig.update_layout(title=title)
    return fig


def plot_long_df(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_by_cols: Optional[List[str]] = None,
    line_colors: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> Dict[str, Union[go.Figure, pd.DataFrame, List[str]]]:
    """Plots data from a "long" DataFrame, transforming it to a "wide" format.

    This function prepares your data using `prep_df_for_grouped_plot` and then
    generates a Plotly figure where `y_col` values are plotted against `x_col`.
    If `group_by_cols` are provided, a separate line is drawn for each unique
    combination of these grouping columns.

    Args:
        df (pd.DataFrame): The input DataFrame in a "long" format.
        x_col (str): The name of the column to be used as the x-axis.
        y_col (str): The name of the column to be used as the y-axis (values).
        group_by_cols (Optional[List[str]]): A list of column names to group by.
            Each unique combination of these columns will result in a separate line.
            If None, a single line representing `y_col` vs `x_col` is plotted.
        line_colors (Optional[List[str]]): A list of color strings to use for each
            line. If None, `plot_lines_markers` will use default Plotly colors.
        title (Optional[str]): The plot title.

    Returns:
        Dict[str, Union[go.Figure, pd.DataFrame, List[str]]]: A dictionary containing:
            'fig' (go.Figure): The generated Plotly interactive figure.
            'df' (pd.DataFrame): The transformed DataFrame (in "wide" format)
                used for plotting.
            'y_cols' (List[str]): A list of the names of the columns that were
                plotted as lines.
    """

    prepared_data = prep_df_for_grouped_plot(
        df=df, x_col=x_col, y_col=y_col, group_by_cols=group_by_cols
    )
    wide_df = prepared_data["df"]
    y_cols = prepared_data["y_cols"]

    # Pass the wide_df to plot_lines_markers, as it contains the pivoted data
    fig = plot_lines_markers(
        df=wide_df,
        x_col=x_col,
        line_cols=y_cols,  # These are the new columns created by prep_df_for_grouped_plot
        line_colors=line_colors,
        title=title,
    )

    return {"fig": fig, "df": wide_df, "y_cols": y_cols}

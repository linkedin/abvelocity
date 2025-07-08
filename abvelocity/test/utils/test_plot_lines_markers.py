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

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # Added for type checking and assertions on Plotly objects
import pytest

from abvelocity.utils.color_utils import get_distinct_colors
from abvelocity.utils.plot_lines_markers import plot_lines_markers, plot_long_df


def test_plot_lines_markers():
    """Tests `plot_lines_markers`."""
    df = pd.DataFrame(
        {
            "ts": [1, 2, 3, 4, 5, 6],
            "line1": [3, 4, 5, 6, 7, 7],
            "line2": [4, 5, 6, 7, 8, 8],
            "marker1": [np.nan, np.nan, 5, 6, np.nan, np.nan],
            "marker2": [np.nan, 5, 6, np.nan, np.nan, 8],
        }
    )

    fig = plot_lines_markers(
        df=df,
        x_col="ts",
        line_cols=["line1", "line2"],
        marker_cols=["marker1", "marker2"],
        line_colors=None,
        marker_colors=None,
    )

    assert len(fig.data) == 4
    assert fig.data[0].line.color is None
    assert fig.data[1].line.color is None
    assert fig.data[2].marker.color is None
    assert fig.data[3].marker.color is None

    # Next we make the marker and line colors consistent
    marker_colors = get_distinct_colors(num_colors=2, opacity=1.0)

    line_colors = get_distinct_colors(num_colors=2, opacity=0.5)

    fig = plot_lines_markers(
        df=df,
        x_col="ts",
        line_cols=["line1", "line2"],
        marker_cols=["marker1", "marker2"],
        line_colors=line_colors,
        marker_colors=marker_colors,
    )

    assert len(fig.data) == 4

    # Length of `line_colors` must be larger than or equal to length of `line_cols` if passed.
    with pytest.raises(ValueError, match="If `line_colors` is passed"):
        plot_lines_markers(
            df=df,
            x_col="ts",
            line_cols=["line1", "line2"],
            marker_cols=["marker1", "marker2"],
            line_colors=line_colors[:1],
            marker_colors=marker_colors,
        )

    # At least one of `line_cols` or `marker_cols` or `band_cols`
    # must be provided (not None).
    with pytest.raises(ValueError, match="At least one of"):
        plot_lines_markers(
            df=df,
            x_col="ts",
            line_cols=None,
            marker_cols=None,
            band_cols=None,
            line_colors=None,
            marker_colors=None,
        )


def test_plot_lines_markers_with_bands():
    """Tests `plot_lines_markers` with bands."""
    df = pd.DataFrame(
        {
            "x": range(4),
            "y": range(4),
            "z1": range(1, 5),
            "z2": range(-1, 3),
            "w": [(0, 1), (1, 3), (1, 5), (3, 5)],
            "u": [(2, 3), (3, 3), (4, 4), (6, 8)],
        }
    )

    fig = plot_lines_markers(df=df, x_col="x", line_cols=["y", "z1"], band_cols=["u", "w"])

    assert len(fig.data) == 6
    assert fig.data[0].line.color is None
    assert fig.data[1].line.color is None
    assert fig.data[2].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[3].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[4].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[5].line.color == "rgba(0, 0, 0, 0)"

    # Names of the band traces are now the column names
    assert fig.data[3].name == "u"
    assert fig.data[5].name == "w"

    # Dynamically get the default colors from get_distinct_colors
    default_band_colors = get_distinct_colors(num_colors=2, opacity=0.2)
    assert fig.data[3].fillcolor == default_band_colors[0]
    assert fig.data[5].fillcolor == default_band_colors[1]
    assert fig.layout.title.text is None

    # Bands with custom colors and a title for the plot.
    fig = plot_lines_markers(
        df=df,
        x_col="x",
        line_cols=["y", "z1"],
        band_cols=["u", "w"],
        band_colors=["rgba(0, 255, 0, 0.2)", "rgba(255, 0, 0, 0.2)"],
        title="custom band colors",
    )

    assert len(fig.data) == 6
    assert fig.data[0].line.color is None
    assert fig.data[1].line.color is None
    assert fig.data[2].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[3].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[4].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[5].line.color == "rgba(0, 0, 0, 0)"

    # Names of the band traces are now the column names
    assert fig.data[3].name == "u"
    assert fig.data[5].name == "w"

    assert fig.data[3].fillcolor == "rgba(0, 255, 0, 0.2)"
    assert fig.data[5].fillcolor == "rgba(255, 0, 0, 0.2)"
    assert fig.layout.title.text == "custom band colors"

    # Bands specified by dictionary.
    df = pd.DataFrame(
        {
            "x": range(4),
            "y": [2, 3, 4, 5],
            "z1": [4, 5, 6, 8],
            "z2": range(-1, 3),
            "w1": [5, 6, 6, 8],
            "w2": [7, 8, 9, 9],
            "u1": [2, 3, 5, 7],
            "u3": [4, 5, 8, 8],
        }
    )

    fig = plot_lines_markers(
        df=df,
        x_col="x",
        line_cols=["y", "z1"],
        band_cols_dict={"u": ["u1", "u3"], "w": ["w1", "w2"]},
        band_colors=["rgba(0, 255, 0, 0.2)", "rgba(255, 0, 0, 0.2)"],
        title="bands via dict",
    )

    assert len(fig.data) == 6
    assert fig.data[0].line.color is None
    assert fig.data[1].line.color is None
    assert fig.data[2].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[3].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[4].line.color == "rgba(0, 0, 0, 0)"
    assert fig.data[5].line.color == "rgba(0, 0, 0, 0)"

    assert fig.data[3].name == "u"  # This should match the key from band_cols_dict
    assert fig.data[5].name == "w"  # This should match the key from band_cols_dict

    assert fig.data[3].fillcolor == "rgba(0, 255, 0, 0.2)"
    assert fig.data[5].fillcolor == "rgba(255, 0, 0, 0.2)"
    assert fig.layout.title.text == "bands via dict"


# --- Tests for plot_long_df function ---


def test_plot_long_df_no_grouping():
    """
    Test plot_long_df when no grouping columns are provided.
    """
    data = {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "sales": [100, 150, 120]}
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "sales"

    result = plot_long_df(df, x_col, y_col, group_by_cols=None)
    fig = result["fig"]
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    # Assert return types and basic structure
    assert isinstance(fig, go.Figure)
    assert isinstance(transformed_df, pd.DataFrame)
    assert isinstance(y_cols, list)

    # Assert transformed_df content
    expected_transformed_df = pd.DataFrame(
        {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "sales": [100, 150, 120]}
    )
    pd.testing.assert_frame_equal(transformed_df, expected_transformed_df)
    assert y_cols == ["sales"]

    # Assert Plotly figure content
    assert len(fig.data) == 1
    assert fig.data[0].name == "sales"
    assert list(fig.data[0].x) == df["date"].tolist()
    assert list(fig.data[0].y) == df["sales"].tolist()


def test_plot_long_df_single_group_by_col():
    """
    Test plot_long_df with a single grouping column.
    """
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
        "region": ["East", "West", "East", "West"],
        "sales": [100, 120, 150, 130],
    }
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "sales"
    group_by_cols = ["region"]

    result = plot_long_df(df, x_col, y_col, group_by_cols)
    fig = result["fig"]
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    assert isinstance(fig, go.Figure)
    assert isinstance(transformed_df, pd.DataFrame)
    assert isinstance(y_cols, list)

    # Expected transformed_df (pivoted)
    expected_transformed_df = pd.DataFrame(
        {"date": ["2023-01-01", "2023-01-02"], "East": [100.0, 150.0], "West": [120.0, 130.0]}
    )
    pd.testing.assert_frame_equal(transformed_df, expected_transformed_df)
    assert sorted(y_cols) == sorted(["East", "West"])  # Order might vary

    # Assert Plotly figure content (2 lines for 'East' and 'West')
    assert len(fig.data) == 2
    # Check if 'East' and 'West' are present as trace names and data
    trace_names = [trace.name for trace in fig.data]
    assert "East" in trace_names
    assert "West" in trace_names

    east_trace = next(trace for trace in fig.data if trace.name == "East")
    west_trace = next(trace for trace in fig.data if trace.name == "West")

    assert list(east_trace.x) == expected_transformed_df["date"].tolist()
    assert list(east_trace.y) == expected_transformed_df["East"].tolist()
    assert list(west_trace.x) == expected_transformed_df["date"].tolist()
    assert list(west_trace.y) == expected_transformed_df["West"].tolist()


def test_plot_long_df_multiple_group_by_cols():
    """
    Test plot_long_df with multiple grouping columns.
    """
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"],
        "region": ["East", "East", "West", "West"],
        "product": ["A", "B", "A", "B"],
        "sales": [10, 20, 15, 25],
    }
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "sales"
    group_by_cols = ["region", "product"]

    result = plot_long_df(df, x_col, y_col, group_by_cols)
    fig = result["fig"]
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    assert isinstance(fig, go.Figure)
    assert isinstance(transformed_df, pd.DataFrame)
    assert isinstance(y_cols, list)

    expected_transformed_df = pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "East_A": [10.0],
            "East_B": [20.0],
            "West_A": [15.0],
            "West_B": [25.0],
        }
    )
    pd.testing.assert_frame_equal(transformed_df, expected_transformed_df)
    assert sorted(y_cols) == sorted(["East_A", "East_B", "West_A", "West_B"])

    assert len(fig.data) == 4
    trace_names = [trace.name for trace in fig.data]
    assert sorted(trace_names) == sorted(y_cols)


def test_plot_long_df_with_colors_and_title():
    """
    Test plot_long_df with custom line colors and title.
    """
    data = {"date": ["2023-01-01", "2023-01-02"], "region": ["North", "South"], "sales": [50, 60]}
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "sales"
    group_by_cols = ["region"]
    custom_colors = ["red", "blue"]
    custom_title = "Sales by Region"

    result = plot_long_df(
        df, x_col, y_col, group_by_cols, line_colors=custom_colors, title=custom_title
    )
    fig = result["fig"]
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    assert transformed_df is not None
    assert y_cols is not None

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == custom_title

    # Check that colors are applied correctly
    # Note: the order of traces might vary due to pivoting, so check colors by name
    trace_names = [trace.name for trace in fig.data]
    assert "North" in trace_names
    assert "South" in trace_names

    north_trace = next(trace for trace in fig.data if trace.name == "North")
    south_trace = next(trace for trace in fig.data if trace.name == "South")

    # Assuming 'North' gets the first color and 'South' the second, or vice-versa
    # Need to verify how get_distinct_colors assigns if order is not fixed.
    # For now, check if colors are among the provided custom colors.
    assert north_trace.line.color in custom_colors
    assert south_trace.line.color in custom_colors
    assert north_trace.line.color != south_trace.line.color  # Ensure different colors are used


def test_plot_long_df_nan_values_in_y_col():
    """
    Test plot_long_df handles NaN values correctly, including shape consistency.
    """
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
        "category": ["A", "B", "A", "B"],
        "value": [10.0, np.nan, 20.0, 30.0],
    }
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "value"
    group_by_cols = ["category"]

    result = plot_long_df(df, x_col, y_col, group_by_cols)
    fig = result["fig"]
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    assert isinstance(fig, go.Figure)
    assert isinstance(transformed_df, pd.DataFrame)
    assert isinstance(y_cols, list)

    expected_transformed_df = pd.DataFrame(
        {"date": ["2023-01-01", "2023-01-02"], "A": [10.0, 20.0], "B": [np.nan, 30.0]}
    )
    pd.testing.assert_frame_equal(transformed_df, expected_transformed_df, check_dtype=False)
    assert sorted(y_cols) == sorted(["A", "B"])

    assert len(fig.data) == 2  # Expect two lines, one for 'A' and one for 'B'

    trace_A = next(trace for trace in fig.data if trace.name == "A")
    trace_B = next(trace for trace in fig.data if trace.name == "B")

    assert list(trace_A.x) == expected_transformed_df["date"].tolist()
    assert list(trace_A.y) == expected_transformed_df["A"].tolist()
    # Use np.testing.assert_array_equal for robust NaN comparison
    np.testing.assert_array_equal(list(trace_B.y), expected_transformed_df["B"].tolist())

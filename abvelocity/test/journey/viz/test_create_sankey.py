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

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from abvelocity.journey.viz.create_sankey import create_sankey

# Create save path for html files (figures)
SAVE_PATH = Path(__file__).parents[6].joinpath("docs/static/test-results/journey/sankey/").resolve()
os.makedirs(SAVE_PATH, exist_ok=True)


def test_sankey_plot_with_color_dict_max_seq():
    """Test generating Sankey plot using max_seq_index and a custom node color dictionary."""
    data = {"s1": ["A", "A", "B"], "s2": ["X", "Y", "X"], "count": [50, 70, 30]}
    df = pd.DataFrame(data)
    color_dict = {"A": "blue", "B": "green", "X": "red", "Y": "purple"}

    fig = create_sankey(
        df,
        max_seq_index=2,
        value_col="count",
        color_dict=color_dict,
        title="Test Sankey: Max Seq with Node Colors",
    )
    assert isinstance(fig, go.Figure), "Should return a Plotly Figure object."
    save_path = SAVE_PATH / "test_sankey_max_seq_with_node_colors.html"
    fig.write_html(str(save_path))


def test_sankey_plot_with_incomplete_color_dict_path_cols():
    """Test Sankey with path_cols and an incomplete node color dictionary (fallback to defaults)."""
    data = {
        "step1": ["Start", "Start", "Mid"],
        "step2": ["Mid", "End", "End"],
        "value": [100, 50, 80],
    }
    df = pd.DataFrame(data)
    color_dict = {"Start": "lightblue", "Mid": "orange"}

    fig = create_sankey(
        df,
        path_cols=["step1", "step2"],
        value_col="value",
        color_dict=color_dict,
        title="Test Sankey: Path Cols with Incomplete Node Colors",
    )
    assert isinstance(fig, go.Figure)
    save_path = SAVE_PATH / "test_sankey_path_cols_incomplete_node_colors.html"
    fig.write_html(str(save_path))


def test_sankey_plot_without_color_dict():
    """Test generating Sankey plot using default node colors."""
    data = {
        "s1": ["Alpha", "Beta"],
        "s2": ["Gamma", "Delta"],
        "s3": ["Epsilon", "Zeta"],
        "users": [40, 60],
    }
    df = pd.DataFrame(data)

    fig = create_sankey(
        df, max_seq_index=3, value_col="users", title="Test Sankey: Default Node Colors"
    )
    assert isinstance(fig, go.Figure)
    save_path = SAVE_PATH / "test_sankey_default_node_colors.html"
    fig.write_html(str(save_path))


def test_sankey_plot_missing_path_column():
    """Test ValueError when a specified path column is missing."""
    data = {"s1": ["A"], "count": [10]}
    df = pd.DataFrame(data)

    try:
        create_sankey(df, max_seq_index=2, value_col="count")
        assert False, "Expected ValueError due to missing path column 's2'."
    except ValueError as e:
        assert "Missing required columns" in str(e) and "'s2'" in str(e), f"Unexpected error: {e}"


def test_sankey_plot_missing_value_column():
    """Test ValueError when the specified value_col is missing."""
    data = {"s1": ["A"], "s2": ["B"]}
    df = pd.DataFrame(data)

    try:
        create_sankey(df, max_seq_index=2, value_col="my_values")
        assert False, "Expected ValueError due to missing value column 'my_values'."
    except ValueError as e:
        assert "Missing required columns" in str(e) and "'my_values'" in str(
            e
        ), f"Unexpected error: {e}"


def test_sankey_no_path_definition():
    """Test ValueError if neither path_cols nor max_seq_index is given."""
    df = pd.DataFrame({"s1": ["A"], "s2": ["B"], "count": [10]})
    try:
        create_sankey(df, value_col="count")
        assert False, "Expected ValueError due to no path definition."
    except ValueError as e:
        assert "Either 'path_cols' or 'max_seq_index' must be provided" in str(
            e
        ), f"Unexpected error: {e}"


def test_sankey_insufficient_path_cols():
    """Test ValueError if path_cols has fewer than 2 columns."""
    df = pd.DataFrame({"s1": ["A"], "count": [10]})
    try:
        create_sankey(df, path_cols=["s1"], value_col="count")
        assert False, "Expected ValueError due to insufficient path_cols."
    except ValueError as e:
        assert "At least two path columns are required" in str(e), f"Unexpected error: {e}"


def test_sankey_orientation_and_suffix():
    """Test with custom default_link_color, orientation, and value_suffix."""
    data = {
        "categoryA": ["X1", "X1", "X2"],
        "categoryB": ["Y1", "Y2", "Y1"],
        "amount": [200, 150, 100],
    }
    df = pd.DataFrame(data)
    fig = create_sankey(
        df,
        path_cols=["categoryA", "categoryB"],
        value_col="amount",
        orientation="v",
        value_suffix=" units",
        title="Test Sankey: Custom Styling",
    )
    assert isinstance(fig, go.Figure)
    assert fig.data[0].orientation == "v"
    assert fig.data[0].valuesuffix == " units"
    save_path = SAVE_PATH / "test_sankey_custom_styling.html"
    fig.write_html(str(save_path))


def test_sankey_with_nans_in_paths():
    """Test Sankey plot where some paths terminate early due to NaNs."""
    data = {
        "step1": ["Homepage", "Homepage", "Product Page", "Homepage", "Homepage"],
        "step2": ["Product Page", "About Us", None, "Product Page", "Blog"],
        "step3": ["Checkout", None, None, "Cart", None],
        "users": [50, 5, 20, 30, 15],
    }
    df = pd.DataFrame(data)
    fig = create_sankey(
        df,
        path_cols=["step1", "step2", "step3"],
        value_col="users",
        title="Test Sankey: Paths with NaNs (Early Exits)",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].node.label) > 0
    assert len(fig.data[0].link.source) > 0
    save_path = SAVE_PATH / "test_sankey_paths_with_nans.html"
    fig.write_html(str(save_path))


def test_sankey_complex_5_stages():
    """Test a more complex Sankey plot with 5 stages, branching, and merging."""
    data = {
        "s1": ["StartA", "StartA", "StartB", "StartA", "StartB", "StartC"],
        "s2": ["MidA1", "MidA2", "MidB1", "MidA1", "MidB1", "MidC1"],
        "s3": ["MidA1-X", "MidA2-Y", "MidB1-X", "MidA1-X", "MidB1-Y", "MidC1-Z"],
        "s4": ["EndP", "EndQ", "EndP", "EndR", "EndQ", "EndS"],
        "s5": ["FinalX", "FinalY", "FinalX", "FinalZ", "FinalY", "FinalX"],
        "volume": [10, 15, 20, 5, 25, 30],
    }
    df = pd.DataFrame(data)
    color_dict = {
        "StartA": "red",
        "StartB": "blue",
        "StartC": "green",
        "MidA1": "lightcoral",
        "MidA2": "lightskyblue",
        "MidB1": "lightblue",
        "MidC1": "lightgreen",
        "MidA1-X": "salmon",
        "MidA2-Y": "deepskyblue",
        "MidB1-X": "dodgerblue",
        "MidB1-Y": "royalblue",
        "MidC1-Z": "lime",
        "EndP": "orange",
        "EndQ": "gold",
        "EndR": "yellow",
        "EndS": "khaki",
        "FinalX": "purple",
        "FinalY": "magenta",
        "FinalZ": "orchid",
    }
    fig = create_sankey(
        df,
        max_seq_index=5,
        value_col="volume",
        color_dict=color_dict,
        title="Test Sankey: Complex 5-Stage Flow",
    )
    assert isinstance(
        fig, go.Figure
    ), "Should return a Plotly Figure object for complex 5-stage test."
    save_path = SAVE_PATH / "test_sankey_complex_5_stages.html"
    fig.write_html(str(save_path))

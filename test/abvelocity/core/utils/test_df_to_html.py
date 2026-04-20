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
import pytest
from abvelocity.core.utils.df_to_html import df_to_html, to_html_format_bg

WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/html_example/")
os.makedirs(WRITE_PATH, exist_ok=True)


def test_df_to_html():
    # Create a simple dataframe
    df = pd.DataFrame({"A": [1, 2897, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    # Convert the dataframe to html
    html_str = df_to_html(df=df, file_name=f"{WRITE_PATH}/test_df_to_html.html")

    # Check if the html string is as expected
    # assert '<table border="1" class="dataframe">' in html_str
    assert "<caption> <big> <strong>  </strong> </big> </caption> <thead>" in html_str
    assert "2897" in html_str


def test_df_to_html_basic_case():
    """Test with a basic DataFrame and bg_colors applied to all columns."""
    df = pd.DataFrame({"name": [1, 2, 3], "value": ["yes", "no", "no"]})
    bg_colors = ["green", "red", "green"]
    html_str = to_html_format_bg(df=df, bg_colors=bg_colors)

    # Check if the HTML contains the expected color
    assert "background-color: green" in html_str
    assert "background-color: red" in html_str


def test_df_to_html_specific_cols():
    """Test with bg_colors applied only to specific columns.
    Best way is to run this and look at the results visually.
    """
    df = pd.DataFrame({"name": [1, 2, 3], "value": ["yes", "no", "no"]})
    bg_colors = ["green", "red", "green"]
    bg_cols = ["value"]

    html_str = to_html_format_bg(df=df, bg_colors=bg_colors, bg_cols=bg_cols)

    assert "background-color: green" in html_str
    assert "background-color: red" in html_str


def test_df_to_html_color_mismatch():
    """Test if ValueError is raised when bg_colors length does not match DataFrame rows."""
    df = pd.DataFrame({"name": [1, 2, 3], "value": ["yes", "no", "no"]})
    bg_colors = ["green", "red"]  # Incorrect length

    with pytest.raises(ValueError, match="bg_colors must be of the same length as data rows"):
        to_html_format_bg(df=df, bg_colors=bg_colors)


def test_df_to_html_no_columns_specified():
    """Test when no format columns are specified (should format all columns)."""
    df = pd.DataFrame({"name": [1, 2, 3], "value": ["yes", "no", "no"]})
    bg_colors = ["blue", "yellow", "pink"]

    html_str = to_html_format_bg(df=df, bg_colors=bg_colors)

    # Check that all columns have color formatting
    assert "background-color: blue" in html_str
    assert "background-color: yellow" in html_str
    assert "background-color: pink" in html_str


def test_df_to_html_p_value():
    df_name = "Reza's df"
    df = pd.DataFrame({"delta": [1, 2, 3, 4, -5], "y": [2, 3, 4, 5, 6], "p_value": [0.02, 0.1, 0.1, 0.02, 0.01]})

    bg_colors = []
    bg_cols = ["delta", "p_value"]

    for row in df.itertuples():
        if row.p_value < 0.05:
            if row.delta >= 0.0:
                bg_colors.append("rgba(0, 250, 0, 0.5)")  # Light green
            else:
                bg_colors.append("rgba(250, 0, 0, 0.5)")  # Light red
        else:
            bg_colors.append(None)

    bg_colors = tuple(bg_colors)

    html_str = df_to_html(
        df=df,
        top_paragraphs=["x", df_name],
        caption=f"x, {df_name}",
        bg_colors=bg_colors,
        bg_cols=bg_cols,
        file_name=f"{WRITE_PATH}/test_df_to_html_p_value.html",
    )
    assert "delta" in html_str

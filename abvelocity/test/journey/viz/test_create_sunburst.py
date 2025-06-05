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

from abvelocity.journey.viz.create_sunburst import create_sunburst

# Create save path for html files (figures)
SAVE_PATH = (
    Path(__file__).parents[6].joinpath("docs/static/test-results/journey/sunburst/").resolve()
)
os.makedirs(SAVE_PATH, exist_ok=True)


def test_sunburst_plot_with_color_dict():
    """Test generating sunburst plot with a custom color dictionary."""
    data = {"s1": ["start", "start"], "s2": ["click", "impression"], "count": [50, 70]}
    df = pd.DataFrame(data)
    color_dict = {"start": "blue", "click": "red", "impression": "green"}

    fig = create_sunburst(
        df, max_seq_index=2, value_col="count", color_dict=color_dict, title="Test Title 1"
    )
    assert isinstance(fig, go.Figure)
    save_path = SAVE_PATH / "test_sunburst_with_color_dict.html"
    fig.write_html(save_path)


def test_sunburst_plot_with_incomplete_color_dict():
    """Test generating sunburst plot with a custom color dictionary."""
    data = {
        "s1": ["start", "start", "start", "click"],
        "s2": ["click", "impression", "end", "end"],
        "s3": ["purchase", "end", "end", "end"],
        "count": [50, 70, 20, 10],
    }
    df = pd.DataFrame(data)

    color_dict = {
        "start": "lightblue",
        "click": "orange",
        "impression": "lightgreen",
        "purchase": "green",
    }

    fig = create_sunburst(
        df, max_seq_index=3, value_col="count", color_dict=color_dict, title="Test Title 1"
    )
    assert isinstance(fig, go.Figure)
    save_path = SAVE_PATH / "test_sunburst_with_incomplete_color_dict.html"
    fig.write_html(save_path)


def test_sunburst_plot_without_color_dict():
    """Test generating sunburst plot without passing a color dictionary."""
    data = {"s1": ["start", "end"], "s2": ["click", "survey"], "count": [30, 20]}
    df = pd.DataFrame(data)

    fig = create_sunburst(df, max_seq_index=2, value_col="count", title="Test Title 2")
    assert isinstance(fig, go.Figure)
    save_path = SAVE_PATH / "test_sunburst_without_color_dict.html"
    fig.write_html(save_path)


def test_sunburst_plot_missing_columns():
    """Test for ValueError when required columns are missing."""
    data = {"s1": ["start", "end"], "count": [30, 20]}
    df = pd.DataFrame(data)

    try:
        create_sunburst(df, max_seq_index=2, value_col="count", title="Test Title 3")
        assert False, "Expected ValueError due to missing columns but did not get one."
    except ValueError as e:
        assert "Missing required columns" in str(e), f"Unexpected error message: {e}"

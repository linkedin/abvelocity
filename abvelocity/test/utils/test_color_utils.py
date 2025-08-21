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

import re

import plotly.graph_objects as go
import pytest

from abvelocity.utils.color_utils import generate_color_shades


def test_valid_hex_input():
    # Test with a valid hex color code
    shades = generate_color_shades("#FF0000", 3)
    assert len(shades) == 3
    for shade in shades:
        assert re.match(r"^#[0-9a-fA-F]{6}$", shade)  # Check hex format


def test_valid_named_input():
    # Test with a valid English color name
    shades = generate_color_shades("blue", 5)
    assert len(shades) == 5
    for shade in shades:
        assert re.match(r"^#[0-9a-fA-F]{6}$", shade)


def test_invalid_color_name():
    # Test with an invalid color name, expect ValueError
    with pytest.raises(ValueError) as excinfo:
        generate_color_shades("unknowncolor", 3)
    assert "Invalid color: 'unknowncolor'" in str(excinfo.value)


def test_invalid_hex_format():
    # Test with an invalid hex format, expect ValueError (due to regex mismatch)
    with pytest.raises(ValueError) as excinfo:
        generate_color_shades("#12345", 3)  # Invalid 5-digit hex
    assert "Invalid color: '#12345'" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        generate_color_shades("not_a_hex", 3)
    assert "Invalid color: 'not_a_hex'" in str(excinfo.value)


def test_n_colors_zero():
    # Test with n_colors = 0, should return an empty list
    shades = generate_color_shades("green", 0)
    assert len(shades) == 0
    assert shades == []


def test_n_colors_one():
    # Test with n_colors = 1, should return a list with one shade
    shades = generate_color_shades("yellow", 1)
    assert len(shades) == 1
    assert re.match(r"^#[0-9a-fA-F]{6}$", shades[0])


def test_multiple_shades_distinctness():
    # Test that shades are distinct (not identical) when n_colors > 1
    shades = generate_color_shades("purple", 5)
    assert len(shades) == 5
    # Convert to a set to check for uniqueness; if all are unique, set size equals list size
    assert len(set(shades)) == 5


def test_case_insensitivity_named_color():
    # Test that color names are case-insensitive
    shades_lower = generate_color_shades("red", 2)
    shades_upper = generate_color_shades("RED", 2)
    shades_mixed = generate_color_shades("rEd", 2)
    assert shades_lower == shades_upper
    assert shades_lower == shades_mixed


def test_short_hex_input():
    # Test with a short hex color code (e.g., #F00 for #FF0000)
    shades = generate_color_shades("#F00", 3)
    assert len(shades) == 3
    for shade in shades:
        assert re.match(r"^#[0-9a-fA-F]{6}$", shade)


def test_plot_generation():
    """
    Test to ensure a Plotly plot is generated and is not empty.
    This test generates a plot of color shades and checks for the existence
    of a Figure object and traces within it.
    """
    color_name = "purple"
    num_shades = 5
    shades = generate_color_shades(color_name, num_shades)

    # Create a Plotly figure
    fig = go.Figure()

    # Add a bar for each shade
    fig.add_trace(
        go.Bar(
            x=list(range(num_shades)),
            y=[1] * num_shades,  # All bars have the same height
            marker_color=shades,
            showlegend=False,
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        title=f"Shades of {color_name.capitalize()}",
        xaxis={"visible": False, "showticklabels": False},  # Hide x-axis
        yaxis={"visible": False, "showticklabels": False},  # Hide y-axis
        plot_bgcolor="white",  # Set background to white for clarity
        margin=dict(l=0, r=0, t=30, b=0),  # Adjust margins
    )

    # Assert that a Figure object was created
    assert fig is not None
    assert isinstance(fig, go.Figure)

    # Assert that the figure contains at least one trace (data series)
    assert len(fig.data) > 0
    assert isinstance(fig.data[0], go.Bar)

    # Note: Plotly figures are interactive and typically rendered in a browser or notebook.
    # We don't save them to a file for a simple 'non-empty' test, as that's more for
    # visual output verification than structural integrity.
    # If you wanted to view it, you could uncomment:
    # fig.show()

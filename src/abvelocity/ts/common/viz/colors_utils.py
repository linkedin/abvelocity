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
# original author: Sayan Patra, Kaixu Yang, Reza Hosseini
"""Color palette for plotting."""

import numpy as np
import plotly.colors as pc
from plotly.colors import DEFAULT_PLOTLY_COLORS, n_colors, validate_colors


def get_color_palette(num, colors=DEFAULT_PLOTLY_COLORS):
    """Returns ``num`` of distinct RGB colors.

    If ``num`` is less than or equal to the length of ``colors``, first ``num``
    elements of ``colors`` are returned.  Else ``num`` elements of colors are
    interpolated between the first and the last colors of ``colors``.

    Args:
        num: Number of colors required.  ``int``.
        colors: Which colors to use to build the color palette, default
            ``DEFAULT_PLOTLY_COLORS``.  This can be a list of RGB colors or a
            ``str`` from ``PLOTLY_SCALES``.  ``str`` or ``list`` of ``str``.

    Returns:
        color_palette: A list consisting of ``num`` RGB colors.  ``list``.
    """
    validate_colors(colors, colortype="rgb")
    if len(colors) == 1:
        return colors * num
    elif len(colors) >= num:
        color_palette = colors[0:num]
    else:
        color_palette = n_colors(colors[0], colors[-1], num, colortype="rgb")
    return color_palette


def get_distinct_colors(num_colors, opacity=0.95):
    """Gets ``num_colors`` most distinguishable colors.

    Uses color maps "tab10", "tab20" or "viridis" depending on the number of
    colors needed.  See above color palettes here:
    https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Args:
        num_colors: The number of colors needed.  ``int``.
        opacity: The opacity of the color, default 0.95.  This has to be a
            number between 0 and 1.  ``float``.

    Returns:
        colors: A list of string colors in RGB.  ``list`` of ``str``.
    """
    if opacity < 0 or opacity > 1:
        raise ValueError("Opacity must be between 0 and 1.")

    if num_colors <= 10:
        hex_colors = pc.qualitative.Plotly[:num_colors]
        rgb_tuples = [(int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)) for c in hex_colors]
    elif num_colors <= 24:
        hex_colors = pc.qualitative.Light24[:num_colors]
        rgb_tuples = [(int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)) for c in hex_colors]
    elif num_colors <= 256:
        rgb_strings = pc.sample_colorscale("Viridis", np.linspace(0, 1, num_colors))
        rgb_tuples = [pc.unlabel_rgb(c) for c in rgb_strings]
    else:
        raise ValueError("The maximum number of colors is 256.")

    return [f"rgba({int(r)}, {int(g)}, {int(b)}, {opacity})" for r, g, b in rgb_tuples]

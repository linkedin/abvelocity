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


import colorsys
import re

from plotly.colors import DEFAULT_PLOTLY_COLORS, n_colors, qualitative, sequential, validate_colors


def get_color_palette(num, colors=DEFAULT_PLOTLY_COLORS):
    """Returns ``num`` of distinct RGB colors.
    If ``num`` is less than or equal to the length of ``colors``, first ``num``
    elements of ``colors`` are returned.
    Else ``num`` elements of colors are interpolated between the first and the last
    colors of ``colors``.

    Args:
        num (int): Number of colors required.
        colors (str or list[str], optional): Which colors to use to build the color palette.
            This can be a list of RGB colors or a `str` from ``PLOTLY_SCALES``.
            Defaults to ``DEFAULT_PLOTLY_COLORS``.

    Returns:
        List: A list consisting ``num`` of RGB colors.
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
    Uses Plotly's qualitative color scales for smaller numbers of colors
    and a sequential color scale ("Viridis") for larger numbers.

    Args:
        num_colors (int): The number of colors needed.
        opacity (float, optional): The opacity of the color. This has to be a number between 0 and 1.
            Defaults to 0.95.

    Returns:
        list[str]: A list of string colors in RGBA format.

    Raises:
        ValueError: If opacity is not between 0 and 1.
        ValueError: If `num_colors` exceeds 256.
        ValueError: If an unsupported color format is encountered during parsing.
    """
    if opacity < 0 or opacity > 1:
        raise ValueError("Opacity must be between 0 and 1.")

    if num_colors <= 10:
        # Use Plotly's "Plotly" discrete color sequence, which is similar to tab10
        colors = qualitative.Plotly
    elif num_colors <= 20:
        # Use Plotly's "Safe" or "T10" which are 10-color scales,
        # we can combine or use "Paired" for 12 colors.
        # For exactly 20, we can interpolate two 10-color scales or use "Dark24"
        # "Plotly" + "D3" gives 20 distinct colors if we ensure no overlap.
        # Let's use "Light24" and truncate for simplicity if >= 20.
        colors = qualitative.Light24
    elif num_colors <= 256:
        # Use a sequential colormap like "Viridis" for a larger number of colors
        # n_colors directly provides the interpolated colors.
        # We need to specify the start and end colors for the Viridis scale.
        # plotly.colors.sequential.Viridis is a list of colors
        # so we take the first and last.
        colors = n_colors(
            sequential.Viridis[0], sequential.Viridis[-1], num_colors, colortype="rgb"
        )
    else:
        raise ValueError("The maximum number of colors is 256.")

    result = []
    for color in colors[:num_colors]:  # Ensure we don't go over num_colors if a palette has more
        # plotly colors are typically "rgb(R, G, B)" or "#RRGGBB"
        # We need to parse them and convert to rgba
        if color.startswith("rgb("):
            # Extract R, G, B values
            rgb_values = [int(x.strip()) for x in color[4:-1].split(",")]
            color_str = f"rgba({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]}, {opacity})"
        elif color.startswith("#"):
            # Convert hex to RGB by extracting substrings directly
            hex_color = color.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            rgb_values = (r, g, b)
            color_str = f"rgba({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]}, {opacity})"
        else:
            raise ValueError(f"Unsupported color format: {color}")
        result.append(color_str)

    return result


def generate_color_shades(color, n_colors):
    """
    Generate n_colors shades from a single color (hex or English name) by varying saturation and value.

    Args:
        color (str): Hex color code (e.g., "#800080") or English color name (e.g., "purple")
        n_colors (int): Number of shades to generate

    Returns:
        list: List of hex color codes
    """
    # Dictionary of common color names to hex codes
    COLOR_MAP = {
        "red": "#FF0000",
        "green": "#008000",
        "purple": "#800080",
        "blue": "#0000FF",
        "yellow": "#FFFF00",
        "pink": "#FF69B4",
        "black": "#000000",
        "white": "#FFFFFF",
        "gray": "#808080",
        "orange": "#FFA500",
        "brown": "#A52A2A",
        "cyan": "#00FFFF",
        "magenta": "#FF00FF",
    }

    # Check if color is a hex code
    hex_pattern = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
    if hex_pattern.match(color):
        hex_color = color
    else:
        # Convert color name to hex
        hex_color = COLOR_MAP.get(color.lower())
        if not hex_color:
            raise ValueError(
                f"Invalid color: '{color}'. Use a hex code (e.g., '#800080') or one of: {', '.join(COLOR_MAP.keys())}"
            )

    # Convert hex to RGB (normalized to 0-1)
    hex_color = hex_color.lstrip("#")

    # Expand 3-digit hex codes to 6-digit hex codes
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])

    # Explicitly parse R, G, B components
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    rgb = (r / 255.0, g / 255.0, b / 255.0)

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(*rgb)

    shades = []
    for i in range(n_colors):
        # Vary saturation (0.5 to 1.0) and value (0.6 to 1.0) for distinct shades
        # Ensure division by zero is handled for n_colors = 0 or 1
        s_var = 0.5 + (0.5 * i / max(n_colors, 1))  # Scale saturation
        v_var = 0.6 + (0.4 * i / max(n_colors, 1))  # Scale value

        rgb_var = colorsys.hsv_to_rgb(h, s_var, v_var)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb_var[0] * 255), int(rgb_var[1] * 255), int(rgb_var[2] * 255)
        )
        shades.append(hex_color)

    return shades

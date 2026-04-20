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

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go


def hist_with_quantiles(
    x: List[float],
    vertical_lines_dict: Optional[Dict[str, float]] = None,
    bands: Optional[List[float]] = None,
    title: str = "Histogram with Quantiles",
    font_size: int = 16,
) -> go.Figure:
    """
    Creates a histogram from an array of data with optional quantile lines,
    vertical lines for estimates, and band lines.

    Args:
        x: A list of numerical values to plot in the histogram.
        vertical_lines_dict: A mapping of labels to x-values for vertical lines.
        bands: A list or tuple containing [lower, upper] bounds for a CI band.
        title: The title of the generated plot.
        font_size: Base font size for titles and labels.

    Returns:
        A plotly.graph_objects.Figure object containing the stylized histogram.
    """
    x = np.array(x, dtype=float)
    if len(x) == 0 or np.isnan(x).any():
        raise ValueError("Array x must be non-empty and contain no NaNs")

    if vertical_lines_dict is not None:
        if not isinstance(vertical_lines_dict, dict):
            raise ValueError("vertical_lines_dict must be a dictionary")
        for key, value in vertical_lines_dict.items():
            if not isinstance(key, str):
                raise ValueError("All keys in 'vertical_lines_dict' must be strings.")
            if not isinstance(value, (int, float)) or np.isnan(value):
                raise ValueError(f"Value for '{key}' must be a valid number")

    if bands is not None:
        if not (isinstance(bands, (list, tuple)) and len(bands) == 2):
            raise ValueError("bands must be [lower, upper]")
        if bands[0] >= bands[1]:
            raise ValueError("lower bound must be less than upper bound")

    # 1. PRE-CALCULATE HISTOGRAM HEIGHT
    # We use 30 bins to match the go.Histogram default.
    # This lets us place markers at the top of the bars using data coordinates.
    counts, _ = np.histogram(x, bins=30)
    max_height = float(np.max(counts))
    marker_y = max_height * 1.05  # Place markers 5% above the highest bar

    quantile_lower = np.percentile(x, 2.5)
    quantile_upper = np.percentile(x, 97.5)

    fig = go.Figure()

    # Main Histogram
    fig.add_trace(go.Histogram(x=x, nbinsx=30, name="Simulation Data", opacity=0.6, marker_color="rgb(100, 149, 237)"))

    # Quantile Lines
    fig.add_vline(x=quantile_lower, line=dict(color="red", dash="dash", width=2))
    fig.add_vline(x=quantile_upper, line=dict(color="red", dash="dash", width=2))

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="95% Quantile Range",
        )
    )

    # CI Bands
    if bands is not None:
        for b_val in bands:
            fig.add_vline(x=b_val, line=dict(color="green", dash="dot", width=2))
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="green", dash="dot", width=2),
                name="CI Band",
            )
        )

    # Estimates with Markers
    if vertical_lines_dict is not None:
        estimate_colors = ["black", "darkorange", "purple", "brown", "deeppink"]
        dash_styles = ["dash", "dashdot", "dot", "longdash", "solid"]
        markers = ["diamond", "square", "cross", "x", "circle"]

        for i, (label, estimate_value) in enumerate(vertical_lines_dict.items()):
            color = estimate_colors[i % len(estimate_colors)]
            dash = dash_styles[i % len(dash_styles)]
            symbol = markers[i % len(markers)]

            curr_width = max(4 - (i * 0.5), 1.5)
            curr_marker_size = 14 - (i * 2)

            # Draw vertical line
            fig.add_vline(x=estimate_value, line=dict(color=color, width=curr_width, dash=dash))

            # Add markers using a Scatter trace on the data coordinates
            # This avoids the 'yref' and 'yaxis domain' string errors
            fig.add_trace(
                go.Scatter(
                    x=[estimate_value],
                    y=[marker_y],
                    mode="markers",
                    marker=dict(
                        symbol=symbol,
                        size=curr_marker_size,
                        color=color,
                        line=dict(width=1, color="white"),
                    ),
                    name=label,
                    line=dict(color=color, width=curr_width, dash=dash),
                    # If you want lines + markers in the legend, keep mode="lines+markers" below
                )
            )

    all_x = [quantile_lower, quantile_upper]
    if vertical_lines_dict:
        all_x.extend(list(vertical_lines_dict.values()))
    if bands:
        all_x.extend(bands)

    min_x, max_x = min(all_x), max(all_x)
    padding = (max_x - min_x) * 0.15
    if padding == 0:
        padding = 1.0
    fig.update_xaxes(range=[min_x - padding, max_x + padding])

    fig.update_layout(
        title=dict(text=title, font=dict(size=font_size + 4)),
        xaxis_title=dict(text="Value", font=dict(size=font_size)),
        yaxis_title=dict(text="Count", font=dict(size=font_size)),
        template="plotly_white",
        legend=dict(font=dict(size=font_size - 2)),
        barmode="overlay",
    )

    return fig

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
) -> go.Figure:
    """
    Creates a histogram from an array of data with optional quantile lines,
    vertical lines for estimates, and band lines.

    Args:
        x (List[float]): Array of simulation data for the histogram.
        vertical_lines_dict (Optional[Dict[str, float]]): Dictionary of estimates
            to display as vertical lines.  Keys are labels, values are the
            estimate values. If None, no vertical lines are plotted.
        bands (Optional[List[float]]): List of two floats representing the lower
            and upper bounds of a comparison band.
        title (str): Title of the plot.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If input data is invalid.
    """
    # Convert x to numpy array and validate
    x = np.array(x, dtype=float)
    if len(x) == 0 or np.isnan(x).any():
        raise ValueError("Array x must be non-empty and contain no NaNs")

    # Validate `vertical_lines_dict`
    if vertical_lines_dict is not None:
        if not isinstance(vertical_lines_dict, dict):
            raise ValueError("vertical_lines_dict must be a dictionary")
        for key, value in vertical_lines_dict.items():
            if not isinstance(key, str):
                raise ValueError("All keys in 'vertical_lines_dict' dictionary must be strings.")
            if not isinstance(value, (int, float)) or np.isnan(value):
                raise ValueError(
                    f"Value for key '{key}' in 'vertical_lines_dict' must be a valid number"
                )

    # Validate bands
    if bands is not None:
        if not (
            isinstance(bands, (list, tuple))
            and len(bands) == 2
            and all(isinstance(v, (int, float)) for v in bands)
        ):
            raise ValueError("bands must be a list or tuple of two numbers [lower, upper]")
        if bands[0] >= bands[1]:
            raise ValueError("bands lower bound must be less than upper bound")

    # Calculate 95% quantile range (2.5th and 97.5th percentiles)
    quantile_lower = np.percentile(x, 2.5)
    quantile_upper = np.percentile(x, 97.5)

    # Create histogram
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=x,
            nbinsx=30,  # Adjust number of bins as needed
            name="Simulation Data",
            opacity=0.75,
        )
    )

    # Add vertical lines for 95% quantile range
    max_height = np.histogram(x, bins=30)[0].max() * 1.1  # Extend lines above histogram
    fig.add_trace(
        go.Scatter(
            x=[quantile_lower, quantile_lower],
            y=[0, max_height],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Quantile Lower (2.5%)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[quantile_upper, quantile_upper],
            y=[0, max_height],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Quantile Upper (97.5%)",
        )
    )

    # Add vertical lines for optional bands, if provided
    if bands is not None:
        band_lower, band_upper = bands
        fig.add_trace(
            go.Scatter(
                x=[band_lower, band_lower],
                y=[0, max_height],
                mode="lines",
                line=dict(color="green", dash="dot", width=3),
                name="Band Lower",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[band_upper, band_upper],
                y=[0, max_height],
                mode="lines",
                line=dict(color="green", dash="dot", width=3),
                name="Band Upper",
            )
        )

    # Add vertical lines for estimates from the dictionary
    if vertical_lines_dict is not None:
        estimate_colors = [
            "blue",
            "orange",
            "purple",
            "brown",
            "pink",
            "cyan",
            "magenta",
            "gray",
        ]  # Add more colors if needed
        for i, (label, estimate_value) in enumerate(vertical_lines_dict.items()):
            color = estimate_colors[i % len(estimate_colors)]  # Cycle through colors
            fig.add_trace(
                go.Scatter(
                    x=[estimate_value, estimate_value],
                    y=[0, max_height],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=label,  # Use the key as the label
                )
            )

        # Adjust x-axis range to include all lines, handling negative values
        all_x = [quantile_lower, quantile_upper] + list(vertical_lines_dict.values())
        if bands is not None:
            all_x.extend(bands)
        min_x = min(all_x)
        max_x = max(all_x)
        padding = (max_x - min_x) * 0.1  # 10% of range as padding
        if padding == 0:  # Handle case where all values are identical
            padding = abs(min_x) * 0.1 or 1.0  # Use 10% of min_x or default to 1
        fig.update_xaxes(range=[min_x - padding, max_x + padding])

    else:
        # Adjust x-axis range to include all lines, handling negative values
        all_x = [quantile_lower, quantile_upper]
        if bands is not None:
            all_x.extend(bands)
        min_x = min(all_x)
        max_x = max(all_x)
        padding = (max_x - min_x) * 0.1  # 10% of range as padding
        if padding == 0:  # Handle case where all values are identical
            padding = abs(min_x) * 0.1 or 1.0  # Use 10% of min_x or default to 1
        fig.update_xaxes(range=[min_x - padding, max_x + padding])

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        showlegend=True,
        barmode="overlay",
    )

    return fig

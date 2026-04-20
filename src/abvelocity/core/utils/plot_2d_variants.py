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

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

NAN_VALUE = "nan"


def plot_2d_variants(
    df: pd.DataFrame,
    variant_col: str = "variant",
    count_column: str = "variant_count",
    title: str = "2D Partition of Space",
    dim_names: Optional[list[str]] = None,
    display_numbers: bool = True,
    num_digits: int = 2,
    legend: bool = False,
) -> go.Figure:
    """
    Create an interactive bubble plot partitioning a 2D space based on two-dimensional categorical tuples using Plotly.

    Args:
        df: DataFrame with variant_col (tuple) and count_column.
        variant_col: Column name for the label we are counting (default: 'variant')
        count_column: Column name for bubble size (default: 'variant_count').
        title: Plot title.
        dim_names: Optional list to determine the label of x and y axis.
        display_numbers: If True, displays the actual count numbers inside the bubbles.
        num_digits: Number of decimal places to display for the numbers in circles.
        legend: If True, displays color scale legend and uses color gradient.
                If False, uses constant light yellow color for better number readability.

    Returns:
        go.Figure: Plotly figure object.
    """
    # Extract unique labels for each dimension, ensuring 'nan' comes first
    dim1_labels = sorted(
        set(x[0] for x in df[variant_col] if isinstance(x, tuple)),
        key=lambda x: (x != NAN_VALUE, x),
    )
    dim2_labels = sorted(
        set(x[1] for x in df[variant_col] if isinstance(x, tuple)),
        key=lambda x: (x != NAN_VALUE, x),
    )

    if not dim_names:
        dim_names = ["Dimension 1", "Dimension 2"]

    # Map labels to numeric coordinates for plotting
    dim1_map = {label: idx for idx, label in enumerate(dim1_labels)}
    dim2_map = {label: idx for idx, label in enumerate(dim2_labels)}

    # Prepare plot data
    x_coords = [dim1_map[v[0]] for v in df[variant_col]]
    y_coords = [dim2_map[v[1]] for v in df[variant_col]]
    sizes = df[count_column].values
    labels = [f"({v[0]}, {v[1]})" for v in df[variant_col]]

    # Scale sizes for better visualization (Plotly uses marker size, adjust as needed)
    max_count = max(sizes)
    scaled_sizes = [50 * (count / max_count) ** 0.5 for count in sizes]  # Square root scaling for better visibility

    # Prepare mode and text values
    mode = "markers+text" if display_numbers else "markers"
    # Formats with commas and the specified number of decimal places
    text_values = [f"{s:,.{num_digits}f}" for s in sizes] if display_numbers else labels

    # Create bubble plot
    fig = go.Figure()

    # Configure marker based on legend parameter
    if legend:
        marker_config = dict(
            size=scaled_sizes,
            sizemode="diameter",
            sizeref=max(scaled_sizes) / 50,
            color=sizes,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=count_column),
        )
    else:
        marker_config = dict(
            size=scaled_sizes,
            sizemode="diameter",
            sizeref=max(scaled_sizes) / 50,
            color="yellow",
            line=dict(color="goldenrod", width=1),
        )

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode=mode,
            marker=marker_config,
            text=text_values,
            textposition="middle center",
            textfont=dict(color="black") if not legend else None,
            customdata=df[count_column].values,
            hovertemplate=f"{variant_col}:" + " %{text}<br>Quantity: %{customdata:,.3f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title=dim_names[0],
            tickvals=list(range(len(dim1_labels))),
            ticktext=dim1_labels,
            range=[-0.5, len(dim1_labels) - 0.5],
        ),
        yaxis=dict(
            title=dim_names[1],
            tickvals=list(range(len(dim2_labels))),
            ticktext=dim2_labels,
            range=[-0.5, len(dim2_labels) - 0.5],
        ),
        showlegend=False,
        width=800,
        height=600,
    )

    return fig

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


import pandas as pd
import plotly.graph_objects as go


def plot_3d_variants(
    df: pd.DataFrame,
    count_column: str = "variant_count",
    title: str = "3D Partition of Variants by Count",
) -> go.Figure:
    """
    Create an interactive 3D bubble plot partitioning a 3D space based on three-dimensional categorical tuples using Plotly.

    Args:
        df: DataFrame with 'variant' (three-element tuple) and count_column.
        count_column: Column name for bubble size (default: 'variant_count').
        title: Plot title.

    Returns:
        go.Figure: Plotly figure object.
    """
    # Extract unique labels for each dimension, ensuring 'nan' comes first
    dim1_labels = sorted(
        set(x[0] for x in df["variant"] if isinstance(x, tuple) and len(x) == 3),
        key=lambda x: (x != "nan", x),
    )
    dim2_labels = sorted(
        set(x[1] for x in df["variant"] if isinstance(x, tuple) and len(x) == 3),
        key=lambda x: (x != "nan", x),
    )
    dim3_labels = sorted(
        set(x[2] for x in df["variant"] if isinstance(x, tuple) and len(x) == 3),
        key=lambda x: (x != "nan", x),
    )

    # Map labels to numeric coordinates for plotting
    dim1_map = {label: idx for idx, label in enumerate(dim1_labels)}
    dim2_map = {label: idx for idx, label in enumerate(dim2_labels)}
    dim3_map = {label: idx for idx, label in enumerate(dim3_labels)}

    # Prepare plot data
    x_coords = [dim1_map[v[0]] for v in df["variant"]]
    y_coords = [dim2_map[v[1]] for v in df["variant"]]
    z_coords = [dim3_map[v[2]] for v in df["variant"]]
    sizes = df[count_column].values
    labels = [f"({v[0]}, {v[1]}, {v[2]})" for v in df["variant"]]

    # Scale sizes for better visualization (Plotly uses marker size, adjust as needed)
    max_count = max(sizes)
    sizes = [
        50 * (count / max_count) ** 0.5 for count in sizes
    ]  # Square root scaling for better visibility

    # Create 3D bubble plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers",
            marker=dict(
                size=sizes,
                sizemode="diameter",
                sizeref=max(sizes) / 50,  # Adjust max bubble size
                color=sizes,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=count_column),
            ),
            text=labels,
            customdata=df[count_column].values,
            hovertemplate="Variant: %{text}<br>Count: %{customdata:,.0f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title="Dimension 1",
                tickvals=list(range(len(dim1_labels))),
                ticktext=dim1_labels,
                range=[-0.5, len(dim1_labels) - 0.5],
            ),
            yaxis=dict(
                title="Dimension 2",
                tickvals=list(range(len(dim2_labels))),
                ticktext=dim2_labels,
                range=[-0.5, len(dim2_labels) - 0.5],
            ),
            zaxis=dict(
                title="Dimension 3",
                tickvals=list(range(len(dim3_labels))),
                ticktext=dim3_labels,
                range=[-0.5, len(dim3_labels) - 0.5],
            ),
        ),
        showlegend=False,
        width=800,
        height=600,
    )

    return fig

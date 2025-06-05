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
import plotly.express as px
import plotly.graph_objects as go


def create_sunburst(
    df: pd.DataFrame,
    max_seq_index: int,
    value_col: str = "count",
    title: str = "Sequence Flow",
    color_dict: Optional[dict[str, str]] = None,
) -> go.Figure:
    """
    Creates a sunburst plot for sequence data using a hierarchical path.

    Args:
        df (pd.DataFrame): A DataFrame containing hierarchical step columns and a value column.
        max_seq_index (int): The maximum number of hierarchical steps to consider (e.g., 5 for `s1` to `s5`).
        value_col (str): The column name representing values to size the segments.
        title (str): The title for the sunburst plot.
        color_dict (Optional[Dict[str, str]]): A dictionary mapping labels to colors. If None, default colors are generated.

    Returns:
        go.Figure: The generated sunburst plot.
    """

    # Dynamically build required column list based on the specified max sequence index
    step_columns = [f"s{i + 1}" for i in range(max_seq_index)]
    required_columns = set(step_columns + [value_col])

    # Ensure required columns are present in the DataFrame
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

    # Create the Sunburst plot using Plotly Express
    fig = px.sunburst(
        df,
        path=step_columns,  # Define hierarchy levels
        values=value_col,  # Size of each segment
        title=title,
    )

    # If color_dict is not provided, assign default colors from Plotly's qualitative palette
    if color_dict is None:
        unique_labels = pd.unique(
            df[step_columns].values.ravel("K")
        )  # Flatten the array of step values
        unique_labels = [
            label for label in unique_labels if pd.notna(label)
        ]  # Remove any NaN values

        # Assign colors cyclically from Plotly's default color palette
        color_dict = {
            label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, label in enumerate(unique_labels)
        }

    # Generate marker colors for the plot segments based on the color mapping
    marker_colors = [color_dict.get(label, "lightgray") for label in fig.data[0].labels]

    # Ensure consistent colors for each action/event type across layers
    fig.update_traces(marker=dict(colors=marker_colors), leaf_opacity=1)

    # Customize the layout for better visuals
    fig.update_layout(
        title_font_size=16,
        legend_title_text="",
        margin=dict(t=30, l=10, r=10, b=10),  # Minimize margins
    )

    return fig

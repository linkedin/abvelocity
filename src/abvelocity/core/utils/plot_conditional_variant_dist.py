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

from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

_NAN_VALUE = "nan"
_NAN_COLOR = "#b0b0b0"
_NAN_OPACITY = 0.25
_NAN_PATTERN = "/"


def _sort_nan_last(values: List[str], nan_value: str = _NAN_VALUE) -> List[str]:
    """Sort a list of strings with nan last, others in alphabetical order."""
    return sorted(values, key=lambda v: (1 if v == nan_value else 0, v))


def _sort_keys_nan_last(keys: List[Tuple], nan_value: str = _NAN_VALUE) -> List[Tuple]:
    """Sort conditional key tuples with any nan-containing tuple last."""
    return sorted(keys, key=lambda k: (1 if any(v == nan_value for v in k) else 0, str(k)))


def plot_conditional_variant_dist(
    df: pd.DataFrame,
    variant_col: str = "variant",
    count_column: str = "variant_count",
    dim_names: Optional[list[str]] = None,
    nan_value: str = _NAN_VALUE,
    nan_color: str = _NAN_COLOR,
    nan_opacity: float = _NAN_OPACITY,
    nan_pattern: str = _NAN_PATTERN,
) -> Dict[int, go.Figure]:
    """
    Creates Plotly bar plots visualizing the conditional probability distribution of
    variant combinations for each dimension in a multi-experiment setup.

    For a variant tuple (V1, V2, V3), it plots the distribution of (V2, V3) conditional on V1=v1i.
    Returns one plot for each experiment dimension (V1, V2, V3). X-axis labels are formatted
    with a placeholder dot (e.g., (v1, ., v3)) to indicate the fixed dimension.

    nan arms (not-triggered users) are sorted last in both the legend and the x-axis,
    rendered in grey at reduced opacity to signal "not a real arm — do not focus here".

    Args:
        df: DataFrame with variant_col (tuple) and count_column.
        variant_col: Column name for the variant label (tuple).
        count_column: Column name for the counts.
        dim_names: Optional list to label the fixed dimension in the plot titles.
        nan_value: Sentinel string for not-triggered variants. Default "nan".
        nan_color: Marker color for the nan trace. Default grey (#b0b0b0).
        nan_opacity: Opacity for the nan trace (0–1). Default 0.4.
        nan_pattern: Hatch pattern shape for the nan trace (Plotly marker.pattern.shape).
            Options: "/", "\\", "x", "-", "|", "+". Default "/".

    Returns:
        Dict[int, go.Figure]: A dictionary mapping the dimension index (1-based) to the Plotly figure object.
    """
    if df.empty:
        return {}

    # 1. Prepare Data and Determine Dimensions
    v0 = df[variant_col].iloc[0]
    if not isinstance(v0, tuple):
        df[variant_col] = df[variant_col].apply(lambda x: (x,))
        v0 = df[variant_col].iloc[0]

    num_expts = len(v0)
    if not dim_names:
        dim_names = [f"Dim {i+1}" for i in range(num_expts)]

    total_count = df[count_column].sum()
    df["percentage"] = df[count_column] / total_count

    # 2. Handle Univariate Case
    if num_expts == 1:
        fig = go.Figure()

        labels = [v[0] for v in df[variant_col]]
        percents = df["percentage"].values * 100.0

        fig.add_trace(go.Bar(x=labels, y=percents, marker_color="skyblue", name=dim_names[0]))

        fig.update_layout(
            title=f"Overall Distribution of {dim_names[0]}",
            xaxis_title=dim_names[0],
            yaxis_title="Percentage (%)",
            yaxis_tickformat=".2f",
            width=600,
            height=450,
        )
        return {1: fig}

    # 3. Handle Multivariate Case (Conditional Distribution)
    result_figs: Dict[int, go.Figure] = {}

    # Helper function to format the conditional key tuple into the (v1, ., v3) string
    def format_key_for_plot(cond_key_tuple, fixed_idx, n_expts):
        label_parts = []
        cond_idx = 0
        # Iterate through all experiment positions
        for k in range(n_expts):
            if k == fixed_idx:
                label_parts.append(".")
            else:
                # Pull from the conditional key tuple for non-fixed positions
                label_parts.append(str(cond_key_tuple[cond_idx]))
                cond_idx += 1
        return f"({', '.join(label_parts)})"

    # Iterate through each dimension to fix (the conditioning variable)
    for i in range(num_expts):
        fixed_dim_index = i
        fixed_dim_name = dim_names[fixed_dim_index]

        df["fixed_val"] = df[variant_col].apply(lambda x: x[fixed_dim_index])

        df["conditional_key"] = df[variant_col].apply(lambda x: tuple(x[j] for j in range(num_expts) if j != fixed_dim_index))

        fixed_totals = df.groupby("fixed_val")[count_column].sum()

        conditional_df = df.groupby(["fixed_val", "conditional_key"])[count_column].sum().reset_index(name="cond_count")

        conditional_df = conditional_df.merge(fixed_totals.rename("fixed_total"), on="fixed_val")

        conditional_df["prob"] = conditional_df["cond_count"] / conditional_df["fixed_total"] * 100.0

        # --- Plotting Setup ---
        fig = go.Figure()
        fixed_values = _sort_nan_last(list(conditional_df["fixed_val"].unique()), nan_value)
        all_conditional_keys = _sort_keys_nan_last(list(conditional_df["conditional_key"].unique()), nan_value)

        # Map the full domain of conditional keys to the plot data (for shared X-axis)
        shared_x_data = pd.DataFrame({"conditional_key": all_conditional_keys})

        # Prepare X-axis labels with the desired format (e.g., (v1, ., v3))
        x_labels = [format_key_for_plot(key, fixed_dim_index, num_expts) for key in shared_x_data["conditional_key"]]

        # --- Traces ---
        for fixed_val in fixed_values:
            # Filter data for the current fixed value
            subset = conditional_df[conditional_df["fixed_val"] == fixed_val]

            # Merge subset onto the shared X-axis domain, filling missing combinations with 0
            plot_data = shared_x_data.merge(subset[["conditional_key", "prob"]], on="conditional_key", how="left").fillna(0)

            # nan arm: grey color + reduced opacity to signal "not a real arm"
            is_nan_trace = fixed_val == nan_value
            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=plot_data["prob"],
                    name=f"{fixed_dim_name} = {fixed_val}",
                    marker_color=nan_color if is_nan_trace else None,
                    marker_pattern_shape=nan_pattern if is_nan_trace else "",
                    opacity=nan_opacity if is_nan_trace else 1.0,
                    hovertemplate="Other Variants: %{x}<br>Prob: %{y:.2f}%<extra></extra>",
                )
            )

        # --- Layout ---
        title = f"Conditional Distribution of Other Variants | {fixed_dim_name}"

        if num_expts == 2:
            conditional_label = dim_names[1 - i]
        else:
            other_dims = [dim_names[j] for j in range(num_expts) if j != fixed_dim_index]
            conditional_label = f"({', '.join(other_dims)})"

        fig.update_layout(
            title=title,
            xaxis_title=f"Other Dimensions: {conditional_label}",
            yaxis_title="Conditional Probability (%)",
            yaxis_tickformat=".2f",
            barmode="group",
            height=600,
            width=900,
            legend_title_text=f"Fixed {fixed_dim_name} Value",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        result_figs[fixed_dim_index + 1] = fig

    return result_figs

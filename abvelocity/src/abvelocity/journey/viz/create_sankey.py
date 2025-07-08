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

import pandas as pd
import plotly.colors  # For color conversion
import plotly.express as px  # For color palettes
import plotly.graph_objects as go
import webcolors


def to_rgba(color_str: str, alpha: float = 0.5) -> str:
    """
    Converts various color formats (named, hex, rgb, rgba) to rgba string
    with a specified alpha. Uses webcolors for names, plotly for hex.
    Handles existing rgba strings by attempting to replace their alpha.
    Provides a fallback grey color for invalid inputs.
    """
    default_fallback = f"rgba(200, 200, 200, {alpha})"
    if not isinstance(color_str, str):
        print(f"Warning: Input to to_rgba is not a string ('{color_str}'). Using fallback.")
        return default_fallback

    color_str_lower = color_str.lower().strip()  # Work with lowercase, stripped string

    # Check if it's already rgba
    if color_str_lower.startswith("rgba"):
        try:
            parts = color_str_lower.split(",")
            if len(parts) == 4:
                # Attempt to replace alpha, handle potential errors in format
                # Ensure the number parts are valid before joining
                r, g, b = map(lambda x: x.strip("rgba( )"), parts[:3])
                # Basic check if r,g,b look like numbers (optional but safer)
                int(r)
                int(g)
                int(b)
                return f"rgba({r}, {g}, {b}, {alpha})"
            else:
                print(f"Warning: Invalid rgba format '{color_str}'. Using fallback.")
                return default_fallback
        except Exception as e_rgba:
            print(f"Warning: Error parsing existing rgba '{color_str}': {e_rgba}. Using fallback.")
            return default_fallback

    # Check if it's rgb
    if color_str_lower.startswith("rgb"):
        try:
            # Extract numbers and add alpha
            parts = color_str_lower.strip("rgb() ").split(",")
            if len(parts) == 3:
                r, g, b = map(str.strip, parts)
                return f"rgba({r}, {g}, {b}, {alpha})"
            else:
                print(f"Warning: Invalid rgb format '{color_str}'. Using fallback.")
                return default_fallback
        except Exception as e_rgb:
            print(f"Warning: Error parsing existing rgb '{color_str}': {e_rgb}. Using fallback.")
            return default_fallback

    # Try converting from named color using webcolors
    try:
        rgb_tuple = webcolors.name_to_rgb(color_str_lower)  # Use lowercase name
        return f"rgba({rgb_tuple.red}, {rgb_tuple.green}, {rgb_tuple.blue}, {alpha})"
    except ValueError:  # Not a recognized name by webcolors
        pass  # Proceed to check if it's hex

    # Try converting from hex color using plotly
    try:
        # Plotly's hex_to_rgb handles '#' prefix optionally
        rgb_tuple_plotly = plotly.colors.hex_to_rgb(color_str)  # Use original case/format for hex
        return f"rgba({rgb_tuple_plotly[0]}, {rgb_tuple_plotly[1]}, {rgb_tuple_plotly[2]}, {alpha})"
    except (
        ValueError,
        TypeError,
        AttributeError,
    ):  # Plotly failed (not valid hex/name known to plotly)
        print(
            f"Warning: Could not parse color '{color_str}' as name, hex, rgb, or rgba. Using fallback."
        )
        return default_fallback
    except Exception as e_other:  # Catch any other unexpected error
        print(
            f"Warning: An unexpected error occurred parsing color '{color_str}': {e_other}. Using fallback."
        )
        return default_fallback


def create_sankey(
    df: pd.DataFrame,
    path_cols: Optional[List[str]] = None,
    max_seq_index: Optional[int] = None,
    value_col: str = "count",
    color_dict: Optional[Dict[str, str]] = None,  # Unified color dictionary
    link_alpha: float = 0.6,  # Transparency for link colors derived from source
    title: str = "Sankey Sequence Flow",
    pad: int = 15,
    thickness: int = 20,
    line_color: str = "black",
    line_width: float = 0.5,
    font_size: int = 10,
    font_color: str = "black",
    width: Optional[int] = None,
    height: Optional[int] = None,
    orientation: str = "h",
    value_format: str = ".0f",
    value_suffix: Optional[str] = None,
    node_hovertemplate: Optional[str] = None,
    link_hovertemplate: Optional[str] = None,
    add_end_state: bool = False,
    distinct_nodes_by_stage: bool = False,
    additional_dimensions: Optional[List[str]] = None,
) -> go.Figure:
    """
    Creates a Sankey plot from sequential path data in a DataFrame.
    Flows are generated between consecutive columns specified in `path_cols`
    or by `max_seq_index` (e.g., s1 -> s2, s2 -> s3). Node and Link colors
    are derived from the `color_dict`.

    Args:
        df (pd.DataFrame): DataFrame containing the path/sequence data.
        path_cols (Optional[List[str]]): Ordered list of column names defining sequence stages.
        max_seq_index (Optional[int]): Alternative way to define sequence stages as s1, s2, ...
        value_col (str): Column name for flow values. Defaults to "count".
        color_dict (Optional[Dict[str, str]]): Dictionary mapping node labels (str) to
             color strings (e.g., "blue", "#FF0000", "rgba(0,0,255,0.5)"). This dictionary
             is used for both node colors and deriving link colors (based on source node).
             If None or if a label is missing, default Plotly colors are used.
        link_alpha (float): The alpha transparency level (0.0 to 1.0) applied to link colors
             when they are derived from source node colors. Defaults to 0.6.
        title (str): The title for the Sankey plot.
        pad (int): Padding between nodes (px).
        thickness (int): Thickness of nodes (px).
        line_color (str): Color of the node border line.
        line_width (float): Width of the node border line.
        font_size (int): Font size for labels and title.
        font_color (str): Font color for labels and title.
        width (Optional[int]): Width of the figure (px).
        height (Optional[int]): Height of the figure (px).
        orientation (str): Diagram orientation ("h" or "v").
        value_format (str): Plotly format string for values.
        value_suffix (Optional[str]): Suffix for values in tooltips.
        node_hovertemplate (Optional[str]): Custom hovertemplate for nodes.
        link_hovertemplate (Optional[str]): Custom hovertemplate for links.
        add_end_state (bool): Parameter to indicate whether we need to add an "End" node to the sequence
        distinct_nodes_by_stage (bool): Parameter to indicate whether we need to append the sequence number for each node. Eg: If there are two sequences
                                        1. A -> B -> C -> D
                                        2. B -> A -> C -> D
                                        If true, these nodes would be named as A_s1, A_s2, B_s1, B_s2, C_s3, D_s4 )
                                        If false, all nodes will be represented with node names as it is ( A, B, C, D).
        additional_dimensions (Optional[List[str]]): Additional columns to group by in the Sankey plot.

    Returns:
        go.Figure: The generated Plotly Figure object.

    Raises:
        ValueError: If path definition is missing/invalid or required columns are missing.
    """

    # Determine path columns
    if path_cols is None and max_seq_index is None:
        raise ValueError("Either 'path_cols' or 'max_seq_index' must be provided.")
    if path_cols and max_seq_index is not None:
        print("Warning: Both 'path_cols' and 'max_seq_index' provided. 'path_cols' will be used.")

    current_path_cols: List[str]
    if path_cols is None:
        if not isinstance(max_seq_index, int) or max_seq_index <= 0:  # type: ignore
            raise ValueError("'max_seq_index' must be a positive integer.")
        current_path_cols = [f"s{i + 1}" for i in range(max_seq_index)]  # type: ignore
    else:
        current_path_cols = path_cols

    if len(current_path_cols) < 2:
        raise ValueError("At least two path columns are required to generate Sankey links.")

    # Validate columns
    required_df_cols = set(current_path_cols + [value_col])
    if additional_dimensions:
        required_df_cols.update(additional_dimensions)
    if not required_df_cols.issubset(df.columns):
        missing = required_df_cols - set(df.columns)
        raise ValueError(f"Missing required columns in DataFrame: {missing}")

    # Transform path data into Sankey links
    generated_source_labels: List[str] = []
    generated_target_labels: List[str] = []
    generated_flow_values: List[float] = []

    for i in range(len(current_path_cols) - 1):
        source_level_col = current_path_cols[i]
        target_level_col = current_path_cols[i + 1]
        level_df_filtered = df.dropna(subset=[source_level_col, target_level_col])
        if level_df_filtered.empty:
            continue
        current_stage_links = level_df_filtered.groupby(
            [source_level_col, target_level_col], as_index=False
        )[value_col].sum()
        src = current_stage_links[source_level_col].astype(str)
        tgt = current_stage_links[target_level_col].astype(str)

        if distinct_nodes_by_stage:
            generated_source_labels.extend((src + f"_s{i + 1}").tolist())
            generated_target_labels.extend((tgt + f"_s{i + 2}").tolist())
        else:
            generated_source_labels.extend(src.tolist())
            generated_target_labels.extend(tgt.tolist())

        generated_flow_values.extend(current_stage_links[value_col].astype(float).tolist())

    # For rows with only one step and no end-state or extra dimensions, emit a self-link so the node shows up
    if not add_end_state and not additional_dimensions:
        single_links = []
        for _, row in df.iterrows():
            # Pick out the non-null steps
            seen = [row[c] for c in current_path_cols if pd.notna(row[c])]
            # If only one step in the journey, append a tuple where source=target to create a loop
            if len(seen) == 1:
                src = str(seen[0])
                single_links.append((src, src, float(row[value_col])))

        # Aggregate single-step rows and append to the label list
        if single_links:
            single_df = pd.DataFrame(single_links, columns=["source", "target", value_col])
            agg = single_df.groupby(["source", "target"], as_index=False)[value_col].sum()
            for _, r in agg.iterrows():
                generated_source_labels.append(r["source"])
                generated_target_labels.append(r["target"])
                generated_flow_values.append(r[value_col])

    if add_end_state:
        leaf_sources = []
        leaf_values = []

        for _, row in df.iterrows():
            # Pick out the non-null stages in order
            seen = [row[c] for c in current_path_cols if pd.notna(row[c])]
            if not seen:
                continue

            last_label = str(seen[-1])
            if distinct_nodes_by_stage:
                stage_num = len(seen)
                # e.g. 2 if you saw s1 and s2
                last_label = f"{last_label}_s{stage_num}"

            leaf_sources.append(last_label)
            leaf_values.append(row[value_col])

        # Aggregate duplicates and append to your lists
        if leaf_sources:
            leaf_df = pd.DataFrame({"source": leaf_sources, value_col: leaf_values})
            agg = leaf_df.groupby("source", as_index=False)[value_col].sum()
            for _, r in agg.iterrows():
                generated_source_labels.append(r["source"])
                generated_target_labels.append("End")
                generated_flow_values.append(r[value_col])

    if additional_dimensions:
        extra_links = []
        for _, row in df.iterrows():
            # Find the last non-null event in this row
            if add_end_state:
                last_label = "End"
            else:
                seen = [row[c] for c in current_path_cols if pd.notna(row[c])]
                if not seen:
                    continue
                last_label = str(seen[-1])
                if distinct_nodes_by_stage:
                    last_label = f"{last_label}_s{len(seen)}"
            # For each requested dimension, link last_event -> dim_value
            for dim in additional_dimensions:
                val = row.get(dim)
                if pd.isna(val) or val is None:
                    continue
                extra_links.append((last_label, str(val), row[value_col]))

        updated_df = pd.DataFrame(extra_links, columns=["source", "target", value_col])
        agg = updated_df.groupby(["source", "target"], as_index=False)[value_col].sum()
        for _, row in agg.iterrows():
            generated_source_labels.append(row["source"])
            generated_target_labels.append(row["target"])
            generated_flow_values.append(row[value_col])

    # If no flows were generated, return an empty figure with a message
    if not generated_source_labels:
        empty_fig = go.Figure()
        empty_fig.update_layout(title_text=f"{title} (No data to display)")
        return empty_fig

    # Consolidate all unique node labels and map them
    combined_labels = pd.Series(generated_source_labels + generated_target_labels)
    all_unique_node_labels = sorted(list(pd.unique(combined_labels)))
    label_to_index_map = {label: idx for idx, label in enumerate(all_unique_node_labels)}
    source_indices = [label_to_index_map[label] for label in generated_source_labels]
    target_indices = [label_to_index_map[label] for label in generated_target_labels]

    # Node Colors (using color_dict)
    node_actual_colors: List[str] = []
    default_colors = px.colors.qualitative.Plotly
    for idx, label in enumerate(all_unique_node_labels):
        if distinct_nodes_by_stage and label.count("_s"):
            base = label.rsplit("_s", 1)[0]
        else:
            base = label
        if color_dict and base in color_dict:
            node_color = color_dict[base]
        else:
            node_color = default_colors[idx % len(default_colors)]
        node_actual_colors.append(node_color)

    # Link Colors (derived from source node color)
    # Always derive from source node color now, using the resolved node_actual_colors
    link_actual_colors: List[str] = [
        to_rgba(node_actual_colors[src_idx], alpha=link_alpha) for src_idx in source_indices
    ]

    # Define Node Properties for Plotly
    node_params = dict(
        pad=pad,
        thickness=thickness,
        line=dict(color=line_color, width=line_width),
        label=all_unique_node_labels,
        color=node_actual_colors,  # Use the fully resolved node colors
    )
    if node_hovertemplate:
        node_params["hovertemplate"] = node_hovertemplate
    else:
        node_params["hovertemplate"] = "<b>%{label}</b><br>Value: %{value}<extra></extra>"

    # Define Link Properties for Plotly
    link_params = dict(
        source=source_indices,
        target=target_indices,
        value=generated_flow_values,
        color=link_actual_colors,  # Use the derived link colors
    )
    if link_hovertemplate:
        link_params["hovertemplate"] = link_hovertemplate

    # Create Sankey Trace and Figure
    sankey_trace = go.Sankey(
        node=node_params,
        link=link_params,
        orientation=orientation,
        valueformat=value_format,
        valuesuffix=value_suffix,
    )

    fig = go.Figure(data=[sankey_trace])

    # Update Layout
    fig.update_layout(
        title_text=title,
        font=dict(size=font_size, color=font_color),
        width=width,
        height=height,
        margin=dict(t=max(30, len(title) * 0.7 + 20 if title else 30), l=10, r=10, b=10),
    )

    return fig

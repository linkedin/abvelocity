# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def plot_compare_variants(
    df: pd.DataFrame,
    x_col: str = "variant",
    y_col: str = "mean",
    split_col: Optional[str] = None,
    err_col: Optional[str] = None,
    sample_size_col: Optional[str] = None,
    ci_coverage: float = 0.95,
    variant_order: Optional[List] = None,
    title_prefix: str = "",
    height_per_plot: int = 600,
    width: int = 900,
    use_lines: bool = False,
) -> Dict[Union[str, int], go.Figure]:
    """Generic function to compare variants using Plotly.

    Categorical data results in a grouped bar chart, while numeric data
    results in a line plot with markers. If sample_size_col is provided,
    it creates a vertically stacked subplot showing sample counts.

    Args:
        df: Input DataFrame containing the data to plot.
        x_col: Column name to use for the x-axis (variants).
        y_col: Column name to use for the y-axis (values/means).
        split_col: Optional column name to split data into multiple plots.
        err_col: Column name for error values, or 'normal_ci' to compute confidence intervals.
        sample_size_col: Column name for sample sizes. If provided, creates a subplot below.
        ci_coverage: The confidence level to use (e.g., 0.95 for 95% CI).
        variant_order: Optional list to define the order of variants on the x-axis.
        title_prefix: String to prepend to the plot titles.
        height_per_plot: Height of each generated figure in pixels.
        width: Width of each generated figure in pixels.
        use_lines: If True, forces a line plot even for categorical data.

    Returns:
        A dictionary mapping group keys (from split_col) to Plotly Figure objects.

    Raises:
        ValueError: If err_col is 'normal_ci' but sample_size_col is None.
    """
    if df.empty:
        return {}

    df = df.copy()
    active_err_col = err_col

    if err_col == "normal_ci":
        if sample_size_col is None:
            raise ValueError("sample_size_col must be provided when err_col='normal_ci'")

        z_score = stats.norm.ppf(1 - (1 - ci_coverage) / 2)
        active_err_col = "_computed_err_ci"
        df[active_err_col] = z_score * (df[y_col] / np.sqrt(df[sample_size_col]))

    variants = df[x_col].dropna().unique()
    is_numeric_x = pd.api.types.is_numeric_dtype(df[x_col]) or all(isinstance(v, (int, float)) for v in variants)

    if variant_order is None:
        if is_numeric_x:
            df = df.sort_values(x_col)
        else:
            unique_variants = sorted(variants, key=str)
            df[x_col] = pd.Categorical(df[x_col], categories=unique_variants, ordered=True)
            df = df.sort_values(x_col)
    else:
        df[x_col] = pd.Categorical(df[x_col], categories=variant_order, ordered=True)
        df = df.sort_values(x_col)

    if split_col is None:
        groups = [(None, df)]
    else:
        groups = df.groupby(split_col)

    result_figs: Dict[Union[str, int], go.Figure] = {}

    for group_key, group_df in groups:
        title = f"{title_prefix}{group_key}" if split_col and group_key is not None else (title_prefix or "Variant Comparison")

        has_sample_size = sample_size_col in group_df.columns

        if has_sample_size:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        else:
            fig = go.Figure()

        if is_numeric_x and not use_lines:
            trace = go.Scatter(
                x=group_df[x_col],
                y=group_df[y_col],
                mode="lines+markers",
                name=y_col.replace("_", " ").title(),
                error_y=dict(
                    type="data",
                    array=group_df[active_err_col] if active_err_col else None,
                    visible=bool(active_err_col),
                ),
                line=dict(width=3),
                marker=dict(size=8),
            )
            fig.add_trace(trace, row=1, col=1) if has_sample_size else fig.add_trace(trace)
        else:
            unique_variants = variant_order if variant_order is not None else sorted(group_df[x_col].unique(), key=str)

            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
            color_map = {var: colors[i % len(colors)] for i, var in enumerate(unique_variants)}

            for variant in unique_variants:
                subset = group_df[group_df[x_col] == variant]
                if subset.empty:
                    continue

                trace = go.Bar(
                    name=str(variant),
                    x=[str(variant)],
                    y=subset[y_col],
                    marker_color=color_map.get(variant),
                    error_y=dict(
                        type="data",
                        array=subset[active_err_col] if active_err_col else None,
                        visible=bool(active_err_col),
                    ),
                    hovertemplate=f"Variant: {variant}<br>{y_col}: %{{y:.3f}}<extra></extra>",
                )
                fig.add_trace(trace, row=1, col=1) if has_sample_size else fig.add_trace(trace)

        if has_sample_size:
            fig.add_trace(
                go.Bar(
                    x=group_df[x_col] if is_numeric_x else group_df[x_col].astype(str),
                    y=group_df[sample_size_col],
                    name="Sample Size",
                    marker_color="lightgrey",
                    showlegend=False,
                    hovertemplate="Sample Size: %{y:,}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        yaxis_title = y_col.replace("_", " ").title()
        if err_col:
            err_label = f"{ci_coverage*100:.0f}% CI" if err_col == "normal_ci" else err_col.upper()
            yaxis_title = f"{yaxis_title} (Error: ±{err_label})"

        fig.update_layout(
            title=title,
            barmode="group",
            height=height_per_plot,
            width=width,
            legend_title="Variants",
            hovermode="x unified" if is_numeric_x else "closest",
        )

        if has_sample_size:
            fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
            fig.update_yaxes(title_text="Sample Size", row=2, col=1)
            fig.update_xaxes(title_text=x_col.replace("_", " ").title(), row=2, col=1)
        else:
            fig.update_yaxes(title_text=yaxis_title)
            fig.update_xaxes(title_text=x_col.replace("_", " ").title())

        result_figs[group_key or 0] = fig

    return result_figs

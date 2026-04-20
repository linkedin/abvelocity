# BSD 2-CLAUSE LICENSE
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
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

"""
Generic 2D heatmap for labeled arrays.

Displays any 2D array of values as a colored heatmap with labeled rows and
columns. NaN cells are shown as blank. Useful for visualizing independence
test residuals, cell means, standard deviations, or any 2D summary statistic.

Intentionally limited to 2D. For higher-dimensional arrays, select a 2D
slice before calling.

Public API:
    plot_2d_heatmap(values, row_labels, col_labels, ...)
        -> plotly Figure
"""

from typing import List, Optional, Tuple

import numpy as np


def plot_2d_heatmap(
    values: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    cell_texts: Optional[List[List[str]]] = None,
    axis_names: Optional[Tuple[str, str]] = None,
    colorbar_title: str = "value",
    colorscale: str = "RdBu_r",
    symmetric: bool = True,
    clim: Optional[float] = None,
    annotation: str = "",
    title: str = "",
) -> object:
    """Generic 2D heatmap for a labeled array.

    Displays a 2D ndarray as a colored grid. NaN values are shown as blank.
    When symmetric=True the colorbar is centered at zero (useful for residuals
    or differences). When symmetric=False, the colorbar spans the data range.

    Args:
        values: 2D ndarray of shape (n_rows, n_cols). NaN cells are blank.
        row_labels: Label string for each row, length n_rows.
        col_labels: Label string for each column, length n_cols.
        cell_texts: Optional 2D list of per-cell annotation strings, shape
            (n_rows, n_cols). If None, each cell shows its rounded value.
            Cells with NaN values show "N/A" regardless.
        axis_names: Optional (row_axis_name, col_axis_name) for axis titles.
            Defaults to ("Row", "Column").
        colorbar_title: Title string for the colorbar. Default "value".
        colorscale: Plotly colorscale name. Default "RdBu_r" (diverging red-blue).
        symmetric: If True, colorbar is centered at zero and uses a symmetric
            range. Default True.
        clim: Color limit. If symmetric=True, range is [-clim, clim]; if
            symmetric=False, range is [0, clim]. If None, derived from data.
        annotation: Optional subtitle string (e.g. stat summary) shown below
            the main title.
        title: Optional main title string.

    Returns:
        plotly go.Figure with the heatmap.

    Raises:
        ImportError: If plotly is not installed.
        ValueError: If values is not 2D.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for plot_2d_heatmap.") from exc

    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape {values.shape}.")

    row_axis_name, col_axis_name = axis_names if axis_names else ("Row", "Column")

    finite_vals = values[np.isfinite(values)]
    if clim is None:
        if len(finite_vals) == 0:
            clim = 1.0
        elif symmetric:
            clim = max(float(np.max(np.abs(finite_vals))), 1e-9)
        else:
            clim = max(float(np.max(finite_vals)), 1e-9)

    # Build cell text: caller-supplied or default (rounded value)
    n_rows, n_cols = values.shape
    if cell_texts is None:
        cell_texts = []
        for row_idx in range(n_rows):
            row_text = []
            for col_idx in range(n_cols):
                val = values[row_idx, col_idx]
                row_text.append("N/A" if not np.isfinite(val) else f"{val:.3g}")
            cell_texts.append(row_text)

    x_labels = [f"{col_axis_name}: {v}" for v in col_labels]
    y_labels = [f"{row_axis_name}: {v}" for v in row_labels]

    clipped = np.where(np.isfinite(values), values, np.nan)
    if symmetric:
        clipped = np.where(np.isfinite(clipped), np.clip(clipped, -clim, clim), np.nan)
        zmin, zmax, zmid = -clim, clim, 0
    else:
        clipped = np.where(np.isfinite(clipped), np.clip(clipped, 0, clim), np.nan)
        zmin, zmax, zmid = 0, clim, clim / 2

    fig = go.Figure(
        go.Heatmap(
            z=clipped,
            x=x_labels,
            y=y_labels,
            text=cell_texts,
            texttemplate="%{text}",
            colorscale=colorscale,
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=colorbar_title),
        )
    )

    subtitle = annotation if annotation else ""
    full_title = f"{title}<br><sup>{subtitle}</sup>" if title and subtitle else (title or subtitle)

    fig.update_layout(
        title=full_title,
        xaxis_title=f"{col_axis_name}",
        yaxis_title=f"{row_axis_name}",
        width=700,
        height=480,
        font=dict(size=13),
    )

    return fig

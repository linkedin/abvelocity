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
# author: Reza Hosseini

import math
from typing import Dict, Optional

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def gen_combined_figs_html(
    fig_dict: Dict[str, go.Figure],
    html_file_name: Optional[str] = None,
    n_cols: Optional[int] = None,
    row_height: int = 300,
) -> str:
    """
    Combines multiple Plotly figures into a single HTML string.

    Without ``n_cols``: figures are stacked vertically, each with an ``<h3>`` caption.
    With ``n_cols``: figures are arranged in a grid using ``make_subplots``, with dict
    keys used as subplot titles. Legends are shown only in the first column to avoid
    duplicates.

    Args:
        fig_dict: Dictionary where keys are figure names and values are Plotly figures.
        html_file_name: Path to save the combined HTML file. If None, not saved to disk.
        n_cols: Number of columns in the grid. When provided, all figures are combined
            into a single subplot grid instead of being stacked with captions.
        row_height: Height in pixels for each row when ``n_cols`` is set. Defaults to 300.

    Returns:
        str: The combined HTML string.
    """
    if n_cols is not None:
        n_rows = math.ceil(len(fig_dict) / n_cols)
        grid_fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(fig_dict.keys()),
        )
        for i, (_, fig) in enumerate(fig_dict.items()):
            row = i // n_cols + 1
            col = i % n_cols + 1
            for trace in fig.data:
                trace.showlegend = col == 1
                grid_fig.add_trace(trace, row=row, col=col)
        grid_fig.update_layout(height=row_height * n_rows, template="plotly_white")
        html_str = f"<html><body>{pio.to_html(grid_fig, full_html=False)}</body></html>"
    else:
        html_str = ""
        for fig_name, fig in fig_dict.items():
            fig_html = pio.to_html(fig, full_html=False)
            html_str += f"<h3>{fig_name}</h3><div style='margin-bottom: 30px;'>{fig_html}</div>"
        html_str = f"<html><body>{html_str}</body></html>"

    if html_file_name:
        with open(html_file_name, "w") as f:
            f.write(html_str)

    return html_str

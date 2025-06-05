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

from typing import Dict, Optional

import plotly.graph_objects as go
import plotly.io as pio


def gen_combined_figs_html(
    fig_dict: Dict[str, go.Figure], html_file_name: Optional[str] = None
) -> str:
    """
    Combines multiple Plotly figures into a single HTML string with captions and spacing.

    Args:
        fig_dict (Dict[str, go.Figure]): Dictionary where keys are figure names and values are Plotly figures.
        html_file_name (Optional[str]): Path to save the combined HTML file. If None, the result is not saved.

    Returns:
        str: The combined HTML string containing all figures with captions and spacing.
    """
    html_str = ""

    # Loop through the dictionary and create HTML for each figure with a caption and spacing
    for fig_name, fig in fig_dict.items():
        fig_html = pio.to_html(fig, full_html=False)
        # Adding a caption with the figure name and some spacing
        html_str += f"<h3>{fig_name}</h3><div style='margin-bottom: 30px;'>{fig_html}</div>"

    # Wrap the combined HTML in a basic HTML structure
    html_str = f"<html><body>{html_str}</body></html>"

    # Optionally save to file
    if html_file_name:
        with open(html_file_name, "w") as f:
            f.write(html_str)

    return html_str

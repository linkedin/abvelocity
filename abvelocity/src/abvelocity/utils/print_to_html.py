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
import pprint
from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go


def print_to_html(
    data_to_convert: Any,
    font_size: str = "18px",
    color: str = "blue",
    caption: Optional[str] = None,
    file_name: Optional[str] = None,
    width: str = "55vw",  # Added width
    height: str = "55vh",  # Added height
) -> str:
    """
    Converts a print result or Plotly figure to HTML string and optionally stores it.

    Args:
        data_to_convert: The object to display (string, list, dictionary, Plotly figure, etc.).
        font_size: The font size in CSS units (e.g., "18px", "20pt").
        color: The text color (e.g., "red", "blue", "#008000"). Defaults to "blue".
        caption: An optional caption to display above the text.
        file_name: File name to store results.
        width: the width of a plotly plot.
        height: the height of a plotly plot.

    Returns:
        The generated HTML string.
    """

    html_output = ""
    if caption:
        html_output += f"<p style='text-align: center;'><strong>{caption}</strong></p>"

    if isinstance(data_to_convert, go.Figure):
        # Handle Plotly figure
        html_output += (
            f'<div style="width: {width}; height: {height};">'
            + data_to_convert.to_html(full_html=False, include_plotlyjs="cdn")
            + "</div>"
        )

    elif isinstance(data_to_convert, str):
        text = data_to_convert
        html_output += f"<pre style='font-size: {font_size}; color: {color};'>{text}</pre>"
    elif hasattr(data_to_convert, "to_html"):
        html_output += data_to_convert.to_html()
    else:
        text = pprint.pformat(data_to_convert)
        html_output += f"<pre style='font-size: {font_size}; color: {color};'>{text}</pre>"

    if file_name:
        try:
            file_path = Path(file_name)
            if file_path.is_dir():
                raise ValueError(f"'{file_name}' is a directory, not a file.")

            mode = "a" if file_path.exists() else "w"

            with open(file_path, mode) as f:
                if mode == "w":
                    f.write("<html><head><title>Output</title></head><body>\n")
                f.write(html_output + "\n")
                if mode == "w":
                    f.write("</body></html>")

        except (IOError, OSError) as e:
            print(f"Error writing to file '{file_name}': {e}")

    return html_output

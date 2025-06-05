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
from typing import Optional

import pandas as pd


def df_to_html(
    df: pd.DataFrame,
    heading: Optional[str] = None,
    top_paragraphs: list[str] = [],
    file_name: Optional[str] = None,
    caption: str = "",
    bg_colors: Optional[tuple] = None,
    bg_cols: Optional[tuple] = None,
) -> str:
    """Converts a pandas dataframe to an html string.
        This is useful to generate quick readable html reports.

    Args:
        df: A pandas dataframe.
        heading: A string for the heading of the html.
            The default is None, in which case no heading will be included.
        top_paragraphs: A list of strings for the top paragraphs of the html.
            The default is an empty list.
        file_name: A string for the name of the html file.
            If None, no file will be written.
        caption: A string for the caption of the table.
            The default is an empty string.
        bg_colors: A tuple containing background color codes. Must be the same length as
            the number of rows in the DataFrame.
        bg_cols: A tuple specifying the columns where the background colors should be applied.
            If None, the colors will be applied to all columns. Defaults to None.
        table_id: This is an internal parameter inside the html file.
            This can be helpful to fix the html file generated and remove randomness.
            This should not have an impact on end user.

    Returns:
        A string of the html.
    """
    if bg_colors is None:
        df_html = df.to_html(index=False)
    else:
        df_html = to_html_format_bg(df=df, bg_colors=bg_colors, bg_cols=bg_cols)

    df_html = df_html.replace(
        "<thead>", f"<caption> <big> <strong> {caption} </strong> </big> </caption> <thead>"
    )

    html_str = ""
    if heading is not None:
        html_str = f"""
        <!DOCTYPE html>
        <html>
        <body>
        <h1>{heading}</h1>
        <p></p>
        <p></p>
        """

    for paragraph in top_paragraphs:
        html_str += f"""
        <p>{paragraph}</p>
        <p></p>
        <p></p>
        """

    html_str += df_html

    if heading is not None:
        html_str += """
        </body>
        </html>
        """

    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(html_str)
            print(f"{file_name} was created.")

    return html_str


def to_html_format_bg(df: pd.DataFrame, bg_colors: tuple, bg_cols: Optional[tuple] = None) -> str:
    """
    Converts a DataFrame to an HTML table with background color formatting.

    This function applies background colors to the rows of the DataFrame based on
    the provided `colors` argument. Optionally, background color formatting can
    be applied only to specified columns.

    Args:
        df: The input DataFrame to be converted to HTML with background formatting.
        bg_colors: A tuple containing background color codes. Must be the same length as
            the number of rows in the DataFrame.
        bg_cols: A tuple specifying the columns where the background colors should be applied.
            If None, the colors will be applied to all columns. Defaults to None.

    Returns:
        str: An HTML string representing the formatted DataFrame.

    Raises:
        ValueError: If the length of `colors` does not match the number of rows in the DataFrame.
    """
    if len(df) != len(bg_colors):
        raise ValueError("bg_colors must be of the same length as data rows")

    df = df.reset_index(drop=True)
    df_cols = list(df.columns)

    def func(row):
        color = f"background-color: {bg_colors[row.name]}"

        if not bg_cols:
            return tuple([color] * len(df_cols))
        else:
            return tuple([color if col in bg_cols else None for col in df_cols])

    df_html = df.style.apply(func, axis=1).to_html(index=False)

    df_html = df_html.replace("<table", '<table border="1"')

    return df_html

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
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

from typing import Dict, Optional

import pandas as pd
from abvelocity.core.utils.df_to_html import df_to_html

FORMAT_DICT: Dict[str, str] = {
    # Added ',' to ensure thousands separation
    "float": "{:,.5f}",
    # Specific simple float columns
    # Added ',' to each specific float format
    "delta": "{:,.5f}",
    "delta_percent": "{:,.3f}",
    "p_value": "{:.6f}",  # P-values are small; typically no comma needed
    "z_value": "{:,.2f}",
    "t_value": "{:,.2f}",
    # Lambda functions for complex/tuple columns
    "ci": lambda x: f"({x[0]:,.4f}, {x[1]:,.4f})",  # Tuple of floats
    "ci_percent": lambda x: f"({x[0]:,.4f}, {x[1]:,.4f})",  # Tuple of floats
    # Handle float or tuple in one column
    "delta_sum": lambda x: (f"({x[0]:,.4f}, {x[1]:,.4f})" if isinstance(x, (tuple, list)) else f"{x:,.5f}"),
    # "delta_sum_ci": lambda x: (
    #    f"({x[0]:,.5f}, {x[1]:,.5f})" if isinstance(x, (tuple, list)) else f"{x:,.5f}"
    # ),
    # Specific formatters for integer tuples (counts) - Commas added for large counts
    "sample_counts": lambda x: f"({x[0]:,d}, {x[1]:,d})",  # Tuple of integers
    "impacted_counts": lambda x: f"({x[0]:,d}, {x[1]:,d})",  # Tuple of integers
}


def publish_df_color_code_p_value(
    df: pd.DataFrame,
    bg_cols: tuple[str],
    df_name: str = "",
    html_str: str = "",
    markdown_str: str = "",
    split_col: Optional[str] = None,
    drop_split_col: bool = True,
):
    """
    Publishes a DataFrame with color-coded values for statistical significance and
    returns the corresponding HTML and Markdown representations.

    This function color-codes cells in the DataFrame based on their `p_value` and `delta_percent`.
    Rows with a `p_value` below 0.05 are highlighted: green for positive `delta_percent` and
    red for negative `delta_percent`. The DataFrame can be split and displayed by a specified column.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        bg_cols (tuple[str]): A tuple of column names to use for background coloring.
        df_name (str, optional): A name for the DataFrame, used in the output captions. Default is an empty string.
        html_str (str, optional): Initial HTML string to append the DataFrame's representation. Default is an empty string.
        markdown_str (str, optional): Initial Markdown string to append the DataFrame's representation. Default is an empty string.
        split_col (str, optional): A column name to split the DataFrame by. If specified, separate tables are
            created for each unique value in this column. Default is None.
        drop_split_col (bool): Deterimines if we should drop the `split_col`
            as it is already mentioned as the sub dataframe title

    Returns:
        tuple[str, str]: A tuple containing:
            - html_str (str): The resulting HTML string with the DataFrame's representation.
            - markdown_str (str): The resulting Markdown string with the DataFrame's representation.

    Notes:
        - Cells in the DataFrame are color-coded based on their `p_value` and `delta_percent` values:
          - Light green (`rgba(0, 250, 0, 0.5)`) for `p_value < 0.05` and `delta_percent >= 0`.
          - Light red (`rgba(250, 0, 0, 0.5)`) for `p_value < 0.05` and `delta_percent < 0`.
        - If `split_col` is specified, the function generates separate tables for each unique value
          in the column and appends them to the HTML and Markdown strings.
    """

    df_name_pretty = " ".join([word.capitalize() for word in df_name.split("_")])

    html_str += f"""
        <br><br><br>
        <h2 style="color: blue; font-size: 28px; margin-top: 20px; margin-bottom: 20px;">
        {df_name_pretty}
        </h2>
    """

    def func(df0, html_str, markdown_str):
        bg_colors = None
        # color coding
        if "delta_percent" in df0.columns and "p_value" in df.columns:
            bg_colors = []
            for row in df0.itertuples():
                if row.p_value < 0.05:
                    if row.delta_percent >= 0.0:
                        bg_colors.append("rgba(0, 250, 0, 0.5)")  # Light green
                    else:
                        bg_colors.append("rgba(250, 0, 0, 0.5)")  # Light red
                else:
                    bg_colors.append(None)
            bg_colors = tuple(bg_colors)

        html_str += df_to_html(
            df=df0,
            top_paragraphs=[],
            caption="",
            bg_colors=bg_colors,
            bg_cols=bg_cols,
            format_dict=FORMAT_DICT,
        )

        markdown_str += f"""\n\n\n\n##  <font color="blue">{df_name_pretty}</font>\n\n\n\n"""
        markdown_str += "\n\n" + df0.to_markdown(index=False)

        return html_str, markdown_str

    if split_col is not None and split_col in df.columns:
        for x in sorted(df[split_col].unique(), key=str):
            html_str += f"\n<h2>{df_name_pretty}: {x}</h2>\n"
            df0 = df[df[split_col] == x].reset_index(drop=True)
            if drop_split_col:
                del df0[split_col]
            if "metric" in df0.columns:
                df0 = df0.sort_values("metric").reset_index(drop=True)
            html_str, markdown_str = func(df0=df0, html_str=html_str, markdown_str=markdown_str)
    else:
        html_str += f"\n<h2>{df_name_pretty}</h2>\n"
        if "metric" in df.columns:
            df = df.sort_values("metric").reset_index(drop=True)
        html_str, markdown_str = func(df0=df, html_str=html_str, markdown_str=markdown_str)

    return html_str, markdown_str

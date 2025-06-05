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


def gen_pcnt_barplot(df: pd.DataFrame, x_col: str, title: Optional[str] = None):
    """
    Generates a percentage bar plot for distinct values in a column.

    Args:
        df: The pandas DataFrame containing the data.
        x_col: The column to use for the x-axis (categorical).
        title: Optional title of the plot. If None, a default title is used.

    Returns:
        A plotly.graph_objects.Figure object.
    """

    # Calculate value counts and percentages
    value_counts = df[x_col].value_counts(normalize=True) * 100
    bar_df = pd.DataFrame(
        {x_col: value_counts.index.astype(str), "percentage": value_counts.values}
    )

    # Determine title
    if title is None:
        title = f"Percentage Distribution of {x_col}"

    # Create the bar plot
    fig = px.bar(
        bar_df,
        x=x_col,
        y="percentage",
        title=title,
        labels={"percentage": "Percentage"},
    )

    return fig

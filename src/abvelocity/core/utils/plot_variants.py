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
import plotly.graph_objects as go
from abvelocity.core.utils.plot_2d_variants import plot_2d_variants
from abvelocity.core.utils.plot_3d_variants import plot_3d_variants

NAN_VALUE = "nan"


def plot_variants(
    df: pd.DataFrame,
    variant_col: str = "variant",
    count_column: str = "variant_count",
    title: str = "Partition of Space",
    dim_names: Optional[list[str]] = None,
) -> go.Figure:
    """
    Create an interactive 2D / 3D bubble plot partitioning space based on three-dimensional categorical tuples using Plotly.
    It simply figures out the length of the tuple for the variant_col and then call either the 2D or 3D function.

    Args:
        df: DataFrame with variant_col and count_column.
        variant_col: Column name for the label we are counting (default: 'variant')
        count_column: Column name for bubble size (default: 'variant_count').
        title: Plot title.
        dim_names: The title of dimension names to be used in the x, y, ... axes.

    Returns:
        go.Figure: Plotly figure object.
    """

    variant = df[variant_col].values[0]

    if len(variant) == 2:
        plot_func = plot_2d_variants
    else:
        plot_func = plot_3d_variants

    return plot_func(df=df, variant_col=variant_col, count_column=count_column, title=title, dim_names=dim_names)

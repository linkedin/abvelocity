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

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from abvelocity.journey.seq.count_seq_query import CountSeqQuery
from abvelocity.journey.viz.create_sunburst import create_sunburst
from abvelocity.param.io_param import IOParam


@dataclass
class SunburstResult:
    """Dataclass to hold the results of sunburst generation."""

    fig: Optional[go.Figure] = None
    df: Optional[pd.DataFrame] = None
    query: Optional[str] = None


class SunburstPlot:
    """Generates sunburst visualizations for sequence data."""

    def __init__(self, io_param: IOParam):
        """
        Initializes the SunburstPlot instance.

        Args:
            io_param (IOParam): Dataclass containing input/output parameters.
        """
        self.cursor = io_param.cursor
        self.print_to_html = io_param.print_to_html
        self.save_path = io_param.save_path
        self.file_name_suffix = io_param.file_name_suffix
        os.makedirs(self.save_path, exist_ok=True)

    def gen(
        self,
        table_name: str,
        value_col: str,
        conditions: List[str] = ["TRUE = TRUE"],
        max_seq_index: int = 5,
        count_distinct_col: Optional[str] = None,
        color_dict: Optional[Dict[str, str]] = None,
        additional_dimensions: Optional[List[str]] = None,
        top_n_dims: Optional[int] = 5,
    ) -> SunburstResult:
        """
        Generate a sunburst visualization for a given sequence table.
        It assumes the ordered events are provided in columns: s1, s2, ...,sk
        where k = `max_seq_index`.

        Args:
            table_name (str): Name of the joined table containing sequence data.
            value_col (str): Column name to be used for sunburst values.
            conditions (Optional[List[str]]): Optional list of SQL conditions.
            max_seq_index (int): Number of elements in the sequence / journey.
            count_distinct_col (Optional[str]): Column to count distinct values from.
            color_dict (Optional[Dict[str, str]]): A dictionary mapping labels to colors. If None, default colors are generated.
            additional_dimensions (Optional[List[str]]): Additional columns to group by in the sunburst plot. These columns will be
                                                     appended as extra levels in the sunburst hierarchy. Defaults to None.
            top_n_dims (Optional[int]): For each extra groupby column, only the top N most frequent values will be shown;
                                        all others will be grouped as 'Others'. This helps keep the plot readable. Defaults to 5.

        Returns:
            SunburstResult: Dataclass containing the generated chart data.
        """

        query = CountSeqQuery(
            table_name=table_name,
            max_seq_index=max_seq_index,
            count_distinct_col=count_distinct_col,
            extra_groupby_cols=additional_dimensions,
            conditions=conditions,
        ).gen()

        query += " ORDER BY percent DESC"
        df = self.cursor.get_df(query=query).df

        if df.empty:
            raise ValueError("df is empty, cannot plot.")

        # Consider the top N dimensions if specified and classifies the rest as 'Others' for the sake for readability.
        if top_n_dims and additional_dimensions:
            for dim in additional_dimensions:
                if dim in df.columns:
                    top_vals = df.groupby(dim)[value_col].sum().nlargest(top_n_dims).index
                    df[dim] = df[dim].where(df[dim].isin(top_vals), "Others")

        if additional_dimensions:
            for i, dim in enumerate(additional_dimensions, start=max_seq_index + 1):
                if dim not in df.columns:
                    continue
                # Prefix to keep labels unique and clear
                df[f"s{i}"] = df[dim].astype(str)
            # Bump the depth so create_sunburst sees them
            max_seq_index += len(additional_dimensions)

        df = df.fillna("end")
        print(f"\n\n\n*** df:\n\n\n {df.head()}")

        title = f"Sunburst plot: {table_name}: quantity:{value_col}: conditions: {conditions}"

        fig = create_sunburst(
            df=df,
            max_seq_index=max_seq_index,
            value_col=value_col,
            color_dict=color_dict,
            title=title,
        )

        file_name = os.path.join(
            self.save_path, f"sunburst{self.file_name_suffix}_{table_name.replace('.', '_')}.html"
        )
        if self.print_to_html:
            self.print_to_html(fig, file_name=file_name)

        return SunburstResult(fig=fig, df=df, query=query)

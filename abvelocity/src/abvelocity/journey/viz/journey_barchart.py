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
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

from abvelocity.param.io_param import IOParam
from abvelocity.utils.get_categ_column_distbn import get_categ_column_distbn


@dataclass
class BarchartResult:
    """Dataclass to hold the results of barchart generation."""

    fig: go.Figure
    df: pd.DataFrame
    plot_df: pd.DataFrame
    query: str
    num_distict_labels: int


class JourneyBarchart:
    """Generates bar charts for event sequences."""

    def __init__(self, io_param: IOParam):
        """
        Initializes the JourneyBarchart instance.

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
        count_distinct_col: Optional[str] = None,
        col: str = "event_seq",
        max_num: int = 25,
        conditions: List[str] = ["TRUE = TRUE"],
        sort_event_array: bool = False,
        is_map: bool = False,
        title: Optional[str] = None,
    ) -> BarchartResult:
        """
        Generates a bar chart for event sequences.

        Args:
            table_name (str): Name of the table containing the sequence data.
            count_distinct_col (Optional[str]): Column to count distinct values from.
            col (str): Column containing the event sequences.
            max_num (int): Maximum number of bars in the chart.
            conditions (List[str]): List of SQL conditions to filter data.
            sort_event_array (bool): Whether to sort events within sequences.
                This is not enabled for maps as conversions will be too complex.
            is_map (bool): Whether the seq is a map or not.
            title (str, optional): Optional title. One will be constructed based on input, if not passed.

        Returns:
            BarchartResult: Dataclass containing the generated chart data.
        """

        if is_map:
            table_name2 = f"""
                (SELECT
                    JSON_FORMAT(CAST(event_seq AS JSON)) AS event_seq_str,
                    *
                FROM
                {table_name}
                )
            """
        else:
            table_name2 = f"""
                (SELECT
                    ARRAY_JOIN({col}, ',') AS event_seq_str,
                    *
                FROM
                    {table_name}
                )
            """

            if sort_event_array:
                table_name2 = f"""
                    (SELECT
                        ARRAY_JOIN(ARRAY_SORT({col}), ',') AS event_seq_str,
                        *
                    FROM
                        {table_name}
                    )
                """

        res_dict = get_categ_column_distbn(
            table_name=table_name2,
            col="event_seq_str",
            conditions=conditions,
            max_num=max_num,
            cursor=self.cursor,
            print_to_html=None,  # We handle the printing to html below
            title=title,
        )

        file_name = os.path.join(
            self.save_path, f"barchart{self.file_name_suffix}_{table_name.replace('.', '_')}.html"
        )
        if self.print_to_html:
            self.print_to_html(res_dict["fig"], file_name=file_name)

        return BarchartResult(**res_dict)

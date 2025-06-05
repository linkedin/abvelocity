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
"""The goal is to merge various experiment assignment DataContainers."""

import pandas as pd

from abvelocity.get_data.data_container import DataContainer
from abvelocity.param.constants import (
    CATEG_NAN_VALUE,
    TRIGGER_TIME_COL,
    UNIT_COL,
    VARIANT_COL,
)
from abvelocity.utils.merge_dfs_number_cols import merge_dfs_number_cols

ON_COLS = [UNIT_COL]
"""This is the expected list of columns which experiment data is to be joined on."""

COMMON_COLS = [VARIANT_COL, TRIGGER_TIME_COL]
"""These are the typical common columns across experiment data."""


def join_expt_dfs(
    dc_list: list[DataContainer],
    on_cols: list[str] = ON_COLS,
    common_cols: list[str] = COMMON_COLS,
    how: str = "outer",
    add_tuple: bool = True,
    drop_numbered_cols: bool = False,
) -> DataContainer:
    """Joins all experiment DataContainers by `UNIT_COL`.
    Typically we expect the common columns to be

        - VARIANT_COL
        - TRIGGER_TIME_COL

    Args:
        dc_list: A list of DataContainer objects containing experiment data.
        on_cols: Columns to join on.
        common_cols: Common columns across DataFrames.
        how: Type of merge to be performed.
        add_tuple: If True, add a tuple column.
        drop_numbered_cols: If True, drop numbered columns.

    Returns:
        A DataContainer containing the joined DataFrame.
    """

    df_list = [dc.df for dc in dc_list if dc.df is not None]

    if not df_list:
        return DataContainer(
            df=pd.DataFrame(), is_df=True
        )  # Return empty DataFrame if input is empty

    joined_df = merge_dfs_number_cols(
        df_list=df_list,
        on_cols=on_cols,
        common_cols=common_cols,
        how=how,
        add_tuple=add_tuple,
        nan_replacement=CATEG_NAN_VALUE,
        drop_numbered_cols=drop_numbered_cols,
    )

    return DataContainer(df=joined_df, is_df=True)

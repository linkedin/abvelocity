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

from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.param.constants import CATEG_NAN_VALUE, TRIGGER_TIME_COL, VARIANT_COL
from abvelocity.core.utils.merge_dfs_number_cols import merge_dfs_number_cols
from abvelocity.core.utils.merge_queries_number_cols import merge_queries_number_cols

COMMON_COLS = [VARIANT_COL]
"""These are the typical common columns across experiment data."""

COMMON_COLS_EXTENDED = [VARIANT_COL, TRIGGER_TIME_COL]


def join_expt_dfs(
    dc_list: list[DataContainer],
    on_cols: list[str],
    common_cols: list[str] = COMMON_COLS,
    how: str = "outer",
    add_tuple: bool = True,
    drop_numbered_cols: bool = False,
) -> DataContainer:
    """Joins all experiment DataContainers by `on_cols` which is typically consists of
        the unit column of the experiment assignment data.
    Typically we expect the common columns to be

        - VARIANT_COL

    OR

        - VARIANT_COL, TRIGGER_TIME_COL

    Args:
        dc_list: A list of DataContainer objects containing experiment data.
        on_cols: Columns to join on. Typically this consists of the unit column of the experiment assignment data.
        common_cols: Common columns across DataFrames.
        how: Type of merge to be performed.
        add_tuple: If True, add a tuple column.
        drop_numbered_cols: If True, drop numbered columns.

    Returns:
        A DataContainer containing the joined DataFrame.
    """

    # There will be three cases
    # Case 1: There are dfs materialized in all of the dcs
    # Case 2: dc_list contains SQL tables names
    # Case 3: dc_list contains SQL tables
    # To determine these we only assess the first element in dc_list
    # If it matches the case, we assume other elements do match as well

    if dc_list[0].pandas_df is not None:  # Case 1
        df_list = [dc.pandas_df for dc in dc_list if dc.pandas_df is not None]

        joined_df = merge_dfs_number_cols(
            df_list=df_list,
            on_cols=on_cols,
            common_cols=common_cols,
            how=how,
            add_tuple=add_tuple,
            nan_replacement=CATEG_NAN_VALUE,
            drop_numbered_cols=drop_numbered_cols,
        )

        return DataContainer(pandas_df=joined_df, is_pandas_df=True)
    elif dc_list[0].table_name is not None:  # Case 2
        query_list = [dc.table_name for dc in dc_list if dc.table_name is not None]
    else:  # Case 3
        query_list = [dc.query for dc in dc_list if dc.query is not None]

    # We have not returned for Case 2 and Case 3 yet and require more steps
    print(f"***\n In `join_expt_dfs`: query_list:\n {query_list}")
    join_query = merge_queries_number_cols(
        queries=query_list,
        on_cols=on_cols,
        common_cols=common_cols,
        how=how,
        add_tuple=add_tuple,
        nan_replacements=[CATEG_NAN_VALUE for col in common_cols],
        drop_numbered_cols=drop_numbered_cols,
    )

    return DataContainer(query=join_query, is_pandas_df=False)

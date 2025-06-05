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
"""The goal is to merge a collection of dataframes and number the common columns."""

import gc
import resource
import sys
from functools import reduce
from typing import Optional

import pandas as pd

from abvelocity.utils.check_df_validity import check_df_validity


def merge_dfs_number_cols(
    df_list: list[pd.DataFrame],
    on_cols: list[str],
    common_cols: Optional[list[str]] = None,
    how: str = "outer",
    add_tuple: bool = True,
    nan_replacement: Optional[str | float] = None,
    drop_numbered_cols: bool = False,
    incremental_df_delete: bool = True,
) -> pd.DataFrame:
    """A function to merge a list of pandas dataframes while numbering the common
        columns by numerics starting from 1.

        This is a different behaviour from Pandas which create {col}_x, {col}_y
        when merging two dataframes with a common column col.
        Here the purpose is to merge potentially more than two dataframes and end up
        with more predictable column names.

        This function will add
        {col}_1, {col}_2, {col}_3 etc as the new columns for each common column col.
        This function will also create tuple columns to combine all the values
        into one tuple as well if `add_tuple` is True.

        This function is cognizant of memory usage and takes steps to save
        memory space by

            - dropping used dataframes as it merges
            - dropping numbered columns as it is creating tuples (if desired).

    Args:
        df_list: A list of pandas dataframes to be merged.
        on_cols: The list of columns to join on.
        common_cols: A list of common columns to be numbered.
            If None, the assumption is all columns besides the `on_cols` are common.
        how: Specifies the method for merging.
            The default is "outer". It can also be "right" or "left".
        add_tuple: If tuple columns are to be added by combining the numbered columns.
        nan_replacement: If passed, it will replace NaN values with this value,
            only in `common_cols`.
        drop_numbered_cols: If it is desired to drop the numbered columns.
        incremental_df_delete: If it is desired to delete the dataframes as they are merged.
            Default is True.

    Returns:
        result: The merged dataframe with merged data for each common column, {col},  given in
            {col}_1, {col}_2, {col}_3, ...
            Also it could contain a tuple column with the same name {col} to store
            a tuple of all those values, if `add_tuple` is True.

    Alters:
        df_list: This argument will be altered to save memory as opposed to be copied.
            If you do not want to alter this argument, pass a copy.

    Raises:
        None.
    """
    all_cols = df_list[0].columns
    if common_cols is None:
        common_cols = [col for col in all_cols if col not in on_cols]

    df_num = len(df_list)

    # Data validation steps.
    err_trigger_source = "merge_dfs_number_cols"
    for i in range(df_num):
        df = df_list[i]
        # Checks the existence of `on_cols`.
        check_df_validity(
            df=df,
            needed_cols=on_cols,
            err_trigger_source=err_trigger_source,
            err_message=f"df_list[{i}] has missing columns for merge (`on_cols`).",
        )
        # Checks the existence of `common_cols`.
        check_df_validity(
            df=df,
            needed_cols=common_cols,
            err_trigger_source=err_trigger_source,
            err_message=f"df_list[{i}] has missing columns for merge (`common_cols`).",
        )

    # Below alters the name of the common columns
    # by adding a number: f"{col}_{i + 1}"" based on the order `i` in `df_list`
    for i in range(df_num):
        df = df_list[i]
        for col in common_cols:
            df_list[i].rename(columns={col: f"{col}_{i + 1}"}, inplace=True)

    def merge_two(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Merges two dataframes.
        It deletes the input dataframes to save memory space.
        Args:
            df1 : First dataframe.
            df2 : Second dataframe.
        Returns:
            df: Merged dataframe.
        Raises:
            None.
        """
        df = pd.merge(df1, df2, on=on_cols, how=how)
        # Removes unnecessary data to retain memory space.
        if incremental_df_delete:
            del df1
            del df2
        # If `nan_replacement` is passed,
        # it will replace NaN values with this value.
        # Since we add numbers suffices to the common cols during merge,
        # we need to find the new column names, before we can replace NaNs.
        # The double for loop below is to achieve that goal.
        # Note that since the number of columns of df and common cols is small,
        # this will not be a computational problem.
        if nan_replacement is not None:
            for col in df.columns:
                for common_col in common_cols:
                    if col.startswith(common_col):
                        df.fillna({col: nan_replacement}, inplace=True)
        return df

    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
    print(f"\n*** Memory used in `merge_dfs_number_cols`, before merging dfs: {mem_usage:.4f} GB")

    df = reduce(merge_two, df_list)
    size_in_megabytes = sys.getsizeof(df) / 10**6
    print(f"\n***`df` size in MB: {size_in_megabytes}")

    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
    print(f"\n*** Memory used in `merge_dfs_number_cols`, after merging dfs: {mem_usage:.4f} GB")

    # Delete `df_list` and Garbage collection
    del df_list
    gc.collect()
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
    print(
        f"\n*** Memory used in `merge_dfs_number_cols`, after del df_list and gc: {mem_usage:.4f} GB"
    )

    if add_tuple:
        for col in common_cols:
            added_cols = [f"{col}_{i + 1}" for i in range(df_num)]
            df[col] = list(zip(*[df[c] for c in added_cols]))
            # If the numbered columns are to be dropped,
            # it is done below as we are adding tuple columns to maintain memory space.
            # This will cause a few lines of (delibrate) repeated code below under `else`.
            if drop_numbered_cols:
                for col in added_cols:
                    del df[col]
    elif drop_numbered_cols:
        # This is the case where no tuple columns are added.
        # But we do want to drop the numbered columns.
        # This is a rare situation, but retains the generality of this function
        # as a generic merge function.
        for col in common_cols:
            added_cols = [f"{col}_{i + 1}" for i in range(df_num)]
            for col in added_cols:
                del df[col]

    return df

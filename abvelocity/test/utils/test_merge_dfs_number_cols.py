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

import numpy as np
import pandas as pd

from abvelocity.utils.merge_dfs_number_cols import merge_dfs_number_cols


def test_merge_dfs_number_cols():
    """Tests `merge_dfs_number_cols`."""

    # With two dataframes.
    df1 = pd.DataFrame({"memberid": [1, 2, 3], "variant": ["1", "2", "3"], "time": ["1", "2", "3"]})

    df2 = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant": ["1", "2", np.nan, "2"],
            "time": ["1", "2", "3", np.nan],
        }
    )

    on_cols = ["memberid"]
    df_list = [df1.copy(), df2.copy()]

    obtained_df = merge_dfs_number_cols(
        df_list=df_list, on_cols=on_cols, common_cols=None, how="outer", add_tuple=True
    )

    expected_tuple_df = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant": [("1", "1"), ("2", "2"), ("3", np.nan), (np.nan, "2")],
            "time": [("1", "1"), ("2", "2"), ("3", "3"), (np.nan, np.nan)],
        }
    )

    expected_new_cols_df = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant_1": ["1", "2", "3", np.nan],
            "time_1": ["1", "2", "3", np.nan],
            "variant_2": ["1", "2", np.nan, "2"],
            "time_2": ["1", "2", "3", np.nan],
        }
    )

    pd.testing.assert_frame_equal(obtained_df[["memberid", "variant", "time"]], expected_tuple_df)
    pd.testing.assert_frame_equal(
        obtained_df[["memberid", "variant_1", "time_1", "variant_2", "time_2"]],
        expected_new_cols_df,
    )

    # Without adding tuple columns.
    df_list = [df1.copy(), df2.copy()]
    obtained_df = merge_dfs_number_cols(
        df_list=df_list, on_cols=on_cols, common_cols=None, how="outer", add_tuple=False
    )

    pd.testing.assert_frame_equal(obtained_df, expected_new_cols_df)

    # With three dataframes.
    df3 = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant": ["11", "22", "33", "44"],
            "time": ["11", "22", "33", "44"],
        }
    )

    df_list = [df1.copy(), df2.copy(), df3.copy()]

    expected_df = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant": [
                ("1", "1", "11"),
                ("2", "2", "22"),
                ("3", np.nan, "33"),
                (np.nan, "2", "44"),
            ],
            "time": [("1", "1", "11"), ("2", "2", "22"), ("3", "3", "33"), (np.nan, np.nan, "44")],
        }
    )

    obtained_df = merge_dfs_number_cols(
        df_list=df_list,
        on_cols=on_cols,
        common_cols=None,
        how="outer",
        add_tuple=True,
        drop_numbered_cols=True,
    )

    pd.testing.assert_frame_equal(obtained_df, expected_df)


def test_merge_dfs_number_cols_with_nan_replacement():
    """Tests `merge_dfs_number_cols`.
    Here we specifically tests the `nan_replacement` parameter.
    """
    # With two dataframes.
    df1 = pd.DataFrame({"memberid": [1, 2, 3], "variant": ["1", "2", "3"], "time": ["1", "2", "3"]})

    df2 = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant": ["1", "2", np.nan, "2"],
            "time": ["1", "2", "3", np.nan],
        }
    )

    on_cols = ["memberid"]
    df_list = [df1.copy(), df2.copy()]

    obtained_df = merge_dfs_number_cols(
        df_list=df_list,
        on_cols=on_cols,
        common_cols=None,
        how="outer",
        add_tuple=True,
        nan_replacement="some_nan",
    )

    print(obtained_df)

    expected_tuple_df = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant": [("1", "1"), ("2", "2"), ("3", "some_nan"), ("some_nan", "2")],
            "time": [("1", "1"), ("2", "2"), ("3", "3"), ("some_nan", "some_nan")],
        }
    )

    pd.testing.assert_frame_equal(obtained_df[["memberid", "variant", "time"]], expected_tuple_df)

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
import pytest
from abvelocity.core.utils.prep_df_for_grouped_plot import prep_df_for_grouped_plot


def test_prep_df_no_grouping():
    """
    Test prep_df_for_grouped_plot when no grouping columns are provided.
    The original y_col name should be preserved.
    """
    data = {"date": ["2023-01-01", "2023-01-02"], "sales": [100, 150], "region": ["East", "West"]}
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "sales"

    result = prep_df_for_grouped_plot(df, x_col, y_col, group_by_cols=None)
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    expected_df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "sales": [100, 150]})
    expected_y_cols = ["sales"]

    pd.testing.assert_frame_equal(transformed_df, expected_df)
    assert y_cols == expected_y_cols


def test_prep_df_single_group_by_col():
    """
    Test prep_df_for_grouped_plot with a single grouping column.
    """
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
        "region": ["East", "West", "East", "West"],
        "sales": [100, 120, 150, 130],
    }
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "sales"
    group_by_cols = ["region"]

    result = prep_df_for_grouped_plot(df, x_col, y_col, group_by_cols)
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    expected_df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "East": [100.0, 150.0], "West": [120.0, 130.0]})
    expected_y_cols = ["East", "West"]

    pd.testing.assert_frame_equal(transformed_df, expected_df)
    assert sorted(y_cols) == sorted(expected_y_cols)  # Order might vary for column names due to pivot


def test_prep_df_multiple_group_by_cols():
    """
    Test prep_df_for_grouped_plot with multiple grouping columns.
    """
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"],
        "region": ["East", "East", "West", "West"],
        "product": ["A", "B", "A", "B"],
        "sales": [10, 20, 15, 25],
    }
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "sales"
    group_by_cols = ["region", "product"]

    result = prep_df_for_grouped_plot(df, x_col, y_col, group_by_cols)
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    expected_df = pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "East_A": [10.0],
            "East_B": [20.0],
            "West_A": [15.0],
            "West_B": [25.0],
        }
    )
    expected_y_cols = ["East_A", "East_B", "West_A", "West_B"]

    pd.testing.assert_frame_equal(transformed_df, expected_df)
    assert sorted(y_cols) == sorted(expected_y_cols)  # Order might vary for column names due to pivot


def test_prep_df_non_unique_x_values_within_group():
    """
    Test prep_df_for_grouped_plot where x_col values are not unique within groups,
    leading to pivot_table aggregating (default is mean).
    """
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"],
        "region": ["East", "East", "West", "West"],
        "sales": [10, 20, 15, 25],
    }
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "sales"
    group_by_cols = ["region"]

    result = prep_df_for_grouped_plot(df, x_col, y_col, group_by_cols)
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    expected_df = pd.DataFrame({"date": ["2023-01-01"], "East": [15.0], "West": [20.0]})  # (10 + 20) / 2  # (15 + 25) / 2
    expected_y_cols = ["East", "West"]

    pd.testing.assert_frame_equal(transformed_df, expected_df)
    assert sorted(y_cols) == sorted(expected_y_cols)


def test_prep_df_with_nan_values_in_y_col():
    """
    Test prep_df_for_grouped_plot with NaN values in the y_col.
    pivot_table should handle these by default (will result in NaN in pivoted columns).
    """
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "region": ["East", "West", "East"],
        "sales": [100.0, np.nan, 150.0],
    }
    df = pd.DataFrame(data)
    x_col = "date"
    y_col = "sales"
    group_by_cols = ["region"]

    result = prep_df_for_grouped_plot(df, x_col, y_col, group_by_cols)
    transformed_df = result["df"]
    y_cols = result["y_cols"]

    expected_df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "East": [100.0, 150.0], "West": [np.nan, np.nan]})
    # Use check_dtype=False to focus on values and shape, ignoring subtle dtype differences
    pd.testing.assert_frame_equal(transformed_df, expected_df, check_dtype=False)
    assert sorted(y_cols) == sorted(["East", "West"])


def test_prep_df_value_error_x_col_missing():
    """
    Test ValueError is raised when x_col is not in DataFrame.
    """
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="x_col 'missing_x' not found in DataFrame columns."):
        prep_df_for_grouped_plot(df, "missing_x", "b")


def test_prep_df_value_error_y_col_missing():
    """
    Test ValueError is raised when y_col is not in DataFrame.
    """
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="y_col 'missing_y' not found in DataFrame columns."):
        prep_df_for_grouped_plot(df, "a", "missing_y")


def test_prep_df_value_error_group_by_col_missing():
    """
    Test ValueError is raised when a group_by_col is not in DataFrame.
    """
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    with pytest.raises(ValueError, match="Group-by column 'missing_group' not found in DataFrame columns."):
        prep_df_for_grouped_plot(df, "a", "b", group_by_cols=["c", "missing_group"])

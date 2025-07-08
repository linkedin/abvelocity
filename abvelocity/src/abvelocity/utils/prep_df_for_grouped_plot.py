from typing import Dict, List, Optional, Union

import pandas as pd


def prep_df_for_grouped_plot(
    df: pd.DataFrame, x_col: str, y_col: str, group_by_cols: Optional[List[str]] = None
) -> Dict[str, Union[pd.DataFrame, List[str]]]:
    """
    Transforms a DataFrame to prepare it for plotting multiple lines/markers
    based on unique combinations of grouping columns.

    If `group_by_cols` are provided, the DataFrame is pivoted such that each
    unique combination of grouping columns becomes a new column, with `x_col`
    as the index. The values for these new columns are taken from `y_col`.

    If `group_by_cols` are not provided, the original DataFrame is returned
    with only `x_col` and `y_col`, and `y_col` retains its original name.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The name of the column to be used as the x-axis.
        y_col (str): The name of the column to be used as the y-axis (values).
        group_by_cols (Optional[List[str]]): A list of column names to group by.
            Each unique combination of these
            columns will result in a separate
            line/marker series. If None, no
            grouping/pivoting is performed.

    Returns:
        Dict[str, Union[pd.DataFrame, List[str]]]: A dictionary containing:
            - 'df' (pd.DataFrame): The transformed DataFrame suitable for `plot_lines_markers`.
                The column names for the new 'y' series will be descriptive
                of the group combinations.
            - 'y_cols' (List[str]): A list of the names of the new 'y' columns that were created
                or identified for plotting (e.g., `[original_y_col_name]` if no grouping).

    Raises:
        ValueError: If `x_col` or `y_col` are not found in the DataFrame.
        ValueError: If any column in `group_by_cols` is not found in the DataFrame.
    """

    # Validate essential columns
    if x_col not in df.columns:
        raise ValueError(f"x_col '{x_col}' not found in DataFrame columns.")
    if y_col not in df.columns:
        raise ValueError(f"y_col '{y_col}' not found in DataFrame columns.")

    if group_by_cols:
        # Validate group_by_cols
        for col in group_by_cols:
            if col not in df.columns:
                raise ValueError(f"Group-by column '{col}' not found in DataFrame columns.")

        # Ensure all columns needed for pivoting are present
        cols_for_pivot = group_by_cols + [x_col, y_col]
        filtered_and_sorted_df = df[cols_for_pivot]

        # Handle potential non-unique index/columns for pivoting
        # Sort by x_col and group_by_cols to ensure consistent ordering
        sort_cols = [x_col] + group_by_cols
        filtered_and_sorted_df = filtered_and_sorted_df.sort_values(by=sort_cols).reset_index(
            drop=True
        )

        # Convert group_by_cols to categorical to ensure all categories are present as columns
        # even if they have no data for certain x_col values after pivoting.
        for col in group_by_cols:
            # Filter out NaN values from categories before creating Categorical
            all_categories = pd.Series(df[col].unique()).dropna().tolist()
            filtered_and_sorted_df[col] = pd.Categorical(
                filtered_and_sorted_df[col], categories=all_categories
            )

        # Pivot the DataFrame
        pivot_df = filtered_and_sorted_df.pivot_table(
            index=x_col,
            columns=group_by_cols,
            values=y_col,
            dropna=False,  # Keep columns even if they become entirely NaN after pivot
            observed=False,  # Silence FutureWarning, ensure all categories are observed
        )

        # Flatten multi-level column index if group_by_cols has multiple columns
        if isinstance(pivot_df.columns, pd.MultiIndex):
            new_y_cols = ["_".join(map(str, col)).strip() for col in pivot_df.columns.values]
        else:
            # For a single group_by_col, the column name is just the value
            new_y_cols = [str(col).strip() for col in pivot_df.columns.values]

        pivot_df.columns = new_y_cols

        # Reset index to make x_col a regular column again
        pivot_df = pivot_df.reset_index()

        return {"df": pivot_df, "y_cols": new_y_cols}
    else:
        # No grouping, just select x_col and y_col, keeping original y_col name
        result_df = df[[x_col, y_col]]
        new_y_cols = [y_col]  # Keep original y_col name
        return {"df": result_df, "y_cols": new_y_cols}

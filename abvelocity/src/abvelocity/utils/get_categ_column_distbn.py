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

from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def get_categ_column_distbn(
    table_name: str,
    col: str,
    count_distinct_col: Optional[str] = None,
    cursor: Optional[Any] = None,
    print_to_html: Optional[Callable] = None,
    max_num: int = 20,
    conditions: List[str] = ["TRUE = TRUE"],
    file_name: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves and visualizes the distribution of a categorical column in a database table.

    This function executes a SQL query to count the occurrences of each unique value in a specified column,
    optionally filtering the results based on a given condition. It then generates a bar chart showing the
    distribution of the top 'max_num' values, along with an 'Other' category for the remaining values.

    Args:
        table_name (str): The name of the database table.
        col (str): The name of the categorical column to analyze.
        curosr (Any): SQL database cursor.
        count_distinct_col (str): The name of the column containing the distinct values to be counted.
        print_to_html (Optional[Callable], optional):
            A function to print output (text or figures) to an HTML file. Defaults to None.
            It should accept a 'message' and optionally 'color', 'font_size', and 'file_name' arguments.
        max_num (int, optional): The number of top values to display in the chart. Defaults to 20.
        conditions (List[str], optional): A list of SQL conditions to filter the data. Defaults to "TRUE = TRUE" (no filtering).
        file_name (str, optional): The name of the HTML file to write output to. Defaults to None.
        title (str, optional): If not passed, one will be generated based on `table_name`, `col`, `conditions` etc.

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - 'fig' (plotly.graph_objects.Figure): The generated bar chart figure.
            - 'df' (pandas.DataFrame): The full DataFrame containing the counts of all unique values.
            - 'plot_df' (pandas.DataFrame): The DataFrame used for plotting, including the top 'max_num' values and 'Other'.
            - 'num_distict_labels' (int): the number of unique values in the column.
    """
    # If col has the form `string1.string2` SQL will return that col as `string2`.
    # Therefore we define `col0` and use it.
    col0 = col.rsplit(".", 1)[1] if "." in col else col

    condition = " AND ".join(conditions)

    count_str = "COUNT(*) AS count"
    if count_distinct_col:
        count_str = f"COUNT(DISTINCT {count_distinct_col}) AS count"

    query = f"""
    SELECT
        COALESCE({col}, 'NA') AS {col0},
        {count_str}
    FROM
        {table_name}
    WHERE {condition}
    GROUP BY
        {col}
    ORDER BY
        count DESC
    """

    header = (
        f"Getting the distribution of possible values for {col} in {table_name}."
        + f" Counting rows for each value of {col}"
        + f" Conditions: {conditions}"
    )

    if count_distinct_col:
        header = (
            f"Getting the distribution of possible values for {col} in {table_name}."
            + f" Counting distinct {count_distinct_col} for each value of {col}"
            f" Conditions: {conditions}"
        )

    if print_to_html:
        print_to_html(
            header,
            file_name=file_name,
        )
        print_to_html(query, color="grey", font_size=8, file_name=file_name)

    if not cursor:
        return

    df_wrapper = cursor.get_df(query=query)
    df = df_wrapper.df

    if not len(df):
        print("df is empty. df is returning.")
        return

    num_distinct_labels = df[col0].nunique()
    if print_to_html:
        print_to_html(
            f"Total number of distinct labels for {col} across {table_name} is: {num_distinct_labels}",
            file_name=file_name,
        )

    top_k = df.head(max_num).copy()
    other_count = df["count"][max_num:].sum()
    total_count = df["count"].sum()

    if other_count > 0:
        other_row = pd.DataFrame([{col0: "Other", "count": other_count}])
        plot_df = pd.concat([top_k, other_row], ignore_index=True)
    else:
        plot_df = top_k

    plot_df["percent"] = 100.0 * plot_df["count"].copy() / total_count

    fig = px.bar(
        plot_df,
        x=col0,
        y="percent",
        title=f"Distribution of {col} in {table_name}: count_distinct_col: {count_distinct_col}, conditions: {conditions}",
    )

    if print_to_html:
        print_to_html(f"Number of distinct {col0}: {num_distinct_labels}", file_name=file_name)

        print_to_html(fig, file_name=file_name)

    return {
        "fig": fig,
        "df": df,
        "plot_df": plot_df,
        "query": query,
        "num_distict_labels": num_distinct_labels,
    }


def num_distinct_values_per_entity(
    table_name: str,
    entity_cols: List[str],
    count_distinct_col: str,
    cursor: Optional[Any] = None,
    print_to_html: Optional[Callable] = None,
    conditions: List[str] = ["TRUE=TRUE"],
    x_range: Optional[List[float]] = None,
    file_name: Optional[str] = None,
) -> Figure:
    """
    Calculates and visualizes the distribution of the number of distinct values per entity.

    This function executes a SQL query to count the number of distinct values in a specified
    column for each unique combination of entity columns. It then generates a histogram
    showing the distribution of these counts.

    Args:
        table_name (str): The name of the database table.
        entity_cols (List[str]): A list of column names representing the entity.
        count_distinct_col (str): The name of the column containing the distinct values to be counted.
        cursor (Optional[Any]): The database cursor object.
        print_to_html (Optional[Callable], optional): A function to print output (text or figures) to an HTML file.
            It should accept a 'message' and optionally 'color', 'font_size', and 'file_name' arguments.
        conditions (List[str], optional): A list of SQL condition clauses to filter the data. Defaults to `["TRUE=TRUE"]` (no filtering).
        x_range (Optional[List[float]], optional): A list specifying the x-axis range for the histogram. Defaults to None.
        file_name (Optional[str], optional): The name of the HTML file to write output to. Defaults to None.

    Returns:
        Figure: The generated histogram figure.
    """
    condition = " AND ".join(conditions)
    if print_to_html:
        print_to_html(
            f"Counts number of distinct {count_distinct_col} per {entity_cols} and displays the distribution.",
            file_name=file_name,
        )

    entity_cols_str = ",".join(entity_cols)

    query = f"""
    SELECT
        {entity_cols_str},
        COUNT(DISTINCT {count_distinct_col}) AS distinct_count
    FROM
        {table_name}
    WHERE {condition}
    GROUP BY
        {entity_cols_str}
    """

    if print_to_html:
        print_to_html(query, color="grey", font_size=8, file_name=file_name)

    df_wrapper = cursor.get_df(query=query)
    df = df_wrapper.df

    fig = px.histogram(
        df,
        x="distinct_count",
        title=f"Distribution of {count_distinct_col} per {entity_cols} in {table_name}: conditions: {conditions}",
        histnorm="percent",
    )

    if x_range:
        fig.update_layout(xaxis=dict(range=x_range))

    if print_to_html:
        print_to_html(fig, file_name=file_name)

    return {"fig": fig, "df": df, "query": query}

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


from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DiffQuery:
    """
    A class to generate SQL queries that calculate the difference of a response variable
    between two slices determined by the value of a column.

    The difference calculated is (pseudo code):

        - (the value of response_variable at slice_values[0]) - (the value of response_variable at slice_values[1])

    The usage is mainly intended for the case where there is only one value per

        - combination of `groupby_cols + [slice_col]`.

    If this is not the case, for each combination user will get multiple values.
    The query does not prevent this behavior as checking this condition requires accessing data and this
    behavior might be useful in some use cases.



    Attributes:
        table_name (str): The name of the table containing the data. This could be a base query as well e.g.
            `(SELECT * FROM table WHERE country = 'US')`
        response_variable (str): The response variable to compare (e.g., count, percent).
        slice_column (str): The column used to define slices (e.g., country, user_type).
        slice_values (List[str]): The two specific values of the slice_column to compare.
        groupby_cols (Optional[List[str]]): Columns to group by when calculating differences.
        conditions (Optional[List[str]]): Conditions applied to the query (default is None).
        normalize_by_slice (bool): If True, the response variable will be normalized by its slice sum.
    """

    table_name: str
    """The name of the table containing the data.
    This could be a base query as well e.g. `(SELECT * FROM table WHERE country = 'US')`
    """

    response_variable: str
    """The response variable to compare (e.g., count, percent)."""

    slice_column: str
    """The column used to define slices (e.g., country, user_type)."""

    slice_values: List[str]
    """The two specific values of the slice_column to compare."""

    groupby_cols: Optional[List[str]] = None
    """Columns to group by when calculating differences."""

    conditions: Optional[List[str]] = None
    """Conditions applied to the query (default is None)."""

    normalize_by_slice: bool = False
    """If True, the response variable will be normalized by its slice sum, multiplied by 100."""

    query: Optional[str] = None
    """The query can be directly provided here. Also each time `self.gen` is called the result will be stored here."""

    def __post_init__(self):
        """Ensure exactly two slice values are specified and handle conditions."""
        if len(self.slice_values) != 2:
            raise ValueError("Exactly two slice values must be specified.")

    def gen(self) -> str:
        """
        Generates the SQL query to calculate the difference in the response variable.

        Returns:
            str: The generated SQL query.
        """
        slice1, slice2 = self.slice_values

        # Base select clause for each slice, avoiding leading commas
        select_clause_1 = (
            f"SELECT {', '.join(self.groupby_cols) + ',' if self.groupby_cols else ''} "
            f"{self.response_variable} AS response_var_1 "
            f"FROM {self.table_name} "
            f"WHERE {self.slice_column} = '{slice1}'"
        )

        select_clause_2 = (
            f"SELECT {', '.join(self.groupby_cols) + ',' if self.groupby_cols else ''} "
            f"{self.response_variable} AS response_var_2 "
            f"FROM {self.table_name} "
            f"WHERE {self.slice_column} = '{slice2}'"
        )

        # Generate the normalized query if normalization is required
        if self.normalize_by_slice:
            select_clause_1 = (
                f"SELECT {', '.join(self.groupby_cols) + ',' if self.groupby_cols else ''} "
                f"({self.response_variable} / SUM({self.response_variable}) OVER (PARTITION BY {self.slice_column})) * 100 AS response_var_1 "
                f"FROM {self.table_name} "
                f"WHERE {self.slice_column} = '{slice1}'"
            )

            select_clause_2 = (
                f"SELECT {', '.join(self.groupby_cols) + ',' if self.groupby_cols else ''} "
                f"({self.response_variable} / SUM({self.response_variable}) OVER (PARTITION BY {self.slice_column})) * 100 AS response_var_2 "
                f"FROM {self.table_name} "
                f"WHERE {self.slice_column} = '{slice2}'"
            )

        # Handle additional conditions
        if self.conditions is not None:
            where_conditions = " AND ".join(self.conditions)
            select_clause_1 += f" AND {where_conditions}"
            select_clause_2 += f" AND {where_conditions}"

        # Join on groupby columns if they exist
        join_conditions = (
            " AND ".join([f"a.{col} = b.{col}" for col in self.groupby_cols])
            if self.groupby_cols
            else "1=1"
        )

        # Final query to compute the difference
        if self.groupby_cols is None:
            final_select = """a.response_var_1 - b.response_var_2 AS response_diff"""
        else:
            final_select = f"""
            {', '.join([f'a.{col}' for col in self.groupby_cols])},
            a.response_var_1 - b.response_var_2 AS response_diff"""

        diff_query = f"""
        WITH slice1 AS (
            {select_clause_1}
        ),
        slice2 AS (
            {select_clause_2}
        )
        SELECT
            {final_select}
        FROM slice1 a
        JOIN slice2 b
        ON {join_conditions}
        """

        self.query = diff_query.strip()

        return self.query

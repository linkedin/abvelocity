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
class CountSeqQuery:
    """
    A class to generate SQL queries for counting distinct units over sequences with optional conditions.

    Attributes:
        table_name (str): The name of the table to query.
        max_seq_index (int): The maximum sequence index for the generated columns (default is 5).
        count_distinct_col (Optional[str]): The unit for which distinct count is calculated (e.g., 'memberid').
            If None, uses COUNT(*) instead.
        end_value (Optional[str]): If provided, will apply COALESCE to replace NULL values with this value.
        extra_groupby_cols (Optional[List[str]]): Additional columns to include in the GROUP BY clause (default is None).
        conditions (Optional[List[str]]): A list of conditions to be used in the WHERE clause
            (default is None, meaning no WHERE clause).
    """

    table_name: str
    max_seq_index: int = 5
    count_distinct_col: Optional[str] = None
    end_value: Optional[str] = None
    extra_groupby_cols: Optional[List[str]] = None
    conditions: Optional[List[str]] = None
    query: Optional[str] = None
    """The query can be directly provided here. Also each time `self.gen` is called the result will be stored here."""

    def gen(self) -> str:
        """
        Generates the SQL query based on the provided parameters.

        Returns:
            str: The generated SQL query.
        """
        # Create list of columns from s1 to sN
        cols = [f"s{i}" for i in range(1, self.max_seq_index + 1)]

        # Select part of the query, conditionally applying COALESCE if end_value is provided
        if self.end_value:
            select_clause = ", ".join(
                [f"COALESCE({col}, '{self.end_value}') AS {col}" for col in cols]
            )
        else:
            select_clause = ", ".join(cols)

        # Determine the COUNT clause
        if self.count_distinct_col:
            count_clause = f"COUNT(DISTINCT {self.count_distinct_col})"
        else:
            count_clause = "COUNT(*)"

        # Add WHERE clause if conditions are provided
        if self.conditions:
            where_clause = " WHERE " + " AND ".join(self.conditions)
        else:
            where_clause = ""

        # Select both count and percentage
        count_alias = " AS count"
        percent_clause = f"{count_clause} * 100.0 / SUM({count_clause}) OVER () AS percent"

        # Add count and percent clause to the select
        select_clause += f", {count_clause}{count_alias}, {percent_clause}"

        # Basic GROUP BY columns (sequence columns)
        groupby_cols = cols

        # Add extra GROUP BY columns if provided
        if self.extra_groupby_cols:
            groupby_cols.extend(self.extra_groupby_cols)
            select_clause += ", " + ", ".join(self.extra_groupby_cols)

        # Create the GROUP BY clause
        groupby_clause = ", ".join(groupby_cols)

        # Build the query
        self.query = f"""
        SELECT
            {select_clause}
        FROM {self.table_name}
        {where_clause}
        """

        # Add GROUP BY clause
        self.query += f" GROUP BY {groupby_clause}"

        return self.query

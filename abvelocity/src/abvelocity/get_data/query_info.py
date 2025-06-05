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

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QueryInfo:
    """
    Represents information for constructing a SQL query.
    """

    table_name: str
    """
    The name of the table for the SQL query.
    """

    columns: Optional[List[str]] = None
    """
    List of columns to select in the SQL query.
    If both this field and `aggregations` are None, all columns will be selected.

    Columns can be simple names (e.g., 'product_id'), aliases (e.g., 'member_id AS memberid'),
    or complex expressions (e.g., 'LOG(sales_amount) AS log_sales').

    If this variable is set to empty list `[]`, we map it to None to simplify the logic.
    """

    aggregations: Optional[List[str]] = None
    """
    List of aggregation expressions to include in the SQL query.

    Each aggregation function should be a valid SQL aggregation expression
    (e.g., 'SUM(sales_amount)', 'AVG(rating)').

    If this variable is set to empty list `[]`, we map it to None to simplify the logic.
    """

    conditions: Optional[List[str]] = None
    """
    List of conditions to apply in the WHERE clause of the SQL query.

    Each condition should be provided as a string (e.g., "department = 'Sales'", "salary > 50000").

    If this variable is set to empty list `[]`, we map it to None to simplify the logic.
    """

    query: str = field(init=False, default="")
    """
    constructed SQL query.
    """

    def construct(self) -> str:
        """
        Constructs a SQL query based on the provided QueryInfo object and stores it in the `query` attribute.

        Returns:
            str: The constructed SQL query string.

        Raises:
            ValueError: If the inputs do not make sense (e.g., missing table name, invalid aggregation).
            TypeError: If conditions are not provided as a list of strings.
        """
        # Extract parameters from the object
        table_name = self.table_name
        columns = self.columns
        aggregations = self.aggregations
        conditions = self.conditions

        # We map empty `columns`, `aggretations` and `conditions` to None to simplify the logic.
        if columns == []:
            columns = None

        if aggregations == []:
            aggregations = None

        if conditions == []:
            conditions = None

        # Validation
        if not table_name:
            raise ValueError("Table name must be provided.")

        if columns is None and aggregations is None:
            select_columns = "*"
        elif columns is None:
            select_columns = ""
        else:
            select_columns = ", ".join(columns)

        if aggregations is not None:
            select_aggregations = ", ".join(aggregations)
        else:
            select_aggregations = ""

        # Start with the SELECT clause
        select_clause = "SELECT "
        if select_columns != "":
            select_clause += select_columns
            if select_aggregations != "":
                select_clause += ", " + select_aggregations
        else:
            select_clause += select_aggregations

        # FROM clause
        from_clause = f"FROM {table_name}"

        # WHERE clause
        where_clause = ""
        if conditions:
            if not isinstance(conditions, list):
                raise TypeError("Conditions should be provided as a list of strings.")
            where_clause = "WHERE " + " AND ".join(conditions)

        # GROUP BY clause based on columns
        group_by_clause = ""
        if aggregations and columns and len(columns) > 0:
            group_by_clause = "GROUP BY " + ", ".join(str(i + 1) for i in range(len(columns)))

        # Combine all parts
        query_parts = [select_clause, from_clause, where_clause, group_by_clause]
        self.query = " ".join(part for part in query_parts if part)

        return self.query

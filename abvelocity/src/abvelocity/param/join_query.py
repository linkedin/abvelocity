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

import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


@dataclass
class JoinQuery:
    """
    Describes a join operation between two tables or dataframes.
    'on' must be provided as a List[Tuple[str, str]] if not a CROSS join
    when gen() is called.
    'select_left_columns' and 'select_right_columns' are lists of strings,
    where each string should be the exact SQL column expression,
    e.g., "column_name" or "column_name AS alias".
    """

    left_table: Optional[str] = None
    right_table: Optional[str] = None
    on: Optional[List[Tuple[str, str]]] = None
    # join_type is now Optional and has a default, allowing it to be after other defaults
    join_type: Optional[Literal["INNER", "LEFT", "RIGHT", "FULL", "CROSS"]] = field(default="LEFT")

    # Columns to select from the left table.
    # Each item is a string that is the exact SQL column expression.
    select_left_columns: Optional[List[str]] = None

    # Columns to select from the right table.
    # Each item is a string that is the exact SQL column expression.
    select_right_columns: Optional[List[str]] = None

    # Optional name for the output table if a CREATE TABLE AS query is desired.
    output_table_name: Optional[str] = None

    # Optional conditions to apply as WHERE clauses to the left table (as a subquery)
    left_conditions: Optional[List[str]] = None

    # Optional conditions to apply as WHERE clauses to the right table (as a subquery)
    right_conditions: Optional[List[str]] = None

    def __post_init__(self):
        """
        Performs essential validation after initialization.
        Checks for empty left_table, right_table, and 'on' condition
        for non-CROSS joins are now in gen().
        """
        if self.output_table_name is not None and not self.output_table_name.strip():
            raise ValueError("output_table_name cannot be an empty string if provided.")

        # If join_type was explicitly passed as None, default it to "LEFT"
        if self.join_type is None:
            self.join_type = "LEFT"

        if self.join_type == "CROSS":
            if self.on:  # If 'on' is provided for a CROSS join, it's an error
                raise ValueError("CROSS join should not have an 'on' condition.")
            self.on = None  # Ensure 'on' is None for CROSS joins internally
        # No 'on' validation here for non-CROSS joins, it will be done in gen()

        # Validate format of 'on' elements if 'on' is provided (not None)
        if self.on is not None:
            if not all(
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
                for item in self.on
            ):
                raise TypeError(
                    "All 'on' conditions must be tuples of (left_column_name, right_column_name)."
                )

        for col_list in [self.select_left_columns, self.select_right_columns]:
            if col_list is not None:
                if not all(isinstance(item, str) and item for item in col_list):
                    raise TypeError("All select columns must be non-empty strings.")

        for cond_list in [self.left_conditions, self.right_conditions]:
            if cond_list is not None:
                if not all(isinstance(item, str) and item.strip() for item in cond_list):
                    raise TypeError("All conditions must be non-empty strings if provided.")

    def _format_select_columns(self, columns: Optional[List[str]], table_alias: str) -> List[str]:
        """
        Helper to format select column list into SQL snippets, prepending alias if needed.
        """
        formatted_columns = []
        if columns is None:
            formatted_columns.append(f"{table_alias}.*")
        else:
            for col_spec in columns:
                match = re.match(r"^(.*?)\s+AS\s+(.+)$", col_spec, re.IGNORECASE)

                if match:
                    column_part = match.group(1).strip()
                    alias_part = match.group(2).strip()

                    if (
                        "." not in column_part
                        and "*" not in column_part
                        and not re.search(r"\w+\(", column_part)
                        and not column_part.startswith("`")
                    ):
                        formatted_columns.append(f"{table_alias}.{column_part} AS {alias_part}")
                    else:
                        formatted_columns.append(f"{column_part} AS {alias_part}")
                else:
                    if (
                        "." not in col_spec
                        and "*" not in col_spec
                        and not re.search(r"\w+\(", col_spec)
                        and not col_spec.startswith("`")
                    ):
                        formatted_columns.append(f"{table_alias}.{col_spec}")
                    else:
                        formatted_columns.append(col_spec)
        return formatted_columns

    def gen(self, left_alias: str = "t1", right_alias: str = "t2") -> str:
        """
        Generates a full SQL query including SELECT and JOIN clauses.
        If output_table_name is provided, it prepends a DROP TABLE IF EXISTS
        and CREATE TABLE AS clause.
        Incorporates left_conditions and right_conditions as subqueries if provided.

        Args:
            left_alias (str): Alias for the left table in the generated query. Defaults to 't1'.
            right_alias (str): Alias for the right table in the generated query. Defaults to 't2'.

        Returns:
            str: The full SQL query.
        """
        # Critical validation: Ensure left_table and right_table are not None *here*
        if self.left_table is None:
            raise ValueError("left_table must be set before gen is called.")
        if self.right_table is None:
            raise ValueError("right_table must be set before gen is called.")

        # Validate 'on' condition for non-CROSS joins just before generating the query
        # Use self.join_type, which now has a default
        if self.join_type != "CROSS" and (self.on is None or not self.on):
            raise ValueError(
                "Join 'on' condition must be specified and non-empty for non-CROSS joins when gen() is called."
            )

        # Build the left table expression (with WHERE clause if conditions exist)
        left_table_expr = self.left_table
        if self.left_conditions:
            where_clause_left = " AND ".join(self.left_conditions)
            left_table_expr = f"(SELECT * FROM {self.left_table} WHERE {where_clause_left})"

        # Build the right table expression (with WHERE clause if conditions exist)
        right_table_expr = self.right_table
        if self.right_conditions:
            where_clause_right = " AND ".join(self.right_conditions)
            right_table_expr = f"(SELECT * FROM {self.right_table} WHERE {where_clause_right})"

        # 1. Generate SELECT clause
        select_parts = []
        select_parts.extend(self._format_select_columns(self.select_left_columns, left_alias))
        select_parts.extend(self._format_select_columns(self.select_right_columns, right_alias))

        if not select_parts:
            select_clause = f"SELECT {left_alias}.*, {right_alias}.*"
        else:
            select_clause = f"SELECT {', '.join(select_parts)}"

        # 2. Generate FROM and JOIN clause
        from_clause = f"FROM {left_table_expr} AS {left_alias}"
        join_clause = f"{self.join_type} JOIN {right_table_expr} AS {right_alias}"

        base_query = ""
        if self.join_type == "CROSS":
            base_query = f"{select_clause} {from_clause} {join_clause}"
        else:
            on_conditions = []
            # 'on' is guaranteed to be not None and not empty here due to validation above
            for l_col, r_col in self.on:
                left_col_ref = f"{left_alias}.{l_col}"
                right_col_ref = f"{right_alias}.{r_col}"
                on_conditions.append(f"{left_col_ref} = {right_col_ref}")
            base_query = f"{select_clause} {from_clause} {join_clause} ON " + " AND ".join(
                on_conditions
            )

        # 3. Prepend DROP TABLE IF EXISTS and CREATE TABLE AS if output_table_name is provided
        final_query = ""
        if self.output_table_name:
            # Use the output_table_name directly without adding quotes.
            # It's assumed the user will provide a valid (potentially quoted) name.
            table_name_for_query = self.output_table_name.strip()

            drop_clause = f"DROP TABLE IF EXISTS {table_name_for_query};"
            create_clause = f"CREATE TABLE {table_name_for_query} AS {base_query};"
            final_query = f"{drop_clause}\n{create_clause}"
        else:
            final_query = f"{base_query};"

        return final_query

    def __str__(self):
        # Handle cases where left_table or right_table might be None for __str__
        left_table_str = self.left_table if self.left_table is not None else "[LEFT TABLE NOT SET]"
        right_table_str = (
            self.right_table if self.right_table is not None else "[RIGHT TABLE NOT SET]"
        )

        on_str = ""
        if self.on:  # This check handles both None and empty list gracefully
            on_parts = []
            for l_col, r_col in self.on:
                if l_col == r_col:
                    on_parts.append(f"'{l_col}'")
                else:
                    on_parts.append(f"('{l_col}', '{r_col}')")
            on_str = f"on {', '.join(on_parts)}".strip()
        elif self.join_type == "CROSS":
            on_str = ""  # No 'on' clause for CROSS join
        else:
            on_str = "ON conditions not set"  # Indicate missing ON for non-CROSS joins

        return f"{self.join_type} JOIN {right_table_str} " f"to {left_table_str} {on_str}".strip()

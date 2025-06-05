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
class ConversionQuery:
    """
    A class to generate SQL queries for calculating conversion rates
    based on the presence of specific values within an array column.

    The conversion is calculated as:
        (Count of entities where `array_col` contains any value from `numerator_list`) /
        (Count of entities where `array_col` contains any value from `denominator_list`)

    Counts can be distinct on a `count_distinct_col` (e.g., user_id) or simple row counts.

    Attributes:
        table_name (str): The name of the table containing the data.
            This can be a base query like `(SELECT * FROM my_table WHERE ...)`
        array_col (str): The name of the array column to analyze (e.g., 'event_tags').
        numerator_list (List[str]): A list of values in the array_col
            that define the numerator (e.g., ['product_view']).
        denominator_list (List[str]): A list of values in the array_col
            that define the denominator (e.g., ['page_load']).
        count_distinct_col (Optional[str]): The column to use for COUNT(DISTINCT).
            If provided, counts unique entities (e.g., 'user_id').
            If None, counts rows satisfying the conditions.
        conditions (Optional[List[str]]): A list of individual SQL WHERE conditions
            (e.g., ["event_date >= DATE     '2023-01-01'", "device_type = 'mobile'"]).
            These will be joined by ' AND '.
        group_by_cols (Optional[List[str]]): A list of columns to group the results by
            (e.g., ['DATE(event_timestamp)', 'country']).
    """

    table_name: str
    """The name of the table containing the data."""

    array_col: str
    """The name of the array column containing the events/tags."""

    numerator_list: List[str]
    """A list of values in the array_col that define the numerator."""

    denominator_list: List[str]
    """A list of values in the array_col that define the denominator."""

    count_distinct_col: Optional[str] = None
    """The column to use for COUNT(DISTINCT).
    If provided, counts unique entities (e.g., 'user_id').
    If None, counts rows satisfying the conditions.
    """

    # Using default_factory for mutable defaults like lists is a dataclass best practice
    conditions: Optional[List[str]] = field(default_factory=list)
    """A list of individual SQL WHERE conditions (e.g., ["event_date >= DATE '2023-01-01'", "device_type = 'mobile'"]).
    These will be joined by ' AND '.
    """

    group_by_cols: Optional[List[str]] = field(default_factory=list)
    """A list of columns to group the results by (e.g., ['DATE(event_timestamp)', 'country'])."""

    def gen(self) -> str:
        """
        Generates a Presto SQL query string to calculate a conversion rate based
        on the parameters provided during the object's initialization.
        The query now also includes the raw numerator and denominator counts.

        Returns:
            str: The generated Presto SQL query.
        """

        if not isinstance(self.numerator_list, list) or not self.numerator_list:
            raise ValueError("numerator_list must be a non-empty list.")
        if not isinstance(self.denominator_list, list) or not self.denominator_list:
            raise ValueError("denominator_list must be a non-empty list.")
        if self.count_distinct_col is not None and not isinstance(self.count_distinct_col, str):
            raise TypeError("count_distinct_col must be a string or None.")
        if not isinstance(self.conditions, list) or not all(
            isinstance(cond, str) for cond in self.conditions
        ):
            raise TypeError("conditions must be a list of strings or None.")
        if not isinstance(self.group_by_cols, list) or not all(
            isinstance(col, str) for col in self.group_by_cols
        ):
            raise TypeError("group_by_cols must be a list of strings or None.")

        numerator_sql = ", ".join(
            [f"'{item}'" if isinstance(item, str) else str(item) for item in self.numerator_list]
        )
        denominator_sql = ", ".join(
            [f"'{item}'" if isinstance(item, str) else str(item) for item in self.denominator_list]
        )

        # Conditions to check if the array contains any of the possibilities
        # Changed from IS_EMPTY to CARDINALITY > 0
        numerator_array_condition = (
            f"CARDINALITY(ARRAY_INTERSECT({self.array_col}, ARRAY[{numerator_sql}])) > 0"
        )
        denominator_array_condition = (
            f"CARDINALITY(ARRAY_INTERSECT({self.array_col}, ARRAY[{denominator_sql}])) > 0"
        )

        count_expr_numerator = ""
        count_expr_denominator = ""

        if self.count_distinct_col:
            count_expr_numerator = (
                f"COUNT(DISTINCT CASE WHEN {numerator_array_condition} "
                f"THEN {self.count_distinct_col} ELSE NULL END)"
            )
            count_expr_denominator = (
                f"COUNT(DISTINCT CASE WHEN {denominator_array_condition} "
                f"THEN {self.count_distinct_col} ELSE NULL END)"
            )
        else:
            count_expr_numerator = (
                f"COUNT(CASE WHEN {numerator_array_condition} THEN 1 ELSE NULL END)"
            )
            count_expr_denominator = (
                f"COUNT(CASE WHEN {denominator_array_condition} THEN 1 ELSE NULL END)"
            )

        select_parts = []
        if self.group_by_cols:
            select_parts.extend(self.group_by_cols)

        # Add numerator and denominator counts to the SELECT statement
        select_parts.append(f"{count_expr_numerator} AS numer_count")
        select_parts.append(f"{count_expr_denominator} AS denom_count")

        # Calculate conversion rate, handling division by zero by returning NULL
        conversion_expression = (
            f"CAST({count_expr_numerator} AS DOUBLE) / NULLIF({count_expr_denominator}, 0)"
        )
        select_parts.append(f"{conversion_expression} AS conversion_rate")

        query = f"SELECT {', '.join(select_parts)} FROM {self.table_name}"

        if self.conditions:
            query += f" WHERE {' AND '.join(self.conditions)}"

        if self.group_by_cols:
            query += f" GROUP BY {', '.join(self.group_by_cols)}"

        self.query = query.strip()

        return self.query

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

CONVERSION_RATE = "conversion_rate"
"""The default name for the conversion rate column."""

NUMER_COUNT = "numer_count"
"""The default name for the numerator of conversion rate computation."""

DENOM_COUNT = "denom_count"
"""Th edefault for the denominator of conversion rate computation."""


@dataclass
class ConversionQuery:
    """
    A class to generate SQL queries for calculating conversion rates
    based on the presence of specific values within an array column.

    The conversion is calculated as:
        (Count of entities where `array_col` contains specified numerator values) /
        (Count of entities where `array_col` contains specified denominator values)

    If `numerator_list` or `denominator_list` is None, all rows are considered
    to satisfy that respective condition.

    `require_all_numerator` and `require_all_denominator` control whether *all*
    elements from the respective list must be present in `array_col` (AND logic)
    or *any* of them (OR logic, default).

    Counts can be distinct on a `count_distinct_col` (e.g., user_id) or simple row counts.

    Attributes:
        table_name (str): The name of the table containing the data.
            This can be a base query like `(SELECT * FROM my_table WHERE ...)`
        array_col (str): The name of the array column to analyze (e.g., 'event_tags').
        numerator_list (Optional[List[str]]): A list of values in the array_col
            that define the numerator (e.g., ['product_view']).
            If None, all rows are included in the numerator count.
        denominator_list (Optional[List[str]]): A list of values in the array_col
            that define the denominator (e.g., ['page_load']).
            If None, all rows are included in the denominator count.
        require_all_numerator (bool): If True, all values in `numerator_list` must be
            present in `array_col`. If False, any value is sufficient. Defaults to False.
        require_all_denominator (bool): If True, all values in `denominator_list` must be
            present in `array_col`. If False, any value is sufficient. Defaults to False.
        count_distinct_col (Optional[str]): The column to use for COUNT(DISTINCT).
            If provided, counts unique entities (e.g., 'user_id').
            If None, counts rows satisfying the conditions.
        conditions (Optional[List[str]]): A list of individual SQL WHERE conditions
            (e.g., ["event_date >= DATE '2023-01-01'", "device_type = 'mobile'"]).
            These will be joined by ' AND '.
        group_by_cols (Optional[List[str]]): A list of columns to group the results by
            (e.g., ['DATE(event_timestamp)', 'country']).
    """

    table_name: str
    """The name of the table containing the data."""

    array_col: str
    """The name of the array column containing the events/tags."""

    numerator_list: Optional[List[str]]
    """A list of values in the array_col that define the numerator, or None to count all rows."""

    denominator_list: Optional[List[str]]
    """A list of values in the array_col that define the denominator, or None to count all rows."""

    require_all_numerator: bool = False
    """If True, all values in `numerator_list` must be present. If False, any is sufficient."""

    require_all_denominator: bool = False
    """If True, all values in `denominator_list` must be present. If False, any is sufficient."""

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

    col_name: Optional[str] = CONVERSION_RATE
    """The name for the conversion rate column."""

    def gen(self) -> str:
        """
        Generates a Presto SQL query string to calculate a conversion rate based
        on the parameters provided during the object's initialization.
        The query now also includes the raw numerator and denominator counts.

        Returns:
            str: The generated Presto SQL query.
        """

        # Validate numerator_list
        if self.numerator_list is not None and (
            not isinstance(self.numerator_list, list) or not self.numerator_list
        ):
            raise ValueError("numerator_list must be a non-empty list or None.")
        # Validate denominator_list
        if self.denominator_list is not None and (
            not isinstance(self.denominator_list, list) or not self.denominator_list
        ):
            raise ValueError("denominator_list must be a non-empty list or None.")

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

        def _get_array_condition(event_list: Optional[List[str]], require_all: bool) -> str:
            """
            Helper function to generate the SQL condition for array containment.
            """
            if event_list is None:
                return "TRUE"
            if not event_list:
                # This case is already handled by outer validation, but defensive check
                raise ValueError("List cannot be empty if not None.")

            # Define single quote characters outside the f-string expression
            _s_quote = "'"
            _d_s_quote = "''"

            # Prepare the SQL list string for ARRAY function
            sql_list_items = ", ".join(
                [
                    (
                        f"{_s_quote}{item.replace(_s_quote, _d_s_quote)}{_s_quote}"
                        if isinstance(item, str)
                        else str(item)
                    )
                    for item in event_list
                ]
            )

            # Condition using ARRAY_INTERSECT for both 'any' and 'all' logic
            array_intersect_clause = (
                f"CARDINALITY(ARRAY_INTERSECT({self.array_col}, ARRAY[{sql_list_items}]))"
            )

            if require_all:
                # For 'all' logic, cardinality of intersection must equal the length of the list
                return f"{array_intersect_clause} = {len(event_list)}"
            else:
                # For 'any' logic, cardinality of intersection must be greater than 0
                return f"{array_intersect_clause} > 0"

        numerator_array_condition = _get_array_condition(
            self.numerator_list, self.require_all_numerator
        )
        denominator_array_condition = _get_array_condition(
            self.denominator_list, self.require_all_denominator
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
        select_parts.append(f"{count_expr_numerator} AS {NUMER_COUNT}")
        select_parts.append(f"{count_expr_denominator} AS {DENOM_COUNT}")

        # Calculate conversion rate, handling division by zero by returning NULL
        conversion_expression = (
            f"CAST({count_expr_numerator} AS DOUBLE) / NULLIF({count_expr_denominator}, 0)"
        )
        select_parts.append(f"{conversion_expression} AS {self.col_name}")

        query = f"SELECT {', '.join(select_parts)} FROM {self.table_name}"

        if self.conditions:
            query += f" WHERE {' AND '.join(self.conditions)}"

        if self.group_by_cols:
            query += f" GROUP BY {', '.join(self.group_by_cols)}"

        self.query = query.strip()

        return self.query

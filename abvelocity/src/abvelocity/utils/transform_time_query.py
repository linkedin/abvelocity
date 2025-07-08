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
from typing import Optional


def get_time_conversion_expression(
    time_col: str,
    time_col_format: Optional[str] = None,
    new_time_col: Optional[str] = None,
    time_unit: Optional[str] = None,
) -> str:
    """
    Generates the SQL expression for converting a time column to a timestamp,
    optionally truncating it to the specified time unit.

    Args:
        time_col (str): The name of the original time column.
        time_col_format (Optional[str]): The format of the original time column ('unix_ms', 'unix_s', 'string', 'timestamp').
        new_time_col (Optional[str]): The desired name for the new timestamp column.
        time_unit (Optional[str]): The unit to truncate the timestamp to ('second', 'minute', 'hour', 'day').
            If None, no truncation is applied. Defaults to None.

    Returns:
        str: The SQL expression to be used in a SELECT statement.

    Raises:
        ValueError: If an unsupported time_col_format or time_unit is provided.
    """
    base_expr = ""
    if not time_col_format or time_col_format == "unix_ms":
        base_expr = f"FROM_UNIXTIME({time_col} / 1000.0)"
    elif time_col_format == "unix_s":
        base_expr = f"FROM_UNIXTIME({time_col})"
    elif time_col_format == "string":
        # Assumes a format like 'YYYY-MM-DD HH:MM:SS' which CAST AS TIMESTAMP can handle
        base_expr = f"CAST({time_col} AS TIMESTAMP)"
    elif time_col_format == "timestamp":
        base_expr = time_col
    else:
        raise ValueError(
            f"Unsupported time_col_format: {time_col_format}. Valid options are 'unix_ms', 'unix_s', 'string', 'timestamp'."
        )

    if not new_time_col:
        new_time_col = "converted_time"

    # Apply truncation if a time_unit is specified
    if time_unit:
        # Map common time units to uppercase for SQL functions
        sql_time_unit = time_unit.upper()
        allowed_units = {
            "SECOND",
            "MINUTE",
            "HOUR",
            "DAY",
        }  # Singular and plural handled by uppercase conversion
        if sql_time_unit not in allowed_units:
            # Handle plural forms by removing 'S' if present
            if sql_time_unit.endswith("S") and sql_time_unit[:-1] in allowed_units:
                sql_time_unit = sql_time_unit[:-1]
            else:
                raise ValueError(
                    f"Unsupported time_unit: '{time_unit}'. Valid options are 'second', 'minute', 'hour', 'day' (and their plurals)."
                )
        return f"DATE_TRUNC('{sql_time_unit}', {base_expr}) AS {new_time_col}"
    else:
        return f"{base_expr} AS {new_time_col}"


@dataclass
class TransformTimeQuery:
    """
    Dataclass to generate a SQL query to transform and add a new timestamp column.

    This class encapsulates the logic for converting an existing time column to a
    timestamp format and optionally truncating it to a specified time unit.
    The generated query can be used as a subquery or for creating new tables/views.
    """

    table_name: str
    """The name of the original table or a subquery (e.g., '(SELECT * FROM my_table WHERE condition)')."""
    original_time_col: str
    """The name of the original time column in the table."""
    time_col_format: str
    """The format of the original time column ('unix_ms', 'unix_s', 'string', 'timestamp')."""
    new_time_col: str
    """The desired name for the new timestamp column."""
    time_unit: Optional[str] = None
    """The unit to truncate the timestamp to ('second', 'minute', 'hour', 'day').
    If None, no truncation is applied. Defaults to None."""

    def gen(self) -> str:
        """
        Generates the final SQL query to add a new column with a converted timestamp.

        Returns:
            str: The generated SQL query.
        """
        # Call the standalone utility function
        time_expr = get_time_conversion_expression(
            self.original_time_col, self.time_col_format, self.new_time_col, self.time_unit
        )
        # Enclosing the entire subquery in parentheses
        query = f"(SELECT *,\n" f"       {time_expr}\n" f"FROM ({self.table_name}))"
        return query

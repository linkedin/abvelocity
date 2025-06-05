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
from dataclasses import dataclass
from typing import Optional


@dataclass
class EventTable:
    table_name: str
    """SQL table name or query."""
    event_label: str
    """Event label."""
    select_cols: list[str]
    """List of column names to be selected in the query."""
    date_col: str = "datepartition"
    """Date column in the table."""
    conditions: Optional[list[str]] = None
    """SQL conditions given as a list of strings."""
    output_table_name: Optional[str] = None
    """The name of output table created based on these specs (if / when created)"""


def convert_to_snake_case(input_string):
    # Helper function to replace all periods with underscores
    input_string = input_string.replace(".", "_")

    # Convert camel case to snake case
    # Insert an underscore before each uppercase letter that follows a lowercase letter
    input_string = re.sub(r"([a-z])([A-Z])", r"\1_\2", input_string)

    # Convert the string to lowercase and return
    return input_string.lower()


def gen_event_query(
    event_table: EventTable,
    start_date: str,
    end_date: str,
    create_table_prefix: Optional[str] = None,
) -> str:
    """
    Generates a SQL query to extract tracking event data from a specified table.
    This assumes a row in the target table is logging an occurrence of an event: `event_label`.
    The resulting created table name (if created) will be `f"{create_table_prefix}_{convert_to_snake_case(event_table.table_name)}"`.

    Args:
        event_table (EventTable): Dataclass containing table details, event label, columns, and conditions.
        start_date (str): The start date for filtering records (format: 'YYYY-MM-DD-00').
        end_date (str): The end date for filtering records (format: 'YYYY-MM-DD-00').
        create_table_prefix (Optional[str], optional): If provided, a new table will be created with this prefix.

    Returns:
        str: The generated SQL query as a string.

    Raises:
        ValueError: If any required parameters (table_name, event_label, start_date, end_date) are empty or None.
    """
    if not all([event_table.table_name, event_table.event_label, start_date, end_date]):
        raise ValueError(
            "All required parameters (table_name, event_label, start_date, end_date) must be provided."
        )

    query = f"""
        SELECT
            {', '.join(event_table.select_cols)},
            '{event_table.event_label}' AS event
        FROM {event_table.table_name}
        WHERE TRUE
            AND {event_table.date_col} BETWEEN '{start_date}' AND '{end_date}'"""

    if event_table.conditions:
        query += " AND " + " AND ".join(event_table.conditions)

    if create_table_prefix:
        output_table_name = f"{create_table_prefix}_{convert_to_snake_case(event_table.table_name)}_{convert_to_snake_case(event_table.event_label)}"
        event_table.output_table_name = output_table_name
        query_prefix = f"""
        DROP TABLE IF EXISTS {output_table_name};
        CREATE TABLE IF NOT EXISTS {output_table_name} AS"""
        query = query_prefix + query

    return query

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

DATE_COL_DEFAULT = "datepartition"


@dataclass
class EventTable:
    table_name: Optional[str] = None
    """SQL table name or a name to describe it in case `table_query` is passed below.
    This name is used to also construct the name of the output table: `output_table_name` as follows:
    `f"{create_table_prefix}_{convert_to_snake_case(event_table.table_name)}_{convert_to_snake_case(event_table.event_label)}"`
    """
    event_label: Optional[str] = None
    """Event label."""
    event_color: Optional[str] = None
    """Event color used in viz."""
    select_cols: Optional[list[str]] = None
    """List of column names to be selected in the query."""
    date_col: Optional[str] = None
    """Date column in the table.
    If not specified, it can be propagated from `common_info`.
    If still not set after propagation,
    it defaults to `DATE_COL_DEFAULT` in the query generation."""
    table_query: Optional[str] = None
    """This is the table query. Note that we allow queries
    rather than only relying on table_name in case some transormations are needed.
    If this is not passed we assume no transformation is needed and use `table_name`.
    """
    conditions: Optional[list[str]] = None
    """SQL conditions given as a list of strings."""
    output_table_name: Optional[str] = None
    """The name of output table created based on these specs (if / when created)"""
    start_date: Optional[str] = None
    """Start date for filtering records. If provided, it overrides the sequence start_date."""
    end_date: Optional[str] = None
    """End date for filtering records. If provided, it overrides the sequence end_date."""


@dataclass
class MultiEventTable:
    event_tables: list[EventTable]
    common_info: Optional[EventTable] = None

    def __post_init__(self):
        """
        Ensures that all event_labels in the event_tables list are unique.
        Raises a ValueError if duplicate event_labels are found.
        Propagation of common info is handled by the separate `propogate_common_info` method.
        """
        event_labels = []
        for event_table in self.event_tables:
            if event_table.event_label is not None:
                event_labels.append(event_table.event_label)

        # Check for uniqueness
        if len(event_labels) != len(set(event_labels)):
            duplicate_labels = {label for label in event_labels if event_labels.count(label) > 1}
            raise ValueError(
                f"All event_labels within 'event_tables' must be unique. Duplicates found: {duplicate_labels}"
            )

    def propogate_common_info(self):
        """propogates the info from `common_info` for missing fields of each event_table"""
        # We've removed the early `return` here so that defaults can be applied even if common_info is None.
        for event_table in self.event_tables:
            # Propagate table_name
            if (
                event_table.table_name is None
                and self.common_info is not None
                and self.common_info.table_name is not None
            ):
                event_table.table_name = self.common_info.table_name

            # Propagate select_cols
            if (
                event_table.select_cols is None
                and self.common_info is not None
                and self.common_info.select_cols is not None
            ):
                event_table.select_cols = self.common_info.select_cols

            # Propagate date_col
            if event_table.date_col is None:
                if self.common_info is not None and self.common_info.date_col is not None:
                    event_table.date_col = self.common_info.date_col
                else:
                    # If both the event table and common info have no date_col, use the default.
                    event_table.date_col = DATE_COL_DEFAULT

            # Propagate table_query
            if (
                event_table.table_query is None
                and self.common_info is not None
                and self.common_info.table_query is not None
            ):
                event_table.table_query = self.common_info.table_query

            # Propagate conditions: Add common conditions to existing ones if they exist
            if self.common_info is not None and self.common_info.conditions is not None:
                if event_table.conditions is None:
                    event_table.conditions = self.common_info.conditions
                else:
                    event_table.conditions = self.common_info.conditions + event_table.conditions

            # Propagate start_date
            if (
                event_table.start_date is None
                and self.common_info is not None
                and self.common_info.start_date is not None
            ):
                event_table.start_date = self.common_info.start_date

            # Propagate end_date
            if (
                event_table.end_date is None
                and self.common_info is not None
                and self.common_info.end_date is not None
            ):
                event_table.end_date = self.common_info.end_date

            # event_label, event_color, and output_table_name are typically specific
            # to each EventTable and thus not propagated from common_info.


def convert_to_snake_case(input_string: str) -> str:
    """
    Converts a given string to snake_case.
    Replaces periods with underscores and converts camel case to snake case.
    """
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
    """Generates a SQL query to extract tracking event data from a specified table.
    This assumes a row in the target table is logging an occurrence of an event: `event_label`.
    The resulting created table name (if created) will be
    `f"{create_table_prefix}_{convert_to_snake_case(event_table.table_name)}_{convert_to_snake_case(event_table.event_label)}"`.

    Args:
        event_table (EventTable): Dataclass containing table details, event label, columns, and conditions.
        start_date (str): The start date for filtering records (format: 'YYYY-MM-DD-00'). This will be used if `event_table.start_date` is not provided.
        end_date (str): The end date for filtering records (format: 'YYYY-MM-DD-00'). This will be used if `event_table.end_date` is not provided.
        create_table_prefix (Optional[str], optional): If provided, a new table will be created with this prefix.

    Returns:
        str: The generated SQL query as a string.

    Raises:
        ValueError: If any required parameters (table_name or table_query, event_label, start_date, end_date) are empty or None.
    """
    table_start_date = event_table.start_date or start_date
    table_end_date = event_table.end_date or end_date

    # Use the date_col from the event table. At this point, it should be set
    # by user input, propagation from common_info, or the default.
    date_col_to_use = event_table.date_col

    # Validate that either table_name or table_query is provided
    if not (event_table.table_name or event_table.table_query):
        raise ValueError("Either 'table_name' or 'table_query' must be provided in EventTable.")

    if not all([event_table.event_label, table_start_date, table_end_date]):
        raise ValueError(
            "All required parameters (event_label, start_date, end_date) must be provided."
        )

    # Determine the source for the query (table_query takes precedence if both are present)
    query_table = (
        f"({event_table.table_query})" if event_table.table_query else event_table.table_name
    )

    # Select columns: if select_cols is None or empty, assume all columns (SELECT *)
    select_clause = ", ".join(event_table.select_cols) if event_table.select_cols else "*"

    query = f"""
        SELECT
            {select_clause},
            '{event_table.event_label}' AS event
        FROM {query_table}
        WHERE TRUE
            AND {date_col_to_use} BETWEEN '{table_start_date}' AND '{table_end_date}'"""

    if event_table.conditions:
        query += " AND " + " AND ".join(event_table.conditions)

    if create_table_prefix:
        # Ensure event_table.table_name is not None for output table naming if table_query is used
        # You might need a more robust naming convention if table_name is Optional and table_query is used.
        # A safer approach might involve using event_table.event_label for naming if table_name is absent.
        name_for_output = (
            event_table.table_name if event_table.table_name else "custom_query"
        )  # Placeholder if table_name is None but table_query is used

        output_table_name = f"{create_table_prefix}_{convert_to_snake_case(name_for_output)}_{convert_to_snake_case(event_table.event_label)}"
        event_table.output_table_name = output_table_name
        query_prefix = f"""
        DROP TABLE IF EXISTS {output_table_name};
        CREATE TABLE IF NOT EXISTS {output_table_name} AS"""
        query = query_prefix + query

    return query

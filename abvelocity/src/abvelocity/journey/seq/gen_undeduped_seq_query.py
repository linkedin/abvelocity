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


from typing import List, Optional


def gen_undeduped_seq_query(
    table_name: str,
    partition_by_cols: List[str],
    time_col: str = "time",
    event_col: str = "event",
    order_list: Optional[List[str]] = None,
) -> str:
    """
    Generate a SQL query to create ordered event sequences without deduplication.

    This function generates a SQL query that processes a table to achieve the following:
    1. Aggregate all events within each partition into an ordered sequence.
    2. Allow for custom ordering of events within the sequence using `order_list`.
       If `order_list` is None, events are ordered by `time_col`.
    3. Calculate the start time (min time), end time (max time), and length of each sequence.

    Args:
        table_name (str):
            The name of the table from which events will be retrieved.
            Must contain `time_col`, `event_col`, and `partition_by_cols`.
        partition_by_cols (List[str]):
            A list of column names used to partition the data. Events will be grouped based on these columns.
        time_col (str, optional):
            The name of the timestamp column used for default ordering and calculating bounds. Defaults to "time".
        event_col (str, optional):
            The name of the column containing event identifiers. Defaults to "event".
        order_list (List[str], optional):
            A list specifying the desired order of events in the output array.
            If provided, the query will order events according to this list first, then by time.
            If None, events are ordered solely by `time_col`. Defaults to None.

    Returns:
        str: The generated SQL query (without internal SQL comments) to produce ordered event sequences.

    Raises:
        ValueError: If `partition_by_cols` is not provided as a non-empty list. (Note: Validation removed)
    """
    # Create the PARTITION BY and SELECT clauses dynamically
    # Assumes partition_by_cols is a non-empty list of strings
    partition_by_clause = ", ".join(partition_by_cols)
    select_partition_cols = ", ".join(partition_by_cols)

    # Determine the ORDER BY clause for the ARRAY_AGG function
    if order_list is None:
        # Default order by time if no custom list is provided
        order_by_clause_agg = f"{time_col}"
    else:
        # Create a CASE statement for custom ordering
        # Events in the list get priority, others are appended, ordered by time
        case_statements = []
        for i, event in enumerate(order_list):
            # Ensure proper quoting for string literals in SQL
            # Basic escaping for single quotes, might need adjustment for specific SQL dialects
            escaped_event = event.replace("'", "''")
            case_statements.append(f"WHEN {event_col} = '{escaped_event}' THEN {i}")
        # Assign a large number to events not in the list so they come last
        order_by_clause_agg = (
            f"CASE {' '.join(case_statements)} ELSE {len(order_list)} END, {time_col}"
        )

    # Generate the SQL query without internal comments, matching original formatting
    query = f"""
    WITH AggregatedSequences AS (
        SELECT
            {select_partition_cols},
            ARRAY_AGG({event_col} ORDER BY {order_by_clause_agg}) AS event_seq,
            MIN({time_col}) AS seq_start_time,
            MAX({time_col}) AS seq_end_time
        FROM {table_name}
        GROUP BY {partition_by_clause}
    )
    SELECT
        {select_partition_cols},
        event_seq,
        seq_start_time,
        seq_end_time,
        cardinality(event_seq) AS seq_length
    FROM AggregatedSequences
    """
    # Return the query string directly
    return query

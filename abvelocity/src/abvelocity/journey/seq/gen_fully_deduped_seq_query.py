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


from typing import Optional


def gen_fully_deduped_seq_query(
    table_name: str,
    partition_by_cols: list[str],
    time_col: str = "time",
    event_col: str = "event",
    order_list: Optional[list[str]] = None,
) -> str:
    """
    Generate a SQL query to remove all duplicate events within each partition and create ordered event sequences.

    This function generates a SQL query that processes a table to achieve the following:
    1. Rank events within partitions based on their timestamp to identify duplicates.
    2. Remove duplicate events, keeping only the first occurrence within each partition.
    3. Aggregate the unique events into an ordered sequence for each partition, with optional custom ordering.

    Args:
        table_name (str):
            The name of the table from which events will be retrieved.
            The table must contain the following columns:
                - A timestamp column specified by `time_col`, used to order events chronologically.
                - An event column specified by `event_col`, containing the event names.
                - Columns specified in `partition_by_cols` used to segment the data.

        partition_by_cols (list[str]):
            A list of column names used to partition the data. Events will be grouped based on these columns.
            These columns must exist in the specified table.

        time_col (str, optional):
            The name of the timestamp column used for ordering events. Default is "time".

        event_col (str, optional):
            The name of the column containing event identifiers. Default is "event".

        order_list (list[str], optional):
            A list specifying the desired order of events in the output array.
            If provided, the query will order the events in the array according to this list.
            If None, the events will be ordered by their timestamp.

    Returns:
        str: The generated SQL query to produce fully deduplicated event sequences.

    Raises:
        ValueError: If `partition_by_cols` is not provided as a list.
    """
    # Ensure `partition_by_cols` is a list
    if not isinstance(partition_by_cols, list):
        raise ValueError("partition_by_cols must be a list of column names.")

    # Create the PARTITION BY clause dynamically
    partition_by_clause = ", ".join(partition_by_cols)

    # Generate the SQL query to generate fully deduplicated event sequences
    if order_list is None:
        order_by_clause = f"{time_col}"
    else:
        # Create a CASE statement to define the custom order.
        case_statements = []
        for i, event in enumerate(order_list):
            case_statements.append(f"WHEN {event_col} = '{event}' THEN {i}")
        order_by_clause = f"CASE {' '.join(case_statements)} ELSE {len(order_list)} END, {time_col}"

    query = f"""
    WITH RankedEvents AS (
        SELECT
            {', '.join(partition_by_cols)},
            {time_col},
            {event_col},
            ROW_NUMBER() OVER (PARTITION BY {partition_by_clause}, {event_col} ORDER BY {time_col}) AS event_rank
        FROM {table_name}
    ),
    DeduplicatedEvents AS (
        SELECT
            {', '.join(partition_by_cols)},
            {time_col},
            {event_col}
        FROM RankedEvents
        WHERE event_rank = 1
    ),
    ArrayedEvents AS (
        SELECT
            {', '.join(partition_by_cols)},
            ARRAY_AGG({event_col} ORDER BY {order_by_clause}) AS event_seq,
            MIN({time_col}) AS seq_start_time,
            MAX({time_col}) AS seq_end_time
        FROM DeduplicatedEvents
        GROUP BY {', '.join(partition_by_cols)}
    )
    SELECT
        {', '.join(partition_by_cols)},
        event_seq,
        seq_start_time,
        seq_end_time,
        cardinality(event_seq) AS seq_length
    FROM ArrayedEvents
    """

    return query

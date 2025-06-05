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


def gen_event_count_map_query(
    table_name: str,
    partition_by_cols: List[str],
    event_col: str = "event",
    time_col: Optional[str] = None,  # Added optional time_col
) -> str:
    """
    Generate a Presto SQL query to count event occurrences within partitions
    and return the result as a map, optionally including sequence start/end times
    and the total sequence length.

    This function generates a SQL query that processes a table to:
    1. Group events by the specified partition columns and the event column.
    2. Count the occurrences of each event within each partition.
    3. Aggregate these counts into a MAP data type for each partition.
    4. Calculate the total number of events (sequence length) for each partition.
    5. Optionally, calculate the minimum (start) and maximum (end) time
       for each partition if `time_col` is provided.

    Args:
        table_name (str):
            The name of the table from which events will be retrieved.
            Must contain `event_col` and `partition_by_cols`.
            If `time_col` is specified, it must also contain that column.
        partition_by_cols (List[str]):
            A list of column names used to partition the data. Events will be
            counted within these groups.
        event_col (str, optional):
            The name of the column containing event identifiers.
            Defaults to "event".
        time_col (Optional[str], optional):
            The name of the timestamp column used for calculating sequence bounds.
            If provided, 'start_seq_time' and 'end_seq_time' columns are added.
            Defaults to None.

    Returns:
        str: The generated Presto SQL query to produce event count maps,
             sequence length, and potentially time bounds.
    """
    # Create the PARTITION BY and SELECT clauses dynamically
    # Assumes partition_by_cols is a non-empty list of strings
    partition_by_clause = ", ".join(partition_by_cols)
    select_partition_cols = ", ".join(partition_by_cols)

    # Base columns to select in the first CTE
    cte1_select_cols = f"{select_partition_cols}, {event_col}"
    # Base grouping for the first CTE
    cte1_group_by_cols = f"{partition_by_clause}, {event_col}"

    # Columns for the final SELECT statement
    # Added SUM(event_count) AS seq_length to calculate the total number of events per partition
    final_select_cols = f"{select_partition_cols}, MAP_AGG({event_col}, event_count) AS event_seq, SUM(event_count) AS seq_length"

    # Add time column and aggregations if time_col is provided
    if time_col:
        cte1_select_cols += f", MIN({time_col}) AS min_t, MAX({time_col}) AS max_t"
        # Updated aliases to start_seq_time and end_seq_time
        final_select_cols += ", MIN(min_t) AS start_seq_time, MAX(max_t) AS end_seq_time"

    # Generate the Presto SQL query
    query = f"""
    WITH EventCountsAndTime AS (
        SELECT
            {cte1_select_cols},
            COUNT(*) AS event_count
        FROM {table_name}
        GROUP BY {cte1_group_by_cols}
    )
    SELECT
        {final_select_cols}
    FROM EventCountsAndTime
    GROUP BY {partition_by_clause}
    """

    return query

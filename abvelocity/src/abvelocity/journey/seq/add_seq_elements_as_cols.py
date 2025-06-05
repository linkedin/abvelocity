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


def add_seq_elements_as_cols(
    base_query: str,
    partition_by_cols: list[str],
    max_seq_index: int,
    seq_col: str = "event_seq",
    extra_cols: list[str] = ["seq_start_time", "seq_end_time", "seq_length"],
) -> str:
    """
    Generate a SQL query that extracts individual sequence elements from an event sequence array
    and adds them as separate columns (s1, s2, ..., sN) in the result.

    Args:
        base_query (str):
            The SQL query that generates the event sequences, typically the output of a deduplication function.
            It must return columns specified in `partition_by_cols` and an array column named by `seq_col`.

        partition_by_cols (list[str]):
            A list of column names to be included in the final output.
            These columns are used to partition the data.

        max_seq_index (int):
            The maximum number of sequence elements to extract as separate columns.
            If `max_seq_index` is 3, columns s1, s2, and s3 will be created.

        seq_col (str, optional):
            The name of the sequence array column from which elements will be extracted. Default is "event_seq".

        extra_cols (list[str], optional):
            Additional column names to be included in the final output. Default is None.
            The default is to maintain start and end time for the seq as well as its length.

    Returns:
        str: A SQL query that extracts sequence elements and includes them as separate columns (s1, s2, ..., sN).

    Raises:
        ValueError: If `partition_by_cols` is not provided as a list or `max_seq_index` is not a positive integer.
    """
    # Validate input types
    if not isinstance(partition_by_cols, list):
        raise ValueError("partition_by_cols must be a list of column names.")

    if not isinstance(max_seq_index, int) or max_seq_index <= 0:
        raise ValueError("max_seq_index must be a positive integer.")

    # Prepare subquery from base query
    subquery = base_query.strip()

    # Construct the SELECT clause
    select_clause = f"""
    SELECT
        {', '.join(partition_by_cols)},
        {seq_col}"""

    if extra_cols:
        select_clause = f"""
    SELECT
        {', '.join(partition_by_cols)},
        {seq_col},
        {', '.join(extra_cols)}"""

    # Add the sequence columns dynamically based on `max_seq_index`
    seq_cols = [f"element_at({seq_col}, {i + 1}) AS s{i + 1}" for i in range(max_seq_index)]
    select_clause += ", " + ", ".join(seq_cols)

    # Formulate the final query
    final_query = f"""
    WITH BaseData AS ({subquery})
    {select_clause}
    FROM BaseData
    """

    return final_query

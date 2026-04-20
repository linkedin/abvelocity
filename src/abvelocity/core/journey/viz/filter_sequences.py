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

"""Utility functions for filtering and re-aggregating sequence data."""

from typing import List, Tuple

import pandas as pd


def filter_and_reaggregate_sequences(
    df: pd.DataFrame,
    max_seq_index: int,
    value_col: str,
    include_events: List[str],
) -> Tuple[pd.DataFrame, int]:
    """
    Filter events from sequences and re-aggregate counts.

    This function handles the complexity of filtering events from positional
    sequence data (s1, s2, ..., sN columns). When events are filtered:
    1. Unwanted events are removed from each sequence
    2. Remaining events shift left to fill gaps (positions stay contiguous)
    3. Counts are re-aggregated for sequences that become identical after filtering

    Example:
        Original data:
            s1=WVMP, s2=TAJ, s3=renewed, count=100
            s1=WVMP, s2=renewed, s3=NULL, count=50

        After filtering to [WVMP, renewed]:
            s1=WVMP, s2=renewed, count=150  (aggregated because both map to same pattern)

    Args:
        df: DataFrame with sequence columns (s1, s2, ...) and value columns.
        max_seq_index: Maximum sequence index (number of s columns).
        value_col: Column name containing the values to aggregate (e.g., 'count', 'percent').
        include_events: List of events to keep in sequences.

    Returns:
        Tuple of:
            - Filtered and re-aggregated DataFrame
            - New max_seq_index (maximum non-null sequence length after filtering)
    """
    if df.empty:
        return df, 0

    seq_cols = [f"s{i}" for i in range(1, max_seq_index + 1)]

    # Verify sequence columns exist
    missing_cols = [col for col in seq_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing sequence columns in DataFrame: {missing_cols}")

    include_events_set = set(include_events)

    def filter_sequence(row):
        """Filter a single row's sequence, keeping only included events."""
        filtered = []
        for col in seq_cols:
            val = row[col]
            # Keep the value if it's in include_events (handle NaN/None)
            if pd.notna(val) and val in include_events_set:
                filtered.append(val)
        # Pad with None to maintain column count
        return filtered + [None] * (max_seq_index - len(filtered))

    # Apply filtering to each row
    filtered_seqs = df.apply(filter_sequence, axis=1, result_type="expand")
    filtered_seqs.columns = seq_cols

    # Create result DataFrame with filtered sequences
    result_df = pd.DataFrame()
    for col in seq_cols:
        result_df[col] = filtered_seqs[col]
    result_df[value_col] = df[value_col].values

    # Re-aggregate: group by sequence columns and sum the value column
    # Convert None to a consistent representation for grouping
    for col in seq_cols:
        result_df[col] = result_df[col].fillna("__NULL__")

    agg_df = result_df.groupby(seq_cols, dropna=False).agg({value_col: "sum"}).reset_index()

    # Convert back from placeholder to actual None/NaN
    for col in seq_cols:
        agg_df[col] = agg_df[col].replace("__NULL__", None)

    # Recalculate percent
    total = agg_df[value_col].sum()
    agg_df["percent"] = (agg_df[value_col] / total * 100) if total > 0 else 0

    # Calculate new max_seq_index (max non-null positions across all rows)
    def count_non_null(row):
        return sum(1 for col in seq_cols if pd.notna(row[col]))

    new_max_seq_index = agg_df.apply(count_non_null, axis=1).max() if len(agg_df) > 0 else 0

    # Filter out rows where all sequence columns are null (empty sequences)
    non_empty_mask = agg_df[seq_cols].notna().any(axis=1)
    agg_df = agg_df[non_empty_mask].reset_index(drop=True)

    # Recalculate percent after removing empty sequences
    if len(agg_df) > 0:
        total = agg_df[value_col].sum()
        agg_df["percent"] = (agg_df[value_col] / total * 100) if total > 0 else 0

    return agg_df, int(new_max_seq_index)

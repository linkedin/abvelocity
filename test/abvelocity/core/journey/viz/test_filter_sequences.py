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

"""Tests for filter_sequences module.

These tests verify the behavior of filter_and_reaggregate_sequences function which:
1. Filters events from sequences, keeping only specified events
2. Shifts remaining events left to fill gaps (positions stay contiguous)
3. Re-aggregates counts for sequences that become identical after filtering
4. Excludes records where all events are filtered out (empty sequences)
"""

import pandas as pd
import pytest
from abvelocity.core.journey.viz.filter_sequences import filter_and_reaggregate_sequences


class TestFilterAndReaggregateSequences:
    """Test suite for filter_and_reaggregate_sequences function."""

    def test_basic_filtering(self):
        """Test basic event filtering - keep only specified events."""
        # Input: sequences with WVMP, TAJ, renewed events
        df = pd.DataFrame(
            {
                "s1": ["WVMP", "TAJ", "WVMP"],
                "s2": ["TAJ", "renewed", "renewed"],
                "s3": ["renewed", None, None],
                "count": [100, 50, 75],
            }
        )

        # Filter to keep only WVMP and renewed
        result_df, new_max_seq = filter_and_reaggregate_sequences(df, max_seq_index=3, value_col="count", include_events=["WVMP", "renewed"])

        # TAJ should be removed, sequences should shift left
        assert "TAJ" not in result_df["s1"].values
        assert "TAJ" not in result_df["s2"].values
        assert "TAJ" not in result_df["s3"].values

    def test_aggregation_after_filtering(self):
        """Test that sequences become identical after filtering are aggregated.

        Example from docstring:
            Original: s1=WVMP, s2=TAJ, s3=renewed, count=100
                      s1=WVMP, s2=renewed, s3=NULL, count=50
            After filtering to [WVMP, renewed]:
                      s1=WVMP, s2=renewed, count=150 (aggregated)
        """
        df = pd.DataFrame(
            {
                "s1": ["WVMP", "WVMP"],
                "s2": ["TAJ", "renewed"],  # TAJ will be filtered out
                "s3": ["renewed", None],
                "count": [100, 50],
            }
        )

        result_df, new_max_seq = filter_and_reaggregate_sequences(df, max_seq_index=3, value_col="count", include_events=["WVMP", "renewed"])

        # Both sequences should become WVMP -> renewed and be aggregated
        assert len(result_df) == 1
        assert result_df.iloc[0]["count"] == 150
        assert result_df.iloc[0]["s1"] == "WVMP"
        assert result_df.iloc[0]["s2"] == "renewed"

    def test_empty_sequences_excluded(self):
        """Test that records with no remaining events after filtering are excluded.

        This is a key behavior: if all events in a sequence are filtered out,
        that record should not appear in the result.
        """
        df = pd.DataFrame(
            {
                "s1": ["WVMP", "TAJ", "OTHER"],  # Third row has no included events
                "s2": ["renewed", "TAJ", "EXCLUDED"],
                "s3": [None, None, None],
                "count": [100, 50, 25],
            }
        )

        result_df, new_max_seq = filter_and_reaggregate_sequences(df, max_seq_index=3, value_col="count", include_events=["WVMP", "renewed"])

        # Only first row should remain (has WVMP and renewed)
        # Second row only has TAJ (not in include_events) - should be excluded
        # Third row has neither WVMP nor renewed - should be excluded
        assert len(result_df) == 1
        assert result_df.iloc[0]["count"] == 100

    def test_all_sequences_filtered_out(self):
        """Test when all sequences become empty after filtering."""
        df = pd.DataFrame(
            {
                "s1": ["TAJ", "OTHER"],
                "s2": ["EXCLUDED", "EXCLUDED"],
                "s3": [None, None],
                "count": [100, 50],
            }
        )

        result_df, new_max_seq = filter_and_reaggregate_sequences(df, max_seq_index=3, value_col="count", include_events=["WVMP", "renewed"])

        # All sequences should be filtered out
        assert len(result_df) == 0
        assert new_max_seq == 0

    def test_empty_input_dataframe(self):
        """Test handling of empty input DataFrame."""
        df = pd.DataFrame({"s1": [], "s2": [], "s3": [], "count": []})

        result_df, new_max_seq = filter_and_reaggregate_sequences(df, max_seq_index=3, value_col="count", include_events=["WVMP"])

        assert len(result_df) == 0
        assert new_max_seq == 0

    def test_new_max_seq_index_calculation(self):
        """Test that new_max_seq_index reflects actual sequence length after filtering."""
        df = pd.DataFrame(
            {
                "s1": ["WVMP", "WVMP"],
                "s2": ["TAJ", "TAJ"],  # Will be filtered out
                "s3": ["OTHER", "OTHER"],  # Will be filtered out
                "s4": ["renewed", None],
                "count": [100, 50],
            }
        )

        result_df, new_max_seq = filter_and_reaggregate_sequences(df, max_seq_index=4, value_col="count", include_events=["WVMP", "renewed"])

        # After filtering: first row is WVMP -> renewed (length 2)
        #                  second row is WVMP (length 1)
        # Max should be 2
        assert new_max_seq == 2

    def test_percent_recalculation(self):
        """Test that percent column is recalculated after filtering and aggregation."""
        df = pd.DataFrame(
            {
                "s1": ["WVMP", "WVMP", "TAJ"],  # Third row will be excluded
                "s2": ["renewed", "renewed", None],
                "count": [100, 100, 50],
            }
        )

        result_df, new_max_seq = filter_and_reaggregate_sequences(df, max_seq_index=2, value_col="count", include_events=["WVMP", "renewed"])

        # After filtering: only rows with WVMP remain, aggregated to count=200
        # Percent should be 100% since it's the only row
        assert len(result_df) == 1
        assert result_df.iloc[0]["count"] == 200
        assert result_df.iloc[0]["percent"] == 100.0

    def test_missing_sequence_columns_raises_error(self):
        """Test that missing sequence columns raise ValueError."""
        df = pd.DataFrame(
            {
                "s1": ["WVMP"],
                # Missing s2, s3
                "count": [100],
            }
        )

        with pytest.raises(ValueError, match="Missing sequence columns"):
            filter_and_reaggregate_sequences(df, max_seq_index=3, value_col="count", include_events=["WVMP"])

    def test_preserves_non_included_none_values(self):
        """Test that None values in original data are handled correctly."""
        df = pd.DataFrame(
            {
                "s1": ["WVMP", "renewed"],
                "s2": [None, None],  # Already None
                "s3": [None, None],
                "count": [100, 50],
            }
        )

        result_df, new_max_seq = filter_and_reaggregate_sequences(df, max_seq_index=3, value_col="count", include_events=["WVMP", "renewed"])

        # Both single-event sequences should be preserved
        assert len(result_df) == 2
        assert new_max_seq == 1

    def test_shift_left_behavior(self):
        """Test that events shift left to fill gaps after filtering.

        If we have s1=A, s2=B, s3=C and filter out B,
        the result should be s1=A, s2=C (C moves to position 2).
        """
        df = pd.DataFrame(
            {
                "s1": ["WVMP"],
                "s2": ["TAJ"],  # Will be filtered out
                "s3": ["renewed"],
                "count": [100],
            }
        )

        result_df, new_max_seq = filter_and_reaggregate_sequences(df, max_seq_index=3, value_col="count", include_events=["WVMP", "renewed"])

        # renewed should shift from s3 to s2
        assert result_df.iloc[0]["s1"] == "WVMP"
        assert result_df.iloc[0]["s2"] == "renewed"
        assert pd.isna(result_df.iloc[0]["s3"]) or result_df.iloc[0]["s3"] is None

# Original author: Reza Hosseini
"""Unit tests for ``time_properties.validate_per_dim_timestamps``."""

from __future__ import annotations

import pandas as pd
import pytest

from abvelocity.ts.common.time_properties import validate_per_dim_timestamps


def _daily(n: int = 10, start: str = "2026-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="D")


def test_validate_passes_for_sorted_single_series():
    """Happy path: one series, sorted, regular daily increments."""
    df = pd.DataFrame({"ts": _daily(10), "y": range(10)})
    validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=())


def test_validate_passes_for_unsorted_single_series():
    """Unsorted is not a violation — duplicates are the only thing
    we raise on; ordering is the algo's problem."""
    df_sorted = pd.DataFrame({"ts": _daily(10), "y": range(10)})
    df_shuffled = df_sorted.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert not df_shuffled["ts"].is_monotonic_increasing
    validate_per_dim_timestamps(df=df_shuffled, time_col="ts", dim_cols=())


def test_validate_raises_on_duplicate_timestamps_single_series():
    df = pd.DataFrame({"ts": list(_daily(5)) + list(_daily(5)), "y": range(10)})
    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=())


def test_validate_passes_for_irregular_gaps_in_real_data():
    """Real-world daily data often has missing days (Wikipedia scraping
    failures, weekend-only operations, source outages).  The validator
    intentionally tolerates these — algos handle gaps fine, and being
    strict here just makes valid data unusable."""
    irregular = list(_daily(3)) + [pd.Timestamp("2026-01-10")] + list(pd.date_range(start="2026-01-11", periods=3, freq="D"))
    df = pd.DataFrame({"ts": irregular, "y": range(7)})
    validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=())


def test_validate_passes_for_clean_panel():
    """Two-segment panel — each segment has its own regular daily series."""
    rows = []
    for country in ("US", "GB"):
        for ts in _daily(10):
            rows.append({"ts": ts, "country": country, "y": 1.0})
    df = pd.DataFrame(rows)
    validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=("country",))


def test_validate_raises_when_dim_col_missing_from_config():
    """Common bug: panel data passed without listing its dim_col → each
    timestamp appears once per segment, looks like a duplicate at the
    aggregated level."""
    rows = []
    for country in ("US", "GB"):
        for ts in _daily(10):
            rows.append({"ts": ts, "country": country, "y": 1.0})
    df = pd.DataFrame(rows)
    # No dim_cols passed — duplicates appear in the single combined "segment".
    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=())


def test_validate_names_offending_segment_in_message():
    rows = []
    for country in ("US", "GB"):
        for ts in _daily(10):
            rows.append({"ts": ts, "country": country, "y": 1.0})
    # Inject one duplicate row into the GB segment only.
    rows.append({"ts": _daily(10)[3], "country": "GB", "y": 99.0})
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match=r"GB"):
        validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=("country",))


def test_validate_skips_segments_with_fewer_than_two_rows():
    """A segment with <2 rows can't be evaluated for regularity — skip
    it rather than raise.  Common at backfill cutoffs where some dim
    combos have only just appeared."""
    df = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")],
            "country": ["US", "US", "GB"],
            "y": [1.0, 2.0, 3.0],
        }
    )
    validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=("country",))


def test_validate_short_circuits_on_empty_df():
    """Empty frame → nothing to validate, no error.  Algos handle the
    empty case themselves with their own (clearer) error messages."""
    df = pd.DataFrame({"ts": [], "y": []})
    validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=())


def test_validate_short_circuits_when_dim_col_missing_from_frame():
    """Caller config names a dim that isn't in the frame yet — let the
    algo surface that with its own error rather than blowing up here
    with a confusing pandas KeyError."""
    df = pd.DataFrame({"ts": _daily(5), "y": range(5)})
    # dim_cols=("country",) but no "country" column → no raise.
    validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=("country",))


def test_validate_short_circuits_when_time_col_missing_from_frame():
    df = pd.DataFrame({"y": range(5)})
    validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=())


def test_validate_does_not_drop_extra_columns_or_mutate_caller_df():
    """The helper must leave the caller's frame untouched — including
    regressor columns and column ordering."""
    df = pd.DataFrame(
        {
            "ts": _daily(10),
            "y": range(10),
            "country": ["US"] * 10,
            "regressor_temperature": [22.0 + i * 0.1 for i in range(10)],
            "regressor_promo": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    snapshot_columns = list(df.columns)
    snapshot_first_row = dict(df.iloc[0])

    validate_per_dim_timestamps(df=df, time_col="ts", dim_cols=("country",))

    assert list(df.columns) == snapshot_columns
    assert dict(df.iloc[0]) == snapshot_first_row

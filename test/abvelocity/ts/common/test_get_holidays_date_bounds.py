# Original author: Reza Hosseini
"""Tests for the ``start_date`` / ``end_date`` clip kwargs of
:func:`abvelocity.ts.common.holiday.get_holidays.get_holiday_df`.

The sibling :mod:`test_get_holidays` file is gated behind
``gk_test_gate`` (heavy snapshot suite — skipped in CI by default), so
the lines in :func:`get_holiday_df` that handle the new optional date
bounds need to be exercised by an *ungated* test file. Hence this
module: small, fast, no greykite import, and always runs in CI.
"""

import pandas as pd

from abvelocity.ts.common.holiday.get_holidays import get_holiday_df


def test_get_holiday_df_no_bounds_returns_full_range():
    """When neither bound is set, every holiday for the requested years is returned."""
    df = get_holiday_df(country_list=["US"], years=[2024, 2025])
    # 12 US holidays per year (11 federal + Halloween injected) → 24 across two years.
    assert len(df) == 24
    assert df["ts"].min() == pd.Timestamp("2024-01-01")
    assert df["ts"].max() == pd.Timestamp("2025-12-25")


def test_get_holiday_df_start_date_clips_lower_bound_inclusive():
    """``start_date`` drops rows strictly before that date — the bound itself is kept."""
    df = get_holiday_df(
        country_list=["US"],
        years=[2024],
        start_date=pd.Timestamp("2024-07-04"),
    )
    # Independence (Jul 4) kept; Memorial (May 27) dropped.
    assert df["ts"].min() == pd.Timestamp("2024-07-04")
    assert (df["ts"] >= pd.Timestamp("2024-07-04")).all()


def test_get_holiday_df_end_date_clips_upper_bound_inclusive():
    """``end_date`` drops rows strictly after that date — the bound itself is kept."""
    df = get_holiday_df(
        country_list=["US"],
        years=[2024],
        end_date=pd.Timestamp("2024-07-04"),
    )
    # Independence (Jul 4) kept; Labor (Sep 2) dropped.
    assert df["ts"].max() == pd.Timestamp("2024-07-04")
    assert (df["ts"] <= pd.Timestamp("2024-07-04")).all()


def test_get_holiday_df_both_bounds_clip_to_window():
    """Both bounds together: keep only holidays inside the inclusive window."""
    df = get_holiday_df(
        country_list=["US"],
        years=[2024, 2025],
        start_date=pd.Timestamp("2024-12-01"),
        end_date=pd.Timestamp("2025-02-01"),
    )
    # Christmas 2024, New Years 2025, MLK 2025 — three rows in the window.
    assert sorted(df["ts"].dt.strftime("%Y-%m-%d").tolist()) == [
        "2024-12-25",
        "2025-01-01",
        "2025-01-20",
    ]


def test_get_holiday_df_window_with_no_matches_returns_empty_frame():
    """Window outside any holiday → empty frame, schema preserved."""
    df = get_holiday_df(
        country_list=["US"],
        years=[2024],
        start_date=pd.Timestamp("2024-08-01"),
        end_date=pd.Timestamp("2024-08-31"),
    )
    # No US holidays in August 2024.
    assert df.empty
    # Schema must still be the canonical 4-column layout so downstream
    # ``df.merge`` etc. doesn't blow up on a missing column.
    assert list(df.columns) == ["ts", "country", "holiday", "country_holiday"]


def test_get_holiday_df_index_is_reset_after_clipping():
    """Clipping must hand back a 0..N-1 contiguous index, not the source's."""
    df = get_holiday_df(
        country_list=["US"],
        years=[2024],
        start_date=pd.Timestamp("2024-11-01"),
    )
    # Range index starting at 0 — easier on downstream callers that do
    # positional reads.
    assert list(df.index) == list(range(len(df)))

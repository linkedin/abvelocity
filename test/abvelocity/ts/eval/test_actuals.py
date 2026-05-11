# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini
"""Tests for :mod:`abvelocity.ts.eval.actuals`."""

import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.param.io_param import IOParam
from abvelocity.ts.eval.actuals import (
    ACTUALS_AGG_QUERY,
    dedupe_actuals_from_partitions_df,
    fetch_actuals_from_forecast_table,
    warn_on_partition_inconsistency,
)


def make_aggregates_df(consistent: bool = True) -> pd.DataFrame:
    """Pre-aggregated frame mirroring the SQL output. ``consistent=False`` injects a CV-violating row."""
    rows = [
        {
            "metric_id": "signups",
            "forecasted_date": pd.Timestamp("2024-01-01"),
            "actual_mean": 100.0,
            "actual_stddev": 0.05,
            "actual_min": 99.95,
            "actual_max": 100.05,
            "actual_partition_count": 5,
        },
        {
            "metric_id": "signups",
            "forecasted_date": pd.Timestamp("2024-01-02"),
            "actual_mean": 110.0,
            "actual_stddev": 0.0,
            "actual_min": 110.0,
            "actual_max": 110.0,
            "actual_partition_count": 5,
        },
    ]
    if not consistent:
        # CV = 5.0 / 100 = 5%, well above default tolerance of 0.1%.
        rows.append(
            {
                "metric_id": "bookings",
                "forecasted_date": pd.Timestamp("2024-01-01"),
                "actual_mean": 100.0,
                "actual_stddev": 5.0,
                "actual_min": 90.0,
                "actual_max": 110.0,
                "actual_partition_count": 5,
            }
        )
    return pd.DataFrame(rows)


def make_partition_rows_df() -> pd.DataFrame:
    """Long-format partition rows: same (metric, date) appears across many cutoffs.

    Two metrics × two dates × three cutoffs each. ``actual`` agrees across
    cutoffs (so no CV warning fires); each cutoff also has ``stage='forecast'``
    rows that should be filtered out of actuals.
    """
    rows = []
    for metric, value in [("signups", 100.0), ("bookings", 50.0)]:
        for date_offset in (0, 1):
            for cutoff_offset in (0, 1, 2):
                rows.append(
                    {
                        "metric_id": metric,
                        "forecasted_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=date_offset),
                        "last_training_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=cutoff_offset),
                        "stage": "fitted",
                        "actual": value + 0.1 * date_offset,
                    }
                )
                rows.append(
                    {
                        "metric_id": metric,
                        "forecasted_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=date_offset),
                        "last_training_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=cutoff_offset),
                        "stage": "forecast",
                        "actual": np.nan,
                    }
                )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# warn_on_partition_inconsistency
# ─────────────────────────────────────────────────────────────────────────────


def test_warn_on_partition_inconsistency_quiet_when_consistent():
    """All rows below CV threshold → no warning emitted; clean output schema."""
    aggregates = make_aggregates_df(consistent=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = warn_on_partition_inconsistency(aggregates=aggregates)
    assert len(caught) == 0
    assert list(result.columns) == ["metric_id", "forecasted_date", "actual"]
    assert len(result) == 2
    assert result["actual"].tolist() == [100.0, 110.0]


def test_warn_on_partition_inconsistency_warns_above_tolerance():
    """A row with CV=5% triggers exactly one warning that names the bad (metric, date)."""
    aggregates = make_aggregates_df(consistent=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = warn_on_partition_inconsistency(aggregates=aggregates, relative_tolerance=1e-3)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "1 (metric_id, forecasted_date) pairs disagree" in message
    assert "bookings" in message
    # Even when warning fires, the result still includes the inconsistent row.
    assert len(result) == 3


def test_warn_on_partition_inconsistency_empty_input_returns_empty_with_schema():
    """Empty input → empty output with the canonical column shape — no crash, no warning."""
    empty = pd.DataFrame(
        columns=["metric_id", "forecasted_date", "actual_mean", "actual_stddev",
                 "actual_min", "actual_max", "actual_partition_count"]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = warn_on_partition_inconsistency(aggregates=empty)
    assert len(caught) == 0
    assert list(result.columns) == ["metric_id", "forecasted_date", "actual"]
    assert len(result) == 0


def test_warn_on_partition_inconsistency_zero_mean_zero_stddev_does_not_warn():
    """``mean=0`` with ``stddev=0`` → CV=0 (clipped denominator); silent. Real zeros stay zero."""
    aggregates = pd.DataFrame(
        [
            {
                "metric_id": "x",
                "forecasted_date": pd.Timestamp("2024-01-01"),
                "actual_mean": 0.0,
                "actual_stddev": 0.0,
                "actual_min": 0.0,
                "actual_max": 0.0,
                "actual_partition_count": 3,
            }
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = warn_on_partition_inconsistency(aggregates=aggregates)
    assert len(caught) == 0
    assert result["actual"].iloc[0] == 0.0


def test_warn_on_partition_inconsistency_zero_mean_nonzero_stddev_warns_loudly():
    """``mean=0`` with ``stddev>0`` → huge CV (clipped denominator) → warning fires."""
    aggregates = pd.DataFrame(
        [
            {
                "metric_id": "x",
                "forecasted_date": pd.Timestamp("2024-01-01"),
                "actual_mean": 0.0,
                "actual_stddev": 1.0,
                "actual_min": -1.0,
                "actual_max": 1.0,
                "actual_partition_count": 3,
            }
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_partition_inconsistency(aggregates=aggregates)
    assert len(caught) == 1


# ─────────────────────────────────────────────────────────────────────────────
# dedupe_actuals_from_partitions_df
# ─────────────────────────────────────────────────────────────────────────────


def test_dedupe_actuals_from_partitions_df_aggregates_correctly():
    """Three cutoffs × two metrics × two dates → 4 deduped rows with the per-(metric, date) mean."""
    df = make_partition_rows_df()
    result = dedupe_actuals_from_partitions_df(df=df)
    assert list(result.columns) == ["metric_id", "forecasted_date", "actual"]
    # 2 metrics × 2 dates = 4 rows.
    assert len(result) == 4
    # signups date 1: all three cutoffs have actual=100.0 → mean=100.0.
    signups_d0 = result[(result["metric_id"] == "signups") & (result["forecasted_date"] == pd.Timestamp("2024-01-01"))]
    assert signups_d0["actual"].iloc[0] == 100.0


def test_dedupe_actuals_from_partitions_df_filters_out_forecast_stage():
    """Rows with ``stage='forecast'`` are excluded from the actuals aggregation."""
    df = make_partition_rows_df()
    # The fixture has 3 forecast rows per (metric, date) — those must NOT contribute to the count.
    result = dedupe_actuals_from_partitions_df(df=df)
    # All four (metric, date) pairs end up with actual = exactly the seeded value (no NaN forecast contamination).
    assert not result["actual"].isna().any()


def test_dedupe_actuals_from_partitions_df_warns_on_disagreeing_partitions():
    """If partitions disagree on the same (metric, date), the CV warning fires."""
    df = make_partition_rows_df()
    # Inject a wildly-divergent fitted row for signups date 0.
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "metric_id": "signups",
                        "forecasted_date": pd.Timestamp("2024-01-01"),
                        "last_training_date": pd.Timestamp("2024-01-04"),
                        "stage": "fitted",
                        "actual": 200.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dedupe_actuals_from_partitions_df(df=df)
    assert len(caught) == 1
    assert "signups" in str(caught[0].message)


def test_dedupe_actuals_from_partitions_df_empty_input():
    """Empty df → empty result with canonical schema."""
    empty = pd.DataFrame(columns=["metric_id", "forecasted_date", "stage", "actual"])
    result = dedupe_actuals_from_partitions_df(df=empty)
    assert len(result) == 0
    assert list(result.columns) == ["metric_id", "forecasted_date", "actual"]


def test_dedupe_actuals_from_partitions_df_all_nan_actuals_returns_empty():
    """Every actual is NaN → no fitted rows survive the filter → empty result."""
    df = pd.DataFrame(
        [
            {
                "metric_id": "x",
                "forecasted_date": pd.Timestamp("2024-01-01"),
                "stage": "fitted",
                "actual": np.nan,
            }
        ]
    )
    result = dedupe_actuals_from_partitions_df(df=df)
    assert len(result) == 0


def test_dedupe_actuals_from_partitions_df_missing_required_column_raises():
    """Caller error: missing ``stage`` column."""
    df = pd.DataFrame({"metric_id": ["x"], "forecasted_date": [pd.Timestamp("2024-01-01")], "actual": [1.0]})
    try:
        dedupe_actuals_from_partitions_df(df=df)
    except ValueError as error:
        assert "stage" in str(error)
    else:
        raise AssertionError("expected ValueError")


# ─────────────────────────────────────────────────────────────────────────────
# fetch_actuals_from_forecast_table — DataContainer adapter
# ─────────────────────────────────────────────────────────────────────────────


def test_fetch_actuals_pandas_mode_delegates_to_df_core():
    """``DataContainer(is_pandas_df=True, pandas_df=...)`` → result == df-core output on the same input."""
    df = make_partition_rows_df()
    dc = DataContainer(is_pandas_df=True, pandas_df=df)
    result = fetch_actuals_from_forecast_table(dc=dc)
    expected = dedupe_actuals_from_partitions_df(df=df)
    pd.testing.assert_frame_equal(result, expected)


def test_fetch_actuals_pandas_mode_missing_pandas_df_raises():
    """``is_pandas_df=True`` with ``pandas_df=None`` is a programming error."""
    dc = DataContainer(is_pandas_df=True, pandas_df=None)
    try:
        fetch_actuals_from_forecast_table(dc=dc)
    except ValueError as error:
        assert "pandas_df" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_fetch_actuals_sql_mode_runs_aggregation_via_cursor():
    """``is_sql_table=True`` → calls ``cursor.get_df`` with ``ACTUALS_AGG_QUERY``, pipes through warn helper."""
    cursor = MagicMock()
    cursor.get_df.return_value = MagicMock(df=make_aggregates_df(consistent=True))
    io_param = IOParam(cursor=cursor)
    dc = DataContainer(is_sql_table=True, table_name="u_clearsight.foo")

    result = fetch_actuals_from_forecast_table(dc=dc, io_param=io_param)

    # Cursor was called once with the formatted aggregation query.
    cursor.get_df.assert_called_once()
    sql_arg = cursor.get_df.call_args[0][0]
    assert "u_clearsight.foo" in sql_arg
    assert "GROUP BY metric_id, forecasted_date" in sql_arg
    # Output schema matches the warn-helper contract.
    assert list(result.columns) == ["metric_id", "forecasted_date", "actual"]
    assert len(result) == 2


def test_fetch_actuals_sql_mode_missing_io_param_raises():
    """``is_sql_table=True`` without ``io_param`` is a programming error — no cursor to run the SQL."""
    dc = DataContainer(is_sql_table=True, table_name="u_clearsight.foo")
    try:
        fetch_actuals_from_forecast_table(dc=dc, io_param=None)
    except ValueError as error:
        assert "io_param" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_fetch_actuals_sql_mode_missing_cursor_raises():
    """``IOParam`` with ``cursor=None`` is also rejected."""
    dc = DataContainer(is_sql_table=True, table_name="u_clearsight.foo")
    io_param = IOParam(cursor=None)
    try:
        fetch_actuals_from_forecast_table(dc=dc, io_param=io_param)
    except ValueError as error:
        assert "cursor" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_fetch_actuals_uninitialized_dc_raises():
    """``DataContainer`` with neither ``is_pandas_df`` nor ``is_sql_table`` set → clear error."""
    dc = DataContainer()
    try:
        fetch_actuals_from_forecast_table(dc=dc)
    except ValueError as error:
        assert "pandas" in str(error) or "SQL" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_fetch_actuals_sql_mode_empty_returns_empty():
    """Cursor returns an empty df → adapter returns empty result with canonical schema."""
    cursor = MagicMock()
    cursor.get_df.return_value = MagicMock(df=pd.DataFrame())
    io_param = IOParam(cursor=cursor)
    dc = DataContainer(is_sql_table=True, table_name="u_clearsight.foo")
    result = fetch_actuals_from_forecast_table(dc=dc, io_param=io_param)
    assert len(result) == 0
    assert list(result.columns) == ["metric_id", "forecasted_date", "actual"]


def test_actuals_agg_query_re_exported_for_callers_pasting_into_cloudNotebook_sql_magic():
    """``ACTUALS_AGG_QUERY`` is part of the public surface — callers paste it into ``%%sql`` cells."""
    assert "GROUP BY metric_id, forecasted_date" in ACTUALS_AGG_QUERY
    assert "{table}" in ACTUALS_AGG_QUERY

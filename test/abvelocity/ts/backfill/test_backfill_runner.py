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
"""Tests for BackfillRunner."""

# Self-registers SimpleForecastAlgo into ALGO_REGISTRY under "simple".
import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.backfill.runner import BackfillRunner
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import ACTUAL_COL, CUTOFF_COL, FORECAST_COL, HORIZON_STEP_COL, METRIC_ID_COL, TIME_COL

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_df():
    """30 days of daily data for structural / behavioural tests."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame({TIME_COL: dates, "y": rng.uniform(10, 20, 30)})


@pytest.fixture
def base_fc():
    """ForecastConfig using SimpleForecastAlgo with a trailing 3-value mean."""
    return ForecastConfig(
        time_col=TIME_COL,
        value_cols=("y",),
        freq="D",
        forecast_horizon=7,
        algo_name="simple",
        algo_params={"period": 1, "k": 3},
    )


@pytest.fixture
def arithmetic_series():
    """12 daily rows with values 1..12 — arithmetic for exact manual verification."""
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    return pd.DataFrame({TIME_COL: dates, "value": np.arange(1.0, 13.0)})


@pytest.fixture
def naive_backfill_config():
    """BackfillConfig using the naive last-value forecast (period=1, k=1, horizon=1).

    period=1, k=1: step-1 forecast = last training value.
    Steps > 1 look outside the training window → NaN, so we use horizon=1.
    """
    forecast_config = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("value",),
        freq="D",
        forecast_horizon=1,
        algo_name="simple",
        algo_params={"period": 1, "k": 1},
    )
    return BackfillConfig(
        forecast_config=forecast_config,
        initial_train_size=5,
        horizon=1,
        step=5,
    )


# ---------------------------------------------------------------------------
# generate_cutoff_indices
# ---------------------------------------------------------------------------


def test_generate_cutoff_indices_expanding(base_fc):
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=10, horizon=5, step=1)
    runner = BackfillRunner(cfg)
    # n=20: range(10, 20-5+1, 1) = range(10, 16) = [10,11,12,13,14,15]
    assert runner.generate_cutoff_indices(20) == list(range(10, 16, 1))


def test_generate_cutoff_indices_step(base_fc):
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=10, horizon=5, step=3)
    runner = BackfillRunner(cfg)
    assert runner.generate_cutoff_indices(20) == [10, 13]


def test_generate_cutoff_indices_n_windows_cap(base_fc):
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=10, horizon=5, step=1, n_windows=3)
    runner = BackfillRunner(cfg)
    # All possible: [10..15], capped to last 3
    assert runner.generate_cutoff_indices(20) == [13, 14, 15]


def test_generate_cutoff_indices_empty_when_no_room(base_fc):
    # initial=50, horizon=7, n=30 → range(50, 24) → empty
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=50, horizon=7, step=1)
    runner = BackfillRunner(cfg)
    assert runner.generate_cutoff_indices(30) == []


# ---------------------------------------------------------------------------
# resolve_cutoff_indices — explicit ``config.cutoffs`` path
# ---------------------------------------------------------------------------


def test_resolve_cutoff_indices_honors_explicit_dates(daily_df, base_fc):
    """Explicit cutoffs are looked up by date, converted to exclusive-end
    indices, and returned in ascending order regardless of input order."""
    cfg = BackfillConfig(
        forecast_config=base_fc,
        initial_train_size=10,
        horizon=5,
        cutoffs=["2024-01-15", "2024-01-10"],  # out-of-order on purpose
    )
    runner = BackfillRunner(cfg)
    indices = runner.resolve_cutoff_indices(daily_df)
    # Jan 10 is row index 9 → cutoff_idx 10. Jan 15 is row index 14 → 15.
    assert indices == [10, 15]


def test_resolve_cutoff_indices_warns_but_keeps_short_horizon_cutoffs(
    daily_df, base_fc, caplog,
):
    """A cutoff date too close to the end of the series produces a logged
    warning but is still honoured — explicit cutoffs are the user's
    explicit choice, and forcing horizon-math at config time is overkill.

    The warning gives the user visibility that those cutoffs' eval metrics
    will be computed over a partial horizon.
    """
    # daily_df has 30 rows ("2024-01-01" → "2024-01-30"). With horizon=5, the
    # last cutoff that gets a full-horizon eval is 2024-01-25 (cutoff_idx=25,
    # 5 rows of actuals after). "2024-01-28" sits 2 rows from the end → its
    # eval will be partial-horizon.
    cfg = BackfillConfig(
        forecast_config=base_fc,
        initial_train_size=10,
        horizon=5,
        cutoffs=["2024-01-20", "2024-01-28"],
    )
    runner = BackfillRunner(cfg)
    with caplog.at_level("WARNING", logger="abvelocity.ts.backfill.runner"):
        indices = runner.resolve_cutoff_indices(daily_df)
    assert indices == [20, 28]                 # both kept
    msg = caplog.text
    assert "partial horizon" in msg
    assert "'2024-01-28'" in msg
    assert "'2024-01-20'" not in msg           # 20 has full 5-row horizon


def test_resolve_cutoff_indices_raises_when_date_not_in_series(daily_df, base_fc):
    """A typo'd or out-of-range date must produce a clear ValueError, not a
    silent KeyError downstream."""
    cfg = BackfillConfig(
        forecast_config=base_fc,
        initial_train_size=10,
        horizon=5,
        cutoffs=["2099-12-31"],  # nowhere in the 2024-01 series
    )
    runner = BackfillRunner(cfg)
    with pytest.raises(ValueError, match="not found in df"):
        runner.resolve_cutoff_indices(daily_df)


# ---------------------------------------------------------------------------
# get_train_df
# ---------------------------------------------------------------------------


def test_get_train_df_expanding(daily_df, base_fc):
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=10, horizon=5)
    runner = BackfillRunner(cfg)
    train = runner.get_train_df(daily_df, 15)
    assert len(train) == 15
    assert list(train[TIME_COL]) == list(daily_df[TIME_COL].iloc[:15])


def test_get_train_df_rolling(daily_df, base_fc):
    cfg = BackfillConfig(
        forecast_config=base_fc,
        initial_train_size=10,
        horizon=5,
        window_type="rolling",
        window_size=10,
    )
    runner = BackfillRunner(cfg)
    train = runner.get_train_df(daily_df, 25)
    assert len(train) == 10
    assert list(train[TIME_COL]) == list(daily_df[TIME_COL].iloc[15:25])


def test_get_train_df_rolling_clips_at_zero(daily_df, base_fc):
    cfg = BackfillConfig(
        forecast_config=base_fc,
        initial_train_size=10,
        horizon=5,
        window_type="rolling",
        window_size=100,  # larger than available rows at early cutoffs
    )
    runner = BackfillRunner(cfg)
    # cutoff_idx=10: start = max(0, 10-100) = 0, gives all 10 rows
    train = runner.get_train_df(daily_df, 10)
    assert len(train) == 10


# ---------------------------------------------------------------------------
# Full run — structural / behavioural checks
# ---------------------------------------------------------------------------


def test_runner_run_produces_result_df(daily_df, base_fc):
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=15, horizon=5, step=5)
    result = BackfillRunner(cfg).run(df=daily_df)
    assert result.result_df is not None
    for col in (CUTOFF_COL, HORIZON_STEP_COL, ACTUAL_COL, FORECAST_COL):
        assert col in result.result_df.columns


def test_runner_run_horizon_step_range(daily_df, base_fc):
    horizon = 5
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=15, horizon=horizon, step=7)
    result = BackfillRunner(cfg).run(df=daily_df)
    assert set(result.result_df[HORIZON_STEP_COL].unique()) == set(range(1, horizon + 1))


def test_runner_run_actuals_filled_from_prepped_df(daily_df, base_fc):
    """Actuals in result_df must come from the prepped df, not from the algo (which returns NaN)."""
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=15, horizon=5, step=5)
    result = BackfillRunner(cfg).run(df=daily_df)
    assert result.result_df[ACTUAL_COL].notna().all()


def test_runner_run_n_windows_limits_cutoffs(daily_df, base_fc):
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=10, horizon=5, step=1, n_windows=3)
    result = BackfillRunner(cfg).run(df=daily_df)
    assert result.result_df[CUTOFF_COL].nunique() == 3


def test_runner_run_rolling_window(daily_df, base_fc):
    cfg = BackfillConfig(
        forecast_config=base_fc,
        initial_train_size=10,
        horizon=5,
        step=5,
        window_type="rolling",
        window_size=10,
    )
    result = BackfillRunner(cfg).run(df=daily_df)
    assert result.result_df is not None


def test_runner_run_raises_when_no_cutoffs(daily_df, base_fc):
    # initial_train_size so large that no cutoff fits: range(29, 30-7+1) = range(29, 24) → empty
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=29, horizon=7, step=1)
    with pytest.raises(ValueError, match="No valid cutoffs"):
        BackfillRunner(cfg).run(df=daily_df)


def test_runner_run_cutoff_is_last_training_timestamp(daily_df, base_fc):
    cfg = BackfillConfig(forecast_config=base_fc, initial_train_size=15, horizon=5, step=5, n_windows=2)
    result = BackfillRunner(cfg).run(df=daily_df)
    for cutoff_ts in result.result_df[CUTOFF_COL].unique():
        assert cutoff_ts in daily_df[TIME_COL].values


# ---------------------------------------------------------------------------
# Deterministic result_df checks using naive last-value forecast (period=1, k=1)
#
# Series: values 1..12, daily from 2024-01-01.
# Config: period=1, k=1, horizon=1, initial_train_size=5, step=5 → 2 cutoffs:
#
#   Cutoff 1 (rows 0-4, values 1..5, n=5):
#     step 1: future_pos=5, lookback at pos 4 → value=5.0
#     actual from prepped df (row 5): 6.0
#
#   Cutoff 2 (rows 0-9, values 1..10, n=10):
#     step 1: future_pos=10, lookback at pos 9 → value=10.0
#     actual from prepped df (row 10): 11.0
# ---------------------------------------------------------------------------


def test_simple_backfill_result_shape(arithmetic_series, naive_backfill_config):
    """Two cutoffs × 1 horizon step × 1 metric = 2 rows."""
    result = BackfillRunner(naive_backfill_config).run(df=arithmetic_series)
    assert len(result.result_df) == 2


def test_simple_backfill_full_result_df(arithmetic_series, naive_backfill_config):
    """Compare the entire result_df against a hand-computed expected DataFrame."""
    result = BackfillRunner(naive_backfill_config).run(df=arithmetic_series)
    cols = [TIME_COL, METRIC_ID_COL, ACTUAL_COL, FORECAST_COL, CUTOFF_COL, HORIZON_STEP_COL]
    rdf = result.result_df[cols].sort_values(CUTOFF_COL).reset_index(drop=True)

    dates = arithmetic_series[TIME_COL]
    expected = pd.DataFrame(
        {
            TIME_COL: [dates[5], dates[10]],
            METRIC_ID_COL: ["value", "value"],
            ACTUAL_COL: [6.0, 11.0],
            FORECAST_COL: [5.0, 10.0],
            CUTOFF_COL: [dates[4], dates[9]],
            HORIZON_STEP_COL: [1, 1],
        }
    )

    pd.testing.assert_frame_equal(rdf, expected, check_exact=False, rtol=1e-5)

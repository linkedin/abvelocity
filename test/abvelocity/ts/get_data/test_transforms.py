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
"""Tests for :mod:`abvelocity.ts.get_data.transforms`.

Helpers (``rows_per_period``, ``lag_orders_to_rows``, ``compute_lag_values``)
and the :class:`Coarsen` transform.  Other transforms have their own files.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from abvelocity.ts.get_data.transforms import (
    ALLOWED_AGGS,
    Coarsen,
    Diff,
    PERIOD_DIVISOR,
    WeightWithinPeriod,
    compute_lag_values,
    lag_orders_to_rows,
    rows_per_period,
)
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig
from abvelocity.ts.get_data.ts_transform import TSTransform


# ---------------------------------------------------------------------------
# Lightweight stand-ins for MetricInfo — we only use ``.dims`` in transforms,
# so a typed shim avoids importing the full mashumaro-driven dataclass with
# its required ``metric_family`` plumbing.
# ---------------------------------------------------------------------------


@dataclass
class StubMetric:
    name: str


@dataclass
class StubMetricInfo:
    dims: Optional[list[str]] = None
    metrics: Optional[list[StubMetric]] = None


def make_ts_config(freq: str = "D") -> TSMetricsConfig:
    return TSMetricsConfig(time_col="event_ts", freq=freq, time_alias="ts")


# ---------------------------------------------------------------------------
# PERIOD_DIVISOR / rows_per_period / lag_orders_to_rows
# ---------------------------------------------------------------------------


def test_period_divisor_known_pairs():
    """Sanity-check the canonical pairs; M / Q intentionally absent."""
    assert PERIOD_DIVISOR[("D", "W")] == 7
    assert PERIOD_DIVISOR[("h", "D")] == 24
    assert PERIOD_DIVISOR[("h", "W")] == 168
    assert PERIOD_DIVISOR[("min", "h")] == 60
    assert ("D", "M") not in PERIOD_DIVISOR
    assert ("D", "Q") not in PERIOD_DIVISOR


def test_rows_per_period_self_pair_returns_one():
    """Same source and target → 1 row per period (identity)."""
    assert rows_per_period("D", "D") == 1
    assert rows_per_period("h", "h") == 1


def test_rows_per_period_known_pair():
    """Daily into weekly = 7."""
    assert rows_per_period("D", "W") == 7


def test_rows_per_period_normalizes_anchored_week_aliases():
    """``"W-SAT"`` and ``"W-SUN"`` both mean "weekly" for row-count purposes
    — anchor only controls the period boundary."""
    assert rows_per_period("D", "W-SAT") == 7
    assert rows_per_period("D", "W-SUN") == 7
    assert rows_per_period("h", "W-FRI") == 24 * 7


def test_rows_per_period_unsupported_raises():
    """Variable-length pairs must raise so completeness logic isn't silently wrong."""
    with pytest.raises(ValueError, match="Unsupported"):
        rows_per_period("D", "M")
    with pytest.raises(ValueError, match="Unsupported"):
        rows_per_period("W", "D")  # upsample


def test_lag_orders_to_rows_translates():
    """Weekly orders [1, 2, 3] at daily freq → row offsets [7, 14, 21]."""
    assert lag_orders_to_rows([1, 2, 3], source_freq="D", lag_period="W") == [7, 14, 21]
    assert lag_orders_to_rows([1], source_freq="h", lag_period="D") == [24]
    assert lag_orders_to_rows([1, 2], source_freq="D", lag_period="D") == [1, 2]


def test_allowed_aggs_membership():
    """Closed enum — keep this stable, the suffix tables in JobConfig depend on it."""
    assert set(ALLOWED_AGGS) == {"mean", "median", "max", "min", "sum"}


# ---------------------------------------------------------------------------
# compute_lag_values
# ---------------------------------------------------------------------------


def test_compute_lag_values_single_lag_returns_shifted_values():
    """Lag 1 day on continuous daily series: each row = previous day's value (NaN at start)."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
    out = compute_lag_values(series, lag_period="D", lag_period_orders=[1])
    assert pd.isna(out.iloc[0])
    assert list(out.iloc[1:]) == [10.0, 20.0, 30.0, 40.0]


def test_compute_lag_values_multi_lag_aggregates():
    """Mean of weekly orders [1, 2, 3]: at day 21, baseline = mean(values at days 0, 7, 14)."""
    idx = pd.date_range("2024-01-01", periods=22, freq="D")
    series = pd.Series(np.arange(22, dtype=float), index=idx)
    out = compute_lag_values(series, lag_period="W", lag_period_orders=[1, 2, 3], agg="mean")
    assert out.loc["2024-01-22"] == pytest.approx((0.0 + 7.0 + 14.0) / 3)


def test_compute_lag_values_date_aware_handles_missing_dates():
    """A missing day mid-series produces NaN at the lookup target — no row-position
    shift that would silently use the wrong neighbor."""
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])
    series = pd.Series([1.0, 2.0, 4.0, 5.0], index=idx)
    out = compute_lag_values(series, lag_period="D", lag_period_orders=[1])
    # 2024-01-04 needs 2024-01-03, which is absent → NaN.
    assert pd.isna(out.loc["2024-01-04"])
    # 2024-01-05 needs 2024-01-04, which exists → value 4.0.
    assert out.loc["2024-01-05"] == 4.0


def test_compute_lag_values_preserves_input_order():
    """Output Series uses the input's original index, even if not sorted."""
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
    series = pd.Series([3.0, 1.0, 2.0], index=idx)
    out = compute_lag_values(series, lag_period="D", lag_period_orders=[1])
    assert list(out.index) == list(idx)


def test_compute_lag_values_rejects_unknown_agg():
    """Closed enum — 'callable' aggs not supported."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    series = pd.Series(range(10), index=idx, dtype=float)
    with pytest.raises(ValueError, match="agg must be one of"):
        compute_lag_values(series, "D", [1, 2], agg="bogus")


def test_compute_lag_values_rejects_multi_lag_without_agg():
    """If multiple orders given, agg is required (ambiguous otherwise)."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    series = pd.Series(range(10), index=idx, dtype=float)
    with pytest.raises(ValueError, match="agg is required"):
        compute_lag_values(series, "D", [1, 2])


def test_compute_lag_values_rejects_non_positive_orders():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    series = pd.Series([1.0] * 5, index=idx)
    with pytest.raises(ValueError, match="must be positive"):
        compute_lag_values(series, "D", [0])
    with pytest.raises(ValueError, match="must be positive"):
        compute_lag_values(series, "D", [-1, 1], agg="mean")


def test_compute_lag_values_requires_datetime_index():
    """Plain integer index → TypeError; signal a user error rather than silently returning junk."""
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        compute_lag_values(series, "D", [1])


# ---------------------------------------------------------------------------
# TSTransform ABC
# ---------------------------------------------------------------------------


def test_tstransform_default_str_name_is_lowercased_classname():
    """A subclass with no suffix() override falls back to lowercased class name."""

    class MyTransform(TSTransform):
        def apply(self, df, ts_config, metric_info):
            return df

    assert MyTransform().str_name() == "mytransform"


# ---------------------------------------------------------------------------
# Coarsen
# ---------------------------------------------------------------------------


def make_daily_signups_df(periods: int = 14) -> pd.DataFrame:
    """Daily wide-format signups: one metric column + sample_count, no dims."""
    return pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=periods, freq="D"),
            "signups": list(range(1, periods + 1)),  # 1, 2, …, periods
            "sample_count": [100] * periods,
        }
    )


def test_coarsen_daily_to_weekly_sum():
    """D → W with sum: 7 daily rows collapse to 1 weekly row whose value is the sum."""
    df = make_daily_signups_df(14)  # two ISO weeks Mon–Sun starting 2024-01-01
    out = Coarsen(freq="W", agg="sum").apply(df, make_ts_config("D"), StubMetricInfo())
    assert len(out) == 2
    # Week 1 (2024-01-01..07): sum(1..7) = 28; Week 2: sum(8..14) = 77.
    assert sorted(out["signups"].tolist()) == [28, 77]
    assert sorted(out["sample_count"].tolist()) == [700, 700]


def test_coarsen_preserves_dim_grouping():
    """Two segments × two weeks → 4 output rows; sums computed within (week × segment)."""
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
                    "country": ["US"] * 14,
                    "signups": [1] * 14,
                }
            ),
            pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
                    "country": ["GB"] * 14,
                    "signups": [2] * 14,
                }
            ),
        ],
        ignore_index=True,
    )
    out = Coarsen(freq="W", agg="sum").apply(df, make_ts_config("D"), StubMetricInfo(dims=["country"]))
    assert len(out) == 4  # 2 weeks × 2 countries
    us_rows = out[out["country"] == "US"]
    gb_rows = out[out["country"] == "GB"]
    assert (us_rows["signups"] == 7).all()
    assert (gb_rows["signups"] == 14).all()


def test_coarsen_does_not_mutate_input():
    """Caller's DataFrame must be untouched — we copy before bucketing."""
    df = make_daily_signups_df(7)
    df_before_ts = df["ts"].copy()
    Coarsen(freq="W", agg="sum").apply(df, make_ts_config("D"), StubMetricInfo())
    pd.testing.assert_series_equal(df["ts"], df_before_ts)


def test_coarsen_mean_agg():
    """W with mean: signups 1..7 in week one → mean = 4.0."""
    df = make_daily_signups_df(7)
    out = Coarsen(freq="W", agg="mean").apply(df, make_ts_config("D"), StubMetricInfo())
    assert len(out) == 1
    assert out["signups"].iloc[0] == 4.0


def test_coarsen_rejects_invalid_agg():
    """Closed enum — instantiation must fail fast."""
    with pytest.raises(ValueError, match="agg must be one of"):
        Coarsen(freq="W", agg="bogus")


def test_coarsen_anchors_output_to_period_end_not_start():
    """Coarsen keys output rows by the END of the period (e.g. Saturday for
    ``"W-SAT"``), NOT the start.  Regression: anchoring on start_time
    produces Sunday-anchored timestamps which pandas mis-infers as
    ``"W-SUN"`` — that mismatch caused greykite to drop every weekly row
    downstream when the model config said ``"W-SAT"``."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
            "signups": list(range(14)),
        }
    )
    out = Coarsen(freq="W-SAT", agg="sum").apply(df, make_ts_config("D"), StubMetricInfo())
    # All output timestamps must fall on Saturday (weekday 5).
    assert (out["ts"].dt.weekday == 5).all()


def test_coarsen_output_freq_round_trips_with_input_freq_alias():
    """``pd.infer_freq`` on the output must match the freq we asked for
    — otherwise downstream gap-fillers / model configs see a different
    freq than the data and silently drop rows."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=70, freq="D"),  # 10 full weeks
            "signups": list(range(70)),
        }
    )
    for week_freq in ("W-SAT", "W-SUN", "W-FRI"):
        out = Coarsen(freq=week_freq, agg="sum").apply(df, make_ts_config("D"), StubMetricInfo())
        assert pd.infer_freq(out["ts"]) == week_freq, f"Coarsen(freq={week_freq!r}) output ts inferred as " f"{pd.infer_freq(out['ts'])!r} — must round-trip"


def test_coarsen_normalizes_end_time_to_midnight():
    """End-of-period in pandas is ``Sat 23:59:59.999999`` — Coarsen must
    normalize to ``Sat 00:00:00`` so equality comparisons against
    plain dates work."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
            "signups": [1.0] * 14,
        }
    )
    out = Coarsen(freq="W-SAT", agg="sum").apply(df, make_ts_config("D"), StubMetricInfo())
    assert (out["ts"].dt.hour == 0).all()
    assert (out["ts"].dt.minute == 0).all()
    assert (out["ts"].dt.second == 0).all()
    assert (out["ts"].dt.microsecond == 0).all()


def test_coarsen_str_name_strings():
    """Suffix shape used for auto-naming JobConfig metric_id_template."""
    assert Coarsen(freq="W", agg="sum").str_name() == "weekly"
    assert Coarsen(freq="W", agg="mean").str_name() == "weekly_mean"
    assert Coarsen(freq="MS", agg="sum").str_name() == "monthly"
    # Unknown freq alias falls back to its lowercased literal.
    assert Coarsen(freq="QS", agg="sum").str_name() == "qs"


# ---------------------------------------------------------------------------
# WeightWithinPeriod
# ---------------------------------------------------------------------------


def make_info_with_signups(dims: Optional[list[str]] = None) -> StubMetricInfo:
    return StubMetricInfo(dims=dims, metrics=[StubMetric(name="signups")])


def test_weight_within_period_complete_week_sums_to_one():
    """Two complete Mon–Sun weeks: weights within each week sum to 1."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=14, freq="D"),  # both full ISO weeks
            "signups": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
        }
    )
    out = WeightWithinPeriod(period="W").apply(df, make_ts_config("D"), make_info_with_signups())
    week1 = out["signups"].iloc[:7]
    week2 = out["signups"].iloc[7:]
    assert week1.sum() == pytest.approx(1.0)
    assert week2.sum() == pytest.approx(1.0)
    # First day of week 1 has share 1/28.
    assert week1.iloc[0] == pytest.approx(1 / 28)


def test_weight_within_period_incomplete_periods_marked_nan():
    """Partial weeks at both ends: those rows must be NaN'd out."""
    # 2024-01-01 is Monday, so the first partial week is 2024-01-08-Sun lonely?
    # Use a span that starts Wednesday and ends Wednesday — both ends partial.
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-03", periods=14, freq="D"),  # Wed → Tue → Tue
            "signups": [1.0] * 14,
        }
    )
    out = WeightWithinPeriod(period="W").apply(df, make_ts_config("D"), make_info_with_signups())
    # Week 1 (Wed Jan 3–Sun Jan 7) has 5 rows; week 3 (Mon Jan 15–Tue Jan 16) has 2.
    # Only the middle week (Jan 8–14) is complete.
    full_week_mask = (out["ts"] >= "2024-01-08") & (out["ts"] <= "2024-01-14")
    assert out.loc[full_week_mask, "signups"].notna().all()
    assert out.loc[~full_week_mask, "signups"].isna().all()
    # Within the complete week each day = 1/7.
    assert out.loc[full_week_mask, "signups"].sum() == pytest.approx(1.0)


def test_weight_within_period_handles_dims_independently():
    """Two dims should be weighted within (week × dim), not pooled."""
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=7, freq="D"),
                    "country": ["US"] * 7,
                    "signups": [1.0] * 7,
                }
            ),
            pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=7, freq="D"),
                    "country": ["GB"] * 7,
                    "signups": [10.0] * 7,
                }
            ),
        ],
        ignore_index=True,
    )
    info = StubMetricInfo(dims=["country"], metrics=[StubMetric(name="signups")])
    out = WeightWithinPeriod(period="W").apply(df, make_ts_config("D"), info)
    us = out[out["country"] == "US"]["signups"]
    gb = out[out["country"] == "GB"]["signups"]
    assert us.sum() == pytest.approx(1.0)
    assert gb.sum() == pytest.approx(1.0)
    # Each US value = 1/7; each GB value = 10/70 = 1/7 — same, since uniform.
    assert np.allclose(us.to_numpy(), 1 / 7)
    assert np.allclose(gb.to_numpy(), 1 / 7)


def test_weight_within_period_zero_total_yields_nan_not_inf():
    """Bucket with all-zeros total → NaN weights, not inf or div-by-zero error."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=7, freq="D"),
            "signups": [0.0] * 7,
        }
    )
    out = WeightWithinPeriod(period="W").apply(df, make_ts_config("D"), make_info_with_signups())
    assert out["signups"].isna().all()


def test_weight_within_period_rejects_unsupported_pair():
    """Daily into monthly is variable-length — must raise via PERIOD_DIVISOR."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=30, freq="D"),
            "signups": [1.0] * 30,
        }
    )
    with pytest.raises(ValueError, match="Unsupported"):
        WeightWithinPeriod(period="M").apply(df, make_ts_config("D"), make_info_with_signups())


def test_weight_within_period_supports_anchored_week_alias():
    """``period="W-SAT"`` (Sun–Sat weeks) groups daily rows correctly.
    Regression: ``rows_per_period('D', 'W-SAT')`` previously raised
    KeyError because the lookup table only held bare ``"W"``."""
    # 2024-01-07 is Sunday → start of a Sun-Sat week. 14 days = 2 full Sun-Sat weeks.
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-07", periods=14, freq="D"),
            "signups": [1.0] * 14,
        }
    )
    out = WeightWithinPeriod(period="W-SAT").apply(df, make_ts_config("D"), make_info_with_signups())
    # Both weeks complete → no NaN, weights sum to 1 per week.
    assert out["signups"].notna().all()
    assert out["signups"].iloc[:7].sum() == pytest.approx(1.0)
    assert out["signups"].iloc[7:].sum() == pytest.approx(1.0)


def test_weight_within_period_str_name_strings():
    assert WeightWithinPeriod(period="W").str_name() == "within_week_weight"
    assert WeightWithinPeriod(period="D").str_name() == "within_day_weight"


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def test_diff_dod_subtracts_previous_day():
    """DoD (lag_period='D', n=1): each row = today − yesterday; first row = NaN."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=5, freq="D"),
            "signups": [10.0, 12.0, 15.0, 14.0, 20.0],
        }
    )
    out = Diff(lag_period="D", n_lag_periods=1).apply(df, make_ts_config("D"), make_info_with_signups())
    assert pd.isna(out["signups"].iloc[0])
    assert list(out["signups"].iloc[1:]) == [2.0, 3.0, -1.0, 6.0]


def test_diff_wow_uses_seven_days_back():
    """WoW (lag_period='W', n=1): each row = today − same-day-last-week."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
            "signups": list(range(1, 15)),  # 1..14
        }
    )
    out = Diff(lag_period="W", n_lag_periods=1).apply(df, make_ts_config("D"), make_info_with_signups())
    # First 7 rows: no week-prior → NaN.
    assert out["signups"].iloc[:7].isna().all()
    # Days 8..14 each have value − value_seven_days_earlier == 7.
    assert (out["signups"].iloc[7:] == 7.0).all()


def test_diff_multi_lag_median_baseline():
    """Median of last 3 weeks: at day 22, baseline = median(value[1], value[8], value[15])."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=22, freq="D"),
            "signups": np.arange(22, dtype=float).tolist(),  # 0..21
        }
    )
    out = Diff(lag_period="W", n_lag_periods=3, agg="median").apply(df, make_ts_config("D"), make_info_with_signups())
    # Day 22 (index 21, value=21): baseline = median(values at days 1, 8, 15)
    # = median(0, 7, 14) = 7. Diff = 21 − 7 = 14.
    assert out["signups"].iloc[21] == pytest.approx(14.0)
    # Earlier rows (no full 3-week history) → NaN somewhere (first 14 rows).
    assert out["signups"].iloc[:14].isna().all()


def test_diff_relative_returns_pct_change():
    """Relative WoW: diff / baseline; with uniform doubling, every value − last week == previous value."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
            "signups": [1.0] * 7 + [2.0] * 7,
        }
    )
    out = Diff(lag_period="W", n_lag_periods=1, relative=True).apply(df, make_ts_config("D"), make_info_with_signups())
    # Days 8..14: (2 − 1) / 1 = 1.0.
    assert np.allclose(out["signups"].iloc[7:].to_numpy(), 1.0)


def test_diff_relative_zero_baseline_yields_nan_not_inf():
    """0-baseline rows → NaN, not ±inf."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
            "signups": [0.0] * 7 + [5.0] * 7,
        }
    )
    out = Diff(lag_period="W", n_lag_periods=1, relative=True).apply(df, make_ts_config("D"), make_info_with_signups())
    # Rows 8..14 see baseline 0 → NaN.
    assert out["signups"].iloc[7:].isna().all()


def test_diff_explicit_lag_period_orders():
    """lag_period_orders=[1, 2] with mean: baseline = mean(week-1, week-2)."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=21, freq="D"),
            "signups": np.arange(21, dtype=float).tolist(),
        }
    )
    out = Diff(lag_period="W", lag_period_orders=[1, 2], agg="mean").apply(df, make_ts_config("D"), make_info_with_signups())
    # Day 21 (index 20, value=20): mean(value[6]=6, value[13]=13) = 9.5; diff = 20 − 9.5 = 10.5.
    assert out["signups"].iloc[20] == pytest.approx(10.5)


def test_diff_per_dim_uses_per_dim_history():
    """Two countries → each gets its own per-dim baseline; no cross-dim leakage."""
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
                    "country": ["US"] * 14,
                    "signups": list(range(100, 114)),  # 100..113
                }
            ),
            pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
                    "country": ["GB"] * 14,
                    "signups": list(range(1000, 1014)),  # 1000..1013
                }
            ),
        ],
        ignore_index=True,
    )
    info = StubMetricInfo(dims=["country"], metrics=[StubMetric(name="signups")])
    out = Diff(lag_period="W", n_lag_periods=1).apply(df, make_ts_config("D"), info)
    # Within US: each diff between days 8..14 = 7. Within GB: same.
    us = out[out["country"] == "US"]["signups"]
    gb = out[out["country"] == "GB"]["signups"]
    assert (us.iloc[7:] == 7.0).all()
    assert (gb.iloc[7:] == 7.0).all()


def test_diff_rejects_both_n_and_orders():
    with pytest.raises(ValueError, match="Exactly one"):
        Diff(lag_period="W", n_lag_periods=2, lag_period_orders=[1, 2], agg="mean")


def test_diff_rejects_neither_n_nor_orders():
    with pytest.raises(ValueError, match="Exactly one"):
        Diff(lag_period="W")


def test_diff_rejects_multi_lag_without_agg():
    with pytest.raises(ValueError, match="agg is required"):
        Diff(lag_period="W", n_lag_periods=3)


def test_diff_rejects_unknown_agg():
    with pytest.raises(ValueError, match="agg must be one of"):
        Diff(lag_period="W", n_lag_periods=2, agg="bogus")


def test_diff_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="must be >= 1"):
        Diff(lag_period="W", n_lag_periods=0)


def test_diff_str_name_strings():
    """Suffix shapes used by JobConfig auto-naming — these are part of the public contract."""
    assert Diff(lag_period="W", n_lag_periods=1).str_name() == "wow_diff"
    assert Diff(lag_period="D", n_lag_periods=1).str_name() == "dod_diff"
    assert Diff(lag_period="Y", n_lag_periods=1).str_name() == "yoy_diff"
    assert Diff(lag_period="W", n_lag_periods=3, agg="median").str_name() == "median_3w_diff"
    assert Diff(lag_period="W", n_lag_periods=3, agg="median", relative=True).str_name() == "rel_median_3w_diff"
    assert Diff(lag_period="W", n_lag_periods=1, relative=True).str_name() == "rel_wow_diff"
    assert Diff(lag_period="W", lag_period_orders=[1, 2, 4], agg="mean").str_name() == "mean_3w_diff"

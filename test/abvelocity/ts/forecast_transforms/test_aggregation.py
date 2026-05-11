# Original author: Reza Hosseini
"""Unit tests for ``forecast_transforms.aggregation`` helpers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from abvelocity.ts.constants import (
    ACTUAL_COL,
    FORECAST_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    METRIC_ID_COL,
    STD_COL,
)
from abvelocity.ts.forecast_transforms.aggregation import (
    SIGMA_METHODS,
    aggregate_actual_side_sum,
    aggregate_forecast_side_sum,
    apply_actual_side_share,
    apply_forecast_side_share,
    grouping_columns,
    sum_strict,
)


def test_sum_strict_all_real():
    assert sum_strict(values=pd.Series([1.0, 2.0, 3.0])) == 6.0


def test_sum_strict_any_nan_returns_nan():
    assert math.isnan(sum_strict(values=pd.Series([1.0, float("nan"), 3.0])))


def test_sum_strict_all_nan_returns_nan():
    assert math.isnan(sum_strict(values=pd.Series([float("nan"), float("nan")])))


def test_sum_strict_empty_series_returns_zero():
    """Empty series has no NaNs by definition; pandas sum returns 0."""
    assert sum_strict(values=pd.Series([], dtype=float)) == 0.0


def test_grouping_columns_picks_string_columns():
    df = pd.DataFrame(
        {
            "metric_id": ["x", "x", "y"],
            "country": ["US", "GB", "US"],
            "forecast": [1.0, 2.0, 3.0],
            "std": [0.5, 0.5, 0.5],
        }
    )
    assert grouping_columns(forecast_df=df) == ["metric_id", "country"]


def test_grouping_columns_excludes_listed():
    df = pd.DataFrame(
        {
            "metric_id": ["x", "y"],
            "_helper": ["a", "b"],
            "forecast": [1.0, 2.0],
        }
    )
    assert grouping_columns(forecast_df=df, exclude=["_helper"]) == ["metric_id"]


def test_aggregate_forecast_side_sum_basic():
    """Two rows with identical group key → one summed row."""
    df = pd.DataFrame(
        {
            "metric_id": ["x", "x"],
            FORECAST_COL: [10.0, 20.0],
            STD_COL: [3.0, 4.0],
            FORECAST_LOWER_COL: [10.0 - 1.2816 * 3.0, 20.0 - 1.2816 * 4.0],
            FORECAST_UPPER_COL: [10.0 + 1.2816 * 3.0, 20.0 + 1.2816 * 4.0],
        }
    )
    out = aggregate_forecast_side_sum(df=df, group_cols=["metric_id"], ci_coverage=0.80)
    assert len(out) == 1
    assert out[FORECAST_COL].iloc[0] == 30.0
    # σ_total = sqrt(9 + 16) = 5
    assert math.isclose(out[STD_COL].iloc[0], 5.0, rel_tol=1e-9)
    # bounds recomputed: forecast ± 1.2816 · σ
    assert math.isclose(out[FORECAST_LOWER_COL].iloc[0], 30.0 - 1.2816 * 5.0, rel_tol=1e-3)
    assert math.isclose(out[FORECAST_UPPER_COL].iloc[0], 30.0 + 1.2816 * 5.0, rel_tol=1e-3)


def test_aggregate_forecast_side_sum_does_not_touch_actual():
    """``aggregate_forecast_side_sum`` aggregates everything except actual."""
    df = pd.DataFrame(
        {
            "metric_id": ["x", "x"],
            ACTUAL_COL: [5.0, 7.0],
            FORECAST_COL: [10.0, 20.0],
        }
    )
    out = aggregate_forecast_side_sum(df=df, group_cols=["metric_id"], ci_coverage=0.80)
    assert ACTUAL_COL not in out.columns


def test_aggregate_forecast_side_sum_strict_propagates_nan():
    """One NaN in forecast → group sum is NaN."""
    df = pd.DataFrame(
        {
            "metric_id": ["x", "x"],
            FORECAST_COL: [10.0, float("nan")],
            STD_COL: [3.0, 4.0],
        }
    )
    out = aggregate_forecast_side_sum(df=df, group_cols=["metric_id"], ci_coverage=0.80)
    assert pd.isna(out[FORECAST_COL].iloc[0])


def test_aggregate_actual_side_sum_basic():
    df = pd.DataFrame(
        {
            "metric_id": ["x", "x", "y"],
            ACTUAL_COL: [5.0, 7.0, 11.0],
        }
    )
    out = aggregate_actual_side_sum(df=df, group_cols=["metric_id"])
    assert len(out) == 2
    out_x = out.loc[out["metric_id"] == "x"]
    assert out_x[ACTUAL_COL].iloc[0] == 12.0


def test_aggregate_actual_side_sum_strict_propagates_nan():
    df = pd.DataFrame(
        {
            "metric_id": ["x", "x"],
            ACTUAL_COL: [5.0, float("nan")],
        }
    )
    out = aggregate_actual_side_sum(df=df, group_cols=["metric_id"])
    assert pd.isna(out[ACTUAL_COL].iloc[0])


def test_aggregate_actual_side_sum_missing_actual_column():
    """No actual column in input → output has just group cols."""
    df = pd.DataFrame({"metric_id": ["x", "y"], FORECAST_COL: [1.0, 2.0]})
    out = aggregate_actual_side_sum(df=df, group_cols=["metric_id"])
    assert ACTUAL_COL not in out.columns
    assert list(out["metric_id"]) == ["x", "y"]


def test_apply_forecast_side_share_basic():
    """Within one group, forecast shares sum to 1."""
    df = pd.DataFrame(
        {
            METRIC_ID_COL: ["x", "x", "x"],
            FORECAST_COL: [10.0, 20.0, 30.0],
            STD_COL: [1.0, 2.0, 3.0],
            FORECAST_LOWER_COL: [10.0, 20.0, 30.0],
            FORECAST_UPPER_COL: [10.0, 20.0, 30.0],
        }
    )
    out = apply_forecast_side_share(
        df=df.copy(),
        denom_group_cols=[METRIC_ID_COL],
        ci_coverage=0.80,
    )
    assert math.isclose(out[FORECAST_COL].sum(), 1.0)


def test_apply_forecast_side_share_zero_total_yields_nan():
    df = pd.DataFrame(
        {
            METRIC_ID_COL: ["x", "x"],
            FORECAST_COL: [0.0, 0.0],
            STD_COL: [1.0, 1.0],
        }
    )
    out = apply_forecast_side_share(
        df=df.copy(),
        denom_group_cols=[METRIC_ID_COL],
        ci_coverage=0.80,
    )
    assert out[FORECAST_COL].isna().all()


def test_apply_forecast_side_share_invalid_method_raises():
    df = pd.DataFrame({METRIC_ID_COL: ["x"], FORECAST_COL: [1.0], STD_COL: [0.1]})
    with pytest.raises(ValueError, match="sigma_method must be"):
        apply_forecast_side_share(
            df=df,
            denom_group_cols=[METRIC_ID_COL],
            ci_coverage=0.80,
            sigma_method="bogus",
        )


def test_apply_actual_side_share_basic():
    df = pd.DataFrame(
        {
            METRIC_ID_COL: ["x", "x", "x"],
            ACTUAL_COL: [10.0, 20.0, 30.0],
        }
    )
    out = apply_actual_side_share(df=df.copy(), denom_group_cols=[METRIC_ID_COL])
    assert math.isclose(out[ACTUAL_COL].sum(), 1.0)


def test_apply_actual_side_share_no_actual_returns_unchanged():
    df = pd.DataFrame({METRIC_ID_COL: ["x"], FORECAST_COL: [1.0]})
    out = apply_actual_side_share(df=df.copy(), denom_group_cols=[METRIC_ID_COL])
    assert out[FORECAST_COL].iloc[0] == 1.0


def test_sigma_methods_constant_present():
    assert "constant" in SIGMA_METHODS


def test_sigma_methods_delta_present():
    assert "delta" in SIGMA_METHODS


# ---------------------------------------------------------------------------
# Demonstrative: hand-computed expected values across the full chain.
# ---------------------------------------------------------------------------


def test_demo_aggregate_forecast_side_sum_full_columns():
    """Three rows, one group.  All numeric columns aggregate per their
    class; bounds are recomputed from new forecast ± 1.2816·new_σ.
    """
    df = pd.DataFrame(
        {
            "metric_id": ["x", "x", "x"],
            "forecast": [10.0, 20.0, 30.0],
            "actual": [11.0, 19.0, 31.0],
            "std": [3.0, 4.0, 12.0],
            "forecast_lower": [10.0 - 1.2816 * 3.0, 20.0 - 1.2816 * 4.0, 30.0 - 1.2816 * 12.0],
            "forecast_upper": [10.0 + 1.2816 * 3.0, 20.0 + 1.2816 * 4.0, 30.0 + 1.2816 * 12.0],
            "longterm_growth": [8.0, 16.0, 24.0],
            "weekly_seasonality": [0.5, -0.3, -0.2],
        }
    )
    out = aggregate_forecast_side_sum(df=df, group_cols=["metric_id"], ci_coverage=0.80)

    assert out["forecast"].iloc[0] == 60.0
    assert "actual" not in out.columns  # actual is the OTHER side's job
    assert math.isclose(out["std"].iloc[0], 13.0, rel_tol=1e-9)  # sqrt(9+16+144) = 13
    assert math.isclose(out["forecast_lower"].iloc[0], 60.0 - 1.2816 * 13.0, rel_tol=1e-3)
    assert math.isclose(out["forecast_upper"].iloc[0], 60.0 + 1.2816 * 13.0, rel_tol=1e-3)
    assert out["longterm_growth"].iloc[0] == 48.0
    assert math.isclose(out["weekly_seasonality"].iloc[0], 0.0, abs_tol=1e-9)


def test_demo_aggregate_forecast_side_sum_multiple_groups():
    """Two metrics → two output rows; groups don't contaminate each other."""
    df = pd.DataFrame(
        {
            "metric_id": ["x", "x", "y", "y"],
            "forecast": [10.0, 20.0, 100.0, 200.0],
            "std": [3.0, 4.0, 5.0, 12.0],
        }
    )
    out = aggregate_forecast_side_sum(df=df, group_cols=["metric_id"], ci_coverage=0.80)
    out_x = out.loc[out["metric_id"] == "x"]
    out_y = out.loc[out["metric_id"] == "y"]
    assert out_x["forecast"].iloc[0] == 30.0
    assert out_y["forecast"].iloc[0] == 300.0
    assert math.isclose(out_x["std"].iloc[0], 5.0)
    assert math.isclose(out_y["std"].iloc[0], 13.0, rel_tol=1e-9)


def test_demo_apply_forecast_side_share_decomposition_sums_to_forecast_share():
    """``Σ component_share = forecast_share`` at each row when components add up to forecast."""
    df = pd.DataFrame(
        {
            METRIC_ID_COL: ["x", "x", "x"],
            FORECAST_COL: [10.0, 20.0, 30.0],
            STD_COL: [1.0, 2.0, 3.0],
            "longterm_growth": [4.0, 8.0, 12.0],
            "weekly_seasonality": [6.0, 12.0, 18.0],
        }
    )
    out = apply_forecast_side_share(
        df=df.copy(),
        denom_group_cols=[METRIC_ID_COL],
        ci_coverage=0.80,
    )
    expected_forecast_share = pd.Series([10.0 / 60.0, 20.0 / 60.0, 30.0 / 60.0])
    pd.testing.assert_series_equal(
        left=out[FORECAST_COL].reset_index(drop=True),
        right=expected_forecast_share,
        check_names=False,
        atol=1e-9,
    )
    component_sum = out["longterm_growth"] + out["weekly_seasonality"]
    pd.testing.assert_series_equal(
        left=component_sum.reset_index(drop=True),
        right=out[FORECAST_COL].reset_index(drop=True),
        check_names=False,
        atol=1e-9,
    )


def test_demo_apply_forecast_side_share_constant_vs_delta_explicit():
    """σ_share for both methods, hand-computed.

    constant: σ_X / |Y|;
    delta:    sqrt(σ_X² + w²·σ_Y²) / |Y|.
    """
    df = pd.DataFrame(
        {
            METRIC_ID_COL: ["x", "x", "x"],
            FORECAST_COL: [10.0, 20.0, 30.0],
            STD_COL: [3.0, 4.0, 12.0],
        }
    )
    out_constant = apply_forecast_side_share(
        df=df.copy(),
        denom_group_cols=[METRIC_ID_COL],
        ci_coverage=0.80,
        sigma_method="constant",
    )
    out_delta = apply_forecast_side_share(
        df=df.copy(),
        denom_group_cols=[METRIC_ID_COL],
        ci_coverage=0.80,
        sigma_method="delta",
    )
    expected_constant = pd.Series([3.0 / 60.0, 4.0 / 60.0, 12.0 / 60.0])
    pd.testing.assert_series_equal(
        left=out_constant[STD_COL].reset_index(drop=True),
        right=expected_constant,
        check_names=False,
        atol=1e-9,
    )
    weights = pd.Series([10.0, 20.0, 30.0]) / 60.0
    sigma_x = pd.Series([3.0, 4.0, 12.0])
    sigma_y = 13.0  # sqrt(9 + 16 + 144)
    expected_delta = ((sigma_x**2 + (weights**2) * (sigma_y**2)) ** 0.5) / 60.0
    pd.testing.assert_series_equal(
        left=out_delta[STD_COL].reset_index(drop=True),
        right=expected_delta,
        check_names=False,
        atol=1e-9,
    )


def test_apply_actual_side_share_zero_total_yields_nan():
    """All zero actuals in a group → shares NaN (avoid 0/0)."""
    df = pd.DataFrame(
        {
            METRIC_ID_COL: ["x", "x"],
            ACTUAL_COL: [0.0, 0.0],
        }
    )
    out = apply_actual_side_share(df=df.copy(), denom_group_cols=[METRIC_ID_COL])
    assert out[ACTUAL_COL].isna().all()

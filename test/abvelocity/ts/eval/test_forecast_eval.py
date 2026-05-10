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
"""Tests for compute_eval and metric helpers in eval.py."""

import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.constants import ACTUAL_COL, FORECAST_COL, FORECAST_LOWER_COL, FORECAST_UPPER_COL, HORIZON_STEP_COL, METRIC_ID_COL
from abvelocity.ts.eval import compute_coverage, compute_eval, compute_mape, compute_medape, compute_r2, compute_smape

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    """Two metrics × 3 horizon steps, perfect and imperfect forecasts."""
    rows = []
    for metric in ("m1", "m2"):
        for step in (1, 2, 3):
            actual = float(step * 10)
            forecast = actual + float(step)  # error grows with step
            rows.append(
                {
                    METRIC_ID_COL: metric,
                    HORIZON_STEP_COL: step,
                    ACTUAL_COL: actual,
                    FORECAST_COL: forecast,
                    FORECAST_LOWER_COL: forecast - 5.0,
                    FORECAST_UPPER_COL: forecast + 5.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def perfect_df():
    """Forecast equals actual — all error metrics should be 0, R2 = NaN (SS_tot = 0 if const)."""
    return pd.DataFrame(
        {
            METRIC_ID_COL: ["m1"] * 4,
            HORIZON_STEP_COL: [1, 2, 3, 4],
            ACTUAL_COL: [10.0, 20.0, 30.0, 40.0],
            FORECAST_COL: [10.0, 20.0, 30.0, 40.0],
        }
    )


# ---------------------------------------------------------------------------
# compute_mape
# ---------------------------------------------------------------------------


def test_compute_mape_known_values():
    actual = np.array([100.0, 200.0, 50.0])
    forecast = np.array([110.0, 190.0, 55.0])
    # |errors|/|actuals|: 10/100, 10/200, 5/50 = 0.1, 0.05, 0.1 → mean=0.0833... × 100
    expected = np.mean([0.1, 0.05, 0.1]) * 100
    assert np.isclose(compute_mape(actual, forecast), expected)


def test_compute_mape_excludes_zero_actuals():
    actual = np.array([0.0, 100.0])
    forecast = np.array([5.0, 110.0])
    # Only second row used: |100-110|/100 × 100 = 10.0
    assert np.isclose(compute_mape(actual, forecast), 10.0)


def test_compute_mape_all_zero_actuals_returns_nan():
    actual = np.array([0.0, 0.0])
    forecast = np.array([1.0, 2.0])
    assert np.isnan(compute_mape(actual, forecast))


# ---------------------------------------------------------------------------
# compute_smape
# ---------------------------------------------------------------------------


def test_compute_medape_known_values():
    actual = np.array([100.0, 200.0, 50.0])
    forecast = np.array([110.0, 180.0, 60.0])
    # |errors|/|actual|: 10/100, 20/200, 10/50 → 0.1, 0.1, 0.2
    # median = 0.1 → 10%
    assert np.isclose(compute_medape(actual, forecast), 10.0)


def test_compute_medape_robust_to_outlier():
    """MedAPE should be insensitive to a single outlier; MAPE should not."""
    actual = np.array([100.0, 100.0, 100.0, 100.0])
    forecast = np.array([105.0, 105.0, 105.0, 1000.0])  # last one is an outlier
    # |errors|/|actual|: 5/100, 5/100, 5/100, 900/100 → 5, 5, 5, 900 (%)
    # median = 5; mean = 228.75
    assert np.isclose(compute_medape(actual, forecast), 5.0)
    assert np.isclose(compute_mape(actual, forecast), 228.75)


def test_compute_medape_excludes_zero_actuals():
    actual = np.array([0.0, 100.0])
    forecast = np.array([5.0, 110.0])
    # Only the second row counts; |10|/100 = 10%
    assert np.isclose(compute_medape(actual, forecast), 10.0)


def test_compute_medape_all_zero_actuals_returns_nan():
    assert np.isnan(compute_medape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))


def test_compute_smape_known_values():
    actual = np.array([100.0, 200.0])
    forecast = np.array([110.0, 180.0])
    # denom = (|actual| + |forecast|) / 2: (210/2, 380/2) = (105, 190)
    # |errors|/denom: 10/105, 20/190
    expected = np.mean([10 / 105, 20 / 190]) * 100
    assert np.isclose(compute_smape(actual, forecast), expected)


def test_compute_smape_excludes_zero_denom():
    # Both 0: excluded. Non-zero row: actual=10, forecast=10 → error=0
    actual = np.array([0.0, 10.0])
    forecast = np.array([0.0, 10.0])
    assert np.isclose(compute_smape(actual, forecast), 0.0)


def test_compute_smape_all_zero_denom_returns_nan():
    actual = np.array([0.0, 0.0])
    forecast = np.array([0.0, 0.0])
    assert np.isnan(compute_smape(actual, forecast))


def test_compute_smape_symmetric():
    """sMAPE(actual, forecast) == sMAPE(forecast, actual)."""
    actual = np.array([100.0, 200.0, 50.0])
    forecast = np.array([120.0, 180.0, 60.0])
    assert np.isclose(compute_smape(actual, forecast), compute_smape(forecast, actual))


# ---------------------------------------------------------------------------
# compute_r2
# ---------------------------------------------------------------------------


def test_compute_r2_perfect_forecast():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(compute_r2(actual, actual), 1.0)


def test_compute_r2_mean_forecast():
    """Forecasting the mean gives R² = 0."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    forecast = np.full(4, np.mean(actual))
    assert np.isclose(compute_r2(actual, forecast), 0.0)


def test_compute_r2_constant_actual_returns_nan():
    actual = np.array([5.0, 5.0, 5.0])
    forecast = np.array([4.0, 5.0, 6.0])
    assert np.isnan(compute_r2(actual, forecast))


def test_compute_r2_can_be_negative():
    actual = np.array([1.0, 2.0, 3.0])
    forecast = np.array([10.0, 20.0, 30.0])  # terrible forecast
    assert compute_r2(actual, forecast) < 0


def test_compute_r2_empty_returns_nan():
    assert np.isnan(compute_r2(np.array([]), np.array([])))


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


def test_compute_coverage_all_inside():
    df = pd.DataFrame(
        {
            ACTUAL_COL: [10.0, 20.0, 30.0],
            FORECAST_LOWER_COL: [5.0, 15.0, 25.0],
            FORECAST_UPPER_COL: [15.0, 25.0, 35.0],
        }
    )
    assert np.isclose(compute_coverage(df), 1.0)


def test_compute_coverage_none_inside():
    df = pd.DataFrame(
        {
            ACTUAL_COL: [0.0, 0.0],
            FORECAST_LOWER_COL: [5.0, 5.0],
            FORECAST_UPPER_COL: [10.0, 10.0],
        }
    )
    assert np.isclose(compute_coverage(df), 0.0)


def test_compute_coverage_missing_ci_raises():
    df = pd.DataFrame({ACTUAL_COL: [10.0], FORECAST_COL: [10.0]})
    with pytest.raises(ValueError, match="forecast_lower"):
        compute_coverage(df)


# ---------------------------------------------------------------------------
# compute_eval — integration
# ---------------------------------------------------------------------------


def test_compute_eval_returns_dataframe(simple_df):
    result = compute_eval(simple_df)
    assert isinstance(result, pd.DataFrame)


def test_compute_eval_group_columns_present(simple_df):
    result = compute_eval(simple_df, group_by=(METRIC_ID_COL, HORIZON_STEP_COL))
    assert METRIC_ID_COL in result.columns
    assert HORIZON_STEP_COL in result.columns


def test_compute_eval_one_row_per_group(simple_df):
    result = compute_eval(simple_df, group_by=(METRIC_ID_COL, HORIZON_STEP_COL))
    # 2 metrics × 3 horizon steps = 6 rows
    assert len(result) == 6


def test_compute_eval_mae_correct(simple_df):
    result = compute_eval(simple_df, metrics=("mae",), group_by=(METRIC_ID_COL, HORIZON_STEP_COL))
    # For step=1: error=1, step=2: error=2, step=3: error=3
    row = result[(result[METRIC_ID_COL] == "m1") & (result[HORIZON_STEP_COL] == 2)]
    assert np.isclose(row["mae"].iloc[0], 2.0)


def test_compute_eval_perfect_forecast_mae_zero(perfect_df):
    result = compute_eval(perfect_df, metrics=("mae",), group_by=(METRIC_ID_COL,))
    assert np.isclose(result["mae"].iloc[0], 0.0)


def test_compute_eval_n_counts_valid_rows(simple_df):
    result = compute_eval(simple_df, group_by=(METRIC_ID_COL, HORIZON_STEP_COL))
    assert (result["n"] == 1).all()


def test_compute_eval_nan_rows_excluded():
    df = pd.DataFrame(
        {
            METRIC_ID_COL: ["m1", "m1", "m1"],
            HORIZON_STEP_COL: [1, 1, 1],
            ACTUAL_COL: [10.0, float("nan"), 20.0],
            FORECAST_COL: [12.0, 15.0, 22.0],
        }
    )
    result = compute_eval(df, metrics=("mae",), group_by=(METRIC_ID_COL,))
    # Only rows 0 and 2 valid: mean(|10-12|, |20-22|) = mean(2, 2) = 2.0
    assert np.isclose(result["mae"].iloc[0], 2.0)
    assert result["n"].iloc[0] == 2


def test_compute_eval_aggregate_group_by(simple_df):
    result = compute_eval(simple_df, metrics=("mae",), group_by=(METRIC_ID_COL,))
    assert len(result) == 2  # one row per metric


def test_compute_eval_unknown_metric_raises():
    df = pd.DataFrame({ACTUAL_COL: [1.0], FORECAST_COL: [1.0], METRIC_ID_COL: ["m"]})
    with pytest.raises(ValueError, match="Unknown metrics"):
        compute_eval(df, metrics=("nonexistent_metric",))


def test_compute_eval_missing_actual_raises():
    df = pd.DataFrame({FORECAST_COL: [1.0], METRIC_ID_COL: ["m"]})
    with pytest.raises(ValueError, match="'actual'"):
        compute_eval(df)


def test_compute_eval_missing_forecast_raises():
    df = pd.DataFrame({ACTUAL_COL: [1.0], METRIC_ID_COL: ["m"]})
    with pytest.raises(ValueError, match="'forecast'"):
        compute_eval(df)


def test_compute_eval_coverage_missing_ci_raises(simple_df):
    df_no_ci = simple_df.drop(columns=[FORECAST_LOWER_COL, FORECAST_UPPER_COL])
    with pytest.raises(ValueError, match="forecast_lower"):
        compute_eval(df_no_ci, metrics=("coverage",), group_by=(METRIC_ID_COL,))


def test_compute_eval_coverage_with_ci(simple_df):
    result = compute_eval(
        simple_df,
        metrics=("coverage",),
        group_by=(METRIC_ID_COL, HORIZON_STEP_COL),
    )
    assert "coverage" in result.columns
    # All actuals equal forecast ± step, within ±5 band → all covered
    assert (result["coverage"] == 1.0).all()


def test_compute_eval_coverage_without_ci_raises():
    df = pd.DataFrame(
        {
            METRIC_ID_COL: ["m1"],
            HORIZON_STEP_COL: [1],
            ACTUAL_COL: [10.0],
            FORECAST_COL: [12.0],
        }
    )
    with pytest.raises(ValueError, match="forecast_lower"):
        compute_eval(df, metrics=("coverage",), group_by=(METRIC_ID_COL,))


def test_compute_eval_all_metrics(simple_df):
    result = compute_eval(
        simple_df,
        metrics=("mae", "rmse", "mape", "smape", "r2", "medae", "coverage"),
        group_by=(METRIC_ID_COL,),
    )
    for col in ("mae", "rmse", "mape", "smape", "r2", "medae", "coverage"):
        assert col in result.columns


# ---------------------------------------------------------------------------
# residual distribution metrics: bias, sigma, q25, q50, q75, iqr
# ---------------------------------------------------------------------------
# simple_df: actual = step*10, forecast = actual + step
# → error (actual - forecast) = -step
# Grouped by (METRIC_ID_COL,): errors = [-1, -2, -3] for each metric


def test_compute_eval_bias_known_value(simple_df):
    result = compute_eval(simple_df, metrics=("bias",), group_by=(METRIC_ID_COL,))
    assert "bias" in result.columns
    # mean(-1, -2, -3) = -2.0
    assert np.isclose(result["bias"].iloc[0], -2.0)


def test_compute_eval_sigma_known_value(simple_df):
    result = compute_eval(simple_df, metrics=("sigma",), group_by=(METRIC_ID_COL,))
    assert "sigma" in result.columns
    # std([-1,-2,-3], ddof=1) = 1.0
    assert np.isclose(result["sigma"].iloc[0], 1.0)


def test_compute_eval_sigma_nan_for_single_row(simple_df):
    # Per (metric, horizon_step) each group has only 1 row → sigma must be NaN.
    result = compute_eval(simple_df, metrics=("sigma",), group_by=(METRIC_ID_COL, HORIZON_STEP_COL))
    assert result["sigma"].isna().all()


def test_compute_eval_quartiles_known_values(simple_df):
    result = compute_eval(simple_df, metrics=("q25", "q50", "q75"), group_by=(METRIC_ID_COL,))
    # sorted errors: [-3, -2, -1]
    # q25 = -2.5, q50 = -2.0, q75 = -1.5
    assert np.isclose(result["q25"].iloc[0], -2.5)
    assert np.isclose(result["q50"].iloc[0], -2.0)
    assert np.isclose(result["q75"].iloc[0], -1.5)


def test_compute_eval_iqr_known_value(simple_df):
    result = compute_eval(simple_df, metrics=("iqr",), group_by=(METRIC_ID_COL,))
    # iqr = q75 - q25 = -1.5 - (-2.5) = 1.0
    assert np.isclose(result["iqr"].iloc[0], 1.0)


def test_compute_eval_perfect_forecast_bias_zero(perfect_df):
    result = compute_eval(perfect_df, metrics=("bias",), group_by=(METRIC_ID_COL,))
    assert np.isclose(result["bias"].iloc[0], 0.0)


def test_compute_eval_perfect_forecast_sigma_zero(perfect_df):
    result = compute_eval(perfect_df, metrics=("sigma",), group_by=(METRIC_ID_COL,))
    assert np.isclose(result["sigma"].iloc[0], 0.0)


def test_compute_eval_residual_metrics_per_horizon_step(simple_df):
    # Step 2 group: single error = -2. bias=-2, q50=-2, iqr=0.
    result = compute_eval(
        simple_df,
        metrics=("bias", "q50", "iqr"),
        group_by=(METRIC_ID_COL, HORIZON_STEP_COL),
    )
    row = result[(result[METRIC_ID_COL] == "m1") & (result[HORIZON_STEP_COL] == 2)]
    assert np.isclose(row["bias"].iloc[0], -2.0)
    assert np.isclose(row["q50"].iloc[0], -2.0)
    assert np.isclose(row["iqr"].iloc[0], 0.0)


def test_compute_eval_all_residual_metrics_columns_present(simple_df):
    result = compute_eval(
        simple_df,
        metrics=("bias", "sigma", "q25", "q50", "q75", "iqr"),
        group_by=(METRIC_ID_COL,),
    )
    for col in ("bias", "sigma", "q25", "q50", "q75", "iqr"):
        assert col in result.columns


# ---------------------------------------------------------------------------
# ForecastEval — class wrapper
# ---------------------------------------------------------------------------


def test_forecast_eval_defaults_match_compute_eval(simple_df):
    """ForecastEval with defaults returns the same thing as compute_eval with defaults."""
    from abvelocity.ts.eval import ForecastEval

    via_class = ForecastEval().run(simple_df)
    via_func = compute_eval(simple_df)
    pd.testing.assert_frame_equal(via_class.reset_index(drop=True), via_func.reset_index(drop=True))


def test_forecast_eval_respects_custom_metrics_and_group_by(simple_df):
    from abvelocity.ts.eval import ForecastEval

    evaluator = ForecastEval(metrics=("mae", "rmse"), group_by=(METRIC_ID_COL,))
    out = evaluator.run(simple_df)
    assert set(out.columns) >= {METRIC_ID_COL, "mae", "rmse"}
    # No horizon_step grouping → one row per metric.
    assert len(out) == simple_df[METRIC_ID_COL].nunique()


def test_forecast_eval_rejects_unknown_metric(simple_df):
    from abvelocity.ts.eval import ForecastEval

    with pytest.raises(ValueError, match="Unknown metrics"):
        ForecastEval(metrics=("not_a_real_metric",)).run(simple_df)


# ---------------------------------------------------------------------------
# Trim — top-|error| rows dropped before computing point-error metrics
# ---------------------------------------------------------------------------


def test_compute_mape_trim_drops_largest_abs_error():
    """One huge outlier in row 5; trim=0.2 (drops ceil(5*0.2)=1 row) gives
    MAPE over the 4 clean rows — exactly 0%."""
    actual = np.array([100.0, 100.0, 100.0, 100.0, 1_000_000.0])
    forecast = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    raw = compute_mape(actual, forecast)
    trimmed = compute_mape(actual, forecast, trim=0.2)
    # Raw MAPE pulls in the outlier: ~99.99% / 5 = ~19.998%.
    assert np.isclose(raw, 19.998, atol=1e-3)
    # Trimmed: outlier dropped, 4 perfect predictions → MAPE=0.
    assert trimmed == 0.0


def test_compute_mape_trim_zero_matches_no_trim():
    actual = np.array([100.0, 110.0, 90.0, 105.0])
    forecast = np.array([95.0, 115.0, 88.0, 100.0])
    assert compute_mape(actual, forecast, trim=0.0) == compute_mape(actual, forecast)


def test_compute_mape_trim_invalid_raises():
    actual = np.array([1.0, 2.0])
    forecast = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match=r"trim must be in \[0, 0.5\)"):
        compute_mape(actual, forecast, trim=0.5)
    with pytest.raises(ValueError, match=r"trim must be in \[0, 0.5\)"):
        compute_mape(actual, forecast, trim=-0.1)


def test_compute_smape_trim_drops_largest_abs_error():
    actual = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    forecast = np.array([100.0, 100.0, 100.0, 100.0, 1_000.0])
    # Without trim the outlier dominates; with trim the 4 perfect rows give 0.
    assert compute_smape(actual, forecast, trim=0.2) == 0.0
    assert compute_smape(actual, forecast) > 30.0  # raw is large


def test_compute_eval_trim_default_is_one_percent():
    """compute_eval defaults trim=0.01 — one outlier in a 100-row group is
    dropped from MAE/RMSE/MAPE/sMAPE/MedAE."""
    n = 100
    actual = np.full(n, 100.0)
    forecast = np.full(n, 100.0)
    forecast[0] = 1_000.0  # one outlier
    df = pd.DataFrame({
        METRIC_ID_COL: ["m1"] * n,
        HORIZON_STEP_COL: list(range(n)),
        ACTUAL_COL: actual,
        FORECAST_COL: forecast,
    })
    # Group all rows together so the trim has something to drop.
    out = compute_eval(df, metrics=("mae", "mape"), group_by=(METRIC_ID_COL,))
    assert len(out) == 1
    # With default trim=0.01 (drops 1 row): MAE on remaining 99 perfect rows = 0.
    assert out.iloc[0]["mae"] == 0.0
    assert out.iloc[0]["mape"] == 0.0


def test_compute_eval_trim_zero_includes_outlier():
    n = 100
    actual = np.full(n, 100.0)
    forecast = np.full(n, 100.0)
    forecast[0] = 1_000.0
    df = pd.DataFrame({
        METRIC_ID_COL: ["m1"] * n,
        HORIZON_STEP_COL: list(range(n)),
        ACTUAL_COL: actual,
        FORECAST_COL: forecast,
    })
    out = compute_eval(df, metrics=("mae",), group_by=(METRIC_ID_COL,), trim=0.0)
    # No trimming: the one 900-error row contributes 900/100 = MAE 9.0.
    assert np.isclose(out.iloc[0]["mae"], 9.0)


def test_compute_eval_r2_unaffected_by_trim():
    """R² is intentionally computed on the un-trimmed series — trimming
    top-|error| rows would inflate R² artificially."""
    rng = np.random.default_rng(7)
    n = 100
    actual = rng.normal(100.0, 10.0, n)
    forecast = actual + rng.normal(0.0, 1.0, n)  # mostly accurate
    forecast[0] = actual[0] + 100.0  # one outlier
    df = pd.DataFrame({
        METRIC_ID_COL: ["m1"] * n,
        HORIZON_STEP_COL: list(range(n)),
        ACTUAL_COL: actual,
        FORECAST_COL: forecast,
    })
    out_default = compute_eval(df, metrics=("r2",), group_by=(METRIC_ID_COL,))
    out_no_trim = compute_eval(df, metrics=("r2",), group_by=(METRIC_ID_COL,), trim=0.0)
    # R² must be identical regardless of trim — by design.
    assert out_default.iloc[0]["r2"] == out_no_trim.iloc[0]["r2"]


def test_compute_eval_residual_metrics_unaffected_by_trim():
    """bias / sigma / quantiles measure distribution shape — trimming the
    extremes would defeat the metric. They must use un-trimmed errors."""
    n = 100
    actual = np.full(n, 100.0)
    forecast = actual.copy()
    forecast[0] = 200.0  # +100 outlier — pulls bias up by 1.0 / 100
    df = pd.DataFrame({
        METRIC_ID_COL: ["m1"] * n,
        HORIZON_STEP_COL: list(range(n)),
        ACTUAL_COL: actual,
        FORECAST_COL: forecast,
    })
    out = compute_eval(df, metrics=("bias", "sigma"), group_by=(METRIC_ID_COL,))
    # bias = mean(actual - forecast) = mean([0]*99 + [-100]) = -1.0.
    # If trim were applied, the outlier would drop and bias would be 0.0.
    assert np.isclose(out.iloc[0]["bias"], -1.0)
    # sigma uses ddof=1; with one outlier it should be > 0.
    assert out.iloc[0]["sigma"] > 5.0


def test_trim_invalid_at_compute_eval_raises():
    df = pd.DataFrame({
        METRIC_ID_COL: ["m1"] * 3,
        HORIZON_STEP_COL: [1, 2, 3],
        ACTUAL_COL: [10.0, 20.0, 30.0],
        FORECAST_COL: [10.0, 20.0, 30.0],
    })
    with pytest.raises(ValueError, match=r"trim must be in \[0, 0.5\)"):
        compute_eval(df, metrics=("mae",), trim=0.5)

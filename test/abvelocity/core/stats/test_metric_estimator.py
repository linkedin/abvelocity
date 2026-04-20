# BSD 2-CLAUSE LICENSE
# Copyright 2024, Blah Corporation. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.stats.metric_estimator import MetricEstimator

# Constants
BASE_SAMPLE_SIZE = 1000
SEED = 1317
CI_COVERAGE = 0.95
NUM_BUCKETS = 20
PLOT_WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/metric-estimator").resolve()


@pytest.fixture(scope="module")
def test_df():
    """Creates a unit-level aggregated DataFrame for testing, with two groups (A and B).

    Returns:
        A pandas DataFrame with columns 'unit', 'group', 'new_signup', 'n_renew', 'n_target', 'score', 'sample_count'.
    """
    np.random.seed(SEED)
    n = BASE_SAMPLE_SIZE // 2
    units_a = [f"u{i}" for i in range(n)]
    units_b = [f"u{i}" for i in range(n, 2 * n)]
    signup_a = np.random.binomial(1, 0.2, n)  # 20% signup rate for A
    signup_b = np.random.binomial(1, 0.25, n)  # 25% signup rate for B
    renew_a = np.random.binomial(1, 0.3, n)  # 30% renewal rate for A
    renew_b = np.random.binomial(1, 0.35, n)  # 35% renewal rate for B
    target_a = np.ones(n)  # All eligible for A
    target_b = np.ones(n)  # All eligible for B
    score_a = np.random.normal(loc=50, scale=10, size=n)
    score_b = np.random.normal(loc=55, scale=10, size=n)
    sample_count_a = np.random.randint(1, 5, n)  # Unit counts 1-4
    sample_count_b = np.random.randint(1, 5, n)
    df = pd.DataFrame(
        {
            "unit": units_a + units_b,
            "group": ["A"] * n + ["B"] * n,
            "new_signup": np.concatenate([signup_a, signup_b]),
            "n_renew": np.concatenate([renew_a, renew_b]),
            "n_target": np.concatenate([target_a, target_b]),
            "score": np.concatenate([score_a, score_b]),
            "sample_count": np.concatenate([sample_count_a, sample_count_b]),
        }
    )
    return df


def simulate_metric_data():
    """
    Simulates data for MetricEstimator convergence tests with known population parameters.
    Returns the sample DataFrame, true unweighted mean, true ratio, and true weighted mean.
    """
    np.random.seed(SEED)
    n = 10000
    units = [f"u{i}" for i in range(n)]
    signup = np.random.binomial(1, 0.2, n)  # 20% signup rate
    target = np.ones(n)  # All eligible
    score = np.random.normal(loc=50, scale=10, size=n)  # Normal scores
    sample_count = np.random.randint(1, 5, n)  # Unit counts 1-4
    df = pd.DataFrame(
        {
            "unit": units,
            "group": ["A"] * n,
            "new_signup": signup,
            "n_target": target,
            "score": score,
            "sample_count": sample_count,
        }
    )
    true_unweighted_mean = df["score"].mean()
    true_ratio = df["new_signup"].sum() / df["n_target"].sum()
    true_weighted_mean = df["score"].sum() / df["sample_count"].sum()
    return df, true_unweighted_mean, true_ratio, true_weighted_mean


@pytest.mark.parametrize(
    "metric_def, condition, expected_name, expected_value",
    [
        (
            Metric(
                numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
                denominator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Users"),
                numerator_agg="SUM",
                denominator_agg="COUNT",
            ),
            "group == 'A'",
            "Signups/Users",
            lambda df: df[df["group"] == "A"]["new_signup"].sum() / df[df["group"] == "A"]["new_signup"].count(),
        ),
        (
            Metric(
                numerator=UMetric(col="n_renew", agg="MAX", fill_na=0, name="Renewals"),
                denominator=UMetric(col="n_target", agg="MAX", fill_na=0, name="Eligible"),
                numerator_agg="SUM",
                denominator_agg="SUM",
            ),
            "group == 'B'",
            "Renewals/Eligible",
            lambda df: df[df["group"] == "B"]["n_renew"].sum() / df[df["group"] == "B"]["n_target"].sum(),
        ),
        (
            Metric(
                numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
                numerator_agg="SUM",
            ),
            lambda df: df[df["group"] == "A"],
            "Signups",
            lambda df: df[df["group"] == "A"]["new_signup"].sum(),
        ),
        (
            Metric(
                numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
                denominator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Users"),
                numerator_agg="SUM",
                denominator_agg="COUNT",
            ),
            None,
            "Signups/Users",
            lambda df: df["new_signup"].sum() / df["new_signup"].count(),
        ),
        (
            Metric(
                numerator=UMetric(col="score", agg="MAX", fill_na=0, name="Score"),
                numerator_agg="MEAN",
                sample_count=UMetric(col="sample_count", agg="MAX", fill_na=0, name="SampleCount"),
            ),
            "group == 'A'",
            "Score",
            lambda df: df[df["group"] == "A"]["score"].sum() / df[df["group"] == "A"]["sample_count"].sum(),
        ),
        (
            Metric(
                numerator=UMetric(col="score", agg="MAX", fill_na=0, name="Score"),
                denominator=UMetric(col="n_renew", agg="MAX", fill_na=0, name="Renewals"),
                numerator_agg="MEAN",
                denominator_agg="MEAN",
                sample_count=UMetric(col="sample_count", agg="MAX", fill_na=0, name="SampleCount"),
            ),
            "group == 'B'",
            "Score/Renewals",
            lambda df: (df[df["group"] == "B"]["score"].sum() / df[df["group"] == "B"]["sample_count"].sum())
            / (df[df["group"] == "B"]["n_renew"].sum() / df[df["group"] == "B"]["sample_count"].sum()),
        ),
    ],
)
def test_metric_estimator(test_df, metric_def, condition, expected_name, expected_value):
    """Tests MetricEstimator for various metrics and conditions on unit-level aggregated data."""
    estimator = MetricEstimator(metric=metric_def, param=condition)
    assert estimator.name == expected_name
    result = estimator.estimator_func_with_inferred_param(test_df)[0]
    expected = expected_value(test_df)
    assert np.isclose(result, expected, rtol=1e-5)
    result_explicit = estimator.estimator_func(test_df, param=condition)[0]
    assert np.isclose(result_explicit, expected, rtol=1e-5)


def test_metric_estimator_dunder_methods(test_df):
    """Tests dunder methods for MetricEstimator (difference and percent difference)."""
    signup_metric = Metric(
        numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
        denominator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Users"),
        numerator_agg="SUM",
        denominator_agg="COUNT",
    )
    est_a = MetricEstimator(signup_metric, param="group == 'A'")
    est_b = MetricEstimator(signup_metric, param="group == 'B'")
    signup_rate_a = test_df[test_df["group"] == "A"]["new_signup"].sum() / test_df[test_df["group"] == "A"]["new_signup"].count()
    signup_rate_b = test_df[test_df["group"] == "B"]["new_signup"].sum() / test_df[test_df["group"] == "B"]["new_signup"].count()
    diff_est = est_a - est_b
    assert diff_est.name == "(Signups/Users - Signups/Users)"
    result_diff = diff_est.estimator_func_with_inferred_param(test_df)[0]
    assert np.isclose(result_diff, signup_rate_a - signup_rate_b, rtol=1e-5)
    percent_diff_est = 100 * (est_a - est_b) / est_b
    assert percent_diff_est.name == "((100 * (Signups/Users - Signups/Users)) / Signups/Users)"
    result_percent_diff = percent_diff_est.estimator_func_with_inferred_param(test_df)[0]
    expected_percent_diff = 100 * (signup_rate_a - signup_rate_b) / signup_rate_b
    assert np.isclose(result_percent_diff, expected_percent_diff, rtol=1e-5)


def test_metric_estimator_jackknife(test_df):
    """Tests calc_jk_stats for MetricEstimator."""
    signup_metric = Metric(
        numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
        denominator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Users"),
        numerator_agg="SUM",
        denominator_agg="COUNT",
    )
    estimator = MetricEstimator(signup_metric, param="group == 'A'")
    result = estimator.calc_jk_stats(test_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
    assert isinstance(result, dict)
    assert "estimator_value" in result
    assert "estimator_varcov" in result
    assert "ci" in result
    assert np.isclose(estimator.jk_value, result["estimator_value"], rtol=1e-5)
    assert np.isclose(estimator.jk_varcov, result["estimator_varcov"][0, 0], rtol=1e-5)
    assert np.allclose(estimator.jk_ci, result["ci"], rtol=1e-5)
    assert estimator.dof == NUM_BUCKETS - 1
    expected_value = test_df[test_df["group"] == "A"]["new_signup"].sum() / test_df[test_df["group"] == "A"]["new_signup"].count()
    assert np.isclose(estimator.jk_value, expected_value, rtol=1e-2)


def test_metric_estimator_error_handling(test_df):
    """Tests error handling in MetricEstimator."""
    signup_metric = Metric(
        numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
        denominator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Users"),
        numerator_agg="SUM",
        denominator_agg="COUNT",
    )
    invalid_metric = Metric(
        numerator=UMetric(col="missing_col", agg="MAX", fill_na=0, name="Missing"),
        numerator_agg="SUM",
    )
    estimator = MetricEstimator(invalid_metric)
    with pytest.raises(KeyError, match="Numerator column 'missing_col' not found"):
        estimator.estimator_func_with_inferred_param(test_df)
    invalid_metric = Metric(
        numerator=UMetric(col="score", agg="MAX", fill_na=0, name="Score"),
        numerator_agg="MEAN",
        sample_count=UMetric(col="missing_sample_count", agg="MAX", fill_na=0, name="SampleCount"),
    )
    estimator = MetricEstimator(invalid_metric)
    with pytest.raises(KeyError, match="Sample count column 'missing_sample_count' not found"):
        estimator.estimator_func_with_inferred_param(test_df)
    invalid_metric = Metric(
        numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
        numerator_agg="INVALID",
    )
    estimator = MetricEstimator(invalid_metric)
    with pytest.raises(NotImplementedError, match="Numerator aggregation INVALID not supported"):
        estimator.estimator_func_with_inferred_param(test_df)
    zero_denom_metric = Metric(
        numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
        denominator=UMetric(col="n_renew", agg="MAX", fill_na=0, name="Renewals"),
        numerator_agg="SUM",
        denominator_agg="SUM",
    )
    df_zero = test_df.copy()
    df_zero["n_renew"] = 0
    estimator = MetricEstimator(zero_denom_metric, param="group == 'A'")
    with pytest.raises(ValueError, match="Denominator aggregated to zero"):
        estimator.estimator_func_with_inferred_param(df_zero)
    zero_sample_metric = Metric(
        numerator=UMetric(col="score", agg="MAX", fill_na=0, name="Score"),
        numerator_agg="MEAN",
        sample_count=UMetric(col="sample_count", agg="MAX", fill_na=0, name="SampleCount"),
    )
    df_zero_sample = test_df.copy()
    df_zero_sample["sample_count"] = 0
    estimator = MetricEstimator(zero_sample_metric, param="group == 'A'")
    with pytest.raises(ValueError, match="Sample count sum is zero for numerator"):
        estimator.estimator_func_with_inferred_param(df_zero_sample)
    estimator = MetricEstimator(signup_metric)
    with pytest.raises(ValueError, match="Invalid query condition: invalid_col == 1"):
        estimator.estimator_func(test_df, param="invalid_col == 1")
    with pytest.raises(ValueError, match="Condition must be a string or callable"):
        estimator.estimator_func(test_df, param=123)


def test_metric_estimator_standard_values_no_denominator(test_df):
    """Tests MetricEstimator with compute_standard_values=True for a metric without a denominator."""
    metric = Metric(
        numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
        numerator_agg="SUM",
    )
    estimator = MetricEstimator(metric=metric, compute_standard_values=True, param="group == 'A'")
    assert estimator.standard_names == ["sum_numer", "mean_numer"]
    result = estimator.estimator_func_with_inferred_param(test_df)
    expected = np.array(
        [
            test_df[test_df["group"] == "A"]["new_signup"].sum(),
            test_df[test_df["group"] == "A"]["new_signup"].mean(),
        ]
    )
    assert np.allclose(result, expected, rtol=1e-5)


def test_metric_estimator_standard_values_with_denominator(test_df):
    """Tests MetricEstimator with compute_standard_values=True for a metric with a denominator."""
    metric = Metric(
        numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
        denominator=UMetric(col="n_target", agg="MAX", fill_na=0, name="Eligible"),
        numerator_agg="SUM",
        denominator_agg="SUM",
    )
    estimator = MetricEstimator(metric=metric, compute_standard_values=True, param="group == 'A'")
    assert estimator.standard_names == [
        "sum_numer",
        "mean_numer",
        "sum_denom",
        "mean_denom",
        "sum_ratio",
        "mean_ratio",
    ]
    df_a = test_df[test_df["group"] == "A"]
    sum_denom = df_a["n_target"].sum()
    mean_denom = df_a["n_target"].mean()
    expected = np.array(
        [
            df_a["new_signup"].sum(),
            df_a["new_signup"].mean(),
            sum_denom,
            mean_denom,
            df_a["new_signup"].sum() / sum_denom if sum_denom != 0 else 0,
            df_a["new_signup"].mean() / mean_denom if mean_denom != 0 else 0,
        ]
    )
    result = estimator.estimator_func_with_inferred_param(test_df)
    assert np.allclose(result, expected, rtol=1e-5)


def test_metric_estimator_standard_values_zero_denominator_ratios(test_df):
    """Tests MetricEstimator with compute_standard_values=True when denominator aggregates are zero."""
    metric = Metric(
        numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
        denominator=UMetric(col="n_renew", agg="MAX", fill_na=0, name="Renewals"),
        numerator_agg="SUM",
        denominator_agg="SUM",
    )
    df_zero = test_df.copy()
    df_zero.loc[df_zero["group"] == "A", "n_renew"] = 0
    estimator = MetricEstimator(metric=metric, compute_standard_values=True, param="group == 'A'")
    assert estimator.standard_names == [
        "sum_numer",
        "mean_numer",
        "sum_denom",
        "mean_denom",
        "sum_ratio",
        "mean_ratio",
    ]
    df_a = df_zero[df_zero["group"] == "A"]
    sum_denom = df_a["n_renew"].sum()
    mean_denom = df_a["n_renew"].mean()
    expected = np.array(
        [
            df_a["new_signup"].sum(),
            df_a["new_signup"].mean(),
            sum_denom,
            mean_denom,
            df_a["new_signup"].sum() / sum_denom if sum_denom != 0 else 0,
            df_a["new_signup"].mean() / mean_denom if mean_denom != 0 else 0,
        ]
    )
    result = estimator.estimator_func_with_inferred_param(df_zero)
    assert np.allclose(result, expected, rtol=1e-5)
    assert result[2] == 0.0
    assert result[3] == 0.0
    assert result[4] == 0.0
    assert result[5] == 0.0


def test_metric_estimator_standard_values_avg_with_sample_count(test_df):
    """Tests MetricEstimator with compute_standard_values=True and AVG aggregation with sample_count."""
    metric = Metric(
        numerator=UMetric(col="score", agg="MAX", fill_na=0, name="Score"),
        numerator_agg="AVG",
        sample_count=UMetric(col="sample_count", agg="MAX", fill_na=0, name="SampleCount"),
    )
    estimator = MetricEstimator(metric=metric, compute_standard_values=True, param="group == 'A'")
    assert estimator.standard_names == ["sum_numer", "mean_numer"]
    df_a = test_df[test_df["group"] == "A"]
    expected = np.array([df_a["score"].sum(), df_a["score"].sum() / df_a["sample_count"].sum()])
    result = estimator.estimator_func_with_inferred_param(test_df)
    assert np.allclose(result, expected, rtol=1e-5)


@pytest.mark.plot
def test_plot_mean_numer_convergence():
    """
    Plots the estimated mean_numer and 95% jackknife CI for increasing sample sizes,
    showing convergence to the true unweighted mean.
    """
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)
    df, true_unweighted_mean, _, _ = simulate_metric_data()
    sample_sizes = list(range(25, 1001, 25))
    estimates, ci_lower, ci_upper = [], [], []
    metric = Metric(
        numerator=UMetric(col="score", agg="MAX", fill_na=0, name="Score"),
        numerator_agg="MEAN",
    )
    estimator = MetricEstimator(metric=metric, compute_standard_values=True, param="group == 'A'")
    for n in sample_sizes:
        sample_df = df[:n]
        jk_result = estimator.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        estimates.append(jk_result["estimator_value"][1])  # mean_numer
        ci_lower.append(jk_result["ci"][1][0])
        ci_upper.append(jk_result["ci"][1][1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_unweighted_mean] * len(sample_sizes),
            mode="lines",
            name=f"True Mean ({true_unweighted_mean:.2f})",
            line=dict(color="blue", dash="dash"),
            legendgroup="True",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="95% Jackknife CI",
            showlegend=True,
            legendgroup="Estimate",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=estimates,
            mode="lines",
            name="Estimated mean_numer",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
        )
    )
    fig.update_layout(
        title="Convergence of mean_numer (Unweighted Mean) with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Estimated Mean",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_mean_numer_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


@pytest.mark.plot
def test_plot_sum_ratio_convergence():
    """
    Plots the estimated sum_ratio and 95% jackknife CI for increasing sample sizes,
    showing convergence to the true population ratio.
    """
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)
    df, _, true_ratio, _ = simulate_metric_data()
    sample_sizes = list(range(25, 1001, 25))
    estimates, ci_lower, ci_upper = [], [], []
    metric = Metric(
        numerator=UMetric(col="new_signup", agg="MAX", fill_na=0, name="Signups"),
        denominator=UMetric(col="n_target", agg="MAX", fill_na=0, name="Eligible"),
        numerator_agg="SUM",
        denominator_agg="SUM",
    )
    estimator = MetricEstimator(metric=metric, compute_standard_values=True, param="group == 'A'")
    for n in sample_sizes:
        sample_df = df[:n]
        jk_result = estimator.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        estimates.append(jk_result["estimator_value"][4])  # sum_ratio
        ci_lower.append(jk_result["ci"][4][0])
        ci_upper.append(jk_result["ci"][4][1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_ratio] * len(sample_sizes),
            mode="lines",
            name=f"True Ratio ({true_ratio:.2f})",
            line=dict(color="blue", dash="dash"),
            legendgroup="True",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="95% Jackknife CI",
            showlegend=True,
            legendgroup="Estimate",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=estimates,
            mode="lines",
            name="Estimated sum_ratio",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
        )
    )
    fig.update_layout(
        title="Convergence of sum_ratio (sum(Signups)/sum(Eligible)) with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Estimated Ratio",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_sum_ratio_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


@pytest.mark.plot
def test_plot_weighted_mean_numer_convergence():
    """
    Plots the estimated mean_numer (with sample_count) and 95% jackknife CI for increasing sample sizes,
    showing convergence to the true weighted mean.
    """
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)
    df, _, _, true_weighted_mean = simulate_metric_data()
    sample_sizes = list(range(25, 1001, 25))
    estimates, ci_lower, ci_upper = [], [], []
    metric = Metric(
        numerator=UMetric(col="score", agg="MAX", fill_na=0, name="Score"),
        numerator_agg="AVG",
        sample_count=UMetric(col="sample_count", agg="MAX", fill_na=0, name="SampleCount"),
    )
    estimator = MetricEstimator(metric=metric, compute_standard_values=True, param="group == 'A'")
    for n in sample_sizes:
        sample_df = df[:n]
        jk_result = estimator.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        estimates.append(jk_result["estimator_value"][1])  # mean_numer
        ci_lower.append(jk_result["ci"][1][0])
        ci_upper.append(jk_result["ci"][1][1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_weighted_mean] * len(sample_sizes),
            mode="lines",
            name=f"True Weighted Mean ({true_weighted_mean:.2f})",
            line=dict(color="blue", dash="dash"),
            legendgroup="True",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="95% Jackknife CI",
            showlegend=True,
            legendgroup="Estimate",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=estimates,
            mode="lines",
            name="Estimated mean_numer",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
        )
    )
    fig.update_layout(
        title="Convergence of mean_numer (Weighted Mean with sample_count) with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Estimated Weighted Mean",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_weighted_mean_numer_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")

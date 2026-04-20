# blah/abvelocity/stats/test_diff_estimator_constructor.py
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
from abvelocity.core.stats.diff_estimator_constructor import DiffEstimatorConstructor
from abvelocity.core.stats.metric_estimator import MetricEstimator
from scipy.stats import norm, poisson

# Constants
POPULATION_SIZE = 20000
BASE_SAMPLE_SIZE = 1000
EXPERIMENT_ASSIGNMENT_WEIGHTS = {"Control": 0.5, "Treatment": 0.5}
SEED = 1317
CI_COVERAGE = 0.95
NUM_BUCKETS = 20
PLOT_WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/diff-estimator-constructor").resolve()


def generate_correlated_poisson(n, mean1, mean2, corr):
    """Generate correlated Poisson variables using a Gaussian copula."""
    np.random.seed(SEED)
    cov_matrix = [[1, corr], [corr, 1]]
    normal_data = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=n)
    uniform_data = norm.cdf(normal_data)
    poisson1 = poisson.ppf(uniform_data[:, 0], mean1)
    poisson2 = poisson.ppf(uniform_data[:, 1], mean2)
    return np.maximum(poisson1, 0), np.maximum(poisson2, 0)  # Ensure non-negative


@pytest.fixture(scope="module")
def test_data():
    """Creates a randomized DataFrame with correlated Poisson impressions and clicks.

    Returns:
        Tuple of (sample_df, pop_df, true_values) where:
        - sample_df: First 2*BASE_SAMPLE_SIZE rows of randomized population DataFrame.
        - pop_df: Full randomized population DataFrame (POPULATION_SIZE rows).
        - true_values: Dict with true mean differences, percentage differences, and CTR differences.
    """
    np.random.seed(SEED)
    # Randomly assign units to control using binomial
    n_control = np.random.binomial(POPULATION_SIZE, EXPERIMENT_ASSIGNMENT_WEIGHTS["Control"])
    n_treatment = POPULATION_SIZE - n_control

    # Control: impressions mean=100, clicks mean=10
    control_impressions, control_clicks = generate_correlated_poisson(n_control, mean1=100, mean2=10, corr=0.5)
    control_df = pd.DataFrame(
        {
            "unit": [f"cu{i}" for i in range(n_control)],
            "variant": ["Control"] * n_control,
            "impressions": control_impressions,
            "clicks": control_clicks,
        }
    )

    # Treatment: impressions mean=110 (10% lift), clicks mean=12 (20% lift)
    treatment_impressions, treatment_clicks = generate_correlated_poisson(n_treatment, mean1=110, mean2=12, corr=0.5)
    treatment_df = pd.DataFrame(
        {
            "unit": [f"tu{i}" for i in range(n_treatment)],
            "variant": ["Treatment"] * n_treatment,
            "impressions": treatment_impressions,
            "clicks": treatment_clicks,
        }
    )

    # Combine and randomize rows
    pop_df = pd.concat([control_df, treatment_df], ignore_index=True)
    pop_df = pop_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Sample DataFrame
    sample_df = pop_df[: 2 * BASE_SAMPLE_SIZE]

    # True population values
    true_values = {
        "impressions_mean_diff": (
            pop_df[pop_df["variant"] == "Treatment"]["impressions"].mean() - pop_df[pop_df["variant"] == "Control"]["impressions"].mean()
        ),
        "clicks_mean_diff": (pop_df[pop_df["variant"] == "Treatment"]["clicks"].mean() - pop_df[pop_df["variant"] == "Control"]["clicks"].mean()),
        "impressions_pcnt_diff": (
            (pop_df[pop_df["variant"] == "Treatment"]["impressions"].mean() - pop_df[pop_df["variant"] == "Control"]["impressions"].mean())
            / pop_df[pop_df["variant"] == "Control"]["impressions"].mean()
            * 100
        ),
        "clicks_pcnt_diff": (
            (pop_df[pop_df["variant"] == "Treatment"]["clicks"].mean() - pop_df[pop_df["variant"] == "Control"]["clicks"].mean())
            / pop_df[pop_df["variant"] == "Control"]["clicks"].mean()
            * 100
        ),
        "ctr_diff": (
            (pop_df[pop_df["variant"] == "Treatment"]["clicks"].sum() / pop_df[pop_df["variant"] == "Treatment"]["impressions"].sum())
            - (pop_df[pop_df["variant"] == "Control"]["clicks"].sum() / pop_df[pop_df["variant"] == "Control"]["impressions"].sum())
        ),
        "ctr_pcnt_diff": (
            (
                (pop_df[pop_df["variant"] == "Treatment"]["clicks"].sum() / pop_df[pop_df["variant"] == "Treatment"]["impressions"].sum())
                - (pop_df[pop_df["variant"] == "Control"]["clicks"].sum() / pop_df[pop_df["variant"] == "Control"]["impressions"].sum())
            )
            / (pop_df[pop_df["variant"] == "Control"]["clicks"].sum() / pop_df[pop_df["variant"] == "Control"]["impressions"].sum())
            * 100
        ),
    }

    return sample_df, pop_df, true_values


def test_diff_estimator_means(test_data):
    """Tests DiffEstimatorConstructor for mean differences and percentage differences."""
    sample_df, _, true_values = test_data

    # Estimators for impressions and clicks means
    impressions_metric = Metric(
        numerator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="MEAN",
    )
    clicks_metric = Metric(
        numerator=UMetric(col="clicks", agg="MAX", fill_na=0, name="Clicks"),
        numerator_agg="MEAN",
    )
    impressions_control_est = MetricEstimator(metric=impressions_metric, param="variant == 'Control'")
    impressions_treatment_est = MetricEstimator(metric=impressions_metric, param="variant == 'Treatment'")
    clicks_control_est = MetricEstimator(metric=clicks_metric, param="variant == 'Control'")
    clicks_treatment_est = MetricEstimator(metric=clicks_metric, param="variant == 'Treatment'")

    # Test simple_diff for impressions and clicks
    constructor_diff = DiffEstimatorConstructor(diff_type="simple_diff", name="Mean_Diff")
    impressions_diff_est = constructor_diff.construct(control_estimator=impressions_control_est, treatment_estimator=impressions_treatment_est)
    clicks_diff_est = constructor_diff.construct(control_estimator=clicks_control_est, treatment_estimator=clicks_treatment_est)

    result_impressions_diff = impressions_diff_est.estimator_func_with_inferred_param(sample_df)
    result_clicks_diff = clicks_diff_est.estimator_func_with_inferred_param(sample_df)

    # Expected differences (Treatment - Control)
    expected_impressions_diff = (
        sample_df[sample_df["variant"] == "Treatment"]["impressions"].mean() - sample_df[sample_df["variant"] == "Control"]["impressions"].mean()
    )
    expected_clicks_diff = sample_df[sample_df["variant"] == "Treatment"]["clicks"].mean() - sample_df[sample_df["variant"] == "Control"]["clicks"].mean()

    assert np.allclose(result_impressions_diff, expected_impressions_diff, rtol=1e-2)
    assert np.allclose(result_clicks_diff, expected_clicks_diff, rtol=1e-2)
    assert impressions_diff_est.name == "Mean_Diff"
    assert clicks_diff_est.name == "Mean_Diff"

    # Compare to true population differences
    assert np.isclose(result_impressions_diff[0], true_values["impressions_mean_diff"], rtol=1e-1)
    assert np.isclose(result_clicks_diff[0], true_values["clicks_mean_diff"], rtol=1e-1)

    # Test pcnt_diff for impressions and clicks
    constructor_pct = DiffEstimatorConstructor(diff_type="pcnt_diff", name="Mean_Pcnt_Diff")
    impressions_pct_est = constructor_pct.construct(control_estimator=impressions_control_est, treatment_estimator=impressions_treatment_est)
    clicks_pct_est = constructor_pct.construct(control_estimator=clicks_control_est, treatment_estimator=clicks_treatment_est)

    result_impressions_pct = impressions_pct_est.estimator_func_with_inferred_param(sample_df)
    result_clicks_pct = clicks_pct_est.estimator_func_with_inferred_param(sample_df)

    control_impressions_mean = sample_df[sample_df["variant"] == "Control"]["impressions"].mean()
    control_clicks_mean = sample_df[sample_df["variant"] == "Control"]["clicks"].mean()
    expected_impressions_pct = (
        (sample_df[sample_df["variant"] == "Treatment"]["impressions"].mean() - sample_df[sample_df["variant"] == "Control"]["impressions"].mean())
        / control_impressions_mean
        * 100
    )
    expected_clicks_pct = (
        (sample_df[sample_df["variant"] == "Treatment"]["clicks"].mean() - sample_df[sample_df["variant"] == "Control"]["clicks"].mean())
        / control_clicks_mean
        * 100
    )

    assert np.allclose(result_impressions_pct, expected_impressions_pct, rtol=1e-2)
    assert np.allclose(result_clicks_pct, expected_clicks_pct, rtol=1e-2)
    assert impressions_pct_est.name == "Mean_Pcnt_Diff"
    assert clicks_pct_est.name == "Mean_Pcnt_Diff"

    # Compare to true population percentage differences
    assert np.isclose(result_impressions_pct[0], true_values["impressions_pcnt_diff"], rtol=1e-1)
    assert np.isclose(result_clicks_pct[0], true_values["clicks_pcnt_diff"], rtol=1e-1)


def test_diff_estimator_ctr(test_data):
    """Tests DiffEstimatorConstructor for CTR (clicks/impressions) differences."""
    sample_df, _, true_values = test_data

    # CTR estimator
    ctr_metric = Metric(
        numerator=UMetric(col="clicks", agg="MAX", fill_na=0, name="Clicks"),
        denominator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="SUM",
        denominator_agg="SUM",
    )
    ctr_control_est = MetricEstimator(metric=ctr_metric, param="variant == 'Control'")
    ctr_treatment_est = MetricEstimator(metric=ctr_metric, param="variant == 'Treatment'")

    # Test simple_diff for CTR
    constructor_diff = DiffEstimatorConstructor(diff_type="simple_diff", name="CTR_Diff")
    ctr_diff_est = constructor_diff.construct(control_estimator=ctr_control_est, treatment_estimator=ctr_treatment_est)
    result_ctr_diff = ctr_diff_est.estimator_func_with_inferred_param(sample_df)

    expected_ctr_diff = (
        sample_df[sample_df["variant"] == "Treatment"]["clicks"].sum() / sample_df[sample_df["variant"] == "Treatment"]["impressions"].sum()
    ) - (sample_df[sample_df["variant"] == "Control"]["clicks"].sum() / sample_df[sample_df["variant"] == "Control"]["impressions"].sum())

    assert np.allclose(result_ctr_diff, expected_ctr_diff, rtol=1e-2)
    assert ctr_diff_est.name == "CTR_Diff"
    assert np.isclose(result_ctr_diff[0], true_values["ctr_diff"], rtol=1e-1)

    # Test pcnt_diff for CTR
    constructor_pct = DiffEstimatorConstructor(diff_type="pcnt_diff", name="CTR_Pcnt_Diff")
    ctr_pct_est = constructor_pct.construct(control_estimator=ctr_control_est, treatment_estimator=ctr_treatment_est)
    result_ctr_pct = ctr_pct_est.estimator_func_with_inferred_param(sample_df)

    control_ctr = sample_df[sample_df["variant"] == "Control"]["clicks"].sum() / sample_df[sample_df["variant"] == "Control"]["impressions"].sum()
    expected_ctr_pct = (
        (
            (sample_df[sample_df["variant"] == "Treatment"]["clicks"].sum() / sample_df[sample_df["variant"] == "Treatment"]["impressions"].sum())
            - (sample_df[sample_df["variant"] == "Control"]["clicks"].sum() / sample_df[sample_df["variant"] == "Control"]["impressions"].sum())
        )
        / control_ctr
        * 100
    )

    assert np.allclose(result_ctr_pct, expected_ctr_pct, rtol=1e-2)
    assert ctr_pct_est.name == "CTR_Pcnt_Diff"
    assert np.isclose(result_ctr_pct[0], true_values["ctr_pcnt_diff"], rtol=1e-1)


def test_diff_estimator_both(test_data):
    """Tests DiffEstimatorConstructor with diff_type='both'."""
    sample_df, _, true_values = test_data

    # Impressions mean estimator
    impressions_metric = Metric(
        numerator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="MEAN",
    )
    impressions_control_est = MetricEstimator(metric=impressions_metric, param="variant == 'Control'")
    impressions_treatment_est = MetricEstimator(metric=impressions_metric, param="variant == 'Treatment'")

    # Test both for impressions
    constructor_both = DiffEstimatorConstructor(diff_type="both", name="Impressions_Both")
    impressions_both_est = constructor_both.construct(control_estimator=impressions_control_est, treatment_estimator=impressions_treatment_est)
    result_both = impressions_both_est.estimator_func_with_inferred_param(sample_df)

    control_impressions_mean = sample_df[sample_df["variant"] == "Control"]["impressions"].mean()
    expected_both = np.array(
        [
            sample_df[sample_df["variant"] == "Treatment"]["impressions"].mean() - sample_df[sample_df["variant"] == "Control"]["impressions"].mean(),
            (sample_df[sample_df["variant"] == "Treatment"]["impressions"].mean() - sample_df[sample_df["variant"] == "Control"]["impressions"].mean())
            / control_impressions_mean
            * 100,
        ]
    )

    assert np.allclose(result_both, expected_both, rtol=1e-2)
    assert impressions_both_est.name == "Impressions_Both"
    assert np.isclose(result_both[0], true_values["impressions_mean_diff"], rtol=1e-1)
    assert np.isclose(result_both[1], true_values["impressions_pcnt_diff"], rtol=1e-1)


def test_plot_diff_convergence(test_data):
    """Plots convergence of impressions mean differences (simple_diff and pcnt_diff) with 95% jackknife CI."""
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)
    _, pop_df, true_values = test_data
    sample_sizes = list(range(50, 2001, 50))  # Total sample size (Control + Treatment)
    diff_estimates, diff_ci_lower, diff_ci_upper = [], [], []
    pcnt_estimates, pcnt_ci_lower, pcnt_ci_upper = [], [], []

    impressions_metric = Metric(
        numerator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="MEAN",
    )
    impressions_control_est = MetricEstimator(metric=impressions_metric, param="variant == 'Control'")
    impressions_treatment_est = MetricEstimator(metric=impressions_metric, param="variant == 'Treatment'")
    constructor_diff = DiffEstimatorConstructor(diff_type="simple_diff")
    constructor_pct = DiffEstimatorConstructor(diff_type="pcnt_diff")

    for n in sample_sizes:
        sample_df = pop_df[:n]
        # Simple diff
        diff_est = constructor_diff.construct(control_estimator=impressions_control_est, treatment_estimator=impressions_treatment_est)
        jk_result_diff = diff_est.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        diff_estimates.append(jk_result_diff["estimator_value"][0])
        diff_ci_lower.append(jk_result_diff["ci"][0][0])
        diff_ci_upper.append(jk_result_diff["ci"][0][1])
        # Percent diff
        pcnt_est = constructor_pct.construct(control_estimator=impressions_control_est, treatment_estimator=impressions_treatment_est)
        jk_result_pct = pcnt_est.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        pcnt_estimates.append(jk_result_pct["estimator_value"][0])
        pcnt_ci_lower.append(jk_result_pct["ci"][0][0])
        pcnt_ci_upper.append(jk_result_pct["ci"][0][1])

    fig = go.Figure()
    # Simple diff
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_values["impressions_mean_diff"]] * len(sample_sizes),
            mode="lines",
            name=f"True Mean Diff ({true_values['impressions_mean_diff']:.2f})",
            line=dict(color="blue", dash="dash"),
            legendgroup="TrueDiff",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=diff_ci_upper + diff_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="95% CI (Simple Diff)",
            showlegend=True,
            legendgroup="SimpleDiff",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=diff_estimates,
            mode="lines",
            name="Estimated Simple Diff",
            line=dict(color="blue", width=2),
            legendgroup="SimpleDiff",
        )
    )
    # Percent diff
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_values["impressions_pcnt_diff"]] * len(sample_sizes),
            mode="lines",
            name=f"True Pcnt Diff ({true_values['impressions_pcnt_diff']:.2f}%)",
            line=dict(color="red", dash="dash"),
            legendgroup="TruePcnt",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=pcnt_ci_upper + pcnt_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 99, 71, 0.2)",
            line=dict(color="rgba(255, 99, 71, 0)"),
            name="95% CI (Pcnt Diff)",
            showlegend=True,
            legendgroup="PcntDiff",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=pcnt_estimates,
            mode="lines",
            name="Estimated Pcnt Diff",
            line=dict(color="red", width=2),
            legendgroup="PcntDiff",
        )
    )
    fig.update_layout(
        title="Convergence of Impressions Mean Differences with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Difference (Simple: units, Pcnt: %)",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_diff_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")

# blah/abvelocity/test/stats/test_weighted_metric_estimator.py
# BSD 2-CLAUSE LICENSE
# Copyright 2024, Blah Corporation. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
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
from abvelocity.core.stats.constants import STANDARD_VALUES_NUMER, STANDARD_VALUES_RATIO
from abvelocity.core.stats.metric_estimator import MetricEstimator
from abvelocity.core.stats.weighted_estimator import StrataInfo
from abvelocity.core.stats.weighted_metric_estimator import WeightedMetricEstimator

# Constants
BASE_SAMPLE_SIZE = 10000
SEED = 1719
CI_COVERAGE = 0.95
NUM_BUCKETS = 20
PLOT_WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/weighted-metric-estimator").resolve()


@pytest.fixture(scope="module")
def get_test_data():
    """
    Creates a single DataFrame containing all columns ('y', 'clicks', 'impressions')
    and StrataInfo for 5 diverse countries to test both numerator and ratio metrics.

    Returns: (df, strata_info, true_mean_y, true_sum_ratio_ctr, true_mean_ratio_ctr)
    """
    np.random.seed(SEED)

    countries = ["USA", "Russia", "Iran", "Japan", "Yemen"]
    props = [0.4, 0.2, 0.2, 0.1, 0.1]
    n = BASE_SAMPLE_SIZE
    df_list = []

    # --- Data Definitions for Strata ---
    # 1. Numerator Metric 'y' (Continuous, for Mean Numerator Test)
    y_true_means = [100.0, 20.0, 10.0, 150.0, 80.0]
    y_sd = 10.0

    # 2. Ratio Metric 'clicks/impressions' (CTR)
    clicks_true_rates = [0.15, 0.05, 0.01, 0.20, 0.10]  # Binomial click rate
    impressions_true_means = [15, 10, 5, 20, 8]  # Poisson mean impressions

    counts = []
    for i in range(len(countries)):
        country = countries[i]
        prop = props[i]
        n_strata = int(n * prop)
        counts.append(n_strata)

        # Generate 'y' data
        y_data = np.random.normal(loc=y_true_means[i], scale=y_sd, size=n_strata)

        # Generate 'clicks' and 'impressions' data
        impressions_data = np.random.poisson(impressions_true_means[i], n_strata)
        # Clicks <= Impressions. clip(min=1) avoids errors if impressions_data contains 0s.
        clicks_data = np.random.binomial(impressions_data.clip(min=1), clicks_true_rates[i], n_strata)

        df_list.append(
            pd.DataFrame(
                {
                    "y": y_data,
                    "clicks": clicks_data,
                    "impressions": impressions_data,
                    "country": country,
                }
            )
        )

    df = pd.concat(df_list, ignore_index=True)
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Create StrataInfo
    strata_df = pd.DataFrame({"variant": countries, "pop_count": counts})
    strata_df.index = countries
    strata_info = StrataInfo(df=strata_df, strata_count_col="pop_count")

    # --- Calculate True Weighted Population Metrics ---
    true_mean_y = sum(prop * mean for prop, mean in zip(props, y_true_means))

    # True Sum Ratio: Sum(E[clicks]) / Sum(E[impressions]) (using expected values)
    true_sum_clicks_per_capita = sum(prop * rate * mean_imp for prop, rate, mean_imp in zip(props, clicks_true_rates, impressions_true_means))
    true_sum_impressions_per_capita = sum(prop * mean_imp for prop, mean_imp in zip(props, impressions_true_means))
    true_sum_ratio_ctr = true_sum_clicks_per_capita / true_sum_impressions_per_capita

    # True Mean Ratio: Approximated using the large sample mean ratio
    true_mean_ratio_ctr = df["clicks"].mean() / df["impressions"].mean()

    return df, strata_info, true_mean_y, true_sum_ratio_ctr, true_mean_ratio_ctr


def test_weighted_numerator_mean(get_test_data):
    """
    Tests WeightedMetricEstimator for mean numerator metrics.
    Ensures:
    1. Unweighted output calculation is correct.
    2. Weighted estimate is close to the true population mean.
    """
    df, strata_info, true_population_mean, _, _ = get_test_data

    metric_def = Metric(
        numerator=UMetric(col="y", agg="MAX", fill_na=0, name="Y_Value"),
        numerator_agg="SUM",
    )

    estimator = WeightedMetricEstimator(
        metric=metric_def,
        strata_info=strata_info,
        variant_values=strata_info.df.index.tolist(),
        variant_col="country",
    )
    assert estimator.standard_names == STANDARD_VALUES_NUMER

    result = estimator.estimator_func_with_inferred_param(df)

    simple_mean = df["y"].mean()
    expected_unweighted_sum_and_mean = np.array([df["y"].sum(), simple_mean])

    assert np.allclose(result, expected_unweighted_sum_and_mean, rtol=0.01)


def test_weighted_ratio_ctr(get_test_data):
    """Tests WeightedMetricEstimator for clicks/impressions ratio metrics."""
    df, strata_info, _, _, _ = get_test_data
    metric_def = Metric(
        numerator=UMetric(col="clicks", agg="MAX", fill_na=0, name="Clicks"),
        denominator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="SUM",
        denominator_agg="SUM",
    )
    estimator = WeightedMetricEstimator(
        metric=metric_def,
        strata_info=strata_info,
        variant_values=strata_info.df.index.tolist(),
        variant_col="country",
    )

    assert estimator.standard_names == STANDARD_VALUES_RATIO

    result = estimator.estimator_func_with_inferred_param(df)

    sum_numer = df["clicks"].sum()
    mean_numer = df["clicks"].mean()
    sum_denom = df["impressions"].sum()
    mean_denom = df["impressions"].mean()
    sum_ratio = sum_numer / sum_denom
    mean_ratio = mean_numer / mean_denom
    expected = np.array([sum_numer, mean_numer, sum_denom, mean_denom, sum_ratio, mean_ratio])

    assert np.allclose(result, expected, rtol=1e-3)


# ------------------------------------------------------------------
# PLOTTING TESTS
# ------------------------------------------------------------------
def test_plot_weighted_mean_ratio_convergence(get_test_data):
    """
    Plots the estimated weighted mean_ratio (CTR, Mean/Mean) vs. simple mean_ratio for increasing sample sizes.
    """
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    # Unpack the necessary data (df, strata_info, true_mean_ratio)
    df, strata_info, _, _, true_mean_ratio = get_test_data

    sample_sizes = list(range(50, 2001, 50))

    # Weighted Estimator Results
    weighted_estimates, weighted_ci_lower, weighted_ci_upper = [], [], []

    # Simple Estimator Results
    simple_estimates, simple_ci_lower, simple_ci_upper = [], [], []

    metric = Metric(
        numerator=UMetric(col="clicks", agg="MAX", fill_na=0, name="Clicks"),
        denominator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="SUM",
        denominator_agg="SUM",
    )

    # Initialize Estimators
    weighted_estimator = WeightedMetricEstimator(
        metric=metric,
        strata_info=strata_info,
        variant_values=strata_info.df.index.tolist(),  # Use all 5 countries
        variant_col="country",
    )
    simple_estimator = MetricEstimator(
        metric=metric,
        compute_standard_values=True,
    )

    # Index 5 is "mean_ratio" in STANDARD_VALUES_RATIO
    RATIO_INDEX = 5

    for n in sample_sizes:
        sample_df = df[:n]

        # Weighted Stats
        w_jk_result = weighted_estimator.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        weighted_estimates.append(w_jk_result["estimator_value"][RATIO_INDEX])
        weighted_ci_lower.append(w_jk_result["ci"][RATIO_INDEX][0])
        weighted_ci_upper.append(w_jk_result["ci"][RATIO_INDEX][1])

        # Simple Stats
        s_jk_result = simple_estimator.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        simple_estimates.append(s_jk_result["estimator_value"][RATIO_INDEX])
        simple_ci_lower.append(s_jk_result["ci"][RATIO_INDEX][0])
        simple_ci_upper.append(s_jk_result["ci"][RATIO_INDEX][1])

    fig = go.Figure()

    # True Value
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_mean_ratio] * len(sample_sizes),
            mode="lines",
            name=f"True Mean Ratio ({true_mean_ratio:.4f})",
            line=dict(color="black", dash="dot"),
            legendgroup="True",
        )
    )

    # Weighted CI Ribbon
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=weighted_ci_upper + weighted_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 99, 71, 0.2)",  # Red shade
            line=dict(color="rgba(255, 99, 71, 0)"),
            name="Weighted 95% CI",
            showlegend=True,
            legendgroup="Weighted",
        )
    )

    # Weighted Estimate Line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=weighted_estimates,
            mode="lines",
            name="Weighted mean_ratio Estimate",
            line=dict(color="red", width=2),
            legendgroup="Weighted",
        )
    )

    # Simple CI Ribbon
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=simple_ci_upper + simple_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(0, 128, 0, 0.2)",  # Green shade
            line=dict(color="rgba(0, 128, 0, 0)"),
            name="Simple 95% CI",
            showlegend=True,
            legendgroup="Simple",
        )
    )

    # Simple Estimate Line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=simple_estimates,
            mode="lines",
            name="Simple mean_ratio Estimate",
            line=dict(color="darkgreen", width=2, dash="dash"),
            legendgroup="Simple",
        )
    )

    fig.update_layout(
        title="Convergence: Weighted vs. Simple mean_ratio (CTR) with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Estimated Mean Ratio",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_weighted_mean_ratio_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_plot_weighted_mean_numer_convergence(get_test_data):
    """
    Plots the estimated weighted mean_numer (Y_Value) vs. simple mean_numer for increasing sample sizes.
    Uses the new, diverse strata data.
    """
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    # Unpack the new return values
    df, strata_info, true_population_mean, _, _ = get_test_data

    sample_sizes = list(range(50, 2001, 50))

    # Weighted Estimator Results
    weighted_estimates, weighted_ci_lower, weighted_ci_upper = [], [], []

    # Simple Estimator Results
    simple_estimates, simple_ci_lower, simple_ci_upper = [], [], []

    # Metric definition using 'y'
    metric = Metric(
        numerator=UMetric(col="y", agg="MAX", fill_na=0, name="Y_Value"),
        numerator_agg="SUM",
    )

    # Initialize Estimators
    weighted_estimator = WeightedMetricEstimator(
        metric=metric,
        strata_info=strata_info,
        variant_values=strata_info.df.index.tolist(),  # Correct: Uses all 5 countries from strata_info
        variant_col="country",  # Correct: Uses the 'country' column
    )
    simple_estimator = MetricEstimator(
        metric=metric,
        compute_standard_values=True,
    )

    # Index 1 is "mean_numer" in STANDARD_VALUES_NUMER
    NUMER_INDEX = 1

    for n in sample_sizes:
        sample_df = df[:n]  # Sample first n rows due to randomization

        # Weighted Stats
        w_jk_result = weighted_estimator.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        weighted_estimates.append(w_jk_result["estimator_value"][NUMER_INDEX])
        weighted_ci_lower.append(w_jk_result["ci"][NUMER_INDEX][0])
        weighted_ci_upper.append(w_jk_result["ci"][NUMER_INDEX][1])

        # Simple Stats
        s_jk_result = simple_estimator.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        simple_estimates.append(s_jk_result["estimator_value"][NUMER_INDEX])
        simple_ci_lower.append(s_jk_result["ci"][NUMER_INDEX][0])
        simple_ci_upper.append(s_jk_result["ci"][NUMER_INDEX][1])

    fig = go.Figure()

    # True Value for Weighted Estimator (True Population Mean)
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_population_mean] * len(sample_sizes),
            mode="lines",
            name=f"True Weighted Mean ({true_population_mean:.2f})",
            line=dict(color="black", dash="dot"),
            legendgroup="True_W",
        )
    )

    # Weighted CI Ribbon
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=weighted_ci_upper + weighted_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="Weighted 95% CI",
            showlegend=True,
            legendgroup="Weighted",
        )
    )

    # Weighted Estimate Line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=weighted_estimates,
            mode="lines",
            name="Weighted mean_numer Estimate",
            line=dict(color="blue", width=2),
            legendgroup="Weighted",
        )
    )

    # True Value for Simple Estimator (Full Sample Mean)
    simple_sample_mean_full = df["y"].mean()
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[simple_sample_mean_full] * len(sample_sizes),
            mode="lines",
            name=f"True Sample Mean ({simple_sample_mean_full:.2f})",
            line=dict(color="darkgrey", dash="dashdot"),
            legendgroup="True_S",
        )
    )

    # Simple CI Ribbon
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=simple_ci_upper + simple_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(148, 0, 211, 0.2)",  # Violet shade
            line=dict(color="rgba(148, 0, 211, 0)"),
            name="Simple 95% CI",
            showlegend=True,
            legendgroup="Simple",
        )
    )

    # Simple Estimate Line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=simple_estimates,
            mode="lines",
            name="Simple mean_numer Estimate",
            line=dict(color="darkviolet", width=2, dash="dash"),
            legendgroup="Simple",
        )
    )

    fig.update_layout(
        title="Convergence: Weighted vs. Simple mean_numer (Y_Value) with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Estimated Mean",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_weighted_mean_numer_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")

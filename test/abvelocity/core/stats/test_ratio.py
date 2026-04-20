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
from abvelocity.core.stats.jackknife import calc_jk_stats
from abvelocity.core.stats.ratio import calc_ratio_stats
from plotly.subplots import make_subplots

# --- Constants and Configuration ---
BASE_SAMPLE_SIZE = 10000
POPULATION_SD_X = 5.0
POPULATION_SD_Y = 10.0
POPULATION_MEAN_X = 50.0
POPULATION_MEAN_Y = 100.0
POPULATION_CORR = 0.5  # Correlation between x and y
TOLERANCE = 1e-4
SEED = 1317

# Baseline tolerance for comparisons
ASYMPTOTIC_TOLERANCE_DEFAULT = 0.55

# --- File Saving Path Setup ---
# Define the path relative to the current file (5 parents up to the root)
# and save plots in docs/static/test-results/ratio/
WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/ratio")
os.makedirs(WRITE_PATH, exist_ok=True)


# --- Fixture to Create Base DataFrame ---
@pytest.fixture(scope="module")
def test_df_fixture():
    """Creates a large base DataFrame for testing, using a bivariate normal distribution."""
    np.random.seed(SEED)
    cov_matrix = [
        [POPULATION_SD_Y**2, POPULATION_CORR * POPULATION_SD_Y * POPULATION_SD_X],
        [POPULATION_CORR * POPULATION_SD_Y * POPULATION_SD_X, POPULATION_SD_X**2],
    ]
    data_array = np.random.multivariate_normal(
        mean=[POPULATION_MEAN_Y, POPULATION_MEAN_X],
        cov=cov_matrix,
        size=BASE_SAMPLE_SIZE,
    )
    df = pd.DataFrame({"y": data_array[:, 0], "x": data_array[:, 1]})
    return df


# --- Parametrized Test Functions (Asserting Statistical Properties) ---
@pytest.mark.parametrize(
    "sample_size, num_buckets",
    [
        (1000, 20),
        (1000, 50),
    ],
)
def test_delta_ratio_check(test_df_fixture, sample_size, num_buckets):
    """
    Parametrized Test (Ratio): Compares the delta method variance and CI against the jackknife
    variance and CI for the ratio estimator using grouped jackknife. Uses relaxed tolerance
    for approximation.

    Args:
        test_df_fixture: Fixture providing a DataFrame with test data.
        sample_size: Number of samples to draw from the fixture.
        num_buckets: Number of jackknife buckets.
    """
    np.random.seed(SEED + sample_size + num_buckets)

    test_df = test_df_fixture.sample(n=sample_size, random_state=SEED + 10)

    numer_col = "y"
    denom_col = "x"

    def estimator_func(df: pd.DataFrame):
        return df[numer_col].mean() / df[denom_col].mean()

    jk_result = calc_jk_stats(test_df, estimator_func, num_buckets=num_buckets)
    delta_result = calc_ratio_stats(test_df, numer_col=numer_col, denom_col=denom_col)

    # FIX 1: Extract the jackknife value, handling potential 1D array return
    jk_value = jk_result["estimator_value"]
    if isinstance(jk_value, np.ndarray) and jk_value.ndim > 0:
        jk_value = jk_value[0]

    # FIX 2: Extract the jackknife variance from the 'estimator_varcov' key
    jk_var = jk_result["estimator_varcov"][0, 0]
    jk_ci = jk_result["ci"]

    # ASSERTION: Ratio estimates should be approximately equal (no bias correction in delta)
    assert np.isclose(
        jk_value,
        delta_result["estimator_value"],
        rtol=ASYMPTOTIC_TOLERANCE_DEFAULT,
    )

    # ASSERTION: Variance Comparison (Delta vs. JK)
    assert np.isclose(
        jk_var,
        delta_result["estimator_var"],
        rtol=ASYMPTOTIC_TOLERANCE_DEFAULT,
    )

    # ASSERTION: CI Comparison (Delta vs. JK)
    assert np.allclose(
        jk_ci,
        delta_result["ci"],
        rtol=ASYMPTOTIC_TOLERANCE_DEFAULT,
    )

    # ASSERTION: Variance must be positive
    assert delta_result["estimator_var"] > 0.0


# --- Plotting Tests (Convergence Visualization) ---
def test_study_ratio_estimator_plot(test_df_fixture):
    """
    Explores how Jackknife Variance (Var_JK) converges to Delta Method Variance (Var_Delta)
    for the Ratio estimator as the number of buckets (b) increases.
    Generates a Plotly plot and saves it as HTML with a matching filename.

    Args:
        test_df_fixture: Fixture providing a DataFrame with test data.
    """
    np.random.seed(SEED + 31)

    # Define the ranges for the analysis
    sample_sizes = [100, 500, 1000]
    bucket_range = range(10, 101, 5)

    all_results = []

    numer_col = "y"
    denom_col = "x"

    def ratio_estimator(df: pd.DataFrame):
        return df[numer_col].mean() / df[denom_col].mean()

    for current_sample_size in sample_sizes:
        max_b = min(current_sample_size, max(bucket_range))
        buckets_to_test = [b for b in bucket_range if b <= max_b]

        if not buckets_to_test:
            continue

        test_df = test_df_fixture.sample(n=current_sample_size, random_state=SEED + current_sample_size)

        # Calculate the delta method variance once for the sample size N
        delta_result = calc_ratio_stats(test_df, numer_col=numer_col, denom_col=denom_col)
        delta_variance = delta_result["estimator_var"]

        for b in buckets_to_test:
            try:
                jk_result = calc_jk_stats(test_df, ratio_estimator, num_buckets=b)

                jk_variance = jk_result["estimator_varcov"][0, 0]

                all_results.append(
                    {
                        "N": current_sample_size,
                        "b": b,
                        "Var_JK": jk_variance,
                        "Var_Delta": delta_variance,
                        "Variance_Type": "Jackknife (Var_JK)",
                        "Relative_Difference": abs(jk_variance - delta_variance) / delta_variance,
                    }
                )
                all_results.append(
                    {
                        "N": current_sample_size,
                        "b": b,
                        "Var_JK": delta_variance,
                        "Var_Delta": delta_variance,
                        "Variance_Type": "Delta Method (Var_Delta)",
                        "Relative_Difference": 0.0,
                    }
                )

            except ValueError:
                continue

    results_df = pd.DataFrame(all_results)

    # Plotting
    fig = make_subplots(
        rows=len(sample_sizes),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Sample Size N={size}" for size in sample_sizes],
        vertical_spacing=0.08,
    )

    colors = {"Jackknife (Var_JK)": "#1f77b4", "Delta Method (Var_Delta)": "#ff7f0e"}

    for i, size in enumerate(sample_sizes):
        n_df = results_df[results_df["N"] == size]
        row = i + 1

        # Plot Delta Method Variance (Var_Delta) - Constant line
        delta_var = n_df[n_df["Variance_Type"] == "Delta Method (Var_Delta)"]["Var_Delta"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=n_df[n_df["Variance_Type"] == "Delta Method (Var_Delta)"]["b"],
                y=[delta_var] * len(n_df[n_df["Variance_Type"] == "Delta Method (Var_Delta)"]),
                mode="lines",
                name="Delta Method Variance (Var_Delta)",
                line=dict(color=colors["Delta Method (Var_Delta)"], dash="dash"),
                showlegend=(i == 0),
            ),
            row=row,
            col=1,
        )

        # Plot Jackknife Variance (Var_JK)
        jk_df = n_df[n_df["Variance_Type"] == "Jackknife (Var_JK)"]
        fig.add_trace(
            go.Scatter(
                x=jk_df["b"],
                y=jk_df["Var_JK"],
                mode="lines+markers",
                name="Jackknife Variance (Var_JK)",
                line=dict(color=colors["Jackknife (Var_JK)"]),
                showlegend=(i == 0),
            ),
            row=row,
            col=1,
        )

        fig.update_yaxes(title_text="Variance Estimate", row=row, col=1)
        if i == len(sample_sizes) - 1:
            fig.update_xaxes(title_text="Number of Buckets (b)", row=row, col=1)

    fig.update_layout(
        height=300 * len(sample_sizes),
        title_text="Grouped Jackknife Variance vs. Bucket Count (Estimator: Ratio) Compared to Delta Method",
        title_font_size=20,
    )

    # File name matches test function name
    file_name = WRITE_PATH.joinpath("test_study_ratio_estimator_plot.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")

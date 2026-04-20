# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

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

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from abvelocity.core.stats.normal_ci import calc_standard_normal_ci
from abvelocity.core.stats.student_ci import calc_student_ci

# --- Constants and Configuration ---
SEED = 1317

# --- File Saving Path Setup ---
# Define the path relative to the current file (5 parents up to the root)
# and save plots in docs/static/test-results/student_ci/
WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/student_ci")
os.makedirs(WRITE_PATH, exist_ok=True)


# Test for known mean and standard deviation with specific values
def test_calc_student_ci():
    # Example 1: Test case where mean=0 and se=1, dof=9
    mean = 0
    se = 1
    dof = 9
    ci_coverage = 0.95
    result = calc_student_ci(mean, se, dof, ci_coverage)

    # Expected t-value, p-value, and confidence interval
    expected_t_value = 0.0
    expected_p_value = 1.0
    expected_ci = np.array([-2.26, 2.26])  # Approximate 95% CI for t with dof=9

    assert np.isclose(result["t_value"], expected_t_value, atol=1e-2)
    assert np.isclose(result["p_value"], expected_p_value, atol=1e-2)
    assert np.allclose(result["ci"], expected_ci, atol=1e-2)

    # Example 2: Test case with non-zero mean, dof=9
    mean = 2
    se = 1
    dof = 9
    ci_coverage = 0.95
    result = calc_student_ci(mean, se, dof, ci_coverage)

    expected_t_value = 2.0
    expected_p_value = 0.077
    expected_ci = np.array([-0.26, 4.26])  # Approximate 95% CI for given parameters

    assert np.isclose(result["t_value"], expected_t_value, atol=1e-2)
    assert np.isclose(result["p_value"], expected_p_value, atol=1e-2)
    assert np.allclose(result["ci"], expected_ci, atol=1e-2)

    # Example 3: Test case with larger standard deviation, dof=19, ci=0.99
    mean = 5
    se = 2
    dof = 19
    ci_coverage = 0.99
    result = calc_student_ci(mean, se, dof, ci_coverage)

    expected_t_value = 2.5
    expected_p_value = 0.022
    expected_ci = np.array([-0.72, 10.72])  # Approximate 99% CI for given parameters

    assert np.isclose(result["t_value"], expected_t_value, atol=1e-2)
    assert np.isclose(result["p_value"], expected_p_value, atol=1e-2)
    assert np.allclose(result["ci"], expected_ci, atol=1e-2)


# Test for edge cases
def test_calc_student_ci_edge_cases():
    # Edge Case: Negative standard deviation (se < 0)
    mean = 5
    se = -1
    dof = 9
    ci_coverage = 0.95
    with pytest.raises(ValueError):
        calc_student_ci(mean, se, dof, ci_coverage)

    # Edge Case: Degrees of freedom not positive (dof <= 0)
    mean = 5
    se = 1
    dof = 0
    ci_coverage = 0.95
    with pytest.raises(ValueError):
        calc_student_ci(mean, se, dof, ci_coverage)

    # Edge Case: Confidence interval coverage outside [0, 1]
    mean = 5
    se = 1
    dof = 9
    ci_coverage = 1.5
    with pytest.raises(ValueError):
        calc_student_ci(mean, se, dof, ci_coverage)


# --------------------------------------------------------------------------
# Plotting Tests (Comparison Visualization)
# --------------------------------------------------------------------------


def test_compare_ci_lengths_plot():
    """
    Compares the confidence interval lengths from the normal approximation and Student's t-distribution
    for three different degrees of freedom settings as the sample size increases.
    Assumes a population standard deviation of 1 for calculating the standard error (SE = 1 / sqrt(n)).
    Generates a Plotly plot and saves it as HTML with a matching filename.
    """
    np.random.seed(SEED + 33)

    # Define the ranges for the analysis
    sample_sizes = np.arange(11, 101, 1)  # Start from 11 to ensure positive df in all settings
    ci_coverage = 0.95
    pop_sd = 1.0  # Assumed population standard deviation

    all_results = []

    for n in sample_sizes:
        se = pop_sd / np.sqrt(n)

        # Normal CI length
        normal_result = calc_standard_normal_ci(mean=0, se=se, ci_coverage=ci_coverage)
        normal_length = normal_result["ci"][1] - normal_result["ci"][0]

        all_results.append(
            {
                "n": n,
                "CI_Length": normal_length,
                "Type": "Normal Approximation",
            }
        )

        # t CI with df = n-1 (e.g., one-sample mean)
        dof1 = n - 1
        t_result1 = calc_student_ci(mean=0, se=se, dof=dof1, ci_coverage=ci_coverage)
        t_length1 = t_result1["ci"][1] - t_result1["ci"][0]

        all_results.append(
            {
                "n": n,
                "CI_Length": t_length1,
                "Type": "t (df = n-1, one-sample)",
            }
        )

        # t CI with df = n-2 (e.g., two independent samples, total n, equal size n/2 each)
        dof2 = n - 2
        t_result2 = calc_student_ci(mean=0, se=se, dof=dof2, ci_coverage=ci_coverage)
        t_length2 = t_result2["ci"][1] - t_result2["ci"][0]

        all_results.append(
            {
                "n": n,
                "CI_Length": t_length2,
                "Type": "t (df = n-2, two-samples)",
            }
        )

        # t CI with df = floor(n/2) - 1 (e.g., paired samples, n total observations, n/2 pairs)
        dof3 = (n // 2) - 1
        if dof3 > 0:
            t_result3 = calc_student_ci(mean=0, se=se, dof=dof3, ci_coverage=ci_coverage)
            t_length3 = t_result3["ci"][1] - t_result3["ci"][0]

            all_results.append(
                {
                    "n": n,
                    "CI_Length": t_length3,
                    "Type": "t (df = n/2 -1, paired)",
                }
            )

    results_df = pd.DataFrame(all_results)

    # Plotting
    fig = go.Figure()

    types = results_df["Type"].unique()
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"]  # Colors for each type

    for i, t in enumerate(types):
        df_subset = results_df[results_df["Type"] == t]
        fig.add_trace(
            go.Scatter(
                x=df_subset["n"],
                y=df_subset["CI_Length"],
                mode="lines",
                name=t,
                line=dict(color=colors[i]),
            )
        )

    fig.update_layout(
        height=600,
        title_text="CI Length Comparison: Normal vs Student's t for Different df Settings (95% Coverage)",
        title_font_size=20,
        xaxis_title="Sample Size (n)",
        yaxis_title="CI Length",
    )

    # File name matches test function name
    file_name = WRITE_PATH.joinpath("test_compare_ci_lengths_plot.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")

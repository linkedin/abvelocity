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
from abvelocity.core.stats.student_ci import calc_student_ci
from plotly.subplots import make_subplots
from scipy import stats

# --- Constants and Configuration ---
BASE_SAMPLE_SIZE = 10000
POPULATION_SD = 10.0
TOLERANCE = 1e-4
SEED = 1317
CI_COVERAGE = 0.95

# Baseline tolerance for stable estimators (Mean, Variance) and stable JK methods.
ASYMPTOTIC_TOLERANCE_DEFAULT = 0.55
# For testing covariance against a theoretical value (which is tighter than zero-check)
COVARIANCE_RELATIVE_TOLERANCE = 0.60

# --- File Saving Path Setup ---
WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/jackknife").resolve()
os.makedirs(WRITE_PATH, exist_ok=True)


# --- Fixtures to Create Base DataFrames ---


@pytest.fixture(scope="module")
def test_df_fixture():
    """Creates a large base DataFrame for testing, using INDEPENDENT normal distributions."""
    np.random.seed(SEED)
    data_array = np.random.normal(loc=100, scale=POPULATION_SD, size=BASE_SAMPLE_SIZE)
    # y2 is independent of y1
    data_array_2 = np.random.normal(loc=50, scale=POPULATION_SD / 2, size=BASE_SAMPLE_SIZE)
    df = pd.DataFrame({"y1": data_array, "y2": data_array_2})
    return df


@pytest.fixture(scope="module")
def correlated_df_fixture():
    """Creates a large base DataFrame for testing with a NON-ZERO covariance."""
    np.random.seed(SEED + 1000)

    # Population Parameters
    mean = [100, 50]
    # Define a covariance matrix with non-zero covariance (e.g., 25)
    # Var(y1)=100, Var(y2)=25, Cov(y1, y2)=25 -> Correlation = 25 / sqrt(100*25) = 0.5
    cov_matrix = [[100, 25], [25, 25]]

    data = np.random.multivariate_normal(mean=mean, cov=cov_matrix, size=BASE_SAMPLE_SIZE)
    df = pd.DataFrame(data, columns=["y1", "y2"])
    return df


# --------------------------------------------------------------------------
# Parametrized Test Functions (Asserting Statistical Properties)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample_size, num_buckets, ci_coverage",
    [
        (1000, 20, 0.95),
        (1000, 50, 0.95),
    ],
)
def test_jackknife_mean_grouped_check(test_df_fixture, sample_size, num_buckets, ci_coverage):
    """
    Parametrized Test (Mean): Checks Var_JK against the theoretical variance (Var_T)
    and CI coverage for the mean using a large sample size (1000) and bucket counts (20, 50).
    Uses relaxed tolerance due to grouped JK bias.
    """
    np.random.seed(SEED + sample_size + num_buckets)

    # Use 'y1' for the univariate tests
    test_df = test_df_fixture[["y1"]].rename(columns={"y1": "y"}).sample(n=sample_size, random_state=SEED + 10)

    def estimator_func(df: pd.DataFrame):
        # Return a 1D array of length 1
        return np.array([df["y"].mean()])

    theta_hat = estimator_func(test_df)[0]  # Extract the scalar value for bias check

    result = calc_jk_stats(test_df, estimator_func, num_buckets=num_buckets, ci_coverage=ci_coverage)

    # --- UPDATED RETRIEVAL FOR K-DIMENSIONAL OUTPUT ---
    # The estimate is now a 1-element array, the CI is a 1x2 array, and the variance is a 1x1 matrix.
    jk_mean_estimator = result["estimator_value"][0]
    jk_variance_estimator = result["estimator_varcov"][0, 0]
    jk_ci = result["ci"][0, :]

    jk_bias_estimate = theta_hat - jk_mean_estimator

    # Calculate Theoretical Variance (Var_T = Sample_Var / sample_size)
    sample_variance_unbiased = test_df["y"].var(ddof=1)
    theoretical_variance = sample_variance_unbiased / sample_size

    # Calculate theoretical CI using Student's t-distribution
    theoretical_ci = calc_student_ci(
        mean=theta_hat,
        se=np.sqrt(theoretical_variance),
        dof=num_buckets - 1,
        ci_coverage=ci_coverage,
    )["ci"]

    # Check that JK bias for the mean is close to zero
    assert np.isclose(jk_bias_estimate, 0.0, atol=1e-3)

    # ASSERTION: Variance Comparison (Var_JK vs. Var_T for the mean)
    assert np.isclose(jk_variance_estimator, theoretical_variance, rtol=ASYMPTOTIC_TOLERANCE_DEFAULT)

    # ASSERTION: CI Comparison
    assert np.allclose(jk_ci, theoretical_ci, rtol=ASYMPTOTIC_TOLERANCE_DEFAULT)


@pytest.mark.parametrize(
    "sample_size, num_buckets, ci_coverage",
    [
        (1000, 20, 0.95),
        (1000, 50, 0.95),
    ],
)
def test_jackknife_median_grouped_check(test_df_fixture, sample_size, num_buckets, ci_coverage):
    """
    Parametrized Test (Median): Checks Var_JK against the theoretical asymptotic variance (Var_T)
    and CI coverage for the median. Focuses on the Grouped Jackknife cases (b < N).
    """
    np.random.seed(SEED + sample_size + num_buckets + 50)

    # Use 'y1' for the univariate tests
    test_df = test_df_fixture[["y1"]].rename(columns={"y1": "y"}).sample(n=sample_size, random_state=SEED + 60)

    # Set a higher tolerance specifically for the known unstable grouped median case (b=50)
    current_tolerance = 0.75 if num_buckets == 50 else ASYMPTOTIC_TOLERANCE_DEFAULT

    def estimator_func(df: pd.DataFrame):
        # Return a 1D array of length 1
        return np.array([df["y"].median()])

    result = calc_jk_stats(test_df, estimator_func, num_buckets=num_buckets, ci_coverage=ci_coverage)

    # --- UPDATED RETRIEVAL FOR K-DIMENSIONAL OUTPUT ---
    # The estimate is now a 1-element array, the CI is a 1x2 array, and the variance is a 1x1 matrix.
    jk_mean_estimator = result["estimator_value"][0]
    jk_variance_estimator = result["estimator_varcov"][0, 0]
    jk_ci = result["ci"][0, :]

    # Calculate Theoretical Variance based on Var(median) approx 1/(4n * f(theta)^2)
    sample_variance_unbiased = test_df["y"].var(ddof=1)
    sample_stdev = np.sqrt(sample_variance_unbiased)

    # Estimate the density f(theta) at the median (theta) for a Normal distribution
    density_at_median = 1.0 / (sample_stdev * np.sqrt(2.0 * np.pi))

    # Apply the general asymptotic variance formula
    theoretical_variance = 1.0 / (4.0 * sample_size * (density_at_median**2))

    # Calculate theoretical CI using Student's t-distribution
    theoretical_ci = calc_student_ci(
        mean=jk_mean_estimator,
        se=np.sqrt(theoretical_variance),
        dof=num_buckets - 1,
        ci_coverage=ci_coverage,
    )["ci"]

    # ASSERTION: Variance Comparison (Var_JK vs. Var_T for the median)
    assert np.isclose(jk_variance_estimator, theoretical_variance, rtol=current_tolerance)

    # ASSERTION: Variance must be positive
    assert jk_variance_estimator > 0.0

    # ASSERTION: CI Comparison
    assert np.allclose(jk_ci, theoretical_ci, rtol=current_tolerance)


@pytest.mark.parametrize(
    "sample_size, num_buckets, ci_coverage",
    [
        (1000, 20, 0.95),
        (1000, 50, 0.95),
    ],
)
def test_jackknife_variance_grouped_check(test_df_fixture, sample_size, num_buckets, ci_coverage):
    """
    Parametrized Test (Variance): Checks Var_JK against theoretical variance
    and CI coverage for the Sample Variance estimator. Uses relaxed tolerance due to grouped JK bias.
    """
    np.random.seed(SEED + sample_size + num_buckets + 70)

    # Use 'y1' for the univariate tests
    test_df = test_df_fixture[["y1"]].rename(columns={"y1": "y"}).sample(n=sample_size, random_state=SEED + 20)

    # Estimator function: Sample Variance (unbiased)
    def estimator_func(df: pd.DataFrame):
        # Return a 1D array of length 1
        return np.array([df["y"].var(ddof=1)])

    theta_hat = estimator_func(test_df)[0]

    result = calc_jk_stats(test_df, estimator_func, num_buckets=num_buckets, ci_coverage=ci_coverage)

    # --- UPDATED RETRIEVAL FOR K-DIMENSIONAL OUTPUT ---
    # The estimate is now a 1-element array, the CI is a 1x2 array, and the variance is a 1x1 matrix.
    jk_mean_estimator = result["estimator_value"][0]
    jk_variance_estimator = result["estimator_varcov"][0, 0]
    jk_ci = result["ci"][0, :]

    # Calculate Theoretical Variance of the Sample Variance (Var_T(s^2))
    # Formula assumes Normality: Var_T(s^2) approx (2 * s^4) / (n - 1)
    theoretical_variance_of_sample_variance = (2 * (theta_hat**2)) / (sample_size - 1)

    # Calculate theoretical CI using Student's t-distribution
    theoretical_ci = calc_student_ci(
        mean=jk_mean_estimator,
        se=np.sqrt(theoretical_variance_of_sample_variance),
        dof=num_buckets - 1,
        ci_coverage=ci_coverage,
    )["ci"]

    # ASSERTION: Variance Comparison (Var_JK vs. Var_T(s^2))
    assert np.isclose(
        jk_variance_estimator,
        theoretical_variance_of_sample_variance,
        rtol=ASYMPTOTIC_TOLERANCE_DEFAULT,
    )

    # ASSERTION: Variance must be positive
    assert jk_variance_estimator > 0.0

    # ASSERTION: CI Comparison
    assert np.allclose(jk_ci, theoretical_ci, rtol=ASYMPTOTIC_TOLERANCE_DEFAULT)


# --------------------------------------------------------------------------
# Plotting Tests (Convergence Visualization)
# --------------------------------------------------------------------------


def test_study_sample_mean_estimator_plot(test_df_fixture):
    """
    Explores how Jackknife Variance (Var_JK) converges to Theoretical Variance (Var_T)
    for the Sample MEAN estimator as the number of buckets (b) increases.
    Generates a Plotly plot and saves it as HTML with a matching filename.
    """
    np.random.seed(SEED + 31)

    # Define the ranges for the analysis
    sample_sizes = [100, 500, 1000]
    bucket_range = range(10, 101, 5)
    ci_coverage = 0.95

    all_results = []

    def sample_mean_estimator(df: pd.DataFrame):
        # Return a 1D array of length 1
        return np.array([df["y1"].mean()])

    for current_sample_size in sample_sizes:
        max_b = min(current_sample_size, max(bucket_range))
        buckets_to_test = [b for b in bucket_range if b <= max_b]

        if not buckets_to_test:
            continue

        test_df = test_df_fixture.sample(n=current_sample_size, random_state=SEED + current_sample_size)

        # Calculate the theoretical variance once for the sample size N
        sample_variance_unbiased = test_df["y1"].var(ddof=1)
        theoretical_variance = sample_variance_unbiased / current_sample_size

        for b in buckets_to_test:
            try:
                result = calc_jk_stats(test_df, sample_mean_estimator, num_buckets=b, ci_coverage=ci_coverage)

                # --- UPDATED RETRIEVAL ---
                var_jk = result["estimator_varcov"][0, 0]

                all_results.append(
                    {
                        "N": current_sample_size,
                        "b": b,
                        "Var_JK": var_jk,
                        "Var_T": theoretical_variance,
                        "Variance_Type": "Jackknife (Var_JK)",
                        "Relative_Difference": abs(var_jk - theoretical_variance) / theoretical_variance,
                    }
                )
                all_results.append(
                    {
                        "N": current_sample_size,
                        "b": b,
                        "Var_JK": theoretical_variance,
                        "Var_T": theoretical_variance,
                        "Variance_Type": "Theoretical (Var_T)",
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

    colors = {"Jackknife (Var_JK)": "#1f77b4", "Theoretical (Var_T)": "#ff7f0e"}

    for i, size in enumerate(sample_sizes):
        n_df = results_df[results_df["N"] == size]
        row = i + 1

        # Plot Theoretical Variance (Var_T) - Constant line
        theoretical_var = n_df[n_df["Variance_Type"] == "Theoretical (Var_T)"]["Var_T"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=n_df[n_df["Variance_Type"] == "Theoretical (Var_T)"]["b"],
                y=[theoretical_var] * len(n_df[n_df["Variance_Type"] == "Theoretical (Var_T)"]),
                mode="lines",
                name="Theoretical Variance (Var_T)",
                line=dict(color=colors["Theoretical (Var_T)"], dash="dash"),
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
        title_text="Grouped Jackknife Variance vs. Bucket Count (Estimator: Sample Mean)",
        title_font_size=20,
    )

    # File name matches test function name
    file_name = WRITE_PATH.joinpath("test_study_sample_mean_estimator_plot.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_study_sample_median_estimator_plot(test_df_fixture):
    """
    Explores how Jackknife Variance (Var_JK) converges to Theoretical Variance (Var_T)
    for the Sample MEDIAN estimator as 'b' increases.
    Generates a Plotly plot and saves it as HTML with a matching filename.
    """
    np.random.seed(SEED + 32)

    # Define the ranges for the analysis
    sample_sizes = [100, 500, 1000]
    bucket_range = range(10, 101, 5)
    ci_coverage = 0.95

    all_results = []

    def sample_median_estimator(df: pd.DataFrame):
        # Return a 1D array of length 1
        return np.array([df["y1"].median()])

    for current_sample_size in sample_sizes:
        max_b = min(current_sample_size, max(bucket_range))
        buckets_to_test = [b for b in bucket_range if b <= max_b]

        if not buckets_to_test:
            continue

        test_df = test_df_fixture.sample(n=current_sample_size, random_state=SEED + current_sample_size + 1)

        # Calculate Theoretical Variance based on Var(median) approx 1/(4n * f(theta)^2)
        sample_variance_unbiased = test_df["y1"].var(ddof=1)
        sample_stdev = np.sqrt(sample_variance_unbiased)
        density_at_median = 1.0 / (sample_stdev * np.sqrt(2.0 * np.pi))
        theoretical_variance = 1.0 / (4.0 * current_sample_size * (density_at_median**2))

        for b in buckets_to_test:
            try:
                result = calc_jk_stats(test_df, sample_median_estimator, num_buckets=b, ci_coverage=ci_coverage)

                # --- UPDATED RETRIEVAL ---
                var_jk = result["estimator_varcov"][0, 0]

                all_results.append(
                    {
                        "N": current_sample_size,
                        "b": b,
                        "Var_JK": var_jk,
                        "Var_T": theoretical_variance,
                        "Variance_Type": "Jackknife (Var_JK)",
                        "Relative_Difference": abs(var_jk - theoretical_variance) / theoretical_variance,
                    }
                )
                all_results.append(
                    {
                        "N": current_sample_size,
                        "b": b,
                        "Var_JK": theoretical_variance,
                        "Var_T": theoretical_variance,
                        "Variance_Type": "Theoretical (Var_T)",
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

    colors = {"Jackknife (Var_JK)": "#1f77b4", "Theoretical (Var_T)": "#ff7f0e"}

    for i, size in enumerate(sample_sizes):
        n_df = results_df[results_df["N"] == size]
        row = i + 1

        # Plot Theoretical Variance (Var_T) - Constant line
        theoretical_var = n_df[n_df["Variance_Type"] == "Theoretical (Var_T)"]["Var_T"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=n_df[n_df["Variance_Type"] == "Theoretical (Var_T)"]["b"],
                y=[theoretical_var] * len(n_df[n_df["Variance_Type"] == "Theoretical (Var_T)"]),
                mode="lines",
                name="Theoretical Variance (Var_T)",
                line=dict(color=colors["Theoretical (Var_T)"], dash="dash"),
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
        title_text="Grouped Jackknife Variance vs. Bucket Count (Estimator: Sample Median)",
        title_font_size=20,
    )

    # File name matches test function name
    file_name = WRITE_PATH.joinpath("test_study_sample_median_estimator_plot.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_study_sample_variance_estimator_plot(test_df_fixture):
    """
    Explores how Jackknife Variance (Var_JK) converges to Theoretical Variance (Var_T)
    for the Sample Variance estimator as 'b' increases.
    Generates a Plotly plot and saves it as HTML with a matching filename.
    """
    np.random.seed(SEED + 30)

    # Define the ranges for the analysis
    sample_sizes = [100, 500, 1000]
    bucket_range = range(10, 101, 5)
    ci_coverage = 0.95

    all_results = []

    def sample_variance_estimator(df: pd.DataFrame):
        # Return a 1D array of length 1
        return np.array([df["y1"].var(ddof=1)])

    for current_sample_size in sample_sizes:
        max_b = min(current_sample_size, max(bucket_range))
        buckets_to_test = [b for b in bucket_range if b <= max_b]

        if not buckets_to_test:
            continue

        test_df = test_df_fixture.sample(n=current_sample_size, random_state=SEED + current_sample_size)
        theta_hat = sample_variance_estimator(test_df)[0]

        # Calculate the theoretical variance once for the sample size N
        theoretical_variance = (2 * (theta_hat**2)) / (current_sample_size - 1)

        for b in buckets_to_test:
            try:
                result = calc_jk_stats(test_df, sample_variance_estimator, num_buckets=b, ci_coverage=ci_coverage)

                # --- UPDATED RETRIEVAL ---
                var_jk = result["estimator_varcov"][0, 0]

                all_results.append(
                    {
                        "N": current_sample_size,
                        "b": b,
                        "Var_JK": var_jk,
                        "Var_T": theoretical_variance,
                        "Variance_Type": "Jackknife (Var_JK)",
                        "Relative_Difference": abs(var_jk - theoretical_variance) / theoretical_variance,
                    }
                )
                all_results.append(
                    {
                        "N": current_sample_size,
                        "b": b,
                        "Var_JK": theoretical_variance,
                        "Var_T": theoretical_variance,
                        "Variance_Type": "Theoretical (Var_T)",
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

    colors = {"Jackknife (Var_JK)": "#1f77b4", "Theoretical (Var_T)": "#ff7f0e"}

    for i, size in enumerate(sample_sizes):
        n_df = results_df[results_df["N"] == size]
        row = i + 1

        # Plot Theoretical Variance (Var_T) - Constant line
        theoretical_var = n_df[n_df["Variance_Type"] == "Theoretical (Var_T)"]["Var_T"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=n_df[n_df["Variance_Type"] == "Theoretical (Var_T)"]["b"],
                y=[theoretical_var] * len(n_df[n_df["Variance_Type"] == "Theoretical (Var_T)"]),
                mode="lines",
                name="Theoretical Variance (Var_T)",
                line=dict(color=colors["Theoretical (Var_T)"], dash="dash"),
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
        title_text="Grouped Jackknife Variance vs. Bucket Count (Estimator: Sample Variance)",
        title_font_size=20,
    )

    # File name matches test function name
    file_name = WRITE_PATH.joinpath("test_study_sample_variance_estimator_plot.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


# --------------------------------------------------------------------------
# K-Dimensional Tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample_size, num_buckets, ci_coverage",
    [
        (1000, 20, 0.95),
    ],
)
def test_jackknife_multivariate_check(test_df_fixture, sample_size, num_buckets, ci_coverage):
    """
    Test case for k-dimensional estimator (k=2) on INDEPENDENT data.
    Estimates a vector of [Mean(y1), Variance(y2)].
    Checks dimensions and asserts the off-diagonal covariance is close to zero.
    """
    np.random.seed(SEED + 99)

    test_df = test_df_fixture.sample(n=sample_size, random_state=SEED + 100)

    # k=2 Estimator function: [Mean(y1), Variance(y2)]
    def estimator_func(df: pd.DataFrame):
        mean_y1 = df["y1"].mean()
        var_y2 = df["y2"].var(ddof=1)
        # Return a 1D array of length k=2
        return np.array([mean_y1, var_y2])

    theta_hat = estimator_func(test_df)

    result = calc_jk_stats(test_df, estimator_func, num_buckets=num_buckets, ci_coverage=ci_coverage)

    # 1. Check Output Dimensions
    assert result["estimator_value"].shape == (2,)
    assert result["estimator_varcov"].shape == (2, 2)
    assert result["ci"].shape == (2, 2)
    assert result["p_values"].shape == (2,)

    jk_varcov = result["estimator_varcov"]
    jk_ci = result["ci"]

    # 2. Component 1: Mean(y1) checks (asymptotic variance Var_T1)
    sample_variance_y1 = test_df["y1"].var(ddof=1)
    theoretical_variance_y1 = sample_variance_y1 / sample_size

    # ASSERTION: Variance Comparison (Var_JK[0,0] vs. Var_T1)
    assert np.isclose(jk_varcov[0, 0], theoretical_variance_y1, rtol=ASYMPTOTIC_TOLERANCE_DEFAULT)

    # 3. Component 2: Variance(y2) checks (asymptotic variance Var_T2)
    sample_variance_y2 = theta_hat[1]  # Unbiased variance estimate s^2_y2
    # Formula assumes Normality: Var_T(s^2) approx (2 * s^4) / (n - 1)
    theoretical_variance_y2 = (2 * (sample_variance_y2**2)) / (sample_size - 1)

    # ASSERTION: Variance Comparison (Var_JK[1,1] vs. Var_T2)
    assert np.isclose(jk_varcov[1, 1], theoretical_variance_y2, rtol=ASYMPTOTIC_TOLERANCE_DEFAULT)

    # 4. Covariance Check (y1 and y2 are independent by construction)
    # The absolute tolerance for covariance is relaxed to 0.1 because the
    # covariance of the sample mean and sample variance is only asymptotically zero,
    # and the grouped jackknife may not be tight enough at N=1000 and B=20.
    COVARIANCE_TOLERANCE_ZERO_CHECK = 0.1

    # ASSERTION: Covariance is close to zero (y1 and y2 are independent)
    assert np.isclose(jk_varcov[0, 1], 0.0, atol=COVARIANCE_TOLERANCE_ZERO_CHECK)
    assert np.isclose(jk_varcov[1, 0], 0.0, atol=COVARIANCE_TOLERANCE_ZERO_CHECK)

    # 5. CI Sanity Check (must be an array of correct size)
    assert np.all(jk_ci[:, 1] > jk_ci[:, 0])  # Upper bound > Lower bound


@pytest.mark.parametrize(
    "sample_size, num_buckets, ci_coverage",
    [
        # Increased num_buckets from 20 to 50 for N=1000 to improve the
        # stability and reduce the bias of the Grouped Jackknife variance estimate.
        (1000, 50, 0.95),
    ],
)
def test_jackknife_multivariate_mean_cov_check(correlated_df_fixture, sample_size, num_buckets, ci_coverage):
    """
    Test case for k-dimensional estimator (k=2) on CORRELATED data.
    Estimates a vector of [Mean(y1), Mean(y2)].
    Checks:
    1. Diagonal elements (Variances of Means) against theoretical Var(mean) = Var(y)/N.
    2. Off-diagonal element (Covariance of Means) against theoretical Cov(mean) = Cov(y1, y2)/N.
    """
    np.random.seed(SEED + 200)

    test_df = correlated_df_fixture.sample(n=sample_size, random_state=SEED + 201)

    # k=2 Estimator function: [Mean(y1), Mean(y2)]
    def estimator_func(df: pd.DataFrame):
        mean_y1 = df["y1"].mean()
        mean_y2 = df["y2"].mean()
        # Return a 1D array of length k=2
        return np.array([mean_y1, mean_y2])

    result = calc_jk_stats(test_df, estimator_func, num_buckets=num_buckets, ci_coverage=ci_coverage)

    # 1. Check Output Dimensions
    assert result["estimator_value"].shape == (2,)
    assert result["estimator_varcov"].shape == (2, 2)
    assert result["ci"].shape == (2, 2)
    assert result["p_values"].shape == (2,)

    jk_varcov = result["estimator_varcov"]

    # Calculate Sample Variance-Covariance Matrix (s^2_i and s_ij)
    sample_varcov_matrix = test_df[["y1", "y2"]].cov(ddof=1).values

    # Theoretical Covariance Matrix of Sample Means (Var_T[i,j] = s_ij / N)
    theoretical_varcov_matrix = sample_varcov_matrix / sample_size

    # --- 2. Check Diagonal Elements (Variances) ---
    # Var(Mean(y1))
    assert np.isclose(jk_varcov[0, 0], theoretical_varcov_matrix[0, 0], rtol=ASYMPTOTIC_TOLERANCE_DEFAULT)
    # Var(Mean(y2))
    assert np.isclose(jk_varcov[1, 1], theoretical_varcov_matrix[1, 1], rtol=ASYMPTOTIC_TOLERANCE_DEFAULT)

    # --- 3. Check Off-Diagonal Elements (Covariances) ---
    # Cov(Mean(y1), Mean(y2))
    # The JK estimate should be close to the theoretical covariance of the means.

    # ASSERTION: Covariance Comparison (Cov_JK[0,1] vs. Cov_T[0,1])
    # The COVARIANCE_RELATIVE_TOLERANCE (0.60) is used because the off-diagonal
    # term is often the most sensitive to grouped jackknife bias.
    assert np.isclose(jk_varcov[0, 1], theoretical_varcov_matrix[0, 1], rtol=COVARIANCE_RELATIVE_TOLERANCE)
    # Check symmetry
    assert np.isclose(jk_varcov[0, 1], jk_varcov[1, 0])

    # 4. CI Sanity Check
    jk_ci = result["ci"]
    assert np.all(jk_ci[:, 1] > jk_ci[:, 0])  # Upper bound > Lower bound


@pytest.mark.parametrize(
    "sample_size, num_buckets, ci_coverage",
    [
        (1000, 20, 0.95),
    ],
)
def test_jackknife_p_value_significant(test_df_fixture, sample_size, num_buckets, ci_coverage):
    """
    Test case for p-value of sample mean estimator expecting a significant p-value.
    Uses data with a large mean (100) relative to standard error, leading to a small p-value.
    Checks that the p-value is less than 0.05 and within [0, 1].
    Compares the jackknife p-value (t-distribution) with a standard normal p-value.
    """
    np.random.seed(SEED + 300)

    # Use 'y1' with mean=100, which is far from 0, ensuring a significant p-value
    test_df = test_df_fixture[["y1"]].rename(columns={"y1": "y"}).sample(n=sample_size, random_state=SEED + 301)

    def estimator_func(df: pd.DataFrame):
        # Return a 1D array of length 1
        return np.array([df["y"].mean()])

    result = calc_jk_stats(test_df, estimator_func, num_buckets=num_buckets, ci_coverage=ci_coverage)

    # Check p-value dimensions and validity
    assert result["p_values"].shape == (1,)
    assert 0.0 <= result["p_values"][0] <= 1.0

    # ASSERTION: Expect a significant p-value (mean far from 0)
    assert result["p_values"][0] < 0.05

    # Calculate standard normal p-value
    sample_mean = test_df["y"].mean()
    sample_variance = test_df["y"].var(ddof=1)
    standard_error = np.sqrt(sample_variance / sample_size)
    z_stat = sample_mean / standard_error if standard_error != 0 else 0
    normal_p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    # ASSERTION: Compare jackknife p-value (t-distribution) with standard normal p-value
    # Use a relative tolerance of 0.1 due to the difference between t and normal distributions
    assert np.isclose(result["p_values"][0], normal_p_value, rtol=0.1)


@pytest.mark.parametrize(
    "sample_size, num_buckets, ci_coverage",
    [
        (1000, 20, 0.95),
    ],
)
def test_jackknife_p_value_non_significant(test_df_fixture, sample_size, num_buckets, ci_coverage):
    """
    Test case for p-value of sample mean estimator expecting a non-significant p-value.
    Adjusts data to have a mean close to 0, leading to a larger p-value.
    Checks that the p-value is greater than or equal to 0.05 and within [0, 1].
    """
    np.random.seed(SEED + 400)

    # Adjust 'y1' to have a mean close to 0 by subtracting the sample mean
    test_df = test_df_fixture[["y1"]].rename(columns={"y1": "y"}).sample(n=sample_size, random_state=SEED + 401)
    test_df["y"] = test_df["y"] - test_df["y"].mean()

    def estimator_func(df: pd.DataFrame):
        # Return a 1D array of length 1
        return np.array([df["y"].mean()])

    result = calc_jk_stats(test_df, estimator_func, num_buckets=num_buckets, ci_coverage=ci_coverage)

    # Check p-value dimensions and validity
    assert result["p_values"].shape == (1,)
    assert 0.0 <= result["p_values"][0] <= 1.0

    # ASSERTION: Expect a non-significant p-value (mean close to 0)
    assert result["p_values"][0] >= 0.05

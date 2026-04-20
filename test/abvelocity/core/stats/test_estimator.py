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
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# Ensure this imports the concrete Estimator class
from abvelocity.core.stats.estimator import Estimator
from abvelocity.core.stats.normal_ci import calc_standard_normal_ci
from abvelocity.core.stats.student_ci import calc_student_ci

# Constants and Configuration
BASE_SAMPLE_SIZE = 1000
POPULATION_SD = 10.0
POPULATION_MEAN = 100.0
SEED = 1317
CI_COVERAGE = 0.95
ASYMPTOTIC_TOLERANCE_DEFAULT = 0.55  # Relaxed tolerance for jackknife variance
PRIOR_PRECISION = 0.01  # Prior precision (1/variance) for Bayesian estimator
PLOT_WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/estimator").resolve()
PLOT_SAMPLE_SIZES = range(10, 201, 10)  # Sample sizes from 10 to 200, step 10
BOOTSTRAP_N = 1000  # Number of bootstrap samples for median CI
NUM_BUCKETS = 20  # Default number of jackknife buckets


@pytest.fixture(scope="module")
def test_df():
    """Creates a DataFrame for testing, using a normal distribution.

    Returns:
        A pandas DataFrame with a single column 'y' containing normally distributed data.
    """
    np.random.seed(SEED)
    data_array = np.random.normal(loc=POPULATION_MEAN, scale=POPULATION_SD, size=BASE_SAMPLE_SIZE)
    df = pd.DataFrame({"y": data_array})
    return df


class MeanEstimator(Estimator):
    """A concrete Estimator implementation for sample mean."""

    # Enforce array return for k=1 consistency
    def estimator_func(self, df: pd.DataFrame, param: None = None) -> np.ndarray:
        """Computes the sample mean, returning a 1-element NumPy array.

        Args:
            df: The input DataFrame containing the data.
            param: Unused parameter for compatibility.

        Returns:
            A 1D numpy array with the sample mean of the 'y' column.
        """
        return np.array([df["y"].mean()])


class BayesianMeanEstimator(Estimator):
    """A concrete Estimator implementation for Bayesian mean with a normal prior."""

    # Enforce array return for k=1 consistency
    def estimator_func(self, df: pd.DataFrame, param: float = None) -> np.ndarray:
        """Computes the Bayesian mean estimate, returning a 1-element NumPy array.

        Args:
            df: The input DataFrame containing the data.
            param: The prior mean. Defaults to 0.0 if None.

        Returns:
            A 1D numpy array with the Bayesian mean estimate.
        """
        # Use self.param if param is None, which is important for  estimator_func_with_inferred_param
        prior_mean = param if param is not None else self.param if self.param is not None else 0.0
        sample_mean = df["y"].mean()
        n = len(df)
        sample_precision = n / (POPULATION_SD**2)  # Assuming known population variance
        total_precision = sample_precision + PRIOR_PRECISION
        bayesian_mean = (sample_precision * sample_mean + PRIOR_PRECISION * prior_mean) / total_precision
        return np.array([bayesian_mean])


class MedianEstimator(Estimator):
    """A concrete Estimator implementation for sample median."""

    # Enforce array return for k=1 consistency
    def estimator_func(self, df: pd.DataFrame, param: None = None) -> np.ndarray:
        """Computes the sample median, returning a 1-element NumPy array.

        Args:
            df: The input DataFrame containing the data.
            param: Unused parameter for compatibility.

        Returns:
            A 1D numpy array with the sample median of the 'y' column.
        """
        return np.array([df["y"].median()])


class ConstantEstimator(Estimator):
    """A concrete Estimator that always returns its 'param' value."""

    # Enforce array return for k=1 consistency
    def estimator_func(self, df: pd.DataFrame, param: float = 0.0) -> np.ndarray:
        """
        Returns the stored param value as a 1-element NumPy array.

        Args:
            df: The input DataFrame (unused).
            param: The constant value to return.

        Returns:
            A 1D numpy array with the constant value.
        """
        # If the function is called directly with param, use it. Otherwise, use self.param.
        val = param if param is not None else self.param if self.param is not None else 0.0
        return np.array([val])


# --- New K-Dimensional Estimator Class for Testing ---


class KDimMeanEstimator(Estimator):
    """A concrete Estimator implementation that computes means of multiple columns (k-dim)."""

    def estimator_func(self, df: pd.DataFrame, param: list[str] = None) -> np.ndarray:
        """Computes the sample mean of multiple specified columns.

        Args:
            df: The input DataFrame.
            param: A list of column names to compute the mean for.

        Returns:
            A 1D numpy array of length k (number of columns) with the means.
        """
        cols = param if param is not None else self.param
        if cols is None or not cols:
            # Ensure array return even for the default
            return np.array([df["y"].mean()])

        # Result is already a 1D numpy array for k-dim compatibility
        return df[cols].mean().values


@pytest.mark.parametrize(
    "num_buckets, ci_coverage",
    [
        (20, 0.95),
        (50, 0.95),
    ],
)
def test_calc_jk_stats(test_df, num_buckets, ci_coverage):
    """Tests calc_jk_stats method for a sample mean estimator (k=1).

    Verifies jk_value, jk_varcov, jk_ci, and dof attributes.
    """
    estimator = MeanEstimator()

    # Now we rely on estimator_func to return the value as an array
    sample_mean_array = estimator.estimator_func(test_df)
    sample_mean = sample_mean_array[0]

    sample_variance = test_df["y"].var(ddof=1)
    theoretical_variance = sample_variance / BASE_SAMPLE_SIZE

    # Initialize value as array and var as matrix for correct internal processing
    estimator.value = sample_mean_array
    estimator.var = np.array([[theoretical_variance]])

    # The calc_jk_stats call internally calculates a (1, 1) variance-covariance matrix
    result = estimator.calc_jk_stats(test_df, num_buckets=num_buckets, ci_coverage=ci_coverage)

    # Compare array elements
    assert np.isclose(estimator.jk_value[0], result["estimator_value"][0])
    assert np.isclose(estimator.jk_varcov[0, 0], result["estimator_varcov"][0, 0])
    assert np.allclose(estimator.jk_ci, result["ci"], rtol=1e-5)
    assert estimator.dof == num_buckets - 1

    # Jackknife bias check
    jk_bias = sample_mean - estimator.jk_value[0]
    assert np.isclose(jk_bias, 0.0, atol=1e-3)

    # Check variance approximation
    assert np.isclose(estimator.jk_varcov[0, 0], theoretical_variance, rtol=ASYMPTOTIC_TOLERANCE_DEFAULT)

    # Theoretical CI check
    theoretical_jk_ci = calc_student_ci(
        mean=estimator.jk_value[0],
        se=np.sqrt(estimator.jk_varcov[0, 0]),
        dof=num_buckets - 1,
        ci_coverage=ci_coverage,
    )["ci"]
    # We compare the (1, 2) result array with the (1, 2) array on the estimator
    assert np.allclose(estimator.jk_ci, theoretical_jk_ci.reshape(1, 2), rtol=ASYMPTOTIC_TOLERANCE_DEFAULT)


@pytest.mark.parametrize(
    "ci_coverage",
    [0.95, 0.99],
)
def test_calc_normal_ci(test_df, ci_coverage):
    """Tests calc_normal_ci method, verifying ci attribute and error handling (k=1)."""
    estimator = MeanEstimator()

    sample_mean = test_df["y"].mean()
    sample_variance = test_df["y"].var(ddof=1)
    theoretical_variance = sample_variance / BASE_SAMPLE_SIZE

    # Ensure initialization uses arrays
    estimator.value = np.array([sample_mean])
    estimator.var = np.array([[theoretical_variance]])

    ci = estimator.calc_normal_ci(ci_coverage=ci_coverage)

    assert np.allclose(estimator.ci, ci)

    theoretical_ci = calc_standard_normal_ci(mean=sample_mean, se=np.sqrt(theoretical_variance), ci_coverage=ci_coverage)["ci"]
    assert np.allclose(ci, theoretical_ci.reshape(1, 2), rtol=1e-2)

    # --- Error Handling Tests ---
    estimator_invalid = MeanEstimator()

    estimator_invalid.var = np.array([[theoretical_variance]])
    # Error message is now 'value' or 'var' is None due to unified check
    with pytest.raises(ValueError, match="Cannot calculate normal CI: 'value' or 'var' is None."):
        estimator_invalid.calc_normal_ci(ci_coverage=ci_coverage)

    estimator_invalid.value = np.array([sample_mean])
    estimator_invalid.var = None
    with pytest.raises(ValueError, match="Cannot calculate normal CI: 'value' or 'var' is None."):
        estimator_invalid.calc_normal_ci(ci_coverage=ci_coverage)

    # Test for non-positive variance in the marginal component
    estimator_invalid.var = np.array([[-1.0]])
    with pytest.raises(ValueError, match="Cannot calculate normal CI: Component 0 variance is non-positive"):
        estimator_invalid.calc_normal_ci(ci_coverage=ci_coverage)


@pytest.mark.parametrize(
    "ci_coverage, dof",
    [
        (0.95, 19),
        (0.99, 49),
    ],
)
def test_calc_student_ci(test_df, ci_coverage, dof):
    """Tests calc_student_ci method, verifying ci attribute and error handling (k=1)."""
    estimator = MeanEstimator()

    sample_mean = test_df["y"].mean()
    sample_variance = test_df["y"].var(ddof=1)
    theoretical_variance = sample_variance / BASE_SAMPLE_SIZE

    # Ensure initialization uses arrays
    estimator.value = np.array([sample_mean])
    estimator.var = np.array([[theoretical_variance]])
    estimator.dof = dof

    ci = estimator.calc_student_ci(ci_coverage=ci_coverage)

    assert np.allclose(estimator.ci, ci)

    theoretical_ci = calc_student_ci(mean=sample_mean, se=np.sqrt(theoretical_variance), dof=dof, ci_coverage=ci_coverage)["ci"]
    assert np.allclose(ci, theoretical_ci.reshape(1, 2), rtol=1e-2)

    # --- Error Handling Tests ---
    estimator_invalid = MeanEstimator()

    estimator_invalid.var = np.array([[theoretical_variance]])
    estimator_invalid.dof = dof
    # Error message is now 'value', 'var', or 'dof' is None due to unified check
    with pytest.raises(ValueError, match="Cannot calculate student CI: 'value', 'var', or 'dof' is None."):
        estimator_invalid.calc_student_ci(ci_coverage=ci_coverage)

    estimator_invalid.value = np.array([sample_mean])
    estimator_invalid.var = None
    estimator_invalid.dof = dof
    with pytest.raises(ValueError, match="Cannot calculate student CI: 'value', 'var', or 'dof' is None."):
        estimator_invalid.calc_student_ci(ci_coverage=ci_coverage)

    # Test for non-positive variance
    estimator_invalid.var = np.array([[-1.0]])
    with pytest.raises(ValueError, match="Cannot calculate student CI: Component 0 variance is non-positive"):
        estimator_invalid.calc_student_ci(ci_coverage=ci_coverage)

    estimator_invalid.var = np.array([[theoretical_variance]])
    estimator_invalid.dof = None
    with pytest.raises(ValueError, match="Cannot calculate student CI: 'value', 'var', or 'dof' is None."):
        estimator_invalid.calc_student_ci(ci_coverage=ci_coverage)

    estimator_invalid.dof = 0
    with pytest.raises(ValueError, match="Cannot calculate student CI: 'dof' must be positive"):
        estimator_invalid.calc_student_ci(ci_coverage=ci_coverage)


@pytest.mark.parametrize(
    "prior_mean",
    [50.0, 200.0],
)
def test_estimator_func_with_inferred_param(test_df, prior_mean):
    """Tests  estimator_func_with_inferred_param with a Bayesian mean estimator (k=1)."""
    estimator = BayesianMeanEstimator(param=prior_mean)

    sample_mean = test_df["y"].mean()
    n = len(test_df)
    sample_precision = n / (POPULATION_SD**2)
    total_precision = sample_precision + PRIOR_PRECISION
    expected_bayesian_mean = (sample_precision * sample_mean + PRIOR_PRECISION * prior_mean) / total_precision

    bayesian_mean_array = estimator.estimator_func_with_inferred_param(test_df)

    # Extract scalar from array for comparison
    assert np.isclose(bayesian_mean_array[0], expected_bayesian_mean, rtol=1e-5)

    direct_bayesian_mean_array = estimator.estimator_func(test_df, param=prior_mean)
    assert np.isclose(bayesian_mean_array[0], direct_bayesian_mean_array[0], rtol=1e-5)


@pytest.mark.parametrize(
    "op1_val, op2_val, prior_mean",
    [
        (5.0, 2.0, 10.0),  # Number + Number operation check (using ConstantEstimator)
        (100.0, 50.0, 50.0),  # Estimator + Constant
    ],
)
def test_composite_estimator(test_df, op1_val, op2_val, prior_mean):
    """Tests the arithmetic dunder methods and name generation for 1-D estimators."""

    # 1. Setup two base estimators
    # Estimator A: Bayesian Mean
    est_a = BayesianMeanEstimator(param=prior_mean, name="BayesianMean")
    est_a_val = est_a.estimator_func_with_inferred_param(test_df)[0]

    # Estimator B: Constant Estimator
    est_b = ConstantEstimator(param=op2_val, name="Constant")
    est_b_val = est_b.estimator_func_with_inferred_param(test_df)[0]

    # Constant C
    constant_c = op1_val

    str_constant_c = str(constant_c)

    # --- Test Addition (Est + Est, Est + Num, Num + Est) ---
    est_sum_est = est_a + est_b
    assert isinstance(est_sum_est, Estimator)
    assert est_sum_est.name == "(BayesianMean + Constant)"
    # Extract scalar from array for comparison
    assert np.isclose(est_sum_est.estimator_func_with_inferred_param(test_df)[0], est_a_val + est_b_val)

    est_sum_num = est_a + constant_c
    assert isinstance(est_sum_num, Estimator)
    assert est_sum_num.name == f"(BayesianMean + {str_constant_c})"
    assert np.isclose(est_sum_num.estimator_func_with_inferred_param(test_df)[0], est_a_val + constant_c)

    num_sum_est = constant_c + est_b
    assert isinstance(num_sum_est, Estimator)
    assert num_sum_est.name == f"({str_constant_c} + Constant)"
    assert np.isclose(num_sum_est.estimator_func_with_inferred_param(test_df)[0], constant_c + est_b_val)

    # --- Test Subtraction (Est - Est, Est - Num, Num - Est) ---
    est_diff_est = est_a - est_b
    assert isinstance(est_diff_est, Estimator)
    assert est_diff_est.name == "(BayesianMean - Constant)"
    assert np.isclose(est_diff_est.estimator_func_with_inferred_param(test_df)[0], est_a_val - est_b_val)

    num_diff_est = constant_c - est_a  # Uses __rsub__
    assert isinstance(num_diff_est, Estimator)
    assert num_diff_est.name == f"({str_constant_c} - BayesianMean)"
    assert np.isclose(num_diff_est.estimator_func_with_inferred_param(test_df)[0], constant_c - est_a_val)

    # --- Test Multiplication (Est * Est, Est * Num, Num * Est) ---
    est_prod_est = est_a * est_b
    assert isinstance(est_prod_est, Estimator)
    assert est_prod_est.name == "(BayesianMean * Constant)"
    assert np.isclose(est_prod_est.estimator_func_with_inferred_param(test_df)[0], est_a_val * est_b_val)

    num_prod_est = constant_c * est_b
    assert isinstance(num_prod_est, Estimator)
    assert num_prod_est.name == f"({str_constant_c} * Constant)"
    assert np.isclose(num_prod_est.estimator_func_with_inferred_param(test_df)[0], constant_c * est_b_val)

    # --- Test Division (Est / Est, Est / Num, Num / Est) ---
    est_div_est = est_a / est_b
    assert isinstance(est_div_est, Estimator)
    assert est_div_est.name == "(BayesianMean / Constant)"
    assert np.isclose(est_div_est.estimator_func_with_inferred_param(test_df)[0], est_a_val / est_b_val)

    num_div_est = constant_c / est_b  # Uses __rtruediv__
    assert isinstance(num_div_est, Estimator)
    assert num_div_est.name == f"({str_constant_c} / Constant)"
    assert np.isclose(num_div_est.estimator_func_with_inferred_param(test_df)[0], constant_c / est_b_val)

    # Test division by zero exception (only possible with constant divisor)
    est_zero = 0
    with pytest.raises(ZeroDivisionError):
        est_a / est_zero


def test_composite_estimator2(test_df):
    """Tests chaining of arithmetic dunder methods for 1-D estimators."""

    # 1. Setup two base estimators
    est_a = BayesianMeanEstimator(param=10, name="BayesianMean")
    est_a_val = est_a.estimator_func_with_inferred_param(test_df)[0]

    est_b = MeanEstimator(name="Mean")
    est_b_val = est_b.estimator_func_with_inferred_param(test_df)[0]

    c = 500

    est_sum_est = est_a + est_b + c
    assert isinstance(est_sum_est, Estimator)
    assert est_sum_est.name == "((BayesianMean + Mean) + 500)"
    assert np.isclose(est_sum_est.estimator_func_with_inferred_param(test_df)[0], est_a_val + est_b_val + c)


def test_composite_estimator3(test_df):
    """Tests complex arithmetic dunder methods for 1-D estimators."""

    est_a = BayesianMeanEstimator(param=10, name="BayesianMean")
    est_a_val = est_a.estimator_func_with_inferred_param(test_df)[0]

    est_b = MeanEstimator(name="Mean")
    est_b_val = est_b.estimator_func_with_inferred_param(test_df)[0]

    new_est = 0.5 * est_a + 0.5 * est_b
    assert isinstance(new_est, Estimator)

    assert new_est.name == "((0.5 * BayesianMean) + (0.5 * Mean))"
    assert np.isclose(new_est.estimator_func_with_inferred_param(test_df)[0], 0.5 * (est_a_val + est_b_val))


def test_plot_ci_vs_sample_size(test_df):
    """Plots CIs for MeanEstimator and BayesianMeanEstimator vs. sample size."""
    # Ensure output directory exists
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    np.random.seed(SEED + 1)

    # Initialize estimators
    mean_estimator = MeanEstimator()
    bayes_estimator_50 = BayesianMeanEstimator(param=50.0)
    bayes_estimator_200 = BayesianMeanEstimator(param=200.0)

    sample_sizes = list(PLOT_SAMPLE_SIZES)
    mean_ci_lower = []
    mean_ci_upper = []
    bayes_50_ci_lower = []
    bayes_50_ci_upper = []
    bayes_200_ci_lower = []
    bayes_200_ci_upper = []

    for n in sample_sizes:
        # Sample data
        sample_df = test_df.sample(n=n, random_state=SEED + n)

        # MeanEstimator: Compute sample mean and variance
        mean_estimator.value = mean_estimator.estimator_func(sample_df)
        sample_variance = sample_df["y"].var(ddof=1)
        mean_estimator.var = np.array([[sample_variance / n]])  # Variance set as (1, 1) matrix
        mean_ci = mean_estimator.calc_normal_ci(ci_coverage=CI_COVERAGE)
        mean_ci_lower.append(mean_ci[0, 0])
        mean_ci_upper.append(mean_ci[0, 1])

        # BayesianMeanEstimator (prior mean = 50.0)
        bayes_estimator_50.value = bayes_estimator_50.estimator_func_with_inferred_param(sample_df)
        # Variance approximation: use sample variance / n (ignores prior for simplicity)
        bayes_estimator_50.var = np.array([[sample_variance / n]])  # Variance set as (1, 1) matrix
        bayes_50_ci = bayes_estimator_50.calc_normal_ci(ci_coverage=CI_COVERAGE)
        bayes_50_ci_lower.append(bayes_50_ci[0, 0])
        bayes_50_ci_upper.append(bayes_50_ci[0, 1])

        # BayesianMeanEstimator (prior mean = 200.0)
        bayes_estimator_200.value = bayes_estimator_200.estimator_func_with_inferred_param(sample_df)
        bayes_estimator_200.var = np.array([[sample_variance / n]])  # Variance set as (1, 1) matrix
        bayes_200_ci = bayes_estimator_200.calc_normal_ci(ci_coverage=CI_COVERAGE)
        bayes_200_ci_lower.append(bayes_200_ci[0, 0])
        bayes_200_ci_upper.append(bayes_200_ci[0, 1])

    # Create Plotly figure
    fig = go.Figure()

    # True mean line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[POPULATION_MEAN] * len(sample_sizes),
            mode="lines",
            name="True Mean",
            line=dict(color="black", dash="dash"),
        )
    )

    # MeanEstimator CI band
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=mean_ci_upper + mean_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.3)",  # Blue with transparency
            line=dict(color="rgba(31, 119, 180, 0)"),  # No border
            name="Mean Estimator CI (Normal)",
        )
    )

    # BayesianMeanEstimator CI band (prior mean = 50.0)
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=bayes_50_ci_upper + bayes_50_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.3)",  # Orange with transparency
            line=dict(color="rgba(255, 127, 14, 0)"),
            name="Bayesian CI (Prior Mean = 50)",
        )
    )

    # BayesianMeanEstimator CI band (prior mean = 200.0)
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=bayes_200_ci_upper + bayes_200_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(44, 160, 44, 0.3)",  # Green with transparency
            line=dict(color="rgba(44, 160, 44, 0)"),
            name="Bayesian CI (Prior Mean = 200)",
        )
    )

    # Layout
    fig.update_layout(
        title="Mean Confidence Intervals vs. Sample Size",
        xaxis_title="Sample Size (n)",
        yaxis_title="Estimate Value",
        height=600,
        showlegend=True,
        template="plotly_white",
    )

    # Save plot
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_mean_ci_vs_sample_size.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_plot_median_ci_vs_sample_size(test_df):
    """Plots bootstrap and jackknife CIs for MedianEstimator vs. sample size."""
    # Ensure output directory exists
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    np.random.seed(SEED + 1)

    # Initialize estimator
    median_estimator = MedianEstimator()

    sample_sizes = list(PLOT_SAMPLE_SIZES)
    bootstrap_ci_lower = []
    bootstrap_ci_upper = []
    jk_ci_lower = []
    jk_ci_upper = []

    for n in sample_sizes:
        # Sample data
        sample_df = test_df.sample(n=n, random_state=SEED + n)

        # Bootstrap CI
        bootstrap_medians = []
        for i in range(BOOTSTRAP_N):
            bootstrap_sample = sample_df.sample(n=n, replace=True, random_state=SEED + n + i)
            bootstrap_medians.append(bootstrap_sample["y"].median())
        bootstrap_medians = np.array(bootstrap_medians)
        ci_lower = np.percentile(bootstrap_medians, (1 - CI_COVERAGE) * 100 / 2)
        ci_upper = np.percentile(bootstrap_medians, (1 + CI_COVERAGE) * 100 / 2)
        bootstrap_ci_lower.append(ci_lower)
        bootstrap_ci_upper.append(ci_upper)

        # Jackknife CI
        # The internal logic of calc_jk_stats will now handle the k=1 array correctly
        num_buckets = min(n, NUM_BUCKETS)  # Ensure num_buckets <= n
        jk_result = median_estimator.calc_jk_stats(sample_df, num_buckets=num_buckets, ci_coverage=CI_COVERAGE)
        jk_ci_lower.append(jk_result["ci"][0, 0])
        jk_ci_upper.append(jk_result["ci"][0, 1])

    # Create Plotly figure
    fig = go.Figure()

    # True median line (same as mean for normal distribution)
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[POPULATION_MEAN] * len(sample_sizes),
            mode="lines",
            name="True Median",
            line=dict(color="black", dash="dash"),
        )
    )

    # Bootstrap CI band
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=bootstrap_ci_upper + bootstrap_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.3)",  # Blue with transparency
            line=dict(color="rgba(31, 119, 180, 0)"),  # No border
            name="Median Estimator CI (Bootstrap)",
        )
    )

    # Jackknife CI band
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=jk_ci_upper + jk_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.3)",  # Orange with transparency
            line=dict(color="rgba(255, 127, 14, 0)"),  # No border
            name="Median Estimator CI (Jackknife)",
        )
    )

    # Layout
    fig.update_layout(
        title="Median Confidence Intervals (Bootstrap vs. Jackknife) vs. Sample Size",
        xaxis_title="Sample Size (n)",
        yaxis_title="Estimate Value",
        height=600,
        showlegend=True,
        template="plotly_white",
    )

    # Save plot
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_median_ci_vs_sample_size.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_composite_estimator_plot_diff_mean_ci_vs_sample_size():
    """
    Simulates A/B test data for various sample sizes and plots the Jackknife
    Confidence Interval for the difference in means (Treatment - Control).
    """

    # === 1. INTERNAL CLASS DEFINITION (Scoped to the Test) ===

    class ConditionalMeanEstimator(Estimator):
        """A concrete Estimator implementation for sample mean conditioned on a group."""

        # Enforce array return
        def estimator_func(self, df: pd.DataFrame, param: tuple[str, Any] = None) -> np.ndarray:
            """Computes the mean of 'y' conditioned on a group filter.

            Returns:
                A 1D numpy array with the sample mean.
            """
            if param is None:
                return np.array([np.nan])

            col, val = param
            filtered_df = df[df[col] == val]

            if filtered_df.empty:
                return np.array([np.nan])

            return np.array([filtered_df["y"].mean()])

    # === 2. CONSTANTS AND SETUP ===

    # Constants for this simulation
    true_diff = 1.0
    control_mean = 2.0
    treatment_mean = 3.0
    population_sd_sim = 1.0

    # Ensure output directory exists
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    np.random.seed(SEED + 2)

    # Simulate a large base DataFrame with T and C groups
    n_base_group = BASE_SAMPLE_SIZE // 2

    data_T = np.random.normal(loc=treatment_mean, scale=population_sd_sim, size=n_base_group)
    data_C = np.random.normal(loc=control_mean, scale=population_sd_sim, size=n_base_group)

    df_T = pd.DataFrame({"y": data_T, "group": "T"})
    df_C = pd.DataFrame({"y": data_C, "group": "C"})
    base_df = pd.concat([df_T, df_C], ignore_index=True)

    # === 3. DEFINE COMPOSITE ESTIMATOR ===
    est_t = ConditionalMeanEstimator(param=("group", "T"), name="Mean(T)")
    est_c = ConditionalMeanEstimator(param=("group", "C"), name="Mean(C)")

    # Define the Difference Estimator using composition
    diff_estimator = est_t - est_c

    # === 4. SIMULATION LOOP ===
    sample_sizes = (20, 100, 200, 500)
    jk_ci_lower = []
    jk_ci_upper = []

    for n in sample_sizes:
        # Sample data for total size n, split as evenly as possible
        n_t = n // 2
        n_c = n - n_t

        sample_df = pd.concat(
            [
                base_df[base_df["group"] == "T"].sample(n=n_t, random_state=SEED + n, replace=False),
                base_df[base_df["group"] == "C"].sample(n=n_c, random_state=SEED + n + 1, replace=False),
            ],
            ignore_index=True,
        )

        # Calculate Jackknife CI for the difference estimator
        jk_result = diff_estimator.calc_jk_stats(df=sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        # Note: Since the result is 1-D, we extract [0, 0] and [0, 1] from the (1, 2) CI matrix
        jk_ci_lower.append(jk_result["ci"][0, 0])
        jk_ci_upper.append(jk_result["ci"][0, 1])

    # === 5. PLOT GENERATION ===
    fig = go.Figure()

    # True difference line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_diff] * len(sample_sizes),
            mode="lines",
            name="True Difference (T - C)",
            line=dict(color="black", dash="dash"),
        )
    )

    # Jackknife CI band
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=jk_ci_upper + jk_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.3)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="Difference Estimator CI (Jackknife)",
        )
    )

    # Layout
    fig.update_layout(
        title="Difference in Means Confidence Intervals (Jackknife) vs. Sample Size",
        xaxis_title="Total Sample Size (n)",
        yaxis_title="Estimated Difference (Mean(T) - Mean(C))",
        height=600,
        showlegend=True,
        template="plotly_white",
    )

    # Save plot
    file_name = PLOT_WRITE_PATH.joinpath("test_composite_estimator_plot_diff_mean_ci_vs_sample_size.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_composite_estimator_plot_percent_diff_mean_ci_vs_sample_size():
    """
    Simulates A/B test data for various sample sizes and plots the Jackknife
    Confidence Interval for the percentage difference in means: (T-C)/C * 100.
    """

    # === 1. INTERNAL CLASS DEFINITION (Scoped to the Test) ===

    class ConditionalMeanEstimator(Estimator):
        """A concrete Estimator implementation for sample mean conditioned on a group."""

        # Enforce array return
        def estimator_func(self, df: pd.DataFrame, param: tuple[str, Any] = None) -> np.ndarray:
            """Computes the mean of 'y' conditioned on a group filter.

            Returns:
                A 1D numpy array with the sample mean.
            """
            if param is None:
                return np.array([np.nan])

            col, val = param
            filtered_df = df[df[col] == val]

            if filtered_df.empty:
                return np.array([np.nan])

            return np.array([filtered_df["y"].mean()])

    # === 2. CONSTANTS AND SETUP ===

    # Constants for this simulation
    control_mean = 100.0
    treatment_mean = 105.0
    true_percent_diff = (treatment_mean - control_mean) / control_mean * 100.0  # 5.0
    population_sd_sim = 10.0

    # Ensure output directory exists
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    np.random.seed(SEED + 3)

    # Simulate a large base DataFrame with T and C groups
    n_base_group = BASE_SAMPLE_SIZE // 2

    data_T = np.random.normal(loc=treatment_mean, scale=population_sd_sim, size=n_base_group)
    data_C = np.random.normal(loc=control_mean, scale=population_sd_sim, size=n_base_group)

    df_T = pd.DataFrame({"y": data_T, "group": "T"})
    df_C = pd.DataFrame({"y": data_C, "group": "C"})
    base_df = pd.concat([df_T, df_C], ignore_index=True)

    # === 3. DEFINE COMPOSITE ESTIMATOR ===
    est_t = ConditionalMeanEstimator(param=("group", "T"), name="Mean(T)")
    est_c = ConditionalMeanEstimator(param=("group", "C"), name="Mean(C)")

    # Define the Percentage Difference Estimator using composition: (T - C) / C * 100
    percent_diff_estimator = (est_t - est_c) / est_c * 100.0

    # === 4. SIMULATION LOOP ===
    sample_sizes = (20, 100, 200, 500)
    jk_ci_lower = []
    jk_ci_upper = []

    for n in sample_sizes:
        # Sample data for total size n, split as evenly as possible
        n_t = n // 2
        n_c = n - n_t

        sample_df = pd.concat(
            [
                base_df[base_df["group"] == "T"].sample(n=n_t, random_state=SEED + n, replace=False),
                base_df[base_df["group"] == "C"].sample(n=n_c, random_state=SEED + n + 1, replace=False),
            ],
            ignore_index=True,
        )

        # Calculate Jackknife CI for the percentage difference estimator
        jk_result = percent_diff_estimator.calc_jk_stats(df=sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        # Note: Since the result is 1-D, we extract [0, 0] and [0, 1] from the (1, 2) CI matrix
        jk_ci_lower.append(jk_result["ci"][0, 0])
        jk_ci_upper.append(jk_result["ci"][0, 1])

    # === 5. PLOT GENERATION ===
    fig = go.Figure()

    # True percentage difference line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_percent_diff] * len(sample_sizes),
            mode="lines",
            name="True % Difference (T-C)/C",
            line=dict(color="black", dash="dash"),
        )
    )

    # Jackknife CI band
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=jk_ci_upper + jk_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.3)",
            line=dict(color="rgba(255, 127, 14, 0)"),
            name="% Difference Estimator CI (Jackknife)",
        )
    )

    # Layout
    fig.update_layout(
        title="Percentage Difference in Means CI (Jackknife) vs. Sample Size",
        xaxis_title="Total Sample Size (n)",
        yaxis_title="Estimated % Difference ((Mean(T) - Mean(C))/Mean(C) * 100)",
        height=600,
        showlegend=True,
        template="plotly_white",
    )

    # Save plot
    file_name = PLOT_WRITE_PATH.joinpath("test_composite_estimator_plot_percent_diff_mean_ci_vs_sample_size.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


# ==============================================================================
#                      SINGLE K-DIMENSIONAL TEST
# ==============================================================================

# ==============================================================================
#                      SINGLE K-DIMENSIONAL TEST
# ==============================================================================


def test_kdim_elementwise_and_jackknife():
    """Tests k-dimensional element-wise arithmetic and jackknife calculation."""

    # 1. Setup Data: Create a DataFrame with k=3 columns
    np.random.seed(SEED + 4)
    df = pd.DataFrame(
        {
            "y1": np.random.normal(loc=10.0, scale=1.0, size=BASE_SAMPLE_SIZE),
            "y2": np.random.normal(loc=20.0, scale=2.0, size=BASE_SAMPLE_SIZE),
            "y3": np.random.normal(loc=30.0, scale=3.0, size=BASE_SAMPLE_SIZE),
        }
    )
    cols = ["y1", "y2", "y3"]
    k = len(cols)

    # 2. Setup Estimators (k-dim)
    est_a = KDimMeanEstimator(param=cols, name="KDimMeanA")
    est_b = KDimMeanEstimator(param=cols, name="KDimMeanB")

    # 3. Test Element-wise Subtraction (A - B)
    diff_estimator = est_a - est_b
    assert diff_estimator.name == "(KDimMeanA - KDimMeanB)"

    # Calculate the actual values for comparison
    # Use  estimator_func_with_inferred_param to ensure parameters are retrieved correctly
    est_a_func_val = est_a.estimator_func_with_inferred_param(df)
    est_b_func_val = est_b.estimator_func_with_inferred_param(df)

    expected_diff = est_a_func_val - est_b_func_val

    # FIX from previous iteration: Use  estimator_func_with_inferred_param
    actual_diff = diff_estimator.estimator_func_with_inferred_param(df)

    assert actual_diff.shape == (k,)
    assert np.allclose(actual_diff, expected_diff, rtol=1e-5)

    # 4. Test Element-wise Scalar Multiplication (A * 2.0)
    scalar_mult_estimator = est_a * 2.0
    expected_mult = est_a_func_val * 2.0

    # FIX from previous iteration: Use  estimator_func_with_inferred_param
    actual_mult = scalar_mult_estimator.estimator_func_with_inferred_param(df)

    assert actual_mult.shape == (k,)
    assert np.allclose(actual_mult, expected_mult, rtol=1e-5)

    # 5. Test Jackknife on a Single Base Estimator (est_a)
    # Rationale: The difference of two identical estimators (est_a - est_b)
    # results in a mathematically zero variance, failing the positive check.
    # We test on a single estimator where a non-zero variance is expected.
    NUM_BUCKETS_KDIM = 50
    jk_result = est_a.calc_jk_stats(df=df, num_buckets=NUM_BUCKETS_KDIM, ci_coverage=CI_COVERAGE)

    # Check the dimensions of the Jackknife results
    assert jk_result["estimator_value"].shape == (k,)  # k-dim vector
    assert jk_result["estimator_varcov"].shape == (k, k)  # k x k matrix
    assert jk_result["ci"].shape == (k, 2)  # k x 2 matrix

    # Check that the Jackknife value is close to the original estimate
    jk_value = jk_result["estimator_value"]
    assert np.allclose(jk_value, est_a_func_val, atol=1e-2)

    # Check that the diagonal elements (marginal variances) are positive and non-trivial
    marginal_variances = np.diag(jk_result["estimator_varcov"])
    assert np.all(marginal_variances > 1e-6)  # Check strictly positive (or greater than a small epsilon)


def test_plot_kdim_ci_vs_sample_size():
    """
    Plots the convergence of the 2D marginal confidence region (a rectangle)
    for a KDimMeanEstimator (k=2) as a 3D filled tube, with fine increments
    and a clearly highlighted convergence path.
    """
    # 1. Setup Constants and Data Generation
    KDIM_MEAN_1 = 100.0
    KDIM_MEAN_2 = 50.0
    KDIM_SD_1 = 5.0
    KDIM_SD_2 = 10.0
    CI_COVERAGE = 0.95

    # Use smaller increments for a smoother tube
    sample_sizes_range = range(10, 501, 10)

    # Ensure output directory exists (assuming PLOT_WRITE_PATH is defined globally/in scope)
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    # Note: SEED, BASE_SAMPLE_SIZE, and other constants are assumed to be defined globally/in scope
    np.random.seed(SEED + 5)

    # Generate k=2 data with a large base size
    data_1 = np.random.normal(loc=KDIM_MEAN_1, scale=KDIM_SD_1, size=BASE_SAMPLE_SIZE)
    data_2 = np.random.normal(loc=KDIM_MEAN_2, scale=KDIM_SD_2, size=BASE_SAMPLE_SIZE)
    base_df = pd.DataFrame({"y1": data_1, "y2": data_2})
    cols = ["y1", "y2"]

    # Note: KDimMeanEstimator is assumed to be defined globally/in scope
    estimator = KDimMeanEstimator(param=cols)

    # 2. Simulation Loop: Collect CI Corners and Mesh Faces

    x_plot, y_plot, z_plot = [], [], []
    i_faces, j_faces, k_faces = [], [], []  # For Mesh3d faces

    sample_sizes = list(sample_sizes_range)

    i_rect_size = 4  # 4 vertices per rectangle
    vertex_count = 0

    for n in sample_sizes:
        # Sample data
        sample_df = base_df.sample(n=n, random_state=SEED + n)

        # Compute mean and variance for k=2
        sample_means = sample_df[cols].mean().values
        sample_vars = np.diag(sample_df[cols].cov().values) / n

        estimator.value = sample_means
        estimator.var = np.diag(sample_vars)

        ci_matrix = estimator.calc_normal_ci(ci_coverage=CI_COVERAGE)

        ci_y1_low, ci_y1_high = ci_matrix[0, 0], ci_matrix[0, 1]
        ci_y2_low, ci_y2_high = ci_matrix[1, 0], ci_matrix[1, 1]

        # Define the 4 vertices of the current rectangle (V0, V1, V2, V3)
        # Order: Low-Low, High-Low, High-High, Low-High

        # V0 (Low Y1, Low Y2)
        x_plot.append(n)
        y_plot.append(ci_y1_low)
        z_plot.append(ci_y2_low)
        # V1 (High Y1, Low Y2)
        x_plot.append(n)
        y_plot.append(ci_y1_high)
        z_plot.append(ci_y2_low)
        # V2 (High Y1, High Y2)
        x_plot.append(n)
        y_plot.append(ci_y1_high)
        z_plot.append(ci_y2_high)
        # V3 (Low Y1, High Y2)
        x_plot.append(n)
        y_plot.append(ci_y1_low)
        z_plot.append(ci_y2_high)

        # Connect current rectangle (i) to the previous rectangle (i-1) if it exists
        if vertex_count >= i_rect_size:
            idx_curr = vertex_count
            idx_prev = vertex_count - i_rect_size

            # Create 4 faces (8 triangles) to connect the two rectangles
            for j in range(i_rect_size):
                curr_v = idx_curr + j
                curr_next_v = idx_curr + (j + 1) % i_rect_size
                prev_v = idx_prev + j
                prev_next_v = idx_prev + (j + 1) % i_rect_size

                # Face 1 (Triangle 1): [prev_v, curr_v, curr_next_v]
                i_faces.extend([prev_v])
                j_faces.extend([curr_v])
                k_faces.extend([curr_next_v])

                # Face 2 (Triangle 2): [prev_v, curr_next_v, prev_next_v]
                i_faces.extend([prev_v])
                j_faces.extend([curr_next_v])
                k_faces.extend([prev_next_v])

        vertex_count += i_rect_size

    # 3. Plot Generation using Plotly

    # True means line (convergence path) - Made red and slightly thicker
    true_mean_line = go.Scatter3d(
        x=sample_sizes,
        y=[KDIM_MEAN_1] * len(sample_sizes),
        z=[KDIM_MEAN_2] * len(sample_sizes),
        mode="lines",
        name=f"True Mean Convergence Path (Mu1={KDIM_MEAN_1}, Mu2={KDIM_MEAN_2})",
        line=dict(color="red", dash="dash", width=3),
    )

    # Mesh3d for the solid tube
    ci_mesh = go.Mesh3d(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        i=i_faces,
        j=j_faces,
        k=k_faces,
        opacity=0.35,
        color="blue",
        name="CI Region (Filled Tube)",
        hoverinfo="none",
    )

    # Combine true mean line and filled CI mesh
    fig = go.Figure(data=[ci_mesh, true_mean_line])

    fig.update_layout(
        title="2D Mean Marginal CI Convergence (K=2)",
        scene=dict(
            xaxis_title="Sample Size (n)",
            yaxis_title=f"Mean 1 CI ({cols[0]})",
            zaxis_title=f"Mean 2 CI ({cols[1]})",
            aspectmode="auto",
        ),
        height=700,
        showlegend=True,
        template="plotly_white",
    )

    # 4. Save Plot - Removed "filled" from filename
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_kdim_mean_ci_vs_sample_size.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_plot_2d_mean_ellipse_vs_sample_size():
    """
    Plots the convergence of the 2D joint confidence ellipse
    for the mean vector (μx, μy) estimated by Jackknife, creating a 3D tube,
    using a KDimMeanEstimator on bivariate data.
    """
    # === 1. INTERNAL IMPORTS AND UTILITIES ===
    from scipy.stats import chi2

    # Note: KDimMeanEstimator, Estimator, calc_jk_stats, np, pd, go, os, and Path
    # are assumed to be imported in the module scope.
    # --- Constants for this Bivariate Simulation ---
    POPULATION_SD_X = 5.0
    POPULATION_SD_Y = 10.0
    POPULATION_MEAN_X = 50.0
    POPULATION_MEAN_Y = 100.0
    POPULATION_CORR = 0.5  # Introduce correlation
    CI_COVERAGE = 0.95

    # --- Utility for Confidence Ellipse ---
    def get_confidence_ellipse(mean, cov_matrix, ci_coverage, num_points=20):
        """Generates points for a confidence ellipse."""
        try:
            confidence_level = chi2.ppf(ci_coverage, df=2)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

            # Use max(0, ...) to handle potential negative eigenvalues from small sample VCV
            major_axis = np.sqrt(max(0, eigenvalues[0]) * confidence_level)
            minor_axis = np.sqrt(max(0, eigenvalues[1]) * confidence_level)

            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

            theta = np.linspace(0, 2 * np.pi, num_points)
            S = np.array([major_axis * np.cos(theta), minor_axis * np.sin(theta)])

            R = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )

            points = R @ S
            # Ellipse is plotted as (Mean(X) on Y-axis, Mean(Y) on Z-axis)
            x_ellipse_data = mean[0] + points[0, :].flatten()  # Mean(X)
            y_ellipse_data = mean[1] + points[1, :].flatten()  # Mean(Y)

            return x_ellipse_data, y_ellipse_data

        except np.linalg.LinAlgError:
            # Fallback if VCV matrix is singular
            return np.array([]), np.array([])

    # --- Joint Mean Estimator (using KDimMeanEstimator) ---
    # This function returns the array [Mean(X), Mean(Y)]
    def joint_means_estimator_func(df: pd.DataFrame):
        return df[["x", "y"]].mean().values

    # === 2. DATA GENERATION & SETUP ===
    num_buckets = 25
    sample_sizes_range = range(100, 1001, 10)
    num_ellipse_points = 20

    # Ensure output directory exists (using module-scoped PLOT_WRITE_PATH)
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    np.random.seed(SEED + 4)

    # Generate bivariate data
    cov_matrix = [
        [POPULATION_SD_X**2, POPULATION_CORR * POPULATION_SD_X * POPULATION_SD_Y],
        [POPULATION_CORR * POPULATION_SD_X * POPULATION_SD_Y, POPULATION_SD_Y**2],
    ]
    data_array = np.random.multivariate_normal(
        mean=[POPULATION_MEAN_X, POPULATION_MEAN_Y],
        cov=cov_matrix,
        size=BASE_SAMPLE_SIZE,
    )
    base_df = pd.DataFrame({"x": data_array[:, 0], "y": data_array[:, 1]})

    # Define the estimator object for its methods
    estimator = KDimMeanEstimator(param=["x", "y"])

    # === 3. SIMULATION LOOP: Collect Ellipse Vertices and Mesh Faces ===
    x_plot, y_plot, z_plot = [], [], []
    i_faces, j_faces, k_faces = [], [], []  # For Mesh3d faces

    sample_sizes = list(sample_sizes_range)
    vertex_count = 0

    for n in sample_sizes:
        # Sample data
        sample_df = base_df.sample(n=n, random_state=SEED + n)

        # Compute mean and VCV using Jackknife
        try:
            # The estimator uses the defined estimator_func on the input dataframe
            jk_result = estimator.calc_jk_stats(df=sample_df, num_buckets=num_buckets, ci_coverage=CI_COVERAGE)
            jk_means = jk_result["estimator_value"].flatten()
            jk_cov_matrix = jk_result["estimator_varcov"]
        except Exception:
            # Skip if Jackknife fails (e.g., sample size too small for buckets)
            continue

        # Generate ellipse points
        x_ellipse, y_ellipse = get_confidence_ellipse(
            mean=jk_means,  # [Mean(X), Mean(Y)]
            cov_matrix=jk_cov_matrix,
            ci_coverage=CI_COVERAGE,
            num_points=num_ellipse_points,
        )

        # Store ellipse vertices for Mesh3d
        current_ellipse_vertices = []
        for j in range(num_ellipse_points):
            x_plot.append(n)  # X-axis is Sample Size
            y_plot.append(x_ellipse[j])  # Y-axis is Mean(X)
            z_plot.append(y_ellipse[j])  # Z-axis is Mean(Y)
            current_ellipse_vertices.append(j)

        # Connect current ellipse (i) to the previous ellipse (i-1)
        if vertex_count >= num_ellipse_points:
            idx_curr = vertex_count
            idx_prev = vertex_count - num_ellipse_points

            # Create faces to connect the two ellipses (a twisted tube)
            for j in range(num_ellipse_points):
                curr_v = idx_curr + j
                curr_next_v = idx_curr + (j + 1) % num_ellipse_points
                prev_v = idx_prev + j
                prev_next_v = idx_prev + (j + 1) % num_ellipse_points

                # Face 1 (Triangle 1): [prev_v, curr_v, curr_next_v]
                i_faces.append(prev_v)
                j_faces.append(curr_v)
                k_faces.append(curr_next_v)

                # Face 2 (Triangle 2): [prev_v, curr_next_v, prev_next_v]
                i_faces.append(prev_v)
                j_faces.append(curr_next_v)
                k_faces.append(prev_next_v)

        vertex_count += num_ellipse_points

    # === 4. PLOT GENERATION ===

    # True means line (convergence path)
    true_mean_line = go.Scatter3d(
        x=sample_sizes,
        y=[POPULATION_MEAN_X] * len(sample_sizes),
        z=[POPULATION_MEAN_Y] * len(sample_sizes),
        mode="lines",
        name=f"True Mean Path (μx={POPULATION_MEAN_X}, μy={POPULATION_MEAN_Y})",
        line=dict(color="red", dash="dash", width=10),
    )

    # Mesh3d for the solid tube
    ci_mesh = go.Mesh3d(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        i=i_faces,
        j=j_faces,
        k=k_faces,
        opacity=0.20,
        color="blue",
        name=f"Jackknife Ellipse Region ({CI_COVERAGE*100:.0f}% CI)",
        hoverinfo="none",
    )

    # Combine true mean line and filled CI mesh
    fig = go.Figure(data=[ci_mesh, true_mean_line])

    fig.update_layout(
        title="2D Joint Mean Ellipse Convergence vs. Sample Size (Jackknife VCV)",
        scene=dict(
            xaxis_title="Sample Size (n)",
            yaxis_title="Mean(X)",
            zaxis_title="Mean(Y)",
            aspectmode="auto",
        ),
        height=700,
        showlegend=True,
        template="plotly_white",
    )

    # 5. Save Plot
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_2d_mean_ellipse_vs_sample_size.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")

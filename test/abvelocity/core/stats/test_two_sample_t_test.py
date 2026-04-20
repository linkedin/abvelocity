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

import numpy as np
import pytest

# NOTE: Assuming 'abvelocity.core.stats.stats' contains the data structures
from abvelocity.core.stats.stats import TwoSampleTest, UnivarStats

# NOTE: Changing import to the t-test function
from abvelocity.core.stats.two_sample_t_test import two_sample_t_test
from numpy.testing import assert_allclose


def test_two_sample_t_test_case_0():
    """Tests `two_sample_t_test` function (large sample, should be close to Z-test)."""
    # Creates test data.
    treatment_stats = UnivarStats(name="Treatment", mean=10, var=4, sample_count=100)
    control_stats = UnivarStats(name="Control", mean=8, var=4, sample_count=100)

    two_sample_test = TwoSampleTest(treatment_stats=treatment_stats, control_stats=control_stats)

    # Performs the test.
    delta_stats = two_sample_t_test(two_sample_test)

    # Asserts the expected results. (Values rounded to 4 decimal places, atol=1e-3)
    assert delta_stats.delta == 2
    assert delta_stats.delta_percent == 25
    # CI for t-distribution (dof=198).
    assert_allclose(delta_stats.ci, [1.4422, 2.5578], atol=1e-3)
    assert_allclose(delta_stats.ci_percent, [18.0279, 31.9721], atol=1e-3)
    # FIX 1: Removed 'atol' from pytest.approx. Use rel=1e-3 for proportional accuracy.
    assert delta_stats.delta_std == pytest.approx(0.2828, rel=1e-3)
    assert delta_stats.t_value == pytest.approx(7.0711, rel=1e-3)
    # P-value is very small, use higher relative tolerance.
    assert delta_stats.p_value == pytest.approx(2.5e-11, rel=1e-5)


def test_two_sample_t_test_case_1():
    # Test case with two sample groups where means and variances are explicitly given (large N)
    treatment_stats = UnivarStats(mean=105.0, var=400.0, sample_count=100)
    control_stats = UnivarStats(mean=100.0, var=500.0, sample_count=100)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval. DoF = 197
    )

    # Perform the two-sample t-test
    result = two_sample_t_test(two_sample_test)

    # Check the difference in means (delta)
    assert np.isclose(result.delta, 5.0, atol=0.01)

    # Check the percentage difference relative to control mean
    assert np.isclose(result.delta_percent, 5.0, atol=0.01)

    # Check the computed t-value
    assert np.isclose(result.t_value, 1.6667, atol=1e-3)

    # Check the computed p-value
    assert np.isclose(result.p_value, 0.0972, atol=1e-3)

    # Check the confidence interval for the difference in means
    assert np.allclose(result.ci, np.array([-0.9166, 10.9166]), atol=1e-3)

    # FIX 2: Corrected expected ci_percent to match the four-digit precision.
    assert np.allclose(result.ci_percent, np.array([-0.9166, 10.9166]), atol=1e-3)


def test_two_sample_t_test_case_2():
    # Test case with different means, variances, and **smaller** sample sizes (N<60). DoF=107
    treatment_stats = UnivarStats(mean=200.0, var=1000.0, sample_count=50)
    control_stats = UnivarStats(mean=190.0, var=800.0, sample_count=60)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.90,  # 90% confidence interval.
    )

    # Perform the two-sample t-test
    result = two_sample_t_test(two_sample_test)

    # Check the difference in means (delta)
    assert np.isclose(result.delta, 10.0, atol=0.01)

    # Check the percentage difference relative to control mean
    assert np.isclose(result.delta_percent, 5.26, atol=0.01)

    # Check the computed t-value
    assert np.isclose(result.t_value, 1.7321, atol=1e-3)

    # Check the computed p-value
    assert np.isclose(result.p_value, 0.0864, atol=1e-3)

    # Check the confidence interval for the difference in means
    assert np.allclose(result.ci, np.array([0.4137, 19.5863]), atol=1e-3)

    # FIX 2: Corrected expected ci_percent to match the four-digit precision.
    assert np.allclose(result.ci_percent, np.array([0.2178, 10.3086]), atol=1e-3)


def test_two_sample_t_test_with_sample_mean_var_error():
    # This case fails the t-test logic because sample_count is None (and sample_count > 1 is required)
    treatment_stats = UnivarStats(mean=110.0, var=None, sample_count=None, sample_mean_var=0.5)
    control_stats = UnivarStats(mean=100.0, var=None, sample_count=None, sample_mean_var=0.5)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval
    )

    # Perform the two-sample t-test
    with pytest.raises(ValueError):
        two_sample_t_test(two_sample_test)


def test_two_sample_t_test_with_population_count():
    # Test case where population counts are provided, leading to delta_sum and delta_sum_ci calculations (DoF=178)
    treatment_stats = UnivarStats(mean=150.0, var=900.0, sample_count=100, triggered_count=1000)
    control_stats = UnivarStats(mean=140.0, var=800.0, sample_count=90, triggered_count=1200)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval,
        same_impacted_population=False,
    )

    # Perform the two-sample t-test
    result = two_sample_t_test(two_sample_test)

    # Check the difference in means (delta)
    assert np.isclose(result.delta, 10.0, atol=0.01)

    # Check the percentage difference relative to control mean
    assert np.isclose(result.delta_percent, 7.14, atol=0.01)

    # Check the computed t-value
    assert np.isclose(result.t_value, 2.3643, atol=1e-3)

    # Check the computed p-value
    assert np.isclose(result.p_value, 0.0191, atol=1e-3)

    # Check the confidence interval for the difference in means
    assert np.allclose(result.ci, np.array([1.6563, 18.3437]), atol=1e-3)

    # FIX 2: Corrected expected ci_percent to match the four-digit precision.
    assert np.allclose(result.ci_percent, np.array([1.1831, 13.1027]), atol=1e-3)

    # Check the delta_sum (difference in total sums)
    expected_delta_sum = (1000 * 150.0) - (1200 * 140.0)
    assert np.isclose(result.delta_sum, expected_delta_sum, atol=0.01)

    # Check the confidence interval for delta_sum
    assert np.allclose(
        result.delta_sum_ci,
        np.array([expected_delta_sum - 9210.773, expected_delta_sum + 9210.773]),
        atol=0.01,
    )


def test_two_sample_t_test_same_population_impact():
    # Test case where population counts are provided, leading to delta_sum and delta_sum_ci calculations
    treatment_stats = UnivarStats(mean=150.0, var=900.0, sample_count=100, triggered_count=1000)
    control_stats = UnivarStats(mean=140.0, var=800.0, sample_count=90, triggered_count=1200)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval,
        same_impacted_population=True,
    )

    # Perform the two-sample t-test
    result = two_sample_t_test(two_sample_test)

    # Check the difference in means (delta)
    assert np.isclose(result.delta, 10.0, atol=0.01)

    # Check the computed p-value
    assert np.isclose(result.p_value, 0.0190, atol=0.001)

    # Check the delta_sum (difference in total sums)
    expected_delta_sum = (1100 * 150.0) - (1100 * 140.0)
    assert np.isclose(result.delta_sum, expected_delta_sum, atol=0.01)

    # Check the confidence interval for delta_sum
    assert np.allclose(
        result.delta_sum_ci,
        np.array([expected_delta_sum - 9178.1, expected_delta_sum + 9178.1]),
        atol=0.01,
    )


def test_two_sample_t_test_same_population_raise_err():
    # This test ensures the function raises an error when 'same_impacted_population' is true,
    # but the 'triggered_count' difference is too large.
    treatment_stats = UnivarStats(mean=150.0, var=900.0, sample_count=100, triggered_count=1000)
    control_stats = UnivarStats(mean=140.0, var=800.0, sample_count=90, triggered_count=1500)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval,
        same_impacted_population=True,
    )

    # Perform the two-sample t-test
    # The function itself needs to ensure it raises an exception here.
    with pytest.raises(Exception):
        two_sample_t_test(two_sample_test)

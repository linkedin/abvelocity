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
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import numpy as np
import pytest
from abvelocity.core.stats.stats import TwoSampleTest, UnivarStats
from abvelocity.core.stats.two_sample_z_test import two_sample_z_test
from numpy.testing import assert_allclose


def test_two_sample_z_test_case_0():
    """Tests `two_sample_z_test` function."""
    # Creates test data.
    treatment_stats = UnivarStats(name="Treatment", mean=10, var=4, sample_count=100)
    control_stats = UnivarStats(name="Control", mean=8, var=4, sample_count=100)

    two_sample_test = TwoSampleTest(treatment_stats=treatment_stats, control_stats=control_stats)

    # Performs the test.
    delta_stats = two_sample_z_test(two_sample_test)

    # Asserts the expected results.
    assert delta_stats.delta == 2
    assert delta_stats.delta_percent == 25
    assert_allclose(delta_stats.ci, [1.44563847, 2.55436153])
    assert_allclose(delta_stats.ci_percent, [18.07, 31.93])
    assert delta_stats.delta_std == pytest.approx(0.282842712474619)
    assert delta_stats.z_value == pytest.approx(7.0710678118654755)
    assert delta_stats.p_value == pytest.approx(1.5374368445009168e-12)


def test_two_sample_z_test_case_1():
    # Test case with two sample groups where means and variances are explicitly given
    treatment_stats = UnivarStats(mean=105.0, var=400.0, sample_count=100)
    control_stats = UnivarStats(mean=100.0, var=500.0, sample_count=100)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval
    )

    # Perform the two-sample z-test
    result = two_sample_z_test(two_sample_test)

    # Check the difference in means (delta)
    assert np.isclose(result.delta, 5.0, atol=0.01)

    # Check the percentage difference relative to control mean
    assert np.isclose(result.delta_percent, 5.0, atol=0.01)

    # Check the computed z-value
    assert np.isclose(result.z_value, 1.666, atol=0.01)

    # Check the computed p-value
    assert np.isclose(result.p_value, 0.0955, atol=0.01)

    # Check the confidence interval for the difference in means
    assert np.allclose(result.ci, np.array([-0.88, 10.88]), atol=0.01)

    # Check the confidence interval for the percentage difference
    assert np.allclose(result.ci_percent, np.array([-0.88, 10.88]), atol=0.01)


def test_two_sample_z_test_case_2():
    # Test case with different means, variances, and sample sizes
    treatment_stats = UnivarStats(mean=200.0, var=1000.0, sample_count=50)
    control_stats = UnivarStats(mean=190.0, var=800.0, sample_count=60)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.90,  # 90% confidence interval
    )

    # Perform the two-sample z-test
    result = two_sample_z_test(two_sample_test)

    # Check the difference in means (delta)
    assert np.isclose(result.delta, 10.0, atol=0.01)

    # Check the percentage difference relative to control mean
    assert np.isclose(result.delta_percent, 5.26, atol=0.01)

    # Check the computed z-value
    assert np.isclose(result.z_value, 1.732, atol=0.01)

    # Check the computed p-value
    assert np.isclose(result.p_value, 0.08326, atol=0.01)

    # Check the confidence interval for the difference in means
    assert np.allclose(result.ci, np.array([0.5034, 19.4966]), atol=0.01)

    # Check the confidence interval for the percentage difference
    assert np.allclose(result.ci_percent, np.array([0.265, 10.261]), atol=0.01)


def test_two_sample_z_test_with_sample_mean_var():
    # Test case where sample mean variances are pre-calculated and provided directly
    treatment_stats = UnivarStats(mean=110.0, var=None, sample_count=None, sample_mean_var=0.5)
    control_stats = UnivarStats(mean=100.0, var=None, sample_count=None, sample_mean_var=0.5)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval
    )

    # Perform the two-sample z-test
    result = two_sample_z_test(two_sample_test)

    # Check the difference in means (delta)
    assert np.isclose(result.delta, 10.0, atol=0.01)

    # Check the percentage difference relative to control mean
    assert np.isclose(result.delta_percent, 10.0, atol=0.01)

    # Check the computed z-value
    assert np.isclose(result.z_value, 10.0, atol=0.01)

    # Check the computed p-value
    assert np.isclose(result.p_value, 0.0, atol=0.01)

    # Check the confidence interval for the difference in means
    assert np.allclose(result.ci, np.array([8.040, 11.96]), atol=0.01)

    # Check the confidence interval for the percentage difference
    assert np.allclose(result.ci_percent, np.array([8.04, 11.96]), atol=0.01)


def test_two_sample_z_test_with_population_count():
    # Test case where population counts are provided, leading to delta_sum and delta_sum_ci calculations
    treatment_stats = UnivarStats(mean=150.0, var=900.0, sample_count=100, triggered_count=1000)
    control_stats = UnivarStats(mean=140.0, var=800.0, sample_count=90, triggered_count=1200)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval,
        same_impacted_population=False,
    )

    # Perform the two-sample z-test
    result = two_sample_z_test(two_sample_test)

    # Check the difference in means (delta)
    assert np.isclose(result.delta, 10.0, atol=0.01)

    # Check the percentage difference relative to control mean
    assert np.isclose(result.delta_percent, 7.14, atol=0.01)

    # Check the computed z-value
    assert np.isclose(result.z_value, 2.364, atol=0.01)

    # Check the computed p-value
    assert np.isclose(result.p_value, 0.0180, atol=0.01)

    # Check the confidence interval for the difference in means
    assert np.allclose(result.ci, np.array([1.71, 18.29]), atol=0.01)

    # Check the confidence interval for the percentage difference
    assert np.allclose(result.ci_percent, np.array([1.222, 13.064]), atol=0.01)

    # Check the delta_sum (difference in total sums)
    expected_delta_sum = (1000 * 150.0) - (1200 * 140.0)
    assert np.isclose(result.delta_sum, expected_delta_sum, atol=0.01)

    # Check the confidence interval for delta_sum
    assert np.allclose(
        result.delta_sum_ci,
        np.array([expected_delta_sum - 9151.164, expected_delta_sum + 9151.164]),
        atol=0.01,
    )


def test_two_sample_z_test_same_population_impact():
    # Test case where population counts are provided, leading to delta_sum and delta_sum_ci calculations
    # In this example we assume population impact is same
    # The triggered counts are slightly off here but they do pass the default threshold: `.triggered_population_diff_thresh`
    # which is 0.1 by default.
    treatment_stats = UnivarStats(mean=150.0, var=900.0, sample_count=100, triggered_count=1000)
    control_stats = UnivarStats(mean=140.0, var=800.0, sample_count=90, triggered_count=1200)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval,
        same_impacted_population=True,
    )

    # Perform the two-sample z-test
    result = two_sample_z_test(two_sample_test)

    # Check the difference in means (delta)
    assert np.isclose(result.delta, 10.0, atol=0.01)

    # Check the percentage difference relative to control mean
    assert np.isclose(result.delta_percent, 7.14, atol=0.01)

    # Check the computed z-value
    assert np.isclose(result.z_value, 2.364, atol=0.01)

    # Check the computed p-value
    assert np.isclose(result.p_value, 0.0180, atol=0.01)

    # Check the confidence interval for the difference in means
    assert np.allclose(result.ci, np.array([1.71, 18.29]), atol=0.01)

    # Check the confidence interval for the percentage difference
    assert np.allclose(result.ci_percent, np.array([1.222, 13.064]), atol=0.01)

    # Check the delta_sum (difference in total sums)
    expected_delta_sum = (1100 * 150.0) - (1100 * 140.0)
    assert np.isclose(result.delta_sum, expected_delta_sum, atol=0.01)

    # Check the confidence interval for delta_sum
    assert np.allclose(
        result.delta_sum_ci,
        np.array([expected_delta_sum - 9118.690, expected_delta_sum + 9118.690]),
        atol=0.01,
    )


def test_two_sample_z_test_same_population_raise_err():
    # Test case where population counts are provided, leading to delta_sum and delta_sum_ci calculations
    # In this example we assume population impact is same
    # The triggered counts are slightly off here but they do pass the default threshold: `.triggered_population_diff_thresh`
    # which is 0.1 by default.
    treatment_stats = UnivarStats(mean=150.0, var=900.0, sample_count=100, triggered_count=1000)
    control_stats = UnivarStats(mean=140.0, var=800.0, sample_count=90, triggered_count=1500)
    two_sample_test = TwoSampleTest(
        treatment_stats=treatment_stats,
        control_stats=control_stats,
        ci_coverage=0.95,  # 95% confidence interval,
        same_impacted_population=True,
    )

    # Perform the two-sample z-test
    with pytest.raises(Exception):
        two_sample_z_test(two_sample_test)

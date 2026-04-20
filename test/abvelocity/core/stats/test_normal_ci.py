# Bse 2-CLAUSE LICENSE

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
from abvelocity.core.stats.normal_ci import calc_standard_normal_ci


# Test for known mean and standard deviation with specific values
def test_calc_standard_normal_ci():
    # Example 1: Test case where mean=0 and se=1 (standard normal distribution)
    mean = 0
    se = 1
    ci_coverage = 0.95
    result = calc_standard_normal_ci(mean, se, ci_coverage)

    # Expected z-value, p-value, and confidence interval for a standard normal distribution
    expected_z_value = 0.0
    expected_p_value = 1.0
    expected_ci = np.array([-1.96, 1.96])  # 95% CI for standard normal distribution

    assert np.isclose(result["z_value"], expected_z_value, atol=1e-2)
    assert np.isclose(result["p_value"], expected_p_value, atol=1e-2)
    assert np.allclose(result["ci"], expected_ci, atol=1e-2)

    # Example 2: Test case with non-zero mean
    mean = 2
    se = 1
    ci_coverage = 0.95
    result = calc_standard_normal_ci(mean, se, ci_coverage)

    expected_z_value = 2.0
    expected_p_value = 0.0455
    expected_ci = np.array([0.04, 3.96])  # 95% CI for given parameters

    assert np.isclose(result["z_value"], expected_z_value, atol=1e-2)
    assert np.isclose(result["p_value"], expected_p_value, atol=1e-2)
    assert np.allclose(result["ci"], expected_ci, atol=1e-2)

    # Example 3: Test case with larger standard deviation
    mean = 5
    se = 2
    ci_coverage = 0.99
    result = calc_standard_normal_ci(mean, se, ci_coverage)

    expected_z_value = 2.5
    expected_p_value = 0.0124
    expected_ci = np.array([-0.152, 10.152])  # 99% CI for given parameters

    assert np.isclose(result["z_value"], expected_z_value, atol=1e-2)
    assert np.isclose(result["p_value"], expected_p_value, atol=1e-2)
    assert np.allclose(result["ci"], expected_ci, atol=1e-2)


# Test for edge cases
def test_calc_standard_normal_ci_edge_cases():
    # Edge Case: Negative standard deviation (se < 0)
    mean = 5
    se = -1
    ci_coverage = 0.95
    with pytest.raises(ValueError):
        calc_standard_normal_ci(mean, se, ci_coverage)

    # Edge Case: Confidence interval coverage outside [0, 1]
    mean = 5
    se = 1
    ci_coverage = 1.5
    with pytest.raises(ValueError):
        calc_standard_normal_ci(mean, se, ci_coverage)

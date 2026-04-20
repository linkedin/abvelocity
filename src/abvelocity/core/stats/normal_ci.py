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
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import math

import numpy as np
from scipy import stats


def calc_standard_normal_ci(mean: float, se: float, ci_coverage: float) -> dict:
    """
    Calculates the z-value, p-value, and confidence interval (CI) for a standard normal distribution.

    Note that here `se` is the standard error for the sample mean (or more generally the estimator) and that is why no sample size is given.
    This is useful when `se` can be estimated for more complex situations e.g. when the sample mean (or more generally the estimator) is estimated
    for weighted samples.

    Args:
        mean: The estimated mean of the estimator. For sample mean estimator this is simply the obsevred sample mean.
        se: The standard error of the estimator. Must be non-negative.
        ci_coverage: The desired coverage of the confidence interval, a value between 0 and 1.

    Returns:
        dict: A dictionary containing the following keys:
            - "z_value" (float): The z-value, which is the number of standard deviations the mean is from zero.
            - "p_value" (float): The two-tailed p-value corresponding to the z-value.
            - "ci" (numpy.ndarray): A 1D array with two elements, representing the lower and upper bounds of the confidence interval.

    Raises:
        ValueError: If `se` is negative.
        ValueError: If `ci_coverage` is not between 0 and 1.

    """
    if se < 0:
        raise ValueError(f"se has to be non-negative: {se}")

    # If se is None, we return
    if not se or math.isnan(se):
        return {"z_value": np.nan, "p_value": np.nan, "ci": np.array([np.nan, np.nan])}

    if not (0 < ci_coverage < 1):
        raise ValueError(f"ci_coverage has to be between 0 and 1: {ci_coverage}")

    # If both se and mean are zero, we do not divide 0/0 and assign a signifcant z_value of magnitude 5
    # If only se is zero z_value will be 5 times the sign of the mean
    if se == 0 and mean == 0:
        z_value = 5.0
    elif se == 0:
        z_value = np.sign(mean) * 5.0
    else:
        z_value = mean / se

    p_value = 2 * (1 - stats.norm.cdf(abs(z_value), loc=0, scale=1))
    # Calculates Z (standard normal) quantile for a given confidence interval coverage level
    tail_mass = 1 - (1 - ci_coverage) / 2
    # Calculate standard normal quantile for that tail
    ci_radius_coef = stats.norm.ppf(tail_mass, loc=0, scale=1)
    ci_radius = ci_radius_coef * se
    ci = np.array([mean - ci_radius, mean + ci_radius])

    return {"z_value": z_value, "p_value": p_value, "ci": ci}

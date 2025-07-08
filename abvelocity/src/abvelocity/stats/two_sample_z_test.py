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
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import math

import numpy as np
from scipy import stats

from abvelocity.stats.stats import DeltaStats, TwoSampleTest

SMALL_SAMPLE_SIZE = 5
"""A constant to handle corner cases where the sample is too small."""


def calc_standard_normal_ci(mean: float, sd: float, ci_coverage: float) -> dict:
    """
    Calculates the z-value, p-value, and confidence interval (CI) for a standard normal distribution.
    Note that here `sd` is the standard deviation for the sample mean and that is why no sample size is given.
    This is useful when `sd` can be estimated for more complex situations e.g. when the sample mean is estimated
    for weighted samples.

    Args:
        mean: The mean of the distribution.
        sd: The standard deviation of the distribution. Must be non-negative.
        ci_coverage: The desired coverage of the confidence interval, a value between 0 and 1.

    Returns:
        dict: A dictionary containing the following keys:
            - "z_value" (float): The z-value, which is the number of standard deviations the mean is from zero.
            - "p_value" (float): The two-tailed p-value corresponding to the z-value.
            - "ci" (numpy.ndarray): A 1D array with two elements, representing the lower and upper bounds of the confidence interval.

    Raises:
        ValueError: If `sd` is negative.
        ValueError: If `ci_coverage` is not between 0 and 1.

    """
    if sd < 0:
        raise ValueError(f"sd has to be non-negative: {sd}")

    # If sd is None, we return
    if not sd or math.isnan(sd):
        return {"z_value": np.nan, "p_value": np.nan, "ci": np.array([np.nan, np.nan])}

    if not (0 < ci_coverage < 1):
        raise ValueError(f"ci_coverage has to be between 0 and 1: {ci_coverage}")

    # If both sd and mean are zero, we do not divide 0/0 and assign a signifcant z_value of magnitude 5
    # If only sd is zero z_value will be 5 times the sign of the mean
    if sd == 0 and mean == 0:
        z_value = 5.0
    elif sd == 0:
        z_value = np.sign(mean) * 5.0
    else:
        z_value = mean / sd

    p_value = 2 * (1 - stats.norm.cdf(abs(z_value), loc=0, scale=1))
    # Calculates Z (standard normal) quantile for a given confidence interval coverage level
    tail_mass = 1 - (1 - ci_coverage) / 2
    # Calculate standard normal quantile for that tail
    ci_radius_coef = stats.norm.ppf(tail_mass, loc=0, scale=1)
    ci_radius = ci_radius_coef * sd
    ci = np.array([mean - ci_radius, mean + ci_radius])

    return {"z_value": z_value, "p_value": p_value, "ci": ci}


def two_sample_z_test(two_sample_test: TwoSampleTest) -> DeltaStats:
    """Z test to compare two populations with their univariate stats given in:

        - two_sample_test["treatment_stats"] (treatment arm)
        - two_sample_test["control_stats"] (control arm)

    This assumes the sample mean is available for both arms in `.mean` attribute.

    Then there are two cases for sample mean variance for each arm:

        - (a) `.sample_mean_var` is already available.
        - (b) `.sample_mean_var` is None. But `.var` and `.sample_count` is available.

    For (a) the test will directly use the sample mean variance to calculate confidence intervals
    and p-values etc.
    For (b), sample mean var will be computed here by dividing `.var` by `.sample_count` attributes.

    It uses pre-calculated means, variances
    and sample sizes (sample_count) from the two samples.

    The assumptions are

        - the two samples are independent of each other
        - each sample consists of independent and identically distributed observations

    Args:
        two_sample_test: A dataclass which includes:

            - the univariate stats for the treatment and control arms
            - the confidence interval coverage.

    Returns:
        delta_stats: A dataclass which includes experiment results e.g. delta, CI, p-value etc.

    Raises:
        None.
    """
    # First for each arm, if `.sample_mean_var` is not available,
    # we calculate it by dividing `.var` by `.sample_count`
    for arm in [two_sample_test.treatment_stats, two_sample_test.control_stats]:
        if arm.sample_mean_var is None:
            if arm.var is None or arm.sample_count is None:
                raise ValueError(
                    f"if `.sample_mean_var` is missing `.var` and `.sample_count` need to be available in {arm}"
                )
            else:
                arm.sample_mean_var = arm.var / arm.sample_count

        # We also check for existence of .mean in both arms
        if arm.mean is None:
            raise ValueError(f"`.mean` cannot be None for an experiment arm: {arm}")

    treatment_stats = two_sample_test.treatment_stats
    control_stats = two_sample_test.control_stats
    ci_coverage = two_sample_test.ci_coverage

    delta = treatment_stats.mean - control_stats.mean

    t_var = treatment_stats.sample_mean_var
    c_var = control_stats.sample_mean_var

    delta_var = 0.0

    # These following operations are to handle rare corner cases
    # Check for NaN cases first, as NaN propagates and needs explicit handling
    if math.isnan(t_var) and math.isnan(c_var):
        delta_var = float("nan")
    # If one of them is Nan we use a conservative variance
    # by multiplying the existing one by 3
    # If both are not Nan, we add them (this is the default situation)
    elif math.isnan(t_var):
        delta_var = c_var * 3
    elif math.isnan(c_var):
        delta_var = t_var * 3
    else:
        delta_var = t_var + c_var

    delta_std = np.sqrt(delta_var)

    z_value = np.nan
    p_value = np.nan
    ci = np.array([np.nan, np.nan])  # Initialize CI to NaN as well for consistency

    if (
        control_stats.sample_count is not None and control_stats.sample_count <= SMALL_SAMPLE_SIZE
    ) or (
        treatment_stats.sample_count is not None
        and treatment_stats.sample_count <= SMALL_SAMPLE_SIZE
    ):
        # If either sample count is less than 4, return NaNs for z_value and p_value.
        # ci will remain NaN from initialization.
        pass  # z_value, p_value, and ci are already initialized to NaN
    else:
        # Only calculate if sample counts are sufficient
        sig_res = calc_standard_normal_ci(mean=delta, sd=delta_std, ci_coverage=ci_coverage)
        z_value = sig_res["z_value"]
        p_value = sig_res["p_value"]
        ci = sig_res["ci"]

    # TODO: Reza, the CI below can be improved by accounting for the randomness in the mean in denominator.
    # This is not a major, issue though as sample sizes are typically large for the control arm.
    # We need to handle potential division by zero if control_stats.mean is 0
    if control_stats.mean != 0:
        delta_percent = round(100 * delta / control_stats.mean, 3)
        ci_percent = (100 * ci / control_stats.mean).round(3)
    else:
        delta_percent = np.nan
        ci_percent = np.array([np.nan, np.nan])

    # Calculate the difference between sum of metric in treatment and control arms.
    # This includes calculating conf. interval for the difference in the sums.
    delta_sum = None
    delta_sum_ci = None
    t_triggered_count = treatment_stats.triggered_count
    c_triggered_count = control_stats.triggered_count
    if t_triggered_count is not None and c_triggered_count is not None:
        if two_sample_test.same_impacted_population:
            # In this case we assume the impacted population is the same.
            # This implies `.triggered_count` to be very close if not identical.
            # Therefore we take their average and will issue a warning if they
            # are not close enough.
            triggered_count = 1 / 2 * (t_triggered_count + c_triggered_count)
            diff = abs(t_triggered_count - c_triggered_count) / (
                t_triggered_count + c_triggered_count
            )

            if diff > two_sample_test.triggered_population_diff_thresh:
                raise ValueError(
                    "`same_impacted_population` is True, yet the `.triggered_count` are very different on the two arms: "
                    f"treatment: {t_triggered_count}, control: {c_triggered_count}"
                )

            delta_sum = (
                triggered_count * treatment_stats.mean - triggered_count * control_stats.mean
            )
            # To calculate variance, note that we need a power of two for constant multipliers
            # Also note that it is summable since the control and treament are assumed to be independent
            delta_sum_var = (triggered_count**2) * treatment_stats.sample_mean_var + (
                triggered_count**2
            ) * control_stats.sample_mean_var

        else:
            delta_sum = (
                t_triggered_count * treatment_stats.mean - c_triggered_count * control_stats.mean
            )
            # To calculate variance, note that we need a power of two for constant multipliers
            # Also note that it is summable since the control and treament are assumed to be independent
            delta_sum_var = (t_triggered_count**2) * treatment_stats.sample_mean_var + (
                c_triggered_count**2
            ) * control_stats.sample_mean_var

        delta_sum_std = np.sqrt(delta_sum_var)
        # Apply the same sample count check for delta_sum_ci as well
        if (
            control_stats.sample_count is not None
            and control_stats.sample_count <= SMALL_SAMPLE_SIZE
        ) or (
            treatment_stats.sample_count is not None
            and treatment_stats.sample_count <= SMALL_SAMPLE_SIZE
        ):
            delta_sum_ci = np.array([np.nan, np.nan])
        else:
            sig_res_sum = calc_standard_normal_ci(
                mean=delta_sum, sd=delta_sum_std, ci_coverage=ci_coverage
            )
            delta_sum_ci = sig_res_sum["ci"]

    return DeltaStats(
        delta=delta,
        delta_percent=delta_percent,
        ci=ci,
        ci_percent=ci_percent,
        delta_std=delta_std,
        delta_sum=delta_sum,
        delta_sum_ci=delta_sum_ci,
        z_value=z_value,
        p_value=p_value,
        sample_counts=(control_stats.sample_count, treatment_stats.sample_count),
        impacted_counts=(control_stats.triggered_count, treatment_stats.triggered_count),
    )

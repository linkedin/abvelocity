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

import math

import numpy as np
from abvelocity.core.stats.stats import DeltaStats, TwoSampleTest
from abvelocity.core.stats.student_ci import calc_student_ci

# The Welch-Satterthwaite equation for degrees of freedom is used to perform
# a two-sample t-test that does not assume equal population variances.
# Reference: B.L. Welch, "The Significance of the Difference Between Two Means When the Population Variances Are Unequal,"
# Biometrika 29 (3/4): 350-362 (1938).

SMALL_SAMPLE_SIZE = 5
"""A constant to handle corner cases where the sample is too small."""


def two_sample_t_test(two_sample_test: TwoSampleTest) -> DeltaStats:
    """T-test (Welch's) to compare two independent populations with their univariate stats given.

    This test is used when the population variance is unknown, and it does not
    assume equal population variances (heteroscedasticity). It calculates the
    effective degrees of freedom using the Welch-Satterthwaite equation.

    Args:
        two_sample_test: A dataclass which includes:
            - the univariate stats for the treatment and control arms
            - the confidence interval coverage.

    Returns:
        delta_stats: A dataclass which includes experiment results e.g. delta, CI, p-value etc.

    Raises:
        ValueError: If required statistics (mean, var, sample_count) are missing or invalid
                    (e.g., sample_count <= 1).
    """
    treatment_stats = two_sample_test.treatment_stats
    control_stats = two_sample_test.control_stats
    ci_coverage = two_sample_test.ci_coverage

    # --- 1. Variance Pre-computation and Checks ---
    for arm in [treatment_stats, control_stats]:
        if arm.sample_mean_var is None:
            if arm.var is None or arm.sample_count is None:
                raise ValueError(f"if `.sample_mean_var` is missing `.var` and `.sample_count` need to be available in {arm}")
            else:
                # Calculate sample mean variance: Var(mean) = Var(X) / N
                arm.sample_mean_var = arm.var / arm.sample_count

        if arm.mean is None:
            raise ValueError(f"`.mean` cannot be None for an experiment arm: {arm}")

        if arm.sample_count is None or arm.sample_count <= 1:
            # Degrees of freedom calculation requires N > 1 (N-1 > 0)
            raise ValueError(f"`.sample_count` must be > 1 for t-test variance estimation: {arm}")

    # --- 2. Delta and Delta Variance Calculation ---
    delta = treatment_stats.mean - control_stats.mean

    t_var = treatment_stats.sample_mean_var
    c_var = control_stats.sample_mean_var

    # Handle NaN variance cases
    if math.isnan(t_var) and math.isnan(c_var):
        delta_var = float("nan")
    elif math.isnan(t_var):
        # Conservative estimate for missing variance
        delta_var = c_var * 3
    elif math.isnan(c_var):
        # Conservative estimate for missing variance
        delta_var = t_var * 3
    else:
        # Variance of the difference
        delta_var = t_var + c_var

    delta_std = np.sqrt(delta_var)

    # --- 3. Degrees of Freedom, T-value, P-value, and CI Calculation ---
    t_dof = treatment_stats.sample_count - 1
    c_dof = control_stats.sample_count - 1

    # Initialize stats to NaN
    t_value = np.nan
    p_value = np.nan
    ci = np.array([np.nan, np.nan])
    dof = np.nan

    if control_stats.sample_count <= SMALL_SAMPLE_SIZE or treatment_stats.sample_count <= SMALL_SAMPLE_SIZE:
        # If either sample count is too small, return NaNs
        pass
    elif delta_var == 0 or math.isnan(delta_var):
        # If variance is zero, the difference is deterministic or indeterminate.
        p_value = 1.0 if delta == 0 else 0.0
        ci = np.array([delta, delta])
    else:
        # Welch-Satterthwaite equation for degrees of freedom:
        numerator = (t_var + c_var) ** 2
        denominator = (t_var**2 / t_dof) + (c_var**2 / c_dof)
        # Round down to the nearest integer for a conservative test
        dof = int(np.floor(numerator / denominator))

        # Use the provided utility function
        sig_res = calc_student_ci(mean=delta, se=delta_std, dof=dof, ci_coverage=ci_coverage)
        t_value = sig_res["t_value"]
        p_value = sig_res["p_value"]
        ci = sig_res["ci"]

    # --- 4. Delta Percent Calculation ---
    if control_stats.mean is not None and control_stats.mean != 0:
        delta_percent = round(100 * delta / control_stats.mean, 3)
        ci_percent = (100 * ci / control_stats.mean).round(3)
    else:
        delta_percent = np.nan
        ci_percent = np.array([np.nan, np.nan])

    # --- 5. Delta Sum Calculation ---
    delta_sum = None
    delta_sum_ci = None
    t_triggered_count = treatment_stats.triggered_count
    c_triggered_count = control_stats.triggered_count

    if t_triggered_count is not None and c_triggered_count is not None:
        if two_sample_test.same_impacted_population:
            # Pooled triggered count
            triggered_count = 1 / 2 * (t_triggered_count + c_triggered_count)

            diff = abs(t_triggered_count - c_triggered_count) / (t_triggered_count + c_triggered_count)

            if diff > two_sample_test.triggered_population_diff_thresh:
                raise ValueError(
                    "`same_impacted_population` is True, yet the `.triggered_count` are very different on the two arms: "
                    f"treatment: {t_triggered_count}, control: {c_triggered_count}"
                )

            delta_sum = triggered_count * delta
            # Note that above is equl to: `triggered_count * treatment_stats.mean - triggered_count * control_stats.mean`
            # To calculate variance, note that we need a power of two for constant multipliers
            # Also note that it is summable since the control and treament are assumed to be independent
            delta_sum_var = (triggered_count**2) * delta_var
            # Note this is equl to: `(t_triggered_count**2) * (treatment_stats.sample_mean_var + control_stats.sample_mean_var)`

        else:
            delta_sum = t_triggered_count * treatment_stats.mean - c_triggered_count * control_stats.mean
            delta_sum_var = (t_triggered_count**2) * t_var + (c_triggered_count**2) * c_var

        delta_sum_std = np.sqrt(delta_sum_var)

        # Apply the same sample count check for delta_sum_ci as well
        if control_stats.sample_count <= SMALL_SAMPLE_SIZE or treatment_stats.sample_count <= SMALL_SAMPLE_SIZE or math.isnan(dof):
            delta_sum_ci = np.array([np.nan, np.nan])
        elif delta_sum_var > 0:
            # Use the Welch-Satterthwaite dof for the sum difference CI
            sig_res_sum = calc_student_ci(mean=delta_sum, se=delta_sum_std, dof=dof, ci_coverage=ci_coverage)
            delta_sum_ci = sig_res_sum["ci"]
        else:
            delta_sum_ci = np.array([delta_sum, delta_sum])

    return DeltaStats(
        delta=delta,
        delta_percent=delta_percent,
        ci=ci,
        ci_percent=ci_percent,
        delta_std=delta_std,
        delta_sum=delta_sum,
        delta_sum_ci=delta_sum_ci,
        t_value=t_value,
        p_value=p_value,
        sample_counts=(control_stats.sample_count, treatment_stats.sample_count),
        impacted_counts=(control_stats.triggered_count, treatment_stats.triggered_count),
    )

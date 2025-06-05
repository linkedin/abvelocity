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

from dataclasses import dataclass
from typing import Optional


@dataclass
class UnivarStats:
    name: Optional[str] = None
    """The variable name."""
    mean: Optional[float] = None
    """The mean of the variable."""
    sd: Optional[float] = None
    """The standard deviation of the variable."""
    var: Optional[float] = None
    """The variance of the variable."""
    sample_count: Optional[float] = None
    """count of the variable in the sample."""
    sum: Optional[float] = None
    """sum of the variable."""
    sum_sq: Optional[float] = None
    """sum of the variable squared."""
    sample_mean_var: Optional[float] = None
    """This is an estimate of the variance of the sample mean.
    This field is useful especially in cases where the sample mean's variance is needed to
    be calculacuted in non-trivial ways, e.g., in the context of stratified sampling.
    In the simple case, the sample mean variance would be easily calculated as `var / count` and
    this field would be somewhat redundant.
    """
    triggered_count: Optional[float] = None
    """
    The triggered count refers to the count of all units,
    when a triggering mechanism is there and after triggering it assigns units to various
    alternatives (variants).
    In order to estimate the impact on the entire population, this value is utilized.
    For example it is used to calculate `.delta_sum` and a confidence interval  for it:
    `.delta_sum_ci` in `DeltaStats`.
    """


@dataclass
class TwoSampleTest:
    treatment_stats: UnivarStats
    """The test arm stats."""
    control_stats: UnivarStats
    """The control arm stats."""
    ci_coverage: float = 0.95
    """This denotes the coverage of the confidence interval."""
    same_impacted_population: bool = True
    """
    This bool determines if the impacted population by the two arms is the same.
    Note that in the experiment, there will be different number of triggers for
    each arm potentially in the sample (`sample_count`).
    But when this is True, we expect `.triggered_count` to be very close if not
    identical when this bool is True.
    """
    triggered_population_diff_thresh: float = 0.1
    """
    This is used to throw ValueError if `same_impacted_population` is True
    but the two arms contain different `.triggered_count` in them which are
    off by more than this threshold in terms of this metric: `abs(x-y) / (x+y)`,
    where x and y are the two different population.
    """


@dataclass
class DeltaStats:
    delta: Optional[float] = None
    """
    The raw difference between the two means of the populations.
    """
    delta_percent: Optional[float] = None
    """
    The percentage difference between the two means of the populations.
    """
    ci: Optional[float] = None
    """
    The confidence interval for the difference between the two means of the populations.
    """
    ci_percent: Optional[float] = None
    """
    The confidence interval for the percentage difference between the two means of the populations.
    """
    delta_std: Optional[float] = None
    """
    The standard deviation of the difference between the two means of the populations.
    """
    delta_sum: Optional[float] = None
    """
    The delta sum over the entire population (if population is finite).
    """
    delta_sum_ci: Optional[float] = None
    """
    The confidence interval for the difference in population sums.
    """
    z_value: Optional[float] = None
    """
    The Z-value for the difference between the two means of the populations.
    """
    p_value: Optional[float] = None
    """
    The P-value for the difference between the two means of the populations.
    """
    sample_counts: Optional[tuple[int]] = None
    """
    A tuple with:

        - The number of the units for the control arm.
        - The number of the units for the treatment arm.

    """
    impacted_counts: Optional[tuple[int]] = None
    """
    A tuple with:

    - The number of triggered (impacted) units across all variants
        with the same trigger state as control.
    - The number of triggered (impacted) units on all variants with the same
        trigger state as treatment.

    Note that these two values should be almost always either equal or very close.
    """


DELTA_STATS_OUTPUT_NAMES = ["delta", "delta(%)", "CI", "CI(%)", "delta_std", "z_value", "p_value"]
"""User-friendly output names for the delta stats.
These names are used in `two_sample_z_test` to generate the output dictionary.
TODO: Reza: Use these for creating user-friendly reports.
"""

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
"""This module contains generic constants for models and algos."""

VARIANT_COL = "variant"
"""
Denotes a column in data which includes experiment variant id.
This could be a tuple for multi-experiments.
The expected type is either `str` or `tuple[str]`.
"""

VARIANT_COUNT_COL = "variant_count"
"""
The count of units for that given variants.
The expected type is `int`.
"""

VARIANT_PERCENT_COL = "variant_percent"
"""
The percent of units for that given variants.
The expected type is `float`.
"""

TRIGGER_STATE_COL = "trigger_state"
"""Denotes a column in data which includes the status of a unit of experiment in terms of triggering.
This is either a `bool` for univariate experiments or `tuple[bool]` for multi-experiments.
"""

TRIGGER_STATE_COUNT_COL = "trigger_state_count"
"""The count of units for that given trigger state.
The expcted type is `int`.
"""

TRIGGER_STATE_PERCENT_COL = "trigger_state_percent"
"""The percent of units for that given trigger state.
The expected type is `float`.
"""

TRIGGER_STATE_OVERALL_COL = "trigger_state_overall"
"""Denotes a column in data which includes the overall trigger state of a unit of experiment.
This is a `bool` column.
"""

TRIGGER_STATE_OVERALL_PERCENT_COL = "trigger_state_overall_weight"
"""Denotes a column in data which includes the overall trigger state of a unit of experiment as a percentage.
This is a `float` column.
"""

TRIGGER_STATE_OVERALL_COUNT_COL = "trigger_state_overall_count"
"""Denotes a column in data which includes the overall trigger state of a unit of experiment as a count.
This is an `int` column.
"""

VARIANT_OVER_TRIGGERED_PERCENT_COL = "variant_over_triggered_pcnt"
"""
Denotes a column in data which includes the percent of the corresponding triggered for the given variant.
This will constitue a distribution of variant occurrence conditional on the given trigger stats.
For example see the following two examples:

    - the trigger state is (True, False).
        Then the corresponding variants could be these three:
        ("v1", "nan") and ("v2", "nan") and ("control", "nan")
        each covering 33.33% of the triggered (sums up to 100.00%).
    - the trigger state is (True, True).
        Then correponding variants could be these six:
        ("v1", "enabled"), ("v2", "enabled"), ("control", "enabled"),
        ("v1", "control"), ("v2", "control"), ("control", "control")
        each covering variuos percentages which also sums up to 100.00%

This is useful information to understand the experiment assignments and can be
used in calculating the impact of the experiment across population (site-wide impact) after experiment is launched.
Note that such information could be available during the experiment design as the assignments are decided.
The two distributions will not be exactly the same due to randomness in the assignment procedure.
"""

TIME_COL = "datepartition"
"""
Denotes a column which includes timestamp / date information.
"""

TRIGGER_TIME_COL = "trigger_date"
"""
Experiment trigger date column.
We assume this column is a tuple column.
"""

MIN_TRIGGER_TIME_COL = "min_trigger_date"
"""
This is the minimum trigger date across experiments.
This is an added column to `expt_df`.
"""

MEAN_COL = "mean"
"""
Denotes a column in data which includes the mean of a metric.
"""

SD_COL = "sd"
"""Denotes a column in data which includes the standard deviation of a metric."""

VAR_COL = "var"
"""Denotes a column in data which includes the variance of a metric."""

SAMPLE_COUNT_COL = "sample_count"
"""
This often maps to `VARIANT_COUNT_COL` in the context of experimentation.
Denotes a column in data which includes the count of units (typically experiment units).
The sample here referes to observed units rather than a larger pool which can be triggered units.

The number of triggerd units (see `TRIGGERED_COUNT_COL`) can be higher than the sample count,
because that might include other variants / conterfactuals in the experiment.

For example for an experiment with arms "control", "v1", "v2".

    - the count for "v1" can be 30
    - the triggered count can be 100 because it includes the observed counts for "v2" and "control" as well.

Also note that situation might be more complicated for multi-experiments but the same logic applies.

For example assume

Expt 1 has three variants v1, v2, control
Expt 2 has three variants enabled, control

a launch (v1, enabled): impact will be in three partitions

    - (v1, nan), (nan, enabled), (v1, enabled)

then these need to compared with (respective to their order)

    - (control, nan), (nan, control), (control, control)

For example for (v1, nan) v.s. (control, nan) each of these have an observed count
and triggered count.

The triggered count for (v1, nan) will consist of the sum of the observed counts for

    - (control, nan)
    - (v1, nan)
    - (v2, nan)

"""

TRIGGERD_COUNT_COL = "triggered_count_col"
"""
This often maps to `TRIGGER_STATE_COUNT_COL` in the context of experimentation.
Denotes a column in data which includes the count of units for all the possible variants
which are to choose from in an experiment.
The number of triggerd units can be higher than the sample count (see `SAMPLE_COUNT_COL`),
because that might include other variants / conterfactuals  in the experiment.

For example for an experiment with arms "control", "v1", "v2".

    - the count for "v1" can be 30
    - the triggered count can be 100 because it includes the observed counts for "v2" and "control" as well.

Also note that situation might be more complicated for multi-experiments but the same logic applies.

For example assume

Expt 1 has three variants v1, v2, control
Expt 2 has three variants enabled, control

a launch (v1, enabled): impact will be in three partitions

    - (v1, nan), (nan, enabled), (v1, enabled)

then these need to compared with (respective to their order)

    - (control, nan), (nan, control), (control, control)

For example for (v1, nan) v.s. (control, nan) each of these have an observed count
and triggered count.

The triggered count for (v1, nan) will consist of the sum of the observed counts for

    - (control, nan)
    - (v1, nan)
    - (v2, nan)
"""

SUM_COL = "sum"
"""Denotes a column in data which includes the sum of a metric."""

SUM_SQ_COL = "sum_sq"
"""Denotes a column in data which includes the sum of a metric squared."""

CI_COL = "ci"
"""Denotes a column in data which includes the confidence interval of a metric."""

CI_PERCENT_COL = "ci_percent"
"""Denotes a column in data which includes the confidence interval of a metric as a percentage."""

DELTA_SUM_CI_COL = "delta_sum_ci"
"""Denotes a column in data which includes the confidence interval for difference of sum of metric between treatment and control."""

CATEG_NAN_VALUE = "nan"
"""Denotes a value in a categorical column which represents a missing value.
This is used to overwrite ad-hoc missing values in categorical columns,
which are usually a result of joins.
"""

CONTROL_LABEL = "control"
"""The typical label for the control group in an experiment."""

TREATMENT_LABEL = "treatment"
"""The typical label for the treatment group in an experiment."""

UNIT_COL = "memberid"
"""This is to specify the experiment unit column name for experiment assignments."""


EXPT_ID_COL = "exptid"
"""This is to specify the experiment ID column."""


METRIC_NAME_COL = "metric"
"""
Denotes a column in data which includes the metric name.
"""

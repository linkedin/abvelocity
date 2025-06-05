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

import pandas as pd

from abvelocity.param.constants import (
    CATEG_NAN_VALUE,
    CONTROL_LABEL,
    TRIGGER_STATE_COL,
    TRIGGER_STATE_COUNT_COL,
    TRIGGER_STATE_OVERALL_COL,
    TRIGGER_STATE_PERCENT_COL,
    VARIANT_COL,
    VARIANT_COUNT_COL,
    VARIANT_OVER_TRIGGERED_PERCENT_COL,
    VARIANT_PERCENT_COL,
)
from abvelocity.param.derived_expt_stats import DerivedExptStats
from abvelocity.param.launch import Launch
from abvelocity.param.variant import TriggerState, Variant, variant_to_trigger_state
from abvelocity.utils.check_df_validity import check_df_validity


def get_expt_stats(df: pd.DataFrame) -> DerivedExptStats:
    """
    This function computes the derived statistics for a multi-experiment (could be univariate as well).

    Args:
        df: The input DataFrame containing the experiment data.
        The df includes `VARIANT_COL` column.

    Returns:
        DerivedExptStats: The derived statistics for the experiment.
    """
    check_df_validity(df=df, needed_cols=[VARIANT_COL], err_trigger_source="get_expt_stats")

    variant_count_df = df.groupby(VARIANT_COL).size().reset_index(name=VARIANT_COUNT_COL)
    variant_count_df[TRIGGER_STATE_COL] = variant_count_df[VARIANT_COL].map(
        lambda v: variant_to_trigger_state(Variant(value=v)).value
    )
    variant_count_df[TRIGGER_STATE_OVERALL_COL] = variant_count_df[VARIANT_COL].map(
        lambda v: variant_to_trigger_state(Variant(value=v)).overall_value
    )

    trigger_state_count_df = (
        variant_count_df.groupby(TRIGGER_STATE_COL)
        .agg(**{TRIGGER_STATE_COUNT_COL: (VARIANT_COUNT_COL, "sum")})
        .reset_index()
    )

    variant_values = variant_count_df[VARIANT_COL].unique()
    variants = [Variant(value=v) for v in variant_values]
    trigger_state_values = variant_count_df[TRIGGER_STATE_COL].unique()
    trigger_states = [TriggerState(value=ts) for ts in trigger_state_values]

    v0 = variant_values[0]
    if isinstance(v0, tuple):  # multi-experiment
        launches = [Launch(value=v) for v in variant_values if CATEG_NAN_VALUE not in v]
        non_control_launches = [
            Launch(value=v)
            for v in variant_values
            if CATEG_NAN_VALUE not in v and v != tuple([CONTROL_LABEL] * len(v))
        ]
    else:  # simple experiment
        launches = [Launch(value=v) for v in variant_values if v != CATEG_NAN_VALUE]
        non_control_launches = [
            Launch(value=v) for v in variant_values if v != CATEG_NAN_VALUE and v != CONTROL_LABEL
        ]

    # Add the trigger state count for each variant.
    # This will first determines which trigger state each variant belong too
    # and returns the total count
    # This is done by aggregation w.r.t `VARIANT_COL` and `transform` method.
    variant_count_df[TRIGGER_STATE_COUNT_COL] = (
        variant_count_df[VARIANT_COUNT_COL]
        .groupby(variant_count_df[TRIGGER_STATE_COL])
        .transform("sum")
    )

    # Sets index for the dataframes to make it easier and faster to extract data.
    # For example in order to get the count of a variant, we can simply execute `variant_count_df.at[variant_value, VARIANT_COUNT_COL]`
    variant_count_df.set_index(VARIANT_COL, inplace=True)
    trigger_state_count_df.set_index(TRIGGER_STATE_COL, inplace=True)

    total_count = variant_count_df[VARIANT_COUNT_COL].sum()
    total_triggered_count = variant_count_df.loc[
        variant_count_df[TRIGGER_STATE_OVERALL_COL], VARIANT_COUNT_COL
    ].sum()
    total_triggered_percent = 100.0 * total_triggered_count / total_count

    variant_count_df[VARIANT_PERCENT_COL] = (
        100.0 * variant_count_df[VARIANT_COUNT_COL] / total_count
    )
    variant_count_df[TRIGGER_STATE_PERCENT_COL] = (
        100.0 * variant_count_df[TRIGGER_STATE_COUNT_COL] / total_count
    )

    # For each trigger state, we find the distribution of the variants with that state.
    # For example see the following two examples:
    # - the trigger state is (True, False).
    #     Then the corresponding variants could be these three:
    #     ("v1", "nan") and ("v2", "nan") and ("control", "nan")
    #     each covering 33.33% of the triggered (sums up to 100.00%).
    # - the trigger state is (True, True).
    #     Then correponding variants could be these six:
    #     ("v1", "enabled"), ("v2", "enabled"), ("control", "enabled"),
    #     ("v1", "control"), ("v2", "control"), ("control", "control")
    #     each covering variuos percentages which also sums up to 100.00%
    variant_count_df[VARIANT_OVER_TRIGGERED_PERCENT_COL] = (
        100.0 * variant_count_df[VARIANT_COUNT_COL] / variant_count_df[TRIGGER_STATE_COUNT_COL]
    )

    trigger_state_count_df[TRIGGER_STATE_PERCENT_COL] = (
        100.0 * trigger_state_count_df[TRIGGER_STATE_COUNT_COL] / total_count
    )

    conditional_trigger_dfs = None
    overlap_rates = None
    if isinstance(v0, tuple):
        num_expts = len(v0)
        conditional_trigger_dfs = {}
        overlap_rates = {}
        for i in range(len(v0)):
            ind = trigger_state_count_df.index.map(lambda x: x[i])
            df = trigger_state_count_df.loc[ind]
            if len(df) > 0:
                conditional_triggered_count = sum(df[TRIGGER_STATE_COUNT_COL])
                df.loc[:, TRIGGER_STATE_PERCENT_COL] = 100 * (
                    df[TRIGGER_STATE_COUNT_COL] / conditional_triggered_count
                )
                conditional_trigger_dfs[i + 1] = df
                # Compute the amount of overlap for experiment i
                # The only possibility where there is no overlap is:
                # the case where other index than `i` is False
                x0 = [False] * num_expts
                x0[i] = True
                t0 = tuple(x0)
                ind = df.index.map(lambda x: x != t0)
                overlap_rate = sum(df.loc[ind][TRIGGER_STATE_PERCENT_COL])
                overlap_rates[i + 1] = overlap_rate

    return DerivedExptStats(
        variants=variants,
        trigger_states=trigger_states,
        launches=launches,
        non_control_launches=non_control_launches,
        variant_count_df=variant_count_df,
        trigger_state_count_df=trigger_state_count_df,
        total_count=total_count,
        total_triggered_count=total_triggered_count,
        total_triggered_percent=total_triggered_percent,
        conditional_trigger_dfs=conditional_trigger_dfs,
        overlap_rates=overlap_rates,
    )

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

from typing import Optional

import numpy as np
import pandas as pd

from abvelocity.param.constants import (
    SAMPLE_COUNT_COL,
    SUM_COL,
    SUM_SQ_COL,
    TRIGGER_STATE_COUNT_COL,
    VARIANT_COL,
)
from abvelocity.param.variant import VariantList
from abvelocity.stats.stats import UnivarStats
from abvelocity.utils.check_df_validity import check_df_validity


def calc_variant_list_metric_stats(
    variant_list: VariantList,
    variant_metric_stats_df: pd.DataFrame,
    variant_col: str = VARIANT_COL,
    variant_count_df: Optional[pd.DataFrame] = None,
    trigger_state_count_col: str = TRIGGER_STATE_COUNT_COL,
) -> UnivarStats:
    """Calculates a (weighted) statistics for a list of variants based on individual variant stats.
        Each variant is assumed to represent a larger strata of the population which we call trigger state.
        The population strata counts are given in trigger state counts within `variant_count_df`.

        Special case:
        This works for the case where there are no weights as well, in which case we do a simple computation which is more stable.
        In that simple case we simply take sums / means etc across variants as we assume their weights are the same in the population.
        For example each variant is a random bucket of the population.

        General weighted case:
        The weights are calculated using observed counts (`trigger_state_count_col`) in a larger set
        which reflects the population. Note that here the input is counts from the population as opposed
        to weights. This is because we want to also calculate the population sum using the counts as well.

        In the context of multi-experiment analysis:
        To calculate the impact of a launch, two appropriate lists of variants (partitions) are needed: aka two `VariantList`s.
        In this case weights are needed because certain variants (partitions) may represent a larger portion of the population.
        The population counts here are calculated using the triggering states.
        The samples are assumed to be large enough to represent the population and the counts (therefore weights) are treated as constants
        in the variance computations.

        Mathematically speaking suppose (in pseudo code):

            - the variants are `v_1, v_2, ..., v_k`
            - the counts for variants are `n_1, n_2, ..., n_k`
            - the population (trigger state) counts are `N_1, ..., N_k`.
            - the within variant sums are `sum_1, sum_2, ..., sum_k` where `sum_i = sum x_i,j`
            - the within variant means are `mean_1, mean_2, ..., mean_k` where `mean_i = sum_i / n_i`.
            - the within variant variances are `var_1, var_2, ..., var_k`
            - var_i denotes the variance for the i-th variant.
            - mu_hat_i denotes the sample mean for the i-th variant.
            - above two lines imply that VAR(mu_hat_i) = var_i / n_i

        Also denote:

            - N = sum N_i
            - n = sum n_i
            - w_i = N_i/N

        Then we have sum(w_i) = 1.

        Then the population mean can be estimated by the weighted sample mean (in pseudo code):

            - mu_hat = sum(w_i * mu_hat_i)

        Then the variance of the weighted sample mean (mu_hat) is calculated as:

            - sample_mean_var = VAR(mu_hat) = VAR(sum(w_i * mu_hat_i)) = sum(w_i^2 * VAR(mu_hat_i)) = sum(w_i^2 * var_i / n_i)

        This is because var_i / n_i is the variance of the sample mean for the i-th variant.

        The population sum can be estimated by:

            - weighted_sum = N * mu_hat

        For the multi-experiments case, usually we expect some of the population count to be roughly speaking k times of the variant count,
        for an integer k. But k would vary depending on the variant.
        Here is a simple example with two experiments:

            - Expt 1: t1, c1
            - Expt 2: t2, c2

        Which means both experiments have a treatment (t_i, i=1,2) and control (c_i, i=1,2) arm only.

        Now consider the `Launch((t1, t2))`. This Launch will impact the users through these variants

            - (t1, nan)
            - (t1, t2)
            - (nan, t2)

        The issue is that for (t1, t2) variant the mean actually represents a larger population post launch. In fact this variant is
        representing all units exposed by following combinations:

            - (t1, t2)
            - (c1, t2)
            - (t1, c2)

        Therefore its count is roughly speaking three times the original count.
        The exact size of that population would be available in the trigger state count for the trigger state `(True, True)`.

        On the other hand (t1, nan) only represents

            - (t1, nan)
            - (c1, nan)

        Therefore its count is roughly speaking two times the original count.
        The exact size of that population would be available in the trigger state count for the trigger state `(True, False)`.
        The same logic applies to the `(nan, t2)` variant.

        For the input: sum, sum squared, and count are available in the statistics for each single variant in `variant_metric_stats_df`.
        The weights are available in `variant_count_df` which is indexed by `variant.value`. This enables us to get the count
        easily using `.at` method.

    Args:
        variant_list: A dataclass which includes a list of variants.
        variant_metric_stats_df: A dataframe which includes individual statistics
            for each variant value in its rows.
            We only require the columns:

            - `variant_col`
            - SUM_COL
            - SUM_SQ_COL
            - COUNT_COL

        This is because mean / variance etc can be calculated from these columns.
        variant_col: The column denoting the variant values in `variant_metric_stats_df`.
            DEFAULT: `VARIANT_COL`
        variant_count_df: A dataframe which will prescribe what "population counts" should be used when computing the statistics.
            The goal is to calculate mean / variance of a estimator corresponding to `variant_list`
            indexed by `variant_col` which must include the count column for each variant
            appearing in the `variant_list` (in it's `tigger_state_count_col` column).
        trigger_state_count_col: The column in `variant_count_df` which includes the count of the trigger states.
            This count is used as the population count during the calculations.
            DEFAULT: `TRIGGER_STATE_COUNT_COL`
    Result: A dataclass instance of `UnivarStats` containing the calculated statistics for the variant list.

    Raises:
        ValueError: If the variants in the variant list are not found in the possible
            variants in the data.
        ValueError: If the required columns are not present in `variant_metric_stats_df`.
        ValueError: If the required columns are not present in `variant_count_df`.

    """
    # Checks for required columns for `variant_metric_stats_df`.
    check_df_validity(
        df=variant_metric_stats_df,
        needed_cols=[variant_col, SAMPLE_COUNT_COL, SUM_COL, SUM_SQ_COL],
        err_trigger_source="calc_variant_list_metric_stats",
    )

    # Checks for required columns for `variant_count_df`.
    if variant_count_df is not None:
        check_df_validity(
            df=variant_count_df,
            needed_cols=[trigger_state_count_col],
            err_trigger_source="calc_variant_list_metric_stats",
        )

    variants = variant_list.variants
    variant_values = [variant.value for variant in variants]

    all_variants_set = set(variant_metric_stats_df[variant_col].values)
    # Raises a `ValueError` if the variants are not in the variants found in data.
    """
    if not set(variant_values).issubset(all_variants_set):
        raise ValueError(
            f"{variant_values} is not in the possible variants. "
            f"Possible variants: {all_variants_set}"
        )
    """
    nonexisting_variants = [v for v in variant_values if v not in all_variants_set]
    if nonexisting_variants:
        raise ValueError(
            f"\n*** Non-existing variants: {nonexisting_variants}."
            f"Possible variants: {all_variants_set}"
        )

    if variant_count_df is None:  # The case where no counts / weighst are given.
        idx = variant_metric_stats_df[variant_col].isin(variant_values)
        sample_count = variant_metric_stats_df[idx][SAMPLE_COUNT_COL].sum()
        sum0 = variant_metric_stats_df[idx][SUM_COL].sum()
        sum_sq = variant_metric_stats_df[idx][SUM_SQ_COL].sum()
        mean = sum0 / sample_count

        # We use this mathematical result: VARIANCE(X) = E(X^2) - E(X)^2 formula.
        var = (sum_sq / sample_count) - (mean**2)

        assert var >= 0, f"Variance is negative: {var}. This is not possible."

        sd = np.sqrt(var)
        sample_mean_var = var / sample_count
        univar_stats = UnivarStats(
            name=variant_list.name,
            mean=mean,
            sd=sd,
            var=var,
            sample_count=sample_count,
            sum=sum0,
            sum_sq=sum_sq,
            sample_mean_var=sample_mean_var,
            triggered_count=None,
            # Here the `triggered_count` is None
            # TODO: can we convert this `triggered_count` to `sample_count`?
            # It depends if the case where no trigger status is calculated
        )
    else:
        trigger_state_counts = [
            variant_count_df.at[v, trigger_state_count_col] for v in variant_values
        ]
        # Let's index the variant stats dataframe by the variant column as well.
        # Then we will extract the means and sums for the variants in the variant list in the same way.
        variant_metric_stats_df = variant_metric_stats_df.copy()
        variant_metric_stats_df.set_index(variant_col, inplace=True)
        sums = [variant_metric_stats_df.at[v, SUM_COL] for v in variant_values]
        sample_counts = [variant_metric_stats_df.at[v, SAMPLE_COUNT_COL] for v in variant_values]
        sample_count = sum(sample_counts)
        means = [s / sample_count for s, sample_count in zip(sums, sample_counts)]
        sum_sqs = [variant_metric_stats_df.at[v, SUM_SQ_COL] for v in variant_values]
        vars = [
            sum_sq / sample_count - (mean**2)
            for sum_sq, sample_count, mean in zip(sum_sqs, sample_counts, means)
        ]
        assert all(
            [not var < 0 for var in vars]
        ), f"Variance is negative for one of: {vars}. This is not possible."
        # We get the population size N by summing the trigger state counts.
        total_trigger_state_count = sum(trigger_state_counts)
        # We define the population weights as w_i = N_i/N where N_i is the trigger state count for the i-th variant.
        # Here N is the total trigger state count: `total_trigger_state_count`.
        weights = [ts_count / total_trigger_state_count for ts_count in trigger_state_counts]

        # The weighted mean is calculated as:
        # weighted_sample_mean = sum(w_i * mean_i)
        # Note that sum of the weights is 1.
        weighted_sample_mean = sum([w * mean for w, mean in zip(weights, means)])
        # The estimated population sum is calculated as:
        # updated_sum = N * weighted_sample_mean
        updated_sum = total_trigger_state_count * weighted_sample_mean
        # The variance of the weighted sample mean variance is:
        # VAR(sample_mean_var) = VAR(mu_hat) = VAR(sum w_i * mu_hat_i) = sum(w_i^2 * VAR(mu_hat_i)) = sum(w_i^2 * var_i / n_i)
        sample_mean_var = sum(
            [
                w**2 * var / sample_count
                for w, var, sample_count in zip(weights, vars, sample_counts)
            ]
        )
        univar_stats = UnivarStats(
            name=variant_list.name,
            mean=weighted_sample_mean,
            sd=None,  # This is skipped because there is no direct estimate of `sd`, and this quantity is not needed.
            var=None,  # This is skipped because there is no direct estimate of `var`, and this quantity is not needed.
            sample_count=sample_count,
            sum=updated_sum,  # This is included as it is useful to calculate the global (aka site-wide) impact.
            sum_sq=None,  # This is skipped because there is no direct estimate of `sum_sq`, and this quantity is not needed.
            sample_mean_var=sample_mean_var,
            triggered_count=total_trigger_state_count,
            # Here the "triggered count" is calculated using weights for trigger state of each variant
        )

    return univar_stats

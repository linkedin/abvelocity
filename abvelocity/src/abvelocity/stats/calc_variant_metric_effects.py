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

from dataclasses import asdict
from typing import Callable, Optional

import pandas as pd

from abvelocity.param.constants import (
    CATEG_NAN_VALUE,
    MEAN_COL,
    SAMPLE_COUNT_COL,
    SD_COL,
    TRIGGER_STATE_COUNT_COL,
)
from abvelocity.param.variant import ComparisonPair, Variant, VariantList
from abvelocity.stats.calc_variant_list_metric_stats import calc_variant_list_metric_stats
from abvelocity.stats.stats import TwoSampleTest
from abvelocity.stats.two_sample_z_test import two_sample_z_test
from abvelocity.utils.check_df_validity import check_df_validity


def calc_variant_metric_effects(
    variant_metric_stats_df: pd.DataFrame,
    variant_col: str,
    comparison_pairs: list[ComparisonPair],
    stats_test_func: Callable = two_sample_z_test,
    ci_coverage: float = 0.95,
    variant_count_df: Optional[pd.DataFrame] = None,
    trigger_state_count_col: str = TRIGGER_STATE_COUNT_COL,
) -> pd.DataFrame:
    """Given statistics for each variant in rows of `variant_metric_stats_df`,
        it compares the variants in `comparison_pairs` using the specified statistical test function.
        When `variant_count_df` is passed (a dataframe index by `variant.value)`,
        it calculates a test based on weighted statistics on the treatment and control arms.
        The weights are based on the population weights given in the `trigger_state_count_col` column.


    The assumptions are that

        - the two samples are independent of each other and
        - each sample is independent and identically distributed.

        Args:
            variant_metric_stats_df: A dataframe with each row representing the stats (mean, sd, count) for each variant.
            variant_col: The column name for the variant assignment in `variant_metric_stats_df`.
            comparison_pairs: A list of comparison pairs (`ComparisonPair`),
                where each comparison pair contains two variant lists (`VariantList`) to be compared.
            stats_test_func: The statistical test function to be used for comparing the variants.
            ci_coverage: The confidence level for the confidence intervals. Default is 0.95.
            variant_count_df: A dataframe which will prescribe what "population counts" should be used when computing the statistics.
                The goal is to calculate mean / variance of a estimator corresponding to `variant_list`
                indexed by `variant_col` which must include the count column for each variant
                appearing in the `variant_list` (in it's `tigger_state_count_col` column).
            trigger_state_count_col: The column in `variant_count_df` which includes the count of the trigger states.
                This count is used as the population count during the calculations.
                DEFAULT: `TRIGGER_STATE_COUNT_COL`

        Returns:
            result: A dataframe with each row representing the effect of a variant (list) compared to another.

        Raises:
            ValueError: If the required columns are not present in `variant_metric_stats_df`.
            ValueError: If the variants in `comparison_pairs` are not in the variants found in data.

        Alters:
            None.

    """
    # Checks for required columns.
    check_df_validity(
        df=variant_metric_stats_df,
        needed_cols=[variant_col, MEAN_COL, SD_COL, SAMPLE_COUNT_COL],
        err_trigger_source="calc_variant_metric_effects",
    )

    variant_effect_dict = {}

    for compare_pair in comparison_pairs:
        # print(f"\n *** calc_variant_metric_effects: compare_pair.treatment:\n {compare_pair.treatment}")
        # print(f"\n *** calc_variant_metric_effects: compare_pair.control:\n {compare_pair.control}")

        treatment_stats = calc_variant_list_metric_stats(
            variant_list=compare_pair.treatment,
            variant_metric_stats_df=variant_metric_stats_df,
            variant_col=variant_col,
            variant_count_df=variant_count_df,
            trigger_state_count_col=trigger_state_count_col,
        )

        control_stats = calc_variant_list_metric_stats(
            variant_list=compare_pair.control,
            variant_metric_stats_df=variant_metric_stats_df,
            variant_col=variant_col,
            variant_count_df=variant_count_df,
            trigger_state_count_col=trigger_state_count_col,
        )

        # print(f"calc_variant_metric_effects: treatment_stats: {treatment_stats}")
        # print(f"calc_variant_metric_effects: control_stats: {control_stats}")

        # TODO: change this to use mean and variance of sample mean.
        # This is needed for the weighted case.
        two_sample_test = TwoSampleTest(
            treatment_stats=treatment_stats, control_stats=control_stats, ci_coverage=ci_coverage
        )

        variant_effect_dict[compare_pair.name] = stats_test_func(two_sample_test)

    # Converts the effect dict to a dataframe.
    effect_records = []
    # Converts each variant effect to a variant effect record and appends to the list.
    for k, effect in variant_effect_dict.items():
        effect_dict = asdict(effect)
        effect_dict["comparison_pair"] = k
        effect_records.append(effect_dict)

    variant_effect_df = pd.DataFrame.from_records(effect_records)
    cols = list(variant_effect_df.columns)
    cols = cols[-1:] + cols[:-1]

    return variant_effect_df[cols]


def compare_variants_with_control(
    variant_metric_stats_df: pd.DataFrame,
    variant_col: str,
    expt_control: str | tuple[str],
    stats_test_func: Callable = two_sample_z_test,
    ci_coverage: float = 0.95,
    variant_count_df: Optional[pd.DataFrame] = None,
    trigger_state_count_col: str = TRIGGER_STATE_COUNT_COL,
) -> pd.DataFrame:
    """Given statistics for each variant in rows of `variant_metric_stats_df`,
        it compares all possible variants to a pre-specified control: `expt_control`.
        The comparison is done using the specified statistical test function.

    The assumptions are

        - the two samples are indepdendent of each other
        - each sample is independent and identically distributed.

    Args:
        variant_metric_stats_df: A dataframe with each row representing the
            stats (mean, sd, count) for each variant.
        variant_col: `float`, default 0.95
            A needed column in `variant_metric_stats_df` which includes,
            the variant assignment for units of the experiment.
        expt_control: The label for the control / baseline variant.
            This could be a string or a tuple of strings (in multi-expt setting).
                    ci_coverage: The confidence level for the confidence intervals. Default is 0.95.
        variant_count_df: A dataframe which will prescribe what "population counts" should be used when computing the statistics.
            The goal is to calculate mean / variance of a estimator corresponding to `variant_list`
            indexed by `variant_col` which must include the count column for each variant
            appearing in the `variant_list` (in it's `tigger_state_count_col` column).
        trigger_state_count_col: The column in `variant_count_df` which includes the count of the trigger states.
            This count is used as the population count during the calculations.
            DEFAULT: `TRIGGER_STATE_COUNT_COL`


    Returns:
        result: A dataframe with each row representing the effect of a variant compared to the control.

    Raises:
        None.

    Alters:
        None.
    """
    all_variants = variant_metric_stats_df[variant_col].values
    non_control_variants = [variant for variant in all_variants if variant != expt_control]

    # We only pick the variants which do not include `CATEG_NAN_VALUE` which is nan
    if isinstance(expt_control, tuple):  # multi-experiment case
        non_nan_variants = [
            variant for variant in non_control_variants if CATEG_NAN_VALUE not in variant
        ]
    else:  # the case where each variant can a string (e.g. univar experiment)
        non_nan_variants = [
            variant for variant in non_control_variants if CATEG_NAN_VALUE != variant
        ]

    comparison_pairs = [
        ComparisonPair(
            treatment=VariantList(variants=[Variant(value=value)]),
            control=VariantList(variants=[Variant(value=expt_control)]),
            name=(value, expt_control),
        )
        for value in non_nan_variants
    ]

    return calc_variant_metric_effects(
        variant_metric_stats_df=variant_metric_stats_df,
        variant_col=variant_col,
        comparison_pairs=comparison_pairs,
        stats_test_func=stats_test_func,
        ci_coverage=ci_coverage,
        variant_count_df=variant_count_df,
        trigger_state_count_col=trigger_state_count_col,
    )

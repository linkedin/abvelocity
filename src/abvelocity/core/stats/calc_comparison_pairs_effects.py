# BSD 2-CLAUSE LICENSE
# Copyright 2024, Blah Corporation. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
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

import warnings
from typing import List, Optional

import pandas as pd
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.get_data.get_expt_stats import get_expt_stats
from abvelocity.core.param.constants import CI_PERCENT_COL, TRIGGER_STATE_COUNT_COL
from abvelocity.core.param.metric import Metric
from abvelocity.core.param.variant import ComparisonPair
from abvelocity.core.stats.cpde_constructor import CPDEConstructor
from abvelocity.core.stats.param import StrataInfo
from abvelocity.core.utils.check_df_validity import check_df_validity

# Index constants for calc_jk_stats output with diff_type="both"
SUM_NUMER_INDEX = 0
MEAN_INDEX = 1
SUM_DENOM_INDEX = 2
MEAN_DENOM_INDEX = 3
SUM_RATIO_INDEX = 4
MEAN_RATIO_INDEX = 5
SIMPLE_DIFF_LENGTH_SIMPLE = 2  # [sum_numer, mean_numer]
SIMPLE_DIFF_LENGTH_RATIO = 6  # [sum_numer, mean_numer, sum_denom, mean_denom, sum_ratio, mean_ratio]
MEAN_PCNT_INDEX_SIMPLE = SIMPLE_DIFF_LENGTH_SIMPLE + 1  # mean_numer_percent
MEAN_PCNT_INDEX_RATIO = SIMPLE_DIFF_LENGTH_RATIO + 5  # mean_ratio_percent


def calc_comparison_pairs_effects(
    dc: DataContainer,
    variant_col: str,
    metric: Metric,
    comparison_pairs: List[ComparisonPair],
    ci_coverage: float = 0.95,
    variant_count_df: Optional[pd.DataFrame] = None,
    trigger_state_count_col: str = TRIGGER_STATE_COUNT_COL,
    num_buckets: Optional[int] = 20,
) -> pd.DataFrame:
    """Computes variant effects for comparison pairs using CPDEConstructor on raw data.
    Returns both simple and percent differences for the specified metric along with their confidence intervals.

    Assumptions:
        - The two samples (control and treatment) are independent.
        - Each sample is independent and identically distributed.

    Args:
        df: Raw DataFrame with columns including `variant_col` and metric columns.
        variant_col: Column name for variant assignments (e.g., 'variant').
        metric: Metric object defining the metric (simple or ratio).
        comparison_pairs: List of `ComparisonPair` objects with control and treatment `VariantList`.
        ci_coverage: Confidence level for confidence intervals (default: 0.95).
        variant_count_df: Optional DataFrame indexed by variant values, with `trigger_state_count_col`
            for impacted_counts and a 'variant' column for sample_counts.
        trigger_state_count_col: Column in `variant_count_df` with trigger state counts
            (default: TRIGGER_STATE_COUNT_COL).
        num_buckets: The number of groups (B) for jackknife iterations. Defaults to 20.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - comparison_pair: Name of the comparison pair.
            - delta: Mean difference (simple_diff).
            - delta_percent: Percent difference (pcnt_diff).
            - ci: Confidence interval for delta as a tuple (lower, upper).
            - CI_PERCENT_COL: Confidence interval for delta_percent as a tuple (lower, upper).
            - p_value: P-value for delta (from jackknife).
            - delta_sum: Sum of differences (float for simple metrics, tuple for ratio metrics).
            - impacted_counts: Tuple of (control, treatment) trigger state counts.
            - sample_counts: Tuple of (control, treatment) sample counts.

    Raises:
        ValueError: If required columns are missing in `df` or `variant_count_df`.
        ValueError: If no variants in a comparison pair are found in `df` or `strata_info`.
    """
    df = dc.pandas_df
    # Validate metric columns in df
    metric_columns = [metric.numerator.col]
    if metric.denominator is not None:
        metric_columns.append(metric.denominator.col)
    needed_cols = [variant_col] + metric_columns
    check_df_validity(
        df=df,
        needed_cols=needed_cols,
        err_trigger_source="calc_comparison_pairs_effects",
    )

    # Create StrataInfo
    if variant_count_df is not None:
        check_df_validity(
            df=variant_count_df,
            needed_cols=[trigger_state_count_col],
            err_trigger_source="calc_comparison_pairs_effects",
        )
        strata_df = variant_count_df[[trigger_state_count_col]].copy()
    else:
        expt_stats_df = get_expt_stats(dc=dc)
        strata_df = expt_stats_df.variant_count_df[[trigger_state_count_col]].copy()

    strata_info = StrataInfo(df=strata_df, strata_count_col=trigger_state_count_col)

    # Validate variants
    available_variants = set(df[variant_col])
    available_strata_variants = set(strata_info.df.index)
    for cp in comparison_pairs:
        control_variants = [v.value for v in cp.control.variants]
        treatment_variants = [v.value for v in cp.treatment.variants]
        control_found_df = any(v in available_variants for v in control_variants)
        treatment_found_df = any(v in available_variants for v in treatment_variants)
        control_found_strata = any(v in available_strata_variants for v in control_variants)
        treatment_found_strata = any(v in available_strata_variants for v in treatment_variants)

        if not (control_found_df or treatment_found_df):
            raise ValueError(f"No variants in comparison pair {cp.name} found in {variant_col} column.")
        if not (control_found_strata or treatment_found_strata):
            raise ValueError(f"No variants in comparison pair {cp.name} found in strata_info.")

        for variant in cp.control.variants + cp.treatment.variants:
            if variant.value not in available_variants:
                warnings.warn(f"Variant {variant.value} in comparison pair {cp.name} not found in {variant_col} column.")
            if variant.value not in available_strata_variants:
                warnings.warn(f"Variant {variant.value} in comparison pair {cp.name} not found in strata_info.")

    # Determine if ratio metric
    is_ratio_metric = metric.denominator is not None
    delta_index = MEAN_RATIO_INDEX if is_ratio_metric else MEAN_INDEX
    mean_pcnt_index = MEAN_PCNT_INDEX_RATIO if is_ratio_metric else MEAN_PCNT_INDEX_SIMPLE

    effect_records = []

    for cp in comparison_pairs:
        # Initialize CPDEConstructor with diff_type="both"
        print(f"\n*** In `calc_comparison_pairs_effects`, jk for metric.name {metric.name}" f" and comparison pair: {cp.name}")
        cpde = CPDEConstructor(
            metric=metric,
            strata_info=strata_info,
            comparison_pair=cp,
            diff_type="both",
            variant_col=variant_col,
            name=cp.name,
        )
        estimator = cpde.construct()
        jk_result = estimator.calc_jk_stats(
            df=df,
            num_buckets=num_buckets,
            ci_coverage=ci_coverage,
        )

        # Extract metrics
        delta = jk_result["estimator_value"][delta_index]
        delta_percent = jk_result["estimator_value"][mean_pcnt_index]
        ci = tuple(jk_result["ci"][delta_index])  # Convert to tuple
        ci_percent = tuple(jk_result["ci"][mean_pcnt_index])  # Convert to tuple
        p_value = jk_result["p_values"][delta_index]  # Use p-value from jackknife

        # Delta sum
        delta_sum = (
            (
                jk_result["estimator_value"][SUM_NUMER_INDEX],
                jk_result["estimator_value"][SUM_DENOM_INDEX],
            )
            if is_ratio_metric
            else jk_result["estimator_value"][SUM_NUMER_INDEX]
        )

        # Counts for both arms
        control_variants = [v.value for v in cp.control.variants]
        treatment_variants = [v.value for v in cp.treatment.variants]
        impacted_counts_control = sum(strata_info.df.at[v, trigger_state_count_col] for v in control_variants if v in strata_info.df.index)
        impacted_counts_treatment = sum(strata_info.df.at[v, trigger_state_count_col] for v in treatment_variants if v in strata_info.df.index)
        # These counts typically should be identical if a reasonable `ComparisonPair` is passed.
        # This is because we expect the two arms target / impact the same units in order to be comparable.
        # Therefore this value can serve as an error catching value as well.
        impacted_counts = (impacted_counts_control, impacted_counts_treatment)

        if variant_count_df is not None and "variant" in variant_count_df.columns:
            sample_counts_control = variant_count_df[variant_count_df.index.isin(control_variants)]["variant"].sum()
            sample_counts_treatment = variant_count_df[variant_count_df.index.isin(treatment_variants)]["variant"].sum()
        else:
            sample_counts_control = len(df[df[variant_col].isin(control_variants)])
            sample_counts_treatment = len(df[df[variant_col].isin(treatment_variants)])
        sample_counts = (sample_counts_control, sample_counts_treatment)

        # Create record
        effect_records.append(
            {
                "comparison_pair": cp.name,
                "delta": delta,
                "delta_percent": delta_percent,
                "ci": ci,
                CI_PERCENT_COL: ci_percent,
                "p_value": p_value,
                "delta_sum": delta_sum,
                "impacted_counts": impacted_counts,
                "sample_counts": sample_counts,
            }
        )

    # Create output DataFrame
    result_df = pd.DataFrame(effect_records)
    if not result_df.empty:
        cols = [
            "comparison_pair",
            "delta",
            "delta_percent",
            "ci",
            CI_PERCENT_COL,
            "delta_sum",
            "p_value",
            "impacted_counts",
            "sample_counts",
        ]
        result_df = result_df[cols]

    return result_df

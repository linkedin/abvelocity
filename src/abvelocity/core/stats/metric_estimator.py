# BSD 2-CLAUSE LICENSE
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
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

from typing import Any, Callable

import numpy as np
import pandas as pd
from abvelocity.core.param.metric import Metric
from abvelocity.core.stats.constants import AGG_MAP, STANDARD_VALUES_NUMER, STANDARD_VALUES_RATIO
from abvelocity.core.stats.estimator import Estimator


class MetricEstimator(Estimator):
    """
    An Estimator based on a Metric definition, computing simple metrics or ratios.

    The input DataFrame is assumed to be aggregated at the unit level, so the Metric's
    numerator and denominator UMetric objects directly reference columns in the DataFrame,
    and their within-unit aggregation (e.g., 'MAX') is ignored. Only across-unit aggregations
    (numerator_agg, denominator_agg) are applied.

    If sample_count is provided and the aggregation is 'MEAN' or 'AVG', the mean is computed as
    sum(column)/sum(sample_count.col). Supports filtering the DataFrame using a condition
    (string or callable) passed as param.

    Attributes:
        metric: The Metric definition specifying numerator, denominator, and aggregations.
        compute_standard_values: If True, computes standard values as defined in
            STANDARD_VALUES_NUMER or STANDARD_VALUES_RATIO.
        standard_names: List of standard value names, set based on presence of denominator
            when compute_standard_values is True.
    """

    def __init__(self, metric: Metric, compute_standard_values: bool = False, param: Any = None):
        """
        Initializes the MetricEstimator with a Metric definition and optional condition.

        Args:
            metric: The Metric definition for the estimator.
            compute_standard_values: If True, there are two cases:
                - (1) There is no denominator: Computes:
                    - sum_numer: Sum of numerator values.
                    - mean_numer: Mean of numerator values (weighted by sample_count if provided).
                - (2) There is a denominator: Computes:
                    - sum_numer: Sum of numerator values.
                    - mean_numer: Mean of numerator values (weighted by sample_count if provided).
                    - sum_denom: Sum of denominator values.
                    - mean_denom: Mean of denominator values (weighted by sample_count if provided).
                    - sum_ratio: Ratio of sums, sum(numerator)/sum(denominator).
                    - mean_ratio: Ratio of means, mean(numerator)/mean(denominator).
                In such a case, we also attach self.standard_names to the instance:
                    (1) - STANDARD_VALUES_NUMER = ["sum_numer", "mean_numer"]
                    (2) - STANDARD_VALUES_RATIO = ["sum_numer", "mean_numer", "sum_denom",
                                                  "mean_denom", "sum_ratio", "mean_ratio"]
            param: Optional condition to filter the DataFrame (e.g., string for df.query or callable).
        """
        super().__init__(param=param, name=metric.name)
        self.metric = metric
        self.compute_standard_values = compute_standard_values
        # Set standard_names based on presence of denominator
        self.standard_names = (
            STANDARD_VALUES_RATIO
            if self.compute_standard_values and metric.denominator is not None
            else STANDARD_VALUES_NUMER if self.compute_standard_values else None
        )

    def estimator_func(self, df: pd.DataFrame, param: Any = None) -> np.ndarray:
        """
        Computes the estimator value based on the Metric definition.

        Assumes the input DataFrame is aggregated at the unit level. Applies a filter (if param
        is provided), then aggregates the numerator and denominator columns across units using
        numerator_agg and denominator_agg. For 'MEAN' or 'AVG' aggregation with sample_count,
        computes sum(column)/sum(sample_count.col). When compute_standard_values is False, returns
        a single-element NumPy array with numerator (no denominator) or ratio (numerator/denominator).
        When compute_standard_values is True, returns a NumPy array containing values specified in
        STANDARD_VALUES_NUMER (no denominator) or STANDARD_VALUES_RATIO (with denominator).
        Missing values are filled with 0.

        Args:
            df: Input DataFrame with unit-level aggregated data, including 'unit' and metric columns.
            param: Optional condition (e.g., string for df.query or callable) to filter the DataFrame.
                If None, uses self.param. If both are None, no filtering is applied.

        Returns:
            np.ndarray: Array of computed values. If compute_standard_values is False, contains a single
                value (numerator or ratio). If compute_standard_values is True, contains values in the order
                defined by STANDARD_VALUES_NUMER (no denominator) or STANDARD_VALUES_RATIO (with denominator).

        Raises:
            ValueError: If denominator aggregates to zero in a ratio metric when compute_standard_values is False.
            NotImplementedError: If aggregation function is not supported.
            ValueError: If string condition is invalid for df.query or param is unsupported.
            KeyError: If required columns are missing in the DataFrame.
            ValueError: If sample count sum is zero when used for MEAN or AVG aggregation.
        """
        # Use param if provided, else fall back to self.param
        condition = param if param is not None else self.param

        # Apply filtering if condition is provided
        df_filtered = df.copy()
        if condition is not None:
            if isinstance(condition, str):
                try:
                    df_filtered = df_filtered.query(condition)
                except Exception as e:
                    raise ValueError(f"Invalid query condition: {condition}. Error: {str(e)}")
            elif isinstance(condition, Callable):
                df_filtered = condition(df_filtered)
            else:
                raise ValueError(f"Condition must be a string or callable, got {type(condition)}")

        # Compute numerator
        num_col = self.metric.numerator.col
        if num_col not in df_filtered.columns:
            raise KeyError(f"Numerator column '{num_col}' not found in DataFrame")
        num_series = df_filtered[num_col].fillna(0)  # Fill NaN with 0

        # Handle MEAN or AVG aggregation with sample_count for numerator
        if self.metric.numerator_agg in ("MEAN", "AVG") and self.metric.sample_count is not None:
            sample_col = self.metric.sample_count.col
            if sample_col not in df_filtered.columns:
                raise KeyError(f"Sample count column '{sample_col}' not found in DataFrame")
            sample_series = df_filtered[sample_col].fillna(0)
            if sample_series.sum() == 0:
                raise ValueError("Sample count sum is zero for numerator")
            numerator = num_series.sum() / sample_series.sum()
        else:
            num_agg_func = AGG_MAP.get(self.metric.numerator_agg)
            if num_agg_func is None:
                raise NotImplementedError(f"Numerator aggregation {self.metric.numerator_agg} not supported")
            numerator = num_agg_func(num_series)

        # Handle standard values case
        if self.compute_standard_values:
            # Compute mean_numer, using weighted mean if sample_count is provided and aggregation is MEAN or AVG
            if self.metric.numerator_agg in ("MEAN", "AVG") and self.metric.sample_count is not None:
                sample_col = self.metric.sample_count.col
                if sample_col not in df_filtered.columns:
                    raise KeyError(f"Sample count column '{sample_col}' not found in DataFrame")
                sample_series = df_filtered[sample_col].fillna(0)
                if sample_series.sum() == 0:
                    raise ValueError("Sample count sum is zero for numerator")
                mean_numer = num_series.sum() / sample_series.sum()
            else:
                mean_numer = AGG_MAP["MEAN"](num_series)

            # Initialize result list for standard values
            result = [
                float(AGG_MAP["SUM"](num_series)),  # sum_numer
                float(mean_numer),  # mean_numer
            ]

            # Handle denominator if present
            if self.metric.denominator is not None:
                denom_col = self.metric.denominator.col
                if denom_col not in df_filtered.columns:
                    raise KeyError(f"Denominator column '{denom_col}' not found in DataFrame")
                denom_series = df_filtered[denom_col].fillna(0)  # Fill NaN with 0

                # Handle MEAN or AVG aggregation with sample_count for denominator
                if self.metric.denominator_agg in ("MEAN", "AVG") and self.metric.sample_count is not None:
                    sample_col = self.metric.sample_count.col
                    if sample_col not in df_filtered.columns:
                        raise KeyError(f"Sample count column '{sample_col}' not found in DataFrame")
                    sample_series = df_filtered[sample_col].fillna(0)
                    if sample_series.sum() == 0:
                        raise ValueError("Sample count sum is zero for denominator")
                    denominator = denom_series.sum() / sample_series.sum()
                else:
                    denom_agg_func = AGG_MAP.get(self.metric.denominator_agg)
                    if denom_agg_func is None:
                        raise NotImplementedError(f"Denominator aggregation {self.metric.denominator_agg} not supported")
                    denominator = denom_agg_func(denom_series)

                # Compute ratio of sums and means, handling zero denominators
                sum_denom = float(AGG_MAP["SUM"](denom_series))
                mean_denom = float(AGG_MAP["MEAN"](denom_series))
                sum_ratio = float(num_series.sum() / sum_denom if sum_denom != 0 else 0)
                mean_ratio = float(num_series.mean() / mean_denom if mean_denom != 0 else 0)

                result.extend(
                    [
                        sum_denom,  # sum_denom
                        mean_denom,  # mean_denom
                        sum_ratio,  # sum_ratio
                        mean_ratio,  # mean_ratio
                    ]
                )
            return np.array(result)

        # Default case: return single-element array
        if self.metric.denominator is None:
            return np.array([numerator])

        # Compute denominator for default ratio case
        denom_col = self.metric.denominator.col
        if denom_col not in df_filtered.columns:
            raise KeyError(f"Denominator column '{denom_col}' not found in DataFrame")
        denom_series = df_filtered[denom_col].fillna(0)  # Fill NaN with 0

        if self.metric.denominator_agg in ("MEAN", "AVG") and self.metric.sample_count is not None:
            sample_col = self.metric.sample_count.col
            if sample_col not in df_filtered.columns:
                raise KeyError(f"Sample count column '{sample_col}' not found in DataFrame")
            sample_series = df_filtered[sample_col].fillna(0)
            if sample_series.sum() == 0:
                raise ValueError("Sample count sum is zero for denominator")
            denominator = denom_series.sum() / sample_series.sum()
        else:
            denom_agg_func = AGG_MAP.get(self.metric.denominator_agg)
            if denom_agg_func is None:
                raise NotImplementedError(f"Denominator aggregation {self.metric.denominator_agg} not supported")
            denominator = denom_agg_func(denom_series)

        # Check for zero denominator in default case
        if denominator == 0:
            raise ValueError("Denominator aggregated to zero in ratio metric")

        return np.array([numerator / denominator])

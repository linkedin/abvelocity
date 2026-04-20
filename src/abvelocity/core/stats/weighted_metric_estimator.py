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

import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from abvelocity.core.param.metric import Metric
from abvelocity.core.stats.constants import AGG_MAP, STANDARD_VALUES_NUMER, STANDARD_VALUES_RATIO
from abvelocity.core.stats.estimator import Estimator
from abvelocity.core.stats.param import StrataInfo


class WeightedMetricEstimator(Estimator):
    """
    An Estimator that computes weighted standard values for a Metric across strata.

    - This estimator uses 'strata population counts' from StrataInfo to compute standardized
      estimates. The weights are standardized (normalized) to sum to 1.
    - Supports `sample_count` for weighted 'mean's within strata (`sum(column)/sum(sample_count.col)`).
    - The 'mean' metrics (`mean_numer`, `mean_denom`) are computed as a weighted average
      of the stratum 'mean's, using the 'standardized weights' (`self.weights`).
    - The 'sum' metrics (`sum_numer`, `sum_denom`) are estimated by multiplying the
      'overall total population count' (stored in `self.total_pop_count`) by the respective
      'overall weighted mean': `sum_metric = total_pop_count * weighted_mean_metric`.
    - For 'ratio' metrics, `sum_ratio` is the ratio of the estimated total 'sum's, and
      `mean_ratio` is the ratio of the overall weighted 'mean's.
    - Applies filtering via `param` within each stratum.

    Attributes:
        metric: The Metric definition specifying numerator, denominator, and aggregations.
        strata_info: StrataInfo with population counts for each variant.
        variant_values: List of variant values to compute estimates for.
        variant_col: Column name in the DataFrame containing variant identifiers.
        param: Optional condition to filter the DataFrame within each stratum.
        standard_names: List of standard value names (`STANDARD_VALUES_NUMER` or `STANDARD_VALUES_RATIO`).
        weights: Pre-calculated 'standardized (normalized) population weights' for each variant (sum=1).
        total_pop_count: The sum of all strata population counts (the total target population size).
    """

    def __init__(
        self,
        metric: Metric,
        strata_info: StrataInfo,
        variant_values: List[Any],
        variant_col: str = "variant",
        param: Any = None,
        name: Optional[str] = None,
    ):
        """
        Initializes the WeightedMetricEstimator with a Metric and strata information.

        Args:
            metric: The Metric definition for the estimator.
            strata_info: StrataInfo with population counts for each variant.
            variant_values: List of variant values to compute estimates for.
            variant_col: Column name in the DataFrame containing variant identifiers.
            param: Optional condition (string or callable) to filter the DataFrame within each stratum.
            name: Optional name for the estimator; defaults to metric.name.

        Raises:
            ValueError: If strata_info is incomplete or variant_values is empty.
        """
        super().__init__(param=param, name=name or metric.name)
        self.metric = metric
        self.strata_info = strata_info
        self.variant_values = variant_values
        self.variant_col = variant_col
        self.standard_names = STANDARD_VALUES_RATIO if metric.denominator is not None else STANDARD_VALUES_NUMER
        self.weights = None  # Standardized weights (sum=1)
        self.total_pop_count = 0.0
        self.calc_weights_and_total_count(strata_info, variant_values)

    def calc_weights_and_total_count(self, strata_info: StrataInfo, variant_values: List[Any]) -> None:
        """
        Calculates and stores the standardized weights and the overall total population count.

        Sets `self.weights` as a list of 'standardized weights' (raw count / total count).
        Sets `self.total_pop_count` as the 'sum' of raw counts.
        Assigns zero weight/count to missing variant_values with a warning.

        Args:
            strata_info: StrataInfo with population counts.
            variant_values: List of variant values.

        Raises:
            ValueError: If strata_info is incomplete or variant_values is empty.
        """
        if not variant_values:
            raise ValueError("variant_values must not be empty")
        if strata_info.df is None or strata_info.strata_count_col is None:
            raise ValueError("StrataInfo is incomplete: missing df or strata_count_col")

        strata_count_col = strata_info.strata_count_col
        df = strata_info.df
        raw_counts = []
        for v in variant_values:
            try:
                # Raw population counts
                count = df.at[v, strata_count_col]
                raw_counts.append(float(count))
            except KeyError:
                warnings.warn(
                    f"Variant value '{v}' not found in the index of strata_info.df. Assigning a zero weight (population count).",
                    UserWarning,
                )
                raw_counts.append(0.0)

        # Store the overall total population count
        self.total_pop_count = sum(raw_counts)

        if self.total_pop_count == 0:
            # If total count is zero, weights are all zero
            self.weights = [0.0] * len(raw_counts)
        else:
            # Standardized weights: raw_count / total_pop_count (sum=1)
            self.weights = [count / self.total_pop_count for count in raw_counts]

    def estimator_func(self, df: pd.DataFrame, param: Any = None) -> np.ndarray:
        """
        Computes weighted standard values for the Metric across strata.

        The 'mean' metrics are computed as a weighted average of the stratum 'mean's,
        using the standardized weights (`self.weights`).
        The 'sum' metrics are estimated as `self.total_pop_count * weighted_mean_metric`.

        Args:
            df: Input DataFrame with unit-level aggregated data, including `variant_col`
                and metric columns.
            param: Optional condition (string or callable) to filter the DataFrame within each stratum.
                If None, uses `self.param`.

        Returns:
            np.ndarray: Array of weighted standard values (`STANDARD_VALUES_NUMER` or `STANDARD_VALUES_RATIO`).

        Raises:
            ValueError: If weights or variant_values are not defined, or `variant_col` is missing.
            KeyError: If required columns are missing in the DataFrame.
            ValueError: If sample_count 'sum' is zero for `MEAN` or `AVG` aggregation.
        """
        if self.weights is None or self.variant_values is None:
            raise ValueError("Weights or variant_values are not defined")
        if self.variant_col not in df.columns:
            raise KeyError(f"Variant column '{self.variant_col}' not found in DataFrame")

        condition = param if param is not None else self.param
        total_pop_count = self.total_pop_count

        if total_pop_count == 0:
            num_results = 6 if self.metric.denominator is not None else 2
            return np.array([0.0] * num_results)

        # 1. Initialize weighted mean metrics
        weighted_mean_numer = 0.0
        weighted_mean_denom = 0.0

        # 2. Iterate and calculate the overall weighted means using standardized weights
        for v, weight in zip(self.variant_values, self.weights):
            if weight == 0.0:
                continue

            sub_df = df[df[self.variant_col] == v].copy()

            # Apply filter
            if isinstance(condition, str):
                try:
                    sub_df = sub_df.query(condition)
                except Exception as e:
                    raise ValueError(f"Invalid query condition: {condition}. Error: {str(e)}")

            if sub_df.empty:
                warnings.warn(
                    f"Stratum '{v}' in column '{self.variant_col}' is empty or empty after filter. Using zero for estimates.",
                    UserWarning,
                )
                continue

            # --- Numerator Calculations ---
            num_col = self.metric.numerator.col
            if num_col not in sub_df.columns:
                raise KeyError(f"Numerator column '{num_col}' not found in DataFrame")
            num_series = sub_df[num_col].fillna(0)

            # Compute stratum_mean_numer
            if self.metric.numerator_agg in ("MEAN", "AVG") and self.metric.sample_count is not None:
                sample_col = self.metric.sample_count.col
                if sample_col not in sub_df.columns:
                    raise KeyError(f"Sample count column '{sample_col}' not found in DataFrame")
                sample_series = sub_df[sample_col].fillna(0)
                if sample_series.sum() == 0:
                    raise ValueError(f"Sample count sum is zero for numerator in stratum '{v}'")
                stratum_mean_numer = num_series.sum() / sample_series.sum()
            else:
                stratum_mean_numer = AGG_MAP["MEAN"](num_series) if not num_series.empty else 0.0

            # Accumulate weighted mean_numer: sum(weight * mean)
            weighted_mean_numer += weight * stratum_mean_numer

            # --- Denominator Calculations (if applicable) ---
            if self.metric.denominator is not None:
                denom_col = self.metric.denominator.col
                if denom_col not in sub_df.columns:
                    raise KeyError(f"Denominator column '{denom_col}' not found in DataFrame")
                denom_series = sub_df[denom_col].fillna(0)

                # Compute stratum_mean_denom
                if self.metric.denominator_agg in ("MEAN", "AVG") and self.metric.sample_count is not None:
                    sample_col = self.metric.sample_count.col
                    if sample_col not in sub_df.columns:
                        raise KeyError(f"Sample count column '{sample_col}' not found in DataFrame")
                    sample_series = sub_df[sample_col].fillna(0)
                    if sample_series.sum() == 0:
                        raise ValueError(f"Sample count sum is zero for denominator in stratum '{v}'")
                    stratum_mean_denom = denom_series.sum() / sample_series.sum()
                else:
                    stratum_mean_denom = AGG_MAP["MEAN"](denom_series) if not denom_series.empty else 0.0

                # Accumulate weighted mean_denom: sum(weight * mean)
                weighted_mean_denom += weight * stratum_mean_denom

        # 3. Estimate Sum Metrics: total_pop_count * weighted_mean
        estimated_sum_numer = total_pop_count * weighted_mean_numer

        # Results start with (Estimated Sum Numerator, Weighted Mean Numerator)
        result = [estimated_sum_numer, weighted_mean_numer]

        if self.metric.denominator is not None:
            estimated_sum_denom = total_pop_count * weighted_mean_denom

            # Ratio of estimated sums
            sum_ratio = estimated_sum_numer / estimated_sum_denom if estimated_sum_denom != 0 else 0.0
            # Ratio of weighted means
            mean_ratio = weighted_mean_numer / weighted_mean_denom if weighted_mean_denom != 0 else 0.0

            # Results extend with (Estimated Sum Denominator, Weighted Mean Denominator, Sum Ratio, Mean Ratio)
            result.extend([estimated_sum_denom, weighted_mean_denom, sum_ratio, mean_ratio])

        return np.array(result)

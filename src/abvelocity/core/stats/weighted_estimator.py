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
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import warnings
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from abvelocity.core.stats.estimator import Estimator
from abvelocity.core.stats.param import StrataInfo


class WeightedEstimator(Estimator):
    def __init__(
        self,
        stratum_estimator: "Estimator",
        strata_info: StrataInfo = None,
        variant_values: Optional[List[Any]] = None,
        variant_col: Optional[str] = "variant",
        standardize_weights: bool = True,
        param: Any = None,
        name: Optional[str] = None,
    ):
        """
        Initializes the WeightedEstimator, which computes a weighted average of an
        underlying estimator across different strata (variants).
        """
        super().__init__(param=None, name=name)
        self.stratum_estimator = stratum_estimator
        self.strata_info = strata_info
        self.variant_values = variant_values
        self.variant_col = variant_col
        self.standardize_weights = standardize_weights

        self.weights: Optional[List[float]] = None

        # Pre-calculate weights if strata info is available
        if self.variant_values is not None and self.strata_info is not None:
            self.calc_weights(self.strata_info, self.variant_values)

    def calc_weights(self, strata_info: StrataInfo, variant_values: List[Any]):
        """
        Calculates and sets self.weights based on strata_info.
        - Handles missing variants in strata_info.df by assigning a 0 weight and warning.
        """
        if strata_info.df is None or strata_info.strata_count_col is None:
            warnings.warn(
                "StrataInfo is incomplete (missing df or strata_count_col). Weights are set to None.",
                UserWarning,
            )
            self.weights = None
            return

        strata_count_col = strata_info.strata_count_col
        df = strata_info.df

        # Collect counts, using 0 if the variant is missing from the strata_info index
        strata_counts = []
        for v in variant_values:
            try:
                # Use .at for fast scalar lookup
                count = df.at[v, strata_count_col]
                strata_counts.append(count)
            except KeyError:
                warnings.warn(
                    f"Variant value '{v}' not found in the index of strata_info.df. Assigning a zero weight.",
                    UserWarning,
                )
                strata_counts.append(0.0)

        total_strata_count = sum(strata_counts)

        if total_strata_count == 0:
            # Handle case where all counts are zero to avoid division by zero
            self.weights = [0.0] * len(strata_counts)
        elif self.standardize_weights:
            # We define the population weights as w_i = N_i/N
            self.weights = [s_count / total_strata_count for s_count in strata_counts]
        else:
            # Non-standardized weights (raw counts)
            self.weights = strata_counts

    def estimator_func(self, df: pd.DataFrame, param: Any = None) -> Union[float, np.ndarray]:
        """
        Computes the weighted average of the stratum_estimator across all variants.

        - If a strata's sub_df is empty, the estimate is assumed to be the scalar value 0.
        """
        if self.weights is None or self.variant_values is None:
            raise ValueError(
                "Weights or variant_values are not defined. Ensure strata_info and variant_values " "were successfully provided during initialization."
            )

        if self.variant_col is None:
            raise ValueError("variant_col must be set to filter the DataFrame.")

        # Initialize the final estimate accumulation variable
        estimator_value = 0

        # Iterate over each variant and its corresponding pre-calculated weight
        for i, v in enumerate(self.variant_values):
            weight = self.weights[i]

            # Simplified: Use a direct boolean mask for filtering.
            sub_df = df[df[self.variant_col] == v]

            if sub_df.empty:
                # If the stratum is empty, use the scalar 0 and warn.
                warnings.warn(
                    f"Strata '{v}' in column '{self.variant_col}' is empty in the input DataFrame. " "Estimator value for this strata assumed to be zero.",
                    UserWarning,
                )
                stratum_estimate = 0
            else:
                # We call `estimator_func_with_param` to apply any inherent parameters
                # of the stratum_estimator.
                stratum_estimate = self.stratum_estimator.estimator_func_with_inferred_param(df=sub_df)

            # Accumulate the weighted estimate using the compound assignment operator (+=)
            estimator_value += weight * stratum_estimate

        # Return the accumulated result
        return np.array(estimator_value)

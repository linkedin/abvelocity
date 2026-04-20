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

from typing import Optional

from abvelocity.core.param.metric import Metric
from abvelocity.core.param.variant import ComparisonPair
from abvelocity.core.stats.diff_estimator_constructor import DiffEstimatorConstructor
from abvelocity.core.stats.estimator import Estimator
from abvelocity.core.stats.param import StrataInfo
from abvelocity.core.stats.weighted_metric_estimator import WeightedMetricEstimator


class CPDEConstructor:
    """
    Comparison Pair Diff Estimator.
    Takes a Metric and ComparisonPair and constructs an estimator.
    Here the estimator we use for each arm is `WeightedMetricEstimator` as it computes
    standard values for all metrics.
    Note the complexity that each arm itself consists of a few variants and as a result
    weighted estimation in envoked.

    The weighted estimation using  strata_info is necessary because the variant weights may not directly
    apply.

    For example for the MEA use case with two experiments, when both of them trigger: the relative count

        (counts of the variant combinations) / (counts of the triggering state)

    is different from when only one of them trigger. Therefore in some sense this combination represent
    a bigger weight than the variant combination will imply and we need to consider the trigger state count.
    """

    def __init__(
        self,
        metric: Metric,
        strata_info: StrataInfo,
        comparison_pair: ComparisonPair,
        diff_type: str = "both",
        variant_col: str = "variant",
        name: Optional[str] = None,
    ):
        """
        Initializes the class.

        Args:
            metric: Metric object defining the metric to estimate.
            strata_info: StrataInfo for weighted estimation.
            comparison_pair: ComparisonPair defining control and treatment variant_values.
            diff_type: Type of difference ('simple_diff', 'pcnt_diff', 'both').
            variant_col: Column name for variant in DataFrame.
            name: Optional name for the estimator.
        """
        self.metric = metric
        self.strata_info = strata_info
        self.comparison_pair = comparison_pair
        self.diff_type = diff_type
        self.variant_col = variant_col
        self.name = name or metric.name

    def construct(self) -> Estimator:
        """
        Constructs the diff estimator.

        Returns:
            Estimator: Configured difference estimator.
        """
        # Extract variant_values for each arm
        self.control_variants = self.comparison_pair.control.variants
        self.treatment_variants = self.comparison_pair.treatment.variants
        self.control_variant_values = [variant.value for variant in self.control_variants]
        self.treatment_variant_values = [variant.value for variant in self.treatment_variants]

        strata_df = self.strata_info.df
        strata_df_control = strata_df.loc[strata_df.index.isin(self.control_variant_values)].copy()
        strata_df_treatment = strata_df.loc[strata_df.index.isin(self.treatment_variant_values)].copy()

        self.strata_info_control = StrataInfo(df=strata_df_control, strata_count_col=self.strata_info.strata_count_col)

        self.strata_info_treatment = StrataInfo(df=strata_df_treatment, strata_count_col=self.strata_info.strata_count_col)

        # Build estimators for each arm
        # Note that the same df can be used as the metric df for both and we do not need to pass a condition through param
        # This is because
        # (1) all variants info are available in df
        # (2) weighted metric estimator will compute the weighted mean using the weights from corresponding strata_df
        self.control_estimator = WeightedMetricEstimator(
            metric=self.metric,
            strata_info=self.strata_info_control,
            variant_values=self.control_variant_values,
            variant_col=self.variant_col,
            param=None,
            name=None,
        )

        self.treatment_estimator = WeightedMetricEstimator(
            metric=self.metric,
            strata_info=self.strata_info_treatment,
            variant_values=self.treatment_variant_values,
            variant_col=self.variant_col,
            param=None,
            name=None,
        )

        # Create the constructor
        diff_estimator_constructor = DiffEstimatorConstructor(diff_type=self.diff_type, name=self.name)

        # Create and return the diff estimator
        return diff_estimator_constructor.construct(control_estimator=self.control_estimator, treatment_estimator=self.treatment_estimator)

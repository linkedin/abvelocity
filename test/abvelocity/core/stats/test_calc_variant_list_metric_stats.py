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

import numpy as np
import pandas as pd
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.param.constants import SAMPLE_COUNT_COL, SUM_COL, SUM_SQ_COL, TRIGGER_STATE_COUNT_COL, VARIANT_COL
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.variant import Variant, VariantList
from abvelocity.core.stats.calc_variant_list_metric_stats import calc_variant_list_metric_stats
from abvelocity.core.stats.calc_variant_metric_stats import calc_variant_metric_stats
from abvelocity.core.stats.stats import UnivarStats


def test_calc_variant_list_metric_stats():
    # Creates a sample DataFrame for testing.
    metric_values = [1, 2, 3, 4, 5]
    df = pd.DataFrame({"variant": [("A",), ("A",), ("B",), ("B",), ("B",)], "metric": metric_values})

    metric = Metric(name="metric", numerator=UMetric(col="metric"))

    # Calls the function under test.
    variant_metric_stats_df = calc_variant_metric_stats(dc=DataContainer(pandas_df=df), metric=metric, variant_col=VARIANT_COL)

    # Asserts the expected output.
    expected_result = pd.DataFrame(
        {
            "variant": [("A",), ("B",)],
            "sample_count": [2, 3],
            "mean": [1.5, 4.0],
            "sd": [0.5, 0.81649],
            "sum": [3, 12],
            "sum_sq": [5, 50],
        }
    )

    pd.testing.assert_frame_equal(variant_metric_stats_df, expected_result, rtol=0.01)

    variant_a = Variant(value=("A",))
    variant_b = Variant(value=("B",))

    variant_list = VariantList(variants=[variant_a, variant_b])

    obtained_univar_stats = calc_variant_list_metric_stats(
        variant_list=variant_list,
        variant_metric_stats_df=variant_metric_stats_df,
        variant_col=VARIANT_COL,
    )

    # Expected stats based on the original vector.
    expected_univer_stats = UnivarStats(
        name="[A, B]",
        mean=np.mean(metric_values),
        sd=np.std(metric_values),
        var=2,
        sample_count=5,
        sum=np.sum(metric_values),
        sum_sq=np.sum(np.array(metric_values) ** 2),
        sample_mean_var=0.4,
        triggered_count=None,
    )

    assert obtained_univar_stats == expected_univer_stats


def test_calc_variant_list_metric_stats_no_weights():
    variant_list_data = VariantList(name="test_variants", variants=[Variant(("v1",)), Variant(("v2",))])

    # Simplified DataFrame with necessary columns only
    variant_metric_stats_df = pd.DataFrame(
        {
            VARIANT_COL: [("v1",), ("v2",), ("v3",)],
            SUM_COL: [10, 40, 90],
            SAMPLE_COUNT_COL: [10, 20, 30],
            SUM_SQ_COL: [15, 100, 560],
        }
    )

    # Raw calculations for expected values
    # We only include v1 and v2
    # Sums: 10 (v1) + 40 (v2) = 50
    # Counts: 10 (v1) + 20 (v2) = 30
    # Sum of Squares: 15 (v1) + 100 (v2) = 115
    # Mean: sum / count = 50 / 30 = 1.6667
    # Variance: (sum_sq / count) - mean^2 = (115 / 30) - (1.6667)^2 = 3.8333 - 2.7778 = 1.0555
    # Standard Deviation: sqrt(variance) = sqrt(1.0555) = 1.0274
    # Sample Mean Variance: variance / count = 1.0555 / 30 = 0.0352

    sum0 = 50
    sample_count = 30
    sum_sq = 115
    mean = sum0 / sample_count
    var = (sum_sq / sample_count) - (mean**2)
    sd = np.sqrt(var)
    sample_mean_var = var / sample_count

    expected = UnivarStats(
        name="test_variants",
        mean=mean,
        sd=sd,
        var=var,
        sample_count=sample_count,
        sum=sum0,
        sum_sq=sum_sq,
        sample_mean_var=sample_mean_var,
    )

    result = calc_variant_list_metric_stats(
        variant_list=variant_list_data,
        variant_metric_stats_df=variant_metric_stats_df,
    )

    assert np.isclose(result.mean, expected.mean)
    assert np.isclose(result.sd, expected.sd)
    assert np.isclose(result.var, expected.var)
    assert result.sample_count == expected.sample_count
    assert np.isclose(result.sum, expected.sum)
    assert np.isclose(result.sum_sq, expected.sum_sq)
    assert np.isclose(result.sample_mean_var, expected.sample_mean_var)


def test_calc_variant_list_metric_stats_with_weights():
    variant_list_data = VariantList(name="test_variants", variants=[Variant(("v1",)), Variant(("v2",))])

    # Simplified DataFrame with necessary columns only
    variant_metric_stats_df = pd.DataFrame(
        {
            VARIANT_COL: [("v1",), ("v2",), ("v3",)],
            SUM_COL: [10, 40, 90],
            SAMPLE_COUNT_COL: [10, 20, 30],
            SUM_SQ_COL: [15, 100, 560],
        }
    )

    variant_count_df = pd.DataFrame({VARIANT_COL: [("v1",), ("v2",), ("v3",)], TRIGGER_STATE_COUNT_COL: [100, 200, 300]}).set_index(VARIANT_COL)

    # Raw calculations for expected values
    # For v1 and v2:
    # Trigger State Counts: 100 (v1) + 200 (v2) = 300
    # Weights: 100/300 = 0.3333 (v1), 200/300 = 0.6667 (v2)
    # Sums: 10 (v1), 40 (v2)
    # Original Counts: 10 (v1), 20 (v2)
    # Means: 10/10 = 1.0 (v1), 40/20 = 2.0 (v2)
    # Variances: (15/10) - (1.0)^2 = 1.5 - 1.0 = 0.5 (v1), (100/20) - (2.0)^2 = 5.0 - 4.0 = 1.0 (v2)
    # Weighted Sample Mean: 0.3333*1.0 + 0.6667*2.0 = 1.6667
    # Updated Sum: 300 * 1.6667 = 500.01
    # Sample Mean Variance: (0.3333^2 * 0.5 / 10) + (0.6667^2 * 1.0 / 20) = 0.005555 + 0.022222 = 0.027777

    trigger_state_counts = [100, 200]  # Trigger state counts for v1 and v2
    total_trigger_state_count = 300
    sums = [10, 40]  # Sums for v1 and v2
    sample_counts = [10, 20]  # Original counts for v1 and v2
    means = [s / c for s, c in zip(sums, sample_counts)]  # Means for v1 and v2
    sum_sqs = [15, 100]  # Sum of squares for v1 and v2
    vars = [(sq / c) - (mean**2) for sq, c, mean in zip(sum_sqs, sample_counts, means)]  # Variances for v1 and v2
    weights = [c / total_trigger_state_count for c in trigger_state_counts]  # Weights for v1 and v2
    weighted_sample_mean = sum(w * mean for w, mean in zip(weights, means))  # Weighted sample mean
    updated_sum = total_trigger_state_count * weighted_sample_mean  # Updated sum
    sample_mean_var = sum(w**2 * var / c for w, var, c in zip(weights, vars, sample_counts))  # Sample mean variance
    sample_count = sum(sample_counts)

    expected = UnivarStats(
        name="test_variants",
        mean=weighted_sample_mean,
        sd=None,  # Not calculated in this case
        var=None,  # Not calculated in this case
        sample_count=sample_count,  # Not calculated in this case
        sum=updated_sum,
        sum_sq=None,  # Not calculated in this case
        sample_mean_var=sample_mean_var,
        triggered_count=total_trigger_state_count,
    )

    result = calc_variant_list_metric_stats(
        variant_list=variant_list_data,
        variant_metric_stats_df=variant_metric_stats_df,
        variant_count_df=variant_count_df,
    )

    assert np.isclose(result.mean, expected.mean)
    assert np.isclose(result.sum, expected.sum)
    assert np.isclose(result.sample_mean_var, expected.sample_mean_var)

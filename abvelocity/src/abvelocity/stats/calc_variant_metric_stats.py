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
import warnings

import numpy as np
import pandas as pd

from abvelocity.param.constants import (
    MEAN_COL,
    SAMPLE_COUNT_COL,
    SD_COL,
    SUM_COL,
    SUM_SQ_COL,
)
from abvelocity.param.metric import Metric
from abvelocity.utils.check_df_validity import check_df_validity


def calc_variant_metric_stats(df: pd.DataFrame, metric: Metric, variant_col: str):
    """Calculates the statistics of a metric for each variant.
    This assumes that in the input dataframe each row corresponds to one unit.

    Args:
        df: The input dataframe with the raw data. This data is supposed to be at unit level.
        metric: The metric of interest. This can be a ratio metric in general (TODO).
            However, while `.denominator` can be None, `.numerator` cannot be None.
        variant_col: The column name in `df` which includes the variant assignment
            for units of the experiment.

    Returns:
        variant_metric_stats_df: A dataframe with each row representing the statistics of a metric for a variant.
            The quantities calculated include:

            - count: The number of units in the variant.
            - mean: The mean of the metric for the variant.
            - sd: The standard deviation of the metric for the variant.
            - sum: The sum of the metric for the variant.
            - sum_sq: The sum of the metric squared for the variant.
    """
    # Checks for required columns.
    needed_cols = [variant_col, metric.numerator.name]
    if metric.denominator is not None:
        needed_cols.append(metric.denominator.name)

    check_df_validity(
        df=df, needed_cols=needed_cols, err_trigger_source="calc_variant_metric_stats"
    )

    metric_col = metric.numerator.name

    # Silence seemingly unnecessary warning.
    # Pandas gives the warning and suggest to concat all added columns at once.
    # However, we deliberately create and delete this column for all metrics,
    # to minimize chance of memory overflow.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        df["temp_metric_squared"] = df[metric_col] ** 2

    # TODO: Check to make sure "count" or "size" should be used in first aggregation. (The diff is about nulls).
    if metric.sample_count is None:
        variant_metric_stats_df = df.groupby([variant_col], as_index=False).agg(
            sample_count=(metric_col, "size"),
            mean=(metric_col, "mean"),
            sd=(metric_col, "std"),
            sum=(metric_col, "sum"),
            sum_sq=("temp_metric_squared", "sum"),
        )
    else:
        # The case where the sample count is more complex and a summable column is given.
        # This specifies a way to count units (sample size) if the number of rows is not correct.
        # Here is an example where this can be useful.
        # Assume we like to estimate the impact of an experiment on retention.
        # The unit data might be as follows:

        # unit, renew, eligible
        # ---------------------
        # u1,   1,     1
        # u2,   1,     1
        # u3,   0,     1
        # u4,   0,     0

        # Note that in this case, u4 is not even eligible for renew, but if we count rows
        # and use a binomial based or appromixation of it, it will be counted in the sample size.
        # In this case the user can pass eligible as a `UMetric` via this field, so that we get zeros and
        # ones for this column.
        variant_metric_stats_df = df.groupby([variant_col], as_index=False).agg(
            sample_count=(metric.sample_count.name, "sum"),
            sum=(metric_col, "sum"),
            sum_sq=("temp_metric_squared", "sum"),
        )

        variant_metric_stats_df["mean"] = (
            variant_metric_stats_df["sum"] / variant_metric_stats_df["sample_count"]
        )
        # To calculate sd, we use the formula: pseudo code: VARIANCE(X) = E(X^2) - E(X)^2
        variant_metric_stats_df["sd"] = np.sqrt(
            (variant_metric_stats_df["sum_sq"] / (variant_metric_stats_df["sample_count"] - 1))
            - variant_metric_stats_df["mean"] ** 2
        )
        # re-order columns to the same order as expected.
        cols = [variant_col, "sample_count", "mean", "sd", "sum", "sum_sq"]
        variant_metric_stats_df = variant_metric_stats_df[cols]

    del df["temp_metric_squared"]

    # Emphasize the column names.
    assert (
        variant_metric_stats_df.columns
        == [variant_col, SAMPLE_COUNT_COL, MEAN_COL, SD_COL, SUM_COL, SUM_SQ_COL]
    ).all()

    return variant_metric_stats_df

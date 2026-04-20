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
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.param.constants import MEAN_COL, SAMPLE_COUNT_COL, SD_COL, SUM_COL, SUM_SQ_COL
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.metric import Metric
from abvelocity.core.utils.check_df_validity import check_df_validity

# The final aggregated DataFrame will be named 'variant_metric_stats_df' regardless of path.
# We initialize it here for scope.
variant_metric_stats_df = None


def calc_variant_metric_stats(
    dc: DataContainer,
    metric: Metric,
    variant_col: str,
    io_param: Optional[IOParam] = None,
) -> pd.DataFrame:
    """Calculates the statistics of a metric for each variant.
    This assumes that in the input dataframe each row corresponds to one unit.
    This stats are for metrics which only have a numerator.
    But it allows for the sample count to be given in another column.
    If the `sample_count` is given we use its sum in the mean calculation,
    (rather than counting the number of rows).

    Args:
        dc: The input with the raw data. This data is supposed to be at unit level.
        metric: The metric of interest.
        variant_col: The column name in `df` which includes the variant assignment
            for units of the experiment.
        io_param: Optional IOParam object for specifying SQL connection.

    Returns:
        variant_metric_stats_df: A dataframe with each row representing the statistics of a metric for a variant.
            The quantities calculated include:

            - count: The number of units in the variant.
            - mean: The mean of the metric for the variant.
            - sd: The standard deviation of the metric for the variant.
            - sum: The sum of the metric for the variant.
            - sum_sq: The sum of the metric squared for the variant.
    """

    cursor = io_param.cursor if io_param is not None else None
    metric_col = metric.numerator.name
    sample_count_col = metric.sample_count.name if metric.sample_count else None

    # Check for metric denominator first (it's the same logic for both SQL/Pandas)
    if metric.denominator:
        raise ValueError(
            "In `calc_variant_metric_stats` a metric with denominator is passed. "
            f"metric with metric.name {metric.name} has a denominator: {metric.denominator}. "
            "Use 'general' method for metrics with denominator."
        )

    if dc.table_name is None and dc.query is None and dc.pandas_df is None:
        raise ValueError(
            "In `calc_variant_metric_stats`, the DataContainer has no data source. "
            "Please provide either a table_name, query, or pandas_df in the DataContainer."
        )

    # Determine if we should use the SQL path
    # This will happen if either table_name or query is given in the DataContainer,
    # but pandas_df is not available.
    is_sql_source = (dc.table_name is not None or dc.query is not None) and dc.pandas_df is None

    if is_sql_source:
        if cursor is None:
            raise ValueError(
                f"A SQL table name or query is provided in the DataContainer: {dc}, "
                "but no 'cursor' instance was supplied (via io_param) to perform the initial aggregation."
            )

        # If `.table_name` we will use that to get the data
        # If not we will use the `.query` provided in dc.
        if dc.table_name is not None:
            sql_source = dc.table_name
            print(f"\n***: In `calc_variant_metric_stats`, " f"using dc.table_name: {dc.table_name} for computation.")
        elif dc.query is not None:
            sql_source = f"({dc.query})"
            print(f"\n***: In `calc_variant_metric_stats`, " f"using dc.query: {dc.query} as subquery for computation.")
        else:
            raise ValueError("In `calc_variant_metric_stats`, both dc.table_name and dc.query are None. " f"dc: {dc}")

        # Determine the COUNT/SUM aggregation for the denominator/sample_count
        # If sample_count is None, we count the rows; otherwise, we sum the sample_count column.
        count_agg = "COUNT(*)" if sample_count_col is None else f"SUM({sample_count_col})"

        # We calculate SUM, SUM_SQ, and the sample count in the database.
        aggregation_query = (
            f"SELECT {variant_col}, "
            f"{count_agg} AS {SAMPLE_COUNT_COL}, "
            f"SUM({metric_col}) AS {SUM_COL}, "
            f"SUM({metric_col} * {metric_col}) AS {SUM_SQ_COL} "
            f"FROM {sql_source} "
            f"GROUP BY 1"
        )

        result = cursor.get_df(aggregation_query)
        variant_metric_stats_df = result.df
    elif dc.pandas_df is not None:
        # Case 2: Data is sourced from a Pandas DataFrame (original path)
        df = dc.pandas_df
        print(f"\n***: In `calc_variant_metric_stats`, using dc.pandas_df: {df.head(3)} for computation.")

        # Checks for required columns.
        needed_cols = [variant_col, metric_col]
        check_df_validity(df=df, needed_cols=needed_cols, err_trigger_source="calc_variant_metric_stats")

        # Silence seemingly unnecessary warning.
        # Pandas gives the warning and suggest to concat all added columns at once.
        # However, we deliberately create and delete this column for all metrics,
        # to minimize chance of memory overflow.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            df["temp_metric_squared"] = df[metric_col] ** 2

        # TODO: Check to make sure "count" or "size" should be used in first aggregation. (The diff is about nulls).
        if sample_count_col is None:
            variant_metric_stats_df = df.groupby([variant_col], as_index=False).agg(
                sample_count=(metric_col, "size"),
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
            # u1, 1, 1
            # u2, 1, 1
            # u3, 0, 1
            # u4, 0, 0

            # Note that in this case, u4 is not even eligible for renew, but if we count rows
            # and use a binomial based or appromixation of it, it will be counted in the sample size.
            # In this case the user can pass eligible as a `UMetric` via this field, so that we get zeros and
            # ones for this column.
            variant_metric_stats_df = df.groupby([variant_col], as_index=False).agg(
                sample_count=(sample_count_col, "sum"),
                sum=(metric_col, "sum"),
                sum_sq=("temp_metric_squared", "sum"),
            )
            # The mean and sd calculation will be done after the aggregation block.

        del df["temp_metric_squared"]

    else:
        # Case 3: DataContainer is empty
        raise ValueError("DataContainer has no data (no pandas_df, table_name, or query set).")

    # Make sure variant_col is tuple (Trino returns NamedRowTuple, DuckDB returns dict)
    variant_metric_stats_df[variant_col] = [tuple(item.values()) if isinstance(item, dict) else tuple(item) for item in variant_metric_stats_df[variant_col]]

    variant_metric_stats_df[MEAN_COL] = variant_metric_stats_df[SUM_COL] / variant_metric_stats_df[SAMPLE_COUNT_COL]

    # To calculate sd, we use the formula: pseudo code: VARIANCE(X) = E(X^2) - E(X)^2
    # This is better than computing sd directly as this is more general
    # This is because in this calculation we use the correct `sample_count`
    # Recall the sample_count variable is useful for metrics for which eligible units
    # might be smaller than those appearing in the data.
    # One example could be when units are subscribers up for renew which are eligible
    # In such case, this is equivalant to filtering the non-eligible out first then compute mean and sd
    variant_metric_stats_df[SD_COL] = np.sqrt(
        (variant_metric_stats_df[SUM_SQ_COL] / (variant_metric_stats_df[SAMPLE_COUNT_COL])) - variant_metric_stats_df[MEAN_COL] ** 2
    )

    # re-order columns to the same order as expected.
    cols = [variant_col, SAMPLE_COUNT_COL, MEAN_COL, SD_COL, SUM_COL, SUM_SQ_COL]
    variant_metric_stats_df = variant_metric_stats_df[cols]

    # Emphasize the column names.
    assert (variant_metric_stats_df.columns == [variant_col, SAMPLE_COUNT_COL, MEAN_COL, SD_COL, SUM_COL, SUM_SQ_COL]).all()

    return variant_metric_stats_df

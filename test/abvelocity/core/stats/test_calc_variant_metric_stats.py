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
from abvelocity.core.get_data.duckdb_cursor import DuckDBCursor
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.stats.calc_variant_metric_stats import calc_variant_metric_stats


def test_calc_variant_metric_stats():
    # Creates a sample DataFrame for testing.
    df = pd.DataFrame({"variant": ["A", "A", "B", "B", "B"], "metric": [1, 2, 3, 4, 5]})

    metric = Metric(name="metric", numerator=UMetric(col="metric"))
    # Calls the function under test.
    result = calc_variant_metric_stats(dc=DataContainer(pandas_df=df), metric=metric, variant_col="variant")

    # Asserts the expected output.
    expected_result = pd.DataFrame(
        {
            "variant": [("A",), ("B",)],
            "sample_count": [2, 3],
            "mean": [1.5, 4.0],
            "sd": [0.5, 0.8164],
            "sum": [3, 12],
            "sum_sq": [5, 50],
        }
    )

    pd.testing.assert_frame_equal(result, expected_result, rtol=0.01)


def test_calc_variant_metric_stats_long():
    # This tests for a larger sample size
    # Several cases are given including those which use `.sample_count`
    rng = np.random.default_rng(1317)
    size = 1000
    a_samples = rng.normal(loc=1.0, scale=1.0, size=size)
    b_samples = rng.normal(loc=-1.0, scale=2.0, size=size)

    df = pd.DataFrame(
        {
            "variant": (["A"] * size + ["B"] * size),
            "metric": list(a_samples) + list(b_samples),
            "eligible": [1 for i in range(2 * size)],
        }
    )

    # Case 1
    # Here we do not use the eligible column as `sample_count` in the `Metric` definition.
    metric = Metric(name="metric", numerator=UMetric(col="metric"))
    # Calls the function under test.
    result1 = calc_variant_metric_stats(dc=DataContainer(pandas_df=df), metric=metric, variant_col="variant")

    # Asserts the expected output.
    expected_result = pd.DataFrame(
        {
            "variant": [("A",), ("B",)],
            "sample_count": [size, size],
            "mean": [1.01, -1.083],
            "sd": [0.954, 1.9399],
            "sum": [1010.681, -1083.6571],
            "sum_sq": [1930.86, 4933.767],
        }
    )

    pd.testing.assert_frame_equal(result1, expected_result, rtol=0.01)

    # Case 2
    # This time we use the eligible column as `.sample_count` in Metric definition.
    # However since we are using uniformly 1 everywhere, all samples will count
    # This means results should be close to the original results
    metric = Metric(name="metric", numerator=UMetric(col="metric"), sample_count=UMetric(col="eligible"))
    # Calls the function under test.
    result2 = calc_variant_metric_stats(dc=DataContainer(pandas_df=df), metric=metric, variant_col="variant")
    pd.testing.assert_frame_equal(result2, expected_result, rtol=0.01)

    # Case 3
    # This time we will use another eligible column where only 10 samples are eligible for A and 20 for B.
    eligible = [1 for i in range(10)] + [0 for i in range(10, size)] + [1 for i in range(20)] + [0 for i in range(20, size)]

    assert len(eligible) == (size * 2)

    a_samples = list(rng.normal(loc=1.0, scale=1.0, size=size)[:10]) + [0] * (size - 10)
    b_samples = list(rng.normal(loc=-1.0, scale=2.0, size=size)[:20]) + [0] * (size - 20)

    assert len(a_samples) == size
    assert len(b_samples) == size

    df = pd.DataFrame(
        {
            "variant": (["A"] * size + ["B"] * size),
            "metric": list(a_samples) + list(b_samples),
            "eligible": eligible,
        }
    )

    metric = Metric(name="metric", numerator=UMetric(col="metric"), sample_count=UMetric(col="eligible"))
    # Calls the function under test.
    result3 = calc_variant_metric_stats(dc=DataContainer(pandas_df=df), metric=metric, variant_col="variant")

    # Now we do expect a different result
    expected_result = pd.DataFrame(
        {
            "variant": [("A",), ("B",)],
            "sample_count": [10, 20],
            "mean": [1.35, -0.779],
            "sd": [1.117, 2.6047],
            "sum": [13.5476, -15.58754],
            "sum_sq": [30.84941, 147.846],
        }
    )

    pd.testing.assert_frame_equal(result3, expected_result, rtol=0.01)


def test_calc_variant_metric_stats_sql():
    # 1. Setup DuckDBCursor (automatically creates in-memory connection)
    cursor = DuckDBCursor()
    table_name = "test_variant_metrics_df"

    # Creates a sample DataFrame for testing.
    metric_values = [1, 2, 3, 4, 5]
    # Variant column values are strings, matching the database storage type.
    df = pd.DataFrame({"variant": ["A", "A", "B", "B", "B"], "metric": metric_values})

    # Register the DataFrame as a temporary view/table.
    cursor._db_connection.register(f"{table_name}_temp", df)
    create_table_query = f"""
    CREATE TABLE {table_name} AS
    SELECT
        STRING_SPLIT_REGEX(variant, ', ') AS variant,
        metric
    FROM {table_name}_temp;
    """

    cursor._db_connection.execute(create_table_query)
    cursor._db_connection.execute(f"DROP VIEW {table_name}_temp;")  # Clean up the registered Pandas view

    # 2. Setup Metric and IOParam
    metric = Metric(name="metric", numerator=UMetric(col="metric"))
    io_param = IOParam(cursor=cursor)

    # 3. Call the function using the SQL path (DataContainer only has table_name)
    variant_metric_stats_df_sql = calc_variant_metric_stats(
        dc=DataContainer(table_name=table_name),
        metric=metric,
        variant_col="variant",
        io_param=io_param,
    )

    # 4. Define the EXPECTED result
    expected_result = pd.DataFrame(
        {
            "variant": [("A",), ("B",)],
            "sample_count": [2, 3],
            "mean": [1.5, 4.0],
            "sd": [0.5, 0.8164],
            "sum": [3.0, 12.0],
            "sum_sq": [5.0, 50.0],
        }
    )

    # Cast necessary columns in the expected result to match the typical output types of
    # `calc_variant_metric_stats` when running through the SQL path (which uses floats for sums).
    expected_result["sample_count"] = expected_result["sample_count"].astype(np.int64)
    expected_result["sum"] = expected_result["sum"].astype(float)
    expected_result["sum_sq"] = expected_result["sum_sq"].astype(float)

    variant_metric_stats_df_sql.sort_values("variant", inplace=True)
    variant_metric_stats_df_sql.reset_index(drop=True, inplace=True)
    expected_result.reset_index(drop=True, inplace=True)

    # 5. Assert the results match (using check_exact=False for float comparisons)
    pd.testing.assert_frame_equal(variant_metric_stats_df_sql, expected_result, check_dtype=False, rtol=0.01)

    # 6. Cleanup: Close the in-memory connection managed by the cursor
    cursor._db_connection.close()

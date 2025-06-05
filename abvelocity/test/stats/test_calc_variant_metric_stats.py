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

from abvelocity.param.metric import Metric, UMetric
from abvelocity.stats.calc_variant_metric_stats import calc_variant_metric_stats


def test_calc_variant_metric_stats():
    # Creates a sample DataFrame for testing.
    df = pd.DataFrame({"variant": ["A", "A", "B", "B", "B"], "metric": [1, 2, 3, 4, 5]})

    metric = Metric(name="metric", numerator=UMetric(col="metric"))
    # Calls the function under test.
    result = calc_variant_metric_stats(df=df, metric=metric, variant_col="variant")

    # Asserts the expected output.
    expected_result = pd.DataFrame(
        {
            "variant": ["A", "B"],
            "sample_count": [2, 3],
            "mean": [1.5, 4.0],
            "sd": [0.7071067811865476, 1.0],
            "sum": [3, 12],
            "sum_sq": [5, 50],
        }
    )

    pd.testing.assert_frame_equal(result, expected_result)


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
    result1 = calc_variant_metric_stats(df=df, metric=metric, variant_col="variant")

    # Asserts the expected output.
    expected_result = pd.DataFrame(
        {
            "variant": ["A", "B"],
            "sample_count": [size, size],
            "mean": [1.0106810178848715, -1.0836571194799272],
            "sd": [0.9540944476059315, 1.9399014576725255],
            "sum": [1010.6810178848716, -1083.6571194799271],
            "sum_sq": [1930.862039, 4933.767200],
        }
    )

    pd.testing.assert_frame_equal(result1, expected_result)

    # Case 2
    # This time we use the eligible column as `.sample_count` in Metric definition.
    # However since we are using uniformly 1 everywhere, all samples will count
    # This means results should be close to the original results
    metric = Metric(
        name="metric", numerator=UMetric(col="metric"), sample_count=UMetric(col="eligible")
    )
    # Calls the function under test.
    result2 = calc_variant_metric_stats(df=df, metric=metric, variant_col="variant")
    pd.testing.assert_frame_equal(result2, expected_result, rtol=0.01)

    # Case 3
    # This time we will use another eligible column where only 10 samples are eligible for A and 20 for B.
    eligible = (
        [1 for i in range(10)]
        + [0 for i in range(10, size)]
        + [1 for i in range(20)]
        + [0 for i in range(20, size)]
    )

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

    metric = Metric(
        name="metric", numerator=UMetric(col="metric"), sample_count=UMetric(col="eligible")
    )
    # Calls the function under test.
    result3 = calc_variant_metric_stats(df=df, metric=metric, variant_col="variant")

    # Now we do expect a different result
    expected_result = pd.DataFrame(
        {
            "variant": ["A", "B"],
            "sample_count": [10, 20],
            "mean": [1.3547615739717729, -0.7793771232823261],
            "sd": [1.2618772858748242, 2.6784201267182732],
            "sum": [13.54761573971773, -15.587542465646523],
            "sum_sq": [30.849418862255593, 147.84589843460193],
        }
    )

    pd.testing.assert_frame_equal(result3, expected_result, rtol=0.01)

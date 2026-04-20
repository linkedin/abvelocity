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

import pandas as pd
from abvelocity.core.param.variant import ComparisonPair, Variant, VariantList
from abvelocity.core.stats.calc_variant_metric_effects_simple import calc_variant_metric_effects_simple, compare_variants_with_control
from numpy.testing import assert_allclose


def test_calc_variant_metric_effects_simple():
    """Tests `calc_variant_metric_effects_simple` function."""
    # Creates a sample `variant_metric_stats_df` and `comparison_pairs`.
    variant_metric_stats_df = pd.DataFrame(
        {
            "variant": [("A",), ("B",), ("C",)],
            "mean": [None] * 3,
            "sd": [None] * 3,
            "sample_count": [10, 10, 10],
            "sum": [10 * 2, 10 * 3, 10 * 4],
            "sum_sq": [10 * 4, 10 * 9, 10 * 16],
        }
    )

    comparison_pairs = [
        ComparisonPair(
            treatment=VariantList(variants=[Variant(value=("A",))]),
            control=VariantList(variants=[Variant(value=("B",))]),
            name=("A", "B"),
        ),
        ComparisonPair(
            treatment=VariantList(variants=[Variant(value=("A",))]),
            control=VariantList(variants=[Variant(value=("C",))]),
            name=("A", "C"),
        ),
        ComparisonPair(
            treatment=VariantList(variants=[Variant(value=("B",))]),
            control=VariantList(variants=[Variant(value=("C",))]),
            name=("B", "C"),
        ),
    ]

    # Calls the function.
    result = calc_variant_metric_effects_simple(variant_metric_stats_df, "variant", comparison_pairs)
    len(result) == 3
    assert_allclose(result["delta"], (-1, -2, -1))


def test_calc_variant_metric_effects_simple_tuples():
    """Tests `calc_variant_metric_effects_simple` function.
    This time, each variant is a tuple.
    """
    # Creates a sample `variant_metric_stats_df` and `comparison_pairs`.
    variant_metric_stats_df = pd.DataFrame(
        {
            "variant": [("A1", "B1"), ("A1", "B2"), ("A2", "B1"), ("A2", "B2")],
            "mean": [None] * 4,
            "sd": [None] * 4,
            "sample_count": [100, 150, 200, 400],
            "sum": [100 * 2, 150 * 3, 200 * 4, 400 * 5],
            "sum_sq": [100 * 4, 150 * 9, 200 * 16, 400 * 25],
        }
    )

    comparison_pairs = [
        ComparisonPair(
            treatment=VariantList(variants=[Variant(value=("A1", "B1"))]),
            control=VariantList(variants=[Variant(value=("A1", "B2"))]),
            name="pair1",
        ),
        ComparisonPair(
            treatment=VariantList(variants=[Variant(value=("A1", "B1"))]),
            control=VariantList(variants=[Variant(value=("A2", "B1"))]),
            name="pair2",
        ),
    ]

    # Calls the function.
    result = calc_variant_metric_effects_simple(variant_metric_stats_df, "variant", comparison_pairs)
    assert len(result) == 2
    assert_allclose(result["delta"], (-1, -2))


def test_compare_variants_with_control():
    """Tests `compare_variants_with_control` function."""
    variant_metric_stats_df = pd.DataFrame(
        {
            "variant": [("A",), ("B",), ("C",)],
            "mean": [None] * 3,
            "sd": [None] * 3,
            "sample_count": [10, 10, 10],
            "sum": [10 * 2, 10 * 3, 10 * 4],
            "sum_sq": [10 * 4, 10 * 9, 10 * 16],
        }
    )

    result = compare_variants_with_control(variant_metric_stats_df=variant_metric_stats_df, variant_col="variant", expt_control=("A",))

    assert len(result) == 2
    assert_allclose(result["delta"], (1, 2))

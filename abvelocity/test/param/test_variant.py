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

import pytest

from abvelocity.param.variant import (
    ComparisonPair,
    TriggerState,
    Variant,
    VariantList,
    variant_to_trigger_state,
)


def test_variant_str():
    v = Variant("v1")
    assert v.value == "v1"
    assert v.name == "v1"


def test_variant_tuple():
    v = Variant(("v1", "w1"))
    assert v.value == ("v1", "w1")
    assert v.name == "(v1, w1)"


def test_variant_list_single_variant():
    v1 = Variant("v1")
    variant_list = VariantList([v1])
    assert variant_list.variants == [v1]
    assert variant_list.name == "[v1]"


def test_variant_list_multiple_variants():
    v1 = Variant(("v1", "w1"))
    v2 = Variant(("v2", "w2"))
    variant_list = VariantList([v1, v2])
    assert variant_list.variants == [v1, v2]
    assert variant_list.name == "[(v1, w1), (v2, w2)]"


def test_variant_list_inconsistent_tuple_lengths():
    v1 = Variant(("v1", "w1"))
    v2 = Variant(("v2", "w2", "x2"))
    with pytest.raises(
        ValueError, match="All tuple elements in the VariantList must be of the same size."
    ):
        VariantList([v1, v2])


def test_variant_list_mixed_types():
    v1 = Variant("v1")
    v2 = Variant(("v2", "w2"))
    with pytest.raises(
        ValueError, match="All elements in the VariantList must be of the same type"
    ):
        VariantList([v1, v2])


def test_comparison_pair_same_size_elements():
    v1 = Variant(("v1", "w1"))
    v2 = Variant(("v2", "w2"))
    v3 = Variant(("v3", "w3"))
    v4 = Variant(("v4", "w4"))
    variant_list1 = VariantList([v1, v2])
    variant_list2 = VariantList([v3, v4])
    cp = ComparisonPair(treatment=variant_list1, control=variant_list2)
    assert cp.treatment == variant_list1
    assert cp.control == variant_list2
    assert cp.name == "[(v1, w1), (v2, w2)] versus [(v3, w3), (v4, w4)]"


def test_comparison_pair_mixed_types_variants():
    v1 = Variant("v1")
    v2 = Variant("v2")
    v3 = Variant(("v1", "w1"))
    v4 = Variant(("v2", "w2"))
    variant_list1 = VariantList([v1, v2])
    variant_list2 = VariantList([v3, v4])
    with pytest.raises(
        ValueError, match="All elements in treatment and control must be of the same type"
    ):
        ComparisonPair(treatment=variant_list1, control=variant_list2)


def test_comparison_pair_with_name():
    v1 = Variant(("v1", "w1"))
    v2 = Variant(("v2", "w2"))
    v3 = Variant(("v3", "w3"))
    v4 = Variant(("v4", "w4"))
    variant_list1 = VariantList([v1, v2])
    variant_list2 = VariantList([v3, v4])
    cp = ComparisonPair(treatment=variant_list1, control=variant_list2, name="Test Comparison")
    assert cp.treatment == variant_list1
    assert cp.control == variant_list2
    assert cp.name == "Test Comparison"


def test_comparison_pair_with_removal():
    v1 = Variant(("v1", "w1"))
    v2 = Variant(("v2", "w2"))
    v3 = v1
    v4 = Variant(("v4", "w4"))
    variant_list1 = VariantList([v1, v2])
    variant_list2 = VariantList([v3, v4])
    cp = ComparisonPair(treatment=variant_list1, control=variant_list2, name="Test Comparison")
    assert cp.treatment == VariantList(variants=[v2], name="[(v1, w1), (v2, w2)]")
    assert cp.control == VariantList(variants=[v4], name="[(v1, w1), (v4, w4)]")
    assert cp.name == "Test Comparison"


def test_comparison_pair_same_raise_value_error():
    v1 = Variant(("v1", "w1"))
    v2 = Variant(("v2", "w2"))
    v3 = v1
    v4 = v2
    variant_list1 = VariantList([v1, v2])
    variant_list2 = VariantList([v3, v4])

    with pytest.raises(ValueError, match="At least one of the arms does not have any variants."):
        ComparisonPair(treatment=variant_list1, control=variant_list2, name="Test Comparison")


def test_comparison_pair_different_number_of_variants():
    v1 = Variant("v1")
    v2 = Variant("v2")
    v3 = Variant(("v1", "w1"))
    variant_list1 = VariantList([v1, v2])
    variant_list2 = VariantList([v3])
    with pytest.raises(
        ValueError, match="All elements in treatment and control must be of the same type"
    ):
        ComparisonPair(treatment=variant_list1, control=variant_list2)


def test_comparison_pair_different_list_size_allowed():
    v1 = Variant("v1")
    v2 = Variant("v2")
    v3 = Variant("v3")
    variant_list1 = VariantList([v1, v2])
    variant_list2 = VariantList([v3])
    # This compares the union of two variants (v1 and v2) with v3.
    cp = ComparisonPair(treatment=variant_list1, control=variant_list2)
    assert cp.name == "[v1, v2] versus [v3]"


def test_trigger_state():
    ts = TriggerState(value=True)
    assert ts.value is True
    assert ts.overall_value is True
    assert ts.name == "True"


def test_trigger_state_false():
    ts = TriggerState(value=False)
    assert ts.value is False
    assert ts.overall_value is False
    assert ts.name == "False"


def test_trigger_state_tuple():
    ts = TriggerState(value=(True, False))
    assert ts.value == (True, False)
    assert ts.overall_value is True
    assert ts.name == "(True, False)"


def test_trigger_state_tuple_false():
    ts = TriggerState(value=(False, False))
    assert ts.value == (False, False)
    assert ts.overall_value is False
    assert ts.name == "(False, False)"


def test_variant_to_trigger_state_single_experiment():
    v = Variant("v1")
    ts = variant_to_trigger_state(v)
    assert ts.value is True
    assert ts.overall_value is True
    assert ts.name == "True"


def test_variant_to_trigger_state_multi_experiment():
    v = Variant(("v1", "w1", "x1"))
    ts = variant_to_trigger_state(v)
    assert ts.value == (True, True, True)
    assert ts.overall_value is True
    assert ts.name == "(True, True, True)"


def test_variant_to_trigger_state_multi_experiment_one_false():
    v = Variant(("v1", "w1", "nan"))
    ts = variant_to_trigger_state(v)
    assert ts.value == (True, True, False)
    assert ts.overall_value is True
    assert ts.name == "(True, True, False)"


def test_variant_to_trigger_state_multi_experiment_all_false():
    v = Variant(("nan", "nan", "nan"))
    ts = variant_to_trigger_state(v)
    assert ts.value == (False, False, False)
    assert ts.overall_value is False
    assert ts.name == "(False, False, False)"

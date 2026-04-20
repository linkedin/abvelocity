# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
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

import pytest
from abvelocity.core.param.variant import ComparisonPair, TriggerState, Variant, VariantList, variant_to_trigger_state


def test_variant_single():
    v = Variant(("v1",))
    assert v.value == ("v1",)
    assert v.name == "v1"


def test_variant_tuple():
    v = Variant(("v1", "w1"))
    assert v.value == ("v1", "w1")
    assert v.name == "(v1, w1)"


def test_variant_is_consistent_with_single():
    v = Variant(("v1",))
    assert v.is_consistent_with((None,))
    assert v.is_consistent_with(("v1",))
    assert not v.is_consistent_with(("v2",))


def test_variant_is_consistent_with_multi():
    v = Variant(("v1", "w1"))
    assert v.is_consistent_with((None, None))
    assert v.is_consistent_with(("v1", None))
    assert v.is_consistent_with((None, "w1"))
    assert v.is_consistent_with(("v1", "w1"))
    assert not v.is_consistent_with(("v1", "w2"))
    assert not v.is_consistent_with(("v2", "w1"))


def test_variant_is_consistent_with_three():
    v = Variant(("v1", "w1", "z1"))
    assert v.is_consistent_with((None, None, None))
    assert v.is_consistent_with(("v1", None, "z1"))
    assert v.is_consistent_with((None, "w1", "z1"))
    assert not v.is_consistent_with(("v1", "w2", "u1"))
    assert not v.is_consistent_with((None, "v2", "w1"))


def test_variant_is_consistent_with_size_mismatch():
    v = Variant(("v1", "w1"))
    with pytest.raises(ValueError, match="Input tuple size 1 does not match variant value size 2"):
        v.is_consistent_with(("v1",))
    with pytest.raises(ValueError, match="Input tuple size 3 does not match variant value size 2"):
        v.is_consistent_with(("v1", "w1", "x1"))


def test_trigger_state_single():
    ts = TriggerState((True,))
    assert ts.value == (True,)
    assert ts.name == "True"
    assert ts.overall_value is True


def test_trigger_state_multi():
    ts = TriggerState((True, False, True))
    assert ts.value == (True, False, True)
    assert ts.name == "(True, False, True)"
    assert ts.overall_value is True


def test_variant_to_trigger_state_single():
    v = Variant(("v1",))
    ts = variant_to_trigger_state(v)
    assert ts.value == (True,)
    assert ts.name == "True"
    assert ts.overall_value is True


def test_variant_to_trigger_state_multi():
    v = Variant(("v1", "w1", "nan"))
    ts = variant_to_trigger_state(v)
    assert ts.value == (True, True, False)
    assert ts.name == "(True, True, False)"
    assert ts.overall_value is True


def test_variant_list_single_variant():
    v1 = Variant(("v1",))
    variant_list = VariantList([v1])
    assert variant_list.variants == [v1]
    assert variant_list.name == "[v1]"


def test_variant_list_multiple_variants():
    v1 = Variant(("v1", "w1"))
    v2 = Variant(("v2", "w2"))
    variant_list = VariantList([v1, v2])
    assert variant_list.variants == [v1, v2]
    assert variant_list.name == "[(v1, w1), (v2, w2)]"


def test_variant_list_empty():
    with pytest.raises(ValueError, match="`.variants` in `VariantList` field cannot be an empty list."):
        VariantList([])


def test_variant_list_inconsistent_tuple_lengths():
    v1 = Variant(("v1", "w1"))
    v2 = Variant(("v2", "w2", "x2"))
    with pytest.raises(ValueError, match="All tuple elements in the VariantList must be of the same size."):
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
        ComparisonPair(treatment=variant_list1, control=variant_list2)

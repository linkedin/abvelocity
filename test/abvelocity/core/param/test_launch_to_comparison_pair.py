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

from abvelocity.core.param.launch import Launch
from abvelocity.core.param.launch_to_comparison_pair import launch_to_comparison_pair
from abvelocity.core.param.variant import Variant, VariantList


def test_single_element():
    launch = Launch(value=("v1",))
    comparison_pair = launch_to_comparison_pair(launch)
    expected_treatment = VariantList(variants=[Variant(value=("v1",))])
    expected_control = VariantList(variants=[Variant(value=("control",))])
    assert comparison_pair.treatment == expected_treatment
    assert comparison_pair.control == expected_control
    assert comparison_pair.name == "v1 launch"


def test_two_elements():
    launch = Launch(value=("a", "b"))
    comparison_pair = launch_to_comparison_pair(launch)
    expected_treatment_values = [("a", "b"), ("nan", "b"), ("a", "nan")]
    expected_control_values = [("control", "control"), ("nan", "control"), ("control", "nan")]
    expected_treatment = VariantList(variants=[Variant(value=val) for val in expected_treatment_values])
    expected_control = VariantList(variants=[Variant(value=val) for val in expected_control_values])
    assert sorted(comparison_pair.treatment.variants, key=lambda x: x.value) == sorted(expected_treatment.variants, key=lambda x: x.value)
    assert sorted(comparison_pair.control.variants, key=lambda x: x.value) == sorted(expected_control.variants, key=lambda x: x.value)
    assert comparison_pair.name == "(a, b) launch"


def test_three_elements():
    launch = Launch(value=("a", "b", "c"))
    comparison_pair = launch_to_comparison_pair(launch)
    expected_treatment_values = [
        ("a", "b", "c"),
        ("nan", "b", "c"),
        ("a", "nan", "c"),
        ("a", "b", "nan"),
        ("nan", "nan", "c"),
        ("nan", "b", "nan"),
        ("a", "nan", "nan"),
    ]
    expected_control_values = [
        ("control", "control", "control"),
        ("nan", "control", "control"),
        ("control", "nan", "control"),
        ("control", "control", "nan"),
        ("nan", "nan", "control"),
        ("nan", "control", "nan"),
        ("control", "nan", "nan"),
    ]
    expected_treatment = VariantList(variants=[Variant(value=val) for val in expected_treatment_values])
    expected_control = VariantList(variants=[Variant(value=val) for val in expected_control_values])
    assert sorted(comparison_pair.treatment.variants, key=lambda x: x.value) == sorted(expected_treatment.variants, key=lambda x: x.value)
    assert sorted(comparison_pair.control.variants, key=lambda x: x.value) == sorted(expected_control.variants, key=lambda x: x.value)
    assert comparison_pair.name == "(a, b, c) launch"


def test_custom_name():
    launch = Launch(value=("v1", "w1", "x1"))
    comparison_pair = launch_to_comparison_pair(launch, name="Custom Comparison")
    assert comparison_pair.name == "Custom Comparison"


def test_default_name():
    launch = Launch(value=("v1", "w1", "x1"))
    comparison_pair = launch_to_comparison_pair(launch)
    assert comparison_pair.name == "(v1, w1, x1) launch"


def test_passed_control_launch():
    launch = Launch(value=("a", "b", "c"))
    control_launch = Launch(value=("d", "e", "f"))
    comparison_pair = launch_to_comparison_pair(launch=launch, control_launch=control_launch)
    expected_treatment_values = [
        ("a", "b", "c"),
        ("nan", "b", "c"),
        ("a", "nan", "c"),
        ("a", "b", "nan"),
        ("nan", "nan", "c"),
        ("nan", "b", "nan"),
        ("a", "nan", "nan"),
    ]
    expected_control_values = [
        ("d", "e", "f"),
        ("nan", "e", "f"),
        ("d", "nan", "f"),
        ("d", "e", "nan"),
        ("nan", "nan", "f"),
        ("nan", "e", "nan"),
        ("d", "nan", "nan"),
    ]
    expected_treatment = VariantList(variants=[Variant(value=val) for val in expected_treatment_values])
    expected_control = VariantList(variants=[Variant(value=val) for val in expected_control_values])
    assert sorted(comparison_pair.treatment.variants, key=lambda x: x.value) == sorted(expected_treatment.variants, key=lambda x: x.value)
    assert sorted(comparison_pair.control.variants, key=lambda x: x.value) == sorted(expected_control.variants, key=lambda x: x.value)
    assert comparison_pair.name == "(a, b, c) vs (d, e, f)"

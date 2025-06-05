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

from abvelocity.utils.gen_tuples_with_replacement import gen_tuples_with_replacement


def test_single_element():
    original_tuple = ("a",)
    replacement = "nan"
    expected = [("a",)]
    result = gen_tuples_with_replacement(original_tuple, replacement)
    assert result == expected


def test_two_elements():
    original_tuple = ("a", "b")
    replacement = "nan"
    expected = [("a", "b"), ("nan", "b"), ("a", "nan")]
    result = gen_tuples_with_replacement(original_tuple, replacement)
    assert sorted(result) == sorted(expected)


def test_three_elements():
    original_tuple = ("a", "b", "c")
    replacement = "nan"
    expected = [
        ("a", "b", "c"),
        ("nan", "b", "c"),
        ("a", "nan", "c"),
        ("a", "b", "nan"),
        ("nan", "nan", "c"),
        ("nan", "b", "nan"),
        ("a", "nan", "nan"),
    ]
    result = gen_tuples_with_replacement(original_tuple, replacement)
    assert sorted(result) == sorted(expected)


def test_no_elements():
    original_tuple = ()
    replacement = "nan"
    expected = []
    result = gen_tuples_with_replacement(original_tuple, replacement)
    assert result == expected


def test_all_elements_same():
    original_tuple = ("a", "a", "a")
    replacement = "nan"
    expected = [
        ("a", "a", "a"),
        ("nan", "a", "a"),
        ("a", "nan", "a"),
        ("a", "a", "nan"),
        ("nan", "nan", "a"),
        ("nan", "a", "nan"),
        ("a", "nan", "nan"),
    ]
    result = gen_tuples_with_replacement(original_tuple, replacement)
    assert sorted(result) == sorted(expected)

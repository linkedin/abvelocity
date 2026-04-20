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

import numpy as np
import pytest
from abvelocity.core.param.constants import CATEG_NAN_VALUE
from abvelocity.core.param.launch import Launch


def test_valid_single():
    launch = Launch(("valid",))
    assert launch.value == ("valid",)
    assert launch.name == "valid"


def test_valid_tuple():
    launch = Launch(("valid", "also_valid"))
    assert launch.value == ("valid", "also_valid")
    assert launch.name == "(valid, also_valid)"


def test_invalid_single_nan():
    with pytest.raises(ValueError, match="Invalid value nan found in tuple \\('nan',\\)"):
        Launch((CATEG_NAN_VALUE,))


def test_invalid_tuple_nan():
    with pytest.raises(ValueError, match="Invalid value nan found in tuple \\('valid', 'nan'\\)"):
        Launch(("valid", CATEG_NAN_VALUE))


def test_invalid_tuple_none():
    with pytest.raises(ValueError, match="Invalid value None found in tuple \\('valid', None\\)"):
        Launch(("valid", None), name="x")


def test_invalid_tuple_np_nan():
    with pytest.raises(ValueError, match="Invalid value nan found in tuple \\('valid', nan\\)"):
        Launch(("valid", np.nan), name="x")


def test_variant_is_consistent_with_multi():
    v = Launch(("v1", "w1", "z1"))
    assert v.is_consistent_with((None, None, None))
    assert v.is_consistent_with(("v1", None, "z1"))
    assert v.is_consistent_with((None, "w1", "z1"))
    assert not v.is_consistent_with(("v1", "w2", "u1"))
    assert not v.is_consistent_with((None, "v2", "w1"))

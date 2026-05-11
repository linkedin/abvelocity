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
"""Tests for ParamConverter base class and IdentityParamConverter."""

from abvelocity.ts.model_selection.param_converter import (
    IdentityParamConverter,
    ParamConverter,
)


def test_identity_returns_equal_dict():
    converter = IdentityParamConverter()
    assert converter.convert({"a": 1, "b": "two"}) == {"a": 1, "b": "two"}


def test_identity_returns_shallow_copy():
    """Caller can mutate the returned dict without touching the input."""
    converter = IdentityParamConverter()
    src = {"a": 1}
    out = converter.convert(src)
    out["a"] = 99
    assert src == {"a": 1}


def test_identity_callable_alias():
    """ParamConverter instances are callable; ``__call__`` is an alias for ``convert``."""
    converter = IdentityParamConverter()
    assert converter({"x": 7}) == converter.convert({"x": 7})


def test_param_converter_abstract_marker_returns_none_when_unimplemented():
    """``ParamConverter.convert`` is marked ``@abstractmethod`` but uses the
    same ``@dataclass`` + abstract pattern as :class:`TSAlgo` — abstract is
    documentary, not enforced via ABCMeta. Calling an unoverridden ``convert``
    falls through the ``...`` body and returns None. This test pins that
    behavior so the convention isn't accidentally tightened or loosened."""

    class IncompleteConverter(ParamConverter):
        pass

    converter = IncompleteConverter()
    assert converter.convert({"a": 1}) is None


def test_param_converter_subclass_can_translate_keys():
    """A real subclass can rename keys to whatever shape it wants."""

    class FlipConverter(ParamConverter):
        def convert(self, params):
            return {f"renamed_{k}": v for k, v in params.items()}

    converter = FlipConverter()
    assert converter.convert({"a": 1, "b": 2}) == {"renamed_a": 1, "renamed_b": 2}


def test_param_converter_subclass_callable():
    """Subclass instance is callable too (inherits __call__)."""

    class DoubleConverter(ParamConverter):
        def convert(self, params):
            return {k: v * 2 for k, v in params.items() if isinstance(v, (int, float))}

    converter = DoubleConverter()
    assert converter({"a": 3, "b": 4.5}) == {"a": 6, "b": 9.0}

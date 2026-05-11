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
"""Single gate for the heavy gk-side tests.

Each gk-flavored test file declares::

    from abvelocity.ts.gk_test_gate import gk_test_gate
    pytestmark = gk_test_gate

— so flipping all those tests on/off is a one-line change here (or via
the ``RUN_ALL_GK_TESTS`` env var) instead of editing every gated file.

The default is OFF (skipped) because the gk fits / silverkite training /
benchmark suites add ~10s+ minutes per file to a normal pytest run.

Two ways to flip on:

1. **Env var** — ``RUN_ALL_GK_TESTS=1 pytest …``.  No code edit, doesn't
   leak across runs.  Preferred for one-off local runs.
2. **In-file** — change :data:`RUN_ALL_GK_TESTS` below to ``True``.
   Affects every gated test file at once.  Use for sustained debugging
   sessions when re-typing the env var gets old.
"""

import os

import pytest

RUN_ALL_GK_TESTS = os.environ.get("RUN_ALL_GK_TESTS", "0") == "1"
"""Whether the heavy gk-side tests should run.  Defaults to False (off).

Toggle by setting ``RUN_ALL_GK_TESTS=1`` in the env, or by overriding
this constant in-place to ``True``.
"""

gk_test_gate = pytest.mark.skipif(
    not RUN_ALL_GK_TESTS,
    reason="heavy gk test — set RUN_ALL_GK_TESTS=1 (env var) to run",
)
"""Pytest marker.  Apply at module level via ``pytestmark = gk_test_gate``."""

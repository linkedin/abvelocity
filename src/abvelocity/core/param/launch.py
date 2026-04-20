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

from dataclasses import dataclass

import numpy as np
from abvelocity.core.param.constants import CATEG_NAN_VALUE
from abvelocity.core.param.variant import Variant


@dataclass
class Launch(Variant):
    """This represents a launch which is a combination of variants.
    The combination cannot include None / `CATEG_NAN_VALUE` ("nan") / np.nan values.
    This is because combinations which include `CATEG_NAN_VALUE` ("nan") do not represent a valid launch.
    Note that the `CATEG_NAN_VALUE` is allowed in `Variant` class to represent all possible partitions of the experiment unit space.
    This is because for some units, they may not experience any of the variants as the variant will not "trigger" for them.

    For a single experiment, use a single-element tuple, e.g., `("v1",)`, and the name will be set to the element itself, e.g., `"v1"`.
    For multi-experiments, use a tuple with multiple elements, e.g., `("v1", "w1", "x1")`, and the name will be set to, e.g., `"(v1, w1, x1)"`.

    Some valid `Launch` examples include:
        - Launch 1: ("v1",)
        - Launch 2: ("v2",)
        - Launch 3: ("control",)
        - Launch 4: ("v1", "w1", "x1")
        - Launch 5: ("v2", "w2", "x2")
        - Launch 6: ("control", "w3", "x3")
        - Launch 7: ("v1", "w1", "control")
        - Launch 8: ("control", "control", "control")

    Some examples of `Variant`s which are not valid launches include:
        - Variant 1: ("v1", "nan", "x1")
        - Variant 2: ("v2", "w2", "nan")
        - Variant 3: ("v1", "w1", "nan")
    """

    def __post_init__(self):
        super().__post_init__()  # Initialize base class (sets name for single/multi-experiments)

        invalid_values = {None, CATEG_NAN_VALUE}

        def is_invalid(value):
            return value in invalid_values or (isinstance(value, float) and np.isnan(value))

        for element in self.value:
            if is_invalid(element):
                raise ValueError(f"Invalid value {element} found in tuple {self.value}")

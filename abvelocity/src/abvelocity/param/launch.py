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

from dataclasses import dataclass

import numpy as np

from abvelocity.param.constants import CATEG_NAN_VALUE
from abvelocity.param.variant import Variant


@dataclass
class Launch(Variant):
    """This represents a launch which is a combination of variants.
    The combination cannot include None / `CATEG_NAN_VALUE` ("nan") / np.nan values.
    This is because combinations which include `CATEG_NAN_VALUE` ("nan") do not represent a valid launch.
    Note that the `CATEG_VALUE_NAN` is allowed in `Variant` class to represent all possible partitions of the experiment unit space.
    This is because for some units, they may not experience any of the variants as the variant will not "trigger" for them.

    For example when there are three experiments with:
        - Experiment 1 having variants v1, v2, control
        - Experiment 2 having variants w1, w2, w3, control
        - Experiment 3 having variants x1, x2, x3, x4, control

    Some valid `Launch` examples include:
        - Launch 1: (v1, w1, x1)
        - Launch 2: (v2, w2, x2)
        - Launch 3: (control, w3, x3)
        - Launch 4: (v1, w1, control)
        - Launch 5: (control, control, control)

    Some examples of `Variant`s which are not launches include:
        - Variant 1: (v1, None, x1)
        - Variant 2: (v2, w2, None)
        - Variant 3: (v1, w1, None)

    Note that we allowed for such Variants in Variant class because we want to have
    a way to reprsent all possible partitions of the space.
    """

    def __post_init__(self):
        super().__post_init__()  # Initialize base class

        invalid_values = {None, CATEG_NAN_VALUE}

        def is_invalid(value):
            return value in invalid_values or (isinstance(value, float) and np.isnan(value))

        if isinstance(self.value, tuple):
            for element in self.value:
                if is_invalid(element):
                    raise ValueError(f"Invalid value {element} found in tuple {self.value}")
        elif isinstance(self.value, str):
            if is_invalid(self.value):
                raise ValueError(f"Invalid value {self.value}")
        else:
            raise TypeError(f"Invalid type for value: {type(self.value)}")

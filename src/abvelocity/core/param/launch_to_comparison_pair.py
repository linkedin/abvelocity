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

from typing import Optional

from abvelocity.core.param.constants import CONTROL_LABEL
from abvelocity.core.param.launch import Launch
from abvelocity.core.param.launch_to_variant_list import launch_to_variant_list
from abvelocity.core.param.variant import ComparisonPair


def launch_to_comparison_pair(launch: Launch, control_launch: Optional[Launch] = None, name: Optional[str] = None) -> ComparisonPair:
    """
    Create a ComparisonPair from a given Launch by generating its control counterpart and converting both
    the launch and its control to VariantList objects.

    Args:
        launch: Launch which is a combination of variants across experiments, with `value` as a tuple of strings
            (e.g., `("v1",)` for single experiments with name `"v1"`, or `("v1", "w1")` for multi-experiments
            with name `"(v1, w1)"`).
        control_launch: The Launch which is used as the baseline for comparison.
            If not passed, we will assume all experiments being on control arm (`CONTROL_LABEL`) is the baseline
            launch (e.g., `("control",)` for single experiments, or `("control", "control")` for multi-experiments).
        name: Optional name for the ComparisonPair. If not provided, defaults to
            "<launch.name> launch" if control_launch is None, or "<launch.name> vs <control_launch.name>" otherwise.

    Returns:
        ComparisonPair: A ComparisonPair object containing the treatment and control VariantList objects.
    """
    # Generate treatment variants
    treatment_variants = launch_to_variant_list(launch)

    if name is None:
        if control_launch is None:
            name = f"{launch.name} launch"
        else:
            name = f"{launch.name} vs {control_launch.name}"

    # Generate control counterpart
    if control_launch is None:
        control_value = tuple(CONTROL_LABEL for _ in launch.value)
        control_launch = Launch(value=control_value)

    # Generate control variants
    control_variants = launch_to_variant_list(control_launch)

    return ComparisonPair(treatment=treatment_variants, control=control_variants, name=name)

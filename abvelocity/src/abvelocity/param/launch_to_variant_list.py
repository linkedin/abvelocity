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

from abvelocity.param.constants import CATEG_NAN_VALUE
from abvelocity.param.launch import Launch
from abvelocity.param.variant import Variant, VariantList
from abvelocity.utils.gen_tuples_with_replacement import gen_tuples_with_replacement


def launch_to_variant_list(launch: Launch) -> VariantList:
    """Converts a given launch (`Launch`) which is a combination of variants across experiments to a union of
    partitions (variants) in the experiment unit space.
    While a launch cannot have CATEG_NAN_VALUE in it's combination,
    these partitions will be allowed to have CATEG_NAN_VALUE.
    This is because the users for which not all variants across experiments are triggered are still part of the impacted unit space.

    For example when there are three experiments with:
        - Experiment 1 having variants v1, v2, control
        - Experiment 2 having variants w1, w2, w3, control
        - Experiment 3 having variants x1, x2, x3, x4, control

    This is a valid `Launch` example:
        - Launch 1: (v1, w1, x1).

    If such a combination (multi-experiment) is launched all the following partitions will be impacted:
        - (v1, w1, x1)
        - (nan, w1, x1)
        - (v1, nan, x1)
        - (v1, w1, nan)
        - (nan, nan, x1)
        - (nan, w1, nan)
        - (v1, nan, nan)

    Note that `nan` in the above example represents `CATEG_NAN_VALUE`.
    Also note that these are the combinations where variants in each experiment are replaced with `nan` with
    the exception of all being replaced with nan:
        - (nan, nan, nan)
    This is because this combination is not impacted.

    In the case where a launch is a single experiment, `launch.value` will be a string.
    In that case, that value itself is the only possibility. For example:
        - Launch 2: "v1"
    In this case, the only partition impacted is: "v1".

    Based on the above explanation, this function will map the launch to a union of these partitions which is encoded in a `VariantList`.

    Args:
        launch: Launch which is a combination of variants across experiments.

    Returns:
        VariantList: A union of partitions in the experiment unit space.

    """
    if isinstance(launch.value, str):  # this is the case for single experiment launches.
        variants = [Variant(value=launch.value)]
    else:  # this is the case for multi-experiment launches where a tuple encodes the combination.
        variant_values = gen_tuples_with_replacement(
            original_tuple=launch.value, replacement=CATEG_NAN_VALUE
        )
        variants = [Variant(value=variant) for variant in variant_values]

    return VariantList(variants=variants)

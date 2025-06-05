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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from abvelocity.param.constants import CATEG_NAN_VALUE


@dataclass
class Variant:
    value: str | tuple[str]
    """The variant label/value in the data which can be a:

        - `str` (for simple experiments) or
        - a `tuple[str]` (for multi-experiments).

    """
    name: Optional[str] = None
    """The name of the variant."""

    def __post_init__(self):
        if self.name is None:
            if isinstance(self.value, str):
                self.name = self.value
            else:
                self.name = "(" + ", ".join(self.value) + ")"

    def __eq__(self, other):
        """We only require the values being equal."""
        if not isinstance(other, Variant):
            return NotImplemented
        # Compare only the value
        return self.value == other.value


@dataclass
class TriggerState:
    value: Optional[bool | tuple[bool]] = None
    """The trigger state of the variant in the data where its `value` can be a:

            - `bool` (for simple experiments) or
            - a `tuple[bool]` (for multi-experiments).

    This determines which experiment is triggered for that unit of the experiment data.

    It also includes an `overall_state` which is a boolean value indicating whether any of the components of the tuple is `True`.
    This is useful for multi-experiments where we want to know if any of the experiments are triggered.
    Note that `overall_value` is inferred from the value if not passed explicitly.
    """
    overall_value: Optional[bool] = None
    """
    bool indicating whether any of the components of the tuple is `True`.
    This is useful for multi-experiments where we want to know if any of the experiments are triggered.
    """
    name: Optional[str] = None
    """The name of the trigger state."""

    def __post_init__(self):
        if self.name is None:
            if isinstance(self.value, bool) or isinstance(self.value, np.bool_):
                self.name = str(self.value)
            else:
                self.name = "(" + ", ".join([str(trigger) for trigger in self.value]) + ")"

        if self.value is not None and self.overall_value is None:
            if isinstance(self.value, bool) or isinstance(self.value, np.bool_):
                self.overall_value = self.value
            else:
                self.overall_value = any(self.value)


@dataclass
class VariantList:
    variants: list[Variant]
    """The list of variants in the list."""
    name: Optional[str] = None
    """The name of the variant list."""

    def __post_init__(self):
        if len(self.variants) == 0:
            raise ValueError("`.variants` in `VariantList` field cannot be an empty list.")
        first_type = type(self.variants[0].value)
        tuple_length = (
            len(self.variants[0].value) if isinstance(self.variants[0].value, tuple) else None
        )

        for variant in self.variants:
            if type(variant.value) != first_type:
                raise ValueError(
                    "All elements in the VariantList must be of the same type (either all str or all tuple)."
                )
            if isinstance(variant.value, tuple):
                if len(variant.value) != tuple_length:
                    raise ValueError(
                        "All tuple elements in the VariantList must be of the same size."
                    )

        if self.name is None:
            self.name = "[" + ", ".join([variant.name for variant in self.variants]) + "]"


@dataclass
class ComparisonPair:
    treatment: VariantList
    """The list of variants in treatment.
    Note that the treatment might be a collection of variants."""
    control: VariantList
    """The list of variants in the control.
    Note that the control might be a collection of variants."""
    name: Optional[str] = None
    """The name of the comparison pair."""

    def remove_common_variants(self):
        """
        - Removes common variants from both arms.
        - Throws an error if either or both of arms have zero variants.
            This could happen after removing common variants (which is also user input error)."""

        # Extract the variants from both arms
        treatment_variant_values = [Variant.value for Variant in self.treatment.variants]
        control_variant_values = [Variant.value for Variant in self.control.variants]

        # Error handling for having non-empty variants on both arms, after removal.
        # If either or both arms have empty variants, we raise error.
        # This implies one is subset of the other.
        set1, set2 = set(treatment_variant_values), set(control_variant_values)
        diff1 = set1 - set2  # Elements in the first list but not in the second
        diff2 = set2 - set1  # Elements in the first list but not in the second
        if not diff1 or not diff2:
            raise ValueError(
                "At least one of the arms does not have any variants."
                " This might happen due to removal of common variants."
                " Which implies one of the arms variants are subset of the other (or both are the same)."
                " These are the original variants passed by user:"
                f" treatment: {treatment_variant_values}"
                f" control: {control_variant_values}"
            )

        # `.name` is not updated by design to be faithful to previous name
        # This does have the caveat that if the name lists old variants adn we drop some,
        # it can create some confusion.
        # But it might be better to keep it, as the comparison still is a comparison
        # for the original lists of variants and the drop is only done to increase statistical power
        self.treatment = VariantList(
            variants=[
                variant
                for variant in self.treatment.variants
                if variant.value not in control_variant_values
            ],
            name=self.treatment.name,
        )

        self.control = VariantList(
            variants=[
                variant
                for variant in self.control.variants
                if variant.value not in treatment_variant_values
            ],
            name=self.control.name,
        )

    def __post_init__(self):
        """
        This accomplishes:
            - Throws an error if either or both of arms have zero variants.
                This could happen after removing common variants.
            - Removes common variants.
            - Assigns a name based on the input, if it's not passed.
        """
        # Remove the common variants.
        # Does error handling.
        self.remove_common_variants()

        combined_variants = self.treatment.variants + self.control.variants

        first_type = type(combined_variants[0].value)
        tuple_length = (
            len(combined_variants[0].value)
            if isinstance(combined_variants[0].value, tuple)
            else None
        )

        for variant in combined_variants:
            if type(variant.value) != first_type:
                raise ValueError(
                    "All elements in treatment and control must be of the same type (either all str or all tuple)."
                )
            if isinstance(variant.value, tuple):
                if len(variant.value) != tuple_length:
                    raise ValueError(
                        "All tuple elements in treatment and control must be of the same size."
                    )

        if self.name is None:
            self.name = f"{self.treatment.name} versus {self.control.name}"


def variant_to_trigger_state(variant: Variant) -> TriggerState:
    """Converts a given variant to a trigger state.
    Example 1: (single experiment)
        - variant.value: "v1"
        - trigger_state.value: True

    Example 2: (multi-experiment)
        - variant.value: ("v1", "w1", "x1")
        - trigger_state.value: (True, True, True)

    Example 3: (multi-experiment)
        - variant.value: ("v1", "w1", "nan")
        - trigger_state.value: (True, True, False)

    Args:
        variant: The variant to convert.

    Returns:
        The trigger state.
    """

    if isinstance(variant.value, str):
        if variant.value == CATEG_NAN_VALUE:
            return TriggerState(value=False)
        return TriggerState(value=True)

    trigger_state_value = tuple([True if v != CATEG_NAN_VALUE else False for v in variant.value])

    return TriggerState(value=trigger_state_value)

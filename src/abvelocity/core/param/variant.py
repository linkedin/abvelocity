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
from typing import Optional, Tuple

from abvelocity.core.param.constants import CATEG_NAN_VALUE
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class Variant(DataClassJSONMixin):
    value: tuple[str]
    """The variant label/value in the data, a tuple of strings.

    For a single experiment, use a single-element tuple, e.g., `("v1",)`, and the name will be set to the element itself, e.g., `"v1"`.
    For multi-experiments, use a tuple with multiple elements, e.g., `("v1", "w1", "x1")`, and the name will be set to, e.g., `"(v1, w1, x1)"`.
    """
    name: Optional[str] = None
    """The name of the variant."""

    def __post_init__(self):
        if self.name is None:
            if len(self.value) == 1:
                self.name = self.value[0]
            else:
                self.name = "(" + ", ".join(self.value) + ")"

    def __eq__(self, other):
        """We only require the values being equal."""
        if not isinstance(other, Variant):
            return NotImplemented
        return self.value == other.value

    def is_consistent_with(self, other: Tuple[Optional[str], ...]) -> bool:
        """Checks if this variant is consistent with another tuple of the same size.
        Consistency means each element in this variant's value matches the corresponding element
        in the input tuple, or the input tuple's element is None.

        Args:
            other: A tuple of strings or None values to compare against.

        Returns:
            bool: True if the tuples are consistent, False otherwise.

        Raises:
            ValueError: If the input tuple has a different size than this variant's value.
        """
        if len(other) != len(self.value):
            raise ValueError(f"Input tuple size {len(other)} does not match variant value size {len(self.value)}")
        return all(other_val is None or self_val == other_val for self_val, other_val in zip(self.value, other))

    def __lt__(self, other):
        """Allows sorting based on the 'value' tuple."""
        if not isinstance(other, Variant):
            return NotImplemented
        return self.value < other.value


@dataclass
class TriggerState(DataClassJSONMixin):
    value: Optional[tuple[bool]] = None
    """The trigger state of the variant in the data, a tuple of booleans.

    For a single experiment, use a single-element tuple, e.g., `(True,)`, and the name will be set to the element itself, e.g., `"True"`.
    For multi-experiments, use a tuple with multiple elements, e.g., `(True, True, False)`, and the name will be set to, e.g., `"(True, True, False)"`.
    This determines which experiment is triggered for that unit of the experiment data.

    It also includes an `overall_value` which is a boolean value indicating whether any of the components of the tuple is `True`.
    This is useful for multi-experiments where we want to know if any of the experiments are triggered.
    Note that `overall_value` is inferred from the value if not passed explicitly.
    """
    overall_value: Optional[bool] = None
    """bool indicating whether any of the components of the tuple is `True`."""
    name: Optional[str] = None
    """The name of the trigger state."""

    def __post_init__(self):
        if self.name is None and self.value is not None:
            if len(self.value) == 1:
                self.name = str(self.value[0])
            else:
                self.name = "(" + ", ".join([str(trigger) for trigger in self.value]) + ")"

        if self.value is not None and self.overall_value is None:
            self.overall_value = any(self.value)

    def __lt__(self, other):
        """Allows sorting based on the 'value' tuple."""
        if not isinstance(other, TriggerState):
            return NotImplemented
        return self.value < other.value


@dataclass
class VariantList(DataClassJSONMixin):
    variants: list[Variant]
    """The list of variants in the list."""
    name: Optional[str] = None
    """The name of the variant list."""

    def __post_init__(self):
        if len(self.variants) == 0:
            raise ValueError("`.variants` in `VariantList` field cannot be an empty list.")
        tuple_length = len(self.variants[0].value)

        for variant in self.variants:
            if not isinstance(variant.value, tuple):
                raise ValueError("All elements in the VariantList must be tuples.")
            if len(variant.value) != tuple_length:
                raise ValueError("All tuple elements in the VariantList must be of the same size.")

        if self.name is None:
            self.name = "[" + ", ".join([variant.name for variant in self.variants]) + "]"


@dataclass
class ComparisonPair(DataClassJSONMixin):
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
        treatment_variant_values = [variant.value for variant in self.treatment.variants]
        control_variant_values = [variant.value for variant in self.control.variants]

        set1, set2 = set(treatment_variant_values), set(control_variant_values)
        diff1 = set1 - set2
        diff2 = set2 - set1
        if not diff1 or not diff2:
            raise ValueError(
                "At least one of the arms does not have any variants."
                " This might happen due to removal of common variants."
                " Which implies one of the arms variants are subset of the other (or both are the same)."
                f" treatment: {treatment_variant_values}"
                f" control: {control_variant_values}"
            )

        self.treatment = VariantList(
            variants=[variant for variant in self.treatment.variants if variant.value not in control_variant_values],
            name=self.treatment.name,
        )

        self.control = VariantList(
            variants=[variant for variant in self.control.variants if variant.value not in treatment_variant_values],
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
        self.remove_common_variants()

        combined_variants = self.treatment.variants + self.control.variants
        tuple_length = len(combined_variants[0].value)

        for variant in combined_variants:
            if not isinstance(variant.value, tuple):
                raise ValueError("All elements in treatment and control must be tuples.")
            if len(variant.value) != tuple_length:
                raise ValueError("All tuple elements in treatment and control must be of the same size.")

        if self.name is None:
            self.name = f"{self.treatment.name} versus {self.control.name}"


def variant_to_trigger_state(variant: Variant) -> TriggerState:
    """Converts a given variant to a trigger state.

    Example 1: (single experiment)
        - variant.value: ("v1",)
        - trigger_state.value: (True,)
        - trigger_state.name: "True"

    Example 2: (multi-experiment)
        - variant.value: ("v1", "w1", "x1")
        - trigger_state.value: (True, True, True)
        - trigger_state.name: "(True, True, True)"

    Example 3: (multi-experiment)
        - variant.value: ("v1", "w1", "nan")
        - trigger_state.value: (True, True, False)
        - trigger_state.name: "(True, True, False)"

    Args:
        variant: The variant to convert, with `value` as a tuple of strings (e.g., `("v1",)` for single experiments).

    Returns:
        The trigger state.
    """
    trigger_state_value = tuple(True if v != CATEG_NAN_VALUE else False for v in variant.value)
    return TriggerState(value=trigger_state_value)

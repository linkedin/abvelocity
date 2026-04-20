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

from typing import Optional


def swi_ratio(
    delta: tuple,
    impacted_pop_count: float,
    control_metric_total: Optional[tuple],
    complement_metric_total: Optional[tuple] = None,
) -> tuple:
    """Returns (swi_raw, swi_perc) for a ratio metric.

    Uses the potential outcomes framework: ``numer_pop_total_given_control / denom_pop_total_given_control``
    is the population-level ratio in the counterfactual control world (all triggered
    units stay on control); ``numer_pop_total_given_treat / denom_pop_total_given_treat``
    is the population-level ratio in the treatment world (triggered units receive treatment).

        numer_pop_total_given_control = control_numer_total + complement_numer_total
        denom_pop_total_given_control = control_denom_total + complement_denom_total
        numer_pop_total_given_treat   = numer_pop_total_given_control + delta_numer * impacted_pop_count
        denom_pop_total_given_treat   = denom_pop_total_given_control + delta_denom * impacted_pop_count
        swi_raw           = numer_pop_total_given_treat / denom_pop_total_given_treat - numer_pop_total_given_control / denom_pop_total_given_control
        swi_relative_perc = swi_raw / (numer_pop_total_given_control / denom_pop_total_given_control) * 100

    Args:
        delta: Tuple (delta_numer, delta_denom) — per-unit treatment effect.
        impacted_pop_count: Total triggered/targeted population.
        control_metric_total: Tuple (control_numer_total, control_denom_total) for the control arm.
        complement_metric_total: Tuple (numer, denom) for the complement set — the units
            not impacted/triggered by the experiment. Combined with ``control_metric_total``
            to reconstruct the full population baseline. None if unavailable.

    Returns:
        (swi_raw, swi_perc), both NaN when inputs are invalid.
    """
    nan = float("nan")

    if not isinstance(control_metric_total, tuple):
        raise ValueError(f"swi_ratio: control_metric_total must be a (numer, denom) tuple, got {type(control_metric_total)}")

    delta_numer, delta_denom = delta
    control_numer_total, control_denom_total = control_metric_total

    complement_numer_total, complement_denom_total = 0.0, 0.0
    if complement_metric_total is not None:
        complement_numer_total, complement_denom_total = complement_metric_total

    numer_pop_total_given_control = control_numer_total + complement_numer_total
    denom_pop_total_given_control = control_denom_total + complement_denom_total

    if not denom_pop_total_given_control:
        return nan, nan

    numer_pop_total_given_treat = numer_pop_total_given_control + delta_numer * impacted_pop_count
    denom_pop_total_given_treat = denom_pop_total_given_control + delta_denom * impacted_pop_count

    ratio_pop_total_given_control = numer_pop_total_given_control / denom_pop_total_given_control
    ratio_pop_total_given_treat = numer_pop_total_given_treat / denom_pop_total_given_treat if denom_pop_total_given_treat else nan

    swi_raw = ratio_pop_total_given_treat - ratio_pop_total_given_control
    swi_perc = swi_raw / ratio_pop_total_given_control * 100 if ratio_pop_total_given_control else nan
    return swi_raw, swi_perc

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

import math
from typing import Optional


def swi_simple(
    delta: float,
    impacted_pop_count: float,
    control_metric_total: Optional[float],
    complement_metric_total: Optional[float] = None,
) -> tuple:
    """Returns (swi_raw, swi_perc) for a simple (non-ratio) metric.

    Uses the potential outcomes framework: ``numer_pop_total_given_control`` is the
    population-level metric in the counterfactual control world (all triggered
    units stay on control); the treatment world shifts it by
    ``delta * impacted_pop_count``.

        numer_pop_total_given_control = control_metric_total + complement_metric_total
        swi_raw           = delta * impacted_pop_count
        swi_relative_perc = swi_raw / numer_pop_total_given_control * 100

    Args:
        delta: Per-unit treatment effect (treatment_avg - control_avg).
        impacted_pop_count: Total triggered/targeted population.
        control_metric_total: Total aggregated metric for the control arm.
        complement_metric_total: Aggregated metric for the complement set — the units
            not impacted/triggered by the experiment. Combined with ``control_metric_total``
            to reconstruct the full population baseline. None if unavailable.

    Returns:
        (swi_raw, swi_perc) where swi_raw = delta * impacted_pop_count and
        swi_perc = swi_raw / numer_pop_total_given_control * 100 (NaN when unavailable).
    """
    swi_raw = delta * impacted_pop_count

    numer_pop_total_given_control = None
    if control_metric_total is not None and not math.isnan(control_metric_total):
        numer_pop_total_given_control = control_metric_total
        if complement_metric_total is not None:
            numer_pop_total_given_control = control_metric_total + complement_metric_total

    swi_perc = swi_raw / numer_pop_total_given_control * 100 if numer_pop_total_given_control else float("nan")
    return swi_raw, swi_perc

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
#
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

import numpy as np
import pytest
from abvelocity.core.stats.swi.swi_ratio import swi_ratio


def test_ratio_swi_raw():
    control_metric_total = (10000.0, 5000.0)
    complement_metric_total = (8000.0, 4000.0)
    delta = (2.0, 1.0)
    impacted_pop_count = 200

    numer_pop_total_given_control = control_metric_total[0] + complement_metric_total[0]
    denom_pop_total_given_control = control_metric_total[1] + complement_metric_total[1]
    expected_swi_raw = (numer_pop_total_given_control + delta[0] * impacted_pop_count) / (
        denom_pop_total_given_control + delta[1] * impacted_pop_count
    ) - numer_pop_total_given_control / denom_pop_total_given_control

    swi_raw, _ = swi_ratio(
        delta=delta,
        impacted_pop_count=impacted_pop_count,
        control_metric_total=control_metric_total,
        complement_metric_total=complement_metric_total,
    )
    assert np.isclose(swi_raw, expected_swi_raw)


def test_ratio_swi_perc():
    control_metric_total = (10000.0, 5000.0)
    complement_metric_total = (8000.0, 4000.0)
    delta = (2.0, 1.0)
    impacted_pop_count = 200

    numer_pop_total_given_control = control_metric_total[0] + complement_metric_total[0]
    denom_pop_total_given_control = control_metric_total[1] + complement_metric_total[1]
    ratio_pop_total_given_control = numer_pop_total_given_control / denom_pop_total_given_control
    ratio_pop_total_given_treat = (numer_pop_total_given_control + delta[0] * impacted_pop_count) / (
        denom_pop_total_given_control + delta[1] * impacted_pop_count
    )
    expected_perc = (ratio_pop_total_given_treat - ratio_pop_total_given_control) / ratio_pop_total_given_control * 100

    _, swi_perc = swi_ratio(
        delta=delta,
        impacted_pop_count=impacted_pop_count,
        control_metric_total=control_metric_total,
        complement_metric_total=complement_metric_total,
    )
    assert np.isclose(swi_perc, expected_perc)


def test_ratio_swi_no_complement_uses_ctrl_as_baseline():
    control_metric_total = (10000.0, 5000.0)
    delta = (2.0, 1.0)
    impacted_pop_count = 200

    expected_swi_raw = (control_metric_total[0] + delta[0] * impacted_pop_count) / (
        control_metric_total[1] + delta[1] * impacted_pop_count
    ) - control_metric_total[0] / control_metric_total[1]

    swi_raw, _ = swi_ratio(
        delta=delta,
        impacted_pop_count=impacted_pop_count,
        control_metric_total=control_metric_total,
    )
    assert np.isclose(swi_raw, expected_swi_raw)


def test_ratio_swi_nan_when_denom_total_zero():
    swi_raw, swi_perc = swi_ratio(
        delta=(2.0, 1.0),
        impacted_pop_count=200,
        control_metric_total=(10000.0, 0.0),
        complement_metric_total=(8000.0, 0.0),
    )
    assert math.isnan(swi_raw)
    assert math.isnan(swi_perc)


def test_ratio_swi_raises_when_control_metric_total_not_tuple():
    with pytest.raises(ValueError, match="control_metric_total must be a"):
        swi_ratio(delta=(2.0, 1.0), impacted_pop_count=200, control_metric_total=10000.0)

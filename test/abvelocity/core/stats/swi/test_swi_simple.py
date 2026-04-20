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
from abvelocity.core.stats.swi.swi_simple import swi_simple


def test_simple_swi_raw():
    delta, impacted_pop_count = 0.05, 1000
    swi_raw, _ = swi_simple(delta=delta, impacted_pop_count=impacted_pop_count, control_metric_total=None)
    assert np.isclose(swi_raw, delta * impacted_pop_count)


def test_simple_swi_perc_with_complement():
    delta, impacted_pop_count = 0.05, 1000
    control_metric_total, complement_metric_total = 20000.0, 30000.0
    _, swi_perc = swi_simple(
        delta=delta,
        impacted_pop_count=impacted_pop_count,
        control_metric_total=control_metric_total,
        complement_metric_total=complement_metric_total,
    )
    assert np.isclose(
        swi_perc,
        delta * impacted_pop_count / (control_metric_total + complement_metric_total) * 100,
    )


def test_simple_swi_perc_no_complement_uses_control_only():
    delta, impacted_pop_count = 0.05, 1000
    control_metric_total = 20000.0
    _, swi_perc = swi_simple(
        delta=delta,
        impacted_pop_count=impacted_pop_count,
        control_metric_total=control_metric_total,
    )
    assert np.isclose(swi_perc, delta * impacted_pop_count / control_metric_total * 100)


def test_simple_swi_perc_nan_when_no_control_no_complement():
    _, swi_perc = swi_simple(delta=0.05, impacted_pop_count=1000, control_metric_total=None)
    assert math.isnan(swi_perc)


def test_simple_swi_perc_nan_when_numer_total_is_zero():
    _, swi_perc = swi_simple(
        delta=0.05,
        impacted_pop_count=1000,
        control_metric_total=0.0,
        complement_metric_total=0.0,
    )
    assert math.isnan(swi_perc)

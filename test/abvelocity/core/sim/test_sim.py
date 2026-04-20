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

import os
from pathlib import Path

import pytest
from abvelocity.core.sim.examples import simulate_data_multi1, simulate_data_three1, simulate_data_uni1


@pytest.fixture
def write_path():
    return Path(__file__).parents[4].joinpath("docs/static/test-results/sim/")


def test_run(write_path):
    sim = simulate_data_multi1()

    assert len(sim.population_df) == 1000

    for expt_data in sim.expt_data_list:
        expt_df = expt_data[0]
        assert len(expt_df) == 1000

    assert sim.expt_df.shape == (1000, 7)
    assert sim.expt_metric_df.shape == (1000, 9)

    path = f"{write_path}/sim1/"
    os.makedirs(path, exist_ok=True)

    log_file = f"{path}/sim_logs.txt"
    with open(log_file, "w") as file:
        sim.publish(file=file)


def test_run_uni(write_path):
    sim = simulate_data_uni1()
    assert len(sim.population_df) == 1000

    for expt_data in sim.expt_data_list:
        expt_df = expt_data[0]
        assert len(expt_df) == 900

    assert sim.expt_df.shape == (900, 6)
    assert sim.expt_metric_df.shape == (900, 8)

    path = f"{write_path}/sim1/"
    os.makedirs(path, exist_ok=True)

    log_file = f"{path}/uni_sim_logs.txt"
    with open(log_file, "w") as file:
        sim.publish(file=file)


def test_run_three(write_path):
    sim = simulate_data_three1()

    assert len(sim.population_df) == 1000

    for expt_data in sim.expt_data_list:
        expt_df = expt_data[0]
        assert len(expt_df) == 900

    assert sim.expt_df.shape == (1000, 8)
    assert sim.expt_metric_df.shape == (1000, 10)

    path = f"{write_path}/sim1/"
    os.makedirs(path, exist_ok=True)

    log_file = f"{path}/sim_three_logs.txt"
    with open(log_file, "w") as file:
        sim.publish(file=file)

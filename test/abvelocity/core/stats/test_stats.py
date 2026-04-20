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

from abvelocity.core.stats.stats import DELTA_STATS_OUTPUT_NAMES, DeltaStats, TwoSampleTest, UnivarStats


def test_univar_stats():
    """Tests `UnivarStats` class."""
    stats = UnivarStats(name="Variable", mean=10.0, sd=2.0, sample_count=100, sum=1000.0)

    assert stats.name == "Variable"
    assert stats.mean == 10.0
    assert stats.sd == 2.0
    assert stats.sample_count == 100
    assert stats.sum == 1000.0


def test_two_sample_test():
    """Tests `TwoSampleTest` class."""
    treatment_stats = UnivarStats(name="Test", mean=20.0, sd=3.0, sample_count=50)

    control_stats = UnivarStats(name="Baseline", mean=15.0, sd=2.5, sample_count=50)

    test = TwoSampleTest(treatment_stats=treatment_stats, control_stats=control_stats, ci_coverage=0.9)
    assert test.treatment_stats == treatment_stats
    assert test.control_stats == control_stats
    assert test.ci_coverage == 0.9
    assert test.same_impacted_population is True
    assert test.triggered_population_diff_thresh == 0.1


def test_delta_stats():
    """Tests `DeltaStats` class."""
    stats = DeltaStats(
        delta=5.0,
        delta_percent=25.0,
        ci=1.0,
        ci_percent=5.0,
        delta_std=2.0,
        z_value=1.96,
        p_value=0.05,
    )
    assert stats.delta == 5.0
    assert stats.delta_percent == 25.0


def test_delta_stats_output_names():
    """Tests `DELTA_STATS_OUTPUT_NAMES` list."""
    assert DELTA_STATS_OUTPUT_NAMES == [
        "delta",
        "delta(%)",
        "CI",
        "CI(%)",
        "delta_std",
        "z_value",
        "p_value",
    ]

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

import pandas as pd
import plotly.graph_objects as go
from abvelocity.core.utils.plot_conditional_variant_dist import plot_conditional_variant_dist


def test_plot_conditional_variant_dist_univariate():
    """
    Tests the univariate case: expects 1 figure showing marginal distribution.
    """
    data = pd.DataFrame(
        [
            {"variant": ("control",), "variant_count": 500},
            {"variant": ("treat1",), "variant_count": 300},
            {"variant": ("treat2",), "variant_count": 200},
        ]
    )

    # Create and test the plot
    figs = plot_conditional_variant_dist(data, count_column="variant_count", dim_names=["My Ex 1"])

    # Expect exactly 1 figure for the univariate case
    assert isinstance(figs, dict)
    assert len(figs) == 1
    assert 1 in figs
    assert isinstance(figs[1], go.Figure)
    # figs[1].show()


def test_plot_conditional_variant_dist_pair():
    """
    Tests the standard 2-dimensional case: expects 2 figures (one conditioned on V1, one on V2).
    """
    data = pd.DataFrame(
        [
            {"variant": ("control", "control"), "variant_count": 400},
            {"variant": ("control", "treat1"), "variant_count": 100},
            {"variant": ("treat2", "control"), "variant_count": 50},
            {"variant": ("treat2", "treat1"), "variant_count": 50},
        ]
    )

    # Create and test the plot
    figs = plot_conditional_variant_dist(data, count_column="variant_count", dim_names=["Ex A", "Ex B"])

    # Expect exactly 2 figures
    assert isinstance(figs, dict)
    assert len(figs) == 2
    assert 1 in figs and 2 in figs
    assert isinstance(figs[1], go.Figure)
    assert isinstance(figs[2], go.Figure)
    # figs[1].show()
    # figs[2].show()


def test_plot_conditional_variant_dist_trio():
    """
    Tests the 3-dimensional case: expects 3 figures.
    """
    data = pd.DataFrame(
        [
            {"variant": ("c", "c", "c"), "variant_count": 100},
            {"variant": ("c", "t", "c"), "variant_count": 50},
            {"variant": ("t", "c", "t"), "variant_count": 20},
            {"variant": ("t", "t", "t"), "variant_count": 30},
            {"variant": ("c", "c", "t"), "variant_count": 10},
        ]
    )

    # Create and test the plot
    figs = plot_conditional_variant_dist(data, count_column="variant_count", dim_names=["D1", "D2", "D3"])

    # Expect exactly 3 figures
    assert isinstance(figs, dict)
    assert len(figs) == 3
    assert 1 in figs and 2 in figs and 3 in figs
    assert isinstance(figs[1], go.Figure)
    assert isinstance(figs[2], go.Figure)
    assert isinstance(figs[3], go.Figure)
    # figs[1].show()
    # figs[2].show()
    # figs[3].show()


def test_plot_conditional_variant_dist_empty_df():
    """
    Tests handling of an empty input DataFrame.
    """
    data = pd.DataFrame(columns=["variant", "variant_count"])

    # Expect an empty dictionary for empty data
    figs = plot_conditional_variant_dist(data, count_column="variant_count")
    assert isinstance(figs, dict)
    assert len(figs) == 0

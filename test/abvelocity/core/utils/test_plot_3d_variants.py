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
from abvelocity.core.utils.plot_3d_variants import plot_3d_variants


def test_plot_3d_variants_basic():
    # Sample data with basic 3D tuples
    data = pd.DataFrame(
        [
            {"variant": ("control", "control", "control"), "variant_count": 4093768},
            {"variant": ("control", "nan", "enabled"), "variant_count": 8619153},
            {"variant": ("enabled", "enabled", "nan"), "variant_count": 225756},
            {"variant": ("enabled", "nan", "control"), "variant_count": 443548},
            {"variant": ("nan", "control", "enabled"), "variant_count": 1028},
            {"variant": ("nan", "enabled", "control"), "variant_count": 34},
        ]
    )

    # Create and test the plot
    fig = plot_3d_variants(data, count_column="variant_count", title="Test: Basic 3D Variants")
    assert fig is not None
    # fig.show()


def test_plot_3d_variants_extended_labels():
    # Sample data with additional labels to test flexibility
    data = pd.DataFrame(
        [
            {"variant": ("nan", "nan", "nan"), "variant_count": 1000},
            {"variant": ("control", "variant1", "variant2"), "variant_count": 5000},
            {"variant": ("enabled", "variant3", "nan"), "variant_count": 3000},
            {"variant": ("variant4", "control", "variant5"), "variant_count": 2000},
            {"variant": ("variant6", "enabled", "control"), "variant_count": 1500},
            {"variant": ("nan", "variant7", "variant8"), "variant_count": 800},
        ]
    )

    # Create and test the plot
    fig = plot_3d_variants(data, count_column="variant_count", title="Test: Extended 3D Labels")
    assert fig is not None
    # fig.show()

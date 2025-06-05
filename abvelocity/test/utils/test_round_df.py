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

from abvelocity.utils.round_df import round_df, round_float_cols, round_tuple_cols


def test_round_float_cols():
    """Tests `round_float_cols` function."""
    df = pd.DataFrame({"col1": [1.123456789, 2.123456789], "col2": [3.123456789, 4.123456789]})

    round_float_cols(df, rounding_digits=2)

    assert df["col1"].tolist() == [1.12, 2.12]
    assert df["col2"].tolist() == [3.12, 4.12]

    df = pd.DataFrame({"col1": [1.123456789, 2.123456789], "col2": [3.123456789, 4.123456789]})

    round_float_cols(df, rounding_digits=2, cols=["col1"])

    assert df["col1"].tolist() == [1.12, 2.12]
    assert df["col2"].tolist() == [3.123456789, 4.123456789]


def test_round_tuple_cols():
    """Tests `round_tuple_cols` function."""
    df = pd.DataFrame(
        {
            "col1": [(1.123456789, 2.123456789), (3.123456789, 4.123456789)],
            "col2": [(5.123456789, 6.123456789), (7.123456789, 8.123456789)],
        }
    )

    round_tuple_cols(df, rounding_digits=2)

    assert df["col1"].tolist() == [(1.12, 2.12), (3.12, 4.12)]
    assert df["col2"].tolist() == [(5.12, 6.12), (7.12, 8.12)]


def test_round_df():
    """Tests `round_df` function."""
    df = pd.DataFrame(
        {
            "col1": [1.123456789, 2.123456789],
            "col2": [3.123456789, 4.123456789],
            "col3": [(5.123456789, 6.123456789), (7.123456789, 8.123456789)],
        }
    )

    round_df(df, rounding_digits=2)

    assert df["col1"].tolist() == [1.12, 2.12]
    assert df["col2"].tolist() == [3.12, 4.12]
    assert df["col3"].tolist() == [(5.12, 6.12), (7.12, 8.12)]

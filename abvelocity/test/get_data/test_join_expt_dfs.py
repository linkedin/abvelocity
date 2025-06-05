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

import numpy as np
import pandas as pd

from abvelocity.get_data.data_container import DataContainer
from abvelocity.get_data.join_expt_dfs import join_expt_dfs
from abvelocity.param.constants import CATEG_NAN_VALUE


def test_join_expt_dfs():
    """Tests `join_expt-dfs`."""
    df1 = pd.DataFrame(
        {"memberid": [1, 2, 3], "variant": ["1", "2", "3"], "trigger_date": ["1", "2", "3"]}
    )

    df2 = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant": ["1", "2", np.nan, "2"],
            "trigger_date": ["1", "2", "3", np.nan],
        }
    )

    df3 = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant": ["11", "22", "33", "44"],
            "trigger_date": ["11", "22", "33", "44"],
        }
    )

    dc_list = [
        DataContainer(df=df1.copy(), is_df=True),
        DataContainer(df=df2.copy(), is_df=True),
        DataContainer(df=df3.copy(), is_df=True),
    ]

    expected_df = pd.DataFrame(
        {
            "memberid": [1, 2, 3, 4],
            "variant": [
                ("1", "1", "11"),
                ("2", "2", "22"),
                ("3", CATEG_NAN_VALUE, "33"),
                (CATEG_NAN_VALUE, "2", "44"),
            ],
            "trigger_date": [
                ("1", "1", "11"),
                ("2", "2", "22"),
                ("3", "3", "33"),
                (CATEG_NAN_VALUE, CATEG_NAN_VALUE, "44"),
            ],
        }
    )

    obtained_dc = join_expt_dfs(dc_list=dc_list, drop_numbered_cols=True)

    pd.testing.assert_frame_equal(obtained_dc.df, expected_df)

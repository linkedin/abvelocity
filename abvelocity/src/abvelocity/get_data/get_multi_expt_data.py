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

from typing import Callable, Optional

from abvelocity.get_data.cursor import Cursor
from abvelocity.get_data.data_container import DataContainer
from abvelocity.get_data.get_asgmnt_data import get_asgmnt_data_multi
from abvelocity.get_data.get_expt_stats import get_expt_stats
from abvelocity.get_data.join_expt_dfs import join_expt_dfs
from abvelocity.param.expt_info import MultiExptInfo


def get_multi_expt_data(
    cursor: Cursor,
    multi_expt_info: MultiExptInfo,
    expt_asgmnt_table: str,
    get_asgmnt_query: Callable,
    condition: Optional[str] = None,
) -> DataContainer:
    """This function gets the multi-experiment data and returns a DataContainer.

    Args:
        cursor: A database cursor.
        multi_expt_info: includes the experiments information.
        expt_asgmnt_table: Table which includes expt data.
        get_asgmnt_query: Callable which creates expt query.

    Returns:
        expt_dc: The DataContainer which contains experiment data.

    Alters:
        multi_expt_info: This function will amend `multi_expt_info` by adding `derived_stats` field.
            It will add `derived_stats` also to all  `expt_info` in the `expt_info_list`.

    """

    print(f"condition: {condition}")

    """Gets User Experiment Assignment Data (Expt Data)"""
    expt_dc_list = get_asgmnt_data_multi(
        expt_asgmnt_table=expt_asgmnt_table,
        get_asgmnt_query=get_asgmnt_query,
        multi_expt_info=multi_expt_info,
        cursor=cursor,
        condition=condition,
    )
    print(f"\n *** expt_dc_list:\n {expt_dc_list}")
    expt_dc = join_expt_dfs(expt_dc_list)

    multi_expt_info.derived_stats = get_expt_stats(df=expt_dc.df)

    return expt_dc

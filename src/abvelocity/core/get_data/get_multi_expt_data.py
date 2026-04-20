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

from typing import Optional

from abvelocity.core.get_data.cursor import Cursor
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.get_data.get_asgmnt_data import get_asgmnt_data_multi
from abvelocity.core.get_data.get_expt_stats import get_expt_stats
from abvelocity.core.get_data.join_expt_dfs import join_expt_dfs
from abvelocity.core.param.constants import TRIGGER_TIME_COL, VARIANT_COL
from abvelocity.core.param.expt_info import MultiExptInfo


def get_multi_expt_data(
    cursor: Cursor,
    multi_expt_info: MultiExptInfo,
    condition: Optional[str] = None,
    materialize_each_expt_to_df: bool = False,
    materialize_joined_data_to_df: bool = False,
    include_trigger_time: bool = False,
) -> DataContainer:
    """This function gets the multi-experiment data and returns a DataContainer.

    Args:
        cursor: A database cursor.
        multi_expt_info: includes the experiments information.
        materialize_each_expt_to_df: A bool to determine if we want to materialize each
        materialize_joined_data_to_df: A bool to determine if we want to meatrialize the joined
            experiment data. If False, a query is generated and returned.
        include_trigger_time: boolean to determine if min trigger time needs to be extracted.
        Default False.

    Returns:
        expt_dc: The DataContainer which contains experiment data.

    Alters:
        multi_expt_info: This function will amend `multi_expt_info` by adding `derived_stats` field.
            It will add `derived_stats` also to all  `expt_info` in the `expt_info_list`.

    """

    print(f"condition: {condition}")

    """Gets User Experiment Assignment Data (Expt Data)"""
    if materialize_each_expt_to_df:
        expt_dc_list = get_asgmnt_data_multi(
            multi_expt_info=multi_expt_info,
            cursor=cursor,
            condition=condition,
            materialize_to_df=True,
        )
        print(f"\n *** expt_dc_list:\n {expt_dc_list}")
        expt_dc = join_expt_dfs(expt_dc_list)
        multi_expt_info.derived_stats = get_expt_stats(df=expt_dc.df)
    else:
        expt_dc_list = get_asgmnt_data_multi(
            multi_expt_info=multi_expt_info,
            cursor=None,  # No cursor is needed here as queries are only generated.
            condition=condition,
            materialize_to_df=False,
        )
        print(f"\n *** expt_dc_list:\n {expt_dc_list}")
        expt_dc = join_expt_dfs(dc_list=expt_dc_list, on_cols=[multi_expt_info.expt_unit_col])
        # In this case the join will only create a join query
        # This means we need to materilize to df now
        if materialize_joined_data_to_df:
            sql_query_result = cursor.get_df(expt_dc.query)
            print(f"\n*** sql_query_result:\n{sql_query_result}")
            common_cols = [VARIANT_COL, TRIGGER_TIME_COL] if include_trigger_time else [VARIANT_COL]
            df = sql_query_result.df
            for col in common_cols:
                if col in df.columns:
                    df[col] = [tuple(item.values()) if isinstance(item, dict) else tuple(item) for item in df[col]]
            expt_dc.df = df
            expt_dc.is_df = True
            multi_expt_info.derived_stats = get_expt_stats(df=expt_dc.df)

    return expt_dc

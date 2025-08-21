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

"""The goal is to get "Expt" data which is experiment assignment data.
It also joins several expts if needed."""

import sys
from typing import Callable, List, Optional

from abvelocity.get_data.cursor import Cursor
from abvelocity.get_data.data_container import DataContainer
from abvelocity.get_data.get_expt_stats import get_expt_stats
from abvelocity.param.constants import UNIT_COL
from abvelocity.param.expt_info import ExptInfo, MultiExptInfo


def get_asgmnt_data(
    expt_asgmnt_table: str,
    get_asgmnt_query: Callable,
    expt_info: ExptInfo,
    cursor: Optional[Cursor] = None,
    condition: Optional[str] = None,
    materialize_to_df: bool = True,
) -> DataContainer:
    """Returns experiment assignment data as a DataContainer.

    Args:
        expt_asgmnt_table: The experiment assignment table.
        get_asgmnt_query: Callable
        expt_info: Experiment info.
            See `~abvelocity.expt_info.ExptInfo` for more details.
        cursor: A Presto Cursor.
        condition: A SQL condition.
        materialize_to_df: A bool to determine if we want to materialize to pandas df.

    Returns:
        result: DataContainer with experiment assignment data.

    Raises:
        None.
    """
    query = expt_info.query
    if query is None:
        query = get_asgmnt_query(
            expt_asgmnt_table=expt_asgmnt_table, expt_info=expt_info, condition=condition
        )

    if not materialize_to_df:
        return DataContainer(query=query, is_df=False)

    result = cursor.get_df(query=query)

    return DataContainer(df=result.df, is_df=True)


def get_asgmnt_data_multi(
    expt_asgmnt_table: str,
    get_asgmnt_query: Callable,
    multi_expt_info: MultiExptInfo,
    cursor: Optional[Cursor] = None,
    condition: Optional[str] = None,
    drop_unit_duplicates: bool = True,
    materialize_to_df: bool = True,
) -> List[DataContainer]:
    """This function queries expt data and returns a list of DataContainers.

    Args:
        expt_asgmnt_table : `str`
            The directory to query expt data.
        get_asgmnt_query: Callable
        multi_expt_info : `MultiExptInfo`
            A MultiExptInfo object with experiment information.
        cursor: A databse Cursor.
        condition: A SQL condition.
        drop_unit_duplicates : If True, this will only keep one row per unit (`UNIT_COL`) for each experiment.
            It keeps the last row, but note that this is a rather arbitrary choice.
            Best practice is to pass data without duplicates in `UNIT_COL` to this function.
            This argument is only supported when we do materialize to df.
        materialize_to_df: A bool to determine if we want to materialize to pandas df.

    Returns
        expt_dc_list : A list of DataContainers, one for each experiment in `multi_expt_info.expt_info_list`.

    Alters:
        expt_info_list : This function will amend each `expt_info` in the list with
        it's derived stats (`DerivedExptState`)
    """

    # Queries expt data for each expt in `multi_expt_info.expt_info_list`.
    expt_dc_list = []
    for expt_info in multi_expt_info.expt_info_list:
        dc = get_asgmnt_data(
            cursor=cursor,
            expt_asgmnt_table=expt_asgmnt_table,
            get_asgmnt_query=get_asgmnt_query,
            expt_info=expt_info,
            condition=condition,
            materialize_to_df=materialize_to_df,
        )
        if materialize_to_df:
            expt_df = dc.df

            if drop_unit_duplicates:
                expt_df.drop_duplicates(subset=[UNIT_COL], keep="first", inplace=True)
            # Get statistics about one individual experiment
            expt_info.derived_stats = get_expt_stats(expt_df)

            expt_dc_list.append(DataContainer(df=expt_df, is_df=True))
        else:
            expt_dc_list.append(dc)

    size_in_megabytes = sys.getsizeof(expt_dc_list) / 10**6
    print(f"`expt_data` size in MB: {size_in_megabytes}")

    return expt_dc_list

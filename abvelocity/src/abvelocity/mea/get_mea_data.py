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

import gc
import resource
from typing import Callable, Optional

from abvelocity.get_data.cursor import Cursor
from abvelocity.get_data.data_container import DataContainer
from abvelocity.get_data.get_multi_expt_data import get_multi_expt_data
from abvelocity.get_data.join_expt_with_metric_df import join_expt_with_metric_df
from abvelocity.param.analysis_info import AnalysisInfo
from abvelocity.param.metric import get_u_metrics


def get_mea_data(
    cursor: Cursor,
    analysis_info: AnalysisInfo,
    expt_asgmnt_table: str,
    get_asgmnt_query: Callable,
    existing_dc: Optional[DataContainer] = None,
    condition: Optional[str] = None,
) -> DataContainer:
    """This is the flow to get data for multi-experiment analysis (MEA), returning a DataContainer.

    Args:
        cursor: A database cursor.
        analysis_info: `AnalysisInfo` which includes the experiments and metrics.
        expt_asgmnt_table: Table which includes expt data.
        get_asgmnt_query: Callable which creates expt query.
        existing_dc: If such DataContainer exists, we augment more metrics to it.
            In particular we will assume there is no need to query expt assignment data.
        condition: Optional condition for sql query, eg to subsample.

    Returns:
        result: Processed DataContainer for MEA.

    Alters:
        analysis_info.multi_expt_info: This function will amend `multi_expt_info` by adding `derived_stats` field.
            It will add `derived_stats` also to all  `expt_info` in the `expt_info_list`.

    """

    multi_expt_info = analysis_info.multi_expt_info
    # Takes intersection from time periods of all experiments
    # This will define the analysis period
    if analysis_info.start_date is None:
        analysis_info.start_date = max(
            [expt_info.start_date for expt_info in multi_expt_info.expt_info_list]
        )

    if analysis_info.end_date is None:
        analysis_info.end_date = min(
            [expt_info.end_date for expt_info in multi_expt_info.expt_info_list]
        )

    print(
        f"\n*** analysis_info start date and end dates after taking intersections: {analysis_info.start_date}, {analysis_info.end_date}"
    )

    # We then update the expt_infos as well to adhere to these dates.
    # Note that it is possible that expt_info has a tighter range for some experiments if desired.
    # However that would be a rare case.
    # The `min`, `max` below will ensure that the experiment range is not wider than the analysis range.
    # Also note that `analysis_info.start_date` and `analysis_info.end_date` are None, the ranges will be all the same.
    # This is because in the above these two quantities are calculated from `expt_info_list` in that case.
    for expt_info in multi_expt_info.expt_info_list:
        expt_info.start_date = max(analysis_info.start_date, expt_info.start_date)
        expt_info.end_date = min(analysis_info.end_date, expt_info.end_date)

    # If either of `expt_df` and `processed_df` are not None.
    # We assume that we only are supposed to join with more metrics.
    # Therefore we do not query the expt assignment data anymore.
    if existing_dc is None:
        existing_dc = get_multi_expt_data(
            cursor=cursor,
            multi_expt_info=multi_expt_info,
            expt_asgmnt_table=expt_asgmnt_table,
            get_asgmnt_query=get_asgmnt_query,
            condition=condition,
        )
        # Expt data

    # print(f"\n*** multi_expt_info (after getting the data):\n{multi_expt_info}")

    print(
        "\n*** multi_expt_info.derived_stats (after getting the data):\n"
        f"{multi_expt_info.derived_stats}"
    )

    # Below we keep augmenting the available data with more metrics.
    # This is done as long as `metric_info_list` is non-empty.
    # Otherwise, `existing_df` will only contain `expt_df`.
    # That would be still useful eg when we want to only return `expt_df`
    # and do joins with metric data in multiple steps.
    # The for loop will keep augmenting `existing_df`
    if analysis_info.metric_info_list:
        for metric_info in analysis_info.metric_info_list:
            metric_family = metric_info.metric_family
            # Info such as table name for metrics is stored in
            # `get_metric_query` which depends (only) on metric_family
            get_metric_query = metric_family.get_metric_query

            process_expt_metric_df = metric_family.process_expt_metric_df
            get_metric_query_params = metric_family.get_metric_query_params
            process_expt_metric_df_params = metric_family.process_expt_metric_df_params

            # Metric data
            u_metrics = get_u_metrics(metric_info.metrics)

            if u_metrics is None:
                raise ValueError("metrics cannot be None.")
            print(f"\n*** u_metrics:\n{u_metrics}")
            metric_query = get_metric_query.construct_query(
                start_date=analysis_info.start_date,
                end_date=analysis_info.end_date,
                u_metrics=u_metrics,
                condition=condition,
                **get_metric_query_params,
            )

            print(f"\n*** `metric_query`:\n{metric_query}")

            metric_query_result = cursor.get_df(query=metric_query)

            metric_df = metric_query_result.df
            print(f"\n*** metric_df:\n{metric_df.head(2)}")
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                1024**3
            )  # Convert to GB
            print(
                f"***\n Memory used in `get_mea_data` after getting metric data: {mem_usage:.4f} GB"
            )

            # TODO: add a check for repeated metric names.
            # This will be a left join as that is the default for the function.
            existing_dc = join_expt_with_metric_df(
                expt_dc=existing_dc,
                metric_dc=DataContainer(df=metric_df, is_df=True),
                u_metrics=u_metrics,
            )

            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                1024**3
            )  # Convert to GB
            print(
                "***\n Memory used in `get_mea_data` after joining `expt_df`, `metric_df`:"
                f" {mem_usage:.4f} GB"
            )

            print(
                f"\n*** existing_dc:\n{existing_dc.df.head(2) if existing_dc.df is not None else None}"
            )
            print("\n*** deletes data we do not need: `metric_df`")

            del metric_df
            del metric_query_result
            gc.collect()
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                1024**3
            )  # Convert to GB
            print(f"***\n Memory used in `get_mea_data` after gc: {mem_usage:.4f} GB")

            if process_expt_metric_df is not None:
                existing_dc.df = process_expt_metric_df(
                    df=existing_dc.df, u_metrics=u_metrics, **process_expt_metric_df_params
                )

    return existing_dc

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

import copy
from typing import Callable, Optional

from abvelocity.get_data.cursor import Cursor
from abvelocity.mea.get_mea_data import get_mea_data
from abvelocity.mea.mea import MEA, MEAResult
from abvelocity.param.analysis_info import AnalysisInfo
from abvelocity.param.launch import Launch
from abvelocity.param.variant import ComparisonPair


def run_mea(
    cursor: Cursor,
    analysis_info: AnalysisInfo,
    expt_asgmnt_table: str,
    get_asgmnt_query: Callable,
    launches: Optional[list[Launch]] = None,
    control_launch: Optional[Launch] = None,
    comparison_pairs: list[ComparisonPair] = None,
    scale: bool = True,
    condition: Optional[str] = None,
) -> Optional[MEAResult]:
    """This is the flow to run multi-experiment analysis (MEA).
    This takes two steps:

        - Get the data for MEA.
        - Run MEA.

    Args:
        cursor: A database Cursor.
        analysis_info: AnalysisInfo which includes the experiments and metrics.
        expt_asgmnt_table: Table which includes expt data.
        get_asgmnt_query: Callable which creates expt query.
        launches: List of launches. Each launch is a combination of variants across experiments (or simply a string for single experiment case).
            See `~abvelocity.param.launch.Launch`.
            For each launch first,

                - we construct the counterpart multi-experiment control
                - for each arm we map them to the corresponding `VariantList`.
                - a `ComparisonPair` is created with these two objects then.

            All the above steps are done using `launch_to_comparison_pair` function.
        control_launch: The Launch which is used as the baseline for comparison.
            If not passed, we will assume all experiments are on the control arm (`CONTROL_LABEL`).
        comparison_pairs: List of comparison pairs.
            This is an advanced feature as `launches` will construct the needed comparison_pairs for typical launches.
            Each pair includes a treatment and control field.
            See `~abvelocity.param.variant.ComparisonPair`.
        scale: If true and metric_info_list has more than one element,

            - we get the assignment data `expt_df` and
            - for each group (can think of this as map phase):
                - join with expt_df
                - perform mea
                - store results
                - delete joined data
            - combine the results (can think of this as reduce phase)
        condition: Optional SQL query condition to be passed to `get_mea_data`

    Returns:
        An instance of `MEAResult`.
        In summary, the result will include the effect sizes and confidence intervals.
        See `~abvelocity.mea.mea.MEAResult` for more details.

    Alters:
        analysis_info:

            - This will be altered upon reading the data and some
                statistics will be added regarding experiments.
            - Also if no metric is passed via `analysis_info`, an attempt
                will be made to infer metrics from `metric_family_name`.


    """
    # Gets the data needed for MEA and updates `analysis_info` by attaching
    # derived expt statistics and metric, metrics are not passed directly
    # in metrics field.
    if scale and analysis_info.metric_info_list and len(analysis_info.metric_info_list) > 1:
        print(
            f"\n*** scale is true and there are more than 1 metric groups: {len(analysis_info.metric_info_list)}"
        )
        print("\n*** we will join metric groups in various stages to limit memory usage.")

        # We start from an empty mea result and keep augmenting it in the for loop.
        mea_result = MEAResult()

        analysis_info_copy = copy.deepcopy(analysis_info)
        analysis_info_copy.metric_info_list = None
        expt_dc = get_mea_data(
            cursor=cursor,
            analysis_info=analysis_info_copy,
            expt_asgmnt_table=expt_asgmnt_table,
            get_asgmnt_query=get_asgmnt_query,
            condition=condition,
        )
        print(f"*** expt_dc was obtained by passing an analysis_info w/o : {expt_dc}")
        # We want to update the `derived_stats` for `analysis_info.multi_expt_info`
        # This is despite passing a copy which is merely done to remove the metric info.
        analysis_info.multi_expt_info.derived_stats = (
            analysis_info_copy.multi_expt_info.derived_stats
        )

        # We also want to update the `start_date` `end_date` of `analysis_info`
        analysis_info.start_date = analysis_info_copy.start_date
        analysis_info.end_date = analysis_info_copy.end_date

        for metric_info in analysis_info.metric_info_list:
            # We reset the `metric_info_list` to only include one `metric_info`
            print(f"\n*** running MEA for metric_info:\n {metric_info}")
            analysis_info_copy.metric_info_list = [metric_info]
            df = get_mea_data(
                cursor=cursor,
                analysis_info=analysis_info_copy,
                expt_asgmnt_table=expt_asgmnt_table,
                get_asgmnt_query=get_asgmnt_query,
                existing_dc=expt_dc,
                condition=condition,
            ).df
            print(f"\n*** MEA data was obtained: df\n {df}")
            mea = MEA(
                df=df,
                analysis_info=analysis_info_copy,
                launches=launches,
                control_launch=control_launch,
                comparison_pairs=comparison_pairs,
            )

            mea.run()
            print("\n*** Delete df generated for this run of MEA.")
            del df

            mea_result_new = mea.result
            print("\n*** Deletes `MEA` object after extracting the result.")
            del mea
            mea_result.combine(mea_result_new)
    else:
        # This will also handle the case with no metrics
        # In such case mea_result will be None
        # However `analysis_info` could get updated with experiment stats
        df = get_mea_data(
            cursor=cursor,
            analysis_info=analysis_info,
            expt_asgmnt_table=expt_asgmnt_table,
            get_asgmnt_query=get_asgmnt_query,
            condition=condition,
        ).df

        mea = MEA(
            df=df,
            analysis_info=analysis_info,
            launches=launches,
            control_launch=control_launch,
            comparison_pairs=comparison_pairs,
        )

        mea.run()

        mea_result = mea.result
        del df
        del mea

    return mea_result

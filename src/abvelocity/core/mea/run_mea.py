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
import re
from typing import Optional, Tuple

from abvelocity.core.get_data.agg_metrics_query import AggMetricsQuery
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.mea.constants import JK_NUM_BUCKETS, MEA_DEFAULT_METHOD
from abvelocity.core.mea.get_mea_data import get_mea_data, get_mea_data_and_materialize
from abvelocity.core.mea.mea import MEA, MEAResult
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.launch import Launch
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.param.variant import ComparisonPair


def run_mea(
    io_param: IOParam,
    analysis_info: AnalysisInfo,
    launches: Optional[list[Launch]] = None,
    control_launch: Optional[Launch] = None,
    comparison_pairs: list[ComparisonPair] = None,
    launch_filter: Optional[tuple[str]] = None,
    scale: bool = True,
    condition: Optional[str] = None,
    metric_inclusion_regex: Optional[str] = None,
    method: str = MEA_DEFAULT_METHOD,
    num_buckets: Optional[int] = JK_NUM_BUCKETS,
    expt_metric_df_materialize_spec: Tuple[str] = ("DBTable",),
    dc: Optional[DataContainer] = None,
) -> Optional[MEAResult]:
    """This is the flow to run multi-experiment analysis (MEA).
    This takes two steps:

        - Get the data for MEA.
        - Run MEA.

    Args:
        io_param: IO parameters to read / write tables and results. This includes a cursor (`Cursor` type) to interact with
        database / read write data. Also it includes other arguments such as `create_table_prefix` to facilitate writing tables
        if needed and apply the same parameter to all tables generated with the flow.
        analysis_info: AnalysisInfo which includes the experiments and metrics.
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
        launch_filter: This is a tuple which is passed to filter launches.
            This tuple needs to be the same size as the Launch / Variant length but it can include None in some positions.
            The `is_consistent_with` method from `Variant` class will be used to filter then and launches not passing
            this filter will not be analyzed. For example if one passed `("v1", None)` only launches involving "v1"
            will be considered.
        scale: If true and metric_info_list has more than one element,
            - we get the assignment data `expt_df` and
            - for each group (can think of this as map phase):
                - join with expt_df
                - perform mea
                - store results
                - delete joined data
            - combine the results (can think of this as reduce phase)
        condition: Optional SQL query condition to be passed to `get_mea_data_and_materialize`
        metric_inclusion_regex: An optional regex string to filter the metrics to include in the analysis.
            This is useful because some metric families could include too many metrics by default, thus overwhelming
            the report in terms of most important metrics to look at. This issue is exacerbated in MEA as we are
            looking at many combinations of variants.
        method: There are two methods to compute MEA

            - (1) "simple": It covers metrics with numerator only, but does allow metrics for which the `sample_count`
            is prescribed in the metric. Therefore, metrics such as survival can be covered if the eligible population
            number is passed through a sample_count column and the number of units surviving is the numerator.
            In this case the means and variance are computed for each variant and combined with weights.
            Since variance computation is simply a weighted combination of variant variances (each of which can be computed using
                VAR(X) = E(X^2) - E(X)^2,
            computing the estimator variance will be fast.
            In particular, we can compute all the variant variances in one step:

                variant_metric_stats_df = calc_variant_metric_stats(df=df, metric=metric, variant_col=VARIANT_COL)

            which will result is a small df and then use it for all comparion pairs.

            - (2) "general": This covers more general metrics including arbitrary ratio metrics. This method
            first constructs the estimator calculation using the estimator class for each metric and then uses
            non-parametric variance computation. This computation will be done for each comparion pair separately
            operating on the original `df`.

            When "simple" is input and metric is a ratio metric we switch to "general" for that metric.
            Default for this argument is `MEA_DEFAULT_METHOD`.
        num_buckets: The number of groups (B) for jackknife iterations.
            Defaults to `JK_NUM_BUCKETS`.
        expt_metric_df_materialize_spec: Spec to determine how to materialize the data.
            See `get_mea_data_and_materialize` for more details.
        dc: Optional DataContainer to be used directly instead of getting the data again.
            This might be useful when flow fails after getting the data and one wants to rerun MEA
            without going through data extraction again.

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
    # If metric filters are to be applied, we update `analysis_info.metric_info_list`
    if metric_inclusion_regex:
        # Compile the metric inclusion regex once if provided
        compiled_regex = re.compile(metric_inclusion_regex) if metric_inclusion_regex else None

        metric_info_list = analysis_info.metric_info_list
        metric_info_list_filtered = []

        for metric_info in metric_info_list:
            # Get the full, unfiltered list of metrics from the MetricFamily object
            metric_family = metric_info.metric_family

            metrics_to_include = []
            if compiled_regex and metric_family.metrics:
                # Filter metrics based on the regex applied to their names
                for metric in metric_family.metrics:
                    # Use metric.name; if None, fall back to numerator.col (assuming it exists for all relevant metrics)
                    metric_name = (
                        metric.name
                        if metric.name is not None
                        else (metric.numerator.col if hasattr(metric, "numerator") and hasattr(metric.numerator, "col") else "")
                    )
                    if compiled_regex.search(metric_name):
                        metrics_to_include.append(metric)
                print(f"\n*** metrics for {metric_family.name} where filtered down to this many:\n {len(metrics_to_include)}")
                print(f"\n*** metrics for {metric_family.name} where filtered down to:\n {metrics_to_include}")
            else:
                # If no regex is provided, or if the family has no predefined metrics, include all (or none)
                metrics_to_include = metric_info.metric_family

            # Create MetricInfo with the (potentially filtered) list of metrics
            metric_info_list_filtered.append(MetricInfo(metric_family=metric_family, metrics=metrics_to_include))

        analysis_info.metric_info_list = metric_info_list_filtered

    if not dc and scale and analysis_info.metric_info_list and len(analysis_info.metric_info_list) > 1:
        print(f"\n*** scale is true and there are more than 1 metric groups: {len(analysis_info.metric_info_list)}")
        print("\n*** we will join metric groups in various stages to limit memory usage.")

        # We start from an empty mea result and keep augmenting it in the for loop.
        mea_result = MEAResult()

        analysis_info_copy = copy.deepcopy(analysis_info)
        analysis_info_copy.metric_info_list = None
        # Notice in this case for the pure experiments data, we do not materizalize in any case
        # In other words this does not depend on the passed `expt_metric_df_materialize_type`
        # We pass "Query" as materialize type to the get_mea_data function
        # Also notice that we use get_mea_data as opposed to get_mea_data_and_materialize
        expt_dc = get_mea_data(
            io_param=io_param,
            analysis_info=analysis_info_copy,
            condition=condition,
            expt_metric_df_materialize_type="Query",
        )
        print(f"*** expt_dc was obtained by passing an analysis_info w/o : {expt_dc}")
        # We want to update the `derived_stats` for `analysis_info.multi_expt_info`
        # This is despite passing a copy which is merely done to remove the metric info.
        analysis_info.multi_expt_info.derived_stats = analysis_info_copy.multi_expt_info.derived_stats

        # We also want to update the `start_date` `end_date` of `analysis_info`
        analysis_info.start_date = analysis_info_copy.start_date
        analysis_info.end_date = analysis_info_copy.end_date

        for metric_info in analysis_info.metric_info_list:
            # We reset the `metric_info_list` to only include one `metric_info`
            print(f"\n*** running MEA for metric_info:\n {metric_info}")
            analysis_info_copy.metric_info_list = [metric_info]
            dc = get_mea_data_and_materialize(
                io_param=io_param,
                analysis_info=analysis_info_copy,
                existing_dc=expt_dc,
                condition=condition,
                expt_metric_df_materialize_spec=expt_metric_df_materialize_spec,
            )
            mea = MEA(
                dc=dc,
                analysis_info=analysis_info_copy,
                launches=launches,
                control_launch=control_launch,
                comparison_pairs=comparison_pairs,
                launch_filter=launch_filter,
                method=method,
                num_buckets=num_buckets,
                io_param=io_param,
            )

            mea.run()
            print("\n*** Delete df generated for this run of MEA.")
            if dc.pandas_df:
                del dc.pandas_df

            mea_result_new = mea.result
            print("\n*** Deletes `MEA` object after extracting the result.")
            del mea
            mea_result.combine(mea_result_new)
    else:
        print("\n*** In `run_mea` for case with either of: dc passed or scale = False, single metric group, or no metrics.")
        # This will also handle the case with no metrics
        # In such case mea_result will be None
        # However `analysis_info` could get updated with experiment stats
        # If `dc` is passed, we skip getting the data again.
        if not dc:
            print("\n*** In `run_mea` dc is not passed (and scale = False, single metric group, or no metrics), therefore dc will be extracted.")
            dc = get_mea_data_and_materialize(
                io_param=io_param,
                analysis_info=analysis_info,
                condition=condition,
                expt_metric_df_materialize_spec=expt_metric_df_materialize_spec,
            )

        # We do not materialize to pandas before passing to MEA
        # MEA class itself will handle materialization to pandas as per need.

        mea = MEA(
            dc=dc,
            analysis_info=analysis_info,
            launches=launches,
            control_launch=control_launch,
            comparison_pairs=comparison_pairs,
            launch_filter=launch_filter,
            method=method,
            num_buckets=num_buckets,
            io_param=io_param,
        )

        mea.run()

        mea_result = mea.result
        if dc.pandas_df is not None:
            del dc.pandas_df
        del mea

    if mea_result is not None and analysis_info.metric_info_list:
        mea_result.agg_metrics = AggMetricsQuery(analysis_info, condition=condition).get_pandas_df(io_param.cursor)

    return mea_result

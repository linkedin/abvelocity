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

import datetime
import os
import warnings
from dataclasses import asdict
from typing import Optional

import pandas as pd
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.get_data.get_expt_stats import get_expt_stats
from abvelocity.core.mea.constants import (
    DEFAULT_MATERIALIZE_TO_PANDAS_STAGE,
    END_USER_COLS,
    END_USER_METRIC_RESULT_KEYS,
    IMPLEMENTED_MATERIALIZE_TO_PANDAS_STAGES,
    JK_NUM_BUCKETS,
    MEA_DEFAULT_METHOD,
)
from abvelocity.core.mea.entities import MEAMetricResult, MEAResult
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.constants import (
    CI_COL,
    CI_PERCENT_COL,
    CONTROL_LABEL,
    DELTA_SUM_CI_COL,
    MEAN_COL,
    METRIC_NAME_COL,
    SAMPLE_COUNT_COL,
    SD_COL,
    SUM_COL,
    SUM_SQ_COL,
    TRIGGER_STATE_COL,
    TRIGGER_STATE_COUNT_COL,
    TRIGGER_STATE_PERCENT_COL,
    VARIANT_COL,
    VARIANT_PERCENT_COL,
)
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.launch import Launch
from abvelocity.core.param.launch_to_comparison_pair import launch_to_comparison_pair
from abvelocity.core.param.variant import ComparisonPair
from abvelocity.core.stats.calc_comparison_pairs_effects import calc_comparison_pairs_effects
from abvelocity.core.stats.calc_variant_metric_effects_simple import calc_variant_metric_effects_simple
from abvelocity.core.stats.calc_variant_metric_stats import calc_variant_metric_stats
from abvelocity.core.stats.two_sample_t_test import two_sample_t_test

# from abvelocity.core.utils.calc_freq import calc_freq
from abvelocity.core.utils.df_to_html import df_to_html
from abvelocity.core.utils.plot_compare_variants import plot_compare_variants
from abvelocity.core.mea.assumption_check import check_trigger_invariance
from abvelocity.core.mea.metric_interaction import check_metric_interaction
from abvelocity.core.utils.plot_conditional_variant_dist import plot_conditional_variant_dist
from abvelocity.core.utils.plot_variants import plot_variants
from abvelocity.core.utils.publish_df_color_code_p_value import FORMAT_DICT, publish_df_color_code_p_value
from abvelocity.core.utils.round_df import round_df


class MEA:
    """This class is used to run multi-experiment analysis (MEA).
    The main method is `run` which generates the results (See `~abvelocity.mea.mea.MEAResult`).
    The class also includes a `publish` method to generate html string and write results to csv files.
    """

    def __init__(
        self,
        dc: Optional[DataContainer] = None,
        analysis_info: Optional[AnalysisInfo] = None,
        launches: Optional[list[Launch]] = None,
        control_launch: Optional[Launch] = None,
        comparison_pairs: Optional[list[ComparisonPair]] = None,
        launch_filter: Optional[tuple[str]] = None,
        ci_coverage: float = 0.95,
        recalculate_expt_stats: bool = False,
        method: str = MEA_DEFAULT_METHOD,
        num_buckets: Optional[int] = JK_NUM_BUCKETS,
        io_param: Optional[IOParam] = None,
        materialize_to_pandas_stage=DEFAULT_MATERIALIZE_TO_PANDAS_STAGE,
    ):
        """Initializes the class with the data, analysis info and comparison pairs.

        Args:
            dc: DataContainer with data in pandas format or in form of a table / query.
            Expected schema is as follows:
                df or query result with these columns.
                    - The data must include VARIANT_COL which specifies the experiment variants
                    - The data should also include one row for each unit (of experiment eg user) and variant
                    - The data will also include columns corresponding to u_metrics (`UMetric`) we might need
                    - Recall that a

                        - `UMetric` is a univar metric already aggregated at unit level
                        - a `Metric` could be a ratio of two such `UMetric`s and it could have a
                            spcification to count the nymber of samples based another `UMetric`
                            which plays the role for certain eligibility i.e. a unit satisfies
                            a certain condition eg up for renew for a subscriber.
                            Currently ratio metrics are supported by passing the denominator as
                            `sample_count` in the `Metric` definition.
                            This will imply that that column will be used to compute the sample size rather than number of rows
                            An example of this could be CTR, or retention.
                            An implementation of more generic ratio metrics fits well with this framework but it is TBD.

                    - Note that no unit col is explicitly mentioned as each row corersponds to a
                    unique combination of (unit)

            analysis_info: AnalysisInfo with the experiments and metrics.
            launches: List of launches.
                Each launch is a combination of variants across experiments
                (or simply a string for single experiment case).
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
            ci_coverage: coverage of conf interval.
            recalculate_expt_stats: A bool to decide if experiment derived stats should be recalculated.
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
            io_param: IO parameters to read / write tables and results. This includes a cursor (`Cursor` type) to interact with
            materialize_to_pandas_stage: This determines when data will be converted to pandas when dc does not inlude pandas_df.
                One could materialize at the beginning of MEA for data of sizes of 10s of millions on machines in 2025 with 10s of rows.
                However, conversions and aggregations could become slow if machine runs out of RAM and Python starts doing disk swaps.
                (That being said we have not seen failures even with data of the size of 100 millions and 20 rows.)
                While this is still fairly scalable, it might be risky or prohibitively slow. Therefore we have option to push the
                materialization further down to internal functions such as `calc_variant_metric_stats` and `get_expt_stats`.

                - "immediate": Will materialize to Pandas Df immediately if pandas_df is not attached to dc yet.
                - "delayed": Will push materialization to Pandas Df to later stages e.g. `calc_variant_metric_stats` and `get_expt_stats`.


        Attributes (other than the input arguments):
            mea_result: Result of MEA. This is attached after running the analysis.
        """
        self.dc = dc
        self.analysis_info = analysis_info
        self.launches = launches
        self.control_launch = control_launch
        self.comparison_pairs = comparison_pairs
        self.launch_filter = launch_filter
        self.ci_coverage = ci_coverage
        self.recalculate_expt_stats = recalculate_expt_stats
        self.method = method
        self.num_buckets = num_buckets
        self.io_param = io_param

        # Attributes
        self.result = None
        self.materialize_to_pandas_stage = materialize_to_pandas_stage

        if materialize_to_pandas_stage is not None:
            if materialize_to_pandas_stage not in IMPLEMENTED_MATERIALIZE_TO_PANDAS_STAGES:
                raise NotImplementedError(
                    f"materialize_to_pandas_stage: {materialize_to_pandas_stage} is not implemented."
                    f"Implemented stages are: {IMPLEMENTED_MATERIALIZE_TO_PANDAS_STAGES}"
                )

    def run(self) -> None:
        """
        This is the main method for the class.
        It runs multi-experiment analysis (MEA).
        Note that the arguments (See `__init__`) can be passed during the class initiation or
        attached later.

        Args:
            None. (Uses attributes of `self`, see `__init__`).

        Alters:
            self: attches the result with type `MEAResult` to the class instance.

        """
        # Materialize before starting the analysis
        # TODO: This behavior will change later on to scale.
        # This means we will be passing the non-materizalized data to lower level functions during estimation
        print(f"\n***: In `MEA`, self.dc is passed as: {self.dc}")
        if self.dc.pandas_df is None and self.materialize_to_pandas_stage == "immediate":
            if self.dc.table_name is not None:
                print(f"\n***: MEA is materializing data to Pandas DF using dc.table_name: {self.dc.table_name}")
                materialized_df_result = self.io_param.cursor.get_df(query=f"SELECT * FROM {self.dc.table_name}")
            elif self.dc.query is not None:
                print("\n***: MEA is materializing data to Pandas DF using dc.query:\n{dc.query}")
                materialized_df_result = self.io_param.cursor.get_df(query=self.dc.query)
            else:
                raise ValueError(f"In MEA, DC has no attribute to extract Data. DC: {self.dc}.")

            self.dc.pandas_df = materialized_df_result.df
            print("\n*** MEA data was materialized as pandas_df.")

        if self.dc.pandas_df is not None:
            df = self.dc.pandas_df
            print(f"""\n*** In class `MEA`: df:\n{df}""")
            if VARIANT_COL in df.columns:
                df[VARIANT_COL] = [tuple(item.values()) if isinstance(item, dict) else tuple(item) for item in df[VARIANT_COL]]

        analysis_info = self.analysis_info
        launches = self.launches
        control_launch = self.control_launch
        comparison_pairs = self.comparison_pairs
        expt_info_list = analysis_info.multi_expt_info.expt_info_list

        if (not analysis_info.multi_expt_info.derived_stats) or self.recalculate_expt_stats:
            analysis_info.multi_expt_info.derived_stats = get_expt_stats(dc=self.dc, io_param=self.io_param)

        print(f"\n*** In class `MEA`: derived_stats variants:\n{analysis_info.multi_expt_info.derived_stats.variants}")
        print(f"\n*** In class `MEA`: derived_stats trigger_states:\n{analysis_info.multi_expt_info.derived_stats.trigger_states}")
        print(f"\n*** In class `MEA`: derived_stats launches:\n{analysis_info.multi_expt_info.derived_stats.launches}")
        print(f"\n*** In class `MEA`: derived_stats variant_count_df:\n{analysis_info.multi_expt_info.derived_stats.variant_count_df}")
        print(f"\n*** In class `MEA`: derived_stats trigger_state_count_df:\n{analysis_info.multi_expt_info.derived_stats.trigger_state_count_df}")
        if not launches:
            # If no `control_launch` is specified, we simply get the non control launches.
            # These are launches for which at least one launch is not on control arm.
            if control_launch is None:
                launches = analysis_info.multi_expt_info.derived_stats.non_control_launches
            else:
                # Get all launches
                launches = analysis_info.multi_expt_info.derived_stats.launches
                # Remove the spcified `control_launch`.
                # The way to do that is to check all launches and compare with `control_launch`.
                # This is done using the equality defined for `Variant` dataclass.
                # This equality only depends on the value field and not the name.
                launches = [launch for launch in launches if launch.value != control_launch.value]

            print("\n*** launches were inferred from derived_stats:" f"\n{launches}")

        # We will convert None to empty list, as we might add new comparison pairs if `launches` is not None.
        if not comparison_pairs:
            comparison_pairs = []

        if launches:
            for launch in launches:
                if not self.launch_filter or launch.is_consistent_with(self.launch_filter):
                    comparison_pair = launch_to_comparison_pair(launch=launch, control_launch=control_launch)
                    comparison_pairs.append(comparison_pair)

        if not analysis_info.metric_info_list:
            self.result = None
            warnings.warn(
                f"\n***: metric_info_list is None or empty = {analysis_info.metric_info_list}" "This means MEA will only generate overlap results.",
                UserWarning,
            )
            return None

        metrics = []
        for metric_info in analysis_info.metric_info_list:
            if metric_info.metrics is None:
                raise ValueError("At this stage metrics need to be available." f"Check analysis_info: {analysis_info}" f"Check metric_info: {metric_info}")
            metrics += metric_info.metrics

        print(f"\n *** MEA: metrics:\n{metrics}")
        num_expts = len(expt_info_list)

        # TODO: deprecate this field as it is unnecessary and `get_expt_stats` already provides equivalent info.
        variant_freq_dict = {}
        # Get stats about experiment variant assignments
        """
        if dc.pandas_df is not None:
            df = dc.pandas_df
            variant_freq_dict = {}
            for i in range(num_expts):
                col = f"{VARIANT_COL}_{i + 1}"
                variant_freq_dict[f"expt_{i + 1}"] = calc_freq(
                    df=df[df[col] != CATEG_NAN_VALUE], cols=[col]
                )
        variant_freq_dict["multi-expt"] = calc_freq(df=df, cols=[VARIANT_COL])
        """

        metric_result_dict = {}
        expt_control = (CONTROL_LABEL,) * num_expts
        print(f"\n*** MEA: expt_control:\n{expt_control}")

        # Get stats about the multi-experiment
        variant_count_df = analysis_info.multi_expt_info.derived_stats.variant_count_df

        for metric in metrics:
            print(f"\n**** MEA for metric name: {metric.name}")
            print(f"\n**** MEA for metric def: {metric}")
            metric_result_dict[metric.name] = MEAMetricResult()

            variant_metric_stats_df = None
            variant_effect_df_pairs = None

            # If method is simple and the metric does not have a denominator
            # we use the simple method.
            # Note that if the metric has a denominator, we have to switch to general.
            if self.method == "simple" and not metric.denominator:
                # This works for metrics with a numerator.
                # However, it does allow for metrics for which the denominator
                # could be considered as the number of samples.
                # In such cases, this could be reflected by passing that column as `sample_count` in `Metric`
                # This will imply that that column will be used to compute the sample size rather than number of rows
                print("\n***: In MEA class (simple): calculating `variant_metric_stats_df`")
                variant_metric_stats_df = calc_variant_metric_stats(dc=self.dc, metric=metric, variant_col=VARIANT_COL, io_param=self.io_param)
                print("\n***: In MEA class ('simple'): finished calculating `variant_metric_stats_df`")
                if comparison_pairs:  # this will make sure its not None or empty.
                    variant_effect_df_pairs = calc_variant_metric_effects_simple(
                        variant_metric_stats_df=variant_metric_stats_df,
                        variant_col=VARIANT_COL,
                        comparison_pairs=comparison_pairs,
                        stats_test_func=two_sample_t_test,
                        ci_coverage=self.ci_coverage,
                        variant_count_df=variant_count_df,
                        trigger_state_count_col=TRIGGER_STATE_COUNT_COL,
                    )
            elif self.method == "general" or metric.denominator:
                # This method works for general metrics including ratio metrics
                # with correlated numerator and denominator.
                if self.method != "general":
                    print(f"In `MEA` method was set to 'simple' but for this metric: {metric.name}, " "we switched to 'general' as this is a ratio metric.")
                # The general method needs a pandas df. If not already materialized
                # (e.g. delayed materialization stage), do it now.
                if self.dc.pandas_df is None:
                    if self.dc.table_name is not None:
                        self.dc.pandas_df = self.io_param.cursor.get_df(query=f"SELECT * FROM {self.dc.table_name}").df
                    elif self.dc.query is not None:
                        self.dc.pandas_df = self.io_param.cursor.get_df(query=self.dc.query).df
                    if self.dc.pandas_df is not None and VARIANT_COL in self.dc.pandas_df.columns:
                        self.dc.pandas_df[VARIANT_COL] = [
                            tuple(item.values()) if isinstance(item, dict) else tuple(item) for item in self.dc.pandas_df[VARIANT_COL]
                        ]
                df = self.dc.pandas_df
                variant_effect_df_pairs = calc_comparison_pairs_effects(
                    dc=DataContainer(pandas_df=df),
                    variant_col=VARIANT_COL,
                    metric=metric,
                    comparison_pairs=comparison_pairs,
                    ci_coverage=self.ci_coverage,
                    variant_count_df=variant_count_df,
                    trigger_state_count_col=TRIGGER_STATE_COUNT_COL,
                    num_buckets=self.num_buckets,
                )
            elif self.method not in ["simple", "general"]:
                raise ValueError(f"In `MEA` class, only methods 'simple' and 'general' are allowed and not {self.method}")

            # Here we maintain `variant_metric_stats_df` in results as extra information.
            # Note that this is not needed for the report and `variant_effect_df_pairs` is enough.
            # Also note that `variant_metric_stats_df` is the stats for the numerator
            # In fact `variant_metric_stats_df` is only needed for the "simple" method
            # where mean diff is the effect.
            metric_result_dict[metric.name].variant_metric_stats_df = variant_metric_stats_df
            metric_result_dict[metric.name].variant_effect_df_pairs = variant_effect_df_pairs

        # Attach the results to the class instance.
        self.result = MEAResult(variant_freq_dict=variant_freq_dict, metric_result_dict=metric_result_dict)

        # Generate the combined metrics results (one table for all metrics for each quantity)
        self.result.gen_combined_mea_result()

        return None

    def publish(
        self,
        mea_result: Optional[MEAResult] = None,
        analysis_info: Optional[AnalysisInfo] = None,
        write_path: Optional[str] = None,
        proj_name: Optional[str] = None,
        add_timestamp_to_path: bool = True,
        rounding_digits: int = 8,
        html_file_name: Optional[str] = None,
        markdown_file_name: Optional[str] = None,
        end_user_report: bool = True,
        publish_metric_variant_stats_fig: bool = True,
    ) -> dict[str, str]:
        """Writes the results of MEA to a specified path in csv format.
            For each metric the results will be written in `f"{write_path}/{metric}/"`.
            If `write_path` is not passed, it will be constructed.
            Also if `add_timestamp_to_path` is True,
            a formatted timestamp is added to the path,
            to avoid overwriting existing results.

        Args:
        mea_result: MEA result
        analysis_info: Analysis info
        write_path: Path to write results. If None one such path is generated.
            The generated path is either of

            - `f"{home}/abvelocity_results/"`
            - `f"{home}/abvelocity_results/{proj_name}"`

            where `home` is the home directory of the user.
        proj_name: This is used in the path name if passed.
        add_timestamp_to_path: If True, a timestamp is added to the path. Default True.
            This is to avoid over-writing existing records.
            The format of the timestamp is `"%Y-%m-%d_%H-%M-%S"`.
        rounding_digits: Number of digits to round the results to. Default 8.
        html_file_name: If passed, the MEA results will be written to an html file.
        markdown_file_name: If passed, the MEA results will be written to markdown file.
        end_user_report: If True, only the end user report will be generated which is based
            on the tables specified in `~abvelocity.mea.mea.END_USER_METRIC_RESULT_KEYS`.
        publish_metric_variant_stats_fig: If True, for each (univariate) metric a figure

        Returns:
            results: A dictionary with keys:

                - "html_str": The constructed html string.
                - "paths": List of paths created.
                - "file_names": List of file names created.

        """
        paths = []
        file_names = []

        if not mea_result:
            mea_result = self.result

        if not analysis_info:
            analysis_info = self.analysis_info

        html_str = ""
        markdown_str = ""
        interaction_figs_to_write = []

        # This will add experiments information to the html report.
        if analysis_info is not None:
            html_str += analysis_info.to_html()
            html_str += analysis_info.metrics_to_html()
            markdown_str += analysis_info.to_html()

            # Here we generate figures for variant counts / trigger state counts
            # This is only done if dimension is less than 3 (we cannot make 4D plots)
            if 1 < len(analysis_info.multi_expt_info.expt_info_list) <= 3:
                derived_stats = analysis_info.multi_expt_info.derived_stats
                variant_count_df = derived_stats.variant_count_df
                trigger_state_count_df = derived_stats.trigger_state_count_df

                dim_names = [x.name for x in analysis_info.multi_expt_info.expt_info_list]

                html_str += "<h2>Understand and validate Experiments' overlap</h2>"
                if trigger_state_count_df is not None:
                    trigger_state_count_df0 = trigger_state_count_df.copy()
                    trigger_state_count_df0.reset_index(inplace=True)
                    html_str += "<h3>Experiments' triggering overlap</h3>"
                    html_str += plot_variants(
                        df=trigger_state_count_df0,
                        variant_col=TRIGGER_STATE_COL,
                        count_column=TRIGGER_STATE_PERCENT_COL,
                        dim_names=dim_names,
                    ).to_html()

                if variant_count_df is not None:
                    variant_count_df0 = variant_count_df.copy()
                    variant_count_df0.reset_index(inplace=True)
                    html_str += "<h3>Experiments' variant overlap</h3>"
                    html_str += plot_variants(
                        df=variant_count_df0,
                        variant_col=VARIANT_COL,
                        count_column=VARIANT_PERCENT_COL,
                        dim_names=dim_names,
                    ).to_html()

        # Here we generate figures for variant counts / trigger state counts
        # This is only done if dimension is greater than 1 (conditional distribution is meaningful for D >= 2)
        if len(analysis_info.multi_expt_info.expt_info_list) > 1:
            derived_stats = analysis_info.multi_expt_info.derived_stats
            variant_count_df = derived_stats.variant_count_df
            trigger_state_count_df = derived_stats.trigger_state_count_df

            dim_names = [x.name for x in analysis_info.multi_expt_info.expt_info_list]
            # html_str += "<h2>Conditional Distributions of Exposure (Triggering) Proportions</h2>"
            """
            # --- Conditional Trigger State Overlap ---
            if trigger_state_count_df is not None:
                trigger_state_count_df0 = trigger_state_count_df.copy()
                trigger_state_count_df0.reset_index(inplace=True)
                html_str += "<h3>Experiments' Triggering Overlap (Conditional Distributions)</h3>"
                conditional_figs_trigger = plot_conditional_variant_dist(
                    df=trigger_state_count_df0,
                    variant_col=TRIGGER_STATE_COL,
                    count_column=TRIGGER_STATE_PERCENT_COL,
                    dim_names=dim_names,
                )
                for idx, fig in conditional_figs_trigger.items():
                    html_str += f"<h4>Conditional Trigger Percent Distribution: Fixed by {dim_names[idx-1]}</h4>"
                    html_str += fig.to_html()
            """
            # --- Conditional Variant Overlap ---
            if variant_count_df is not None:
                variant_count_df0 = variant_count_df.copy()
                variant_count_df0.reset_index(inplace=True)

                html_str += "<h3>Experiments' Variant Overlap (Conditional Distributions)</h3>"
                conditional_figs_variant = plot_conditional_variant_dist(
                    df=variant_count_df0,
                    variant_col=VARIANT_COL,
                    count_column=VARIANT_PERCENT_COL,
                    dim_names=[d[:20] for d in dim_names],
                )

                for idx, fig in conditional_figs_variant.items():
                    html_str += f"<h4>Conditional Variant (Unit Count) Distribution: Fixed by {dim_names[idx-1]}</h4>"
                    html_str += fig.to_html()

                # --- MEA Arm-Trigger Invariance check ---
                # K joint tests (one per source): tests whether the source arm shifts the
                # joint distribution of all other experiments' variants (including nan).
                # Nan-containing columns = trigger contamination signal (Assumption 1).
                # K(K-1) pairwise tests also computed; stored in by_pair for drill-down.
                K = len(analysis_info.multi_expt_info.expt_info_list)
                if K >= 2:
                    html_str += "<h3>MEA Arm-Trigger Invariance Check</h3>"
                    try:
                        invar_result = check_trigger_invariance(
                            variant_count_df=variant_count_df0,
                        )
                        overall_status = "PASS ✓" if invar_result.passed_by_source else "FLAG ✗"
                        overall_color = "green" if invar_result.passed_by_source else "#cc0000"
                        html_str += (
                            f"<p><b style='color:{overall_color}'>Overall: {overall_status}</b> "
                            f"({invar_result.n_sources} source tests, "
                            f"Bonferroni α={invar_result.alpha_bonferroni_by_source:.4f})</p>"
                        )
                        # Primary: K source tests
                        rows = ""
                        for src, sr in invar_result.by_source.items():
                            name_src = dim_names[src][:20]
                            status = "PASS ✓" if sr.passed else "FLAG ✗"
                            row_bg = "" if sr.passed else " style='background:#fff0f0'"
                            status_color = "green" if sr.passed else "#cc0000"
                            rows += (
                                f"<tr{row_bg}><td>{name_src}</td>"
                                f"<td style='color:{status_color}'><b>{status}</b></td>"
                                f"<td>{sr.test_result.chi2_value:.2f}</td>"
                                f"<td>{sr.p_value:.2e}</td>"
                                f"<td>{sr.cramers_v:.4f}</td>"
                                f"<td>{int(sr.test_result.n_total):,}</td></tr>"
                            )
                        html_str += (
                            "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
                            "<tr><th>Source experiment</th><th>Result</th>"
                            "<th>χ²</th><th>p_value</th><th>Cramér's V</th><th>N</th></tr>" + rows + "</table>"
                        )
                        # Drill-down: K(K-1) pairwise tests
                        pair_rows = ""
                        for (src, tgt), pr in invar_result.by_pair.items():
                            name_src = dim_names[src][:20]
                            name_tgt = dim_names[tgt][:20]
                            status = "PASS ✓" if pr.passed else "FLAG ✗"
                            row_bg = "" if pr.passed else " style='background:#fff0f0'"
                            status_color = "green" if pr.passed else "#cc0000"
                            pair_rows += (
                                f"<tr{row_bg}><td>{name_src} → {name_tgt}</td>"
                                f"<td style='color:{status_color}'><b>{status}</b></td>"
                                f"<td>{pr.test_result.chi2_value:.2f}</td>"
                                f"<td>{pr.p_value:.2e}</td>"
                                f"<td>{pr.cramers_v:.4f}</td>"
                                f"<td>{int(pr.test_result.n_total):,}</td></tr>"
                            )
                        html_str += (
                            f"<details><summary>Pairwise drill-down "
                            f"({invar_result.n_pairs} pairs, "
                            f"Bonferroni α={invar_result.by_pair[(0, 1)].alpha_bonferroni:.4f})</summary>"
                            "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
                            "<tr><th>Pair (source → target)</th><th>Result</th>"
                            "<th>χ²</th><th>p_value</th><th>Cramér's V</th><th>N</th></tr>"
                            + pair_rows + "</table></details>"
                        )
                    except Exception as e:
                        html_str += f"<p>Arm-Trigger Invariance check unavailable: {e}</p>"

                # Metric interaction diagnostics table — appears right after assumption check.
                # File writes (heatmaps) are deferred until write_path is resolved below.
                if not end_user_report and mea_result is not None and mea_result.combined_mea_result is not None:
                    combined_vmstats = mea_result.combined_mea_result.variant_metric_stats_df
                    required_cols = {METRIC_NAME_COL, VARIANT_COL, SAMPLE_COUNT_COL, MEAN_COL, SD_COL, SUM_COL, SUM_SQ_COL}
                    if isinstance(combined_vmstats, pd.DataFrame) and required_cols.issubset(combined_vmstats.columns):
                        interaction_rows = []
                        flagged_metrics = []
                        for metric_name in combined_vmstats[METRIC_NAME_COL].unique():
                            metric_df = combined_vmstats[combined_vmstats[METRIC_NAME_COL] == metric_name].copy()
                            try:
                                check = check_metric_interaction(
                                    metric_stats_df=metric_df,
                                    expt_names=[n[:20] for n in dim_names],
                                    metric_name=str(metric_name),
                                )
                                interaction_result = check["interaction_result"]
                                if interaction_result.flagged:
                                    flagged_metrics.append(str(metric_name))
                                safe_metric = str(metric_name).replace("/", "_")
                                for ts, ts_result in interaction_result.by_trigger_state.items():
                                    r = ts_result.test_result
                                    ts_str = "R" + "".join("1" if b else "0" for b in ts)
                                    interaction_rows.append(
                                        {
                                            "metric": metric_name,
                                            "trigger_state": ts_str,
                                            "dims": str(ts_result.triggered_dims),
                                            "flagged": ts_result.flagged,
                                            "interaction_stat": round(r.f_value, 3),
                                            "p_value": r.p_value,
                                            "dof": r.dof,
                                            "sigma_within_cell": round(r.sigma_within_cell, 4),
                                            "ss_interaction": round(r.ss_interaction, 4),
                                        }
                                    )
                                    for (dim_a, dim_b), pair_viz in check["by_trigger_state"][ts]["pair_heatmaps"].items():
                                        safe_pair = f"{ts_str}_d{dim_a}_x_d{dim_b}"
                                        interaction_figs_to_write.append((safe_metric, safe_pair, pair_viz["means_fig"], pair_viz["residuals_fig"]))
                            except Exception as e:
                                interaction_rows.append(
                                    {
                                        "metric": metric_name,
                                        "trigger_state": None,
                                        "dims": None,
                                        "flagged": None,
                                        "interaction_stat": None,
                                        "p_value": None,
                                        "dof": None,
                                        "sigma_within_cell": None,
                                        "ss_interaction": None,
                                    }
                                )
                                warnings.warn(f"\n*** In `mea.publish`: metric interaction check failed " f"for metric={metric_name}: {e}")

                        if interaction_rows:
                            interaction_summary_df = pd.DataFrame(interaction_rows).sort_values(["metric", "trigger_state"]).reset_index(drop=True)
                            alpha = 0.05
                            bg_colors = tuple(
                                "rgba(250, 0, 0, 0.3)" if (row.p_value is not None and row.p_value < alpha) else "white"
                                for _, row in interaction_summary_df.iterrows()
                            )
                            overall_flagged = bool(flagged_metrics)
                            overall_status = "FLAG ✗" if overall_flagged else "PASS ✓"
                            html_str += "<h2>Metric Interaction Diagnostics</h2>"
                            html_str += (
                                "<p>Post-hoc diagnostic: does the metric respond additively across "
                                "experiment cells within each trigger state? This is NOT a MEA assumption — "
                                "MEA estimates remain valid regardless. A small p_value (highlighted) "
                                "indicates a real interaction effect between the experiments for that "
                                "metric and trigger state.</p>"
                            )
                            # Overall summary box
                            valid_pvals = [
                                (row.p_value, row.metric, row.trigger_state) for _, row in interaction_summary_df.iterrows() if row.p_value is not None
                            ]
                            min_p, min_metric, min_ts = min(valid_pvals, key=lambda x: x[0]) if valid_pvals else (None, None, None)
                            box_color = "rgba(250, 0, 0, 0.15)" if overall_flagged else "rgba(0, 180, 0, 0.1)"
                            border_color = "rgba(200, 0, 0, 0.6)" if overall_flagged else "rgba(0, 150, 0, 0.5)"
                            min_p_str = f"{min_p:.2e}" if min_p is not None else "N/A"
                            flagged_detail = (
                                f"Flagged: {', '.join(flagged_metrics)}&nbsp;&nbsp;|&nbsp;&nbsp;"
                                f"Most significant: {min_metric} / {min_ts} &nbsp;(p = {min_p_str})"
                                if overall_flagged
                                else f"Min p_value: {min_p_str} ({min_metric} / {min_ts})"
                            )
                            html_str += (
                                f'<div style="display:inline-block; border:1px solid {border_color}; background:{box_color}; '
                                f'padding:10px 16px; margin:8px 0; border-radius:4px; font-size:1.05em;">'
                                f"<strong>Overall: {overall_status}</strong>"
                                f"&nbsp;&nbsp;&nbsp;{flagged_detail}"
                                f"</div><br>"
                            )
                            html_str += df_to_html(
                                df=interaction_summary_df,
                                bg_colors=bg_colors,
                                bg_cols=("metric", "trigger_state", "flagged", "interaction_stat", "p_value", "ss_interaction"),
                            )

        if write_path is not None:
            paths.append(write_path)
        else:
            home = os.path.expanduser("~")
            write_path = f"{home}/abvelocity_results/"
            paths.append(write_path)
        if proj_name is not None:
            write_path = f"{write_path}/{proj_name}"
            paths.append(write_path)

        if add_timestamp_to_path:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            write_path = f"{write_path}/{timestamp}"
            print(f"\n*** write_path is:\n {write_path}")
            paths.append(write_path)

        print(f"\n***: `MEA.publish`: write_path: {write_path}")
        for path in paths:
            os.makedirs(path, exist_ok=True)

        # Write deferred interaction heatmap files now that write_path is resolved.
        if interaction_figs_to_write:
            metric_figs_path_int = f"{write_path}/variant_combo_metric_figs/"
            os.makedirs(metric_figs_path_int, exist_ok=True)
            for safe_metric, safe_pair, means_fig, residuals_fig in interaction_figs_to_write:
                means_fig.write_html(f"{metric_figs_path_int}/{safe_metric}_{safe_pair}_interaction_means.html", include_plotlyjs="cdn")
                residuals_fig.write_html(f"{metric_figs_path_int}/{safe_metric}_{safe_pair}_interaction_residuals.html", include_plotlyjs="cdn")

        if mea_result is None:
            print(
                "***\n`MEA.publish`: You do not have any MEA results for any metrics."
                "It might be because you have not passed any metrics."
                "The report still might be generated and include assignment stats."
            )
            print(f"\n*** `mea.publish`: html_file_name: {html_file_name}")
            print(f"\n*** `mea.publish`: markdown_file_name: {markdown_file_name}")
            if html_file_name is not None:
                with open(f"{write_path}/{html_file_name}", "w") as f:
                    f.write(html_str)
                    print(f"\n*** data was written to {write_path}/{html_file_name}.")

            if markdown_file_name is not None:
                with open(f"{write_path}/{markdown_file_name}", "w") as f:
                    f.write(markdown_str)
                    print(f"\n*** data was written to {write_path}/{markdown_file_name}.")

            return {
                "html_str": html_str,
                "markdown_str": markdown_str,
                "paths": paths,
                "file_names": file_names,
            }

        variant_freq_dict = mea_result.variant_freq_dict

        # Define float format for writing dataframes based on `rounding_digits`.
        float_format = f"%.{rounding_digits}f"

        # Write information for the experiments.
        if not end_user_report:
            path = f"{write_path}/expt_stats/"
            os.makedirs(path, exist_ok=True)
            for df_name, df in variant_freq_dict.items():
                if df is not None:
                    print(f"\n*** attempting to write {df_name} from variant_freq_dict")
                    file_name = f"{path}/{df_name}.csv"
                    file_names.append(file_name)
                    print(f"\n*** file_name: {file_name} being written.")

                    # Round float and tuple cols before writing data.
                    round_df(
                        df=df,
                        rounding_digits=rounding_digits,
                        tuple_cols=[CI_COL, CI_PERCENT_COL, DELTA_SUM_CI_COL],
                    )

                    df.to_csv(file_name, index=False, float_format=float_format)

                    df_name_pretty = " ".join([word.capitalize() for word in df_name.split("_")])

                    html_str += f"\n<h2>{df_name_pretty}</h2>\n"
                    html_str += df_to_html(df=df, top_paragraphs=[df_name], caption=df_name, format_dict=FORMAT_DICT)

                    markdown_str += f"""\n\n\n\n##  <font color="blue">{df_name_pretty} </font>\n\n\n\n"""
                    markdown_str += df.to_markdown(index=False)

        # Write raw agg metrics table (no experiment join).
        # Published for both end-user and full reports.
        if mea_result.agg_metrics is not None:
            df_sw = mea_result.agg_metrics.copy()
            round_df(df=df_sw, rounding_digits=rounding_digits)
            html_str += "<h2>Raw Aggregated Metrics (No Experiment Join)</h2>"
            html_str += df_to_html(df=df_sw, caption="Raw Aggregated Metrics (No Experiment Join)")
            file_name = f"{write_path}/raw_agg_metrics.csv"
            file_names.append(file_name)
            df_sw.to_csv(file_name, index=False, float_format=float_format)

        # Write combined tables which include all metrics in one
        for df_name, df in asdict(mea_result.combined_mea_result).items():
            if df is not None:
                print(f"\n*** attempting to write {df_name}")
                print(f"\n*** df.head(): {df.head()}")
                df_name_pretty = " ".join([word.capitalize() for word in df_name.split("_")])
                path = write_path
                file_name = f"{path}/all_metrics_{df_name}.csv"
                file_names.append(file_name)
                print(f"\n*** file_name: {file_name} being written.")
                if len(df) == 0:
                    print(f"\n*** df {df_name} is empty.")
                    if df_name in END_USER_METRIC_RESULT_KEYS or (not end_user_report):
                        html_str += f"""
                            <br><br><br>
                            <h2 style="color: blue; font-size: 28px; margin-top: 20px; margin-bottom: 20px;">
                            {df_name_pretty}
                            </h2>

                            <br>No data as there are no significant or close to significant results.
                        """
                        markdown_str += (
                            f"""\n\n\n\n##  <font color="blue">{df_name_pretty}</font>\n\n\n\n"""
                            """\nNo data as there are no significant or close to significant results.\n"""
                        )

                else:
                    # Round float and tuple cols before writing data.
                    round_df(
                        df=df,
                        rounding_digits=rounding_digits,
                        tuple_cols=[CI_COL, CI_PERCENT_COL, DELTA_SUM_CI_COL],
                    )

                    if "comparison_pair" in df.columns:
                        df.rename(columns={"comparison_pair": "launch"}, inplace=True)

                    df.to_csv(file_name, index=False, float_format=float_format)

                    if end_user_report:
                        cols = [col for col in df.columns if col in END_USER_COLS]
                        df = df[cols]

                    # If it is not an end user report, or it is end user and it qualifies,
                    # we add the table to the html
                    if df_name in END_USER_METRIC_RESULT_KEYS or (not end_user_report):
                        html_str, makdown_str = publish_df_color_code_p_value(
                            df=df,
                            bg_cols=(
                                METRIC_NAME_COL,
                                "launch",
                                "delta_percent",
                                "p_value",
                                "delta_sum",
                            ),
                            df_name=df_name,
                            html_str=html_str,
                            markdown_str=markdown_str,
                            split_col="launch",
                        )

        # Let us publish metric variant mean/SD diagnostics figures if eligible.
        if not end_user_report and publish_metric_variant_stats_fig and mea_result.combined_mea_result.variant_metric_stats_df is not None:
            metric_figs_path = f"{write_path}/variant_combo_metric_figs/"
            # Create the path.
            os.makedirs(metric_figs_path, exist_ok=True)

            variant_metric_stats_df = mea_result.combined_mea_result.variant_metric_stats_df
            # Two more checks before we attempt to create the fugures.
            if isinstance(variant_metric_stats_df, pd.DataFrame) and MEAN_COL in variant_metric_stats_df.columns:
                if SD_COL in variant_metric_stats_df.columns:
                    err_col = SD_COL
                else:
                    err_col = None

                figs = plot_compare_variants(
                    df=variant_metric_stats_df,
                    x_col=VARIANT_COL,
                    y_col=MEAN_COL,
                    split_col=METRIC_NAME_COL,
                    err_col=err_col,
                    title_prefix="Metric: ",
                    width=900,
                    height_per_plot=500,
                )

                for metric, fig in figs.items():
                    html_path = f"{metric_figs_path}/{metric}_variant_mean_sd_diagnostics.html"
                    fig.write_html(str(html_path), include_plotlyjs="cdn")
            else:
                warnings.warn(
                    "\n*** In `mea.publish`: we attempted to create metric diagnostic figs. "
                    f"However, `variant_metric_stats_df`: {variant_metric_stats_df.head()} did not include {MEAN_COL}."
                )

        # Finally we publish the final reports.
        print(f"\n*** `mea.publish`: html_file_name: {html_file_name}")
        print(f"\n*** `mea.publish`: markdown_file_name: {markdown_file_name}")
        if html_file_name is not None:
            with open(f"{write_path}/{html_file_name}", "w") as f:
                f.write(html_str)
                print(f"\n*** data was written to {write_path}/{html_file_name}.")

        if markdown_file_name is not None:
            with open(f"{write_path}/{markdown_file_name}", "w") as f:
                f.write(markdown_str)
                print(f"\n*** data was written to {write_path}/{markdown_file_name}.")

        return {
            "html_str": html_str,
            "markdown_str": markdown_str,
            "paths": paths,
            "file_names": file_names,
        }

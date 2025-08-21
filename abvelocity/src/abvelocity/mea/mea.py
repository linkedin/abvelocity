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
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from abvelocity.get_data.get_expt_stats import get_expt_stats
from abvelocity.param.analysis_info import AnalysisInfo
from abvelocity.param.constants import (
    CATEG_NAN_VALUE,
    CI_COL,
    CI_PERCENT_COL,
    CONTROL_LABEL,
    DELTA_SUM_CI_COL,
    METRIC_NAME_COL,
    TRIGGER_STATE_COUNT_COL,
    VARIANT_COL,
)
from abvelocity.param.launch import Launch
from abvelocity.param.launch_to_comparison_pair import launch_to_comparison_pair
from abvelocity.param.variant import ComparisonPair
from abvelocity.stats.calc_variant_metric_effects import (
    calc_variant_metric_effects,
    compare_variants_with_control,
)
from abvelocity.stats.calc_variant_metric_stats import calc_variant_metric_stats
from abvelocity.stats.two_sample_z_test import two_sample_z_test
from abvelocity.utils.calc_freq import calc_freq
from abvelocity.utils.df_to_html import df_to_html
from abvelocity.utils.round_df import round_df


@dataclass
class MEAMetricResult:
    """This dataclass stores MEA (Multi-Expt Analysis) results for one metric."""

    variant_metric_stats_df: Optional[pd.DataFrame] = None
    """Variant statistics dataframe."""
    variant_effect_df_pairs: Optional[pd.DataFrame] = None
    """Variant effects dataframe for comparison pairs."""
    variant_effect_df_pairs_sig: Optional[pd.DataFrame] = None
    """Variant effects dataframe for comparison pairs for sig results only."""
    variant_effect_df_vs_control: Optional[pd.DataFrame] = None
    """Variant effects dataframe for all variants versus control."""
    univar_variant_metric_stats_df: Optional[pd.DataFrame] = None
    """Univariate variant statistics dataframe."""
    univar_variant_effect_df_vs_control: Optional[pd.DataFrame] = None
    """Univariate variant effects dataframe for all variants versus control."""


# END_USER_METRIC_RESULT_KEYS = ["variant_effect_df_pairs", "variant_effect_df_pairs_sig"]
END_USER_METRIC_RESULT_KEYS = ["variant_effect_df_pairs_sig"]
"""The end user does not need all the genarted tables.
This constant is used to decide what is to be shown in the final report.
Currently we only include the comparisons.
These could be launch effects for example passed using `Launch` and then converted to `ComparisonPair`.
"""

END_USER_COLS = [
    METRIC_NAME_COL,
    "launch",
    "delta",
    "delta_percent",
    CI_PERCENT_COL,
    "p_value",
    "delta_sum",
]
"""
The columns to include in metric effect tables for end user.
"""


@dataclass
class MEAResult:
    """This dataclass stores the results of MEA (Multi-Expt Analysis).
    The results include variant frequencies and MEA results for each metric.
    """

    variant_freq_dict: Optional[dict[str, pd.DataFrame]] = None
    """Dictionary of variant assignment frequencies.
        The keys are:
            - expt_1
            - expt_2
            ...
            - expt_k (k is the number of experiments)
            - multi-expt
    """
    metric_result_dict: Optional[dict[str, MEAMetricResult]] = None
    """Dictionary of MEA results for each metric.
        The keys are the metric names.
    """
    combined_mea_result: Optional[MEAMetricResult] = None
    """
    This is an `MEAMetricResult` which includes all metrics in its corresponding tables.
    This is done by concatenating all the dataframes in individual metric results.
    To each dataframe a new "metric" column added and then data are concatenated
    vertically.
    """

    def gen_combined_mea_result(self) -> None:
        """
        This creates an `MEAMetricResult` which includes all metrics in its corresponding tables.
            This is done by concatenating all the dataframes in individual metric results.
            To each dataframe a new "metric" column added and then data are concatenated
            vertically.

        Alters: `self.combined_mea_result`
        """
        self.combined_mea_result = MEAMetricResult()
        # We only combine the key tables.
        variant_metric_stats_df = None
        variant_effect_df_pairs = None
        # add a metric  name column and concat al dataframes.
        for metric, mea_metric_result in self.metric_result_dict.items():
            variant_metric_stats_df0 = mea_metric_result.variant_metric_stats_df
            if variant_metric_stats_df0 is not None:
                variant_metric_stats_df0[METRIC_NAME_COL] = metric
                cols = list(variant_metric_stats_df0.columns)
                # re-arrange the columns
                cols = [cols[-1]] + cols[:-1]
                variant_metric_stats_df0 = variant_metric_stats_df0[cols]

                variant_metric_stats_df = pd.concat(
                    [variant_metric_stats_df0, variant_metric_stats_df], axis=0
                )
            variant_effect_df_pairs0 = mea_metric_result.variant_effect_df_pairs
            if variant_effect_df_pairs0 is not None:
                variant_effect_df_pairs0[METRIC_NAME_COL] = metric
                cols = list(variant_effect_df_pairs0.columns)
                # re-arrange the columns
                cols = [cols[-1]] + cols[:-1]
                variant_effect_df_pairs0 = variant_effect_df_pairs0[cols]

                variant_effect_df_pairs = pd.concat(
                    [variant_effect_df_pairs0, variant_effect_df_pairs], axis=0
                )

        variant_effect_df_pairs_sig = variant_effect_df_pairs.loc[
            variant_effect_df_pairs["p_value"] < 0.10
        ]
        self.combined_mea_result.variant_metric_stats_df = variant_metric_stats_df
        self.combined_mea_result.variant_effect_df_pairs = variant_effect_df_pairs
        self.combined_mea_result.variant_effect_df_pairs_sig = (
            variant_effect_df_pairs_sig.reset_index(drop=True)
        )

    def combine(self, other):
        """
        This function will combine the results from two mea runs.
        For dict objects it will update them using other.

        For the dataframes given in `combined_mea_result`, it will concat them
        vertically.
        """
        if self.variant_freq_dict and other.variant_freq_dict:
            self.variant_freq_dict.update(other.variant_freq_dict)
        elif other.variant_freq_dict:
            self.variant_freq_dict = other.variant_freq_dict

        if self.metric_result_dict and other.metric_result_dict:
            self.metric_result_dict.update(other.metric_result_dict)
        elif other.metric_result_dict:
            self.metric_result_dict = other.metric_result_dict

        if self.combined_mea_result and other.combined_mea_result:
            field_names = [
                "variant_metric_stats_df",
                "variant_effect_df_pairs",
                "variant_effect_df_pairs_sig",
            ]
            for field_name in field_names:
                res1 = getattr(self.combined_mea_result, field_name)
                res2 = getattr(other.combined_mea_result, field_name)

                if res1 is not None and res2 is not None:
                    res = pd.concat([res1, res2], ignore_index=True)
                elif res2 is not None:
                    res = res2

                setattr(self.combined_mea_result, field_name, res)
        elif other.combined_mea_result:
            self.combined_mea_result = other.combined_mea_result


@dataclass
class MEAReport:
    """This encodes the MEA report.
    The main component is `html_str` which can be published.
    """

    html_str: str = ""
    """
    html string which can be stored in an html file.
    """
    file_names: Optional[list] = None
    """
    file names generated during publish, if any.
    """
    paths: Optional[list] = None
    """
    paths generated during publish if any.
    """


class MEA:
    """This class is used to run multi-experiment analysis (MEA).
    The main method is `run` which generates the results (See `~abvelocity.mea.mea.MEAResult`).
    The class also includes a `publish` method to generate html string and write results to csv files.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        analysis_info: Optional[AnalysisInfo] = None,
        launches: Optional[list[Launch]] = None,
        control_launch: Optional[Launch] = None,
        comparison_pairs: Optional[list[ComparisonPair]] = None,
        ci_coverage: float = 0.95,
        recalculate_expt_stats: bool = False,
    ):
        """Initializes the class with the data, analysis info and comparison pairs.

        Args:
            df: Dataframe with the data.
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
            ci_coverage: coverage of conf interval.
            recalculate_expt_stats: A bool to decide if experiment derived stats should be recalculated.

        Attributes (other than the input arguments):
            mea_result: Result of MEA. This is attached after running the analysis.
        """
        self.df = df
        self.analysis_info = analysis_info
        self.launches = launches
        self.control_launch = control_launch
        self.comparison_pairs = comparison_pairs
        self.ci_coverage = ci_coverage
        self.recalculate_expt_stats = recalculate_expt_stats

        # Attributes
        self.result = None

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
        df = self.df
        analysis_info = self.analysis_info
        launches = self.launches
        control_launch = self.control_launch
        comparison_pairs = self.comparison_pairs
        expt_info_list = analysis_info.multi_expt_info.expt_info_list

        if (not analysis_info.multi_expt_info.derived_stats) or self.recalculate_expt_stats:
            analysis_info.multi_expt_info.derived_stats = get_expt_stats(df=df)

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
                comparison_pair = launch_to_comparison_pair(
                    launch=launch, control_launch=control_launch
                )
                comparison_pairs.append(comparison_pair)

        if not analysis_info.metric_info_list:
            self.result = None
            warnings.warn(
                f"\n***: metric_info_list is None or empty = {analysis_info.metric_info_list}"
                "This means MEA will only generate overlap results.",
                UserWarning,
            )
            return None

        metrics = []
        for metric_info in analysis_info.metric_info_list:
            if metric_info.metrics is None:
                raise ValueError(
                    "At this stage metrics need to be available."
                    f"Check analysis_info: {analysis_info}"
                    f"Check metric_info: {metric_info}"
                )
            metrics += metric_info.metrics

        print(f"\n *** MEA: metrics:\n{metrics}")
        num_expts = len(expt_info_list)

        print(f"\n*** MEA: df:\n{df.head(2)}")

        # Get stats about experiment variant assignments
        variant_freq_dict = {}
        for i in range(num_expts):
            col = f"{VARIANT_COL}_{i + 1}"
            variant_freq_dict[f"expt_{i + 1}"] = calc_freq(
                df=df[df[col] != CATEG_NAN_VALUE], cols=[col]
            )

        variant_freq_dict["multi-expt"] = calc_freq(df=df, cols=[VARIANT_COL])

        metric_result_dict = {}
        expt_control = (CONTROL_LABEL,) * num_expts
        print(f"\n*** MEA: expt_control:\n{expt_control}")

        # Get stats about the multi-experiment
        variant_count_df = analysis_info.multi_expt_info.derived_stats.variant_count_df

        for metric in metrics:
            print(f"\n**** MEA for metric name: {metric.name}")
            print(f"\n**** MEA for metric def: {metric}")
            metric_result_dict[metric.name] = MEAMetricResult()
            # TODO: extend to more general metrics (allow denominators)
            # Currently it works only for metrics with a numerator.
            variant_metric_stats_df = calc_variant_metric_stats(
                df=df, metric=metric, variant_col=VARIANT_COL
            )

            variant_effect_df_pairs = None
            if comparison_pairs:  # this will make sure its not None or empty.
                variant_effect_df_pairs = calc_variant_metric_effects(
                    variant_metric_stats_df=variant_metric_stats_df,
                    variant_col=VARIANT_COL,
                    comparison_pairs=comparison_pairs,
                    stats_test_func=two_sample_z_test,
                    ci_coverage=self.ci_coverage,
                    variant_count_df=variant_count_df,
                    trigger_state_count_col=TRIGGER_STATE_COUNT_COL,
                )

            variant_effect_df_vs_control = compare_variants_with_control(
                variant_metric_stats_df=variant_metric_stats_df,
                variant_col=VARIANT_COL,
                expt_control=expt_control,
                stats_test_func=two_sample_z_test,
                ci_coverage=self.ci_coverage,
                variant_count_df=variant_count_df,
                trigger_state_count_col=TRIGGER_STATE_COUNT_COL,
            )

            metric_result_dict[metric.name].variant_metric_stats_df = variant_metric_stats_df
            metric_result_dict[metric.name].variant_effect_df_pairs = variant_effect_df_pairs
            metric_result_dict[metric.name].variant_effect_df_vs_control = (
                variant_effect_df_vs_control
            )

        # Univar results
        for metric in metrics:
            univar_variant_metric_stats_df = None
            univar_variant_effect_df_vs_control = None
            for i in range(num_expts):
                # TODO: extend to more general metrics (allow denominators)
                variant_metric_stats_df = calc_variant_metric_stats(
                    df=df, metric=metric, variant_col=f"{VARIANT_COL}_{i + 1}"
                )
                cols = list(variant_metric_stats_df.columns)
                # re-arrange the columns
                cols = [cols[-1]] + cols[:-1]
                variant_metric_stats_df = variant_metric_stats_df[cols]
                # Change the variant column name from `f"{VARIANT_COL}_{i + 1}"` to `VARIANT_COL`.
                # This is needed for merging univariate data across experiments.
                variant_metric_stats_df.rename(
                    columns={f"{VARIANT_COL}_{i + 1}": VARIANT_COL}, inplace=True
                )

                univar_variant_metric_stats_df = pd.concat(
                    [univar_variant_metric_stats_df, variant_metric_stats_df], axis=0
                )

                if expt_info_list[i].derived_stats:
                    variant_count_df_uni = expt_info_list[i].derived_stats.variant_count_df
                else:
                    variant_count_df_uni = None

                variant_effect_df_vs_control = compare_variants_with_control(
                    variant_metric_stats_df=variant_metric_stats_df,
                    variant_col=VARIANT_COL,
                    expt_control=CONTROL_LABEL,
                    stats_test_func=two_sample_z_test,
                    ci_coverage=self.ci_coverage,
                    variant_count_df=variant_count_df_uni,
                    trigger_state_count_col=TRIGGER_STATE_COUNT_COL,
                )

                variant_effect_df_vs_control["expt"] = f"expt_{i}"

                univar_variant_effect_df_vs_control = pd.concat(
                    [univar_variant_effect_df_vs_control, variant_effect_df_vs_control], axis=0
                )
                cols = list(univar_variant_effect_df_vs_control.columns)
                # Reshuffles column order.
                cols = [cols[-1]] + cols[:-1]
                univar_variant_effect_df_vs_control = univar_variant_effect_df_vs_control[cols]

                metric_result_dict[metric.name].univar_variant_metric_stats_df = (
                    univar_variant_metric_stats_df
                )
                metric_result_dict[metric.name].univar_variant_effect_df_vs_control = (
                    univar_variant_effect_df_vs_control
                )

        # Attach the results to the class instance.
        self.result = MEAResult(
            variant_freq_dict=variant_freq_dict, metric_result_dict=metric_result_dict
        )

        # Generate the combined metrics results (one table for all metrics for each quantity)
        self.result.gen_combined_mea_result()

        return None

    def publish_df_color_code_p_value(
        self,
        df: pd.DataFrame,
        bg_cols: tuple[str],
        df_name: str = "",
        html_str: str = "",
        markdown_str: str = "",
        split_col: Optional[str] = None,
    ):
        """
        Publishes a DataFrame with color-coded values for statistical significance and
        returns the corresponding HTML and Markdown representations.

        This function color-codes cells in the DataFrame based on their `p_value` and `delta_percent`.
        Rows with a `p_value` below 0.05 are highlighted: green for positive `delta_percent` and
        red for negative `delta_percent`. The DataFrame can be split and displayed by a specified column.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            bg_cols (tuple[str]): A tuple of column names to use for background coloring.
            df_name (str, optional): A name for the DataFrame, used in the output captions. Default is an empty string.
            html_str (str, optional): Initial HTML string to append the DataFrame's representation. Default is an empty string.
            markdown_str (str, optional): Initial Markdown string to append the DataFrame's representation. Default is an empty string.
            split_col (str, optional): A column name to split the DataFrame by. If specified, separate tables are
                created for each unique value in this column. Default is None.

        Returns:
            tuple[str, str]: A tuple containing:
                - html_str (str): The resulting HTML string with the DataFrame's representation.
                - markdown_str (str): The resulting Markdown string with the DataFrame's representation.

        Notes:
            - Cells in the DataFrame are color-coded based on their `p_value` and `delta_percent` values:
              - Light green (`rgba(0, 250, 0, 0.5)`) for `p_value < 0.05` and `delta_percent >= 0`.
              - Light red (`rgba(250, 0, 0, 0.5)`) for `p_value < 0.05` and `delta_percent < 0`.
            - If `split_col` is specified, the function generates separate tables for each unique value
              in the column and appends them to the HTML and Markdown strings.
        """

        df_name_pretty = " ".join([word.capitalize() for word in df_name.split("_")])

        html_str += f"""
            <br><br><br>
            <h2 style="color: blue; font-size: 28px; margin-top: 20px; margin-bottom: 20px;">
            {df_name_pretty}
            </h2>
        """

        def func(df0, html_str, markdown_str):
            bg_colors = None
            # color coding
            if "delta_percent" in df0.columns and "p_value" in df.columns:
                bg_colors = []
                for row in df0.itertuples():
                    if row.p_value < 0.05:
                        if row.delta_percent >= 0.0:
                            bg_colors.append("rgba(0, 250, 0, 0.5)")  # Light green
                        else:
                            bg_colors.append("rgba(250, 0, 0, 0.5)")  # Light red
                    else:
                        bg_colors.append(None)
                bg_colors = tuple(bg_colors)

            html_str += df_to_html(
                df=df0,
                top_paragraphs=[],
                caption="",
                bg_colors=bg_colors,
                bg_cols=bg_cols,
            )

            markdown_str += f"""\n\n\n\n##  <font color="blue">{df_name_pretty}</font>\n\n\n\n"""
            markdown_str += "\n\n" + df0.to_markdown(index=False)

            return html_str, markdown_str

        if split_col is not None and split_col in df.columns:
            for x in df[split_col].unique():
                html_str += f"\n<h2>{df_name_pretty}: {x}</h2>\n"
                df0 = df[df[split_col] == x].reset_index(drop=True)
                html_str, markdown_str = func(df0=df0, html_str=html_str, markdown_str=markdown_str)
        else:
            html_str += f"\n<h2>{df_name_pretty}</h2>\n"
            html_str, markdown_str = func(df0=df, html_str=html_str, markdown_str=markdown_str)

        return html_str, markdown_str

    def publish(
        self,
        mea_result: Optional[MEAResult] = None,
        analysis_info: Optional[AnalysisInfo] = None,
        write_path: Optional[str] = None,
        proj_name: Optional[str] = None,
        add_timestamp_to_path: bool = True,
        rounding_digits: int = 4,
        html_file_name: Optional[str] = None,
        markdown_file_name: Optional[str] = None,
        end_user_report: bool = True,
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
        rounding_digits: Number of digits to round the results to. Default 4.
        html_file_name: If passed, the MEA results will be written to an html file.
        markdown_file_name: If passed, the MEA results will be written to markdown file.
        end_user_report: If True, only the end user report will be generated which is based
            on the tables specified in `~abvelocity.mea.mea.END_USER_METRIC_RESULT_KEYS`.

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

        # This will add experiments information to the html report.
        if analysis_info is not None:
            html_str += analysis_info.to_html()
            markdown_str += analysis_info.to_html()

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
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                print(f"***\n path: {path} already exists")

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

        # metric_result_dict = mea_result.metric_result_dict
        variant_freq_dict = mea_result.variant_freq_dict

        # Define float format for writing dataframes based on `rounding_digits`.
        float_format = f"%.{rounding_digits}f"

        # Write information for the experiments.
        if not end_user_report:
            path = f"{write_path}/expt_stats/"
            if not os.path.exists(path):
                os.makedirs(path)
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
                    html_str += df_to_html(df=df, top_paragraphs=[df_name], caption=df_name)

                    markdown_str += (
                        f"""\n\n\n\n##  <font color="blue">{df_name_pretty} </font>\n\n\n\n"""
                    )
                    markdown_str += df.to_markdown(index=False)

        # Write MEA results for each metric.
        """
        if not end_user_report:
            for metric, mea_metric_result in metric_result_dict.items():
                metric_result = asdict(mea_metric_result)
                if end_user_report:
                    metric_result = {
                        k: v for k, v in metric_result.items() if k in END_USER_METRIC_RESULT_KEYS
                    }

                path = f"{write_path}/{metric}"
                paths.append(path)
                if not os.path.exists(path):
                    os.makedirs(path)
                else:
                    print(f"\n*** path: {path} already exists")

                for df_name, df in metric_result.items():
                    if df is not None:
                        print(f"\n*** attempting to write {df_name} for {metric}")
                        print(df)
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
                        html_str += df_to_html(
                            df=df,
                            top_paragraphs=[f"metric: {metric}, {df_name}"],
                            caption=f"metric: {metric}; {df_name}",
                        )
                        markdown_str += df.to_markdown()
        """

        # Write combined tables which include all metrics in one
        for df_name, df in asdict(mea_result.combined_mea_result).items():
            if df is not None:
                print(f"\n*** attempting to write {df_name}")
                print(df)
                path = write_path
                file_name = f"{path}/all_metrics_{df_name}.csv"
                file_names.append(file_name)
                print(f"\n*** file_name: {file_name} being written.")
                if len(df) == 0:
                    print(f"\n*** df {df_name} is empty.")
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

                    # Limit the number of columns if this is end user report
                    """
                    if end_user_report:
                        cols = [col for col in df.columns if col in END_USER_COLS]
                        df = df[cols]
                    """

                    df_name_pretty = " ".join([word.capitalize() for word in df_name.split("_")])

                    # If it is not an end user report, or it is end user and it qualifies,
                    # we add the table to the html
                    if df_name in END_USER_METRIC_RESULT_KEYS or (not end_user_report):
                        html_str, makdown_str = self.publish_df_color_code_p_value(
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

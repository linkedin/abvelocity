# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
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

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from abvelocity.core.mea.constants import METRIC_NAME_COL
from abvelocity.core.utils.serialization import DataFrameConfig
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class MEAMetricResult(DataClassJSONMixin):
    """This dataclass stores MEA (Multi-Expt Analysis) results for one metric."""

    variant_metric_stats_df: Optional[pd.DataFrame] = None
    """Variant statistics dataframe."""
    variant_effect_df_pairs: Optional[pd.DataFrame] = None
    """Variant effects dataframe for comparison pairs."""
    variant_effect_df_pairs_sig: Optional[pd.DataFrame] = None
    """Variant effects dataframe for comparison pairs for sig results only."""

    class Config(DataFrameConfig):
        pass

    # variant_effect_df_vs_control: Optional[pd.DataFrame] = None
    """Variant effects dataframe for all variants versus control."""


@dataclass
class MEAResult(DataClassJSONMixin):
    """This dataclass stores the results of MEA (Multi-Experiment Analysis).
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
    agg_metrics: Optional[pd.DataFrame] = None
    """
    Raw aggregated metrics computed on all data (no experiment join).
    Produced by `get_agg_metrics_df`. Includes a leading
    `metric_family` column followed by the agg metric columns for each
    MetricInfo in the analysis: {metric}_numer, {metric}_denom (if ratio),
    {metric}, {metric}_sample_count (if specified), dim columns, sample_count.
    """

    class Config(DataFrameConfig):
        pass

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

                variant_metric_stats_df = pd.concat([variant_metric_stats_df0, variant_metric_stats_df], axis=0)
            variant_effect_df_pairs0 = mea_metric_result.variant_effect_df_pairs
            if variant_effect_df_pairs0 is not None:
                variant_effect_df_pairs0[METRIC_NAME_COL] = metric
                cols = list(variant_effect_df_pairs0.columns)
                # re-arrange the columns
                cols = [cols[-1]] + cols[:-1]
                variant_effect_df_pairs0 = variant_effect_df_pairs0[cols]

                variant_effect_df_pairs = pd.concat([variant_effect_df_pairs0, variant_effect_df_pairs], axis=0)

        variant_effect_df_pairs_sig = variant_effect_df_pairs.loc[variant_effect_df_pairs["p_value"] < 0.10]
        self.combined_mea_result.variant_metric_stats_df = variant_metric_stats_df
        self.combined_mea_result.variant_effect_df_pairs = variant_effect_df_pairs
        self.combined_mea_result.variant_effect_df_pairs_sig = variant_effect_df_pairs_sig.reset_index(drop=True)

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
class MEAReport(DataClassJSONMixin):
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

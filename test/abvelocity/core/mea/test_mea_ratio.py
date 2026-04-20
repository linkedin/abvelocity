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
# Original Author: Reza Hosseini

import os
from pathlib import Path

import pytest
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.mea.mea import MEA
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.sim.ctr_example import COMPARISON_PAIRS, sim_ctr_data
from abvelocity.core.sim.sim import EXPT_UNIT_COL

# Create save path for html files (figures)
SAVE_PATH = str(Path(__file__).parents[4].joinpath("docs/static/test-results/mea-ratio/").resolve())
os.makedirs(SAVE_PATH, exist_ok=True)

# Define the MEA method constant
METHOD = "general"
"""Constant to denote the MEA method used in this test."""

# --- Fixtures ---


@pytest.fixture
def example_analysis_info():
    """
    Creates AnalysisInfo for a multi-variate experiment, defining unit metrics (impressions, clicks)
    and the CTR ratio metric.
    """
    # 1. Define the constituent experiments (two experiments in the simulation)
    expt1 = ExptInfo(test_key="expt_A")  # Implicitly uses variants 'control', 'enabled'
    expt2 = ExptInfo(test_key="expt_B")  # Implicitly uses variants 'control', 'v1'

    # 2. Define the MultiExptInfo, merging the two experiments
    multi_expt_info = MultiExptInfo(
        expt_info_list=[expt1, expt2],
        merge_method="cross",  # Cartesian product of variants
        expt_unit_col=EXPT_UNIT_COL,
    )

    # 3. Define metrics
    impressions_metric = Metric(numerator=UMetric(col="impressions"), name="impressions")

    clicks_metric = Metric(numerator=UMetric(col="clicks"), name="clicks")

    ctr_metric = Metric(numerator=UMetric(col="clicks"), denominator=UMetric(col="impressions"), name="CTR")

    metrics = [impressions_metric, clicks_metric, ctr_metric]

    # 4. Combine all into AnalysisInfo
    analysis_info = AnalysisInfo(
        multi_expt_info=multi_expt_info,
        metric_info_list=[MetricInfo(metrics=metrics)],
    )

    return analysis_info


# --- Test Function ---


def test_mea_ratio_multi_expt_sim(example_analysis_info):
    """
    Tests MEA calculation for unit and ratio metrics in a multi-experiment
    setting using the simulated data and validates against true effects.
    """
    # 1. Setup: Generate data using the multi-experiment simulator
    # Unpack the tuple: (df, strata_info, true_values)
    joined_expt_metric_df, _, true_effects = sim_ctr_data()
    joined_expt_metric_df2 = joined_expt_metric_df[:80]
    analysis_info = example_analysis_info

    # 2. Execution: Run MEA
    # comparison_pairs is passed directly to MEA as the AnalysisInfo did not include it
    mea = MEA(
        dc=DataContainer(pandas_df=joined_expt_metric_df2),
        analysis_info=analysis_info,
        comparison_pairs=COMPARISON_PAIRS,
        launches=None,
        method=METHOD,
        # num_buckets=10,
    )
    mea.run()

    # 3. Validation: Get results
    effect_df = mea.result.combined_mea_result.variant_effect_df_pairs

    # 4. Prepare paths and write summary HTML (MINIMAL VERSION)
    write_path = f"{SAVE_PATH}/mea-ratio-test1/"
    os.makedirs(write_path, exist_ok=True)  # Ensure path exists before writing custom file
    true_values_file = Path(write_path) / "mea-ratio-test1-true-values.html"

    # Generate simple string representation for true effects
    true_effects_str = str(true_effects).replace(", ", ",\n")

    # Minimal HTML content using only df.to_html() and simple headers
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MEA Calculated and True Effects Summary</title>
        </head>
        <body>
            <h1>1. Calculated MEA Variant Effects (effect_df)</h1>
            {effect_df.to_html(index=False)}

            <h1>2. True Simulated Effects (true_effects)</h1>
            <pre>{true_effects_str}</pre>
        </body>
        </html>
    """

    # Write the custom HTML file
    with open(true_values_file, "w") as f:
        f.write(html_content)

    # Print statement for file creation confirmation
    print(f"\nSuccessfully wrote MEA and True Values Summary to: {true_values_file}")

    # Assert the core ratio calculation correctness (CTR)
    calculated_ctr_pct = effect_df[(effect_df["metric"] == "CTR") & (effect_df["comparison_pair"] == "sim focus: launch ('enabled', 'v1')")][
        "delta_percent"
    ].iloc[0]

    true_ctr_pct = true_effects["ctr_pct_diff"]

    # Use a low tolerance to confirm the ratio calculation is accurate
    assert pytest.approx(true_ctr_pct, rel=0.2) == calculated_ctr_pct

    # Assert the unit metric calculation correctness (Impressions)
    calculated_impressions_pct = effect_df[(effect_df["metric"] == "impressions") & (effect_df["comparison_pair"] == "sim focus: launch ('enabled', 'v1')")][
        "delta_percent"
    ].iloc[0]

    true_impressions_pct = true_effects["impressions_pct_diff"]

    # Use a low tolerance to confirm the unit metric calculation is accurate
    assert pytest.approx(true_impressions_pct, rel=0.2) == calculated_impressions_pct

    # 5. Publish the official MEA report
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=3,
        html_file_name="mea-ratio-test1.html",
        end_user_report=False,
    )

    assert mea_report is not None

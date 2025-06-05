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

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abvelocity.get_data.data_container import DataContainer
from abvelocity.get_data.join_expt_dfs import join_expt_dfs
from abvelocity.get_data.join_expt_with_metric_df import join_expt_with_metric_df
from abvelocity.mea.mea import MEA, MEAMetricResult, MEAResult
from abvelocity.param.analysis_info import AnalysisInfo
from abvelocity.param.constants import TRIGGER_TIME_COL, UNIT_COL, VARIANT_COL
from abvelocity.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.param.launch import Launch
from abvelocity.param.metric import Metric, UMetric
from abvelocity.param.metric_info import MetricInfo
from abvelocity.param.variant import ComparisonPair, Variant, VariantList
from abvelocity.sim.examples import (
    simulate_data_multi1,
    simulate_data_multi2,
    simulate_data_multi3,
    simulate_data_multi4,
    simulate_data_multi5,
    simulate_data_three1,
    simulate_data_three2,
    simulate_data_uni1,
    simulate_data_uni2,
    simulate_data_uni3,
)

# Create save path for html files (figures)
SAVE_PATH = str(Path(__file__).parents[5].joinpath("docs/static/test-results/mea/").resolve())
os.makedirs(SAVE_PATH, exist_ok=True)


@pytest.fixture
def mea_result_1():
    df1 = pd.DataFrame(
        {
            "metric_col": ["A", "B"],
            "value": [10, 20],
        }
    )
    df2 = pd.DataFrame(
        {
            "metric_col": ["C", "D"],
            "value": [30, 40],
        }
    )
    metric_result_1 = MEAMetricResult(variant_metric_stats_df=df1, variant_effect_df_pairs=df2)
    return MEAResult(
        variant_freq_dict={"expt_1": pd.DataFrame({"freq": [0.1, 0.2]})},
        metric_result_dict={"metric_1": metric_result_1},
        combined_mea_result=metric_result_1,
    )


@pytest.fixture
def mea_result_2():
    df1 = pd.DataFrame(
        {
            "metric_col": ["E", "F"],
            "value": [50, 60],
        }
    )
    df2 = pd.DataFrame(
        {
            "metric_col": ["G", "H"],
            "value": [70, 80],
        }
    )
    metric_result_2 = MEAMetricResult(variant_metric_stats_df=df1, variant_effect_df_pairs=df2)

    return MEAResult(
        variant_freq_dict={"expt_2": pd.DataFrame({"freq": [0.3, 0.4]})},
        metric_result_dict={"metric_2": metric_result_2},
        combined_mea_result=metric_result_2,
    )


def test_mea_result_combine_method(mea_result_1, mea_result_2):
    # Perform the combination
    mea_result_1.combine(mea_result_2)

    # Assert that the variant_freq_dict is updated
    assert "expt_2" in mea_result_1.variant_freq_dict
    assert mea_result_1.variant_freq_dict["expt_2"].equals(pd.DataFrame({"freq": [0.3, 0.4]}))

    # Assert that the metric_result_dict is updated
    assert "metric_2" in mea_result_1.metric_result_dict

    # Assert combined dataframes in combined_mea_result
    combined_variant_metric_stats_df = pd.concat(
        [
            pd.DataFrame({"metric_col": ["A", "B"], "value": [10, 20]}),
            pd.DataFrame({"metric_col": ["E", "F"], "value": [50, 60]}),
        ],
        ignore_index=True,
    )
    combined_variant_effect_df_pairs = pd.concat(
        [
            pd.DataFrame({"metric_col": ["C", "D"], "value": [30, 40]}),
            pd.DataFrame({"metric_col": ["G", "H"], "value": [70, 80]}),
        ],
        ignore_index=True,
    )

    # Validate combined result dataframes
    assert mea_result_1.combined_mea_result.variant_metric_stats_df.equals(
        combined_variant_metric_stats_df
    )
    assert mea_result_1.combined_mea_result.variant_effect_df_pairs.equals(
        combined_variant_effect_df_pairs
    )


@pytest.fixture
def simulated_data0():
    # This is a simple data simulation done on the spot within this module.
    # More sophisticated simulations are imported in the above from the `sim` module.
    sampler = np.random.default_rng(1317)

    # Population and sample sizes.
    total_unit_num = 5000
    expt1_unit_num = int(total_unit_num * 0.9)
    expt2_unit_num = int(total_unit_num * 0.9)
    metric_unit_num = int(total_unit_num * 0.95)

    expt_df1 = pd.DataFrame(
        {
            UNIT_COL: sampler.choice(range(total_unit_num), size=expt1_unit_num),
            VARIANT_COL: sampler.choice(["v1", "v2", "control"], size=expt1_unit_num),
            TRIGGER_TIME_COL: sampler.choice(range(total_unit_num), size=expt1_unit_num),
        }
    )

    expt_df2 = pd.DataFrame(
        {
            UNIT_COL: sampler.choice(range(total_unit_num), size=expt2_unit_num),
            VARIANT_COL: sampler.choice(["test", "control"], size=expt2_unit_num),
            TRIGGER_TIME_COL: sampler.choice(range(total_unit_num), size=expt2_unit_num),
        }
    )

    # We assume we do find metric data for most of the units.
    metric_df = pd.DataFrame(
        {
            UNIT_COL: sampler.choice(range(total_unit_num), size=metric_unit_num),
            "metric1": sampler.normal(loc=0.0, scale=1.0, size=metric_unit_num),
            "metric2": sampler.normal(loc=0.0, scale=1.0, size=metric_unit_num),
        }
    )

    expt_dc_list = [
        DataContainer(df=expt_df1.copy(), is_df=True),
        DataContainer(df=expt_df2.copy(), is_df=True),
    ]

    metric_dc = DataContainer(df=metric_df, is_df=True)
    expt_dc = join_expt_dfs(dc_list=expt_dc_list)

    joined_expt_metric_dc = join_expt_with_metric_df(expt_dc=expt_dc, metric_dc=metric_dc)
    joined_expt_metric_df = joined_expt_metric_dc.df

    # Now we impose some signals on the data
    joined_expt_metric_df.loc[joined_expt_metric_df["variant_1"] == "control", "metric1"] += 2.0
    joined_expt_metric_df.loc[joined_expt_metric_df["variant_1"] == "v1", "metric1"] += 3.0
    joined_expt_metric_df.loc[joined_expt_metric_df["variant_1"] == "v2", "metric1"] += 3.0

    joined_expt_metric_df.loc[joined_expt_metric_df["variant_2"] == "control", "metric1"] += 2.0
    joined_expt_metric_df.loc[joined_expt_metric_df["variant_2"] == "test", "metric1"] += 3.0

    # Define a boolen index based on interaction between expt 1 and expt 2.
    # We impose a synergic effect between v1 in expt 1 and test in experiment 2.
    ind = (joined_expt_metric_df["variant_1"] == "v1") & (
        joined_expt_metric_df["variant_2"] == "test"
    )

    joined_expt_metric_df.loc[ind, "metric1"] += 3.0

    return joined_expt_metric_df


@pytest.fixture
def example_analysis_info():
    # Create AnalysisInfo.
    # Note that we do not query data, all fields for ExptInfo are dummy.
    expt1 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    expt2 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    metrics = [Metric(numerator=UMetric(col="metric1")), Metric(numerator=UMetric(col="metric2"))]

    analysis_info = AnalysisInfo(
        MultiExptInfo(expt_info_list=[expt1, expt2], merge_method="cross"),
        metric_info_list=[MetricInfo(metrics=metrics)],
    )

    return analysis_info


@pytest.fixture
def example_uni_analysis_info():
    # Create AnalysisInfo for a univariate experiment.
    # Note that we do not query data, all fields for ExptInfo are dummy.
    expt1 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    metrics = [Metric(numerator=UMetric(col="metric1")), Metric(numerator=UMetric(col="metric2"))]
    analysis_info = AnalysisInfo(
        MultiExptInfo(expt_info_list=[expt1], merge_method="cross"),
        metric_info_list=[MetricInfo(metrics=metrics)],
    )

    return analysis_info


@pytest.fixture
def example_three_analysis_info():
    # Create AnalysisInfo.
    # Note that we do not query data, all fields for ExptInfo are dummy.
    expt1 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    expt2 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    expt3 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    metrics = [Metric(numerator=UMetric(col="metric1")), Metric(numerator=UMetric(col="metric2"))]
    analysis_info = AnalysisInfo(
        MultiExptInfo(expt_info_list=[expt1, expt2, expt3], merge_method="cross"),
        metric_info_list=[MetricInfo(metrics=metrics)],
    )

    return analysis_info


@pytest.fixture
def example_analysis_info_metric_info_list():
    # Create AnalysisInfo.
    # Note that we do not query data, all fields for ExptInfo are dummy.
    expt1 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    expt2 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    expt3 = ExptInfo(
        test_key=None,
        experiment_id=None,
        segment_id=None,
        start_date=None,
        end_date=None,
        variants=None,
        control_label=None,
    )

    metrics1 = [Metric(numerator=UMetric(col="metric1"))]
    metrics2 = [Metric(numerator=UMetric(col="metric2"))]
    analysis_info = AnalysisInfo(
        MultiExptInfo(expt_info_list=[expt1, expt2, expt3], merge_method="cross"),
        metric_info_list=[MetricInfo(metrics=metrics1), MetricInfo(metrics=metrics2)],
    )

    return analysis_info


def test_mea(simulated_data0, example_analysis_info):
    """Tests `mea` function."""
    joined_expt_metric_df = simulated_data0.copy()
    analysis_info = example_analysis_info

    comparison_pairs = [
        # Assesssing expt1: "v1", if expt 2 is not launched ("control" variant).
        # Impact of (v1 | expt2:control launched)
        ComparisonPair(
            treatment=VariantList(
                variants=[Variant(value=("v1", "control")), Variant(value=("v1", "nan"))]
            ),
            control=VariantList(
                variants=[Variant(value=("control", "control")), Variant(value=("control", "nan"))]
            ),
        ),
        # Assessing expt 1: "v1", if expt 2 is launched ("test" variant).
        # Impact of (v1 | expt2:test launched)
        ComparisonPair(
            treatment=VariantList(
                variants=[Variant(value=("v1", "test")), Variant(value=("v1", "nan"))]
            ),
            control=VariantList(
                variants=[Variant(value=("control", "test")), Variant(value=("control", "nan"))]
            ),
        ),
    ]

    mea = MEA(
        df=joined_expt_metric_df, analysis_info=analysis_info, comparison_pairs=comparison_pairs
    )
    mea.run()

    write_path = f"{SAVE_PATH}/mea/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
        markdown_file_name="mea_test.md",
    )

    assert mea_report is not None


def test_mea_launches(simulated_data0, example_analysis_info):
    """Tests `mea` function with launches."""
    joined_expt_metric_df = simulated_data0.copy()
    analysis_info = example_analysis_info

    launches_values = [
        ("v1", "control"),
        ("v1", "test"),
        ("v2", "control"),
        ("v2", "test"),
        ("control", "test"),
    ]
    launches = [Launch(value=value) for value in launches_values]

    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=launches)
    mea.run()

    write_path = write_path = f"{SAVE_PATH}/mea_launches/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test_launches.html",
    )

    assert mea_report is not None


def test_mea_with_sim1(example_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_multi1()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_analysis_info

    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=None)
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim1/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
        markdown_file_name="mea_test.md",
    )

    assert mea_report is not None


def test_mea_with_sim2(example_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_multi2()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_analysis_info

    launches_values = [
        ("v1", "control"),
        ("v1", "enabled"),
        ("v2", "control"),
        ("v2", "enabled"),
        ("control", "enabled"),
    ]
    launches = [Launch(value=value) for value in launches_values]

    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=launches)
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim2/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim3(example_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_multi3()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_analysis_info

    launches_values = [
        ("v1", "control"),
        ("v1", "enabled"),
        ("v2", "control"),
        ("v2", "enabled"),
        ("control", "enabled"),
    ]
    launches = [Launch(value=value) for value in launches_values]

    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=launches)
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim3/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim4(example_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_multi4()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_analysis_info

    launches_values = [
        ("v1", "control"),
        ("v1", "enabled"),
        ("v2", "control"),
        ("v2", "enabled"),
        ("control", "enabled"),
    ]
    launches = [Launch(value=value) for value in launches_values]

    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=launches)
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim4/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim4_launches_inferred(example_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_multi4()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_analysis_info

    # No need to specify launches
    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=None)
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim4_launches_inferred/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim4_scenario_based_default(example_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_multi4()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_analysis_info

    # Specify the `control_launch`
    # In this case we use the default, so results should remain the same
    control_launch = Launch(value=("control", "control"))
    mea = MEA(
        df=joined_expt_metric_df,
        analysis_info=analysis_info,
        launches=None,
        control_launch=control_launch,
    )

    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim4_scenario_based_default/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
        end_user_report=False,
    )

    assert mea_report is not None


def test_mea_with_sim4_uni_version_expt1(example_uni_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_multi4()
    df = sim.expt_metric_df
    cols = ["memberid", "variant_1", "metric1", "metric2"]
    df = df[cols]
    df["variant"] = df["variant_1"].map(lambda x: (x,))

    analysis_info = example_uni_analysis_info

    mea = MEA(df=df, analysis_info=analysis_info, launches=None, control_launch=None)

    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim4_uni_version_expt1/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
        end_user_report=False,
    )

    assert mea_report is not None


def test_mea_with_sim4_scenario_based_non_default(example_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_multi4()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_analysis_info

    # Specify the `control_launch`
    # In this case we use a value different from default
    control_launch = Launch(value=("v1", "enabled"))
    mea = MEA(
        df=joined_expt_metric_df,
        analysis_info=analysis_info,
        launches=None,
        control_launch=control_launch,
    )

    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim4_scenario_based_non_default/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
        end_user_report=False,
    )

    assert mea_report is not None


def test_mea_with_sim5(example_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_multi5()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_analysis_info

    launches_values = [
        ("v1", "control"),
        ("v1", "enabled"),
        ("control", "enabled"),
    ]
    launches = [Launch(value=value) for value in launches_values]

    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=launches)
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim5/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim6(example_uni_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_uni1()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_uni_analysis_info

    launches_values = [("v1",), ("v2",)]
    launches = [Launch(value=value) for value in launches_values]

    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=launches)
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim_uni1/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim7(example_uni_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_uni2()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_uni_analysis_info

    launches_values = [
        ("v1",),
        ("v2",),
    ]
    launches = [Launch(value=value) for value in launches_values]

    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=launches)
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim_uni2/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim8(example_uni_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_uni3()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_uni_analysis_info

    launches_values = [("enabled",)]
    launches = [Launch(value=value) for value in launches_values]

    mea = MEA(df=joined_expt_metric_df, analysis_info=analysis_info, launches=launches)
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim_uni3/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim9(example_three_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_three1()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_three_analysis_info

    launches_values = [
        ("v1", "control", "control"),
        ("v1", "enabled", "control"),
        ("control", "enabled", "control"),
        ("v1", "control", "enabled"),
        ("v1", "enabled", "enabled"),
        ("control", "enabled", "enabled"),
    ]
    launches = [Launch(value=value) for value in launches_values]

    mea = MEA(
        df=joined_expt_metric_df,
        analysis_info=analysis_info,
        launches=launches,
        recalculate_expt_stats=True,
    )
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim_three1/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim10(example_three_analysis_info):
    """A test using simulated data from sim module."""
    sim = simulate_data_three2()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_three_analysis_info

    mea = MEA(
        df=joined_expt_metric_df,
        analysis_info=analysis_info,
        launches=None,
        recalculate_expt_stats=True,
    )
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim_three2/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None


def test_mea_with_sim11(example_analysis_info_metric_info_list):
    """A test using simulated data from sim module."""
    sim = simulate_data_three2()
    joined_expt_metric_df = sim.expt_metric_df
    analysis_info = example_analysis_info_metric_info_list

    mea = MEA(
        df=joined_expt_metric_df,
        analysis_info=analysis_info,
        launches=None,
        recalculate_expt_stats=True,
    )
    mea.run()

    write_path = f"{SAVE_PATH}/mea_sim_three2/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_test.html",
    )

    assert mea_report is not None

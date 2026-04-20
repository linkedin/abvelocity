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
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.get_data.join_expt_dfs import join_expt_dfs
from abvelocity.core.get_data.join_expt_with_metric_df import join_expt_with_metric_df
from abvelocity.core.mea.mea import MEA
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.constants import TRIGGER_TIME_COL, VARIANT_COL
from abvelocity.core.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.sim.sim import EXPT_UNIT_COL

# Create save path for html files (figures)
SAVE_PATH = str(Path(__file__).parents[4].joinpath("docs/static/test-results/mea_edge_cases/").resolve())
os.makedirs(SAVE_PATH, exist_ok=True)


METHOD = "simple"
"""Constant to denote which MEA method we are testing across this module."""


def simulated_data0(total_unit_num=5000):
    # This is a simple data simulation done on the spot within this module.
    # More sophisticated simulations are imported in the above from the `sim` module.
    sampler = np.random.default_rng(1317)

    expt1_unit_num = int(total_unit_num * 0.2)
    expt2_unit_num = int(total_unit_num * 0.2)
    metric_unit_num = int(total_unit_num * 0.95)

    expt_df1 = pd.DataFrame(
        {
            EXPT_UNIT_COL: sampler.choice(range(total_unit_num), size=expt1_unit_num),
            VARIANT_COL: sampler.choice(["v1", "v2", "v3", "v4", "control"], size=expt1_unit_num),
            TRIGGER_TIME_COL: sampler.choice(range(total_unit_num), size=expt1_unit_num),
        }
    )

    expt_df2 = pd.DataFrame(
        {
            EXPT_UNIT_COL: sampler.choice(range(total_unit_num), size=expt2_unit_num),
            VARIANT_COL: sampler.choice(["test", "control"], size=expt2_unit_num),
            TRIGGER_TIME_COL: sampler.choice(range(total_unit_num), size=expt2_unit_num),
        }
    )

    # We assume we do find metric data for most of the units.
    metric_df = pd.DataFrame(
        {
            EXPT_UNIT_COL: sampler.choice(range(total_unit_num), size=metric_unit_num),
            "metric1": sampler.normal(loc=0.0, scale=1.0, size=metric_unit_num),
            "metric2": sampler.normal(loc=0.0, scale=1.0, size=metric_unit_num),
        }
    )

    expt_dc_list = [
        DataContainer(pandas_df=expt_df1.copy(), is_pandas_df=True),
        DataContainer(pandas_df=expt_df2.copy(), is_pandas_df=True),
    ]

    metric_dc = DataContainer(pandas_df=metric_df, is_pandas_df=True)
    expt_dc = join_expt_dfs(dc_list=expt_dc_list, on_cols=[EXPT_UNIT_COL])

    joined_expt_metric_dc = join_expt_with_metric_df(
        expt_dc=expt_dc,
        metric_dc=metric_dc,
        expt_unit_col=EXPT_UNIT_COL,
        metric_join_unit_col=EXPT_UNIT_COL,
    )
    joined_expt_metric_df = joined_expt_metric_dc.pandas_df

    # Now we impose some signals on the data
    joined_expt_metric_df.loc[joined_expt_metric_df["variant_1"] == "control", "metric1"] += 2.0
    joined_expt_metric_df.loc[joined_expt_metric_df["variant_1"] == "v1", "metric1"] += 3.0
    joined_expt_metric_df.loc[joined_expt_metric_df["variant_1"] == "v2", "metric1"] += 3.0

    joined_expt_metric_df.loc[joined_expt_metric_df["variant_2"] == "control", "metric1"] += 2.0
    joined_expt_metric_df.loc[joined_expt_metric_df["variant_2"] == "test", "metric1"] += 3.0

    # Define a boolen index based on interaction between expt 1 and expt 2.
    # We impose a synergic effect between v1 in expt 1 and test in experiment 2.
    ind = (joined_expt_metric_df["variant_1"] == "v1") & (joined_expt_metric_df["variant_2"] == "test")

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
        multi_expt_info=MultiExptInfo(expt_info_list=[expt1, expt2], merge_method="cross", expt_unit_col=EXPT_UNIT_COL),
        metric_info_list=[MetricInfo(metrics=metrics)],
    )

    return analysis_info


def test_mea_default(example_analysis_info):
    """Tests `mea` function with launches."""
    joined_expt_metric_df = simulated_data0(200)
    analysis_info = example_analysis_info

    dc = DataContainer(pandas_df=joined_expt_metric_df)
    mea = MEA(dc, analysis_info=analysis_info, launches=None)
    mea.run()

    write_path = write_path = f"{SAVE_PATH}/mea_default/"
    mea_report = mea.publish(
        write_path=write_path,
        add_timestamp_to_path=False,
        rounding_digits=2,
        html_file_name="mea_report.html",
        end_user_report=False,
    )

    assert mea_report is not None

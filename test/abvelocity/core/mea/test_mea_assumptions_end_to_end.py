# BSD 2-CLAUSE LICENSE
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

"""
End-to-end tests for MEA Arm-Trigger Invariance assumption check.

Tests the full pipeline: simulated data → MEA run → published HTML report
with the assumption check section included. Two scenarios per dimensionality:

  1. Clean data — arm assignments independent, assumption PASSES.
  2. Contaminated data — Expt 0's treated arm has a lower trigger rate in
     Expt 1 (rows removed from the sim output to simulate the forbidden edge
     A_0 → S_1). Assumption FLAGS on source 0.

Contamination is injected by taking the sim's final expt_metric_df and
removing a fraction of Expt-1-triggered rows for members assigned to a
specific Expt-0 arm. This is the simplest way to bias the trigger rate
without changing the sim code.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.mea.mea import MEA
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.constants import CATEG_NAN_VALUE, VARIANT_COL
from abvelocity.core.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.sim.examples import simulate_data_multi1, simulate_data_three2
from abvelocity.core.sim.sim import EXPT_UNIT_COL

SAVE_PATH = str(
    Path(__file__).parents[4].joinpath("docs/static/test-results/mea_assumption_check/").resolve()
)
os.makedirs(SAVE_PATH, exist_ok=True)

METHOD = "simple"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_analysis_info(n_expts):
    expt_info_list = [
        ExptInfo(
            name=f"Expt {i}",
            test_key=None, experiment_id=None, segment_id=None,
            start_date=None, end_date=None, variants=None, control_label=None,
        )
        for i in range(n_expts)
    ]
    metrics = [Metric(numerator=UMetric(col="metric1")), Metric(numerator=UMetric(col="metric2"))]
    return AnalysisInfo(
        multi_expt_info=MultiExptInfo(
            expt_info_list=expt_info_list, merge_method="cross", expt_unit_col=EXPT_UNIT_COL
        ),
        metric_info_list=[MetricInfo(metrics=metrics)],
    )


def _inject_trigger_contamination(
    df: pd.DataFrame,
    source_col: str,
    source_arm_value: str,
    target_col: str,
    remove_fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Remove a fraction of triggered-target rows for a specific source arm.

    Simulates the forbidden edge A_source → S_target: members assigned to
    `source_arm_value` in experiment `source_col` are less likely to be
    triggered in `target_col`. Implements this by dropping the trigger for
    `remove_fraction` of those rows, setting target_col to CATEG_NAN_VALUE.

    Rebuilds the `variant` tuple column to stay consistent.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Rows where source arm matches AND target is currently triggered
    mask = (df[source_col] == source_arm_value) & (df[target_col] != CATEG_NAN_VALUE)
    candidate_idx = df[mask].index
    n_remove = int(len(candidate_idx) * remove_fraction)
    remove_idx = rng.choice(candidate_idx, size=n_remove, replace=False)

    df.loc[remove_idx, target_col] = CATEG_NAN_VALUE

    # Rebuild variant tuple from individual variant columns
    variant_cols = sorted([c for c in df.columns if c.startswith("variant_")])
    df[VARIANT_COL] = list(zip(*[df[c] for c in variant_cols]))

    return df


# ---------------------------------------------------------------------------
# K=2: clean — assumption passes
# ---------------------------------------------------------------------------


def test_clean_k2_assumption_passes():
    """K=2 with independent arm assignment: assumption check should pass."""
    sim = simulate_data_multi1()
    df = sim.expt_metric_df
    analysis_info = _make_analysis_info(n_expts=2)

    mea = MEA(dc=DataContainer(pandas_df=df), analysis_info=analysis_info, method=METHOD)
    mea.run()

    report = mea.publish(
        write_path=f"{SAVE_PATH}/k2_clean/",
        add_timestamp_to_path=False,
        html_file_name="mea_report.html",
        end_user_report=False,
    )

    assert report is not None
    assert "MEA Arm-Trigger Invariance Check" in report["html_str"]
    assert "PASS ✓" in report["html_str"]


# ---------------------------------------------------------------------------
# K=2: contaminated — Expt 0 treat arm depresses Expt 1 trigger rate
# ---------------------------------------------------------------------------


def test_contaminated_k2_assumption_flags():
    """K=2: Expt 0 v1 arm has lower Expt 1 trigger rate (30% removed).

    Injects A_0 → S_1: removes 30% of Expt-1-triggered rows for members
    assigned to v1 in Expt 0. The assumption check should flag source 0.
    """
    sim = simulate_data_multi1()
    df = _inject_trigger_contamination(
        df=sim.expt_metric_df,
        source_col="variant_1",
        source_arm_value="v1",
        target_col="variant_2",
        remove_fraction=0.30,
        seed=7,
    )
    analysis_info = _make_analysis_info(n_expts=2)

    mea = MEA(dc=DataContainer(pandas_df=df), analysis_info=analysis_info, method=METHOD)
    mea.run()

    report = mea.publish(
        write_path=f"{SAVE_PATH}/k2_contaminated/",
        add_timestamp_to_path=False,
        html_file_name="mea_report.html",
        end_user_report=False,
    )

    assert report is not None
    assert "MEA Arm-Trigger Invariance Check" in report["html_str"]
    assert "FLAG ✗" in report["html_str"]


# ---------------------------------------------------------------------------
# K=3: contaminated — Expt 0 v1 arm depresses Expt 1 trigger rate
# ---------------------------------------------------------------------------


def test_contaminated_k3_assumption_flags():
    """K=3: Expt 0 v1 arm has lower Expt 1 trigger rate (30% removed).

    Injects A_0 → S_1. Source 0 should flag; sources 1 and 2 should pass.
    """
    sim = simulate_data_three2()
    df = _inject_trigger_contamination(
        df=sim.expt_metric_df,
        source_col="variant_1",
        source_arm_value="v1",
        target_col="variant_2",
        remove_fraction=0.30,
        seed=7,
    )
    analysis_info = _make_analysis_info(n_expts=3)

    mea = MEA(
        dc=DataContainer(pandas_df=df),
        analysis_info=analysis_info,
        recalculate_expt_stats=True,
        method=METHOD,
    )
    mea.run()

    report = mea.publish(
        write_path=f"{SAVE_PATH}/k3_contaminated/",
        add_timestamp_to_path=False,
        html_file_name="mea_report.html",
        end_user_report=False,
    )

    assert report is not None
    assert "MEA Arm-Trigger Invariance Check" in report["html_str"]
    assert "FLAG ✗" in report["html_str"]

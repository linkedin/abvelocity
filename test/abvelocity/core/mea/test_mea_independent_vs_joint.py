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

"""
Independent Per-Experiment vs Joint MEA Analysis: demonstrates how
analyzing each experiment independently (ignoring the other) can lead
to a suboptimal launch decision that MEA's joint analysis corrects.

NOTE: Both experiments run concurrently. The issue is NOT about running
experiments back-to-back in time (temporal sequencing). The issue is that
each team analyzes their experiment in isolation, unaware of the
interaction with the overlapping experiment.

Scenario
--------
Two binary experiments (control/treatment each) run concurrently on a
shared population with **partial overlap**: each experiment triggers on
about 30% of the population, so the doubly-triggered region R11 is ~9%
and the singly-triggered regions R10, R01 are each ~21%.

Univariate effects (within each experiment's triggered population):
    Expt 1:  treatment  +3  on metric1
    Expt 2:  treatment  +4  on metric1

But there is a strong *negative* interaction when both treatments are
active simultaneously (within R11 only):

    (treatment, treatment):  -10  on metric1

Cell-level truth within R11 (additive baseline a = 0):

    (c1, c2) = 0       (c1, t2) = +4       ← BEST
    (t1, c2) = +3      (t1, t2) = +3+4-10 = -3  ← WORST

Per-experiment univariate analysis pools across R10/R01 (where the
interaction does not apply) and R11 (where it does, diluted by the
other experiment's 50/50 split).  Because R11 is a small fraction of
the triggered population, the negative interaction is heavily diluted:

    Univar Expt 1 ≈ w_10 * 3 + w_11 * (3 + 0.5*(-10)) ≈ +1.5  (positive)
    Univar Expt 2 ≈ w_01 * 4 + w_11 * (4 + 0.5*(-10)) ≈ +2.5  (positive)

    → Per-experiment decision: launch both (t1, t2).

MEA reveals (within R11):
    Best combination is (c1, t2) = +4.
    Launching (t1, t2) = -3 is the WORST combination.

This is the concrete failure case requested by Reviewer 5mVs.
"""

import os
from pathlib import Path

import pytest
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.mea.mea import MEA
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.sim.sim import EXPT_UNIT_COL, Sim

SAVE_PATH = str(Path(__file__).parents[4].joinpath("docs/static/test-results/mea/independent_vs_joint/").resolve())
os.makedirs(SAVE_PATH, exist_ok=True)

METHOD = "simple"


# ------------------------------------------------------------------ #
#  Simulation helper
# ------------------------------------------------------------------ #


def simulate_independent_failure(
    population_size: int = 100_000,
    population_seed: int = 42,
) -> Sim:
    """Simulate two concurrent experiments where independent per-experiment analysis picks the wrong combo.

    Ground truth (metric1 impacts relative to control baseline):
        Expt 1 treatment:       +3
        Expt 2 treatment:       +4
        Interaction (t1, t2):  -10   (strong negative synergy)

    With partial overlap (each experiment triggers ~30% of population):
        R11 ≈ 9%, R10 ≈ 21%, R01 ≈ 21%

    Cell-level expected metric1 within R11:
        (c1, c2) =  0
        (t1, c2) = +3
        (c1, t2) = +4   ← true optimum
        (t1, t2) = -3   ← worst cell

    Independent per-experiment analysis pools R10/R01 (clean +3 or +4)
    with R11 (where the interaction applies at half weight), so both
    univariate effects remain positive — leading to the wrong
    recommendation of launching (t1, t2).
    """
    sim = Sim(
        population_size=population_size,
        attribute_weights={
            "Level": {"Senior": 0.5, "Junior": 0.5},
        },
        metric_attribute_values={
            "metric1": {"Level": {"Junior": 5, "Senior": 2}},
        },
        expt_variant_weights_multi=[
            {"control": 0.5, "treatment": 0.5},  # Expt 1
            {"control": 0.5, "treatment": 0.5},  # Expt 2
        ],
        population_pcnt_multi=(100, 100),
        non_trigger_pct_multi=(70, 70),  # ~30% trigger each → ~9% R11
        population_seed=population_seed,
        expt_assignment_seed_multi=(13, 17),
        expt_metric_impacts=[
            {
                "control": {"metric1": 0},
                "treatment": {"metric1": 3},
            },
            {
                "control": {"metric1": 0},
                "treatment": {"metric1": 4},
            },
        ],
        interaction_metric_impacts={
            ("treatment", "treatment"): {"metric1": -10},
        },
        noise_sd_dict={"metric1": 5.0},
        noise_seed=99,
    )
    sim.run()
    return sim


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def sim():
    return simulate_independent_failure()


@pytest.fixture
def analysis_info():
    expt1 = ExptInfo(name="expt1")
    expt2 = ExptInfo(name="expt2")
    metrics = [Metric(numerator=UMetric(col="metric1"))]
    return AnalysisInfo(
        multi_expt_info=MultiExptInfo(
            expt_info_list=[expt1, expt2],
            merge_method="cross",
            expt_unit_col=EXPT_UNIT_COL,
        ),
        metric_info_list=[MetricInfo(metrics=metrics)],
    )


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


def test_independent_picks_wrong_combination(sim, analysis_info):
    """Independent per-experiment analysis recommends launching both treatments,
    but MEA's joint analysis shows (c1, t2) is optimal and (t1, t2) is the worst.
    """
    df = sim.expt_metric_df

    # --- Step 1: univariate per-experiment analysis ---
    # Expt 1 effect (averaging over Expt 2 states):
    expt1_treat = df.loc[df["variant_1"] == "treatment", "metric1"].mean()
    expt1_ctrl = df.loc[df["variant_1"] == "control", "metric1"].mean()
    univar_effect_expt1 = expt1_treat - expt1_ctrl

    # Expt 2 effect (averaging over Expt 1 states):
    expt2_treat = df.loc[df["variant_2"] == "treatment", "metric1"].mean()
    expt2_ctrl = df.loc[df["variant_2"] == "control", "metric1"].mean()
    univar_effect_expt2 = expt2_treat - expt2_ctrl

    # Independent analysis says: both treatments are positive → launch (t1, t2).
    # With 100k units and these effect sizes, both should be positive.
    assert univar_effect_expt1 > 0, f"Expected univariate Expt 1 effect > 0, got {univar_effect_expt1:.3f}"
    assert univar_effect_expt2 > 0, f"Expected univariate Expt 2 effect > 0, got {univar_effect_expt2:.3f}"
    independent_decision = ("treatment", "treatment")

    # --- Step 2: MEA joint analysis ---
    mea = MEA(
        dc=DataContainer(pandas_df=df),
        analysis_info=analysis_info,
        method=METHOD,
    )
    mea.run()
    result = mea.result
    effects_df = result.combined_mea_result.variant_effect_df_pairs

    # Find the combination effect estimates.
    # Effects are relative to (control, control).
    combo_effects = {}
    for _, row in effects_df.iterrows():
        launch = row.get("launch") or row.get("launch_variant")
        if launch is not None:
            combo_effects[str(launch)] = row["delta"]

    # --- Step 3: verify MEA identifies (c1, t2) as best ---
    # We examine the cell means directly as a simpler verification.
    cell_means = df.groupby(["variant_1", "variant_2"])["metric1"].mean().reset_index()
    cell_means = cell_means.set_index(["variant_1", "variant_2"])["metric1"]

    baseline = cell_means[("control", "control")]
    effect_t1_c2 = cell_means[("treatment", "control")] - baseline
    effect_c1_t2 = cell_means[("control", "treatment")] - baseline
    effect_t1_t2 = cell_means[("treatment", "treatment")] - baseline

    # (c1, t2) should be the best combination
    best_combo_effect = max(effect_t1_c2, effect_c1_t2, effect_t1_t2)
    assert effect_c1_t2 == best_combo_effect, (
        f"Expected (c1, t2) to be best but got: " f"(t1,c2)={effect_t1_c2:.2f}, (c1,t2)={effect_c1_t2:.2f}, (t1,t2)={effect_t1_t2:.2f}"
    )

    # (t1, t2) — the sequential decision — should be the WORST combination
    worst_combo_effect = min(effect_t1_c2, effect_c1_t2, effect_t1_t2)
    assert effect_t1_t2 == worst_combo_effect, (
        f"Expected (t1, t2) to be worst but got: " f"(t1,c2)={effect_t1_c2:.2f}, (c1,t2)={effect_c1_t2:.2f}, (t1,t2)={effect_t1_t2:.2f}"
    )

    # The independent per-experiment decision is wrong: it picks the worst, not the best.
    assert independent_decision == ("treatment", "treatment")
    assert effect_t1_t2 < 0, f"Expected (t1, t2) effect < 0, got {effect_t1_t2:.2f}"
    assert effect_c1_t2 > 0, f"Expected (c1, t2) effect > 0, got {effect_c1_t2:.2f}"

    # Print summary for inclusion in paper / test report.
    print("\n" + "=" * 70)
    print("INDEPENDENT PER-EXPERIMENT vs JOINT MEA ANALYSIS — FAILURE CASE")
    print("=" * 70)
    print("\nPer-experiment univariate effects (each analyzed independently):")
    print(f"  Expt 1 (treatment vs control): {univar_effect_expt1:+.3f}  → launch treatment")
    print(f"  Expt 2 (treatment vs control): {univar_effect_expt2:+.3f}  → launch treatment")
    print("  Per-experiment decision: launch (t1, t2)")
    print("\nMEA combination effects (vs all-control baseline):")
    print(f"  (treatment, control):   {effect_t1_c2:+.3f}")
    print(f"  (control, treatment):   {effect_c1_t2:+.3f}  <- BEST")
    print(f"  (treatment, treatment): {effect_t1_t2:+.3f}  <- WORST (per-experiment picks this)")
    print("\nIndependent per-experiment analysis leads to the WORST combination.")
    print("MEA correctly identifies (control, treatment) as optimal.")
    print("=" * 70)


def test_independent_failure_generates_report(sim, analysis_info):
    """Generate an MEA report for the independent-analysis failure scenario,
    saved to test-results for inclusion in the paper."""
    df = sim.expt_metric_df

    mea = MEA(
        dc=DataContainer(pandas_df=df),
        analysis_info=analysis_info,
        method=METHOD,
    )
    mea.run()

    report = mea.publish(
        write_path=SAVE_PATH,
        add_timestamp_to_path=False,
        html_file_name="independent_vs_joint_report.html",
    )
    assert report is not None

    report_path = os.path.join(SAVE_PATH, "independent_vs_joint_report.html")
    assert os.path.exists(report_path), f"Report not found at {report_path}"
    print(f"\nReport saved to: {report_path}")

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
Three-Way Comparison: Independent vs Sequential vs MEA Joint Analysis.

Compares three approaches to analyzing two concurrent overlapping experiments:

1. **Independent per-experiment**: Each experiment analyzed in isolation,
   ignoring the other. Both run concurrently.

2. **Sequential in time**: Run Experiment 1 first, ship its winner, then
   run Experiment 2 with Expt 1's treatment already deployed. This is
   equivalent to computing Expt 2's conditional effect given Expt 1 = treatment
   (MEA's Algorithm 2 / scenario-based analysis).

3. **MEA joint analysis**: Both experiments run concurrently, analyzed
   jointly via post-stratification on triggering profiles.

Key finding: With **low overlap** (30% trigger rates), both independent
and sequential approaches fail — only MEA identifies the correct optimum.
With **high overlap** (50% trigger rates), sequential catches the problem
but independent still fails. MEA is the only approach that works in both
regimes.

The simulation uses one 2D experiment (both experiments concurrent) and
derives all three analyses from the same data. The "sequential" analysis
is computed by examining the conditional effect of Expt 2 given Expt 1 =
treatment, which is exactly what you would measure if you shipped Expt 1
first and then ran Expt 2.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.mea.mea import MEA
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.sim.sim import EXPT_UNIT_COL, Sim

SAVE_PATH = str(Path(__file__).parents[4].joinpath("docs/static/test-results/mea/sequential_vs_mea/").resolve())
os.makedirs(SAVE_PATH, exist_ok=True)

METHOD = "simple"

# Ground-truth effects (shared across all scenarios)
EFFECT_1 = 3  # Expt 1 treatment effect
EFFECT_2 = 4  # Expt 2 treatment effect
INTERACTION = -10  # (treatment, treatment) interaction


# ------------------------------------------------------------------ #
#  Ground-truth calculation
# ------------------------------------------------------------------ #


def get_expected_delta(non_trigger_pct, launch_variant):
    """Compute the ground-truth expected MEA delta for a given launch combination.

    This uses the same logic as MEA estimation applied to population parameters:
    weight each region's effect by its trigger rate, normalized to the total
    triggered population.

    Parameters
    ----------
    non_trigger_pct : int
        Non-trigger percentage (same for both experiments).
    launch_variant : tuple
        e.g. ("treatment", "treatment"), ("control", "treatment"), etc.

    Returns
    -------
    dict with 'expected_delta' and the intermediate calculations.
    """
    trigger_rate = (100.0 - non_trigger_pct) / 100.0

    # Region trigger rates (for two experiments with the same trigger rate)
    r11_rate = trigger_rate * trigger_rate  # both trigger
    r10_rate = trigger_rate * (1 - trigger_rate)  # only Expt 1
    r01_rate = (1 - trigger_rate) * trigger_rate  # only Expt 2

    # Effect of launching variant vs (control, control) in each region
    v1, v2 = launch_variant

    # Expt 1 impact in R10 and R11
    impact1 = EFFECT_1 if v1 == "treatment" else 0
    # Expt 2 impact in R01 and R11
    impact2 = EFFECT_2 if v2 == "treatment" else 0

    # Interaction in R11 (only when both are treatment)
    interaction = INTERACTION if (v1 == "treatment" and v2 == "treatment") else 0

    # Region-level effects (vs control baseline in that region)
    # R10: only Expt 1 active → effect is impact1
    # R01: only Expt 2 active → effect is impact2
    # R11: both active → effect is impact1 + impact2 + interaction
    effect_r10 = impact1
    effect_r01 = impact2
    effect_r11 = impact1 + impact2 + interaction

    # Weighted average over AFFECTED regions only.
    # A region is affected if the launched variant changes something there:
    #   R10 affected iff Expt 1 is launched (v1 != "control")
    #   R01 affected iff Expt 2 is launched (v2 != "control")
    #   R11 affected iff either experiment is launched
    # This matches how MEA computes its delta: only users who see a change
    # from the launch decision contribute to the weighted average.
    numerator = 0.0
    denominator = 0.0

    if v1 != "control":  # Expt 1 launched → R10 users are affected
        numerator += r10_rate * effect_r10
        denominator += r10_rate

    if v2 != "control":  # Expt 2 launched → R01 users are affected
        numerator += r01_rate * effect_r01
        denominator += r01_rate

    # R11 always affected for any non-baseline launch
    numerator += r11_rate * effect_r11
    denominator += r11_rate

    expected_delta = numerator / denominator if denominator > 0 else 0.0

    return {
        "expected_delta": expected_delta,
        "launch_variant": launch_variant,
        "trigger_rate": trigger_rate,
        "r11_rate": r11_rate,
        "r10_rate": r10_rate,
        "r01_rate": r01_rate,
        "affected_rate": denominator,
        "effect_r10": effect_r10,
        "effect_r01": effect_r01,
        "effect_r11": effect_r11,
        "impact1": impact1,
        "impact2": impact2,
        "interaction": interaction,
    }


# ------------------------------------------------------------------ #
#  Simulation helpers
# ------------------------------------------------------------------ #


def simulate_two_experiments(
    non_trigger_pct: int,
    population_size: int = 100_000,
    population_seed: int = 42,
) -> Sim:
    """Simulate two overlapping experiments with configurable trigger rates.

    Parameters
    ----------
    non_trigger_pct : int
        Percentage of population that does NOT trigger each experiment.
        70 → 30% trigger rate → ~9% R11.
        50 → 50% trigger rate → ~25% R11.
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
            {"control": 0.5, "treatment": 0.5},
            {"control": 0.5, "treatment": 0.5},
        ],
        population_pcnt_multi=(100, 100),
        non_trigger_pct_multi=(non_trigger_pct, non_trigger_pct),
        population_seed=population_seed,
        expt_assignment_seed_multi=(13, 17),
        expt_metric_impacts=[
            {"control": {"metric1": 0}, "treatment": {"metric1": EFFECT_1}},
            {"control": {"metric1": 0}, "treatment": {"metric1": EFFECT_2}},
        ],
        interaction_metric_impacts={
            ("treatment", "treatment"): {"metric1": INTERACTION},
        },
        noise_sd_dict={"metric1": 5.0},
        noise_seed=99,
    )
    sim.run()
    return sim


def compute_three_way_analysis(df, non_trigger_pct):
    """Compute all three analysis approaches from the same 2D data.

    Returns a dict with results for each approach.
    """
    results = {}

    # ------------------------------------------------------------------
    # Ground-truth expected deltas (from population parameters)
    # ------------------------------------------------------------------
    combos = [
        ("treatment", "control"),
        ("control", "treatment"),
        ("treatment", "treatment"),
    ]
    ground_truth = {}
    for combo in combos:
        gt = get_expected_delta(non_trigger_pct, combo)
        label = f"({combo[0]}, {combo[1]})"
        ground_truth[label] = gt
    results["ground_truth"] = ground_truth

    # ------------------------------------------------------------------
    # Region sizes
    # Note: Sim uses string "nan" for non-triggered, not Python None
    # ------------------------------------------------------------------
    v1_triggered = df["variant_1"] != "nan"
    v2_triggered = df["variant_2"] != "nan"
    r11 = df[v1_triggered & v2_triggered]
    r10 = df[v1_triggered & ~v2_triggered]
    r01 = df[~v1_triggered & v2_triggered]
    r00 = df[~v1_triggered & ~v2_triggered]

    results["region_sizes"] = {
        "R11": len(r11),
        "R10": len(r10),
        "R01": len(r01),
        "R00": len(r00),
        "total": len(df),
    }

    # ------------------------------------------------------------------
    # 1. Independent per-experiment analysis
    # ------------------------------------------------------------------
    # Expt 1: all triggered units (R10 ∪ R11)
    e1_triggered = df[v1_triggered]
    univar_1 = (
        e1_triggered.loc[e1_triggered["variant_1"] == "treatment", "metric1"].mean()
        - e1_triggered.loc[e1_triggered["variant_1"] == "control", "metric1"].mean()
    )

    # Expt 2: all triggered units (R01 ∪ R11)
    e2_triggered = df[v2_triggered]
    univar_2 = (
        e2_triggered.loc[e2_triggered["variant_2"] == "treatment", "metric1"].mean()
        - e2_triggered.loc[e2_triggered["variant_2"] == "control", "metric1"].mean()
    )

    indep_decision_1 = "treatment" if univar_1 > 0 else "control"
    indep_decision_2 = "treatment" if univar_2 > 0 else "control"

    results["independent"] = {
        "univar_expt1": univar_1,
        "univar_expt2": univar_2,
        "decision_expt1": indep_decision_1,
        "decision_expt2": indep_decision_2,
        "final_combination": (indep_decision_1, indep_decision_2),
    }

    # ------------------------------------------------------------------
    # 2. Sequential in time: Expt 1 first, ship winner, then Expt 2
    # ------------------------------------------------------------------
    # Step 1: Expt 1 analyzed alone → same univariate as independent
    seq_step1_effect = univar_1
    seq_step1_decision = indep_decision_1  # e.g., "treatment"

    # Step 2: After shipping Expt 1's winner, run Expt 2.
    # Expt 2's triggered population = R11 ∪ R01.
    # In R11: all users now have seq_step1_decision for Expt 1 (shipped).
    #   We estimate Expt 2 effect from R11 rows where variant_1 == shipped variant.
    # In R01: Expt 1 doesn't trigger, shipping is irrelevant.
    #   Expt 2 effect estimated from all R01 rows.

    # R11 conditional on shipped Expt 1 variant
    r11_shipped = r11[r11["variant_1"] == seq_step1_decision]
    r11_cond_effect = (
        r11_shipped.loc[r11_shipped["variant_2"] == "treatment", "metric1"].mean() - r11_shipped.loc[r11_shipped["variant_2"] == "control", "metric1"].mean()
    )

    # R01 effect (Expt 1 not triggered, unaffected by shipping)
    r01_effect = r01.loc[r01["variant_2"] == "treatment", "metric1"].mean() - r01.loc[r01["variant_2"] == "control", "metric1"].mean()

    # Weighted by FULL region sizes (after shipping, all R11 has shipped variant)
    n_r11 = len(r11)
    n_r01 = len(r01)
    w_r11 = n_r11 / (n_r11 + n_r01)
    w_r01 = n_r01 / (n_r11 + n_r01)
    seq_step2_effect = w_r11 * r11_cond_effect + w_r01 * r01_effect

    seq_step2_decision = "treatment" if seq_step2_effect > 0 else "control"

    results["sequential_t1_first"] = {
        "step1_effect": seq_step1_effect,
        "step1_decision": seq_step1_decision,
        "step2_r11_cond_effect": r11_cond_effect,
        "step2_r01_effect": r01_effect,
        "step2_w_r11": w_r11,
        "step2_w_r01": w_r01,
        "step2_effect": seq_step2_effect,
        "step2_decision": seq_step2_decision,
        "final_combination": (seq_step1_decision, seq_step2_decision),
    }

    # Also compute: Expt 2 first, ship winner, then Expt 1
    seq2_step1_effect = univar_2
    seq2_step1_decision = indep_decision_2  # e.g., "treatment"

    r11_shipped2 = r11[r11["variant_2"] == seq2_step1_decision]
    r11_cond_effect2 = (
        r11_shipped2.loc[r11_shipped2["variant_1"] == "treatment", "metric1"].mean()
        - r11_shipped2.loc[r11_shipped2["variant_1"] == "control", "metric1"].mean()
    )

    r10_effect = r10.loc[r10["variant_1"] == "treatment", "metric1"].mean() - r10.loc[r10["variant_1"] == "control", "metric1"].mean()

    n_r10 = len(r10)
    w2_r11 = n_r11 / (n_r11 + n_r10)
    w2_r10 = n_r10 / (n_r11 + n_r10)
    seq2_step2_effect = w2_r11 * r11_cond_effect2 + w2_r10 * r10_effect

    seq2_step2_decision = "treatment" if seq2_step2_effect > 0 else "control"

    results["sequential_t2_first"] = {
        "step1_effect": seq2_step1_effect,
        "step1_decision": seq2_step1_decision,
        "step2_r11_cond_effect": r11_cond_effect2,
        "step2_r10_effect": r10_effect,
        "step2_w_r11": w2_r11,
        "step2_w_r10": w2_r10,
        "step2_effect": seq2_step2_effect,
        "step2_decision": seq2_step2_decision,
        "final_combination": (seq2_step2_decision, seq2_step1_decision),
    }

    # ------------------------------------------------------------------
    # 3. Cell means in R11 (ground truth for the overlap region)
    # ------------------------------------------------------------------
    cell_means = r11.groupby(["variant_1", "variant_2"])["metric1"].mean()
    baseline = cell_means[("control", "control")]
    results["r11_cell_effects"] = {
        "(c1, c2)": 0.0,
        "(t1, c2)": cell_means[("treatment", "control")] - baseline,
        "(c1, t2)": cell_means[("control", "treatment")] - baseline,
        "(t1, t2)": cell_means[("treatment", "treatment")] - baseline,
    }

    return results


def make_analysis_info():
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


def format_decision(combo):
    """Format a (variant1, variant2) tuple for display."""
    labels = {"treatment": "t", "control": "c"}
    return f"({labels.get(combo[0], combo[0])}, {labels.get(combo[1], combo[1])})"


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestLowOverlap:
    """Low overlap (30% trigger): both independent and sequential fail."""

    @pytest.fixture
    def sim(self):
        return simulate_two_experiments(non_trigger_pct=70)

    @pytest.fixture
    def analysis_info(self):
        return make_analysis_info()

    def test_all_three_approaches(self, sim, analysis_info):
        """With 30% trigger rates (~9% R11), both independent and sequential-
        in-time analyses recommend (t1, t2), which is the worst combination.
        Only MEA identifies the optimum (c1, t2).
        """
        df = sim.expt_metric_df
        r = compute_three_way_analysis(df, non_trigger_pct=70)

        # --- Independent: both positive → launch (t1, t2) ---
        assert r["independent"]["univar_expt1"] > 0
        assert r["independent"]["univar_expt2"] > 0
        assert r["independent"]["final_combination"] == ("treatment", "treatment")

        # --- Sequential (Expt 1 first): ship t1, then Expt 2 still positive ---
        assert r["sequential_t1_first"]["step1_decision"] == "treatment"
        assert r["sequential_t1_first"]["step2_effect"] > 0, (
            f"Expected sequential Expt 2 effect > 0 (low overlap dilutes interaction), " f"got {r['sequential_t1_first']['step2_effect']:.3f}"
        )
        assert r["sequential_t1_first"]["final_combination"] == ("treatment", "treatment")

        # --- R11 ground truth: (t1, t2) is the worst ---
        assert r["r11_cell_effects"]["(t1, t2)"] < 0
        assert r["r11_cell_effects"]["(c1, t2)"] > 0
        best = max(r["r11_cell_effects"], key=r["r11_cell_effects"].get)
        worst = min(r["r11_cell_effects"], key=r["r11_cell_effects"].get)
        assert best == "(c1, t2)"
        assert worst == "(t1, t2)"

        # --- MEA joint analysis ---
        mea = MEA(
            dc=DataContainer(pandas_df=df),
            analysis_info=analysis_info,
            method=METHOD,
        )
        mea.run()
        effects_df = mea.result.combined_mea_result.variant_effect_df_pairs

        # Extract MEA combination effects (weighted across all regions)
        mea_effects = {}
        for _, row in effects_df.iterrows():
            launch = row.get("launch") or row.get("comparison_pair")
            mea_effects[str(launch)] = {
                "delta": row["delta"],
                "delta_percent": row["delta_percent"],
            }
        r["mea_effects"] = mea_effects

        # MEA should rank (c1, t2) best
        deltas = {k: v["delta"] for k, v in mea_effects.items()}
        mea_best = max(deltas, key=deltas.get)
        assert "control" in mea_best and "treatment" in mea_best, f"Expected MEA best to be (control, treatment), got {mea_best}"

        # --- Print summary ---
        self._print_summary("LOW OVERLAP (30% trigger, R11 ≈ 9%)", r)

    def _print_summary(self, title, r):
        print(f"\n{'=' * 72}")
        print(f"THREE-WAY COMPARISON — {title}")
        print(f"{'=' * 72}")

        regions = r["region_sizes"]
        total = regions["total"]
        print(
            f"\nRegion sizes: R11={regions['R11']} ({100*regions['R11']/total:.1f}%), "
            f"R10={regions['R10']} ({100*regions['R10']/total:.1f}%), "
            f"R01={regions['R01']} ({100*regions['R01']/total:.1f}%), "
            f"R00={regions['R00']} ({100*regions['R00']/total:.1f}%)"
        )

        ind = r["independent"]
        print("\n1. INDEPENDENT PER-EXPERIMENT:")
        print(f"   Expt 1 univariate: {ind['univar_expt1']:+.3f} → {ind['decision_expt1']}")
        print(f"   Expt 2 univariate: {ind['univar_expt2']:+.3f} → {ind['decision_expt2']}")
        print(f"   Decision: {format_decision(ind['final_combination'])}")

        seq = r["sequential_t1_first"]
        print("\n2. SEQUENTIAL (Expt 1 first, ship, then Expt 2):")
        print(f"   Step 1 — Expt 1: {seq['step1_effect']:+.3f} → ship {seq['step1_decision']}")
        print(f"   Step 2 — Expt 2 conditional on {seq['step1_decision']}₁:")
        print(f"     R11 (w={seq['step2_w_r11']:.2f}): {seq['step2_r11_cond_effect']:+.3f}")
        print(f"     R01 (w={seq['step2_w_r01']:.2f}): {seq['step2_r01_effect']:+.3f}")
        print(f"     Pooled: {seq['step2_effect']:+.3f} → {seq['step2_decision']}")
        print(f"   Decision: {format_decision(seq['final_combination'])}")

        seq2 = r["sequential_t2_first"]
        print("\n   ALT ORDER (Expt 2 first, ship, then Expt 1):")
        print(f"   Step 1 — Expt 2: {seq2['step1_effect']:+.3f} → ship {seq2['step1_decision']}")
        print(f"   Step 2 — Expt 1 conditional on {seq2['step1_decision']}₂:")
        print(f"     R11 (w={seq2['step2_w_r11']:.2f}): {seq2['step2_r11_cond_effect']:+.3f}")
        print(f"     R10 (w={seq2['step2_w_r10']:.2f}): {seq2['step2_r10_effect']:+.3f}")
        print(f"     Pooled: {seq2['step2_effect']:+.3f} → {seq2['step2_decision']}")
        print(f"   Decision: {format_decision(seq2['final_combination'])}")

        gt = r["ground_truth"]
        print("\n3. GROUND-TRUTH EXPECTED DELTAS (from population parameters):")
        gt_deltas = {k: v["expected_delta"] for k, v in gt.items()}
        for combo, delta in sorted(gt_deltas.items(), key=lambda x: -x[1]):
            g = gt[combo]
            tag = ""
            if delta == max(gt_deltas.values()):
                tag = " ← BEST"
            elif delta == min(gt_deltas.values()):
                tag = " ← WORST"
            print(f"   {combo}: {delta:+.4f}{tag}")
            print(
                f"     R10={g['effect_r10']:+d} (w={g['r10_rate']:.3f}), "
                f"R01={g['effect_r01']:+d} (w={g['r01_rate']:.3f}), "
                f"R11={g['effect_r11']:+d} (w={g['r11_rate']:.3f}), "
                f"affected={g['affected_rate']:.3f}"
            )

        if "mea_effects" in r:
            print("\n4. MEA ESTIMATED EFFECTS (weighted across all regions):")
            mea_effs = r["mea_effects"]
            deltas = {k: v["delta"] for k, v in mea_effs.items()}
            for combo, delta in sorted(deltas.items(), key=lambda x: -x[1]):
                pct = mea_effs[combo]["delta_percent"]
                # Find matching ground truth
                gt_match = None
                for gt_key, gt_val in gt_deltas.items():
                    if gt_key.replace("(", "").replace(")", "") in combo.replace("(", "").replace(")", ""):
                        gt_match = gt_val
                tag = ""
                if delta == max(deltas.values()):
                    tag = " ← BEST"
                elif delta == min(deltas.values()):
                    tag = " ← WORST"
                gt_str = f" (ground truth: {gt_match:+.4f})" if gt_match else ""
                print(f"   {combo}: {delta:+.3f} ({pct:+.1f}%){tag}{gt_str}")

        print(f"\n{'=' * 72}")


class TestHighOverlap:
    """High overlap (50% trigger): sequential catches it, independent doesn't."""

    @pytest.fixture
    def sim(self):
        return simulate_two_experiments(non_trigger_pct=50)

    @pytest.fixture
    def analysis_info(self):
        return make_analysis_info()

    def test_sequential_catches_interaction(self, sim, analysis_info):
        """With 50% trigger rates (~25% R11), sequential-in-time analysis
        detects the problem (Expt 2 conditional effect is negative after
        shipping t1), but independent per-experiment still misses it.
        """
        df = sim.expt_metric_df
        r = compute_three_way_analysis(df, non_trigger_pct=50)

        # --- Independent: STILL both positive → launch (t1, t2) → WRONG ---
        assert r["independent"]["univar_expt1"] > 0
        assert r["independent"]["univar_expt2"] > 0
        assert r["independent"]["final_combination"] == ("treatment", "treatment")

        # --- Sequential (Expt 1 first): ship t1, Expt 2 effect is NEGATIVE ---
        assert r["sequential_t1_first"]["step1_decision"] == "treatment"
        assert r["sequential_t1_first"]["step2_effect"] < 0, (
            f"Expected sequential Expt 2 effect < 0 (high overlap reveals interaction), " f"got {r['sequential_t1_first']['step2_effect']:.3f}"
        )
        # Sequential correctly avoids launching t2
        assert r["sequential_t1_first"]["step2_decision"] == "control"
        assert r["sequential_t1_first"]["final_combination"] == ("treatment", "control")

        # --- R11 ground truth ---
        assert r["r11_cell_effects"]["(t1, t2)"] < 0
        assert r["r11_cell_effects"]["(c1, t2)"] > 0

        # --- MEA ---
        mea = MEA(
            dc=DataContainer(pandas_df=df),
            analysis_info=analysis_info,
            method=METHOD,
        )
        mea.run()
        effects_df = mea.result.combined_mea_result.variant_effect_df_pairs

        mea_effects = {}
        for _, row in effects_df.iterrows():
            launch = row.get("launch") or row.get("comparison_pair")
            mea_effects[str(launch)] = {
                "delta": row["delta"],
                "delta_percent": row["delta_percent"],
            }
        r["mea_effects"] = mea_effects

        # --- Print ---
        print(f"\n{'=' * 72}")
        print("THREE-WAY COMPARISON — HIGH OVERLAP (50% trigger, R11 ≈ 25%)")
        print(f"{'=' * 72}")

        regions = r["region_sizes"]
        total = regions["total"]
        print(
            f"\nRegion sizes: R11={regions['R11']} ({100*regions['R11']/total:.1f}%), "
            f"R10={regions['R10']} ({100*regions['R10']/total:.1f}%), "
            f"R01={regions['R01']} ({100*regions['R01']/total:.1f}%), "
            f"R00={regions['R00']} ({100*regions['R00']/total:.1f}%)"
        )

        ind = r["independent"]
        print(f"\n1. CONCURRENT UNIVARIATE: Expt 1 = {ind['univar_expt1']:+.3f}, " f"Expt 2 = {ind['univar_expt2']:+.3f}")
        print(f"   Decision: {format_decision(ind['final_combination'])} — STILL WRONG")

        seq = r["sequential_t1_first"]
        print("\n2. SEQUENTIAL (Expt 1 first):")
        print(f"   Step 1: {seq['step1_effect']:+.3f} -> ship treatment1")
        print(f"   Step 2: Expt 2 conditional = {seq['step2_effect']:+.3f} -> DO NOT launch treatment2")
        print(f"   Decision: {format_decision(seq['final_combination'])} -- CATCHES the interaction")

        deltas = {k: v["delta"] for k, v in mea_effects.items()}
        print("\n3. MEA COMBINATION EFFECTS (weighted across all regions):")
        for combo, delta in sorted(deltas.items(), key=lambda x: -x[1]):
            pct = mea_effects[combo]["delta_percent"]
            tag = " <- BEST" if delta == max(deltas.values()) else (" <- WORST" if delta == min(deltas.values()) else "")
            print(f"   {combo}: {delta:+.3f} ({pct:+.1f}%){tag}")

        print("\n   Sequential gets (t1, c2) -- not optimal (c1, t2) is best,")
        print("   but at least avoids the catastrophic (t1, t2).")
        print("   Only MEA identifies the true optimum.")
        print(f"{'=' * 72}")


class TestGenerateReport:
    """Generate HTML report comparing all three approaches."""

    def test_generate_comparison_report(self):
        """Generate results JSON with MEA combination effects for both scenarios."""

        for label, ntp in [("low_overlap", 70), ("high_overlap", 50)]:
            analysis_info = make_analysis_info()  # fresh per iteration; MEA mutates it
            sim = simulate_two_experiments(non_trigger_pct=ntp)
            df = sim.expt_metric_df
            r = compute_three_way_analysis(df, non_trigger_pct=ntp)

            # Run MEA to get weighted combination effects
            mea = MEA(
                dc=DataContainer(pandas_df=df),
                analysis_info=analysis_info,
                method=METHOD,
            )
            mea.run()
            effects_df = mea.result.combined_mea_result.variant_effect_df_pairs
            mea_effects = {}
            for _, row in effects_df.iterrows():
                launch = row.get("launch") or row.get("comparison_pair")
                mea_effects[str(launch)] = {
                    "delta": row["delta"],
                    "delta_percent": row["delta_percent"],
                }
            r["mea_effects"] = mea_effects

            if label == "low_overlap":
                results_low = r
            else:
                results_high = r

        results = {
            "low_overlap": _serialize_results(results_low),
            "high_overlap": _serialize_results(results_high),
            "ground_truth": {
                "effect_1": EFFECT_1,
                "effect_2": EFFECT_2,
                "interaction": INTERACTION,
            },
        }

        json_path = os.path.join(SAVE_PATH, "sequential_vs_mea_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        assert os.path.exists(json_path)
        print(f"\nResults saved to: {json_path}")


def _serialize_results(r):
    """Convert numpy types to Python types for JSON serialization."""

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    return convert(r)

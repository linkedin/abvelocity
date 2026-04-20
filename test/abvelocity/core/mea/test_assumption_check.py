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
Tests for mea/assumption_check.py.

Includes a self-contained simulation of MEA-style data so we can verify the
test genuinely fires when the MEA Arm-Trigger Invariance assumption is violated
and stays quiet when it holds. Two scenarios are simulated at member level:

  1. Independent assignment (H0 holds):
     - Each member is independently bucketed into Expt A and Expt B using
       separate hash IDs (mirroring how ABPlatform works). Triggering for each
       experiment is also independent. Expected: assumption PASSES.

  2. Correlated assignment (H0 violated):
     - Members assigned to Expt A treatment are *more likely* to be assigned
       to Expt B treatment (simulates within-arm imbalance or a routing bug).
       Expected: assumption FLAGS.

These tests do not use the full abvelocity Sim/MEA stack -- they build
variant_count_df directly from a simulated member-level DataFrame, keeping
the test self-contained and fast.
"""

import numpy as np
import pandas as pd
import pytest

from abvelocity.core.mea.assumption_check import (
    ArmTriggerInvarianceResult,
    PairTriggerInvarianceResult,
    SourceTriggerInvarianceResult,
    _build_variant_count_array,
    _build_source_joint_table,
    check_trigger_invariance,
)
from abvelocity.core.mea.metric_interaction import (
    MetricInteractionResult,
    MetricTriggerStateTestResult,
    _build_cell_arrays_for_trigger_state,
    check_metric_interaction,
)
from abvelocity.core.param.constants import (
    CATEG_NAN_VALUE,
    MEAN_COL,
    SAMPLE_COUNT_COL,
    SD_COL,
    SUM_COL,
    SUM_SQ_COL,
    VARIANT_COL,
    VARIANT_COUNT_COL,
)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def simulate_mea_members(
    n_members: int = 50000,
    trigger_rate_a: float = 0.6,
    trigger_rate_b: float = 0.5,
    treat_rate_a: float = 0.5,
    treat_rate_b: float = 0.5,
    assignment_correlation: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a member-level MEA assignment DataFrame.

    Each member gets:
      - A binary trigger for Expt A (based on trigger_rate_a)
      - A binary trigger for Expt B (based on trigger_rate_b)
      - A variant for Expt A: "ctrl" or "treat" (with treat_rate_a among triggered)
      - A variant for Expt B: "ctrl" or "treat" (with treat_rate_b among triggered)

    When assignment_correlation == 0 (default), assignment in Expt A is
    independent of Expt B -- the invariance assumption holds.

    When assignment_correlation > 0, members assigned to Expt A treatment are
    more likely to be assigned to Expt B treatment, violating within-arm balance
    (the non-nan columns of the contingency table become non-uniform).

    Args:
        n_members: Number of members to simulate.
        trigger_rate_a: Fraction of members who trigger Expt A.
        trigger_rate_b: Fraction of members who trigger Expt B.
        treat_rate_a: Fraction of Expt-A-triggered members assigned to treatment.
        treat_rate_b: Baseline fraction of Expt-B-triggered members assigned to
            treatment (before correlation is applied).
        assignment_correlation: How much Expt A treatment assignment inflates the
            probability of Expt B treatment. 0 = independent; 0.3 means P(B=treat
            | A=treat) = treat_rate_b + 0.3, clipped to [0,1].
        seed: Random seed.

    Returns:
        DataFrame with columns: "variant" (tuple of two strings), "variant_count".
        Untriggered members have CATEG_NAN_VALUE for their variant slot.
    """
    rng = np.random.default_rng(seed)

    triggered_a = rng.random(n_members) < trigger_rate_a
    triggered_b = rng.random(n_members) < trigger_rate_b

    # Expt A variant: independent of B
    variant_a_treat = rng.random(n_members) < treat_rate_a
    variant_a = np.where(triggered_a, np.where(variant_a_treat, "treat", "ctrl"), CATEG_NAN_VALUE)

    # Expt B variant: may be correlated with Expt A treatment
    prob_b_treat = np.where(
        variant_a == "treat",
        np.clip(treat_rate_b + assignment_correlation, 0, 1),
        treat_rate_b,
    )
    variant_b_treat = rng.random(n_members) < prob_b_treat
    variant_b = np.where(triggered_b, np.where(variant_b_treat, "treat", "ctrl"), CATEG_NAN_VALUE)

    # Exclude R00 (members triggering neither) -- MEA's design
    in_analysis = triggered_a | triggered_b

    df_members = pd.DataFrame(
        {
            "variant_a": variant_a[in_analysis],
            "variant_b": variant_b[in_analysis],
        }
    )
    df_members[VARIANT_COL] = list(zip(df_members["variant_a"], df_members["variant_b"]))

    variant_count_df = df_members.groupby(VARIANT_COL).size().reset_index(name=VARIANT_COUNT_COL)
    return variant_count_df


# ---------------------------------------------------------------------------
# _build_variant_count_array
# ---------------------------------------------------------------------------


def make_simple_variant_count_df():
    """3x3 table (2 variants each + nan) with known counts."""
    rows = [
        (("ctrl", "ctrl"), 400),
        (("ctrl", "treat"), 390),
        (("ctrl", CATEG_NAN_VALUE), 200),
        (("treat", "ctrl"), 410),
        (("treat", "treat"), 395),
        (("treat", CATEG_NAN_VALUE), 210),
        ((CATEG_NAN_VALUE, "ctrl"), 300),
        ((CATEG_NAN_VALUE, "treat"), 295),
        # (nan, nan) excluded -- structural zero
    ]
    return pd.DataFrame(rows, columns=[VARIANT_COL, VARIANT_COUNT_COL])


def test_build_variant_count_array_shape():
    vcdf = make_simple_variant_count_df()
    arr, row_labels, col_labels = _build_variant_count_array(vcdf)
    assert arr.shape == (3, 3)
    assert len(row_labels) == 3
    assert len(col_labels) == 3


def test_build_variant_count_array_nan_last():
    vcdf = make_simple_variant_count_df()
    _, row_labels, col_labels = _build_variant_count_array(vcdf)
    assert row_labels[-1] == CATEG_NAN_VALUE
    assert col_labels[-1] == CATEG_NAN_VALUE


def test_build_variant_count_array_counts():
    vcdf = make_simple_variant_count_df()
    arr, row_labels, col_labels = _build_variant_count_array(vcdf)
    ctrl_idx_a = row_labels.index("ctrl")
    ctrl_idx_b = col_labels.index("ctrl")
    assert arr[ctrl_idx_a, ctrl_idx_b] == 400


def test_build_variant_count_array_nan_nan_corner_is_zero():
    vcdf = make_simple_variant_count_df()
    arr, _, _ = _build_variant_count_array(vcdf)
    # (nan, nan) corner should be 0 since it's excluded by MEA design
    assert arr[-1, -1] == 0.0


# ---------------------------------------------------------------------------
# check_trigger_invariance: simulated independent data (should PASS)
# ---------------------------------------------------------------------------


def test_independent_assignment_passes():
    """Under proper independent randomization the trigger invariance test should pass."""
    vcdf = simulate_mea_members(
        n_members=100000,
        assignment_correlation=0.0,
        seed=42,
    )
    result = check_trigger_invariance(vcdf, alpha=0.001, min_cramers_v=0.01)
    assert isinstance(result, ArmTriggerInvarianceResult)
    assert result.passed_by_source is True


def test_independent_assignment_low_cramers_v():
    vcdf = simulate_mea_members(n_members=100000, assignment_correlation=0.0, seed=7)
    result = check_trigger_invariance(vcdf, alpha=0.001, min_cramers_v=0.01)
    for pair_result in result.by_pair.values():
        assert pair_result.cramers_v < 0.05


# ---------------------------------------------------------------------------
# check_trigger_invariance: simulated correlated data (should FLAG)
# ---------------------------------------------------------------------------


def test_correlated_assignment_flags():
    """When assignment is correlated the trigger invariance test should flag."""
    vcdf = simulate_mea_members(
        n_members=100000,
        assignment_correlation=0.3,
        seed=42,
    )
    result = check_trigger_invariance(vcdf, alpha=0.001, min_cramers_v=0.01)
    assert result.passed_by_source is False


def test_correlated_assignment_high_chi2():
    vcdf = simulate_mea_members(n_members=50000, assignment_correlation=0.3, seed=0)
    result = check_trigger_invariance(vcdf)
    # At least one pair should show a large chi2
    max_chi2 = max(pr.test_result.chi2_value for pr in result.by_pair.values())
    assert max_chi2 > 100.0


# ---------------------------------------------------------------------------
# check_trigger_invariance: known independent table (exact verification)
# ---------------------------------------------------------------------------


def test_exact_independent_table_passes():
    """Perfectly balanced table should pass."""
    vcdf = make_simple_variant_count_df()
    result = check_trigger_invariance(vcdf, alpha=0.001, min_cramers_v=0.01)
    assert result.passed_by_source is True


# ---------------------------------------------------------------------------
# check_trigger_invariance: result structure
# ---------------------------------------------------------------------------


def test_result_structure_k2():
    """K=2: n_sources == 2, n_pairs == 2, keys present in both dicts."""
    vcdf = simulate_mea_members(n_members=20000, seed=1)
    result = check_trigger_invariance(vcdf)
    assert result.n_sources == 2
    assert 0 in result.by_source
    assert 1 in result.by_source
    assert result.n_pairs == 2
    assert (0, 1) in result.by_pair
    assert (1, 0) in result.by_pair


def test_source_result_type():
    vcdf = simulate_mea_members(n_members=20000, seed=2)
    result = check_trigger_invariance(vcdf)
    for sr in result.by_source.values():
        assert isinstance(sr, SourceTriggerInvarianceResult)


def test_pair_result_type():
    vcdf = simulate_mea_members(n_members=20000, seed=2)
    result = check_trigger_invariance(vcdf)
    for pair_result in result.by_pair.values():
        assert isinstance(pair_result, PairTriggerInvarianceResult)


def test_source_result_properties():
    vcdf = simulate_mea_members(n_members=20000, seed=3)
    result = check_trigger_invariance(vcdf)
    sr = result.by_source[0]
    assert sr.source_arm_index == 0
    assert sr.test_result.chi2_value >= 0.0
    assert 0.0 <= sr.p_value <= 1.0
    assert sr.cramers_v >= 0.0
    # row_labels should not contain nan (source is triggered-only)
    assert CATEG_NAN_VALUE not in sr.row_labels
    # col_labels include nan-containing combinations (joint other dims)
    assert any(CATEG_NAN_VALUE in label for label in sr.col_labels)


def test_pair_result_properties():
    vcdf = simulate_mea_members(n_members=20000, seed=3)
    result = check_trigger_invariance(vcdf)
    pr = result.by_pair[(0, 1)]
    assert pr.source_arm_index == 0
    assert pr.target_trigger_index == 1
    assert pr.test_result.chi2_value >= 0.0
    assert 0.0 <= pr.p_value <= 1.0
    assert pr.cramers_v >= 0.0
    # row_labels should not contain nan (source is triggered-only)
    assert CATEG_NAN_VALUE not in pr.row_labels
    # col_labels should contain nan (target includes not-triggered column)
    assert CATEG_NAN_VALUE in pr.col_labels


def test_result_str_contains_pass_or_flag():
    vcdf = make_simple_variant_count_df()
    result = check_trigger_invariance(vcdf)
    s = str(result)
    assert "PASS" in s or "FLAG" in s
    assert "MEA Arm-Trigger Invariance" in s


def test_source_alpha_fields():
    """Each source result carries alpha, alpha_bonferroni (= alpha/K), and p_value."""
    vcdf = simulate_mea_members(n_members=10000, seed=1)
    result = check_trigger_invariance(vcdf, alpha=0.002)
    for sr in result.by_source.values():
        assert sr.alpha == pytest.approx(0.002)
        assert sr.alpha_bonferroni == pytest.approx(0.002 / 2)  # K=2
        assert 0.0 <= sr.p_value <= 1.0


def test_pair_alpha_fields():
    """Each pair result carries alpha, alpha_bonferroni (= alpha/K*(K-1)), and p_value."""
    vcdf = simulate_mea_members(n_members=10000, seed=1)
    result = check_trigger_invariance(vcdf, alpha=0.002)
    for pr in result.by_pair.values():
        assert pr.alpha == pytest.approx(0.002)
        assert pr.alpha_bonferroni == pytest.approx(0.002 / 2)  # K=2, K*(K-1)=2
        assert 0.0 <= pr.p_value <= 1.0


def test_alpha_bonferroni_property():
    """result.alpha_bonferroni_by_source reflects K joint tests (alpha / K)."""
    vcdf = simulate_mea_members(n_members=10000, seed=1)
    result = check_trigger_invariance(vcdf, alpha=0.05)
    assert result.alpha_bonferroni_by_source == pytest.approx(0.05 / 2)  # K=2


def test_p_values_property():
    """p_values is keyed by source_arm_index (K joint tests)."""
    vcdf = simulate_mea_members(n_members=10000, seed=1)
    result = check_trigger_invariance(vcdf)
    assert set(result.p_values_by_source.keys()) == set(result.by_source.keys())
    for src, p in result.p_values_by_source.items():
        assert p == pytest.approx(result.by_source[src].p_value)


def test_min_p_value_property():
    vcdf = simulate_mea_members(n_members=10000, seed=1)
    result = check_trigger_invariance(vcdf)
    assert result.min_p_value_by_source == pytest.approx(min(result.p_values_by_source.values()))


# ---------------------------------------------------------------------------
# check_trigger_invariance: three-experiment case
# ---------------------------------------------------------------------------


def simulate_3_expt_variant_count_df(n_members=30000, seed=0):
    """Simulate three independent experiments and build a variant_count_df."""
    rng = np.random.default_rng(seed)
    trigger_rates = [0.6, 0.5, 0.7]
    treat_rate = 0.5

    triggered = [rng.random(n_members) < rate for rate in trigger_rates]
    variants = [
        np.where(trig, np.where(rng.random(n_members) < treat_rate, "treat", "ctrl"), CATEG_NAN_VALUE)
        for trig in triggered
    ]

    in_analysis = triggered[0] | triggered[1] | triggered[2]
    df_members = pd.DataFrame(
        {
            "v0": variants[0][in_analysis],
            "v1": variants[1][in_analysis],
            "v2": variants[2][in_analysis],
        }
    )
    df_members[VARIANT_COL] = list(zip(df_members["v0"], df_members["v1"], df_members["v2"]))
    return df_members.groupby(VARIANT_COL).size().reset_index(name=VARIANT_COUNT_COL)


def test_three_expt_n_sources_and_pairs():
    """K=3 should have n_sources=3 and n_pairs=6."""
    vcdf = simulate_3_expt_variant_count_df(seed=5)
    result = check_trigger_invariance(vcdf, alpha=0.001, min_cramers_v=0.01)
    assert result.n_sources == 3
    assert len(result.by_source) == 3
    assert result.n_pairs == 6
    assert len(result.by_pair) == 6


def test_three_expt_all_pairs_present():
    vcdf = simulate_3_expt_variant_count_df(seed=5)
    result = check_trigger_invariance(vcdf)
    expected_pairs = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
    assert set(result.by_pair.keys()) == expected_pairs


def test_three_expt_alpha_bonferroni_is_over_k():
    """For K=3, result.alpha_bonferroni_by_source = alpha / 3 (K joint tests)."""
    vcdf = simulate_3_expt_variant_count_df(seed=5)
    result = check_trigger_invariance(vcdf, alpha=0.003)
    assert result.alpha_bonferroni_by_source == pytest.approx(0.003 / 3)
    for sr in result.by_source.values():
        assert sr.alpha_bonferroni == pytest.approx(0.003 / 3)
    for pr in result.by_pair.values():
        assert pr.alpha_bonferroni == pytest.approx(0.003 / 6)


def test_three_expt_independent_all_pass():
    vcdf = simulate_3_expt_variant_count_df(n_members=60000, seed=99)
    result = check_trigger_invariance(vcdf, alpha=0.001, min_cramers_v=0.01)
    assert result.passed_by_source is True
    for src, sr in result.by_source.items():
        assert sr.passed is True, f"Source {src} unexpectedly flagged: {sr}"
    for (src, tgt), pr in result.by_pair.items():
        assert pr.passed is True, f"Pair ({src}->{tgt}) unexpectedly flagged: {pr}"


# ---------------------------------------------------------------------------
# exclude_tuples
# ---------------------------------------------------------------------------


def test_exclude_tuples_zeros_count_cell():
    vcdf = make_simple_variant_count_df()
    arr_full, _, _ = _build_variant_count_array(vcdf)
    arr_excl, row_labels, col_labels = _build_variant_count_array(vcdf, exclude_tuples=[("ctrl", "ctrl")])
    ri = row_labels.index("ctrl")
    ci = col_labels.index("ctrl")
    assert arr_full[ri, ci] == 400
    assert arr_excl[ri, ci] == 0.0


def test_exclude_tuples_nonexistent_ignored():
    """Excluding a non-existent tuple should not raise or change counts."""
    vcdf = make_simple_variant_count_df()
    arr_full, _, _ = _build_variant_count_array(vcdf)
    arr_excl, _, _ = _build_variant_count_array(vcdf, exclude_tuples=[("bogus", "bogus")])
    np.testing.assert_array_equal(arr_full, arr_excl)


# ---------------------------------------------------------------------------
# _build_source_joint_table
# ---------------------------------------------------------------------------


def test_build_source_joint_table_shape_k2():
    """K=2: rows = non-nan source arms, cols = other dim values (including nan)."""
    vcdf = make_simple_variant_count_df()
    arr, row_labels, col_labels = _build_source_joint_table(vcdf, source_idx=0)
    # source_idx=0: rows = non-nan arms of expt 0 (ctrl, treat), cols = expt 1 values
    assert CATEG_NAN_VALUE not in row_labels
    assert len(row_labels) == 2  # ctrl, treat
    assert any(CATEG_NAN_VALUE in c for c in col_labels)
    assert arr.shape == (len(row_labels), len(col_labels))


def test_build_source_joint_table_nan_cols_last_k2():
    """nan-containing column should sort last."""
    vcdf = make_simple_variant_count_df()
    _, _, col_labels = _build_source_joint_table(vcdf, source_idx=0)
    nan_positions = [i for i, c in enumerate(col_labels) if CATEG_NAN_VALUE in c]
    non_nan_positions = [i for i, c in enumerate(col_labels) if CATEG_NAN_VALUE not in c]
    if nan_positions and non_nan_positions:
        assert min(nan_positions) > max(non_nan_positions)


def test_build_source_joint_table_counts_sum():
    """Total count in source joint table equals count of triggered-in-source users."""
    vcdf = make_simple_variant_count_df()
    arr, row_labels, _ = _build_source_joint_table(vcdf, source_idx=0)
    # Total should equal sum of counts where source is non-nan
    df = vcdf.copy()
    df["_src"] = df[VARIANT_COL].map(lambda v: v[0])
    expected = df[df["_src"] != CATEG_NAN_VALUE][VARIANT_COUNT_COL].sum()
    assert arr.sum() == pytest.approx(expected)


def test_build_source_joint_table_k3_col_labels_are_joint():
    """K=3, source=1: col_labels should be 'v0|v2' joined strings."""
    vcdf = simulate_3_expt_variant_count_df(seed=7)
    _, _, col_labels = _build_source_joint_table(vcdf, source_idx=1)
    # Each col_label is a "|"-joined combination of expt 0 and expt 2 values
    assert all("|" in c for c in col_labels)


# ---------------------------------------------------------------------------
# Simulation helpers for metric interaction tests
# ---------------------------------------------------------------------------


def _make_metric_stats_df(
    n_members=20000,
    interaction_strength=0.0,
    base_mean=10.0,
    treat_effect_a=1.0,
    treat_effect_b=0.5,
    within_sd=5.0,
    trigger_rate_a=0.6,
    trigger_rate_b=0.5,
    seed=42,
):
    """Simulate a variant_metric_stats_df for one metric.

    Each member's metric:
      y = base_mean + treat_a * treat_effect_a + treat_b * treat_effect_b
          + treat_a * treat_b * interaction_strength + noise

    interaction_strength=0  → additive, no interaction → test should NOT fire
    interaction_strength!=0 → non-additive interaction → test SHOULD fire
    """
    rng = np.random.default_rng(seed)
    triggered_a = rng.random(n_members) < trigger_rate_a
    triggered_b = rng.random(n_members) < trigger_rate_b
    in_inner = triggered_a & triggered_b

    v_a = np.where(rng.random(n_members) < 0.5, "treat", "ctrl")
    v_b = np.where(rng.random(n_members) < 0.5, "treat", "ctrl")

    treat_a = (v_a == "treat").astype(float)
    treat_b = (v_b == "treat").astype(float)
    noise = rng.normal(0, within_sd, n_members)
    y = base_mean + treat_effect_a * treat_a + treat_effect_b * treat_b + interaction_strength * treat_a * treat_b + noise

    df = pd.DataFrame(
        {
            "variant_a": v_a[in_inner],
            "variant_b": v_b[in_inner],
            "y": y[in_inner],
        }
    )
    df[VARIANT_COL] = list(zip(df["variant_a"], df["variant_b"]))

    grouped = (
        df.groupby(VARIANT_COL)["y"]
        .agg(
            sample_count="count",
            mean_val="mean",
            sum_val="sum",
            sum_sq_val=lambda x: (x**2).sum(),
        )
        .reset_index()
    )
    grouped[MEAN_COL] = grouped["mean_val"]
    grouped[SAMPLE_COUNT_COL] = grouped["sample_count"]
    grouped[SUM_COL] = grouped["sum_val"]
    grouped[SUM_SQ_COL] = grouped["sum_sq_val"]
    grouped[SD_COL] = np.sqrt(grouped[SUM_SQ_COL] / grouped[SAMPLE_COUNT_COL] - grouped[MEAN_COL] ** 2)
    return grouped[[VARIANT_COL, SAMPLE_COUNT_COL, MEAN_COL, SD_COL, SUM_COL, SUM_SQ_COL]]


# ---------------------------------------------------------------------------
# _build_cell_arrays_for_trigger_state
# ---------------------------------------------------------------------------


def test_build_cell_arrays_for_trigger_state_shape():
    df = _make_metric_stats_df(seed=1)
    means, sds, counts, labels = _build_cell_arrays_for_trigger_state(df, (True, True), [0, 1])
    # 2 variants each (ctrl, treat) -> 2x2 inner block
    assert means.shape == (2, 2)
    assert sds.shape == (2, 2)
    assert counts.shape == (2, 2)
    assert len(labels[0]) == 2
    assert len(labels[1]) == 2


def test_build_cell_arrays_for_trigger_state_no_nan_labels():
    df = _make_metric_stats_df(seed=2)
    _, _, _, labels = _build_cell_arrays_for_trigger_state(df, (True, True), [0, 1])
    assert CATEG_NAN_VALUE not in labels[0]
    assert CATEG_NAN_VALUE not in labels[1]


def test_build_cell_arrays_for_trigger_state_means_finite():
    df = _make_metric_stats_df(seed=3)
    means, sds, counts, _ = _build_cell_arrays_for_trigger_state(df, (True, True), [0, 1])
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(sds))
    assert np.all(np.isfinite(counts))


def test_build_cell_arrays_for_trigger_state_sds_nonneg():
    df = _make_metric_stats_df(seed=4)
    _, sds, _, _ = _build_cell_arrays_for_trigger_state(df, (True, True), [0, 1])
    assert np.all(sds >= 0.0)


def test_build_cell_arrays_for_trigger_state_counts_match_total():
    """Total count in cell arrays matches total members in inner block."""
    df = _make_metric_stats_df(n_members=10000, seed=5)
    _, _, counts, _ = _build_cell_arrays_for_trigger_state(df, (True, True), [0, 1])
    assert np.nansum(counts) == pytest.approx(df[SAMPLE_COUNT_COL].sum(), rel=1e-9)


def test_build_cell_arrays_for_trigger_state_mean_plausible():
    """Cell means should be near the expected values given the simulation."""
    # base=10, treat_a=2, treat_b=1, no interaction, large n -> low noise
    df = _make_metric_stats_df(
        n_members=200000,
        base_mean=10.0,
        treat_effect_a=2.0,
        treat_effect_b=1.0,
        within_sd=5.0,
        seed=0,
    )
    means, _, _, labels = _build_cell_arrays_for_trigger_state(df, (True, True), [0, 1])
    rows, cols = labels[0], labels[1]
    ri_ctrl = rows.index("ctrl")
    ri_treat = rows.index("treat")
    ci_ctrl = cols.index("ctrl")
    ci_treat = cols.index("treat")
    # ctrl/ctrl ~ 10, treat/ctrl ~ 12, ctrl/treat ~ 11, treat/treat ~ 13
    assert means[ri_ctrl, ci_ctrl] == pytest.approx(10.0, abs=0.3)
    assert means[ri_treat, ci_ctrl] == pytest.approx(12.0, abs=0.3)
    assert means[ri_ctrl, ci_treat] == pytest.approx(11.0, abs=0.3)
    assert means[ri_treat, ci_treat] == pytest.approx(13.0, abs=0.3)


# ---------------------------------------------------------------------------
# check_metric_interaction: no interaction (additive) -- should not fire
# ---------------------------------------------------------------------------


def test_check_metric_interaction_additive_high_p():
    """Pure additive model: interaction test p-value should be large in R11."""
    df = _make_metric_stats_df(n_members=100000, interaction_strength=0.0, within_sd=5.0, seed=10)
    out = check_metric_interaction(df, metric_name="sessions")
    r11 = out["interaction_result"].by_trigger_state[(True, True)]
    assert r11.test_result.p_value > 0.05


def test_check_metric_interaction_no_interaction_small_f():
    df = _make_metric_stats_df(n_members=50000, interaction_strength=0.0, seed=11)
    out = check_metric_interaction(df)
    r11 = out["interaction_result"].by_trigger_state[(True, True)]
    assert r11.test_result.f_value < 20.0


# ---------------------------------------------------------------------------
# check_metric_interaction: strong interaction -- should fire
# ---------------------------------------------------------------------------


def test_check_metric_interaction_strong_interaction_low_p():
    """Strong interaction: R11 test should flag."""
    df = _make_metric_stats_df(n_members=200000, interaction_strength=5.0, within_sd=5.0, seed=20)
    out = check_metric_interaction(df)
    assert out["interaction_result"].flagged is True
    r11 = out["interaction_result"].by_trigger_state[(True, True)]
    assert r11.test_result.p_value < 0.001


def test_check_metric_interaction_strong_interaction_large_f():
    df = _make_metric_stats_df(n_members=200000, interaction_strength=5.0, seed=21)
    out = check_metric_interaction(df)
    r11 = out["interaction_result"].by_trigger_state[(True, True)]
    assert r11.test_result.f_value > 10.0


# ---------------------------------------------------------------------------
# check_metric_interaction: output structure
# ---------------------------------------------------------------------------


def test_check_metric_interaction_returns_correct_keys():
    df = _make_metric_stats_df(seed=30)
    out = check_metric_interaction(df)
    assert "interaction_result" in out
    assert "by_trigger_state" in out
    assert isinstance(out["interaction_result"], MetricInteractionResult)


def test_check_metric_interaction_r11_has_figs():
    df = _make_metric_stats_df(seed=31)
    out = check_metric_interaction(df, metric_name="bookings")
    pair_viz = out["by_trigger_state"][(True, True)]["pair_heatmaps"][(0, 1)]
    assert hasattr(pair_viz["means_fig"], "data")
    assert hasattr(pair_viz["residuals_fig"], "data")


def test_check_metric_interaction_expt_names_in_axis():
    df = _make_metric_stats_df(seed=32)
    out = check_metric_interaction(df, expt_names=["SJS A", "SJS B"])
    means_fig = out["by_trigger_state"][(True, True)]["pair_heatmaps"][(0, 1)]["means_fig"]
    assert "SJS A" in means_fig.layout.yaxis.title.text
    assert "SJS B" in means_fig.layout.xaxis.title.text


def test_check_metric_interaction_residuals_shape():
    df = _make_metric_stats_df(seed=33)
    out = check_metric_interaction(df)
    r11 = out["interaction_result"].by_trigger_state[(True, True)]
    assert r11.test_result.cell_residuals is not None
    assert r11.test_result.cell_residuals.shape == (2, 2)


def test_check_metric_interaction_trigger_state_result_type():
    df = _make_metric_stats_df(seed=34)
    out = check_metric_interaction(df)
    for ts, ts_result in out["interaction_result"].by_trigger_state.items():
        assert isinstance(ts_result, MetricTriggerStateTestResult)
        assert ts_result.trigger_state == ts
        assert isinstance(ts_result.triggered_dims, list)
        assert len(ts_result.arm_labels) == len(ts_result.triggered_dims)


def test_check_metric_interaction_k2_pair_heatmap_exists():
    """For K=2, pair_heatmaps[(0,1)] must be populated in by_trigger_state."""
    df = _make_metric_stats_df(seed=35)
    out = check_metric_interaction(df)
    assert (0, 1) in out["by_trigger_state"][(True, True)]["pair_heatmaps"]


def test_check_metric_interaction_k2_pair_heatmap_keys():
    """pair_heatmaps viz dict has expected keys for K=2."""
    df = _make_metric_stats_df(seed=36)
    out = check_metric_interaction(df)
    pair_viz = out["by_trigger_state"][(True, True)]["pair_heatmaps"][(0, 1)]
    assert set(pair_viz.keys()) == {"means_fig", "residuals_fig", "row_labels", "col_labels"}


# ---------------------------------------------------------------------------
# _build_cell_arrays_for_trigger_state: extra checks
# ---------------------------------------------------------------------------


def test_trigger_state_metric_cell_arrays_counts_positive():
    df = _make_metric_stats_df(seed=40)
    _, _, counts, labels = _build_cell_arrays_for_trigger_state(df, (True, True), [0, 1])
    ri = labels[0].index("ctrl")
    ci = labels[1].index("ctrl")
    assert counts[ri, ci] > 0


# ---------------------------------------------------------------------------
# K=3 metric interaction tests
# ---------------------------------------------------------------------------


def _make_3expt_metric_stats_df(
    n_members=30000,
    interaction_ab=0.0,
    interaction_abc=0.0,
    base_mean=10.0,
    treat_effect_a=1.0,
    treat_effect_b=0.5,
    treat_effect_c=0.3,
    within_sd=5.0,
    trigger_rate_a=0.6,
    trigger_rate_b=0.5,
    trigger_rate_c=0.4,
    seed=42,
):
    """Simulate a 3-experiment variant_metric_stats_df.

    y = base + a*ea + b*eb + c*ec + a*b*interaction_ab + a*b*c*interaction_abc + noise

    interaction_ab=0, interaction_abc=0  -> fully additive
    interaction_ab!=0                    -> 2-way A×B interaction (visible in R110, R111)
    interaction_abc!=0                   -> pure 3-way A×B×C interaction:
        visible only in R111 (all three triggered); R110/R101/R011 are unaffected
        because within each of those strata one factor is always 0 (not triggered)

    Produces all 8 trigger state combinations with appropriate arm tuples.
    """
    rng = np.random.default_rng(seed)
    trig_a = rng.random(n_members) < trigger_rate_a
    trig_b = rng.random(n_members) < trigger_rate_b
    trig_c = rng.random(n_members) < trigger_rate_c

    v_a = np.where(rng.random(n_members) < 0.5, "treat", "ctrl")
    v_b = np.where(rng.random(n_members) < 0.5, "treat", "ctrl")
    v_c = np.where(rng.random(n_members) < 0.5, "treat", "ctrl")

    ta = (v_a == "treat").astype(float)
    tb = (v_b == "treat").astype(float)
    tc = (v_c == "treat").astype(float)
    noise = rng.normal(0, within_sd, n_members)
    # interaction_ab: gated on both A and B triggering (ta*tb always 0 in R01x/R10x strata)
    # interaction_abc: gated on ALL THREE triggering — ta*tb*tc is still nonzero for R110
    #   members (tc is a real assignment even if var_c=nan), so we must gate explicitly on
    #   trig_c (and trig_a, trig_b) to isolate the effect to R111 only.
    trig_abc = trig_a.astype(float) * trig_b.astype(float) * trig_c.astype(float)
    y = (
        base_mean
        + treat_effect_a * ta
        + treat_effect_b * tb
        + treat_effect_c * tc
        + interaction_ab * ta * tb
        + interaction_abc * ta * tb * tc * trig_abc
        + noise
    )

    nan_val = CATEG_NAN_VALUE
    var_a = np.where(trig_a, v_a, nan_val)
    var_b = np.where(trig_b, v_b, nan_val)
    var_c = np.where(trig_c, v_c, nan_val)

    df = pd.DataFrame({VARIANT_COL: list(zip(var_a, var_b, var_c)), "y": y})
    grouped = (
        df.groupby(VARIANT_COL)["y"]
        .agg(
            sample_count="count",
            sum_val="sum",
            sum_sq_val=lambda x: (x**2).sum(),
        )
        .reset_index()
    )
    grouped[SAMPLE_COUNT_COL] = grouped["sample_count"]
    grouped[SUM_COL] = grouped["sum_val"]
    grouped[SUM_SQ_COL] = grouped["sum_sq_val"]
    grouped[MEAN_COL] = grouped[SUM_COL] / grouped[SAMPLE_COUNT_COL]
    grouped[SD_COL] = np.sqrt(grouped[SUM_SQ_COL] / grouped[SAMPLE_COUNT_COL] - grouped[MEAN_COL] ** 2)
    return grouped[[VARIANT_COL, SAMPLE_COUNT_COL, MEAN_COL, SD_COL, SUM_COL, SUM_SQ_COL]]


def test_k3_metric_interaction_all_trigger_states_tested():
    """K=3 should test R110, R101, R011, R111 -- 4 strata with >= 2 triggered."""
    df = _make_3expt_metric_stats_df(seed=50)
    out = check_metric_interaction(df)
    tested = set(out["interaction_result"].by_trigger_state.keys())
    assert (True, True, False) in tested
    assert (True, False, True) in tested
    assert (False, True, True) in tested
    assert (True, True, True) in tested
    assert len(tested) == 4


def test_k3_metric_interaction_r111_is_3d():
    """R111 K-dim test should have 3D residuals array."""
    df = _make_3expt_metric_stats_df(seed=51)
    out = check_metric_interaction(df)
    r111 = out["interaction_result"].by_trigger_state[(True, True, True)]
    assert r111.triggered_dims == [0, 1, 2]
    assert r111.test_result.cell_residuals is not None
    assert r111.test_result.cell_residuals.ndim == 3


def test_k3_metric_interaction_r111_no_pair_heatmaps():
    """R111 (K=3) should have an empty pair_heatmaps — heatmaps only for K=2 strata."""
    df = _make_3expt_metric_stats_df(seed=53)
    out = check_metric_interaction(df)
    r111_heatmaps = out["by_trigger_state"][(True, True, True)]["pair_heatmaps"]
    assert len(r111_heatmaps) == 0


def test_k3_metric_interaction_r110_is_2d():
    """R110 K-dim test should have 2D residuals (only dims 0,1 triggered)."""
    df = _make_3expt_metric_stats_df(seed=54)
    out = check_metric_interaction(df)
    r110 = out["interaction_result"].by_trigger_state[(True, True, False)]
    assert r110.triggered_dims == [0, 1]
    assert r110.test_result.cell_residuals.ndim == 2


def test_k3_metric_interaction_additive_no_flag():
    """No interaction in any stratum -> should not flag."""
    df = _make_3expt_metric_stats_df(n_members=80000, interaction_ab=0.0, within_sd=5.0, seed=60)
    out = check_metric_interaction(df)
    assert out["interaction_result"].flagged is False


def test_k3_metric_interaction_strong_ab_flags_r111():
    """Strong A×B interaction should flag strata where both A and B are triggered."""
    df = _make_3expt_metric_stats_df(n_members=200000, interaction_ab=5.0, within_sd=5.0, seed=61)
    out = check_metric_interaction(df)
    assert out["interaction_result"].flagged is True
    r111 = out["interaction_result"].by_trigger_state[(True, True, True)]
    assert r111.test_result.p_value < 0.001


def test_k3_metric_interaction_three_way_only_flags_r111():
    """Pure 3-way A×B×C interaction: only R111 should flag; R110/R101/R011 stay clean.

    Within R110, R101, R011 one factor is never triggered so ta*tb*tc=0 for all
    members in those strata — the model is purely additive there. Only R111 has
    all three factors active and the 3D ANOVA detects the anomalous (treat,treat,treat)
    cell. This validates the K-dim test's ability to catch interactions that are
    invisible to every pairwise stratum test.
    """
    df = _make_3expt_metric_stats_df(n_members=300000, interaction_abc=6.0, within_sd=2.0, seed=70)
    out = check_metric_interaction(df)
    results = out["interaction_result"].by_trigger_state

    # Pairwise strata: one factor always 0 -> no 3-way signal -> should not flag
    assert results[(True, True, False)].test_result.p_value > 0.05, "R110 should not flag (C never triggered)"
    assert results[(True, False, True)].test_result.p_value > 0.05, "R101 should not flag (B never triggered)"
    assert results[(False, True, True)].test_result.p_value > 0.05, "R011 should not flag (A never triggered)"

    # R111: all three triggered -> 3D test detects 3-way interaction
    assert results[(True, True, True)].test_result.p_value < 0.001, "R111 3D test should flag"
    assert out["interaction_result"].flagged is True

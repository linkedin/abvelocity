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
MEA Assumption Check — Arm-Trigger Invariance.

--- Causal graph (K=2, null -- Assumption 1 holds) ---

Nodes (top to bottom): L, {A_0, A_1}, {S_0, S_1}, Y, U

    L:   the arm combination launched — corresponds to the `Launch` concept in
         MEA (e.g. (ctrl, treat), (treat, ctrl), etc.). This is a
         do-intervention: the experimenter sets L, which then independently
         hashes users into each experiment's arm.
    A_i: latent arm assignment for experiment i (e.g. ctrl / treat).
         Determined by L via independent hashing; A_0 and A_1 are independent
         of each other given L.
         Note: A_i != V_i. V_i is the *observed* variant in the data
         (= A_i if the user triggers experiment i, else nan).
    S_i: trigger state -- 1 if user enters experiment i's eligibility surface.
    Y:   outcome metric.
    U:   unobserved user characteristics -- common cause of triggering and outcome.

Edges (null -- Assumption 1 holds):
    L   -> A_0, A_1      launch sets arm assignments
    A_i -> Y             arms directly affect outcome  (i = 0, 1)
    S_i -> Y             trigger states affect outcome (i = 0, 1)
    U   -> S_0, S_1, Y   common cause of triggering and outcome

Forbidden (Assumption 1 violation -- trigger contamination):
    A_0 - - > S_1        arm of exp 0 must not shift trigger rate of exp 1
    A_1 - - > S_0        arm of exp 1 must not shift trigger rate of exp 0

If either forbidden edge exists, the nan proportion in V_i will be uneven
across A_j arms -- that is the trigger-contamination signal detected by
check_trigger_invariance.

--- The core assumption ---

MEA Arm-Trigger Invariance (Assumption 1): the distribution of trigger states
is invariant to which arm combination is launched. Formally, for all l:

    P(S | do(L = l)) = P(S)

This fails when the treatment arm of one experiment causally shifts the trigger
rate of another (trigger contamination: the A_j -> S_i edge in the causal graph).

Under independent hashing, assignment independence within strata (A_i ⊥ A_j | S=s)
follows as a corollary and does not need to be tested separately — it is guaranteed
by Arm-Trigger Invariance alone. See the formal proof in the MEA Assumption paper.

--- The test ---

`check_trigger_invariance` runs K joint chi-squared homogeneity tests (default),
one per source experiment j:

  - Restrict to users triggered in experiment j (V_j != nan).
  - Build a p_j x M contingency table:
        rows    = non-nan arm values of V_j  (A_j arms, triggered users only)
        columns = all joint combinations of (V_i)_{i != j}, including nan
  - Chi-squared homogeneity test: null = each row has the same conditional
    joint distribution of all other experiments' variants.

Bonferroni correction applied across all K sources (alpha / K).

The K joint tests align exactly with the K conditional distribution plots
(plot_conditional_variant_dist) — one test per fixed source dimension.

`check_trigger_invariance` also computes K(K-1) pairwise tests (one per ordered
pair (j, i) with j != i) for diagnostic drill-down. These are stored in
`result.by_pair` but do not affect `passed`, `min_p_value`, or `alpha_bonferroni`.
For K=2 the K and K(K-1) tests are identical.

--- Input format ---

All public functions take a DataFrame with a `variant_col` column whose values
are tuples of variant strings, one element per experiment. For example:
  ("ctrl", "treat")        — two-experiment MEA, both triggered
  ("ctrl", "nan", "treat") — three-experiment MEA, experiment 1 not triggered

Public API:
    check_trigger_invariance(variant_count_df, ...)
        -> ArmTriggerInvarianceResult
           .by_source  -- dict[source_arm_index] -> SourceTriggerInvarianceResult  (K joint)
           .by_pair    -- dict[(arm_expt_j, trigger_expt_i)] -> PairTriggerInvarianceResult  (K*(K-1) pairwise)
           .passed     -- overall pass/fail (all K sources must pass)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from abvelocity.core.param.constants import (
    CATEG_NAN_VALUE,
    VARIANT_COL,
    VARIANT_COUNT_COL,
)
from abvelocity.core.stats.count_independence_test import (
    CountIndependenceTestResult,
    count_independence_test,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SourceTriggerInvarianceResult:
    """Result for one source experiment j — K-joint test across all other dims.

    Tests whether arm assignment of experiment j shifts the joint distribution
    of all other experiments' variants:
        H0: P(V_{-j} | V_j = v_j) = P(V_{-j})   for all v_j in A_j.

    Contingency table:
        rows    = non-nan arm values of V_j   (source arm, triggered users only)
        columns = all joint combinations of (V_i)_{i != j}, including nan in
                  each dimension — nan-containing columns sorted last

    A flagged result identifies experiment j as a potential cause of trigger
    contamination. For the specific target dimension affected, see by_pair in
    ArmTriggerInvarianceResult.

    Attributes:
        source_arm_index: Index of experiment j — whose arm forms the table rows.
        test_result: CountIndependenceTestResult from the chi-squared test.
        p_value: Chi-squared p-value.
        alpha: Nominal significance level (before Bonferroni correction).
        alpha_bonferroni: Bonferroni-adjusted threshold (= alpha / K).
        passed: True if p >= alpha_bonferroni OR cramers_v <= min_cramers_v.
        min_cramers_v: Minimum effect size threshold for flagging.
        row_labels: Non-nan arm values of source experiment.
        col_labels: Joint combination labels for other dims, nan-containing last.
    """

    source_arm_index: int
    test_result: CountIndependenceTestResult
    passed: bool
    alpha: float
    alpha_bonferroni: float
    min_cramers_v: float
    row_labels: List[str]
    col_labels: List[str]

    @property
    def p_value(self) -> float:
        return self.test_result.p_value

    @property
    def cramers_v(self) -> float:
        return self.test_result.cramers_v

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FLAG"
        return (
            f"[{status}] Source {self.source_arm_index} | "
            f"chi2={self.test_result.chi2_value:.2f}, "
            f"p={self.p_value:.2e}, alpha_bonf={self.alpha_bonferroni:.2e}, "
            f"V={self.cramers_v:.4f}, N={int(self.test_result.n_total):,}"
        )


@dataclass
class PairTriggerInvarianceResult:
    """Result for one ordered pair: arm experiment j -> trigger experiment i.

    Tests the forbidden causal edge A_j -> S_i: does the arm assignment in
    experiment j shift the trigger probability of experiment i?

    Causal graph for this pair (K=2 shown; same logic applies per-pair for K>2):

        A_j  ------?------>  S_i        <- the edge being tested (should not exist)
         |                    |
         v                    v
        V_j                  V_i        <- observed variants (arm value or nan)

    Contingency table:
        rows    = non-nan arm values of V_j   (A_j arms, triggered users only)
        columns = full distribution of V_i including nan  (S_i trigger state + arms)

    The nan column of V_i is the direct trigger-contamination signal:
    if the proportions of nan differ across rows, A_j -> S_i exists.
    The non-nan columns test within-arm balance (Corollary under indep. hashing).

    Attributes:
        source_arm_index: Index of experiment j — whose arm assignment A_j forms
            the contingency table rows. This is the potential cause in the tested
            causal edge A_j -> S_i.
        target_trigger_index: Index of experiment i — whose trigger state S_i is
            under test. The nan column of V_i is the contamination signal.
        test_result: CountIndependenceTestResult from the chi-squared test.
        p_value: Chi-squared p-value for this pair.
        alpha: Nominal significance level (before Bonferroni correction).
        n_pairs: Total number of ordered pairs tested (= K*(K-1)); used to
            derive alpha_bonferroni.
        alpha_bonferroni: Bonferroni-adjusted threshold (= alpha / K*(K-1)).
        passed: True if p >= alpha_bonferroni OR cramers_v <= min_cramers_v.
        min_cramers_v: Minimum effect size threshold for flagging.
        row_labels: Non-nan arm values of experiment j (A_j arms).
        col_labels: Full V_i labels (arms + nan last).
    """

    source_arm_index: int
    target_trigger_index: int
    test_result: CountIndependenceTestResult
    passed: bool
    alpha: float
    n_pairs: int
    min_cramers_v: float
    row_labels: List[str]
    col_labels: List[str]

    @property
    def alpha_bonferroni(self) -> float:
        return self.alpha / self.n_pairs if self.n_pairs > 0 else self.alpha

    @property
    def p_value(self) -> float:
        return self.test_result.p_value

    @property
    def cramers_v(self) -> float:
        return self.test_result.cramers_v

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FLAG"
        return (
            f"[{status}] Pair ({self.source_arm_index} -> {self.target_trigger_index}) | "
            f"chi2={self.test_result.chi2_value:.2f}, "
            f"p={self.p_value:.2e}, alpha_bonf={self.alpha_bonferroni:.2e}, "
            f"V={self.cramers_v:.4f}, N={int(self.test_result.n_total):,}"
        )


@dataclass
class ArmTriggerInvarianceResult:
    """Combined result of MEA Arm-Trigger Invariance tests.

    The primary test is K joint tests (one per source experiment), stored in
    by_source. passed, alpha_bonferroni, and min_p_value all reflect this default.

    K(K-1) pairwise results are stored in by_pair for diagnostic drill-down
    (identifying which specific A_j -> S_i edge is at fault when a source flags).
    For K=2 the two designs are identical.

    Attributes:
        by_source: Dict keyed by source_arm_index -> SourceTriggerInvarianceResult.
            K joint tests — the default. Bonferroni alpha / K.
        by_pair: Dict keyed by (source_arm_index, target_trigger_index) ->
            PairTriggerInvarianceResult. K*(K-1) pairwise tests — for drill-down.
        passed_by_source: True only if all K joint source tests pass.
        alpha: Nominal significance level.
        min_cramers_v: Effect size threshold.
        n_sources: Number of source tests run (= K).
        n_pairs: Number of pairwise tests run (= K*(K-1)).
        alpha_bonferroni_by_source: Bonferroni threshold for K joint tests (= alpha / K).
        passed_by_pair: True only if all K*(K-1) pairwise tests pass.
        p_values_by_source: Dict of p-values from K joint tests, keyed by source_arm_index.
        min_p_value_by_source: Smallest p-value across K joint tests.
    """

    by_source: Dict[int, "SourceTriggerInvarianceResult"]
    by_pair: Dict[Tuple[int, int], "PairTriggerInvarianceResult"]
    passed_by_source: bool
    alpha: float
    min_cramers_v: float

    @property
    def n_sources(self) -> int:
        return len(self.by_source)

    @property
    def n_pairs(self) -> int:
        return len(self.by_pair)

    @property
    def alpha_bonferroni_by_source(self) -> float:
        return self.alpha / self.n_sources if self.n_sources > 0 else self.alpha

    @property
    def passed_by_pair(self) -> bool:
        return all(pr.passed for pr in self.by_pair.values())

    @property
    def p_values_by_source(self) -> Dict[int, float]:
        return {src: sr.p_value for src, sr in self.by_source.items()}

    @property
    def min_p_value_by_source(self) -> float:
        return min(self.p_values_by_source.values()) if self.p_values_by_source else 1.0

    def __str__(self) -> str:
        src_status = "PASS" if self.passed_by_source else "FLAG"
        pair_status = "PASS" if self.passed_by_pair else "FLAG"
        lines = [
            f"[{src_status}] MEA Arm-Trigger Invariance — {self.n_sources} source test(s) | "
            f"pairwise [{pair_status}]"
        ]
        for sr in self.by_source.values():
            lines.append("  " + str(sr))
        if self.by_pair:
            lines.append(f"  [{self.n_pairs} pairwise detail]")
            for pr in self.by_pair.values():
                lines.append("    " + str(pr))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _build_variant_count_array(
    variant_count_df: pd.DataFrame,
    expt_a_index: int = 0,
    expt_b_index: int = 1,
    variant_col: str = VARIANT_COL,
    count_col: str = VARIANT_COUNT_COL,
    nan_value: str = CATEG_NAN_VALUE,
    exclude_tuples: Optional[List[Tuple]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a 2D variant combination count array for one experiment pair.

    Marginalizes over all experiments other than expt_a_index and expt_b_index.
    Rows = arms of experiment A (including nan for non-triggered).
    Columns = arms of experiment B (including nan for non-triggered).

    Args:
        variant_count_df: DataFrame with columns [variant_col, count_col].
        expt_a_index: Index of experiment A within the variant tuple.
        expt_b_index: Index of experiment B within the variant tuple.
        variant_col: Column name for the variant tuple column.
        count_col: Column name for the count column.
        nan_value: Sentinel string for "not triggered".
        exclude_tuples: Optional list of (arm_a, arm_b) pairs to zero out.

    Returns:
        count_array: 2D ndarray of shape (n_arms_a, n_arms_b), nan last in each dim.
        row_labels: Arm names for experiment A, nan last.
        col_labels: Arm names for experiment B, nan last.
    """
    df = variant_count_df.copy()
    df["_a"] = df[variant_col].map(lambda v: v[expt_a_index])
    df["_b"] = df[variant_col].map(lambda v: v[expt_b_index])
    aggregated = df.groupby(["_a", "_b"])[count_col].sum().reset_index()
    pivot = aggregated.pivot(index="_a", columns="_b", values=count_col).fillna(0)

    def _sort_nan_last(index_vals):
        non_nan = sorted(v for v in index_vals if v != nan_value)
        return non_nan + ([nan_value] if nan_value in index_vals else [])

    row_labels = _sort_nan_last(pivot.index)
    col_labels = _sort_nan_last(pivot.columns)
    pivot = pivot.reindex(index=row_labels, columns=col_labels, fill_value=0)
    count_array = pivot.values.astype(float)

    if exclude_tuples:
        row_idx_map = {v: i for i, v in enumerate(row_labels)}
        col_idx_map = {v: i for i, v in enumerate(col_labels)}
        for val_a, val_b in exclude_tuples:
            if val_a in row_idx_map and val_b in col_idx_map:
                count_array[row_idx_map[val_a], col_idx_map[val_b]] = 0.0

    return count_array, row_labels, col_labels


def _build_source_joint_table(
    variant_count_df: pd.DataFrame,
    source_idx: int,
    variant_col: str = VARIANT_COL,
    count_col: str = VARIANT_COUNT_COL,
    nan_value: str = CATEG_NAN_VALUE,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build contingency table for the K-joint test: source j vs joint of all other dims.

    Rows = non-nan arms of V_source (triggered users only).
    Columns = all joint combinations of (V_i)_{i != source_idx}, represented as
              "|"-joined strings, with nan-containing combinations sorted last.

    Args:
        variant_count_df: DataFrame with columns [variant_col, count_col].
        source_idx: Index of the source experiment j.
        variant_col: Column name for the variant tuple column.
        count_col: Column name for the count column.
        nan_value: Sentinel string for "not triggered".

    Returns:
        count_array: 2D ndarray (n_source_arms x n_joint_combos).
        row_labels: Non-nan arm values of the source experiment.
        col_labels: Joint combination labels, nan-containing last.
    """
    df = variant_count_df.copy()
    n_expts = len(df[variant_col].iloc[0])
    other_indices = [i for i in range(n_expts) if i != source_idx]

    df["_source"] = df[variant_col].map(lambda v: v[source_idx])
    df["_other"] = df[variant_col].map(
        lambda v: "|".join(str(v[i]) for i in other_indices)
    )

    aggregated = df.groupby(["_source", "_other"])[count_col].sum().reset_index()

    def _sort_nan_last(vals):
        return sorted(vals, key=lambda v: (1 if nan_value in v else 0, v))

    source_arms = _sort_nan_last([a for a in aggregated["_source"].unique() if a != nan_value])
    col_labels = _sort_nan_last(list(aggregated["_other"].unique()))

    row_map = {v: i for i, v in enumerate(source_arms)}
    col_map = {k: i for i, k in enumerate(col_labels)}
    count_array = np.zeros((len(source_arms), len(col_labels)), dtype=float)

    for _, row in aggregated.iterrows():
        src = row["_source"]
        other = row["_other"]
        if src in row_map:
            count_array[row_map[src], col_map[other]] = row[count_col]

    return count_array, source_arms, col_labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_trigger_invariance(
    variant_count_df: pd.DataFrame,
    variant_col: str = VARIANT_COL,
    count_col: str = VARIANT_COUNT_COL,
    nan_value: str = CATEG_NAN_VALUE,
    alpha: float = 0.001,
    min_cramers_v: float = 0.01,
) -> "ArmTriggerInvarianceResult":
    """Run the MEA Arm-Trigger Invariance check.

    Runs K joint tests (default) — one per source experiment j, testing whether
    the source arm shifts the joint distribution of all other experiments' variants.
    Bonferroni correction: alpha / K.

    Also runs K(K-1) pairwise tests for drill-down (stored in result.by_pair),
    each targeting one forbidden causal edge A_j -> S_i. For K=2 the two are identical.

    passed, alpha_bonferroni, and min_p_value on the result all reflect the K
    joint tests.

    Args:
        variant_count_df: DataFrame with columns [variant_col, count_col].
        variant_col: Column holding variant tuples (one element per experiment).
        count_col: Column holding counts.
        nan_value: Sentinel string for untriggered variants.
        alpha: Nominal significance level; Bonferroni threshold = alpha / K.
        min_cramers_v: Minimum effect size for flagging.

    Returns:
        ArmTriggerInvarianceResult with K joint results, K*(K-1) pairwise results,
        and overall pass/fail.
    """
    sample_variant = variant_count_df[variant_col].iloc[0]
    n_expts = len(sample_variant)
    n_pairs = n_expts * (n_expts - 1)
    alpha_bonferroni_source = alpha / n_expts if n_expts > 0 else alpha
    alpha_bonferroni_pair = alpha / n_pairs if n_pairs > 0 else alpha  # used for pair passed flag

    # --- K joint tests (primary) ---
    by_source = {}
    for source_idx in range(n_expts):
        count_array, row_labels, col_labels = _build_source_joint_table(
            variant_count_df,
            source_idx=source_idx,
            variant_col=variant_col,
            count_col=count_col,
            nan_value=nan_value,
        )
        test_result = count_independence_test(count_array)
        passed = not (test_result.p_value < alpha_bonferroni_source and test_result.cramers_v > min_cramers_v)
        source_result = SourceTriggerInvarianceResult(
            source_arm_index=source_idx,
            test_result=test_result,
            passed=passed,
            alpha=alpha,
            alpha_bonferroni=alpha_bonferroni_source,
            min_cramers_v=min_cramers_v,
            row_labels=row_labels,
            col_labels=col_labels,
        )
        by_source[source_idx] = source_result
        print(source_result)

    # --- K(K-1) pairwise tests (drill-down) ---
    by_pair = {}
    for source_idx in range(n_expts):
        for target_idx in range(n_expts):
            if source_idx == target_idx:
                continue

            count_array, row_labels, col_labels = _build_variant_count_array(
                variant_count_df,
                expt_a_index=source_idx,
                expt_b_index=target_idx,
                variant_col=variant_col,
                count_col=count_col,
                nan_value=nan_value,
            )
            # Strip nan row: condition on source being triggered (V_source != nan).
            # Keep nan column: V_target = nan is the trigger-contamination signal.
            row_end = -1 if row_labels and row_labels[-1] == nan_value else None
            table = count_array[:row_end, :]
            row_lbls = row_labels[:row_end]

            test_result = count_independence_test(table)
            passed = not (test_result.p_value < alpha_bonferroni_pair and test_result.cramers_v > min_cramers_v)

            pair_result = PairTriggerInvarianceResult(
                source_arm_index=source_idx,
                target_trigger_index=target_idx,
                test_result=test_result,
                passed=passed,
                alpha=alpha,
                n_pairs=n_pairs,
                min_cramers_v=min_cramers_v,
                row_labels=row_lbls,
                col_labels=col_labels,
            )
            by_pair[(source_idx, target_idx)] = pair_result

    return ArmTriggerInvarianceResult(
        by_source=by_source,
        by_pair=by_pair,
        passed_by_source=all(r.passed for r in by_source.values()),
        alpha=alpha,
        min_cramers_v=min_cramers_v,
    )

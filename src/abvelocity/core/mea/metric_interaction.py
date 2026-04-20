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
MEA metric interaction diagnostic — trigger-state-stratified.

Detects whether experiments interfere with each other's effects on a metric.
Metric additivity is NOT a MEA assumption: MEA causal estimates are valid
regardless of whether the experiments interact. This module is a post-hoc
diagnostic that answers a different question: did the experiments genuinely
amplify or attenuate each other's effects?

--- Testing plan ---

For each trigger state with >= 2 triggered experiments, run a k-way ANOVA
interaction test on all triggered arms within that exact stratum, where k is
the number of triggered experiments in that stratum. Each trigger state is
tested independently.

The k-way test checks whether cell means fit a purely additive main-effects
model. Rejection means some combination of 2-way, 3-way, ..., k-way
interactions is present.

Example for k=3 (experiments 0, 1, 2):
  - R110 (True, True, False):  2D test on dims (0, 1) — primary flag.
  - R101 (True, False, True):  2D test on dims (0, 2) — primary flag.
  - R011 (False, True, True):  2D test on dims (1, 2) — primary flag.
  - R111 (True, True, True):   3D test on dims (0, 1, 2) — primary flag.

Why stratify by trigger state?
  Pooling trigger states (e.g. pooling R110 and R111 for pair (0,1)) mixes
  two behaviorally distinct populations. This can produce Simpson's paradox
  in the interaction test. Testing within each exact trigger state gives a
  clean, unambiguous result.

Why k-way rather than only pairwise?
  The k-way test within a stratum is more powerful: it detects any interaction
  structure (including higher-order terms) in a single test. Pairwise tests
  marginalize over the remaining dimensions, which can both miss and inflate
  certain interaction patterns.

Stratification and the linear additivity assumption:
  By testing within each trigger state separately, we allow the additive
  baseline and the effect magnitudes to differ across strata. This is
  analogous to an interaction-with-stratum term in a regression model:
  we are not assuming that the linear structure is homogeneous across
  trigger states -- only that it holds within each one. This makes the
  test robust to behavioral heterogeneity across triggering populations.

Combining rule: flagged if any trigger state has p < alpha.

--- Input data format ---

All public functions take a DataFrame with a `variant_col` column whose values
are tuples of variant strings, one element per experiment, plus per-cell
summary statistics: SAMPLE_COUNT_COL, MEAN_COL, SD_COL, SUM_COL, SUM_SQ_COL.

Public API:
    check_metric_interaction(metric_stats_df, ...)
        -> dict with:
            "interaction_result": MetricInteractionResult
                .by_trigger_state  -- dict[ts] -> MetricTriggerStateTestResult
                    .test_result   -- k-dim ContinuousIndependenceTestResult
                .flagged           -- any stratum flags -> True
            "by_trigger_state": dict[ts] -> dict with
                "pair_heatmaps": dict[(dim_a, dim_b)] -> {
                    "means_fig", "residuals_fig", "row_labels", "col_labels"
                }
                (only populated for strata where len(triggered_dims) == 2)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from abvelocity.core.param.constants import (
    CATEG_NAN_VALUE,
    MEAN_COL,
    SAMPLE_COUNT_COL,
    SD_COL,
    SUM_COL,
    SUM_SQ_COL,
    VARIANT_COL,
)
from abvelocity.core.stats.continuous_independence_test import (
    ContinuousIndependenceTestResult,
    continuous_independence_test,
)
from abvelocity.core.utils.plot_2d_heatmap import plot_2d_heatmap


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricTriggerStateTestResult:
    """Result of the metric interaction test for one trigger state stratum.

    For k triggered experiments within this stratum, runs a k-way ANOVA
    interaction test.

    Attributes:
        trigger_state: Tuple of booleans identifying the stratum.
        triggered_dims: Experiment indices that are True in trigger_state.
        test_result: k-way ContinuousIndependenceTestResult (primary).
        flagged: True if k-way test p < alpha.
        alpha: p-value threshold used.
        arm_labels: Arm-name lists, one list per triggered dim.
    """

    trigger_state: tuple
    triggered_dims: List[int]
    test_result: ContinuousIndependenceTestResult
    flagged: bool
    alpha: float
    arm_labels: List[List[str]]

    def __str__(self) -> str:
        ts_str = "(" + ", ".join(str(b) for b in self.trigger_state) + ")"
        status = "FLAG" if self.flagged else "PASS"
        r = self.test_result
        return f"[{status}] TriggerState {ts_str} | dims={self.triggered_dims} | " f"F={r.f_value:.2f}, p={r.p_value:.2e}, dof={r.dof}, N={int(r.n_total):,}"


@dataclass
class MetricInteractionResult:
    """Combined metric interaction result across all trigger state strata.

    For each trigger state with >= 2 triggered experiments, runs a k-way
    ANOVA interaction test within that exact stratum. Each stratum is tested
    independently.

    Combining rule: flagged = any stratum is flagged (p < alpha).
    This is conservative — any stratum showing interaction is noteworthy.

    Attributes:
        by_trigger_state: Dict keyed by trigger state tuple with
            MetricTriggerStateTestResult for each stratum tested.
        flagged: True if any stratum flags.
        alpha: p-value threshold used.
    """

    by_trigger_state: Dict[tuple, MetricTriggerStateTestResult]
    flagged: bool
    alpha: float

    def __str__(self) -> str:
        status = "FLAG" if self.flagged else "PASS"
        lines = [f"[{status}] Metric interaction — {len(self.by_trigger_state)} trigger state(s) tested"]
        for ts_result in self.by_trigger_state.values():
            lines.append("  " + str(ts_result))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _build_cell_arrays_for_trigger_state(
    metric_stats_df: pd.DataFrame,
    trigger_state_value: tuple,
    project_dims: List[int],
    variant_col: str = VARIANT_COL,
    nan_value: str = CATEG_NAN_VALUE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]]]:
    """Build k-dim variant combination cell summary arrays for one trigger state stratum.

    Called by: `check_metric_interaction` — once per trigger state for the k-dim
    primary test.

    Filters to rows matching trigger_state_value exactly (only members who
    triggered the exact set of experiments defined by the trigger state), then
    projects to the specified experiment dimensions. Each cell in the output
    arrays corresponds to one variant combination — a specific tuple of arms
    across the projected experiments.

    Args:
        metric_stats_df: DataFrame with columns [variant_col, SAMPLE_COUNT_COL,
            MEAN_COL, SD_COL, SUM_COL, SUM_SQ_COL] for a single metric.
        trigger_state_value: Tuple of bools — exact trigger state to filter to.
        project_dims: Experiment indices to include. E.g. [0, 1] for a 2D test,
            [0, 1, 2] for a 3D test. Must be triggered in trigger_state_value.
        variant_col: Column name for the variant tuple column.
        nan_value: Sentinel string for untriggered variants.

    Returns:
        cell_means_kdim: k-dim array of per-variant-combination cell means.
        cell_sds_kdim: k-dim array of per-variant-combination cell SDs.
        cell_counts_kdim: k-dim array of per-variant-combination cell unit counts.
        arm_labels: Arm-name lists, one per projected experiment dimension.
    """

    def _matches(v):
        return all((v[i] != nan_value) == b for i, b in enumerate(trigger_state_value))

    df = metric_stats_df[metric_stats_df[variant_col].map(_matches)].copy()

    dim_cols = []
    for k in project_dims:
        col = f"_dim{k}"
        df[col] = df[variant_col].map(lambda v, k=k: v[k])
        dim_cols.append(col)

    aggregated = df.groupby(dim_cols)[[SAMPLE_COUNT_COL, SUM_COL, SUM_SQ_COL]].sum().reset_index()
    total_n = aggregated[SAMPLE_COUNT_COL]
    total_sum = aggregated[SUM_COL]
    total_sum_sq = aggregated[SUM_SQ_COL]
    aggregated[MEAN_COL] = total_sum / total_n
    ss_within = total_sum_sq - total_n * aggregated[MEAN_COL] ** 2
    aggregated[SD_COL] = np.sqrt(np.clip(ss_within / np.maximum(total_n - 1, 1), 0.0, None))

    arm_labels = [sorted(aggregated[col].unique()) for col in dim_cols]
    shape = tuple(len(lbl) for lbl in arm_labels)
    cell_means_kdim = np.full(shape, np.nan)
    cell_sds_kdim = np.full(shape, np.nan)
    cell_counts_kdim = np.zeros(shape)
    idx_maps = [{v: i for i, v in enumerate(lbls)} for lbls in arm_labels]

    for _, row in aggregated.iterrows():
        idx = tuple(idx_maps[pos][row[col]] for pos, col in enumerate(dim_cols))
        cell_means_kdim[idx] = row[MEAN_COL]
        cell_sds_kdim[idx] = row[SD_COL]
        cell_counts_kdim[idx] = row[SAMPLE_COUNT_COL]

    return cell_means_kdim, cell_sds_kdim, cell_counts_kdim, arm_labels


def _make_pair_viz(
    cell_means_2d: np.ndarray,
    cell_sds_2d: np.ndarray,
    cell_counts_2d: np.ndarray,
    arm_labels: List[List[str]],
    test_result: ContinuousIndependenceTestResult,
    name_a: str,
    name_b: str,
    ts_str: str,
    title_prefix: str,
    interaction_clim: float,
) -> Dict:
    """Build means and interaction residuals heatmaps for one 2D variant combination cell grid.

    Called by: `check_metric_interaction` — once per (trigger_state, pair) to
    visualize the per-cell means and interaction residuals for a pair of experiments.

    Inputs are 2D arrays indexed by (arm_expt_a, arm_expt_b), where each cell
    is a specific variant combination within the given trigger state stratum.
    """
    stat_str = f"F={test_result.f_value:.2f}, p={test_result.p_value:.2e}, dof={test_result.dof} | {ts_str}"
    sigma = test_result.sigma_within_cell if test_result.sigma_within_cell > 0 else 1.0
    row_labels, col_labels = arm_labels[0], arm_labels[1]

    valid_mask = np.isfinite(cell_means_2d) & (cell_counts_2d > 0)
    denom = float(np.nansum(np.where(valid_mask, cell_counts_2d, 0.0)))
    grand_mean = float(np.nansum(np.where(valid_mask, cell_counts_2d * cell_means_2d, 0.0))) / denom if denom > 0 else 0.0
    standardized_means = np.where(valid_mask, (cell_means_2d - grand_mean) / sigma, np.nan)

    n_rows, n_cols = cell_means_2d.shape
    means_cell_texts = []
    for ri in range(n_rows):
        row_text = []
        for ci in range(n_cols):
            if valid_mask[ri, ci]:
                row_text.append(f"{cell_means_2d[ri, ci]:.3g}<br>±{cell_sds_2d[ri, ci]:.3g}")
            else:
                row_text.append("N/A")
        means_cell_texts.append(row_text)

    means_fig = plot_2d_heatmap(
        values=standardized_means,
        row_labels=row_labels,
        col_labels=col_labels,
        cell_texts=means_cell_texts,
        axis_names=(name_a, name_b),
        colorbar_title="(mean − grand mean) / σ",
        symmetric=True,
        annotation=stat_str,
        title=f"{title_prefix}Cell means ({name_a} × {name_b}) — trigger state {ts_str}",
    )

    normalized_residuals = test_result.cell_residuals / sigma
    residuals_fig = plot_2d_heatmap(
        values=normalized_residuals,
        row_labels=row_labels,
        col_labels=col_labels,
        axis_names=(name_a, name_b),
        colorbar_title="interaction<br>residual (σ)",
        symmetric=True,
        clim=interaction_clim,
        annotation=stat_str,
        title=f"{title_prefix}Interaction residuals ({name_a} × {name_b}) — trigger state {ts_str}",
    )

    return {
        "means_fig": means_fig,
        "residuals_fig": residuals_fig,
        "row_labels": row_labels,
        "col_labels": col_labels,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_metric_interaction(
    metric_stats_df: pd.DataFrame,
    expt_names: Optional[List[str]] = None,
    metric_name: str = "",
    variant_col: str = VARIANT_COL,
    nan_value: str = CATEG_NAN_VALUE,
    interaction_clim: float = 2.0,
    alpha: float = 0.05,
) -> Dict:
    """Detect metric interaction among experiments, stratified by trigger state.

    For each trigger state with >= 2 triggered experiments, runs a k-way ANOVA
    interaction test on the triggered arms within that exact stratum. This tests
    whether cell means fit a purely additive main-effects model -- i.e. whether
    the experiments' effects on this metric are independent of each other within
    each triggered population.

    Combining rule: flagged if any stratum has p < alpha.

    Interpretation per stratum:
      - p large: effects are additive within this population. MEA was not
        strictly necessary for this metric in this stratum.
      - p small: real interaction (synergy or antagonism). MEA decomposition
        is important; report the interaction alongside main effects.

    Stratification note:
      By testing within each trigger state independently, the additive model
      is fit separately per stratum. This allows the baseline and effect sizes
      to differ across trigger states -- analogous to a stratum-specific
      regression model. We only assume additivity within each stratum, not
      homogeneity of effects across strata.

    Args:
        metric_stats_df: DataFrame with columns [variant_col, SAMPLE_COUNT_COL,
            MEAN_COL, SD_COL, SUM_COL, SUM_SQ_COL] for a single metric.
        expt_names: Optional list of experiment name strings, indexed by
            experiment position. Used for axis labels in heatmaps.
        metric_name: Optional metric name for plot titles.
        variant_col: Column name for the variant tuple column.
        nan_value: Sentinel string for untriggered variants.
        interaction_clim: Fixed colorbar half-range for interaction residual
            heatmaps, in sigma_within_cell units. Default 2.0.
        alpha: p-value threshold for flagging. Default 0.05.

    Returns:
        Dict with keys:
            "interaction_result": MetricInteractionResult
                .by_trigger_state  -- dict[ts] -> MetricTriggerStateTestResult
                    .test_result   -- k-way primary ContinuousIndependenceTestResult
                .flagged           -- True if any stratum flags
            "by_trigger_state": dict[ts] -> dict with
                "pair_heatmaps": dict[(dim_a, dim_b)] -> {
                    "means_fig", "residuals_fig", "row_labels", "col_labels"
                }
                (only populated for strata where len(triggered_dims) == 2)
    """

    def _trigger_state(v):
        return tuple(val != nan_value for val in v)

    trigger_states = sorted(set(metric_stats_df[variant_col].map(_trigger_state)))
    title_prefix = f"{metric_name} — " if metric_name else ""

    by_trigger_state_result: Dict[tuple, MetricTriggerStateTestResult] = {}
    by_trigger_state_viz: Dict[tuple, Dict] = {}
    overall_flagged = False

    for ts in trigger_states:
        triggered_dims = [i for i, b in enumerate(ts) if b]
        if len(triggered_dims) < 2:
            continue

        ts_str = "(" + ", ".join(str(b) for b in ts) + ")"

        # k-way primary test on all triggered dims.
        cell_means_kdim, cell_sds_kdim, cell_counts_kdim, arm_labels = _build_cell_arrays_for_trigger_state(
            metric_stats_df,
            trigger_state_value=ts,
            project_dims=triggered_dims,
            variant_col=variant_col,
            nan_value=nan_value,
        )
        test_result = continuous_independence_test(
            cell_means=cell_means_kdim,
            cell_sds=cell_sds_kdim,
            cell_counts=cell_counts_kdim,
        )
        flagged = test_result.p_value < alpha
        if flagged:
            overall_flagged = True

        # For K=2 strata: generate 2D heatmaps (means + interaction residuals).
        # For K>=3 strata: the K-dim test result is the output; no per-pair heatmaps.
        pair_heatmaps: Dict[Tuple[int, int], Dict] = {}
        if len(triggered_dims) == 2:
            dim_a, dim_b = triggered_dims
            name_a = expt_names[dim_a] if expt_names and dim_a < len(expt_names) else f"Expt {dim_a}"
            name_b = expt_names[dim_b] if expt_names and dim_b < len(expt_names) else f"Expt {dim_b}"
            pair_heatmaps[(dim_a, dim_b)] = _make_pair_viz(
                cell_means_2d=cell_means_kdim,
                cell_sds_2d=cell_sds_kdim,
                cell_counts_2d=cell_counts_kdim,
                arm_labels=arm_labels,
                test_result=test_result,
                name_a=name_a,
                name_b=name_b,
                ts_str=ts_str,
                title_prefix=title_prefix,
                interaction_clim=interaction_clim,
            )

        ts_result = MetricTriggerStateTestResult(
            trigger_state=ts,
            triggered_dims=triggered_dims,
            test_result=test_result,
            flagged=flagged,
            alpha=alpha,
            arm_labels=arm_labels,
        )
        by_trigger_state_result[ts] = ts_result
        by_trigger_state_viz[ts] = {"pair_heatmaps": pair_heatmaps}
        print(ts_result)

    # TODO: the any-flag rule has inflated family-wise error when many strata
    # are tested. A future PR should apply multiple-comparison correction across
    # strata (e.g. Bonferroni or Benjamini-Hochberg on the per-stratum p-values).
    interaction_result = MetricInteractionResult(
        by_trigger_state=by_trigger_state_result,
        flagged=overall_flagged,
        alpha=alpha,
    )

    return {
        "interaction_result": interaction_result,
        "by_trigger_state": by_trigger_state_viz,
    }

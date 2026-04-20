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
Multi-way ANOVA interaction test for continuous outcomes using cell summary statistics.

--- Terminology ---

A **cell** is a specific variant combination: a k-tuple of arms, one per
experiment. For a 2-experiment MEA, cells are things like ("control", "treat")
or ("treat", "control"). For k=3 they are ("control", "treat", "nan"), etc.
Each cell has n_cell unit observations, a cell mean, and a cell SD.

The arrays passed to this module are indexed by these cells: axis i corresponds
to experiment i, and index j along that axis corresponds to the j-th arm of
that experiment. The array shape is (p_1, ..., p_k) where k is the number of
experiments and p_i is the number of arms in experiment i.

--- What is tested ---

Tests whether cell means factor additively (no interaction), using only
per-cell summary statistics (mean, SD, count) -- no raw data required.
Cell summaries can be pre-computed via SQL (e.g. Trino/Spark GROUP BY).

--- Test statistic ---

    SS_interaction     = sum over cells:  n_cell * (mu_cell - mu_fitted_cell)^2
    sigma2_within_cell = sum over cells: (n_cell - 1) * sd_cell^2
                         -------------------------------------------------
                         sum over cells: (n_cell - 1)
    F = (SS_interaction / dof_interaction) / sigma2_within_cell
      ~ F(dof_interaction, dof_within)

where mu_fitted_cell is the cell mean predicted by the additive (no-interaction)
null model, sigma2_within_cell is the pooled variance of individual unit
observations within each variant combination cell (not across cells or strata),
and dof_within = sum over cells of (n_cell - 1).

--- Degrees of freedom ---

The additive model has 1 + sum_i(p_i - 1) = 1 + sum(p_i) - k free parameters:
one grand mean mu_0, and (p_i - 1) free main-effect parameters per experiment
(first arm is the reference, so mu_{i,0} = 0 by convention).
There are prod(p_i) cells to fit, so the model is identifiable only when
prod(p_i) > 1 + sum(p_i) - k, i.e. cells outnumber parameters.

    dof_interaction = prod(p_i) - (1 + sum(p_i) - k)
                    = prod(p_i) - sum(p_i) + k - 1

For two-way: dof = (p_1-1)(p_2-1), matching standard two-way ANOVA interaction dof.

Applicable to any continuous metric where per-cell (mean, sd, n) is available.
Cells with n <= 1 are excluded from the within-cell variance estimate.

Called by: `mea.metric_interaction.check_metric_interaction` — one call per
(metric, trigger_state) stratum within a MEA analysis.

References:
    - Montgomery, D.C. (2017). Design and Analysis of Experiments (9th ed.).
      Wiley. https://www.wiley.com/en-us/9781119492443
      [Ch. 5-7 for factorial designs and interaction tests]

    - Kutner, M.H., Nachtsheim, C.J., Neter, J. & Li, W. (2005).
      Applied Linear Statistical Models (5th ed.). McGraw-Hill.
      https://www.mheducation.com/highered/product/applied-linear-statistical-models-kutner-nachtsheim/M9780073108742.html
      [Ch. 19-23 for multi-way ANOVA and interaction decomposition]

Public API:
    continuous_independence_test(cell_means, cell_sds, cell_counts, excluded_cells)
        -> ContinuousIndependenceTestResult

    Utilities (in stats.interaction_statistics):
        fit_additive_kdim_cells(cell_means, cell_counts) -> fitted_cell_means
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import f as f_dist

from abvelocity.core.stats.interaction_statistics import fit_additive_kdim_cells


@dataclass
class ContinuousIndependenceTestResult:
    """Result of a multi-dimensional ANOVA interaction test on continuous cell summaries.

    Attributes:
        f_value: F statistic = MS_interaction / MS_within
            = (SS_interaction / dof_interaction) / sigma2_within_cell.
        p_value: p-value from F(dof_interaction, dof_within) distribution.
        dof: Degrees of freedom for the interaction term (dof_interaction).
        dof_within: Degrees of freedom for the within-cell variance estimate
            = sum over cells of (n_cell - 1).
        n_cells: Number of non-empty variant combination cells used in the test.
        n_total: Total number of unit observations across all valid cells.
        sigma_within_cell: Pooled standard deviation of individual unit
            observations within each variant combination cell (i.e. within
            each specific arm tuple like ("control", "treat")). Pooled across
            all cells by weighting each cell's SD by (n_cell - 1).
        ss_interaction: Interaction sum of squares.
        fitted_means: Multi-dim array of additive null-model fitted cell means.
        cell_residuals: Multi-dim array of (observed cell mean - fitted cell mean).
    """

    f_value: float
    p_value: float
    dof: int
    dof_within: float
    n_cells: int
    n_total: float
    sigma_within_cell: float
    ss_interaction: float
    fitted_means: Optional[np.ndarray] = None
    cell_residuals: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return (
            f"ContinuousIndependenceTest | "
            f"F={self.f_value:.2f}, p={self.p_value:.2e}, "
            f"dof={self.dof}, dof_within={int(self.dof_within)}, N={int(self.n_total):,}, "
            f"sigma_within_cell={self.sigma_within_cell:.4f}"
        )


def continuous_independence_test(
    cell_means: np.ndarray,
    cell_sds: np.ndarray,
    cell_counts: np.ndarray,
    excluded_cells: Optional[List[Tuple]] = None,
) -> ContinuousIndependenceTestResult:
    """Multi-way ANOVA interaction test for continuous outcomes from variant combination cell summaries.

    A cell is a specific variant combination — a k-tuple of arms, one per
    experiment. Inputs are k-dimensional arrays indexed by (arm_expt0, arm_expt1,
    ..., arm_expt_{k-1}). Each cell holds the summary statistics (mean, SD, count)
    for all members with that exact variant combination.

    Tests H0: no interaction -- cell means follow an additive main-effects model.
    Cell summaries can be computed upstream via SQL (Trino / Spark GROUP BY).

    Args:
        cell_means: k-dim array of per-variant-combination cell means.
            Shape (p_1, p_2, ..., p_k) where p_i = number of arms in experiment i.
            Set to NaN for empty cells (no members with that variant combination).
        cell_sds: k-dim array of per-variant-combination cell standard deviations
            (SD of individual unit observations within each cell), same shape.
            Used to estimate pooled within-cell variance. NaN for empty cells.
        cell_counts: k-dim array of unit counts per variant combination cell,
            same shape. Zero for empty cells.
        excluded_cells: Optional list of cell index tuples to exclude
            (e.g. structural zeros). Negative indices supported.

    Returns:
        ContinuousIndependenceTestResult with test statistics.

    Raises:
        ValueError: If fewer than 2 valid cells or within-cell variance
            cannot be estimated.
    """
    cell_means = np.array(cell_means, dtype=float)
    cell_sds = np.array(cell_sds, dtype=float)
    cell_counts = np.array(cell_counts, dtype=float)
    shape = cell_means.shape
    num_dims = cell_means.ndim

    if excluded_cells:
        for cell in excluded_cells:
            idx = tuple(s % d for s, d in zip(cell, shape))
            cell_counts[idx] = 0.0
            cell_means[idx] = np.nan
            cell_sds[idx] = np.nan

    valid = (cell_counts > 0) & np.isfinite(cell_means)
    n_valid_cells = int(valid.sum())
    if n_valid_cells < 2:
        raise ValueError(f"Only {n_valid_cells} valid variant combination cell(s) -- need at least 2.")

    n_total = float(cell_counts[valid].sum())

    # sigma2_within_cell: pooled variance of individual unit observations
    # within each variant combination cell. Each cell contributes
    # (n_cell - 1) * sd_cell^2 to the numerator. Cells with n <= 1 are
    # excluded (no within-cell variance estimable from a single observation).
    has_var = valid & (cell_counts > 1) & np.isfinite(cell_sds) & (cell_sds >= 0)
    if not has_var.any():
        raise ValueError("No variant combination cells with n > 1 and valid SD -- cannot estimate within-cell variance.")

    dof_within_cell = float((cell_counts[has_var] - 1).sum())
    ss_within_cell = float(((cell_counts[has_var] - 1) * cell_sds[has_var] ** 2).sum())
    sigma2_within_cell = ss_within_cell / dof_within_cell
    if not np.isfinite(sigma2_within_cell) or sigma2_within_cell <= 0:
        raise ValueError("Within-cell variance estimate is non-positive or undefined.")

    fitted_cell_means = fit_additive_kdim_cells(cell_means, cell_counts)
    cell_residuals = np.where(valid, cell_means - fitted_cell_means, np.nan)
    ss_interaction = float(np.nansum(cell_counts * np.where(valid, cell_residuals**2, 0.0)))

    dof_interaction = int(np.prod(shape) - sum(shape) + num_dims - 1)
    if excluded_cells:
        dof_interaction = max(dof_interaction - len(excluded_cells), 1)
    dof_interaction = max(dof_interaction, 1)

    f_value = (ss_interaction / dof_interaction) / sigma2_within_cell
    p_value = float(f_dist.sf(f_value, dof_interaction, dof_within_cell))

    return ContinuousIndependenceTestResult(
        f_value=f_value,
        p_value=p_value,
        dof=dof_interaction,
        dof_within=dof_within_cell,
        n_cells=n_valid_cells,
        n_total=n_total,
        sigma_within_cell=float(np.sqrt(sigma2_within_cell)),
        ss_interaction=ss_interaction,
        fitted_means=fitted_cell_means,
        cell_residuals=cell_residuals,
    )

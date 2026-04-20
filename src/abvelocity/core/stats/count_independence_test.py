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
Chi-squared test of mutual independence for multi-dimensional count arrays.

--- Terminology ---

A **cell** is a specific variant combination: a k-tuple of arms, one per
experiment. For a 2-experiment MEA, cells are things like ("control", "treat").
Each cell holds the count of members with that exact variant combination.

The count array passed to this module is indexed by these cells: axis i
corresponds to experiment i, and index j along that axis corresponds to the
j-th arm of that experiment. The array shape is (p_1, ..., p_k) where k is
the number of experiments and p_i is the number of arms in experiment i.

--- What is tested ---

Applicable to count data only. Each cell holds a non-negative integer count.
For continuous outcomes (means + SDs per cell) use `continuous_independence_test`.

The test checks whether the joint distribution across variant combinations
factorizes into the product of the marginal arm distributions:

    P(v1, ..., vk) = prod_i P(v_i)

Expected counts under independence are:

    E[n_{j1,...,jk}] = N * prod_i f_i(j_i)

where f_i(j_i) = marginal fraction of arm j_i in experiment i.

--- Degrees of freedom ---

    dof = prod(p_i) - sum(p_i) + k - 1

Same formula as the continuous test: total cells minus the number of free
parameters in the independence model (one marginal probability per arm minus
one constraint per dimension).

One or more cells known to be impossible by design can be excluded via
`excluded_cells`. Each excluded cell is zeroed before computing marginals
and marked NaN in the cell_residuals output. The degrees of freedom are reduced
by the number of excluded cells.

Called by: `mea.assumption_check.check_mea_orthogonality` — one call per
trigger state stratum to test assignment independence across experiments.

References:
    - Pearson, K. (1900). "On the criterion that a given system of deviations
      from the probable in the case of a correlated system of variables is such
      that it can be reasonably supposed to have arisen from random sampling."
      Philosophical Magazine, 50(302), 157-175.
      https://doi.org/10.1080/14786440009463897

    - Agresti, A. (2002). Categorical Data Analysis (2nd ed.). Wiley.
      https://doi.org/10.1002/0471249688
      [Ch. 2-3 for 2-way; Ch. 9 for multi-way independence]

    - Cramer, H. (1946). Mathematical Methods of Statistics.
      Princeton University Press.
      [Sec.21.4 for Cramer's V]

Public API:
    build_count_array(count_df, dim_cols, count_col, ...)
        -> (count_array, arm_labels)
        Builds a k-dim count array from a flat DataFrame of variant combination counts.

    count_array_for_trigger_state(variants, trigger_state)
        -> count_array
        Builds a k-dim count array from a unit-level (n_units, k) int array,
        filtered to one exact trigger state. Useful for simulation and testing.

    count_independence_test(count_array, excluded_cells)
        -> CountIndependenceTestResult

    Utilities (in stats.chi2):
        chi2_statistic(arr, excluded_cells) -> (chi2_val, expected_cell_counts, cell_residuals)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import chi2 as chi2_dist

from abvelocity.core.stats.chi2 import chi2_statistic


@dataclass
class CountIndependenceTestResult:
    """Result of a multi-dimensional chi-squared independence test on count data.

    Attributes:
        chi2_value: Chi-squared test statistic.
        p_value: p-value from asymptotic chi-squared distribution.
        dof: Degrees of freedom.
        n_total: Total unit count (sum of all non-excluded cells).
        cramers_v: Generalised Cramer's V effect size.
            V = sqrt(chi2_value / (n_total * min(p_i - 1))). V = 0 -> perfect independence.
        count_array: The original multi-dimensional count array.
        expected_cell_counts: Expected counts under mutual independence.
        cell_residuals: Pearson residuals (O - E) / sqrt(E); excluded cells = NaN.
    """

    chi2_value: float
    p_value: float
    dof: int
    n_total: float
    cramers_v: float
    count_array: Optional[np.ndarray] = None
    expected_cell_counts: Optional[np.ndarray] = None
    cell_residuals: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return (
            f"CountIndependenceTest | " f"chi2={self.chi2_value:.2f}, p={self.p_value:.2e}, " f"V={self.cramers_v:.4f}, N={int(self.n_total):,}, dof={self.dof}"
        )


def build_count_array(
    count_df,
    dim_cols: List[str],
    count_col: str,
    nan_last: bool = True,
    nan_value: str = "nan",
) -> Tuple[np.ndarray, List[List[str]]]:
    """Build a multi-dimensional count array from a flat DataFrame.

    Args:
        count_df: DataFrame with one row per cell combination.
        dim_cols: Ordered list of column names, one per dimension.
        count_col: Column name holding the count for each cell.
        nan_last: If True, sort so nan_value appears last in each dimension.
        nan_value: The string sentinel for the "not observed" category.

    Returns:
        count_array: ndarray of shape (p_1, ..., p_k) where p_i is the number
            of arms (distinct values) in dimension i.
        arm_labels: List of arm-name lists, one per dimension, in array index order.
    """

    def _sort_labels(vals):
        non_nan = sorted(v for v in vals if v != nan_value)
        return non_nan + ([nan_value] if nan_value in vals else [])

    arm_labels = [_sort_labels(count_df[col].unique()) if nan_last else sorted(count_df[col].unique()) for col in dim_cols]
    shape = tuple(len(lbl) for lbl in arm_labels)
    count_array = np.zeros(shape, dtype=float)

    idx_maps = [{v: i for i, v in enumerate(lbls)} for lbls in arm_labels]
    for _, row in count_df.iterrows():
        idx = tuple(idx_maps[dim][row[col]] for dim, col in enumerate(dim_cols))
        count_array[idx] = row[count_col]

    return count_array, arm_labels


def count_array_for_trigger_state(
    variants: np.ndarray,
    trigger_state: tuple,
) -> np.ndarray:
    """Build a k-dim count array from a unit-level variants array for one trigger state.

    Useful for simulation and testing, where data exists as a unit-level array
    rather than a pre-aggregated DataFrame.

    Args:
        variants: (n_units, k) int array. 0=ctrl, 1=treat, -1=not triggered.
            One row per unit, one column per experiment.
        trigger_state: Tuple of k booleans. Only units whose trigger pattern
            matches exactly are included: triggered experiments must have a
            real arm (>= 0), non-triggered must be -1.

    Returns:
        count_array: ndarray of shape (2,) * num_triggered. Each axis corresponds
            to one triggered experiment (in the order they appear in trigger_state).
            count_array[0, 1] = count of units with ctrl in the first triggered
            experiment and treat in the second.
    """
    triggered_dims = [i for i, t in enumerate(trigger_state) if t]
    mask = np.ones(len(variants), dtype=bool)
    for i, t in enumerate(trigger_state):
        mask &= (variants[:, i] >= 0) if t else (variants[:, i] < 0)
    sub = variants[mask][:, triggered_dims]
    arr = np.zeros((2,) * len(triggered_dims), dtype=float)
    for row in sub:
        arr[tuple(row)] += 1
    return arr


def count_independence_test(
    count_array: np.ndarray,
    excluded_cells: Optional[List[Tuple]] = None,
) -> CountIndependenceTestResult:
    """Chi-squared test of mutual independence for multi-dimensional count data.

    Under mutual independence, expected counts equal the outer product of
    marginal distributions scaled by N.

    See module docstring for full references.

    Args:
        count_array: Multi-dimensional ndarray of non-negative counts.
            Shape (p_1, p_2, ..., p_k) where p_i is the number of arms
            (categories) in experiment i.
        excluded_cells: Optional list of cell index tuples to exclude before
            testing (e.g. cells known to be impossible by design). Each cell
            is zeroed, its residual is set to NaN, and dof is reduced by one
            per excluded cell. Negative indices are supported (e.g. [(-1, -1)]).

    Returns:
        CountIndependenceTestResult with test statistics and arrays.
    """
    arr = count_array.astype(float).copy()
    shape = arr.shape
    num_dims = arr.ndim

    if excluded_cells is not None:
        for cell in excluded_cells:
            idx = tuple(s % d for s, d in zip(cell, shape))
            arr[idx] = 0.0

    chi2_val, expected_cell_counts, cell_residuals = chi2_statistic(arr, excluded_cells)

    n_total = float(arr.sum())
    dof = int(np.prod(shape) - sum(shape) + num_dims - 1)
    if excluded_cells is not None:
        dof = max(dof - len(excluded_cells), 1)
    dof = max(dof, 1)

    min_dim = min(s - 1 for s in shape)
    cramers_v = float(np.sqrt(chi2_val / (n_total * min_dim))) if n_total > 0 and min_dim > 0 else 0.0

    p_val = float(1 - chi2_dist.cdf(chi2_val, dof))

    return CountIndependenceTestResult(
        chi2_value=chi2_val,
        p_value=p_val,
        dof=dof,
        n_total=n_total,
        cramers_v=cramers_v,
        count_array=count_array,
        expected_cell_counts=expected_cell_counts,
        cell_residuals=cell_residuals,
    )

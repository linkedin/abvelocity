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
Chi-squared statistic for multi-dimensional count arrays.

Computes the Pearson chi-squared statistic for a k-dimensional count array
under the null hypothesis of mutual independence across all dimensions.

Expected counts under independence are the outer product of marginal
distributions across all axes:

    E[n_{j1,...,jk}] = N * prod_i f_i(j_i)

where f_i(j_i) = marginal fraction of arm j_i in dimension i.

Works for any number of dimensions >= 2. Used by `count_independence_test`.

Public API:
    chi2_statistic(arr, excluded_cells)
        -> (chi2_val, expected_cell_counts, cell_residuals)
"""

from typing import List, Optional, Tuple

import numpy as np


def chi2_statistic(
    arr: np.ndarray,
    excluded_cells: Optional[List[Tuple]] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the Pearson chi-squared statistic for a k-dimensional count array.

    Expected counts are computed from the outer product of marginal distributions
    across all dimensions. Works for any k >= 2.

    Args:
        arr: k-dim count array of non-negative counts. Shape (p_1, ..., p_k).
            Each cell arr[j1,...,jk] = number of observations with index j1
            along axis 0, j2 along axis 1, etc.
        excluded_cells: Optional list of cell index tuples to exclude (zero out)
            before computing marginals. Negative indices are supported
            (e.g. [(-1, -1)] for the last cell along each axis).

    Returns:
        chi2_val: Chi-squared test statistic (sum of squared Pearson residuals).
        expected_cell_counts: Expected count array under mutual independence,
            same shape as arr. Excluded cells are zeroed.
        cell_residuals: Pearson residuals (O - E) / sqrt(E), same shape.
            Excluded cells and cells with zero expected count are NaN.
    """
    num_dims = arr.ndim
    shape = arr.shape

    mask = np.zeros(shape, dtype=bool)
    if excluded_cells is not None:
        arr = arr.copy()
        for cell in excluded_cells:
            idx = tuple(s % d for s, d in zip(cell, shape))
            mask[idx] = True
            arr[idx] = 0.0

    total_n = arr.sum()
    marginals = [arr.sum(axis=tuple(dim for dim in range(num_dims) if dim != axis_dim)) for axis_dim in range(num_dims)]

    expected_cell_counts = marginals[0].copy().astype(float)
    for dim_idx in range(1, num_dims):
        expected_cell_counts = np.multiply.outer(expected_cell_counts, marginals[dim_idx])
    expected_cell_counts = expected_cell_counts / (total_n ** (num_dims - 1))

    if excluded_cells is not None:
        for cell in excluded_cells:
            idx = tuple(s % d for s, d in zip(cell, shape))
            expected_cell_counts[idx] = 0.0

    cell_residuals = np.full(shape, np.nan)
    valid = (expected_cell_counts > 0) & (~mask)
    cell_residuals[valid] = (arr[valid] - expected_cell_counts[valid]) / np.sqrt(expected_cell_counts[valid])
    chi2_val = float(np.nansum(cell_residuals[valid] ** 2))

    return chi2_val, expected_cell_counts, cell_residuals

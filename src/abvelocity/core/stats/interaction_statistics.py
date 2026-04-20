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
Additive main-effects model for k-dimensional cell mean arrays.

Fits the null model:

    mu_fitted(j_1, ..., j_k) = mu_0 + mu_{1,j_1} + mu_{2,j_2} + ... + mu_{k,j_k}

where j_i in {0, ..., p_i-1} is the arm index along dimension i.
Each mu_{i,*} is a (p_i - 1)-vector of free main-effect parameters for
dimension i (arm 0 is the reference; mu_{i,0} = 0 by convention).
Total free parameters: 1 + sum_i(p_i - 1).

Works for any number of dimensions k >= 2 and any shape (p_1, ..., p_k).
Used by `continuous_independence_test` to fit the additive null model.

Public API:
    fit_additive_kdim_cells(cell_means, cell_counts) -> fitted_cell_means
"""

import numpy as np


def fit_additive_kdim_cells(
    cell_means: np.ndarray,
    cell_counts: np.ndarray,
) -> np.ndarray:
    """Fit the additive main-effects null model to k-dimensional cell means via WLS.

    The null model (H0: no interaction) is:

        mu_fitted(j_1, ..., j_k) = mu_0 + mu_{1,j_1} + mu_{2,j_2} + ... + mu_{k,j_k}

    where j_i in {0, ..., p_i-1} is the arm index along dimension i.
    mu_{i,j_i} is a lookup into a length-(p_i - 1) vector of free main-effect
    parameters for dimension i (arm 0 is the reference, mu_{i,0} = 0 by convention).
    mu_0 is the grand mean. Total free parameters: 1 + sum_i(p_i - 1).
    No cross terms — deviations of observed cell means from this fit are the
    interaction residuals that drive the F statistic.

    Fitting method: WLS where each cell is weighted by its unit count. A
    cell mean from n_i observations has variance sigma^2 / n_i, so precision
    is n_i / sigma^2. Since sigma^2 is common (pooled), the WLS weights are
    just n_i. Cell SDs are not needed here -- they enter the F statistic via
    sigma2_within_cell, not via the fit itself.

    Implementation: treatment-coded dummy variables (first arm of each
    dimension as reference), solved via numpy.linalg.lstsq after absorbing
    sqrt(weights) into X and y. Exact WLS, no iteration needed.

    Args:
        cell_means: k-dim array of observed cell means. Shape (p_1, ..., p_k)
            where p_i = number of arms in dimension i.
            NaN = empty cell (no members with this exact variant combination).
        cell_counts: k-dim array of unit counts per cell, same shape.
            Zero = empty cell.

    Returns:
        fitted_cell_means: k-dim array of additive null-model fitted cell means,
            same shape as cell_means.
    """
    shape = cell_means.shape
    num_dims = cell_means.ndim
    n_cells = int(np.prod(shape))

    flat_means = cell_means.flatten()
    flat_counts = cell_counts.flatten()
    valid = (flat_counts > 0) & np.isfinite(flat_means)

    # Multi-dimensional index for each cell: shape (n_cells, num_dims)
    multi_idx = np.array(np.unravel_index(np.arange(n_cells), shape)).T

    # Design matrix: intercept + treatment-coded dummies for each dimension
    # (first arm of each dimension is the reference category)
    cols = [np.ones(n_cells)]
    for dim_idx in range(num_dims):
        for level in range(1, shape[dim_idx]):
            cols.append((multi_idx[:, dim_idx] == level).astype(float))
    design_matrix = np.column_stack(cols)

    # WLS: absorb sqrt(cell_counts) into X and y, then solve via lstsq
    x_valid = design_matrix[valid]
    y_valid = flat_means[valid]
    weights_valid = flat_counts[valid]
    weight_matrix = np.diag(np.sqrt(weights_valid))
    beta, _, _, _ = np.linalg.lstsq(weight_matrix @ x_valid, weight_matrix @ y_valid, rcond=None)

    fitted_cell_means = (design_matrix @ beta).reshape(shape)
    return fitted_cell_means

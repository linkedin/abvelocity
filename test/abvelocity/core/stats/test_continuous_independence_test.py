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

import numpy as np
import pytest

from abvelocity.core.stats.continuous_independence_test import (
    ContinuousIndependenceTestResult,
    continuous_independence_test,
)
from abvelocity.core.stats.interaction_statistics import fit_additive_kdim_cells


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def additive_means(shape=(3, 3), row_effects=None, col_effects=None, grand_mean=10.0):
    """Build a cell means array that is perfectly additive (no interaction)."""
    if row_effects is None:
        row_effects = np.arange(shape[0], dtype=float)
    if col_effects is None:
        col_effects = np.arange(shape[1], dtype=float)
    means = np.zeros(shape)
    for row_idx in range(shape[0]):
        for col_idx in range(shape[1]):
            means[row_idx, col_idx] = grand_mean + row_effects[row_idx] + col_effects[col_idx]
    return means


def uniform_inputs(shape, mean_val=10.0, sd_val=1.0, count_val=1000):
    """Build uniform means/sds/counts arrays."""
    means = np.full(shape, mean_val, dtype=float)
    sds = np.full(shape, sd_val, dtype=float)
    counts = np.full(shape, count_val, dtype=float)
    return means, sds, counts


# ---------------------------------------------------------------------------
# No interaction (additive model): T should be near 0
# ---------------------------------------------------------------------------


def test_additive_means_zero_f():
    means = additive_means(shape=(3, 3))
    sds = np.ones((3, 3))
    counts = np.full((3, 3), 1000.0)
    result = continuous_independence_test(means, sds, counts)
    assert isinstance(result, ContinuousIndependenceTestResult)
    assert result.f_value == pytest.approx(0.0, abs=1e-6)
    assert result.p_value == pytest.approx(1.0, abs=1e-6)


def test_additive_means_2x2_zero_f():
    means = additive_means(shape=(2, 2), row_effects=[0, 5], col_effects=[0, 3])
    sds = np.ones((2, 2)) * 2.0
    counts = np.full((2, 2), 500.0)
    result = continuous_independence_test(means, sds, counts)
    assert result.f_value == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Strong interaction: F should be large, p small
# ---------------------------------------------------------------------------


def test_interaction_large_f():
    # Means with a clear crossover interaction
    means = np.array(
        [
            [100.0, 10.0],
            [10.0, 100.0],
        ]
    )
    sds = np.ones((2, 2)) * 5.0
    counts = np.full((2, 2), 10000.0)
    result = continuous_independence_test(means, sds, counts)
    assert result.f_value > 100.0
    assert result.p_value < 1e-10


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_fields():
    means, sds, counts = uniform_inputs((2, 3))
    result = continuous_independence_test(means, sds, counts)
    assert result.n_cells == 6
    assert result.n_total == pytest.approx(6 * 1000.0)
    assert result.sigma_within_cell == pytest.approx(1.0, abs=1e-6)
    # dof_within = sum of (n_cell - 1) = 6 * 999 = 5994
    assert result.dof_within == pytest.approx(5994.0)
    assert result.fitted_means is not None
    assert result.fitted_means.shape == (2, 3)
    assert result.cell_residuals is not None
    assert result.cell_residuals.shape == (2, 3)


def test_dof_2x2():
    means, sds, counts = uniform_inputs((2, 2))
    result = continuous_independence_test(means, sds, counts)
    # dof = 4 - (2+2) + 2 - 1 = 4 - 4 + 1 = 1
    assert result.dof == 1


def test_dof_3x4():
    means, sds, counts = uniform_inputs((3, 4))
    result = continuous_independence_test(means, sds, counts)
    # dof = 12 - (3+4) + 2 - 1 = 12 - 7 + 1 = 6
    assert result.dof == 6


# ---------------------------------------------------------------------------
# Excluded cells
# ---------------------------------------------------------------------------


def test_excluded_cell_ignored():
    means = additive_means(shape=(3, 3))
    # Corrupt one cell with a large interaction signal
    means[2, 2] = 9999.0
    sds = np.ones((3, 3))
    counts = np.full((3, 3), 1000.0)
    # Excluding the corrupted cell should restore near-zero F
    result = continuous_independence_test(means, sds, counts, excluded_cells=[(2, 2)])
    assert result.f_value == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_raises_fewer_than_2_valid_cells():
    means = np.array([[10.0, np.nan], [np.nan, np.nan]])
    sds = np.array([[1.0, np.nan], [np.nan, np.nan]])
    counts = np.array([[1000.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="valid variant combination cell"):
        continuous_independence_test(means, sds, counts)


def test_raises_no_within_cell_variance():
    # All cells have count = 1, so no within-cell variance can be estimated
    means = np.array([[10.0, 12.0], [11.0, 13.0]])
    sds = np.array([[1.0, 1.0], [1.0, 1.0]])
    counts = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="within-cell variance"):
        continuous_independence_test(means, sds, counts)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------


def test_3d_additive_zero_t():
    shape = (2, 2, 2)
    row_effects = [0.0, 1.0]
    means = np.zeros(shape)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                means[i, j, k] = 10.0 + row_effects[i] + 2 * row_effects[j] + 3 * row_effects[k]
    sds = np.ones(shape)
    counts = np.full(shape, 500.0)
    result = continuous_independence_test(means, sds, counts)
    assert result.f_value == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# fit_additive_kdim_cells
# ---------------------------------------------------------------------------


def test_fit_additive_kdim_cells_recovers_additive_input():
    # Additive cell means: fitted values should equal observed exactly.
    cell_means = additive_means(shape=(3, 3))
    cell_counts = np.full((3, 3), 1000.0)
    fitted_cell_means = fit_additive_kdim_cells(cell_means, cell_counts)
    assert fitted_cell_means.shape == cell_means.shape
    np.testing.assert_allclose(fitted_cell_means, cell_means, atol=1e-6)


def test_fit_additive_kdim_cells_shape_preserved():
    cell_means = additive_means(shape=(2, 4))
    cell_counts = np.full((2, 4), 500.0)
    fitted_cell_means = fit_additive_kdim_cells(cell_means, cell_counts)
    assert fitted_cell_means.shape == (2, 4)


def test_fit_additive_kdim_cells_3d():
    shape = (2, 2, 2)
    cell_means = np.zeros(shape)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                cell_means[i, j, k] = 10.0 + i + 2 * j + 3 * k
    cell_counts = np.full(shape, 500.0)
    fitted_cell_means = fit_additive_kdim_cells(cell_means, cell_counts)
    np.testing.assert_allclose(fitted_cell_means, cell_means, atol=1e-6)


def test_fit_additive_kdim_cells_interaction_not_recovered():
    # Cell means with a crossover interaction: fitted (additive) should differ.
    cell_means = np.array([[100.0, 10.0], [10.0, 100.0]])
    cell_counts = np.full((2, 2), 1000.0)
    fitted_cell_means = fit_additive_kdim_cells(cell_means, cell_counts)
    assert not np.allclose(fitted_cell_means, cell_means, atol=1.0)


def test_fit_additive_kdim_cells_output_satisfies_additive_constraint():
    # For any input, the fitted values must satisfy the additive constraint:
    # column differences must be constant across rows (no interaction term).
    cell_means = np.array([[100.0, 10.0], [10.0, 100.0]])  # crossover interaction
    cell_counts = np.full((2, 2), 1000.0)
    fitted_cell_means = fit_additive_kdim_cells(cell_means, cell_counts)
    # additive: fitted[0,0] - fitted[0,1] == fitted[1,0] - fitted[1,1]
    assert (fitted_cell_means[0, 0] - fitted_cell_means[0, 1]) == pytest.approx(fitted_cell_means[1, 0] - fitted_cell_means[1, 1], abs=1e-6)

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
import pandas as pd
import pytest

from abvelocity.core.stats.count_independence_test import (
    CountIndependenceTestResult,
    build_count_array,
    count_independence_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_independent_2d(n=10000, seed=0):
    """Build a 2D count array that is perfectly (multiplicatively) independent."""
    rng = np.random.default_rng(seed)
    row_probs = np.array([0.3, 0.4, 0.3])
    col_probs = np.array([0.5, 0.5])
    counts = np.outer(row_probs, col_probs) * n
    # Add small Poisson noise so it is not exactly rank-1
    counts = counts + rng.poisson(1, size=counts.shape).astype(float)
    return counts


def make_dependent_2d():
    """Build a 2D array with a strong dependency (diagonal dominance)."""
    arr = np.array(
        [
            [500, 50],
            [50, 500],
        ],
        dtype=float,
    )
    return arr


# ---------------------------------------------------------------------------
# count_independence_test: basic 2D
# ---------------------------------------------------------------------------


def test_independent_table_low_chi2():
    arr = make_independent_2d(n=100000)
    result = count_independence_test(arr)
    assert isinstance(result, CountIndependenceTestResult)
    # For near-independent data chi2 should be small
    assert result.chi2_value < 20.0
    assert result.p_value > 0.01
    assert result.cramers_v < 0.05


def test_dependent_table_high_chi2():
    arr = make_dependent_2d()
    result = count_independence_test(arr)
    assert result.chi2_value > 500.0
    assert result.p_value < 1e-10
    assert result.cramers_v > 0.5


def test_exact_independence_zero_chi2():
    # Perfectly multiplicative table -> chi2 = 0
    row_probs = np.array([0.4, 0.6])
    col_probs = np.array([0.3, 0.7])
    arr = np.outer(row_probs, col_probs) * 10000
    result = count_independence_test(arr)
    assert result.chi2_value == pytest.approx(0.0, abs=1e-6)
    assert result.cramers_v == pytest.approx(0.0, abs=1e-6)
    assert result.p_value == pytest.approx(1.0, abs=1e-6)


def test_dof_2x2():
    arr = np.array([[100, 200], [300, 400]], dtype=float)
    result = count_independence_test(arr)
    # dof for 2x2 = prod(2,2) - sum(2,2) + 2 - 1 = 4 - 4 + 1 = 1
    assert result.dof == 1


def test_dof_3x4():
    arr = np.ones((3, 4), dtype=float) * 100
    result = count_independence_test(arr)
    # dof = 12 - (3+4) + 2 - 1 = 12 - 7 + 1 = 6
    assert result.dof == 6


def test_n_total_equals_sum():
    arr = np.array([[100, 200], [300, 400]], dtype=float)
    result = count_independence_test(arr)
    assert result.n_total == pytest.approx(1000.0)


def test_cell_residuals_shape():
    arr = np.array([[100, 200], [300, 400]], dtype=float)
    result = count_independence_test(arr)
    assert result.cell_residuals is not None
    assert result.cell_residuals.shape == arr.shape


def test_expected_cell_counts_sums_to_n_total():
    arr = make_dependent_2d()
    result = count_independence_test(arr)
    assert result.expected_cell_counts is not None
    assert result.expected_cell_counts.sum() == pytest.approx(result.n_total, rel=1e-6)


# ---------------------------------------------------------------------------
# excluded_cells
# ---------------------------------------------------------------------------


def test_excluded_cell_excluded_from_chi2():
    # Table where (nan, nan) corner is always zero by design
    arr = np.array(
        [
            [400, 200, 0],
            [300, 100, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )
    result = count_independence_test(arr, excluded_cells=[(-1, -1)])
    assert result.cell_residuals is not None
    # The excluded cell should have NaN residual
    assert np.isnan(result.cell_residuals[-1, -1])
    # DOF should be reduced by 1 compared to no exclusion
    result_no_excl = count_independence_test(arr)
    assert result.dof == max(result_no_excl.dof - 1, 1)


def test_excluded_cell_does_not_affect_independent_inner():
    # Inner block is perfectly independent; excluded cell at corner
    inner = np.outer([0.4, 0.6], [0.5, 0.5]) * 1000
    arr = np.zeros((3, 3))
    arr[:2, :2] = inner
    arr[0, 2] = 200  # A-triggered, B-nan
    arr[1, 2] = 300
    arr[2, 0] = 100  # A-nan, B-triggered
    arr[2, 1] = 150
    arr[2, 2] = 0  # excluded corner
    result = count_independence_test(arr, excluded_cells=[(-1, -1)])
    assert np.isnan(result.cell_residuals[-1, -1])


def test_multiple_excluded_cells():
    arr = np.array(
        [
            [400, 200, 50],
            [300, 100, 60],
            [80, 70, 0],
        ],
        dtype=float,
    )
    result_one = count_independence_test(arr, excluded_cells=[(-1, -1)])
    result_two = count_independence_test(arr, excluded_cells=[(-1, -1), (0, 2)])
    # Two exclusions reduce dof by one more than one exclusion
    assert result_two.dof == max(result_one.dof - 1, 1)
    assert np.isnan(result_two.cell_residuals[-1, -1])
    assert np.isnan(result_two.cell_residuals[0, 2])


# ---------------------------------------------------------------------------
# 3D independence test
# ---------------------------------------------------------------------------


def test_3d_independent():
    # Build a 3D array from the outer product of three marginals
    p1, p2, p3 = np.array([0.5, 0.5]), np.array([0.4, 0.6]), np.array([0.3, 0.7])
    arr = np.einsum("i,j,k", p1, p2, p3) * 10000
    result = count_independence_test(arr)
    assert result.chi2_value == pytest.approx(0.0, abs=1e-6)
    # dof for 2x2x2 = 8 - (2+2+2) + 3 - 1 = 8 - 6 + 2 = 4
    assert result.dof == 4


# ---------------------------------------------------------------------------
# build_count_array
# ---------------------------------------------------------------------------


def test_build_count_array_basic():
    df = pd.DataFrame(
        {
            "dim1": ["a", "a", "b", "b"],
            "dim2": ["x", "y", "x", "y"],
            "cnt": [100, 200, 300, 400],
        }
    )
    arr, labels = build_count_array(df, dim_cols=["dim1", "dim2"], count_col="cnt")
    assert arr.shape == (2, 2)
    assert labels[0] == ["a", "b"]
    assert labels[1] == ["x", "y"]
    assert arr.sum() == 1000.0


def test_build_count_array_nan_last():
    df = pd.DataFrame(
        {
            "dim1": ["a", "nan", "b"],
            "dim2": ["x", "x", "x"],
            "cnt": [100, 200, 300],
        }
    )
    arr, labels = build_count_array(df, dim_cols=["dim1", "dim2"], count_col="cnt", nan_last=True)
    assert labels[0][-1] == "nan"

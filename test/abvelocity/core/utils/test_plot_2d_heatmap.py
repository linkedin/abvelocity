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

from abvelocity.core.utils.plot_2d_heatmap import plot_2d_heatmap


def make_residuals():
    count_array = np.array([[500, 50], [50, 500]], dtype=float)
    total = count_array.sum()
    row_sums = count_array.sum(axis=1, keepdims=True)
    col_sums = count_array.sum(axis=0, keepdims=True)
    expected = row_sums * col_sums / total
    return (count_array - expected) / np.sqrt(expected)


def test_returns_figure():
    values = make_residuals()
    fig = plot_2d_heatmap(values, ["ctrl", "treat"], ["ctrl", "treat"])
    assert hasattr(fig, "layout")
    assert hasattr(fig, "data")
    assert len(fig.data) == 1


def test_raises_for_non_2d():
    with pytest.raises(ValueError, match="2D"):
        plot_2d_heatmap(np.ones((2, 2, 2)), ["a", "b"], ["x", "y"])


def test_annotation_in_title():
    values = make_residuals()
    fig = plot_2d_heatmap(
        values,
        row_labels=["ctrl", "treat"],
        col_labels=["ctrl", "treat"],
        annotation="chi2=900, p=1e-200",
        title="Test heatmap",
    )
    assert "chi2=900" in fig.layout.title.text


def test_nan_cell_shows_na_by_default():
    values = np.array([[0.5, np.nan], [-0.5, 0.3]])
    fig = plot_2d_heatmap(values, ["a", "b"], ["x", "y"])
    texts = fig.data[0].text
    assert texts[0][1] == "N/A"


def test_symmetric_clim_derived_from_data():
    values = np.array([[3.0, -1.0], [0.5, -3.0]])
    fig = plot_2d_heatmap(values, ["a", "b"], ["x", "y"], symmetric=True)
    assert fig.data[0].zmax == pytest.approx(3.0)
    assert fig.data[0].zmin == pytest.approx(-3.0)


def test_clim_override():
    values = np.array([[10.0, -10.0], [5.0, -5.0]])
    fig = plot_2d_heatmap(values, ["a", "b"], ["x", "y"], clim=7.0)
    assert fig.data[0].zmax == pytest.approx(7.0)
    assert fig.data[0].zmin == pytest.approx(-7.0)


def test_custom_axis_names():
    values = make_residuals()
    fig = plot_2d_heatmap(
        values,
        row_labels=["ctrl", "treat"],
        col_labels=["ctrl", "treat"],
        axis_names=("Experiment A", "Experiment B"),
    )
    assert "Experiment A" in fig.layout.yaxis.title.text
    assert "Experiment B" in fig.layout.xaxis.title.text


def test_custom_cell_texts():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    cell_texts = [["a", "b"], ["c", "d"]]
    fig = plot_2d_heatmap(values, ["r0", "r1"], ["c0", "c1"], cell_texts=cell_texts)
    assert fig.data[0].text[0][0] == "a"
    assert fig.data[0].text[1][1] == "d"


def test_asymmetric_colorbar():
    values = np.array([[0.0, 5.0], [2.0, 8.0]])
    fig = plot_2d_heatmap(values, ["a", "b"], ["x", "y"], symmetric=False)
    assert fig.data[0].zmin == pytest.approx(0.0)
    assert fig.data[0].zmax == pytest.approx(8.0)


def test_means_heatmap_use_case():
    # Verify the function works equally well for a means array (not residuals)
    means = np.array([[10.5, 12.3], [9.8, 11.1]])
    fig = plot_2d_heatmap(
        means,
        row_labels=["ctrl", "treat"],
        col_labels=["ios", "android"],
        colorbar_title="mean",
        symmetric=False,
        title="Cell means",
    )
    assert fig.layout.title.text == "Cell means"

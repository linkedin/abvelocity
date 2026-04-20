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

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from abvelocity.core.stats.stat_test_compare_distbns import stat_test_compare_distbns

SEED = 1317
WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/stat_test_compare_distbns")
os.makedirs(WRITE_PATH, exist_ok=True)

# --- Basic Unit Tests ---


def test_stat_test_compare_distbns_tuple_basic():
    distbns = [(100, 150), (110, 140)]
    result = stat_test_compare_distbns(distbns=distbns, method="LR")
    assert result["test_type"] == "Homogeneity"
    assert result["dof"] == 1


def test_stat_test_compare_distbns_tuple_multiple():
    distbns = [(100, 100), (100, 100), (100, 100), (180, 20)]
    result = stat_test_compare_distbns(distbns=distbns, method="LR")
    assert result["dof"] == 3
    assert result["p_value"] < 0.01


def test_stat_test_compare_distbns_tuple_true_baseline():
    distbns = [(200, 50, 50)]
    true_distbn = (1, 1, 1)
    result = stat_test_compare_distbns(distbns=distbns, true_distbn=true_distbn, method="Pearson")
    assert result["test_type"] == "Goodness-of-Fit"
    assert result["p_value"] < 0.001


def test_stat_test_compare_distbns_validation_error():
    with pytest.raises(ValueError):
        stat_test_compare_distbns(distbns=[(1, 2), (1, 2, 3)])


# --- Simulation Infrastructure ---


def run_simulation_and_plot(p_h0_a, p_h0_b, p_h1_a, p_h1_b, file_suffix, true_distbn=None, n_range=None):
    np.random.seed(SEED)
    sample_sizes = n_range if n_range is not None else np.arange(50, 1001, 150)
    iterations = 2000
    alpha = 0.05
    results = []

    for n in sample_sizes:
        pwr_lr, pwr_p = 0, 0
        t1_lr, t1_p = 0, 0

        for _ in range(iterations):
            # Sample Generation
            s0_a = tuple(np.random.multinomial(n, p_h0_a))
            s1_a = tuple(np.random.multinomial(n, p_h1_a))

            if true_distbn:
                # Goodness-of-Fit Comparison
                res_h0_lr = stat_test_compare_distbns([s0_a], true_distbn=true_distbn, method="LR")
                res_h0_p = stat_test_compare_distbns([s0_a], true_distbn=true_distbn, method="Pearson")
                res_h1_lr = stat_test_compare_distbns([s1_a], true_distbn=true_distbn, method="LR")
                res_h1_p = stat_test_compare_distbns([s1_a], true_distbn=true_distbn, method="Pearson")
            else:
                # Homogeneity Comparison (Two groups)
                s0_b = tuple(np.random.multinomial(n, p_h0_b))
                s1_b = tuple(np.random.multinomial(n, p_h1_b))
                res_h0_lr = stat_test_compare_distbns([s0_a, s0_b], method="LR")
                res_h0_p = stat_test_compare_distbns([s0_a, s0_b], method="Pearson")
                res_h1_lr = stat_test_compare_distbns([s1_a, s1_b], method="LR")
                res_h1_p = stat_test_compare_distbns([s1_a, s1_b], method="Pearson")

            if res_h0_lr["p_value"] < alpha:
                t1_lr += 1
            if res_h0_p["p_value"] < alpha:
                t1_p += 1
            if res_h1_lr["p_value"] < alpha:
                pwr_lr += 1
            if res_h1_p["p_value"] < alpha:
                pwr_p += 1

        results.extend(
            [
                {"n": n, "Rate": t1_lr / iterations, "Method": "LR", "Metric": "Type 1 Error"},
                {"n": n, "Rate": t1_p / iterations, "Method": "Pearson", "Metric": "Type 1 Error"},
                {"n": n, "Rate": pwr_lr / iterations, "Method": "LR", "Metric": "Power"},
                {"n": n, "Rate": pwr_p / iterations, "Method": "Pearson", "Metric": "Power"},
            ]
        )

    df = pd.DataFrame(results)
    fig = go.Figure()

    configs = {
        ("LR", "Power"): dict(color="rgba(31, 119, 180, 0.4)", width=10, dash="solid"),
        ("Pearson", "Power"): dict(color="rgba(255, 127, 14, 1.0)", width=2, dash="solid"),
        ("LR", "Type 1 Error"): dict(color="rgba(31, 119, 180, 0.4)", width=10, dash="dot"),
        ("Pearson", "Type 1 Error"): dict(color="rgba(255, 127, 14, 1.0)", width=2, dash="dot"),
    }

    for (meth, met), style in configs.items():
        sub = df[(df["Metric"] == met) & (df["Method"] == meth)]
        fig.add_trace(go.Scatter(x=sub["n"], y=sub["Rate"], mode="lines+markers", name=f"{meth} {met}", line=style))

    title = f"Sim: {file_suffix}<br>H0_p: {p_h0_a} | H1_p: {p_h1_a}"
    fig.add_hline(y=0.05, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title="n", yaxis_title="Rate", height=700)

    full_filename = f"sim_{file_suffix.replace(' ', '_').lower()}.html"
    fig.write_html(str(WRITE_PATH.joinpath(full_filename)), include_plotlyjs="cdn")


@pytest.mark.parametrize(
    "p_h0_a, p_h0_b, p_h1_a, p_h1_b, suffix, true_dist, n_range",
    [
        # 1. Standard Balanced Homogeneity
        ([0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.44, 0.56], "homogeneity_balanced", None, None),
        # 2. GoF Skewed Baseline (0.1/0.9)
        (
            [0.1, 0.9],
            [0.1, 0.9],
            [0.13, 0.87],
            [0.13, 0.87],
            "gof_skewed_baseline",
            (0.1, 0.9),
            None,
        ),
        # 3. Homogeneity Multi-category (K=5)
        (
            [0.2] * 5,
            [0.2] * 5,
            [0.2] * 5,
            [0.16, 0.24, 0.2, 0.2, 0.2],
            "homogeneity_multicat_k5",
            None,
            None,
        ),
        # 4. Homogeneity High-K (K=10) - Should show Type 1 error closer to 0.05
        (
            [0.1] * 10,
            [0.1] * 10,
            [0.1] * 10,
            [0.05, 0.15] + [0.1] * 8,
            "homogeneity_k10_smooth",
            None,
            None,
        ),
        # 5. Tiny Sample Sizes (Small N)
        (
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.3, 0.7],
            "tiny_n_homogeneity",
            None,
            np.arange(5, 51, 5),
        ),
        # 6. GoF with Uniform Baseline (K=4)
        (
            [0.25] * 4,
            [0.25] * 4,
            [0.30, 0.20, 0.25, 0.25],
            [0.30, 0.20, 0.25, 0.25],
            "gof_uniform_k4",
            (0.25, 0.25, 0.25, 0.25),
            None,
        ),
    ],
)
def test_stat_test_simulations(p_h0_a, p_h0_b, p_h1_a, p_h1_b, suffix, true_dist, n_range):
    run_simulation_and_plot(p_h0_a, p_h0_b, p_h1_a, p_h1_b, suffix, true_dist, n_range)

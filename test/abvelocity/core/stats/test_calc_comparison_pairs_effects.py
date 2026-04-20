# BSD 2-CLAUSE LICENSE
# Copyright 2024, Blah Corporation. All rights reserved.
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
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.param.constants import CI_PERCENT_COL
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.sim.ctr_example import COMPARISON_PAIRS, POPULATION_SIZE, SEED, sim_ctr_data
from abvelocity.core.stats.calc_comparison_pairs_effects import calc_comparison_pairs_effects

# Constants
CI_COVERAGE = 0.95
NUM_BUCKETS = 20
PLOT_WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/calc_comparison_pairs_effects").resolve()
BASE_SAMPLE_SIZE = 1000
MEAN_INDEX = 1
MEAN_RATIO_INDEX = 5


@pytest.fixture(scope="module")
def get_test_data():
    return sim_ctr_data(seed=SEED, population_size=POPULATION_SIZE)


@pytest.mark.parametrize("n", [500, 1000, 5000, 10000, 20000])
def test_calc_comparison_pairs_effects_impressions(get_test_data, n):
    """Test calc_comparison_pairs_effects for impressions with varying sample sizes."""
    df, strata_info, true_values = get_test_data
    df_sample = df.sample(n=n, random_state=SEED) if n <= len(df) else df
    strata_df = strata_info.df.copy()

    metric = Metric(
        numerator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="MEAN",
        name="impressions",
    )

    result = calc_comparison_pairs_effects(
        dc=DataContainer(pandas_df=df_sample),
        variant_col="variant",
        metric=metric,
        comparison_pairs=COMPARISON_PAIRS,  # Fixed: Added missing constant
        ci_coverage=CI_COVERAGE,
        variant_count_df=strata_df,
    )

    assert not result.empty, "Result DataFrame should not be empty"
    expected_cols = {
        "comparison_pair",
        "delta",
        "delta_percent",
        "ci",
        CI_PERCENT_COL,
        "p_value",
        "delta_sum",
        "impacted_counts",
        "sample_counts",
    }
    assert set(result.columns) == expected_cols, "Incorrect columns"
    assert isinstance(result["impacted_counts"].iloc[0], tuple), "impacted_counts should be a tuple"
    assert isinstance(result["sample_counts"].iloc[0], tuple), "sample_counts should be a tuple"
    assert isinstance(result["ci"].iloc[0], tuple), "ci should be a tuple"
    assert isinstance(result[CI_PERCENT_COL].iloc[0], tuple), f"{CI_PERCENT_COL} should be a tuple"
    assert not isinstance(result["delta_sum"].iloc[0], tuple), "delta_sum should be float for simple metrics"
    assert np.isclose(
        result["delta"].iloc[0], true_values["impressions_diff"], rtol=0.2
    ), f"Expected delta ~{true_values['impressions_diff']}, got {result['delta'].iloc[0]} for n={n}"
    assert np.isclose(
        result["delta_percent"].iloc[0], true_values["impressions_pct_diff"], rtol=0.2
    ), f"Expected delta_percent ~{true_values['impressions_pct_diff']}, got {result['delta_percent'].iloc[0]} for n={n}"


@pytest.mark.parametrize("n", [500, 1000, 5000, 10000, 20000])
def test_calc_comparison_pairs_effects_ctr(get_test_data, n):
    """Test calc_comparison_pairs_effects for CTR with varying sample sizes."""
    df, strata_info, true_values = get_test_data
    df_sample = df.sample(n=n, random_state=SEED) if n <= len(df) else df
    strata_df = strata_info.df.copy()

    metric = Metric(
        numerator=UMetric(col="clicks", agg="MAX", fill_na=0, name="Clicks"),
        denominator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="SUM",
        denominator_agg="SUM",
        name="ctr",
    )

    result = calc_comparison_pairs_effects(
        dc=DataContainer(pandas_df=df_sample),
        variant_col="variant",
        metric=metric,
        comparison_pairs=COMPARISON_PAIRS,
        ci_coverage=CI_COVERAGE,
        variant_count_df=strata_df,
    )

    assert not result.empty, "Result DataFrame should not be empty"
    expected_cols = {
        "comparison_pair",
        "delta",
        "delta_percent",
        "ci",
        CI_PERCENT_COL,
        "p_value",
        "delta_sum",
        "impacted_counts",
        "sample_counts",
    }
    assert set(result.columns) == expected_cols, "Incorrect columns"
    assert isinstance(result["impacted_counts"].iloc[0], tuple), "impacted_counts should be a tuple"
    assert isinstance(result["sample_counts"].iloc[0], tuple), "sample_counts should be a tuple"
    assert isinstance(result["ci"].iloc[0], tuple), "ci should be a tuple"
    assert isinstance(result[CI_PERCENT_COL].iloc[0], tuple), f"{CI_PERCENT_COL} should be a tuple"
    assert isinstance(result["delta_sum"].iloc[0], tuple), "delta_sum should be a tuple for ratio metrics"
    assert np.isclose(
        result["delta"].iloc[0], true_values["ctr_diff"], rtol=0.2
    ), f"Expected delta ~{true_values['ctr_diff']}, got {result['delta'].iloc[0]} for n={n}"
    assert np.isclose(
        result["delta_percent"].iloc[0], true_values["ctr_pct_diff"], rtol=0.2
    ), f"Expected delta_percent ~{true_values['ctr_pct_diff']}, got {result['delta_percent'].iloc[0]} for n={n}"


def test_convergence(get_test_data, tmp_path):
    """Test convergence of delta and delta_percent for impressions and CTR."""
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    df, strata_info, true_values = get_test_data
    sample_sizes = list(range(500, POPULATION_SIZE, 200))

    # Impressions
    impressions_results = []
    metric_impressions = Metric(
        numerator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="MEAN",
        name="impressions",
    )
    for n in sample_sizes:
        df_sample = df.sample(n=n, random_state=SEED) if n <= len(df) else df
        result = calc_comparison_pairs_effects(
            dc=DataContainer(pandas_df=df_sample),
            variant_col="variant",
            metric=metric_impressions,
            comparison_pairs=COMPARISON_PAIRS,
            ci_coverage=CI_COVERAGE,
            variant_count_df=strata_info.df,
        )
        impressions_results.append(result.assign(n=n))

    impressions_df = pd.concat(impressions_results, ignore_index=True)
    impressions_csv = PLOT_WRITE_PATH / "impressions_results.csv"
    impressions_df.to_csv(impressions_csv, index=False)

    # CTR
    ctr_results = []
    metric_ctr = Metric(
        numerator=UMetric(col="clicks", agg="MAX", fill_na=0, name="Clicks"),
        denominator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="SUM",
        denominator_agg="SUM",
        name="ctr",
    )
    for n in sample_sizes:
        df_sample = df.sample(n=n, random_state=SEED) if n <= len(df) else df
        result = calc_comparison_pairs_effects(
            dc=DataContainer(pandas_df=df_sample),
            variant_col="variant",
            metric=metric_ctr,
            comparison_pairs=COMPARISON_PAIRS,
            ci_coverage=CI_COVERAGE,
            variant_count_df=strata_info.df,
        )
        ctr_results.append(result.assign(n=n))

    ctr_df = pd.concat(ctr_results, ignore_index=True)
    ctr_csv = PLOT_WRITE_PATH / "ctr_results.csv"
    ctr_df.to_csv(ctr_csv, index=False)

    # Convergence plots
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    # Impressions Delta
    fig_impressions_delta = go.Figure()
    fig_impressions_delta.add_trace(
        go.Scatter(
            x=impressions_df["n"],
            y=impressions_df["delta"],
            mode="lines+markers",
            name="Delta",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
            hovertemplate="Sample Size: %{x}<br>Delta: %{y:.4f}<extra></extra>",
        )
    )
    fig_impressions_delta.add_trace(
        go.Scatter(
            x=impressions_df["n"],
            y=[ci[0] for ci in impressions_df["ci"]],
            mode="lines",
            name="95% CI (Lower)",
            line=dict(color="rgba(31, 119, 180, 0.3)", dash="dash"),
            showlegend=True,
            legendgroup="CI",
            hovertemplate="Sample Size: %{x}<br>CI Lower: %{y:.4f}<extra></extra>",
        )
    )
    fig_impressions_delta.add_trace(
        go.Scatter(
            x=impressions_df["n"],
            y=[ci[1] for ci in impressions_df["ci"]],
            mode="lines",
            name="95% CI (Upper)",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0.3)", dash="dash"),
            showlegend=True,
            legendgroup="CI",
            hovertemplate="Sample Size: %{x}<br>CI Upper: %{y:.4f}<extra></extra>",
        )
    )
    fig_impressions_delta.add_trace(
        go.Scatter(
            x=[sample_sizes[0], sample_sizes[-1]],
            y=[true_values["impressions_diff"], true_values["impressions_diff"]],
            mode="lines",
            name=f"True Delta ({true_values['impressions_diff']:.4f})",
            line=dict(color="black", dash="dash"),
            showlegend=True,
            legendgroup="True",
            hovertemplate="Sample Size: %{x}<br>True Delta: %{y:.4f}<extra></extra>",
        )
    )
    fig_impressions_delta.update_layout(
        title="Impressions Delta Convergence with 95% CI",
        xaxis_title="Sample Size (n)",
        yaxis_title="Delta",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    fig_impressions_delta.write_html(
        str(PLOT_WRITE_PATH / "impressions_delta_convergence.html"),
        include_plotlyjs="cdn",
    )

    # Impressions Delta Percent
    fig_impressions_delta_percent = go.Figure()
    fig_impressions_delta_percent.add_trace(
        go.Scatter(
            x=impressions_df["n"],
            y=impressions_df["delta_percent"],
            mode="lines+markers",
            name="Delta Percent",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
            hovertemplate="Sample Size: %{x}<br>Delta Percent: %{y:.4f}<extra></extra>",
        )
    )
    fig_impressions_delta_percent.add_trace(
        go.Scatter(
            x=impressions_df["n"],
            y=[ci[0] for ci in impressions_df[CI_PERCENT_COL]],
            mode="lines",
            name="95% CI (Lower)",
            line=dict(color="rgba(31, 119, 180, 0.3)", dash="dash"),
            showlegend=True,
            legendgroup="CI",
            hovertemplate="Sample Size: %{x}<br>CI Lower: %{y:.4f}<extra></extra>",
        )
    )
    fig_impressions_delta_percent.add_trace(
        go.Scatter(
            x=impressions_df["n"],
            y=[ci[1] for ci in impressions_df[CI_PERCENT_COL]],
            mode="lines",
            name="95% CI (Upper)",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0.3)", dash="dash"),
            showlegend=True,
            legendgroup="CI",
            hovertemplate="Sample Size: %{x}<br>CI Upper: %{y:.4f}<extra></extra>",
        )
    )
    fig_impressions_delta_percent.add_trace(
        go.Scatter(
            x=[sample_sizes[0], sample_sizes[-1]],
            y=[true_values["impressions_pct_diff"], true_values["impressions_pct_diff"]],
            mode="lines",
            name=f"True Delta Percent ({true_values['impressions_pct_diff']:.4f})",
            line=dict(color="black", dash="dash"),
            showlegend=True,
            legendgroup="True",
            hovertemplate="Sample Size: %{x}<br>True Delta Percent: %{y:.4f}<extra></extra>",
        )
    )
    fig_impressions_delta_percent.update_layout(
        title="Impressions Delta Percent Convergence with 95% CI",
        xaxis_title="Sample Size (n)",
        yaxis_title="Delta Percent",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    fig_impressions_delta_percent.write_html(
        str(PLOT_WRITE_PATH / "impressions_delta_percent_convergence.html"),
        include_plotlyjs="cdn",
    )

    # CTR Delta
    fig_ctr_delta = go.Figure()
    fig_ctr_delta.add_trace(
        go.Scatter(
            x=ctr_df["n"],
            y=ctr_df["delta"],
            mode="lines+markers",
            name="Delta",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
            hovertemplate="Sample Size: %{x}<br>Delta: %{y:.4f}<extra></extra>",
        )
    )
    fig_ctr_delta.add_trace(
        go.Scatter(
            x=ctr_df["n"],
            y=[ci[0] for ci in ctr_df["ci"]],
            mode="lines",
            name="95% CI (Lower)",
            line=dict(color="rgba(31, 119, 180, 0.3)", dash="dash"),
            showlegend=True,
            legendgroup="CI",
            hovertemplate="Sample Size: %{x}<br>CI Lower: %{y:.4f}<extra></extra>",
        )
    )
    fig_ctr_delta.add_trace(
        go.Scatter(
            x=ctr_df["n"],
            y=[ci[1] for ci in ctr_df["ci"]],
            mode="lines",
            name="95% CI (Upper)",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0.3)", dash="dash"),
            showlegend=True,
            legendgroup="CI",
            hovertemplate="Sample Size: %{x}<br>CI Upper: %{y:.4f}<extra></extra>",
        )
    )
    fig_ctr_delta.add_trace(
        go.Scatter(
            x=[sample_sizes[0], sample_sizes[-1]],
            y=[true_values["ctr_diff"], true_values["ctr_diff"]],
            mode="lines",
            name=f"True Delta ({true_values['ctr_diff']:.4f})",
            line=dict(color="black", dash="dash"),
            showlegend=True,
            legendgroup="True",
            hovertemplate="Sample Size: %{x}<br>True Delta: %{y:.4f}<extra></extra>",
        )
    )
    fig_ctr_delta.update_layout(
        title="CTR Delta Convergence with 95% CI",
        xaxis_title="Sample Size (n)",
        yaxis_title="Delta",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    fig_ctr_delta.write_html(
        str(PLOT_WRITE_PATH / "ctr_delta_convergence.html"),
        include_plotlyjs="cdn",
    )

    # CTR Delta Percent
    fig_ctr_delta_percent = go.Figure()
    fig_ctr_delta_percent.add_trace(
        go.Scatter(
            x=ctr_df["n"],
            y=ctr_df["delta_percent"],
            mode="lines+markers",
            name="Delta Percent",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
            hovertemplate="Sample Size: %{x}<br>Delta Percent: %{y:.4f}<extra></extra>",
        )
    )
    fig_ctr_delta_percent.add_trace(
        go.Scatter(
            x=ctr_df["n"],
            y=[ci[0] for ci in ctr_df[CI_PERCENT_COL]],
            mode="lines",
            name="95% CI (Lower)",
            line=dict(color="rgba(31, 119, 180, 0.3)", dash="dash"),
            showlegend=True,
            legendgroup="CI",
            hovertemplate="Sample Size: %{x}<br>CI Lower: %{y:.4f}<extra></extra>",
        )
    )
    fig_ctr_delta_percent.add_trace(
        go.Scatter(
            x=ctr_df["n"],
            y=[ci[1] for ci in ctr_df[CI_PERCENT_COL]],
            mode="lines",
            name="95% CI (Upper)",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0.3)", dash="dash"),
            showlegend=True,
            legendgroup="CI",
            hovertemplate="Sample Size: %{x}<br>CI Upper: %{y:.4f}<extra></extra>",
        )
    )
    fig_ctr_delta_percent.add_trace(
        go.Scatter(
            x=[sample_sizes[0], sample_sizes[-1]],
            y=[true_values["ctr_pct_diff"], true_values["ctr_pct_diff"]],
            mode="lines",
            name=f"True Delta Percent ({true_values['ctr_pct_diff']:.4f})",
            line=dict(color="black", dash="dash"),
            showlegend=True,
            legendgroup="True",
            hovertemplate="Sample Size: %{x}<br>True Delta Percent: %{y:.4f}<extra></extra>",
        )
    )
    fig_ctr_delta_percent.update_layout(
        title="CTR Delta Percent Convergence with 95% CI",
        xaxis_title="Sample Size (n)",
        yaxis_title="Delta Percent",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    fig_ctr_delta_percent.write_html(
        str(PLOT_WRITE_PATH / "ctr_delta_percent_convergence.html"),
        include_plotlyjs="cdn",
    )

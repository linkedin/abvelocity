# test_cpde_constructor.py
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
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.get_data.get_expt_stats import get_expt_stats
from abvelocity.core.param.constants import CATEG_NAN_VALUE
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.variant import ComparisonPair, Variant, VariantList
from abvelocity.core.sim.sim import Sim
from abvelocity.core.stats.cpde_constructor import CPDEConstructor
from abvelocity.core.stats.param import StrataInfo

# Constants
CI_COVERAGE = 0.95
NUM_BUCKETS = 20
PLOT_WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/cpde").resolve()
POPULATION_SIZE = 20000
SEED = 1317
BASE_SAMPLE_SIZE = 1000
TRIGGER_STATE_COUNT_COL = "trigger_state_count"
MEAN_INDEX = 1  # Index for weighted_mean_numer in WeightedMetricEstimator
MEAN_RATIO_INDEX = 5  # Index for mean_ratio in WeightedMetricEstimator

# Simulation parameters
attribute_weights = {
    "Level": {"Senior": 0.5, "Junior": 0.5},
    "Country": {"Iran": 0.1, "Canada": 0.3, "UK": 0.2, "US": 0.4},
    "Device": {"Smartphone": 0.6, "Tablet": 0.3, "Laptop": 0.1},
}
metric_attribute_values = {
    "impressions": {"Level": {"Junior": 100, "Senior": 90}},
    "clicks": {"Level": {"Junior": 10, "Senior": 8}},
}
expt_variant_weights_multi = [
    {"control": 0.5, "enabled": 0.5},  # Experiment 1
    {"control": 0.5, "v1": 0.5},  # Experiment 2
]
population_pcnt_multi = [100, 100]  # 100% of population is reflected in the data
non_trigger_pct_multi = [10, 10]  # 10% non-trigger
expt_metric_impacts = [
    {
        "control": {"impressions": 0, "clicks": 0},
        "enabled": {"impressions": 10, "clicks": 2},
    },  # Experiment 1
    {
        "control": {"impressions": 0, "clicks": 0},
        "v1": {"impressions": 5, "clicks": 1},
    },  # Experiment 2
]
interaction_metric_impacts = {("enabled", "v1"): {"impressions": 15, "clicks": 3}}

# Control and treatment variant lists for launch ("enabled", "v1")
control_variant_values = [
    ("control", CATEG_NAN_VALUE),
    (CATEG_NAN_VALUE, "control"),
    ("control", "control"),
]
treatment_variant_values = [
    ("enabled", CATEG_NAN_VALUE),
    (CATEG_NAN_VALUE, "v1"),
    ("enabled", "v1"),
]

# True values using get_expected_delta for impressions
launch_value = ("enabled", "v1")
control_value = ("control", "control")

# Initialize comparison pair for plotting tests
comparison_pair = ComparisonPair(
    control=VariantList(variants=[Variant(value=v) for v in control_variant_values]),
    treatment=VariantList(variants=[Variant(value=v) for v in treatment_variant_values]),
)


# Weighted mean function
def weighted_mean(df, metric_col, variant_values, strata_info):
    """Compute weighted mean for a metric using standardized StrataInfo weights."""
    counts = [strata_info.df.at[v, TRIGGER_STATE_COUNT_COL] if v in strata_info.df.index else 0 for v in variant_values]
    total_count = sum(counts)
    weights = [c / total_count if total_count > 0 else 0 for c in counts]
    total = 0.0
    for v, w in zip(variant_values, weights):
        if v in strata_info.df.index:
            variant_mean = df[df["variant"] == v][metric_col].mean() if len(df[df["variant"] == v]) > 0 else 0
            total += w * variant_mean
    return total


def get_expected_delta(
    non_trigger_pct_multi: list[float],
    expt_metric_impacts,
    interaction_metric_impacts,
    launch_value: tuple,
    metric: str,
) -> float:
    """Calculate expected delta based on population parameters for two experiments.
    launch_value here represent the variant combination we are launching.
    It is different from a variant in the sense that a launch includes a list of variants.
    For example launch (enabled, v1) effect is obtained by comparing:
        - [("enabled", CATEG_NAN_VALUE), (CATEG_NAN_VALUE, "v1"), ("enabled", "v1")]
        - [("control", CATEG_NAN_VALUE), (CATEG_NAN_VALUE, "control"), ("control", "control")]
    The reason is the launch impact includes three triggering states.
        - only Expt 1 triggers
        - only Expt 2 triggers
        - Both trigger
    """
    # Get trigger rates
    trigger_rates = [(100.0 - x) / 100.0 for x in non_trigger_pct_multi]
    # Trigger rate for Expt 1
    trigger_rate1 = trigger_rates[0]
    # Trigger rate for Expt 2
    trigger_rate2 = trigger_rates[1]
    # Both trigger rate (both experiments trigger)
    trigger_rate_both = trigger_rates[0] * trigger_rates[1]
    # Only Expt 1 triggers
    trigger_rate_only1 = trigger_rate1 - trigger_rate_both
    # Only Expt 2 triggers
    trigger_rate_only2 = trigger_rate2 - trigger_rate_both

    # Univariate impact of launch_value in Expt 1
    impact1 = expt_metric_impacts[0][launch_value[0]][metric]
    # Univariate impact of launch_value in Expt 2
    impact2 = expt_metric_impacts[1][launch_value[1]][metric]
    # Extra impact on the common trigger population
    impact_both = interaction_metric_impacts.get(launch_value, {}).get(metric, 0)
    # Add univariate impacts to get final impact for both
    impact_both += impact1 + impact2

    trigger_rate_sum = trigger_rate_only1 + trigger_rate_both + trigger_rate_only2
    assert np.isclose(trigger_rate_sum, 0.99, rtol=0.001)

    # Expected Delta
    expected_delta = (trigger_rate_only1 * impact1 + trigger_rate_both * impact_both + trigger_rate_only2 * impact2) / (trigger_rate_sum)
    return expected_delta


@pytest.fixture(scope="module")
def get_test_data():
    """Creates DataFrame, StrataInfo, true values, and exploratory plots from a single Sim run."""
    sim = Sim(
        population_size=POPULATION_SIZE,
        attribute_weights=attribute_weights,
        metric_attribute_values=metric_attribute_values,
        expt_variant_weights_multi=expt_variant_weights_multi,
        population_pcnt_multi=population_pcnt_multi,
        non_trigger_pct_multi=non_trigger_pct_multi,
        population_seed=SEED,
        expt_assignment_seed_multi=[SEED, SEED + 1],
        expt_metric_impacts=expt_metric_impacts,
        interaction_metric_impacts=interaction_metric_impacts,
    )
    sim.run()
    df = sim.expt_metric_df.copy()
    # Randomize DataFrame rows to ensure robust testing
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Create StrataInfo using get_expt_stats
    expt_stats = get_expt_stats(dc=DataContainer(pandas_df=df))

    strata_df = expt_stats.variant_count_df[[TRIGGER_STATE_COUNT_COL]].copy()
    strata_info = StrataInfo(df=strata_df, strata_count_col=TRIGGER_STATE_COUNT_COL)

    # Impressions difference
    impressions_t = get_expected_delta(
        non_trigger_pct_multi,
        expt_metric_impacts,
        interaction_metric_impacts,
        launch_value,
        "impressions",
    )
    impressions_c = get_expected_delta(
        non_trigger_pct_multi,
        expt_metric_impacts,
        interaction_metric_impacts,
        control_value,
        "impressions",
    )

    assert np.isclose(impressions_t, 25, rtol=0.1)
    assert np.isclose(impressions_c, 0, rtol=0.1)

    true_impressions_diff = impressions_t - impressions_c
    # Impressions percent difference (base impressions ~95)
    true_impressions_pct = 100 * (true_impressions_diff / 95)

    # CTR difference using weighted_mean as it is not possible to easily compute that in closed form as the above
    # This is because ratio is not additive and the impacts info will not suffice
    true_ctr_diff = weighted_mean(df, "clicks", treatment_variant_values, strata_info) / weighted_mean(
        df, "impressions", treatment_variant_values, strata_info
    ) - weighted_mean(df, "clicks", control_variant_values, strata_info) / weighted_mean(df, "impressions", control_variant_values, strata_info)
    true_ctr_pct = (
        true_ctr_diff
        / (weighted_mean(df, "clicks", control_variant_values, strata_info) / weighted_mean(df, "impressions", control_variant_values, strata_info))
        * 100
    )

    # Exploratory Plots
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    # Get all variants from strata_info.df
    variants = list(strata_info.df.index)
    variant_labels = [f"{v[0]},{v[1]}" for v in variants]  # String representation for plotting

    # 1. Mean Impressions per Variant
    mean_impressions = []
    for v in variants:
        mean_val = df[df["variant"] == v]["impressions"].mean() if len(df[df["variant"] == v]) > 0 else 0
        mean_impressions.append(mean_val)

    fig_impressions = go.Figure()
    fig_impressions.add_trace(go.Bar(x=variant_labels, y=mean_impressions, name="Mean Impressions", marker_color="blue"))
    fig_impressions.update_layout(
        title="Mean Impressions per Variant",
        xaxis_title="Variant (Expt1, Expt2)",
        yaxis_title="Mean Impressions",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    fig_impressions.write_html(
        str(PLOT_WRITE_PATH.joinpath("exploratory_impressions_per_variant.html")),
        include_plotlyjs="cdn",
    )

    # 2. Mean Clicks per Variant
    mean_clicks = []
    for v in variants:
        mean_val = df[df["variant"] == v]["clicks"].mean() if len(df[df["variant"] == v]) > 0 else 0
        mean_clicks.append(mean_val)

    fig_clicks = go.Figure()
    fig_clicks.add_trace(go.Bar(x=variant_labels, y=mean_clicks, name="Mean Clicks", marker_color="green"))
    fig_clicks.update_layout(
        title="Mean Clicks per Variant",
        xaxis_title="Variant (Expt1, Expt2)",
        yaxis_title="Mean Clicks",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    fig_clicks.write_html(str(PLOT_WRITE_PATH.joinpath("exploratory_clicks_per_variant.html")), include_plotlyjs="cdn")

    # 3. Variant Weights
    counts = [strata_info.df.at[v, TRIGGER_STATE_COUNT_COL] if v in strata_info.df.index else 0 for v in variants]
    total_count = sum(counts)
    weights = [c / total_count if total_count > 0 else 0 for c in counts]

    fig_weights = go.Figure()
    fig_weights.add_trace(go.Bar(x=variant_labels, y=weights, name="Variant Weights", marker_color="purple"))
    fig_weights.update_layout(
        title="Normalized Weights per Variant",
        xaxis_title="Variant (Expt1, Expt2)",
        yaxis_title="Weight",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    fig_weights.write_html(
        str(PLOT_WRITE_PATH.joinpath("exploratory_weights_per_variant.html")),
        include_plotlyjs="cdn",
    )

    return (
        df,
        strata_info,
        {
            "impressions_diff": true_impressions_diff,
            "impressions_pct": true_impressions_pct,
            "ctr_diff": true_ctr_diff,
            "ctr_pct": true_ctr_pct,
        },
    )


# Unit Tests
def test_weighted_mean_standardization(get_test_data):
    """Test that weights in weighted_mean are standardized correctly."""
    df, strata_info, true_values = get_test_data
    df_subset = df[df["variant"].isin(control_variant_values)].copy()
    df_subset.loc[:, "impressions"] = 100.0  # Uniform value for testing

    result = weighted_mean(df_subset, "impressions", control_variant_values, strata_info)
    expected = 100.0  # All means are 100, weights sum to 1
    assert np.isclose(result, expected, rtol=1e-5), f"Expected {expected}, got {result}"

    # Check weight standardization
    counts = [strata_info.df.at[v, TRIGGER_STATE_COUNT_COL] if v in strata_info.df.index else 0 for v in control_variant_values]
    total_count = sum(counts)
    weights = [c / total_count if total_count > 0 else 0 for c in counts]
    assert np.isclose(sum(weights), 1.0), f"Weights should sum to 1, got {sum(weights)}"


def test_weighted_mean_empty_variant(get_test_data):
    """Test weighted_mean with empty variant data."""
    df, strata_info, true_values = get_test_data
    df_subset = df[df["variant"] == ("control", CATEG_NAN_VALUE)].copy()
    df_subset.loc[:, "impressions"] = 100.0

    result = weighted_mean(df_subset, "impressions", control_variant_values, strata_info)
    expected = 100.0 * (
        strata_info.df.at[("control", CATEG_NAN_VALUE), TRIGGER_STATE_COUNT_COL]
        / sum(strata_info.df.at[v, TRIGGER_STATE_COUNT_COL] for v in control_variant_values)
    )
    assert np.isclose(result, expected, rtol=1e-5), f"Expected {expected}, got {result}"


# Plotting Tests
def test_plot_impressions_simple_diff_convergence(get_test_data):
    """Plots convergence of CPDE estimates vs. true value for impressions simple difference."""
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    df, strata_info, true_values = get_test_data
    sample_sizes = list(range(50, 2001, 50))

    impressions_metric = Metric(
        numerator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="MEAN",
    )
    estimator_constructor = CPDEConstructor(
        metric=impressions_metric,
        strata_info=strata_info,
        comparison_pair=comparison_pair,
        diff_type="simple_diff",
        variant_col="variant",
        name="Impressions_Diff",
    )

    # Collect estimates and CIs
    results = {"estimates": [], "ci_lower": [], "ci_upper": []}
    for n in sample_sizes:
        sample_df = df[:n]  # First n rows (data is shuffled)
        est = estimator_constructor.construct()
        jk_result = est.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        results["estimates"].append(jk_result["estimator_value"][MEAN_INDEX])
        results["ci_lower"].append(jk_result["ci"][MEAN_INDEX][0])
        results["ci_upper"].append(jk_result["ci"][MEAN_INDEX][1])

    # Create plot
    fig = go.Figure()
    # True Value
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_values["impressions_diff"]] * len(sample_sizes),
            mode="lines",
            name=f"True Value ({true_values['impressions_diff']:.4f})",
            line=dict(color="black", dash="dot"),
            legendgroup="True",
        )
    )
    # CI Ribbon
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=results["ci_upper"] + results["ci_lower"][::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="95% CI",
            showlegend=True,
            legendgroup="Estimate",
        )
    )
    # Estimate Line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=results["estimates"],
            mode="lines",
            name="Estimate",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
        )
    )
    fig.update_layout(
        title="Convergence: Impressions Simple Diff with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Estimate",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_impressions_diff_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_plot_impressions_pct_diff_convergence(get_test_data):
    """Plots convergence of CPDE estimates vs. true value for impressions percent difference."""
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    df, strata_info, true_values = get_test_data
    sample_sizes = list(range(50, 2001, 50))

    impressions_metric = Metric(
        numerator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="MEAN",
    )
    estimator_constructor = CPDEConstructor(
        metric=impressions_metric,
        strata_info=strata_info,
        comparison_pair=comparison_pair,
        diff_type="pcnt_diff",
        variant_col="variant",
        name="Impressions_Pcnt_Diff",
    )

    # Collect estimates and CIs
    results = {"estimates": [], "ci_lower": [], "ci_upper": []}
    for n in sample_sizes:
        sample_df = df[:n]  # First n rows (data is shuffled)
        est = estimator_constructor.construct()
        jk_result = est.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        results["estimates"].append(jk_result["estimator_value"][MEAN_INDEX])
        results["ci_lower"].append(jk_result["ci"][MEAN_INDEX][0])
        results["ci_upper"].append(jk_result["ci"][MEAN_INDEX][1])

    # Create plot
    fig = go.Figure()
    # True Value
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_values["impressions_pct"]] * len(sample_sizes),
            mode="lines",
            name=f"True Value ({true_values['impressions_pct']:.4f})",
            line=dict(color="black", dash="dot"),
            legendgroup="True",
        )
    )
    # CI Ribbon
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=results["ci_upper"] + results["ci_lower"][::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="95% CI",
            showlegend=True,
            legendgroup="Estimate",
        )
    )
    # Estimate Line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=results["estimates"],
            mode="lines",
            name="Estimate",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
        )
    )
    fig.update_layout(
        title="Convergence: Impressions Percent Diff with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Estimate (%)",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_impressions_pct_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_plot_ctr_simple_diff_convergence(get_test_data):
    """Plots convergence of CPDE estimates vs. true value for CTR simple difference."""
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    df, strata_info, true_values = get_test_data
    sample_sizes = list(range(50, 2001, 50))

    ctr_metric = Metric(
        numerator=UMetric(col="clicks", agg="MAX", fill_na=0, name="Clicks"),
        denominator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="SUM",
        denominator_agg="SUM",
    )
    estimator_constructor = CPDEConstructor(
        metric=ctr_metric,
        strata_info=strata_info,
        comparison_pair=comparison_pair,
        diff_type="simple_diff",
        variant_col="variant",
        name="CTR_Diff",
    )

    # Collect estimates and CIs
    results = {"estimates": [], "ci_lower": [], "ci_upper": []}
    for n in sample_sizes:
        sample_df = df[:n]  # First n rows (data is shuffled)
        est = estimator_constructor.construct()
        jk_result = est.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        results["estimates"].append(jk_result["estimator_value"][MEAN_RATIO_INDEX])
        results["ci_lower"].append(jk_result["ci"][MEAN_RATIO_INDEX][0])
        results["ci_upper"].append(jk_result["ci"][MEAN_RATIO_INDEX][1])

    # Create plot
    fig = go.Figure()
    # True Value
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_values["ctr_diff"]] * len(sample_sizes),
            mode="lines",
            name=f"True Value ({true_values['ctr_diff']:.4f})",
            line=dict(color="black", dash="dot"),
            legendgroup="True",
        )
    )
    # CI Ribbon
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=results["ci_upper"] + results["ci_lower"][::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="95% CI",
            showlegend=True,
            legendgroup="Estimate",
        )
    )
    # Estimate Line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=results["estimates"],
            mode="lines",
            name="Estimate",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
        )
    )
    fig.update_layout(
        title="Convergence: CTR Simple Diff with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Estimate",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_ctr_diff_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")


def test_plot_ctr_pct_diff_convergence(get_test_data):
    """Plots convergence of CPDE estimates vs. true value for CTR percent difference."""
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    df, strata_info, true_values = get_test_data
    sample_sizes = list(range(50, 2001, 50))

    ctr_metric = Metric(
        numerator=UMetric(col="clicks", agg="MAX", fill_na=0, name="Clicks"),
        denominator=UMetric(col="impressions", agg="MAX", fill_na=0, name="Impressions"),
        numerator_agg="SUM",
        denominator_agg="SUM",
    )
    estimator_constructor = CPDEConstructor(
        metric=ctr_metric,
        strata_info=strata_info,
        comparison_pair=comparison_pair,
        diff_type="pcnt_diff",
        variant_col="variant",
        name="CTR_Pcnt_Diff",
    )

    # Collect estimates and CIs
    results = {"estimates": [], "ci_lower": [], "ci_upper": []}
    for n in sample_sizes:
        sample_df = df[:n]  # First n rows (data is shuffled)
        est = estimator_constructor.construct()
        jk_result = est.calc_jk_stats(sample_df, num_buckets=NUM_BUCKETS, ci_coverage=CI_COVERAGE)
        results["estimates"].append(jk_result["estimator_value"][MEAN_RATIO_INDEX])
        results["ci_lower"].append(jk_result["ci"][MEAN_RATIO_INDEX][0])
        results["ci_upper"].append(jk_result["ci"][MEAN_RATIO_INDEX][1])

    # Create plot
    fig = go.Figure()
    # True Value
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_values["ctr_pct"]] * len(sample_sizes),
            mode="lines",
            name=f"True Value ({true_values['ctr_pct']:.4f})",
            line=dict(color="black", dash="dot"),
            legendgroup="True",
        )
    )
    # CI Ribbon
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=results["ci_upper"] + results["ci_lower"][::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name="95% CI",
            showlegend=True,
            legendgroup="Estimate",
        )
    )
    # Estimate Line
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=results["estimates"],
            mode="lines",
            name="Estimate",
            line=dict(color="blue", width=2),
            legendgroup="Estimate",
        )
    )
    fig.update_layout(
        title="Convergence: CTR Percent Diff with 95% Jackknife CI",
        xaxis_title="Sample Size",
        yaxis_title="Estimate (%)",
        height=600,
        showlegend=True,
        template="plotly_white",
    )
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_ctr_pct_convergence.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")

# BSD 2-CLAUSE LICENSE
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini


import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from abvelocity.core.stats.estimator import Estimator
from abvelocity.core.stats.weighted_estimator import StrataInfo, WeightedEstimator

# --- Constants and Configuration ---
SEED = 1317
PLOT_WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/weighted-estimator").resolve()


# --- Helper Estimator Classes ---


class MeanEstimator(Estimator):
    """A concrete Estimator implementation for sample mean."""

    def estimator_func(self, df: pd.DataFrame, param: None = None) -> np.ndarray:
        """Computes the sample mean, returning a 1-element NumPy array."""
        return np.array([df["y"].mean()])

    # NOTE: It is assumed that the base Estimator class implements
    # calc_normal_ci and estimator_func_with_inferred_param (as used in WeightedEstimator)
    # in a way that is compatible with this MeanEstimator implementation.


@pytest.fixture
def weighted_test_data():
    """Creates a base DataFrame and StrataInfo for WeightedEstimator unit tests."""
    np.random.seed(SEED)

    data = {
        "variant": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        "y": np.concatenate(
            [
                np.repeat(10, 10),  # Mean = 10
                np.repeat(20, 10),  # Mean = 20
                np.repeat(30, 10),  # Mean = 30
            ]
        ),
    }
    df = pd.DataFrame(data)

    strata_data = {
        "variant": ["A", "B", "Missing"],
        "pop_count": [100, 400, 500],
    }
    strata_df = pd.DataFrame(strata_data).set_index("variant")

    strata_info = StrataInfo(df=strata_df, strata_count_col="pop_count")

    return df, strata_info


# ----------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------


def test_weighted_estimator_numerical_correctness(weighted_test_data):
    """Tests the final weighted estimate using known, non-stochastic means and weights."""
    df, strata_info = weighted_test_data

    variant_values = ["A", "B"]

    w_a = 0.2
    w_b = 0.8

    mean_a = df[df["variant"] == "A"]["y"].mean()
    mean_b = df[df["variant"] == "B"]["y"].mean()

    expected_weighted_mean = w_a * mean_a + w_b * mean_b

    weighted_est = WeightedEstimator(
        stratum_estimator=MeanEstimator(),
        strata_info=strata_info,
        variant_values=variant_values,
        variant_col="variant",
        standardize_weights=True,
    )

    # NOTE: The test passes here if the internal WeightedEstimator correctly handles
    # the call to stratum_estimator.estimator_func_with_inferred_param
    result = weighted_est.estimator_func(df)

    assert np.isclose(result, expected_weighted_mean)


def test_weighted_estimator_empty_strata_and_missing_variant_handling(weighted_test_data):
    """Tests that empty strata and missing variants are handled correctly with zero estimates/weights and warnings."""
    df, strata_info = weighted_test_data

    variant_values = ["A", "C", "Missing"]

    w_a = 100 / 600
    w_c = 0.0
    w_missing = 500 / 600

    mean_a = df[df["variant"] == "A"]["y"].mean()
    mean_c = df[df["variant"] == "C"]["y"].mean()

    expected_weighted_mean = w_a * mean_a + w_c * mean_c + w_missing * 0.0

    # FIX: Wrap initialization and execution in the warning block to catch all warnings
    # (weight calculation is done during init and missing strata warning is done during func call).
    with pytest.warns(UserWarning) as record:
        weighted_est = WeightedEstimator(
            stratum_estimator=MeanEstimator(),
            strata_info=strata_info,
            variant_values=variant_values,
            variant_col="variant",
            standardize_weights=True,
        )
        result = weighted_est.estimator_func(df)

    warnings_list = [w.message.args[0] for w in record]

    # Assertions for the expected warnings
    assert any("Variant value 'C' not found" in w for w in warnings_list)
    assert any("Strata 'Missing' in column 'variant' is empty" in w for w in warnings_list)

    assert np.isclose(result, expected_weighted_mean)


# ----------------------------------------------------------------------
# Plotting Test
# ----------------------------------------------------------------------


def simulate_heterogeneous_data():
    """
    Simulates data using 5 countries as strata.
    Returns the sample DataFrame, strata info, true population mean, and biased simple mean limit.
    """
    countries = ["USA", "Russia", "Iran", "Japan", "Yemen"]
    true_means = [100, 20, 10, 150, 80]

    sd = 10.0
    props = [0.4, 0.2, 0.2, 0.1, 0.1]
    n = 10000
    df_list = []

    counts = []
    for i in range(len(countries)):
        prop = props[i]
        country = countries[i]
        true_mean = true_means[i]
        n_strata = int(n * prop)
        counts.append(n_strata)
        data_strata = np.random.normal(loc=true_mean, scale=sd, size=n_strata)
        df_list.append(pd.DataFrame({"y": data_strata, "country": country}))

    df = pd.concat(df_list, ignore_index=True)
    df = df.sample(frac=1, random_state=None).reset_index(drop=True)

    strata_df = pd.DataFrame({"country": countries, "count": counts})
    strata_df.index = countries
    strata_info = StrataInfo(df=strata_df, strata_count_col="count")
    true_population_mean = sum(prop * true_mean for prop, true_mean in zip(props, true_means))

    return df, strata_info, true_population_mean


def test_plot_weighted_vs_simple_mean():
    """
    Plots the estimated value and Jackknife Confidence Interval (CI) for both the
    Weighted Mean and Simple Mean across increasing sample sizes.
    """
    os.makedirs(PLOT_WRITE_PATH, exist_ok=True)

    simple_mean_estimator = MeanEstimator(name="Simple Mean")
    sample_sizes = list(range(25, 501, 5))

    simple_means, simple_ci_lower, simple_ci_upper = [], [], []
    weighted_means, weighted_ci_lower, weighted_ci_upper = [], [], []

    # Get true targets for plotting purposes
    df, strata_info, true_mean = simulate_heterogeneous_data()
    variant_values = list(strata_info.df.index)
    ci_coverage = 0.95
    num_buckets = 20

    for n in sample_sizes:
        sample_df = df[:n]

        # WeightedEstimator must be re-initialized inside the loop
        weighted_mean_estimator = WeightedEstimator(
            stratum_estimator=MeanEstimator(),
            strata_info=strata_info,
            variant_values=variant_values,
            variant_col="country",
            standardize_weights=True,
            name="Weighted Mean",
        )

        # --- Weighted Estimator (using calc_jk_stats) ---
        weighted_jk_result = weighted_mean_estimator.calc_jk_stats(df=sample_df, num_buckets=num_buckets, ci_coverage=ci_coverage)
        # Extract mean and CI bounds (assuming k=1, so we take the first element)
        weighted_means.append(weighted_jk_result["estimator_value"][0])
        weighted_ci_lower.append(weighted_jk_result["ci"][0][0])
        weighted_ci_upper.append(weighted_jk_result["ci"][0][1])

        # --- Simple Estimator (using calc_jk_stats) ---
        simple_jk_result = simple_mean_estimator.calc_jk_stats(df=sample_df, num_buckets=num_buckets, ci_coverage=ci_coverage)
        # Extract mean and CI bounds (assuming k=1, so we take the first element)
        simple_means.append(simple_jk_result["estimator_value"][0])
        simple_ci_lower.append(simple_jk_result["ci"][0][0])
        simple_ci_upper.append(simple_jk_result["ci"][0][1])

    # Create Plotly figure
    fig = go.Figure()

    # --- True Mean Lines ---
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=[true_mean] * len(sample_sizes),
            mode="lines",
            name=f"True Mean ({true_mean:.2f})",
            line=dict(color="blue", dash="dash"),
            legendgroup="True_W",
        )
    )

    # --- Confidence Interval Bands ---

    # 1. Weighted Mean CI Band (Light Blue)
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=weighted_ci_upper + weighted_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(31, 119, 180, 0)"),
            name=f"Weighted CI ({int(ci_coverage*100)}%)",
            showlegend=True,
            legendgroup="Weighted_Est",
        )
    )

    # 2. Simple Mean CI Band (Light Red)
    fig.add_trace(
        go.Scatter(
            x=sample_sizes + sample_sizes[::-1],
            y=simple_ci_upper + simple_ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.1)",
            line=dict(color="rgba(255, 0, 0, 0)"),
            name=f"Simple CI ({int(ci_coverage*100)}%)",
            showlegend=True,
            legendgroup="Simple_Est",
        )
    )

    # --- Estimated Values (Line on top of CI) ---

    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=weighted_means,
            mode="lines",
            name="Weighted Mean Estimate",
            line=dict(color="blue", width=2),
            legendgroup="Weighted_Est",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=simple_means,
            mode="lines",
            name="Simple Mean Estimate",
            line=dict(color="red", width=2),
            legendgroup="Simple_Est",
        )
    )

    # Layout
    fig.update_layout(
        title="Weighted vs. Simple Mean: Bias and Precision (95% CI - Jackknife)",
        xaxis_title="Total Sample Size (n)",
        yaxis_title="Estimated Mean Value",
        height=600,
        showlegend=True,
        template="plotly_white",
    )

    # Save plot
    file_name = PLOT_WRITE_PATH.joinpath("test_plot_weighted_vs_simple_mean_with_jk_ci.html")
    fig.write_html(str(file_name), include_plotlyjs="cdn")

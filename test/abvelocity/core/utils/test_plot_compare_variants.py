# BSD 2-CLAUSE LICENSE
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from abvelocity.core.utils.plot_compare_variants import plot_compare_variants

# Single shared output directory for all tests in this module
WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/plot_compare_variants/")
WRITE_PATH.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def sample_variant_df():
    """Categorical variant fixture with 'nan' in both positions."""
    data = {
        "metric": [
            "yield",
            "yield",
            "yield",
            "yield",
            "yield",
            "defect_rate",
            "defect_rate",
            "defect_rate",
            "defect_rate",
            "cycle_time",
            "cycle_time",
            "cycle_time",
        ],
        "variant": [
            ("control", "control"),
            ("control", "v1"),
            ("control", "v2"),
            ("control", "nan"),
            ("treat", "control"),
            ("treat", "v1"),
            ("treat", "nan"),
            ("nan", "control"),
            ("nan", "v2"),
            ("control", "control"),
            ("treat", "v1"),
            ("nan", "nan"),
        ],
        "sample_count": [
            40000,
            34000,
            32000,
            5000,
            36000,
            35000,
            8000,
            2000,
            1500,
            40000,
            35000,
            3000,
        ],
        "mean": [28.35, 47.63, 42.10, 25.50, 39.15, 33.54, 37.80, 0.050, 0.060, 5.50, 6.20, 5.80],
        "sd": [55.23, 143.0, 130.5, 50.0, 73.18, 72.02, 78.5, 0.08, 0.09, 1.08, 1.30, 1.20],
    }
    df = pd.DataFrame(data)
    df["variant_str"] = df["variant"].apply(lambda x: str(x))
    return df


@pytest.fixture
def sample_continuous_df():
    """Continuous variant fixture – variant is a float 'time' from 0 to 10."""
    np.random.seed(42)

    times = [0.0, 1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]

    data = {
        "metric": ["response_value"] * len(times),
        "variant": times,
        "sample_count": [5000] * len(times),
        "mean": [10 + 2 * t + np.random.normal(0, 1) for t in times],
        "sd": [1.5 + 0.2 * t for t in times],
    }
    df = pd.DataFrame(data)
    return df


def test_plot_compare_variants_per_metric(sample_variant_df):
    """Generate per-metric grouped bar plots (with error bars)."""
    figs = plot_compare_variants(
        df=sample_variant_df,
        x_col="variant_str",
        y_col="mean",
        split_col="metric",
        err_col="sd",
        title_prefix="Metric: ",
        width=900,
        height_per_plot=500,
    )

    assert len(figs) == sample_variant_df["metric"].nunique()

    for metric, fig in figs.items():
        html_path = WRITE_PATH / f"{metric}_per_metric.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        assert html_path.exists()


def test_plot_compare_variants_single_plot_with_errors(sample_variant_df):
    """Single grouped bar plot with error bars."""
    df_one = sample_variant_df[sample_variant_df["metric"] == "yield"].copy()

    figs = plot_compare_variants(
        df=df_one,
        x_col="variant_str",
        y_col="mean",
        split_col=None,
        err_col="sd",
        title_prefix="Yield Across Variants",
    )

    fig = list(figs.values())[0]
    html_path = WRITE_PATH / "yield_single_with_errors.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    assert html_path.exists()


def test_plot_compare_variants_single_plot_no_errors(sample_variant_df):
    """Single grouped bar plot without error bars."""
    df_one = sample_variant_df[sample_variant_df["metric"] == "yield"].copy()

    figs = plot_compare_variants(
        df=df_one,
        x_col="variant_str",
        y_col="mean",
        split_col=None,
        err_col=None,
        title_prefix="Yield Across Variants (No Error Bars)",
    )

    fig = list(figs.values())[0]
    html_path = WRITE_PATH / "yield_single_no_errors.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    assert html_path.exists()


def test_plot_compare_variants_normal_ci(sample_variant_df):
    """Single grouped bar plot with computed normal confidence intervals."""
    df_one = sample_variant_df[sample_variant_df["metric"] == "yield"].copy()

    figs = plot_compare_variants(
        df=df_one,
        x_col="variant_str",
        y_col="mean",
        split_col=None,
        err_col="normal_ci",
        sample_size_col="sample_count",
        ci_coverage=0.99,
        title_prefix="Yield with 99% Normal CI",
    )

    fig = list(figs.values())[0]
    html_path = WRITE_PATH / "yield_normal_ci_99.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    assert html_path.exists()


def test_plot_compare_variants_normal_ci_raises_error(sample_variant_df):
    """Ensure ValueError is raised when sample_size_col is missing for normal_ci."""
    df_one = sample_variant_df[sample_variant_df["metric"] == "yield"].copy()

    with pytest.raises(ValueError, match="sample_size_col must be provided"):
        plot_compare_variants(
            df=df_one,
            x_col="variant_str",
            y_col="mean",
            err_col="normal_ci",
            sample_size_col=None,
        )


def test_plot_compare_variants_custom_order(sample_variant_df):
    """Grouped bar plot with explicit variant ordering."""
    custom_order = [
        ("treat", "nan"),
        ("treat", "v1"),
        ("treat", "control"),
        ("control", "nan"),
        ("control", "v2"),
        ("control", "v1"),
        ("control", "control"),
        ("nan", "control"),
        ("nan", "v2"),
        ("nan", "nan"),
    ]
    custom_order_str = [str(v) for v in custom_order]

    df_one = sample_variant_df[sample_variant_df["metric"] == "defect_rate"].copy()

    figs = plot_compare_variants(
        df=df_one,
        x_col="variant_str",
        y_col="mean",
        split_col=None,
        err_col="sd",
        variant_order=custom_order_str,
    )

    fig = list(figs.values())[0]
    html_path = WRITE_PATH / "defect_rate_ordered.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    assert html_path.exists()


def test_plot_compare_variants_continuous_with_errors(sample_continuous_df):
    """Line plot with error bands for continuous variant."""
    figs = plot_compare_variants(
        df=sample_continuous_df,
        x_col="variant",
        y_col="mean",
        split_col=None,
        err_col="sd",
        title_prefix="Response over Time",
    )

    fig = list(figs.values())[0]
    html_path = WRITE_PATH / "continuous_time_line_with_errors.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    assert html_path.exists()


def test_plot_compare_variants_continuous_no_errors(sample_continuous_df):
    """Line plot without error bands for continuous variant."""
    figs = plot_compare_variants(
        df=sample_continuous_df,
        x_col="variant",
        y_col="mean",
        split_col=None,
        err_col=None,
        title_prefix="Response over Time (No Error Bands)",
    )

    fig = list(figs.values())[0]
    html_path = WRITE_PATH / "continuous_time_line_no_errors.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    assert html_path.exists()

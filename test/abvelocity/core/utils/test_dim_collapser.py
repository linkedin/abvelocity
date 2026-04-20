# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
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
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import pandas as pd
import pytest
from abvelocity.core.utils.dim_collapser import DimCollapser

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def single_dim_df():
    """Daily values for 4 regions across 3 dates."""
    return pd.DataFrame(
        {
            "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4 + ["2024-01-03"] * 4,
            "region": ["us", "gb", "ca", "au"] * 3,
            "value": [100, 80, 30, 10, 110, 70, 25, 5, 120, 90, 20, 8],
        }
    )


@pytest.fixture
def multi_dim_df():
    """Daily values for (product, platform) combinations."""
    return pd.DataFrame(
        {
            "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
            "product": ["pro", "pro", "basic", "basic"] * 2,
            "platform": ["web", "mobile", "web", "mobile"] * 2,
            "value": [200, 150, 50, 10, 210, 140, 45, 8],
        }
    )


# ── single dim ────────────────────────────────────────────────────────────────


def test_single_dim_top_k_kept(single_dim_df):
    collapser = DimCollapser(dims=["region"], k=2, rank_by="value", group_by=["date"])
    result = collapser.apply(single_dim_df)
    regions = set(result["region"].unique())
    # top 2 by total: us (330), gb (240); rest → "other"
    assert regions == {"us", "gb", "other"}


def test_single_dim_other_values_summed(single_dim_df):
    collapser = DimCollapser(dims=["region"], k=2, rank_by="value", group_by=["date"])
    result = collapser.apply(single_dim_df)
    # On 2024-01-01: ca=30, au=10 → other=40
    other_day1 = result.loc[(result["date"] == "2024-01-01") & (result["region"] == "other"), "value"].values[0]
    assert other_day1 == 40


def test_single_dim_top_values_unchanged(single_dim_df):
    collapser = DimCollapser(dims=["region"], k=2, rank_by="value", group_by=["date"])
    result = collapser.apply(single_dim_df)
    us_day1 = result.loc[(result["date"] == "2024-01-01") & (result["region"] == "us"), "value"].values[0]
    assert us_day1 == 100


def test_single_dim_k_larger_than_unique(single_dim_df):
    # k=10 > 4 unique regions — no "other" bucket should appear
    collapser = DimCollapser(dims=["region"], k=10, rank_by="value", group_by=["date"])
    result = collapser.apply(single_dim_df)
    assert "other" not in result["region"].values


def test_single_dim_custom_fallback(single_dim_df):
    collapser = DimCollapser(dims=["region"], k=1, rank_by="value", group_by=["date"], fallback="rest")
    result = collapser.apply(single_dim_df)
    assert "rest" in result["region"].values
    assert "other" not in result["region"].values


def test_single_dim_k1(single_dim_df):
    collapser = DimCollapser(dims=["region"], k=1, rank_by="value", group_by=["date"])
    result = collapser.apply(single_dim_df)
    assert set(result["region"].unique()) == {"us", "other"}


# ── multi dim ─────────────────────────────────────────────────────────────────


def test_multi_dim_top_k_combinations(multi_dim_df):
    collapser = DimCollapser(dims=["product", "platform"], k=2, rank_by="value", group_by=["date"])
    result = collapser.apply(multi_dim_df)
    combos = set(zip(result["product"], result["platform"]))
    # top 2 by total: (pro, web)=410, (pro, mobile)=290; rest → (other, other)
    assert ("pro", "web") in combos
    assert ("pro", "mobile") in combos
    assert ("other", "other") in combos


def test_multi_dim_other_bucket_summed(multi_dim_df):
    collapser = DimCollapser(dims=["product", "platform"], k=2, rank_by="value", group_by=["date"])
    result = collapser.apply(multi_dim_df)
    # On 2024-01-01: (basic, web)=50 + (basic, mobile)=10 → (other, other)=60
    other_day1 = result.loc[
        (result["date"] == "2024-01-01") & (result["product"] == "other") & (result["platform"] == "other"),
        "value",
    ].values[0]
    assert other_day1 == 60


def test_multi_dim_no_group_by(multi_dim_df):
    # Without group_by, collapse and sum across all dates
    collapser = DimCollapser(dims=["product", "platform"], k=1, rank_by="value")
    result = collapser.apply(multi_dim_df)
    combos = set(zip(result["product"], result["platform"]))
    assert ("pro", "web") in combos
    assert ("other", "other") in combos


# ── original df not mutated ────────────────────────────────────────────────────


def test_does_not_mutate_input(single_dim_df):
    original_regions = single_dim_df["region"].tolist()
    collapser = DimCollapser(dims=["region"], k=1, rank_by="value", group_by=["date"])
    collapser.apply(single_dim_df)
    assert single_dim_df["region"].tolist() == original_regions

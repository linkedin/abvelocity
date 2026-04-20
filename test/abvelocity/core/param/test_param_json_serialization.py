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
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import pandas as pd
from abvelocity.core.param.derived_expt_stats import DerivedExptStats
from abvelocity.core.param.expt_info import ExptInfo, MultiExptInfo
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.variant import Variant


def test_umetric_json_round_trip():
    umetric = UMetric(col="signups", agg="SUM", fill_na=0, name="signups")
    result = UMetric.from_json(umetric.to_json())

    assert result.col == umetric.col
    assert result.agg == umetric.agg
    assert result.fill_na == umetric.fill_na
    assert result.name == umetric.name


def test_metric_json_round_trip_with_nested_umetric():
    metric = Metric(
        numerator=UMetric(col="signups", agg="SUM", fill_na=0),
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0),
        numerator_agg="SUM",
        denominator_agg="SUM",
        name="signup_rate",
    )
    result = Metric.from_json(metric.to_json())

    assert result.name == metric.name
    assert result.numerator.col == metric.numerator.col
    assert result.denominator.col == metric.denominator.col
    assert result.numerator_agg == metric.numerator_agg


def test_variant_json_round_trip_preserves_tuple():
    variant = Variant(value=("control",))
    result = Variant.from_json(variant.to_json())

    assert result.value == ("control",)
    assert isinstance(result.value, tuple)
    assert result.name == variant.name


def test_expt_info_json_round_trip():
    expt_info = ExptInfo(
        test_key="test_key_1",
        start_date="2024-01-01-00",
        end_date="2024-02-01-00",
        control_label="control",
    )
    result = ExptInfo.from_json(expt_info.to_json())

    assert result.test_key == expt_info.test_key
    assert result.start_date == expt_info.start_date
    assert result.end_date == expt_info.end_date
    assert result.control_label == expt_info.control_label
    assert result.derived_stats is None


def test_multi_expt_info_json_round_trip():
    multi_expt_info = MultiExptInfo(
        expt_info_list=[
            ExptInfo(test_key="expt_a", start_date="2024-01-01-00"),
            ExptInfo(test_key="expt_b", start_date="2024-01-01-00"),
        ],
        merge_method="cross",
    )
    result = MultiExptInfo.from_json(multi_expt_info.to_json())

    assert len(result.expt_info_list) == 2
    assert result.expt_info_list[0].test_key == "expt_a"
    assert result.expt_info_list[1].test_key == "expt_b"
    assert result.merge_method == "cross"


def test_derived_expt_stats_json_round_trip_with_dataframes():
    df = pd.DataFrame(
        {"variant_count": [100, 200], "variant_percent": [33.3, 66.7]},
        index=["control", "enabled"],
    )
    stats = DerivedExptStats(
        total_count=300,
        total_triggered_count=280,
        total_triggered_percent=93.3,
        variant_count_df=df,
    )
    result = DerivedExptStats.from_json(stats.to_json())

    assert result.total_count == stats.total_count
    assert result.total_triggered_count == stats.total_triggered_count
    pd.testing.assert_frame_equal(result.variant_count_df, stats.variant_count_df)


def test_derived_expt_stats_json_round_trip_with_dict_of_dataframes():
    df0 = pd.DataFrame({"overlap": [0.1, 0.2]})
    df1 = pd.DataFrame({"overlap": [0.3, 0.4]})
    stats = DerivedExptStats(
        total_count=500,
        conditional_trigger_dfs={0: df0, 1: df1},
    )
    result = DerivedExptStats.from_json(stats.to_json())

    assert len(result.conditional_trigger_dfs) == 2
    pd.testing.assert_frame_equal(result.conditional_trigger_dfs[0], df0)
    pd.testing.assert_frame_equal(result.conditional_trigger_dfs[1], df1)

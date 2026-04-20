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
import pytest
from abvelocity.core.mea.entities import MEAMetricResult, MEAReport, MEAResult


@pytest.fixture
def effect_df():
    return pd.DataFrame(
        {
            "variant": ["enabled", "enabled"],
            "metric": ["signups", "retention"],
            "effect": [0.05, -0.01],
            "p_value": [0.03, 0.42],
        }
    )


@pytest.fixture
def stats_df():
    return pd.DataFrame(
        {
            "variant": ["control", "enabled"],
            "mean": [0.10, 0.105],
            "std": [0.01, 0.01],
        }
    )


def test_mea_metric_result_json_round_trip(effect_df, stats_df):
    result = MEAMetricResult(
        variant_metric_stats_df=stats_df,
        variant_effect_df_pairs=effect_df,
    )
    restored = MEAMetricResult.from_json(result.to_json())

    pd.testing.assert_frame_equal(restored.variant_metric_stats_df, stats_df)
    pd.testing.assert_frame_equal(restored.variant_effect_df_pairs, effect_df)
    assert restored.variant_effect_df_pairs_sig is None


def test_mea_result_json_round_trip(effect_df, stats_df):
    metric_result = MEAMetricResult(
        variant_metric_stats_df=stats_df,
        variant_effect_df_pairs=effect_df,
    )
    freq_df = pd.DataFrame({"variant": ["control", "enabled"], "count": [500, 510]})
    mea_result = MEAResult(
        metric_result_dict={"signups": metric_result},
        variant_freq_dict={"expt_1": freq_df},
    )
    restored = MEAResult.from_json(mea_result.to_json())

    assert set(restored.metric_result_dict.keys()) == {"signups"}
    pd.testing.assert_frame_equal(restored.metric_result_dict["signups"].variant_effect_df_pairs, effect_df)
    pd.testing.assert_frame_equal(restored.variant_freq_dict["expt_1"], freq_df)


def test_mea_report_json_round_trip():
    report = MEAReport(
        html_str="<h1>Results</h1>",
        file_names=["report.html"],
        paths=["/tmp/report.html"],
    )
    restored = MEAReport.from_json(report.to_json())

    assert restored.html_str == report.html_str
    assert restored.file_names == report.file_names
    assert restored.paths == report.paths

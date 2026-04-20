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
#
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

import math

import numpy as np
import pandas as pd
import pytest
from abvelocity.core.param.constants import METRIC_NAME_COL
from abvelocity.core.stats.swi.compute_swi import (
    CONTROL_METRIC_COL,
    DELTA_COL,
    IMPACTED_POP_COUNT_COL,
    SITE_WIDE_IMPACT_COL,
    SITE_WIDE_IMPACT_PERC_COL,
    compute_swi,
)


def create_effect_df(metric_name, delta, impacted_pop_count, control_metric_total=None):
    return pd.DataFrame(
        [
            {
                METRIC_NAME_COL: metric_name,
                DELTA_COL: delta,
                IMPACTED_POP_COUNT_COL: impacted_pop_count,
                CONTROL_METRIC_COL: control_metric_total,
            }
        ]
    )


def create_complement_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_complement_metrics_df_duplicate_metric_raises():
    complement_metrics_df = create_complement_df(
        [
            {METRIC_NAME_COL: "clicks", "numer": 30000.0},
            {METRIC_NAME_COL: "clicks", "numer": 25000.0},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        compute_swi(
            variant_effect_df=create_effect_df(
                metric_name="clicks",
                delta=0.05,
                impacted_pop_count=1000,
                control_metric_total=20000.0,
            ),
            complement_metrics_df=complement_metrics_df,
        )


def test_missing_impacted_pop_count_gives_nan():
    df = compute_swi(
        variant_effect_df=create_effect_df(
            metric_name="clicks",
            delta=0.05,
            impacted_pop_count=float("nan"),
            control_metric_total=20000.0,
        )
    )
    assert math.isnan(df[SITE_WIDE_IMPACT_COL].iloc[0])
    assert math.isnan(df[SITE_WIDE_IMPACT_PERC_COL].iloc[0])


def test_missing_delta_gives_nan():
    df = compute_swi(
        variant_effect_df=create_effect_df(
            metric_name="clicks",
            delta=None,
            impacted_pop_count=1000,
            control_metric_total=20000.0,
        )
    )
    assert math.isnan(df[SITE_WIDE_IMPACT_COL].iloc[0])
    assert math.isnan(df[SITE_WIDE_IMPACT_PERC_COL].iloc[0])


def test_multiple_metrics_mixed():
    effect_df = pd.DataFrame(
        [
            {
                METRIC_NAME_COL: "clicks",
                DELTA_COL: 0.1,
                IMPACTED_POP_COUNT_COL: 500,
                CONTROL_METRIC_COL: 20000.0,
            },
            {
                METRIC_NAME_COL: "sessions_per_member",
                DELTA_COL: (3.0, 1.0),
                IMPACTED_POP_COUNT_COL: 500,
                CONTROL_METRIC_COL: (8000.0, 4000.0),
            },
            {
                METRIC_NAME_COL: "broken",
                DELTA_COL: 0.2,
                IMPACTED_POP_COUNT_COL: float("nan"),
                CONTROL_METRIC_COL: None,
            },
        ]
    )
    complement_metrics_df = create_complement_df(
        [
            {METRIC_NAME_COL: "clicks", "numer": 30000.0},
            {METRIC_NAME_COL: "sessions_per_member", "numer": 6000.0, "denom": 3000.0},
        ]
    )

    result = compute_swi(variant_effect_df=effect_df, complement_metrics_df=complement_metrics_df)

    # clicks: simple
    exp_raw = 0.1 * 500
    exp_perc = exp_raw / (20000.0 + 30000.0) * 100
    clicks_row = result.loc[result[METRIC_NAME_COL] == "clicks"]
    assert np.isclose(clicks_row[SITE_WIDE_IMPACT_COL].iloc[0], exp_raw)
    assert np.isclose(clicks_row[SITE_WIDE_IMPACT_PERC_COL].iloc[0], exp_perc)

    # sessions_per_member: ratio
    numer_pop_total_given_control = 8000.0 + 6000.0
    denom_pop_total_given_control = 4000.0 + 3000.0
    exp_swi = (numer_pop_total_given_control + 3.0 * 500) / (
        denom_pop_total_given_control + 1.0 * 500
    ) - numer_pop_total_given_control / denom_pop_total_given_control
    ratio_row = result.loc[result[METRIC_NAME_COL] == "sessions_per_member"]
    assert np.isclose(ratio_row[SITE_WIDE_IMPACT_COL].iloc[0], exp_swi)

    # broken: NaN
    assert math.isnan(result.loc[result[METRIC_NAME_COL] == "broken", SITE_WIDE_IMPACT_COL].iloc[0])


def test_output_is_copy():
    effect_df = create_effect_df(
        metric_name="clicks",
        delta=0.05,
        impacted_pop_count=1000,
        control_metric_total=20000.0,
    )
    original_cols = list(effect_df.columns)
    compute_swi(variant_effect_df=effect_df)
    assert list(effect_df.columns) == original_cols

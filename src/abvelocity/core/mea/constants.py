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

from abvelocity.core.param.constants import CI_PERCENT_COL, METRIC_NAME_COL

JK_NUM_BUCKETS = 30
"""Defaulyt Jackknife estimation number of buckets."""

MEA_DEFAULT_METHOD = "simple"
"""MEA default method."""

# END_USER_METRIC_RESULT_KEYS = ["variant_effect_df_pairs", "variant_effect_df_pairs_sig"]
END_USER_METRIC_RESULT_KEYS = ["variant_effect_df_pairs_sig"]
"""The end user does not need all the genarted tables.
This constant is used to decide what is to be shown in the final report.
Currently we only include the comparisons.
These could be launch effects for example passed using `Launch` and then converted to `ComparisonPair`.
"""

END_USER_COLS = [
    METRIC_NAME_COL,
    "launch",
    "delta",
    "delta_percent",
    CI_PERCENT_COL,
    "p_value",
    "delta_sum",
    "impacted_counts",
    "sample_counts",
]
"""
The columns to include in metric effect tables for end user.
"""

DEFAULT_MATERIALIZE_TO_PANDAS_STAGE = "delayed"
IMPLEMENTED_MATERIALIZE_TO_PANDAS_STAGES = ["immediate", "delayed"]

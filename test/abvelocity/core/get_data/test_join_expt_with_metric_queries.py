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

from abvelocity.core.get_data.join_expt_with_metric_queries import join_expt_with_metric_queries
from abvelocity.core.param.metric import UMetric


def test_no_u_metrics_selects_all_cols():
    """
    Tests the case where no u_metrics are passed.
    The function should select all columns from both tables.
    """
    expt_query = "SELECT user_id, assignment_group FROM assignments"
    metric_query = "SELECT user_id, clicks, conversions FROM metrics"

    actual_sql = join_expt_with_metric_queries(
        expt_assignment_query=expt_query,
        metric_query=metric_query,
        expt_unit_col="user_id",
        metric_join_unit_col="user_id",
        u_metrics=None,
    )

    expected_sql = """
WITH
    expt_data AS (
        SELECT user_id, assignment_group FROM assignments
    ),
    metric_data AS (
        SELECT user_id, clicks, conversions FROM metrics
    )
SELECT
    expt_data.*,
        metric_data.*
FROM
    expt_data
LEFT JOIN
    metric_data
ON
    expt_data.user_id = metric_data.user_id
    """.strip()

    assert actual_sql.strip() == expected_sql, "Test 'no u_metrics' failed."


def test_u_metrics_with_fill_na_integers():
    """
    Tests the case with u_metrics containing integer fill_na values.
    It should use COALESCE for the specified metrics.
    """
    expt_query = "SELECT user_id, assignment_group FROM assignments"
    metric_query = "SELECT user_id, clicks, conversions FROM metrics"

    u_metrics = [
        UMetric(col="clicks", name="clicks", fill_na=0),
        UMetric(col="conversions", name="conversions", fill_na=-1),
    ]

    actual_sql = join_expt_with_metric_queries(
        expt_assignment_query=expt_query,
        metric_query=metric_query,
        expt_unit_col="user_id",
        metric_join_unit_col="user_id",
        u_metrics=u_metrics,
    )

    expected_sql = """
WITH
    expt_data AS (
        SELECT user_id, assignment_group FROM assignments
    ),
    metric_data AS (
        SELECT user_id, clicks, conversions FROM metrics
    )
SELECT
    expt_data.*,
        COALESCE(metric_data.clicks, 0) AS clicks,
        COALESCE(metric_data.conversions, -1) AS conversions
FROM
    expt_data
LEFT JOIN
    metric_data
ON
    expt_data.user_id = metric_data.user_id
    """.strip()

    assert actual_sql.strip() == expected_sql, "Test 'u_metrics with integers' failed."


def test_u_metrics_with_fill_na_string():
    """
    Tests the case with a u_metric containing a string fill_na value.
    It should use COALESCE and correctly wrap the string in quotes.
    """
    expt_query = "SELECT user_id, assignment FROM assignments"
    metric_query = "SELECT user_id, status FROM metrics"
    u_metrics = [UMetric(col="status", name="status", fill_na="N/A")]

    actual_sql = join_expt_with_metric_queries(
        expt_assignment_query=expt_query,
        metric_query=metric_query,
        expt_unit_col="user_id",
        metric_join_unit_col="user_id",
        u_metrics=u_metrics,
    )

    expected_sql = """
WITH
    expt_data AS (
        SELECT user_id, assignment FROM assignments
    ),
    metric_data AS (
        SELECT user_id, status FROM metrics
    )
SELECT
    expt_data.*,
        COALESCE(metric_data.status, 'N/A') AS status
FROM
    expt_data
LEFT JOIN
    metric_data
ON
    expt_data.user_id = metric_data.user_id
    """.strip()

    assert actual_sql.strip() == expected_sql, "Test 'u_metrics with string' failed."

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

from abvelocity.get_data.get_metric_query import GetMetricQuery
from abvelocity.param.metric import UMetric


def test_construct_query():
    get_metric_query = GetMetricQuery(table_name="some_table", date_col="datepartition")

    query = get_metric_query.construct_query(
        start_date="2022-01-01", end_date="2022-01-31", u_metrics=None, condition=None
    )

    assert (
        query
        == "SELECT * FROM some_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31'"
    )


def test_construct_query_with_metrics():
    get_metric_query = GetMetricQuery(
        table_name="some_table", date_col="datepartition", metric_table_unit_col="member_id"
    )

    u_metrics = [UMetric(col="n_sessions", agg="SUM"), UMetric(col="has_sessions", agg="MAX")]

    query = get_metric_query.construct_query(
        start_date="2022-01-01", end_date="2022-01-31", u_metrics=u_metrics, condition=None
    )

    assert query == (
        "SELECT member_id AS memberid, SUM(n_sessions) AS n_sessions, MAX(has_sessions) AS "
        "has_sessions FROM some_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' GROUP BY 1"
    )

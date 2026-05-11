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

from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.metric import COUNT_DISTINCT, UMetric


def test_construct_query():
    u_metrics_query = UMetricsQuery(
        table_name="some_table",
        date_col="datepartition",
    )

    query = u_metrics_query.construct(
        start_date="2022-01-01",
        end_date="2022-01-31",
        metric_join_unit_col="member_id",
        u_metrics=None,
        condition=None,
    )

    assert query == "SELECT * FROM some_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31'"


def test_construct_query_with_metrics():
    u_metrics_query = UMetricsQuery(
        table_name="some_table",
        date_col="datepartition",
    )

    u_metrics = [UMetric(col="n_sessions", agg="SUM"), UMetric(col="has_sessions", agg="MAX")]

    query = u_metrics_query.construct(
        start_date="2022-01-01",
        end_date="2022-01-31",
        metric_join_unit_col="member_id",
        u_metrics=u_metrics,
        condition=None,
    )

    assert query == (
        "SELECT member_id, SUM(n_sessions) AS n_sessions, MAX(has_sessions) AS "
        "has_sessions FROM some_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' GROUP BY 1"
    )


def test_construct_query_with_count_distinct():
    u_metrics_query = UMetricsQuery(
        table_name="some_table",
        date_col="datepartition",
    )

    u_metrics = [UMetric(col="member_id", agg=COUNT_DISTINCT, name="dau")]

    query = u_metrics_query.construct(
        start_date="2022-01-01",
        end_date="2022-01-31",
        metric_join_unit_col="member_id",
        u_metrics=u_metrics,
        condition=None,
    )

    assert query == (
        "SELECT member_id, COUNT(DISTINCT member_id) AS dau" " FROM some_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' GROUP BY 1"
    )


def test_pad_zero_hour_suffix_appends_dash_00_to_both_bounds():
    """``pad_zero_hour_suffix=True`` is the Blah ``datepartition`` case.

    Without padding, ``BETWEEN '2026-04-30' AND '2026-04-30'`` against
    a column whose values are like ``'2026-04-30-00'`` would silently
    drop the entire end-date partition (lex-string compare:
    ``'2026-04-30-00' > '2026-04-30'``). Padding both bounds with
    ``-00`` keeps the BETWEEN inclusive on both ends.
    """
    u_metrics_query = UMetricsQuery(
        table_name="some_table",
        date_col="datepartition",
        pad_zero_hour_suffix=True,
    )
    query = u_metrics_query.construct(
        start_date="2022-01-01",
        end_date="2022-01-31",
        metric_join_unit_col="member_id",
        u_metrics=None,
        condition=None,
    )
    assert query == "SELECT * FROM some_table WHERE datepartition BETWEEN '2022-01-01-00' AND '2022-01-31-00'"


def test_pad_zero_hour_suffix_default_is_false_for_plain_date_columns():
    """Default behavior (no padding) is unchanged for plain DATE columns."""
    u_metrics_query = UMetricsQuery(
        table_name="some_table",
        date_col="event_date",
    )
    query = u_metrics_query.construct(
        start_date="2022-01-01",
        end_date="2022-01-31",
        metric_join_unit_col="member_id",
        u_metrics=None,
        condition=None,
    )
    assert "BETWEEN '2022-01-01' AND '2022-01-31'" in query
    # The padded form must NOT appear.
    assert "-00" not in query

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
import pytest
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal
from abvelocity.ts.get_data.ts_metrics_config import TSMetricsConfig
from abvelocity.ts.get_data.ts_metrics_query import TSMetricsQuery


@pytest.fixture
def signup_metric() -> Metric:
    return Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        name="total_signups",
    )


@pytest.fixture
def signup_rate_metric() -> Metric:
    return Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="SUM",
        name="signup_rate",
    )


@pytest.fixture
def metric_family() -> MetricFamily:
    return MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )


@pytest.fixture
def metric_info(metric_family: MetricFamily, signup_metric: Metric) -> MetricInfo:
    return MetricInfo(
        metric_family=metric_family,
        metrics=[signup_metric],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )


def test_ts_metrics_query_simple_metric(metric_info: MetricInfo):
    ts_cfg = TSMetricsConfig(time_col="event_ts", freq="D")
    q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_cfg)
    sql = q.construct()

    assert_query_is_equal(
        sql,
        """
        SELECT
            DATE_TRUNC('day', event_ts) AS ts,
            SUM(signup) AS total_signups_numer,
            SUM(signup) AS total_signups,
            COUNT(*) AS sample_count
        FROM (
            SELECT
                user_id,
                event_ts,
                MAX(signup) AS signup
            FROM user_events
            WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
            GROUP BY 1, 2
        ) AS u_metrics
        GROUP BY 1
        ORDER BY 1
    """,
    )


def test_ts_metrics_query_ratio_metric(metric_family: MetricFamily, signup_rate_metric: Metric):
    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=[signup_rate_metric],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    ts_cfg = TSMetricsConfig(time_col="event_ts", freq="ME")
    q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_cfg)
    sql = q.construct()

    assert_query_is_equal(
        sql,
        """
        SELECT
            DATE_TRUNC('month', event_ts) AS ts,
            SUM(signup) AS signup_rate_numer,
            SUM(eligible) AS signup_rate_denom,
            SUM(signup) / NULLIF(SUM(eligible), 0) AS signup_rate,
            COUNT(*) AS sample_count
        FROM (
            SELECT
                user_id,
                event_ts,
                MAX(signup) AS signup,
                MAX(eligible) AS eligible
            FROM user_events
            WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
            GROUP BY 1, 2
        ) AS u_metrics
        GROUP BY 1
        ORDER BY 1
    """,
    )


def test_ts_metrics_query_with_dims(metric_family: MetricFamily, signup_metric: Metric):
    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=[signup_metric],
        start_date="2024-01-01",
        end_date="2024-01-31",
        dims=["country", "platform"],
    )
    ts_cfg = TSMetricsConfig(time_col="event_ts", freq="W")
    q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_cfg)
    sql = q.construct()

    assert_query_is_equal(
        sql,
        """
        SELECT
            DATE_TRUNC('week', event_ts) AS ts,
            country,
            platform,
            SUM(signup) AS total_signups_numer,
            SUM(signup) AS total_signups,
            COUNT(*) AS sample_count
        FROM (
            SELECT
                user_id,
                event_ts,
                country,
                platform,
                MAX(signup) AS signup
            FROM user_events
            WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
            GROUP BY 1, 2, 3, 4
        ) AS u_metrics
        GROUP BY 1, 2, 3
        ORDER BY 1
    """,
    )


def test_ts_metrics_query_with_time_format(metric_info: MetricInfo):
    ts_cfg = TSMetricsConfig(time_col="event_ts", freq="D", time_format="%Y-%m-%d:%H")
    q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_cfg)
    sql = q.construct()

    assert "DATE_TRUNC('day', DATE_PARSE(event_ts, '%Y-%m-%d:%H')) AS ts" in sql


def test_ts_metrics_query_with_condition(metric_info: MetricInfo):
    ts_cfg = TSMetricsConfig(time_col="event_ts", freq="D")
    q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_cfg, condition="country = 'US'")
    sql = q.construct()

    assert "country = 'US'" in sql


def test_ts_metrics_query_missing_dates_raises(metric_family: MetricFamily, signup_metric: Metric):
    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_metric])
    ts_cfg = TSMetricsConfig(time_col="event_ts", freq="D")
    q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_cfg)

    with pytest.raises(ValueError, match="start_date"):
        q.construct()


def test_ts_metrics_query_construct_stores_query(metric_info: MetricInfo):
    ts_cfg = TSMetricsConfig(time_col="event_ts", freq="D")
    q = TSMetricsQuery(metric_info=metric_info, ts_config=ts_cfg)
    assert q.query is None
    q.construct()
    assert q.query is not None

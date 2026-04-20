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

from abvelocity.core.get_data.agg_metrics_query import get_agg_metrics_query_from_info
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal


def test_simple_metric_no_denominator():
    """Test a simple metric without a denominator (just aggregated numerator)."""
    # Create a simple metric: total signups
    signup_metric = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        name="total_signups",
    )

    # Create metric family and info
    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_metric])

    # Get the aggregated query
    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        SUM(signup) AS total_signups_numer,
        SUM(signup) AS total_signups,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            MAX(signup) AS signup
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1
    ) AS u_metrics
    """

    assert_query_is_equal(query, expected_query)


def test_ratio_metric_with_denominator():
    """Test a ratio metric with both numerator and denominator."""
    # Create a ratio metric: signup rate
    signup_rate = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="COUNT",
        name="signup_rate",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_rate])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        SUM(signup) AS signup_rate_numer,
        COUNT(eligible) AS signup_rate_denom,
        SUM(signup) / NULLIF(COUNT(eligible), 0) AS signup_rate,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            MAX(signup) AS signup,
            MAX(eligible) AS eligible
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1
    ) AS u_metrics
    """

    assert_query_is_equal(query, expected_query)


def test_metric_with_sample_count():
    """Test a metric with explicit sample_count specification."""
    # Create a metric with sample_count
    retention_rate = Metric(
        numerator=UMetric(col="renewed", agg="MAX", fill_na=0, name="renewed"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="SUM",
        sample_count=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        name="retention_rate",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[retention_rate])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        SUM(renewed) AS retention_rate_numer,
        SUM(eligible) AS retention_rate_denom,
        SUM(renewed) / NULLIF(SUM(eligible), 0) AS retention_rate,
        SUM(eligible) AS retention_rate_sample_count,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            MAX(renewed) AS renewed,
            MAX(eligible) AS eligible
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1
    ) AS u_metrics
    """

    assert_query_is_equal(query, expected_query)


def test_multiple_metrics():
    """Test multiple metrics in the same query."""
    # Create multiple metrics
    signup_rate = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="COUNT",
        name="signup_rate",
    )

    retention_rate = Metric(
        numerator=UMetric(col="renewed", agg="MAX", fill_na=0, name="renewed"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="SUM",
        name="retention_rate",
    )

    avg_sessions = Metric(
        numerator=UMetric(col="sessions", agg="SUM", fill_na=0, name="sessions"),
        numerator_agg="AVG",
        name="avg_sessions",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_rate, retention_rate, avg_sessions])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        SUM(signup) AS signup_rate_numer,
        COUNT(eligible) AS signup_rate_denom,
        SUM(signup) / NULLIF(COUNT(eligible), 0) AS signup_rate,
        SUM(renewed) AS retention_rate_numer,
        SUM(eligible) AS retention_rate_denom,
        SUM(renewed) / NULLIF(SUM(eligible), 0) AS retention_rate,
        AVG(sessions) AS avg_sessions_numer,
        AVG(sessions) AS avg_sessions,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            MAX(signup) AS signup,
            MAX(eligible) AS eligible,
            MAX(renewed) AS renewed,
            SUM(sessions) AS sessions
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1
    ) AS u_metrics
    """

    assert_query_is_equal(query, expected_query)


def test_with_condition():
    """Test that conditions are properly propagated."""
    signup_rate = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="COUNT",
        name="signup_rate",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_rate])

    query = get_agg_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2024-01-01",
        end_date="2024-01-31",
        condition="country = 'US'",
    )

    expected_query = """
    SELECT
        SUM(signup) AS signup_rate_numer,
        COUNT(eligible) AS signup_rate_denom,
        SUM(signup) / NULLIF(COUNT(eligible), 0) AS signup_rate,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            MAX(signup) AS signup,
            MAX(eligible) AS eligible
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31' AND country = 'US'
        GROUP BY 1
    ) AS u_metrics
    """

    assert_query_is_equal(query, expected_query)


def test_query_structure():
    """Test the overall structure of the generated query."""
    signup_rate = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="COUNT",
        name="signup_rate",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_rate])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        SUM(signup) AS signup_rate_numer,
        COUNT(eligible) AS signup_rate_denom,
        SUM(signup) / NULLIF(COUNT(eligible), 0) AS signup_rate,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            MAX(signup) AS signup,
            MAX(eligible) AS eligible
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1
    ) AS u_metrics
    """

    assert_query_is_equal(query, expected_query)


def test_different_aggregation_functions():
    """Test metrics with different aggregation functions (AVG, MAX, MIN, etc.)."""
    max_sessions = Metric(
        numerator=UMetric(col="sessions", agg="SUM", fill_na=0, name="sessions"),
        numerator_agg="MAX",
        name="max_sessions",
    )

    min_sessions = Metric(
        numerator=UMetric(col="sessions", agg="SUM", fill_na=0, name="sessions"),
        numerator_agg="MIN",
        name="min_sessions",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[max_sessions, min_sessions])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        MAX(sessions) AS max_sessions_numer,
        MAX(sessions) AS max_sessions,
        MIN(sessions) AS min_sessions_numer,
        MIN(sessions) AS min_sessions,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            SUM(sessions) AS sessions
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1
    ) AS u_metrics
    """

    assert_query_is_equal(query, expected_query)


def test_multiple_metrics_with_different_sample_counts():
    """Test multiple metrics where each has its own sample_count specification."""
    # Signup rate with eligible_for_signup as sample count
    signup_rate = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible_for_signup", agg="MAX", fill_na=0, name="eligible_for_signup"),
        denominator_agg="SUM",
        sample_count=UMetric(col="eligible_for_signup", agg="MAX", fill_na=0, name="eligible_for_signup"),
        name="signup_rate",
    )

    # Retention rate with eligible_for_renewal as sample count
    retention_rate = Metric(
        numerator=UMetric(col="renewed", agg="MAX", fill_na=0, name="renewed"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible_for_renewal", agg="MAX", fill_na=0, name="eligible_for_renewal"),
        denominator_agg="SUM",
        sample_count=UMetric(col="eligible_for_renewal", agg="MAX", fill_na=0, name="eligible_for_renewal"),
        name="retention_rate",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_rate, retention_rate])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        SUM(signup) AS signup_rate_numer,
        SUM(eligible_for_signup) AS signup_rate_denom,
        SUM(signup) / NULLIF(SUM(eligible_for_signup), 0) AS signup_rate,
        SUM(eligible_for_signup) AS signup_rate_sample_count,
        SUM(renewed) AS retention_rate_numer,
        SUM(eligible_for_renewal) AS retention_rate_denom,
        SUM(renewed) / NULLIF(SUM(eligible_for_renewal), 0) AS retention_rate,
        SUM(eligible_for_renewal) AS retention_rate_sample_count,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            MAX(signup) AS signup,
            MAX(eligible_for_signup) AS eligible_for_signup,
            MAX(renewed) AS renewed,
            MAX(eligible_for_renewal) AS eligible_for_renewal
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1
    ) AS u_metrics
    """

    assert_query_is_equal(query, expected_query)


def test_simple_metric_with_single_dim():
    """Test a simple metric with a single dimension."""
    # Create a simple metric: total signups
    signup_metric = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        name="total_signups",
    )

    # Create metric family with country as dimension
    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_metric], dims=["country"])

    # Get the aggregated query
    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        country,
        SUM(signup) AS total_signups_numer,
        SUM(signup) AS total_signups,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            country,
            MAX(signup) AS signup
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1, 2
    ) AS u_metrics
    GROUP BY 1
    """

    assert_query_is_equal(query, expected_query)


def test_ratio_metric_with_multiple_dims():
    """Test a ratio metric with multiple dimensions."""
    # Create a ratio metric: signup rate
    signup_rate = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="COUNT",
        name="signup_rate",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_rate], dims=["country", "device_type"])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        country,
        device_type,
        SUM(signup) AS signup_rate_numer,
        COUNT(eligible) AS signup_rate_denom,
        SUM(signup) / NULLIF(COUNT(eligible), 0) AS signup_rate,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            country,
            device_type,
            MAX(signup) AS signup,
            MAX(eligible) AS eligible
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1, 2, 3
    ) AS u_metrics
    GROUP BY 1, 2
    """

    assert_query_is_equal(query, expected_query)


def test_multiple_metrics_with_dims():
    """Test multiple metrics with dimensions."""
    # Create multiple metrics
    signup_rate = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="COUNT",
        name="signup_rate",
    )

    avg_sessions = Metric(
        numerator=UMetric(col="sessions", agg="SUM", fill_na=0, name="sessions"),
        numerator_agg="AVG",
        name="avg_sessions",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_rate, avg_sessions], dims=["country"])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        country,
        SUM(signup) AS signup_rate_numer,
        COUNT(eligible) AS signup_rate_denom,
        SUM(signup) / NULLIF(COUNT(eligible), 0) AS signup_rate,
        AVG(sessions) AS avg_sessions_numer,
        AVG(sessions) AS avg_sessions,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            country,
            MAX(signup) AS signup,
            MAX(eligible) AS eligible,
            SUM(sessions) AS sessions
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1, 2
    ) AS u_metrics
    GROUP BY 1
    """

    assert_query_is_equal(query, expected_query)


def test_dims_with_date_col():
    """Test using date column as a dimension to get daily metrics."""
    signup_metric = Metric(
        numerator=UMetric(col="signup", agg="MAX", fill_na=0, name="signup"),
        numerator_agg="SUM",
        name="total_signups",
    )

    # Using event_date as a dimension
    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[signup_metric], dims=["event_date"])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        event_date,
        SUM(signup) AS total_signups_numer,
        SUM(signup) AS total_signups,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            event_date,
            MAX(signup) AS signup
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1, 2
    ) AS u_metrics
    GROUP BY 1
    """

    assert_query_is_equal(query, expected_query)


def test_dims_with_sample_count():
    """Test dimensions with metric-specific sample counts."""
    retention_rate = Metric(
        numerator=UMetric(col="renewed", agg="MAX", fill_na=0, name="renewed"),
        numerator_agg="SUM",
        denominator=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        denominator_agg="SUM",
        sample_count=UMetric(col="eligible", agg="MAX", fill_na=0, name="eligible"),
        name="retention_rate",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=UMetricsQuery(table_name="user_events", date_col="event_date"),
        metric_join_unit_col="user_id",
    )

    metric_info = MetricInfo(metric_family=metric_family, metrics=[retention_rate], dims=["country", "device_type"])

    query = get_agg_metrics_query_from_info(metric_info=metric_info, start_date="2024-01-01", end_date="2024-01-31")

    expected_query = """
    SELECT
        country,
        device_type,
        SUM(renewed) AS retention_rate_numer,
        SUM(eligible) AS retention_rate_denom,
        SUM(renewed) / NULLIF(SUM(eligible), 0) AS retention_rate,
        SUM(eligible) AS retention_rate_sample_count,
        COUNT(*) AS sample_count
    FROM (
        SELECT
            user_id,
            country,
            device_type,
            MAX(renewed) AS renewed,
            MAX(eligible) AS eligible
        FROM user_events
        WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
        GROUP BY 1, 2, 3
    ) AS u_metrics
    GROUP BY 1, 2
    """

    assert_query_is_equal(query, expected_query)

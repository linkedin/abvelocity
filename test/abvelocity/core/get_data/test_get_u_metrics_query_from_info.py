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

import pytest
from abvelocity.core.get_data.get_u_metrics_query_from_info import get_u_metrics_query_from_info
from abvelocity.core.get_data.u_metrics_query import UMetricsQuery
from abvelocity.core.param.metric import Metric, UMetric
from abvelocity.core.param.metric_family import MetricFamily
from abvelocity.core.param.metric_info import MetricInfo


def test_get_u_metrics_query_from_info_basic():
    """Test basic query generation without conditions."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    metrics = [
        Metric(
            name="sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
    ]

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="member_id",
        metrics=metrics,
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=metrics,
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2022-01-01",
        end_date="2022-01-31",
    )

    # Assert
    expected_query = (
        "SELECT member_id, SUM(n_sessions) AS n_sessions " "FROM test_metrics_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' " "GROUP BY 1"
    )
    assert query == expected_query


def test_get_u_metrics_query_from_info_with_condition():
    """Test query generation with a condition in metric_info."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    metrics = [
        Metric(
            name="sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
    ]

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="member_id",
        metrics=metrics,
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=metrics,
        condition="country = 'US'",
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2022-01-01",
        end_date="2022-01-31",
    )

    # Assert - condition should be added to the WHERE clause
    expected_query = (
        "SELECT member_id, SUM(n_sessions) AS n_sessions "
        "FROM test_metrics_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' "
        "AND country = 'US' GROUP BY 1"
    )
    assert query == expected_query


def test_get_u_metrics_query_from_info_multiple_metrics():
    """Test query generation with multiple metrics."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    metrics = [
        Metric(
            name="sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
        Metric(
            name="clicks",
            numerator=UMetric(col="n_clicks", agg="SUM", fill_na=0),
        ),
        Metric(
            name="max_duration",
            numerator=UMetric(col="session_duration", agg="MAX", fill_na=0),
        ),
    ]

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="user_id",
        metrics=metrics,
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=metrics,
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2022-01-01",
        end_date="2022-01-31",
    )

    # Assert
    expected_query = (
        "SELECT user_id, SUM(n_sessions) AS n_sessions, SUM(n_clicks) AS n_clicks, "
        "MAX(session_duration) AS session_duration "
        "FROM test_metrics_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' "
        "GROUP BY 1"
    )
    assert query == expected_query


def test_get_u_metrics_query_from_info_with_query_params():
    """Test query generation with additional u_metrics_query_params."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    metrics = [
        Metric(
            name="sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
    ]

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="member_id",
        metrics=metrics,
        u_metrics_query_params={},  # Additional params can be passed here
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=metrics,
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2022-01-01",
        end_date="2022-01-31",
    )

    # Assert
    expected_query = (
        "SELECT member_id, SUM(n_sessions) AS n_sessions " "FROM test_metrics_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' " "GROUP BY 1"
    )
    assert query == expected_query


def test_get_u_metrics_query_from_info_raises_error_when_metrics_none():
    """Test that TypeError is raised when metrics resolve to None (from get_u_metrics)."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="member_id",
        metrics=None,  # No default metrics
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=None,  # No metrics specified
    )

    # Execute & Assert
    # Note: get_u_metrics raises TypeError when passed None, not ValueError
    with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
        get_u_metrics_query_from_info(
            metric_info=metric_info,
            start_date="2022-01-01",
            end_date="2022-01-31",
        )


def test_get_u_metrics_query_from_info_uses_metric_family_defaults():
    """Test that metrics default from metric_family when not specified in metric_info."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    default_metrics = [
        Metric(
            name="default_sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
    ]

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="member_id",
        metrics=default_metrics,
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=None,  # Should use default from metric_family
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2022-01-01",
        end_date="2022-01-31",
    )

    # Assert - should use the default metrics from metric_family
    expected_query = (
        "SELECT member_id, SUM(n_sessions) AS n_sessions " "FROM test_metrics_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' " "GROUP BY 1"
    )
    assert query == expected_query


def test_get_u_metrics_query_from_info_different_date_ranges():
    """Test query generation with different date ranges."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="event_date",
    )

    metrics = [
        Metric(
            name="sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
    ]

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="member_id",
        metrics=metrics,
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=metrics,
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2023-06-01",
        end_date="2023-12-31",
    )

    # Assert
    expected_query = (
        "SELECT member_id, SUM(n_sessions) AS n_sessions " "FROM test_metrics_table WHERE event_date BETWEEN '2023-06-01' AND '2023-12-31' " "GROUP BY 1"
    )
    assert query == expected_query


def test_get_u_metrics_query_from_info_with_single_dim():
    """Test query generation with a single dimension."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    metrics = [
        Metric(
            name="sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
    ]

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="member_id",
        metrics=metrics,
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=metrics,
        dims=["country"],
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2022-01-01",
        end_date="2022-01-31",
    )

    # Assert - country should be in SELECT and GROUP BY
    expected_query = (
        "SELECT member_id, country, SUM(n_sessions) AS n_sessions "
        "FROM test_metrics_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' "
        "GROUP BY 1, 2"
    )
    assert query == expected_query


def test_get_u_metrics_query_from_info_with_multiple_dims():
    """Test query generation with multiple dimensions."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    metrics = [
        Metric(
            name="sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
        Metric(
            name="clicks",
            numerator=UMetric(col="n_clicks", agg="SUM", fill_na=0),
        ),
    ]

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="user_id",
        metrics=metrics,
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=metrics,
        dims=["country", "device_type"],
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2022-01-01",
        end_date="2022-01-31",
    )

    # Assert - both dims should be in SELECT and GROUP BY
    expected_query = (
        "SELECT user_id, country, device_type, SUM(n_sessions) AS n_sessions, SUM(n_clicks) AS n_clicks "
        "FROM test_metrics_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' "
        "GROUP BY 1, 2, 3"
    )
    assert query == expected_query


def test_get_u_metrics_query_from_info_with_date_col_as_dim():
    """Test query generation when date column is used as a dimension."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    metrics = [
        Metric(
            name="sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
    ]

    # Using datepartition as a dimension to get daily metrics
    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="member_id",
        metrics=metrics,
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=metrics,
        dims=["datepartition"],
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2022-01-01",
        end_date="2022-01-31",
    )

    # Assert - datepartition should be in SELECT and GROUP BY
    expected_query = (
        "SELECT member_id, datepartition, SUM(n_sessions) AS n_sessions "
        "FROM test_metrics_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' "
        "GROUP BY 1, 2"
    )
    assert query == expected_query


def test_get_u_metrics_query_from_info_with_dims_and_condition():
    """Test query generation with both dimensions and condition."""
    # Setup
    u_metrics_query = UMetricsQuery(
        table_name="test_metrics_table",
        date_col="datepartition",
    )

    metrics = [
        Metric(
            name="sessions",
            numerator=UMetric(col="n_sessions", agg="SUM", fill_na=0),
        ),
    ]

    metric_family = MetricFamily(
        name="test_family",
        u_metrics_query=u_metrics_query,
        metric_join_unit_col="member_id",
        metrics=metrics,
    )

    metric_info = MetricInfo(
        metric_family=metric_family,
        metrics=metrics,
        dims=["country", "device_type"],
        condition="is_active = TRUE",
    )

    # Execute
    query = get_u_metrics_query_from_info(
        metric_info=metric_info,
        start_date="2022-01-01",
        end_date="2022-01-31",
    )

    # Assert - dims in SELECT/GROUP BY and condition in WHERE
    expected_query = (
        "SELECT member_id, country, device_type, SUM(n_sessions) AS n_sessions "
        "FROM test_metrics_table WHERE datepartition BETWEEN '2022-01-01' AND '2022-01-31' "
        "AND is_active = TRUE GROUP BY 1, 2, 3"
    )
    assert query == expected_query

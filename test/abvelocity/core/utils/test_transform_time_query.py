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
from abvelocity.core.utils.transform_time_query import TransformTimeQuery, get_time_conversion_expression


def test_get_time_conversion_expression_unix_ms_no_truncation():
    """Test UNIX milliseconds conversion without truncation."""
    expected_sql = "FROM_UNIXTIME(event_time / 1000.0) AS new_timestamp_col"
    actual_sql = get_time_conversion_expression("event_time", "unix_ms", "new_timestamp_col")
    assert actual_sql == expected_sql


def test_get_time_conversion_expression_unix_s_no_truncation():
    """Test UNIX seconds conversion without truncation."""
    expected_sql = "FROM_UNIXTIME(event_time_s) AS new_timestamp_col"
    actual_sql = get_time_conversion_expression("event_time_s", "unix_s", "new_timestamp_col")
    assert actual_sql == expected_sql


def test_get_time_conversion_expression_string_no_truncation():
    """Test string to timestamp conversion without truncation."""
    expected_sql = "CAST(event_datetime_str AS TIMESTAMP) AS new_timestamp_col"
    actual_sql = get_time_conversion_expression("event_datetime_str", "string", "new_timestamp_col")
    assert actual_sql == expected_sql


def test_get_time_conversion_expression_timestamp_no_truncation():
    """Test already timestamp type without truncation."""
    expected_sql = "existing_ts AS new_timestamp_col"
    actual_sql = get_time_conversion_expression("existing_ts", "timestamp", "new_timestamp_col")
    assert actual_sql == expected_sql


def test_get_time_conversion_expression_unix_ms_with_second_truncation():
    """Test UNIX milliseconds with 'second' truncation."""
    expected_sql = "DATE_TRUNC('SECOND', FROM_UNIXTIME(event_time / 1000.0)) AS new_timestamp_col"
    actual_sql = get_time_conversion_expression("event_time", "unix_ms", "new_timestamp_col", "second")
    assert actual_sql == expected_sql


def test_get_time_conversion_expression_unix_s_with_minute_truncation():
    """Test UNIX seconds with 'minute' truncation."""
    expected_sql = "DATE_TRUNC('MINUTE', FROM_UNIXTIME(event_time_s)) AS new_timestamp_col"
    actual_sql = get_time_conversion_expression("event_time_s", "unix_s", "new_timestamp_col", "minutes")
    assert actual_sql == expected_sql


def test_get_time_conversion_expression_string_with_hour_truncation():
    """Test string to timestamp with 'hour' truncation."""
    expected_sql = "DATE_TRUNC('HOUR', CAST(event_datetime_str AS TIMESTAMP)) AS new_timestamp_col"
    actual_sql = get_time_conversion_expression("event_datetime_str", "string", "new_timestamp_col", "hour")
    assert actual_sql == expected_sql


def test_get_time_conversion_expression_timestamp_with_day_truncation():
    """Test already timestamp type with 'day' truncation."""
    expected_sql = "DATE_TRUNC('DAY', existing_ts) AS new_timestamp_col"
    actual_sql = get_time_conversion_expression("existing_ts", "timestamp", "new_timestamp_col", "days")
    assert actual_sql == expected_sql


def test_get_time_conversion_expression_unsupported_time_col_format():
    """Test unsupported time_col_format raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        get_time_conversion_expression("time_col", "invalid_format", "new_time_col")
    assert "Unsupported time_col_format: invalid_format" in str(excinfo.value)  # Updated assertion


def test_get_time_conversion_expression_unsupported_time_unit():
    """Test unsupported time_unit raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        get_time_conversion_expression("time_col", "unix_ms", "new_time_col", "week")
    assert "Unsupported time_unit: 'week'" in str(excinfo.value)  # Updated assertion


def test_transform_time_query_basic_generation_unix_ms():
    """Test basic query generation with unix_ms format and no truncation."""
    query_generator = TransformTimeQuery(
        table_name="my_raw_events",
        original_time_col="event_timestamp_ms",
        time_col_format="unix_ms",
        new_time_col="processed_ts",
    )
    expected_query = "(SELECT *,\n" "       FROM_UNIXTIME(event_timestamp_ms / 1000.0) AS processed_ts\n" "FROM (my_raw_events))"
    assert query_generator.gen() == expected_query


def test_transform_time_query_string_format_no_truncation():
    """Test query generation with string format and no truncation."""
    query_generator = TransformTimeQuery(
        table_name="sales_data",
        original_time_col="order_datetime_str",
        time_col_format="string",
        new_time_col="order_ts",
    )
    expected_query = "(SELECT *,\n" "       CAST(order_datetime_str AS TIMESTAMP) AS order_ts\n" "FROM (sales_data))"
    assert query_generator.gen() == expected_query


def test_transform_time_query_unix_s_format_minute_truncation():
    """Test query generation with unix_s format and minute truncation."""
    query_generator = TransformTimeQuery(
        table_name="(SELECT * FROM login_attempts WHERE status = 'SUCCESS')",
        original_time_col="login_time_s",
        time_col_format="unix_s",
        new_time_col="login_minute_ts",
        time_unit="minute",
    )
    expected_query = (
        "(SELECT *,\n"
        "       DATE_TRUNC('MINUTE', FROM_UNIXTIME(login_time_s)) AS login_minute_ts\n"
        "FROM ((SELECT * FROM login_attempts WHERE status = 'SUCCESS')))"
    )
    assert query_generator.gen() == expected_query


def test_transform_time_query_subquery_day_truncation():
    """Test query generation when table_name is a subquery and with day truncation."""
    subquery = "(SELECT user_id, action, create_time FROM user_activity WHERE date >= '2023-01-01')"
    query_generator = TransformTimeQuery(
        table_name=subquery,
        original_time_col="create_time",
        time_col_format="timestamp",
        new_time_col="activity_day_ts",
        time_unit="day",
    )
    expected_query = f"(SELECT *,\n" f"       DATE_TRUNC('DAY', create_time) AS activity_day_ts\n" f"FROM ({subquery}))"
    assert query_generator.gen() == expected_query


def test_transform_time_query_plural_time_unit():
    """Test query generation with a plural time unit (e.g., 'seconds')."""
    query_generator = TransformTimeQuery(
        table_name="my_table",
        original_time_col="my_time",
        time_col_format="unix_ms",
        new_time_col="truncated_time",
        time_unit="seconds",
    )
    expected_query = "(SELECT *,\n" "       DATE_TRUNC('SECOND', FROM_UNIXTIME(my_time / 1000.0)) AS truncated_time\n" "FROM (my_table))"
    assert query_generator.gen() == expected_query

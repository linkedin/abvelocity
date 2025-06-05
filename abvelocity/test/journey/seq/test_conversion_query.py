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

from abvelocity.journey.seq.conversion_query import ConversionQuery


def test_basic_conversion_query():
    """
    Test case: Basic conversion query without optional parameters.
    """
    converter = ConversionQuery(
        table_name="user_events",
        array_col="event_tags",
        numerator_list=["product_page_view"],
        denominator_list=["landing_page_view"],
    )
    expected_query = (
        "SELECT "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['product_page_view'])) > 0 "
        "THEN 1 ELSE NULL END) AS numer_count, "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['landing_page_view'])) > 0 "
        "THEN 1 ELSE NULL END) AS denom_count, "
        "CAST(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['product_page_view'])) > 0 "
        "THEN 1 ELSE NULL END) AS DOUBLE) "
        "/ NULLIF(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['landing_page_view'])) > 0 THEN 1 ELSE NULL END), 0) AS conversion_rate "
        "FROM user_events"
    )
    assert converter.gen().strip() == expected_query.strip()


def test_distinct_conversion_query():
    """
    Test case: Conversion with a distinct count column (`count_distinct_col`).
    """
    converter = ConversionQuery(
        table_name="user_events",
        array_col="event_tags",
        numerator_list=["purchase_completed"],
        denominator_list=["add_to_cart"],
        count_distinct_col="user_id",
    )
    expected_query = (
        "SELECT "
        "COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['purchase_completed'])) > 0 THEN user_id ELSE NULL END) AS numer_count, "
        "COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['add_to_cart'])) > 0 THEN user_id ELSE NULL END) AS denom_count, "
        "CAST(COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['purchase_completed'])) > 0 THEN user_id ELSE NULL END) AS DOUBLE) "
        "/ NULLIF(COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['add_to_cart'])) > 0 THEN user_id ELSE NULL END), 0) AS conversion_rate "
        "FROM user_events"
    )
    assert converter.gen().strip() == expected_query.strip()


def test_conversion_with_conditions():
    """
    Test case: Conversion with multiple WHERE conditions.
    """
    converter = ConversionQuery(
        table_name="user_events",
        array_col="event_tags",
        numerator_list=["click"],
        denominator_list=["impression"],
        conditions=["event_date >= DATE '2023-01-01'", "device = 'mobile'"],
    )
    expected_query = (
        "SELECT "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['click'])) > 0 THEN 1 ELSE NULL END) AS numer_count, "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['impression'])) > 0 THEN 1 ELSE NULL END) AS denom_count, "
        "CAST(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['click'])) > 0 THEN 1 ELSE NULL END) AS DOUBLE) "
        "/ NULLIF(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['impression'])) > 0 THEN 1 ELSE NULL END), 0) AS conversion_rate "
        "FROM user_events WHERE event_date >= DATE '2023-01-01' AND device = 'mobile'"
    )
    assert converter.gen().strip() == expected_query.strip()


def test_conversion_with_group_by():
    """
    Test case: Conversion grouped by specified columns.
    """
    converter = ConversionQuery(
        table_name="user_events",
        array_col="event_tags",
        numerator_list=["conversion"],
        denominator_list=["visit"],
        group_by_cols=["country", "DATE(event_timestamp)"],
    )
    expected_query = (
        "SELECT country, DATE(event_timestamp), "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['conversion'])) > 0 "
        "THEN 1 ELSE NULL END) AS numer_count, "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['visit'])) > 0 "
        "THEN 1 ELSE NULL END) AS denom_count, "
        "CAST(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['conversion'])) > 0 "
        "THEN 1 ELSE NULL END) AS DOUBLE) "
        "/ NULLIF(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['visit'])) > 0 THEN 1 ELSE NULL END), 0) AS conversion_rate "
        "FROM user_events GROUP BY country, DATE(event_timestamp)"
    )
    assert converter.gen().strip() == expected_query.strip()


def test_conversion_all_params():
    """
    Test case: Conversion using all optional parameters (distinct count, conditions, group by).
    """
    converter = ConversionQuery(
        table_name="user_events",
        array_col="event_tags",
        numerator_list=["signup_complete"],
        denominator_list=["signup_start"],
        count_distinct_col="session_id",
        conditions=["platform = 'web'", "region = 'EU'"],
        group_by_cols=["channel"],
    )
    expected_query = (
        "SELECT channel, "
        "COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['signup_complete'])) > 0 THEN session_id ELSE NULL END) AS numer_count, "
        "COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['signup_start'])) > 0 THEN session_id ELSE NULL END) AS denom_count, "
        "CAST(COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['signup_complete'])) > 0 THEN session_id ELSE NULL END) AS DOUBLE) "
        "/ NULLIF(COUNT(DISTINCT CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['signup_start'])) > 0 THEN session_id ELSE NULL END), 0) AS conversion_rate "
        "FROM user_events WHERE platform = 'web' AND region = 'EU' GROUP BY channel"
    )
    assert converter.gen().strip() == expected_query.strip()


def test_empty_numerator_list_raises_error():
    """
    Test case: Ensure ValueError is raised when numerator_list is empty during initialization.
    """
    with pytest.raises(ValueError, match="numerator_list must be a non-empty list."):
        ConversionQuery(
            table_name="user_events",
            array_col="event_tags",
            numerator_list=[],
            denominator_list=["any"],
        ).gen()


def test_empty_denominator_list_raises_error():
    """
    Test case: Ensure ValueError is raised when denominator_list is empty during initialization.
    """
    with pytest.raises(ValueError, match="denominator_list must be a non-empty list."):
        ConversionQuery(
            table_name="user_events",
            array_col="event_tags",
            numerator_list=["any"],
            denominator_list=[],
        ).gen()


def test_invalid_count_distinct_col_type_raises_error():
    """
    Test case: Ensure TypeError is raised for invalid count_distinct_col type during initialization.
    """
    with pytest.raises(TypeError, match="count_distinct_col must be a string or None."):
        ConversionQuery(
            table_name="user_events",
            array_col="event_tags",
            numerator_list=["a"],
            denominator_list=["b"],
            count_distinct_col=123,  # Invalid type
        ).gen()


def test_invalid_conditions_type_raises_error():
    """
    Test case: Ensure TypeError is raised for invalid conditions type or content during initialization.
    """
    with pytest.raises(TypeError, match="conditions must be a list of strings or None."):
        ConversionQuery(
            table_name="user_events",
            array_col="event_tags",
            numerator_list=["a"],
            denominator_list=["b"],
            conditions="not_a_list",  # Invalid type
        ).gen()
    with pytest.raises(TypeError, match="conditions must be a list of strings or None."):
        ConversionQuery(
            table_name="user_events",
            array_col="event_tags",
            numerator_list=["a"],
            denominator_list=["b"],
            conditions=["valid", 123],  # Invalid content
        ).gen()


def test_invalid_group_by_cols_type_raises_error():
    """
    Test case: Ensure TypeError is raised for invalid group_by_cols type or content during initialization.
    """
    with pytest.raises(TypeError, match="group_by_cols must be a list of strings or None."):
        ConversionQuery(
            table_name="user_events",
            array_col="event_tags",
            numerator_list=["a"],
            denominator_list=["b"],
            group_by_cols="not_a_list",  # Invalid type
        ).gen()
    with pytest.raises(TypeError, match="group_by_cols must be a list of strings or None."):
        ConversionQuery(
            table_name="user_events",
            array_col="event_tags",
            numerator_list=["a"],
            denominator_list=["b"],
            group_by_cols=["valid", False],  # Invalid content
        ).gen()


def test_no_optional_params():
    """
    Test case: Query generation when no optional parameters are explicitly provided
        (relying on default_factory for empty lists).
    """
    converter = ConversionQuery(
        table_name="user_events",
        array_col="event_tags",
        numerator_list=["X"],
        denominator_list=["Y"],
    )
    expected_query = (
        "SELECT "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['X'])) > 0 "
        "THEN 1 ELSE NULL END) AS numer_count, "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['Y'])) > 0 "
        "THEN 1 ELSE NULL END) AS denom_count, "
        "CAST(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(event_tags, ARRAY['X'])) > 0 "
        "THEN 1 ELSE NULL END) AS DOUBLE) "
        "/ NULLIF(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['Y'])) > 0 THEN 1 ELSE NULL END), 0) AS conversion_rate "
        "FROM user_events"
    )
    assert converter.gen().strip() == expected_query.strip()


def test_nullif_in_query():
    """
    Test case: Verify that NULLIF is correctly included in the generated query
        to prevent division by zero.
    """
    converter = ConversionQuery(
        table_name="user_events",
        array_col="event_tags",
        numerator_list=["eventA"],
        denominator_list=["eventB"],
    )
    query = converter.gen()
    assert "NULLIF(" in query and ", 0)" in query


def test_different_table_and_array_names():
    """
    Test case: Verify query generation with different table and array column names.
    """
    converter = ConversionQuery(
        table_name="product_logs",
        array_col="actions",
        numerator_list=["view"],
        denominator_list=["search"],
    )
    expected_query = (
        "SELECT "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(actions, ARRAY['view'])) > 0 "
        "THEN 1 ELSE NULL END) AS numer_count, "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(actions, ARRAY['search'])) > 0 "
        "THEN 1 ELSE NULL END) AS denom_count, "
        "CAST(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT(actions, ARRAY['view'])) > 0 "
        "THEN 1 ELSE NULL END) AS DOUBLE) "
        "/ NULLIF(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "actions, ARRAY['search'])) > 0 THEN 1 ELSE NULL END), 0) AS conversion_rate "
        "FROM product_logs"
    )
    assert converter.gen().strip() == expected_query.strip()


def test_conversion_with_multiple_list_items():
    """
    Test case: Conversion with multiple values in both numerator and denominator lists.
    """
    converter = ConversionQuery(
        table_name="user_events",
        array_col="event_tags",
        numerator_list=["product_page_view", "detail_view"],
        denominator_list=["landing_page_view", "homepage_visit"],
    )
    expected_query = (
        "SELECT "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['product_page_view', 'detail_view'])) > 0 THEN 1 ELSE NULL END) AS numer_count, "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['landing_page_view', 'homepage_visit'])) > 0 THEN 1 ELSE NULL END) AS denom_count, "
        "CAST(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['product_page_view', 'detail_view'])) > 0 THEN 1 ELSE NULL END) "
        "AS DOUBLE) "
        "/ NULLIF(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['landing_page_view', 'homepage_visit'])) > 0 THEN 1 ELSE NULL END), 0) "
        "AS conversion_rate "
        "FROM user_events"
    )
    assert converter.gen().strip() == expected_query.strip()


def test_denominator_includes_numerator_plus_extra():
    """
    Test case: Denominator list contains all numerator elements plus additional ones.
    This verifies correct handling of overlapping (superset) tag sets.
    """
    converter = ConversionQuery(
        table_name="user_events",
        array_col="event_tags",
        numerator_list=["add_to_cart", "product_view"],
        denominator_list=["add_to_cart", "product_view", "page_load", "session_start"],
    )
    expected_query = (
        "SELECT "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['add_to_cart', 'product_view'])) > 0 THEN 1 ELSE NULL END) AS numer_count, "
        "COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['add_to_cart', 'product_view', 'page_load', 'session_start'])) > 0 "
        "THEN 1 ELSE NULL END) AS denom_count, "
        "CAST(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['add_to_cart', 'product_view'])) > 0 THEN 1 ELSE NULL END) AS DOUBLE) "
        "/ NULLIF(COUNT(CASE WHEN CARDINALITY(ARRAY_INTERSECT("
        "event_tags, ARRAY['add_to_cart', 'product_view', 'page_load', 'session_start'])) > 0 "
        "THEN 1 ELSE NULL END), 0) AS conversion_rate "
        "FROM user_events"
    )
    assert converter.gen().strip() == expected_query.strip()

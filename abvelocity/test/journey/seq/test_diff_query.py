# BSD 2-CLAUSE LICENSE
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
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

from abvelocity.journey.seq.diff_query import DiffQuery
from abvelocity.testing.assert_query_is_equal import assert_query_is_equal


def test_gen_diff_query_basic_no_normalization():
    query = DiffQuery(
        table_name="my_event_table",
        response_variable="count",
        slice_column="country",
        slice_values=["US", "Canada"],
        normalize_by_slice=False,  # Normalization is turned off
    )

    expected_query = """
    WITH slice1 AS (
        SELECT count AS response_var_1 FROM my_event_table WHERE country = 'US'
    ),
    slice2 AS (
        SELECT count AS response_var_2 FROM my_event_table WHERE country = 'Canada'
    )
    SELECT a.response_var_1 - b.response_var_2 AS response_diff
    FROM slice1 a
    JOIN slice2 b
    ON 1=1
    """.strip()

    assert_query_is_equal(query.gen(), expected_query)


def test_gen_diff_query_basic_with_normalization():
    query = DiffQuery(
        table_name="my_event_table",
        response_variable="count",
        slice_column="country",
        slice_values=["US", "Canada"],
        normalize_by_slice=True,  # Normalization by slice is enabled
    )

    expected_query = """
    WITH slice1 AS (
        SELECT (count / SUM(count) OVER (PARTITION BY country)) * 100 AS response_var_1
        FROM my_event_table WHERE country = 'US'
    ),
    slice2 AS (
        SELECT (count / SUM(count) OVER (PARTITION BY country)) * 100 AS response_var_2
        FROM my_event_table WHERE country = 'Canada'
    )
    SELECT a.response_var_1 - b.response_var_2 AS response_diff
    FROM slice1 a
    JOIN slice2 b
    ON 1=1
    """.strip()

    assert_query_is_equal(query.gen(), expected_query)


def test_gen_diff_query_with_groupby_and_normalization():
    query = DiffQuery(
        table_name="my_event_table",
        response_variable="percent",
        slice_column="region",
        slice_values=["East", "West"],
        groupby_cols=["event_type"],
        normalize_by_slice=True,  # Normalization by slice is enabled
    )

    expected_query = """
    WITH slice1 AS (
        SELECT event_type, (percent / SUM(percent) OVER (PARTITION BY region)) * 100 AS response_var_1
        FROM my_event_table WHERE region = 'East'
    ),
    slice2 AS (
        SELECT event_type, (percent / SUM(percent) OVER (PARTITION BY region)) * 100 AS response_var_2
        FROM my_event_table WHERE region = 'West'
    )
    SELECT a.event_type, a.response_var_1 - b.response_var_2 AS response_diff
    FROM slice1 a
    JOIN slice2 b
    ON a.event_type = b.event_type
    """.strip()

    assert_query_is_equal(query.gen(), expected_query)


def test_gen_diff_query_with_conditions_and_normalization():
    query = DiffQuery(
        table_name="sales_data",
        response_variable="revenue",
        slice_column="quarter",
        slice_values=["Q1", "Q2"],
        groupby_cols=["region", "product_type"],
        conditions=["year = 2023", "status = 'confirmed'"],
        normalize_by_slice=True,  # Normalization by slice is enabled
    )

    expected_query = """
    WITH slice1 AS (
        SELECT region, product_type, (revenue / SUM(revenue) OVER (PARTITION BY quarter)) * 100 AS response_var_1
        FROM sales_data
        WHERE quarter = 'Q1' AND year = 2023 AND status = 'confirmed'
    ),
    slice2 AS (
        SELECT region, product_type, (revenue / SUM(revenue) OVER (PARTITION BY quarter)) * 100 AS response_var_2
        FROM sales_data
        WHERE quarter = 'Q2' AND year = 2023 AND status = 'confirmed'
    )
    SELECT a.region, a.product_type, a.response_var_1 - b.response_var_2 AS response_diff
    FROM slice1 a
    JOIN slice2 b
    ON a.region = b.region AND a.product_type = b.product_type
    """.strip()

    assert_query_is_equal(query.gen(), expected_query)


def test_gen_diff_query_no_normalization_with_conditions():
    query = DiffQuery(
        table_name="user_activity",
        response_variable="login_count",
        slice_column="device_type",
        slice_values=["mobile", "desktop"],
        conditions=["active = 1", "country IN ('US', 'UK')"],
        normalize_by_slice=False,  # Normalization is turned off
    )

    expected_query = """
    WITH slice1 AS (
        SELECT login_count AS response_var_1 FROM user_activity
        WHERE device_type = 'mobile' AND active = 1 AND country IN ('US', 'UK')
    ),
    slice2 AS (
        SELECT login_count AS response_var_2 FROM user_activity
        WHERE device_type = 'desktop' AND active = 1 AND country IN ('US', 'UK')
    )
    SELECT a.response_var_1 - b.response_var_2 AS response_diff
    FROM slice1 a
    JOIN slice2 b
    ON 1=1
    """.strip()

    assert_query_is_equal(query.gen(), expected_query)


def test_table_name_is_query():
    # Create a `DiffQuery` instance where `table_name` is a subquery
    query = DiffQuery(
        table_name="(SELECT * FROM user_activity WHERE active = 1)",  # subquery as table_name
        response_variable="login_count",
        slice_column="device_type",
        slice_values=["mobile", "desktop"],
        groupby_cols=["country"],
        normalize_by_slice=True,
    )

    # Expected query result
    expected_query = """
        WITH slice1 AS (
            SELECT country, (login_count / SUM(login_count) OVER (PARTITION BY device_type)) * 100 AS response_var_1
            FROM (SELECT * FROM user_activity WHERE active = 1)
            WHERE device_type = 'mobile'
        ),
        slice2 AS (
            SELECT country, (login_count / SUM(login_count) OVER (PARTITION BY device_type)) * 100 AS response_var_2
            FROM (SELECT * FROM user_activity WHERE active = 1)
            WHERE device_type = 'desktop'
        )
        SELECT
            a.country,
            a.response_var_1 - b.response_var_2 AS response_diff
        FROM slice1 a
        JOIN slice2 b
        ON a.country = b.country
    """.strip()

    # Generate the query using the DiffQuery class
    generated_query = query.gen()

    assert_query_is_equal(generated_query, expected_query)

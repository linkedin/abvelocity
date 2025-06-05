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

import re

import pytest

from abvelocity.param.join_query import JoinQuery


def test_inner_join_select_all():
    """Test a basic INNER JOIN with SELECT * (default behavior)."""
    join = JoinQuery(
        left_table="orders",
        right_table="customers",
        join_type="INNER",
        on=[("customer_id", "customer_id")],
    )
    expected_query = "SELECT o.*, c.* FROM orders AS o INNER JOIN customers AS c ON o.customer_id = c.customer_id;"
    assert join.gen(left_alias="o", right_alias="c") == expected_query


def test_left_join_specific_columns():
    """Test a LEFT JOIN with specific column selections and aliases."""
    join = JoinQuery(
        left_table="products",
        right_table="categories",
        join_type="LEFT",
        on=[("category_id", "category_id")],
        select_left_columns=["product_name", "product_price AS price"],
        select_right_columns=["category_name AS cat_name", "description"],
    )
    expected_query = (
        "SELECT p.product_name, p.product_price AS price, c.category_name AS cat_name, c.description"
        + " FROM products AS p LEFT JOIN categories AS c ON p.category_id = c.category_id;"
    )
    assert join.gen(left_alias="p", right_alias="c") == expected_query


def test_right_join_partial_select():
    """Test a RIGHT JOIN selecting only from the right table."""
    join = JoinQuery(
        left_table="employees",
        right_table="departments",
        join_type="RIGHT",
        on=[("dept_id", "id")],
        select_left_columns=[],
        select_right_columns=["id", "department_name AS dept_name"],
    )
    expected_query = (
        "SELECT d.id, d.department_name AS dept_name FROM"
        + " employees AS e RIGHT JOIN departments AS d ON e.dept_id = d.id;"
    )

    assert join.gen(left_alias="e", right_alias="d") == expected_query


def test_cross_join_specific_columns():
    """Test a CROSS JOIN with specific column selection (no ON clause)."""
    join = JoinQuery(
        left_table="table_a",
        right_table="table_b",
        join_type="CROSS",
        # 'on' should be None or empty for CROSS, explicit None makes it clear
        on=None,
        select_left_columns=["id AS a_id", "name"],
        select_right_columns=["value AS b_value"],
    )
    expected_query = (
        "SELECT a.id AS a_id, a.name, b.value AS b_value FROM table_a AS a CROSS JOIN table_b AS b;"
    )
    assert join.gen(left_alias="a", right_alias="b") == expected_query


def test_create_table_as_query():
    """Test generation of a CREATE TABLE AS query."""
    join = JoinQuery(
        left_table="orders",
        right_table="customers",
        join_type="INNER",
        on=[("customer_id", "customer_id")],
        output_table_name="order_customer_summary",  # No quotes here, it's a raw identifier
    )

    # Expected query now uses the raw table name without added quotes
    expected_query = (
        "DROP TABLE IF EXISTS order_customer_summary;\nCREATE TABLE order_customer_summary AS SELECT o.*, c.* FROM"
        + " orders AS o INNER JOIN customers AS c ON o.customer_id = c.customer_id;"
    )

    assert join.gen(left_alias="o", right_alias="c") == expected_query


def test_create_table_with_quoted_name_and_spaces():
    """Test CREATE TABLE AS when the output table name contains spaces, which should be quoted."""
    join = JoinQuery(
        left_table="data",
        right_table="lookup",
        join_type="INNER",
        on=[("lk_id", "id")],
        output_table_name='"my complex table name"',  # User provides the quotes
    )
    # Expected query now uses the raw table name as provided by the user
    expected_query = (
        'DROP TABLE IF EXISTS "my complex table name";\nCREATE TABLE "my complex table name" AS '
        + "SELECT t1.*, t2.* FROM data AS t1 INNER JOIN lookup AS t2 ON t1.lk_id = t2.id;"
    )
    assert join.gen() == expected_query


def test_create_table_with_embedded_quotes_escaped():
    """Test CREATE TABLE AS when the output table name has embedded quotes, which should be escaped."""
    join = JoinQuery(
        left_table="data",
        right_table="lookup",
        join_type="INNER",
        on=[("lk_id", "id")],
        output_table_name='"table_with_""quotes""_inside"',  # User provides the escaped quotes
    )

    # Expected query now uses the raw table name as provided by the user
    expected_query = (
        'DROP TABLE IF EXISTS "table_with_""quotes""_inside";\nCREATE TABLE "table_with_""quotes""_inside" AS'
        + " SELECT t1.*, t2.* FROM data AS t1 INNER JOIN lookup AS t2 ON t1.lk_id = t2.id;"
    )

    assert join.gen() == expected_query


def test_complex_select_expressions():
    """Test selecting columns with functions and other complex expressions."""
    join = JoinQuery(
        left_table="sales",
        right_table="regions",
        join_type="INNER",
        on=[("region_id", "id")],
        select_left_columns=[
            "SUM(amount) AS total_sales",
            "EXTRACT(YEAR FROM sale_date) AS sale_year",
        ],
        select_right_columns=["region_name"],
    )

    expected_query = (
        "SELECT SUM(amount) AS total_sales, EXTRACT(YEAR FROM sale_date) AS sale_year, r.region_name FROM"
        + " sales AS s INNER JOIN regions AS r ON s.region_id = r.id;"
    )

    assert join.gen(left_alias="s", right_alias="r") == expected_query


def test_select_star_with_alias():
    """Test selecting * from a specific alias within select_columns."""
    join = JoinQuery(
        left_table="users",
        right_table="user_details",
        join_type="LEFT",
        on=[("id", "user_id")],
        select_left_columns=["u.*"],
        select_right_columns=["details"],
    )
    expected_query = (
        "SELECT u.*, ud.details FROM users AS u LEFT JOIN user_details AS ud ON u.id = ud.user_id;"
    )
    assert join.gen(left_alias="u", right_alias="ud") == expected_query


def test_select_function_no_alias():
    """Test selecting a function result without an explicit AS alias."""
    join = JoinQuery(
        left_table="logs",
        right_table="errors",
        join_type="INNER",
        on=[("log_id", "error_log_id")],
        select_left_columns=["COUNT(*)"],
        select_right_columns=["error_message"],
    )
    expected_query = "SELECT COUNT(*), e.error_message FROM logs AS l INNER JOIN errors AS e ON l.log_id = e.error_log_id;"
    assert join.gen(left_alias="l", right_alias="e") == expected_query


def test_default_join_type_is_left():
    """Test that join_type defaults to 'LEFT' when not specified."""
    join = JoinQuery(left_table="table_x", right_table="table_y", on=[("id", "id")])
    # The expected query should use LEFT JOIN as the default
    expected_query = (
        "SELECT t1.*, t2.* FROM table_x AS t1 LEFT JOIN table_y AS t2 ON t1.id = t2.id;"
    )
    assert join.gen() == expected_query
    assert join.join_type == "LEFT"  # Also check the attribute directly


def test_empty_output_table_name_raises_value_error():
    """Test that creating a JoinQuery with an empty (whitespace-only) output_table_name raises ValueError."""
    with pytest.raises(
        ValueError, match="output_table_name cannot be an empty string if provided."
    ):
        JoinQuery(
            left_table="orders",
            right_table="customers",
            join_type="INNER",
            on=[("id", "id")],
            output_table_name="   ",
        )


def test_non_cross_join_without_on_raises_value_error_at_gen():
    """Test that a non-CROSS join without an 'on' condition raises ValueError when gen() is called."""
    # It's now allowed to initialize without 'on', but gen() should fail
    join = JoinQuery(left_table="t1", right_table="t2", join_type="INNER", on=None)
    # Use re.escape() to handle special characters in the regex match
    expected_match = re.escape(
        "Join 'on' condition must be specified and non-empty for non-CROSS joins when gen() is called."
    )
    with pytest.raises(ValueError, match=expected_match):
        join.gen()


def test_cross_join_with_on_raises_value_error_at_init():
    """Test that a CROSS join with an 'on' condition raises ValueError during initialization."""
    with pytest.raises(ValueError, match="CROSS join should not have an 'on' condition."):
        JoinQuery(left_table="t1", right_table="t2", join_type="CROSS", on=[("id", "id")])


def test_invalid_on_condition_type_raises_type_error_at_init():
    """Test that 'on' conditions with incorrect types raise TypeError during initialization."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "All 'on' conditions must be tuples of (left_column_name, right_column_name)."
        ),
    ):
        JoinQuery(left_table="t1", right_table="t2", join_type="INNER", on=[("id", 123)])


def test_invalid_on_condition_tuple_length_raises_type_error_at_init():
    """Test that 'on' conditions with incorrect tuple length raise TypeError."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "All 'on' conditions must be tuples of (left_column_name, right_column_name)."
        ),
    ):
        JoinQuery(left_table="t1", right_table="t2", join_type="INNER", on=[("id", "id", "extra")])


def test_empty_string_in_select_columns_raises_type_error():
    """Test that an empty string in select_columns raises TypeError."""
    with pytest.raises(TypeError, match="All select columns must be non-empty strings."):
        JoinQuery(
            left_table="t1",
            right_table="t2",
            join_type="INNER",
            on=[("x", "y")],
            select_left_columns=["col1", ""],
        )


def test_non_string_in_select_columns_raises_type_error():
    """Test that a non-string item in select_columns raises TypeError."""
    with pytest.raises(TypeError, match="All select columns must be non-empty strings."):
        JoinQuery(
            left_table="t1",
            right_table="t2",
            join_type="INNER",
            on=[("x", "y")],
            select_right_columns=["col2", 123],
        )


def test_left_table_not_set_raises_value_error_at_gen():
    """Test that calling gen() without left_table set raises ValueError."""
    join = JoinQuery(right_table="customers", join_type="INNER", on=[("id", "id")])
    with pytest.raises(ValueError, match="left_table must be set before gen is called."):
        join.gen()


def test_right_table_not_set_raises_value_error_at_gen():
    """Test that calling gen() without right_table set raises ValueError."""
    join = JoinQuery(left_table="orders", join_type="INNER", on=[("id", "id")])
    with pytest.raises(ValueError, match="right_table must be set before gen is called."):
        join.gen()


def test_str_representation_simple_join():
    """Test the string representation for a simple join."""
    join = JoinQuery(
        left_table="users", right_table="posts", join_type="INNER", on=[("user_id", "author_id")]
    )
    assert str(join) == "INNER JOIN posts to users on ('user_id', 'author_id')"


def test_str_representation_common_column_name():
    """Test the string representation for a common ON column name."""
    join = JoinQuery(
        left_table="products", right_table="categories", join_type="LEFT", on=[("id", "id")]
    )
    assert str(join) == "LEFT JOIN categories to products on 'id'"


def test_str_representation_cross_join():
    """Test the string representation for a CROSS JOIN."""
    join = JoinQuery(left_table="metrics", right_table="dates", join_type="CROSS")
    assert str(join) == "CROSS JOIN dates to metrics"


def test_str_representation_tables_not_set():
    """Test the string representation when left_table and right_table are None."""
    join = JoinQuery(join_type="INNER", on=[("id", "id")])
    # Updated expected string for 'on' clause to match the current __str__ implementation
    assert str(join) == "INNER JOIN [RIGHT TABLE NOT SET] to [LEFT TABLE NOT SET] on 'id'"


def test_str_representation_on_not_set_non_cross():
    """Test the string representation when 'on' is None for a non-CROSS join."""
    join = JoinQuery(left_table="table_a", right_table="table_b", join_type="LEFT", on=None)
    assert str(join) == "LEFT JOIN table_b to table_a ON conditions not set"


def test_right_conditions():
    """Test a join with a WHERE clause applied to the right table as a subquery."""
    join = JoinQuery(
        left_table="orders",
        right_table="products",
        join_type="INNER",
        on=[("product_id", "id")],
        right_conditions=["price > 100", "category = 'Electronics'"],
    )
    expected_query = (
        "SELECT t1.*, t2.* FROM orders AS t1 INNER JOIN "
        + "(SELECT * FROM products WHERE price > 100 AND category = 'Electronics') AS t2 ON t1.product_id = t2.id;"
    )

    assert join.gen() == expected_query

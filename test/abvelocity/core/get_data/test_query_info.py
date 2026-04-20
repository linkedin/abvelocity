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
from abvelocity.core.get_data.query_info import QueryInfo


def test_construct_with_all_fields():
    query_info = QueryInfo(
        table_name="sales",
        columns=["product_id", "product_name"],
        aggregations=["SUM(sales_amount) AS total_sales"],
        conditions=["department = 'Sales'", "year = 2023"],
    )
    expected_query = (
        "SELECT product_id, product_name, SUM(sales_amount) AS total_sales " "FROM sales WHERE department = 'Sales' AND year = 2023 " "GROUP BY 1, 2"
    )
    assert query_info.construct() == expected_query
    assert query_info.query == expected_query


def test_construct_with_no_columns():
    query_info = QueryInfo(
        table_name="sales",
        columns=None,
        aggregations=["SUM(sales_amount) AS total_sales"],
        conditions=["department = 'Sales'", "year = 2023"],
    )
    expected_query = "SELECT SUM(sales_amount) AS total_sales " "FROM sales WHERE department = 'Sales' AND year = 2023"
    assert query_info.construct() == expected_query
    assert query_info.query == expected_query


def test_construct_with_no_aggregations():
    query_info = QueryInfo(
        table_name="sales",
        columns=["product_id", "product_name"],
        aggregations=None,
        conditions=["department = 'Sales'", "year = 2023"],
    )
    expected_query = "SELECT product_id, product_name " "FROM sales WHERE department = 'Sales' AND year = 2023"
    assert query_info.construct() == expected_query
    assert query_info.query == expected_query


def test_construct_with_no_conditions():
    query_info = QueryInfo(
        table_name="sales",
        columns=["product_id", "product_name"],
        aggregations=["SUM(sales_amount) AS total_sales"],
        conditions=None,
    )
    expected_query = "SELECT product_id, product_name, SUM(sales_amount) AS total_sales " "FROM sales GROUP BY 1, 2"
    assert query_info.construct() == expected_query
    assert query_info.query == expected_query


def test_construct_with_only_table_name():
    query_info = QueryInfo(table_name="sales", columns=None, aggregations=None, conditions=None)
    expected_query = "SELECT * FROM sales"
    assert query_info.construct() == expected_query
    assert query_info.query == expected_query


def test_construct_missing_table_name():
    query_info = QueryInfo(
        table_name="",
        columns=["product_id", "product_name"],
        aggregations=["SUM(sales_amount) AS total_sales"],
        conditions=["department = 'Sales'", "year = 2023"],
    )
    with pytest.raises(ValueError, match="Table name must be provided."):
        query_info.construct()


def test_construct_invalid_conditions_type():
    query_info = QueryInfo(
        table_name="sales",
        columns=["product_id", "product_name"],
        aggregations=["SUM(sales_amount) AS total_sales"],
        conditions="department = 'Sales'",
    )
    with pytest.raises(TypeError, match="Conditions should be provided as a list of strings."):
        query_info.construct()

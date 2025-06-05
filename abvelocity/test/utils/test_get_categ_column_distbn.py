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

import pandas as pd
from plotly.graph_objects import Figure

from abvelocity.utils.get_categ_column_distbn import (
    get_categ_column_distbn,
    num_distinct_values_per_entity,
)


def test_get_categ_column_distbn():
    """Tests the get_categ_column_distbn function."""

    class MockCursor:
        def get_df(self, query):
            if "COUNT(DISTINCT" in query:
                data = {
                    "country": ["USA", "Canada", "USA", "Mexico", "Canada", "UK"],
                    "count": [2, 2, 1, 1, 1, 1],
                }
            else:
                data = {
                    "country": [
                        "USA",
                        "Canada",
                        "USA",
                        "Mexico",
                        "Canada",
                        "UK",
                        "France",
                        "Germany",
                        "Italy",
                        "Spain",
                        "Japan",
                        "China",
                    ],
                    "count": [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                }
            return type("obj", (object,), {"df": pd.DataFrame(data)})

    def mock_print_to_html(message, color=None, font_size=None, file_name=None):
        pass

    cursor = MockCursor()

    result = get_categ_column_distbn(
        table_name="customers",
        col="country",
        cursor=cursor,
        print_to_html=mock_print_to_html,
        max_num=3,
        conditions=["age > 18"],
        file_name="customer_countries.html",
    )

    assert result["num_distict_labels"] == 10
    assert len(result["plot_df"]) == 4

    # Test with count_distinct_col
    result_distinct = get_categ_column_distbn(
        table_name="customers",
        col="country",
        cursor=cursor,
        print_to_html=mock_print_to_html,
        max_num=3,
        count_distinct_col="user_id",
        conditions=["age > 18"],
        file_name="customer_countries.html",
    )

    assert result_distinct["num_distict_labels"] == 4
    assert len(result_distinct["plot_df"]) == 4

    # Test with an empty dataframe
    class EmptyMockCursor:
        def get_df(self, query):
            return type("obj", (object,), {"df": pd.DataFrame()})

    empty_cursor = EmptyMockCursor()

    empty_result = get_categ_column_distbn(
        table_name="empty_table",
        col="empty_col",
        cursor=empty_cursor,
        print_to_html=mock_print_to_html,
        max_num=5,
    )
    assert empty_result is None

    # Test with cursor is None
    none_result = get_categ_column_distbn(
        table_name="none_table",
        col="none_col",
        cursor=None,
        print_to_html=mock_print_to_html,
        max_num=5,
    )
    assert none_result is None


def test_num_distinct_values_per_entity():
    """Tests the num_distinct_values_per_entity function."""

    class MockCursor:
        def get_df(self, query):
            data = {"entity_id": [1, 1, 2, 2, 3], "distinct_count": [3, 3, 2, 2, 1]}
            return type("obj", (object,), {"df": pd.DataFrame(data)})

    def mock_print_to_html(message, color=None, font_size=None, file_name=None):
        pass

    cursor = MockCursor()

    fig = num_distinct_values_per_entity(
        table_name="entity_data",
        entity_cols=["entity_id"],
        count_distinct_col="value",
        cursor=cursor,
        print_to_html=mock_print_to_html,
        conditions=["some_condition = True"],
        x_range=[0, 5],
        file_name="distbn.html",
    )["fig"]

    assert isinstance(fig, Figure)  # Check if the return value is a plotly Figure.

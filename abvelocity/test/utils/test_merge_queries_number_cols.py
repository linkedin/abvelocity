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

from abvelocity.utils.merge_queries_number_cols import merge_queries_number_cols


def print_debug(actual, expected):
    print("\n--- ACTUAL ---")
    print(actual)
    print("\n--- EXPECTED ---")
    print(expected)
    print("\n--- END ---\n")


def test_single_query_no_tuple():
    queries = ["alpha"]
    on_cols = ["user_id"]
    common_cols = ["metric"]
    how = "inner"

    query = merge_queries_number_cols(
        queries=queries,
        on_cols=on_cols,
        common_cols=common_cols,
        how=how,
        add_tuple=False,
        drop_numbered_cols=False,
    )

    expected_sql = (
        "WITH\n"
        "  cte_0 AS (SELECT src_0.user_id, src_0.metric AS metric_1 FROM (alpha) AS src_0),\n"
        "  _initial_merged_data AS (\n"
        "SELECT cte_0.user_id, cte_0.metric_1 AS metric_1 FROM cte_0\n"
        ")\n"
        "SELECT _merged_output.user_id, _merged_output.metric_1 AS metric_1 FROM _initial_merged_data AS _merged_output"
    )

    if query.strip() != expected_sql.strip():
        print_debug(query.strip(), expected_sql.strip())
    assert query.strip() == expected_sql.strip()


def test_merge_two_tables_with_tuple_and_drop_numbered():
    queries = ["alpha", "bravo"]
    on_cols = ["user_id"]
    common_cols = ["metric"]
    how = "inner"

    query = merge_queries_number_cols(
        queries=queries,
        on_cols=on_cols,
        common_cols=common_cols,
        how=how,
        add_tuple=True,
        drop_numbered_cols=True,
    )

    expected_sql = (
        "WITH\n"
        "  cte_0 AS (SELECT src_0.user_id, src_0.metric AS metric_1 FROM (alpha) AS src_0),\n"
        "  cte_1 AS (SELECT src_1.user_id, src_1.metric AS metric_2 FROM (bravo) AS src_1),\n"
        "  _initial_merged_data AS (\n"
        "SELECT cte_0.user_id, cte_0.metric_1 AS metric_1, cte_1.metric_2 AS metric_2"
        " FROM (cte_0 INNER JOIN cte_1 ON cte_0.user_id = cte_1.user_id)\n"
        ")\n"
        "SELECT _merged_output.user_id, ROW(_merged_output.metric_1, _merged_output.metric_2) AS metric"
        " FROM _initial_merged_data AS _merged_output"
    )

    if query.strip() != expected_sql.strip():
        print_debug(query.strip(), expected_sql.strip())
    assert query.strip() == expected_sql.strip()


def test_merge_two_tables_with_tuple_and_keep_numbered_cols():
    queries = ["alpha", "bravo"]
    on_cols = ["user_id"]
    common_cols = ["metric"]
    how = "inner"
    nan_replacements = [0]

    query = merge_queries_number_cols(
        queries=queries,
        on_cols=on_cols,
        common_cols=common_cols,
        how=how,
        add_tuple=True,
        drop_numbered_cols=False,
        nan_replacements=nan_replacements,
    )

    expected_sql = (
        "WITH\n"
        "  cte_0 AS (SELECT src_0.user_id, COALESCE(src_0.metric, 0) AS metric_1 FROM (alpha) AS src_0),\n"
        "  cte_1 AS (SELECT src_1.user_id, COALESCE(src_1.metric, 0) AS metric_2 FROM (bravo) AS src_1),\n"
        "  _initial_merged_data AS (\n"
        "SELECT cte_0.user_id, COALESCE(cte_0.metric_1, 0) AS metric_1, COALESCE(cte_1.metric_2, 0) AS metric_2"
        " FROM (cte_0 INNER JOIN cte_1 ON cte_0.user_id = cte_1.user_id)\n"
        ")\n"
        "SELECT _merged_output.user_id, ROW(_merged_output.metric_1, _merged_output.metric_2) AS metric,"
        " COALESCE(_merged_output.metric_1, 0) AS metric_1, COALESCE(_merged_output.metric_2, 0) AS metric_2"
        " FROM _initial_merged_data AS _merged_output"
    )

    if query.strip() != expected_sql.strip():
        print_debug(query.strip(), expected_sql.strip())
    assert query.strip() == expected_sql.strip()


def test_merge_with_nan_replacements():
    queries = ["alpha", "bravo"]
    on_cols = ["user_id"]
    common_cols = ["metric"]
    nan_replacements = [0]
    how = "left"

    query = merge_queries_number_cols(
        queries=queries,
        on_cols=on_cols,
        common_cols=common_cols,
        how=how,
        nan_replacements=nan_replacements,
    )

    expected_sql = (
        "WITH\n"
        "  cte_0 AS (SELECT src_0.user_id, COALESCE(src_0.metric, 0) AS metric_1 FROM (alpha) AS src_0),\n"
        "  cte_1 AS (SELECT src_1.user_id, COALESCE(src_1.metric, 0) AS metric_2 FROM (bravo) AS src_1),\n"
        "  _initial_merged_data AS (\n"
        "SELECT cte_0.user_id, COALESCE(cte_0.metric_1, 0) AS metric_1,"
        " COALESCE(cte_1.metric_2, 0) AS metric_2 FROM (cte_0 LEFT OUTER JOIN cte_1 ON cte_0.user_id = cte_1.user_id)\n"
        ")\n"
        "SELECT _merged_output.user_id, COALESCE(_merged_output.metric_1, 0) AS metric_1,"
        " COALESCE(_merged_output.metric_2, 0) AS metric_2 FROM _initial_merged_data AS _merged_output"
    )

    if query.strip() != expected_sql.strip():
        print_debug(query.strip(), expected_sql.strip())
    assert query.strip() == expected_sql.strip()


def test_merge_three_queries_with_outer_and_tuple():
    queries = ["alpha", "bravo", "charlie"]
    on_cols = ["user_id"]
    common_cols = ["metric"]
    how = "outer"

    query = merge_queries_number_cols(
        queries=queries,
        on_cols=on_cols,
        common_cols=common_cols,
        how=how,
        add_tuple=True,
        drop_numbered_cols=True,
    )

    expected_sql = (
        "WITH\n"
        "  cte_0 AS (SELECT src_0.user_id, src_0.metric AS metric_1 FROM (alpha) AS src_0),\n"
        "  cte_1 AS (SELECT src_1.user_id, src_1.metric AS metric_2 FROM (bravo) AS src_1),\n"
        "  cte_2 AS (SELECT src_2.user_id, src_2.metric AS metric_3 FROM (charlie) AS src_2),\n"
        "  _initial_merged_data AS (\n"
        "SELECT COALESCE(cte_0.user_id, cte_1.user_id, cte_2.user_id) AS user_id,"
        " cte_0.metric_1 AS metric_1, cte_1.metric_2 AS metric_2, cte_2.metric_3 AS metric_3"
        " FROM ((cte_0 FULL OUTER JOIN cte_1 ON cte_0.user_id = cte_1.user_id) FULL OUTER JOIN cte_2 ON cte_1.user_id = cte_2.user_id)\n"
        ")\n"
        "SELECT user_id, ROW(_merged_output.metric_1, _merged_output.metric_2, _merged_output.metric_3)"
        " AS metric FROM _initial_merged_data AS _merged_output"
    )

    if query.strip() != expected_sql.strip():
        print_debug(query.strip(), expected_sql.strip())
    assert query.strip() == expected_sql.strip()


def test_merge_with_no_nan_replacements_on_some_cols():
    queries = ["alpha", "bravo"]
    on_cols = ["id"]
    common_cols = ["value_1", "value_2"]
    nan_replacements = [0, -1]
    how = "inner"

    query = merge_queries_number_cols(
        queries=queries,
        on_cols=on_cols,
        common_cols=common_cols,
        how=how,
        nan_replacements=nan_replacements,
    )

    expected_sql = (
        "WITH\n"
        "  cte_0 AS (SELECT src_0.id, COALESCE(src_0.value_1, 0) AS value_1_1, COALESCE(src_0.value_2, -1) AS value_2_1 FROM (alpha) AS src_0),\n"
        "  cte_1 AS (SELECT src_1.id, COALESCE(src_1.value_1, 0) AS value_1_2, COALESCE(src_1.value_2, -1) AS value_2_2 FROM (bravo) AS src_1),\n"
        "  _initial_merged_data AS (\n"
        "SELECT cte_0.id, COALESCE(cte_0.value_1_1, 0) AS value_1_1, COALESCE(cte_1.value_1_2, 0)"
        " AS value_1_2, COALESCE(cte_0.value_2_1, -1) AS value_2_1, COALESCE(cte_1.value_2_2, -1) AS value_2_2"
        " FROM (cte_0 INNER JOIN cte_1 ON cte_0.id = cte_1.id)\n"
        ")\n"
        "SELECT _merged_output.id, COALESCE(_merged_output.value_1_1, 0) AS value_1_1,"
        " COALESCE(_merged_output.value_1_2, 0) AS value_1_2, COALESCE(_merged_output.value_2_1, -1)"
        " AS value_2_1, COALESCE(_merged_output.value_2_2, -1) AS value_2_2 FROM _initial_merged_data AS _merged_output"
    )

    if query.strip() != expected_sql.strip():
        print_debug(query.strip(), expected_sql.strip())
    assert query.strip() == expected_sql.strip()


def test_merge_with_string_nan_replacements():
    queries = ["alpha", "bravo"]
    on_cols = ["id"]
    common_cols = ["score"]
    nan_replacements = ["cat"]
    how = "inner"

    query = merge_queries_number_cols(
        queries=queries,
        on_cols=on_cols,
        common_cols=common_cols,
        how=how,
        nan_replacements=nan_replacements,
    )

    expected_sql = (
        "WITH\n"
        "  cte_0 AS (SELECT src_0.id, COALESCE(src_0.score, 'cat') AS score_1 FROM (alpha) AS src_0),\n"
        "  cte_1 AS (SELECT src_1.id, COALESCE(src_1.score, 'cat') AS score_2 FROM (bravo) AS src_1),\n"
        "  _initial_merged_data AS (\n"
        "SELECT cte_0.id, COALESCE(cte_0.score_1, 'cat') AS score_1,"
        " COALESCE(cte_1.score_2, 'cat') AS score_2 FROM (cte_0 INNER JOIN cte_1 ON cte_0.id = cte_1.id)\n"
        ")\n"
        "SELECT _merged_output.id, COALESCE(_merged_output.score_1, 'cat') AS"
        " score_1, COALESCE(_merged_output.score_2, 'cat') AS score_2 FROM _initial_merged_data AS _merged_output"
    )

    if query.strip() != expected_sql.strip():
        print_debug(query.strip(), expected_sql.strip())
    assert query.strip() == expected_sql.strip()

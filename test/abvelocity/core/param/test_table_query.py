# BSD 2-CLAUSE LICENSE

from abvelocity.core.journey.param.table_query import TableQuery
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal


def test_gen_rebuild_query_default():
    tq = TableQuery(
        table_name="schema.my_table",
        main_query="SELECT a, b FROM source_table",
    )
    expected = """
    DROP TABLE IF EXISTS schema.my_table;
    CREATE TABLE schema.my_table AS
    SELECT a, b FROM source_table
    """.strip()
    assert_query_is_equal(tq.gen_rebuild_query(), expected)


def test_gen_rebuild_query_if_not_exists():
    tq = TableQuery(
        table_name="schema.my_table",
        main_query="SELECT a, b FROM source_table",
    )
    expected = """
    DROP TABLE IF EXISTS schema.my_table;
    CREATE TABLE IF NOT EXISTS schema.my_table AS
    SELECT a, b FROM source_table
    """.strip()
    assert_query_is_equal(tq.gen_rebuild_query(if_not_exists=True), expected)


def test_gen_create_query_default():
    tq = TableQuery(
        table_name="schema.new_table",
        main_query="SELECT x, y FROM staging",
    )
    expected = """
    CREATE TABLE IF NOT EXISTS schema.new_table AS
    SELECT x, y FROM staging
    """.strip()
    assert_query_is_equal(tq.gen_create_query(), expected)


def test_gen_create_query_no_if_not_exists():
    tq = TableQuery(
        table_name="schema.new_table",
        main_query="SELECT x, y FROM staging",
    )
    expected = """
    CREATE TABLE schema.new_table AS
    SELECT x, y FROM staging
    """.strip()
    assert_query_is_equal(tq.gen_create_query(if_not_exists=False), expected)


def test_gen_view_query_or_replace_default():
    tq = TableQuery(
        table_name="schema.vw_items",
        main_query="SELECT id, name FROM items",
    )
    expected = """
    CREATE OR REPLACE VIEW schema.vw_items AS
    SELECT id, name FROM items
    """.strip()
    assert_query_is_equal(tq.gen_view_query(), expected)


def test_gen_view_query_no_replace():
    tq = TableQuery(
        table_name="schema.vw_items",
        main_query="SELECT id, name FROM items",
    )
    expected = """
    CREATE VIEW schema.vw_items AS
    SELECT id, name FROM items
    """.strip()
    assert_query_is_equal(tq.gen_view_query(or_replace=False), expected)


def test_gen_overwrite_partition_query_without_create_static():
    # Static single-partition mode (partition_value supplied, partition col not in main_query)
    tq = TableQuery(
        table_name="schema.part_table",
        main_query="SELECT col1, col2 FROM raw_src",
    )
    result = tq.gen_overwrite_partition_query(partition_col="etl_date", partition_value="2024-09-29", create_table=False)
    expected = """
    INSERT INTO schema.part_table
    WITH _src AS (
        SELECT col1, col2 FROM raw_src
    )
    SELECT _src.*, '2024-09-29' AS etl_date
    FROM _src
    """.strip()
    assert_query_is_equal(result, expected)


def test_gen_overwrite_partition_query_with_create():
    # Static single-partition with create table path
    tq = TableQuery(
        table_name="schema.part_table",
        main_query="SELECT col1, col2 FROM raw_src",
    )
    result = tq.gen_overwrite_partition_query(partition_col="etl_date", partition_value="2024-09-29", create_table=True)
    expected = """
    CREATE TABLE IF NOT EXISTS schema.part_table
                            WITH (
                            partitioned_by = ARRAY['etl_date']
                            )
                            AS
                             WITH _src AS (
                                SELECT col1, col2 FROM raw_src
                            )
                            SELECT _src.*, '2024-09-29' AS etl_date
                            FROM _src
                            LIMIT 0;INSERT INTO  schema.part_table
                    WITH _src AS (
                        SELECT col1, col2 FROM raw_src
                    )
                    SELECT _src.*, '2024-09-29' AS etl_date
                    FROM _src
    """.strip()
    assert_query_is_equal(result, expected)


def test_gen_overwrite_partition_query_without_create_multiple_parts():
    # Dynamic mode unsupported explicitly: emulate by passing a dummy partition value to match implementation
    tq = TableQuery(
        table_name="schema.part_table",
        main_query="SELECT col1, col2 FROM raw_src",
    )
    result = tq.gen_overwrite_partition_query(partition_col="etl_date", partition_value="2024-09-29", create_table=False)
    expected = """
    INSERT INTO schema.part_table
    WITH _src AS (
        SELECT col1, col2 FROM raw_src
    )
    SELECT _src.*, '2024-09-29' AS etl_date
    FROM _src
    """.strip()
    assert_query_is_equal(result, expected)


def test_gen_overwrite_partition_query_escaping():
    tq = TableQuery(
        table_name="schema.part_table",
        main_query="SELECT col1 FROM raw_src",
    )
    result = tq.gen_overwrite_partition_query(partition_col="etl_date", partition_value="2025-09-25-00", create_table=False)
    expected = """
    INSERT INTO schema.part_table
    WITH _src AS (
        SELECT col1 FROM raw_src
    )
    SELECT _src.*, '2025-09-25-00' AS etl_date
    FROM _src
    """.strip()
    assert_query_is_equal(result, expected)

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

import pytest
from abvelocity.core.journey.param.table_materialization import MaterializationType, QueryEngine, TableMaterialization
from abvelocity.core.journey.param.table_query import TableQuery, gen_insert_partition_from_source
from abvelocity.core.testing.assert_query_is_equal import assert_query_is_equal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_table_query() -> TableQuery:
    return TableQuery(
        table_name="u_test_schema.journey_table",
        main_query="SELECT id, label FROM raw_events",
    )


# ---------------------------------------------------------------------------
# Tests: Trino (default engine) — INSERT INTO
# ---------------------------------------------------------------------------
def test_trino_overwrite_partition_insert_only(sample_table_query: TableQuery) -> None:
    """Trino incremental without CREATE should produce INSERT INTO (no OVERWRITE)."""
    sql = sample_table_query.gen_overwrite_partition_query(
        partition_col="partition_dt",
        partition_value="2025-06-01",
        create_table=False,
        engine=QueryEngine.TRINO,
    )

    expected = """
        INSERT INTO  u_test_schema.journey_table
            WITH _src AS (
                SELECT id, label FROM raw_events
            )
            SELECT _src.*, '2025-06-01' AS partition_dt
            FROM _src
    """
    assert_query_is_equal(sql, expected)


def test_trino_overwrite_partition_with_create(sample_table_query: TableQuery) -> None:
    """Trino incremental with CREATE should emit CREATE TABLE IF NOT EXISTS + INSERT INTO."""
    sql = sample_table_query.gen_overwrite_partition_query(
        partition_col="partition_dt",
        partition_value="2025-06-01",
        create_table=True,
        engine=QueryEngine.TRINO,
    )

    expected = """
        CREATE TABLE IF NOT EXISTS u_test_schema.journey_table
            WITH (
                partitioned_by = ARRAY['partition_dt']
            )
            AS
                WITH _src AS (
                    SELECT id, label FROM raw_events
                )
                SELECT _src.*, '2025-06-01' AS partition_dt
                FROM _src
                LIMIT 0;INSERT INTO  u_test_schema.journey_table
            WITH _src AS (
                SELECT id, label FROM raw_events
            )
            SELECT _src.*, '2025-06-01' AS partition_dt
            FROM _src
    """
    assert_query_is_equal(sql, expected)


# ---------------------------------------------------------------------------
# Tests: Spark — INSERT OVERWRITE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", [QueryEngine.SPARK, QueryEngine.CUSTOM_SPARK])
def test_spark_overwrite_partition_insert_only(sample_table_query: TableQuery, engine: QueryEngine) -> None:
    """Spark/CUSTOM_SPARK incremental without CREATE should produce INSERT OVERWRITE."""
    sql = sample_table_query.gen_overwrite_partition_query(
        partition_col="partition_dt",
        partition_value="2025-06-02",
        create_table=False,
        engine=engine,
    )

    expected = """
        INSERT OVERWRITE  u_test_schema.journey_table
            WITH _src AS (
                SELECT id, label FROM raw_events
            )
            SELECT _src.*, '2025-06-02' AS partition_dt
            FROM _src
    """
    assert_query_is_equal(sql, expected)


@pytest.mark.parametrize("engine", [QueryEngine.SPARK, QueryEngine.CUSTOM_SPARK])
def test_spark_overwrite_partition_with_create(sample_table_query: TableQuery, engine: QueryEngine) -> None:
    """Spark/CUSTOM_SPARK incremental with CREATE should emit CREATE TABLE + INSERT OVERWRITE."""
    sql = sample_table_query.gen_overwrite_partition_query(
        partition_col="partition_dt",
        partition_value="2025-06-03",
        create_table=True,
        engine=engine,
    )

    expected = """
        CREATE TABLE IF NOT EXISTS u_test_schema.journey_table
            WITH (
                partitioned_by = ARRAY['partition_dt']
            )
            AS
                WITH _src AS (
                    SELECT id, label FROM raw_events
                )
                SELECT _src.*, '2025-06-03' AS partition_dt
                FROM _src
                LIMIT 0;INSERT OVERWRITE  u_test_schema.journey_table
            WITH _src AS (
                SELECT id, label FROM raw_events
            )
            SELECT _src.*, '2025-06-03' AS partition_dt
            FROM _src
    """
    assert_query_is_equal(sql, expected)


# ---------------------------------------------------------------------------
# Tests: gen_insert_partition_from_source
# ---------------------------------------------------------------------------
def test_gen_insert_partition_from_source_spark() -> None:
    """gen_insert_partition_from_source should use INSERT OVERWRITE for Spark materializations."""
    mat = TableMaterialization(
        materialization_type=MaterializationType.INCREMENTAL,
        partition_col="partition_dt",
        partition_value="2025-06-10",
        engine=QueryEngine.SPARK,
    )

    sql = gen_insert_partition_from_source(
        source_table="tmp_view",
        target_table="u_test_schema.final_table",
        materialization=mat,
        create_table=False,
    )

    expected = """
        INSERT OVERWRITE  u_test_schema.final_table
            WITH _src AS (
                SELECT * FROM tmp_view
            )
            SELECT _src.*, '2025-06-10' AS partition_dt
            FROM _src
    """
    assert_query_is_equal(sql, expected)


def test_gen_insert_partition_from_source_trino() -> None:
    """gen_insert_partition_from_source should use INSERT INTO for Trino materializations."""
    mat = TableMaterialization(
        materialization_type=MaterializationType.INCREMENTAL,
        partition_col="partition_dt",
        partition_value="2025-06-10",
        engine=QueryEngine.TRINO,
    )

    sql = gen_insert_partition_from_source(
        source_table="tmp_view",
        target_table="u_test_schema.final_table",
        materialization=mat,
        create_table=False,
    )

    expected = """
        INSERT INTO  u_test_schema.final_table
            WITH _src AS (
                SELECT * FROM tmp_view
            )
            SELECT _src.*, '2025-06-10' AS partition_dt
            FROM _src
    """
    assert_query_is_equal(sql, expected)


def test_gen_insert_partition_from_source_spark_with_create() -> None:
    """gen_insert_partition_from_source with create_table=True for Spark."""
    mat = TableMaterialization(
        materialization_type=MaterializationType.INCREMENTAL,
        partition_col="partition_dt",
        partition_value="2025-06-10",
        engine=QueryEngine.SPARK,
    )

    sql = gen_insert_partition_from_source(
        source_table="tmp_view",
        target_table="u_test_schema.final_table",
        materialization=mat,
        create_table=True,
    )

    expected = """
        CREATE TABLE IF NOT EXISTS u_test_schema.final_table
            WITH (
                partitioned_by = ARRAY['partition_dt']
            )
            AS
                WITH _src AS (
                    SELECT * FROM tmp_view
                )
                SELECT _src.*, '2025-06-10' AS partition_dt
                FROM _src
                LIMIT 0;INSERT OVERWRITE  u_test_schema.final_table
            WITH _src AS (
                SELECT * FROM tmp_view
            )
            SELECT _src.*, '2025-06-10' AS partition_dt
            FROM _src
    """
    assert_query_is_equal(sql, expected)


def test_gen_insert_partition_from_source_rejects_non_incremental() -> None:
    """gen_insert_partition_from_source should raise for non-INCREMENTAL materializations."""
    mat = TableMaterialization(
        materialization_type=MaterializationType.OVERWRITE,
        partition_col="partition_dt",
        partition_value="2025-06-10",
        engine=QueryEngine.SPARK,
    )

    with pytest.raises(ValueError, match="INCREMENTAL"):
        gen_insert_partition_from_source(
            source_table="tmp_view",
            target_table="u_test_schema.final_table",
            materialization=mat,
        )


# ---------------------------------------------------------------------------
# Edge case: partition value with quotes
# ---------------------------------------------------------------------------
def test_spark_partition_value_with_quotes(sample_table_query: TableQuery) -> None:
    """Partition values containing single quotes should be properly escaped."""
    sql = sample_table_query.gen_overwrite_partition_query(
        partition_col="partition_dt",
        partition_value="it's a test",
        create_table=False,
        engine=QueryEngine.SPARK,
    )

    expected = """
        INSERT OVERWRITE  u_test_schema.journey_table
            WITH _src AS (
                SELECT id, label FROM raw_events
            )
            SELECT _src.*, 'it''s a test' AS partition_dt
            FROM _src
    """
    assert_query_is_equal(sql, expected)

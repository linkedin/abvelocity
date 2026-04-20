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

from dataclasses import dataclass

from abvelocity.core.journey.param.table_materialization import MaterializationType, QueryEngine
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class TableQuery(DataClassJSONMixin):
    """Container for a SQL body  and associated target table name.

    Attributes:
        table_name: Name of the table to (re)build.
        main_query: Core SELECT / WITH ... SELECT statement (no trailing semicolon required).
    """

    table_name: str
    main_query: str

    def gen_rebuild_query(self, if_not_exists: bool = False) -> str:
        """Return a full rebuild SQL with DROP TABLE IF EXISTS + CREATE TABLE (optionally IF NOT EXISTS) AS.
        The method always produces idempotent rebuild logic.
        """
        create_kw = "CREATE TABLE"
        if if_not_exists:
            create_kw += " IF NOT EXISTS"
        return f"DROP TABLE IF EXISTS {self.table_name};\n{create_kw} {self.table_name} AS\n{self.main_query.strip()}"  # no semicolon to allow chaining

    def gen_create_query(self, if_not_exists: bool = True) -> str:
        create_kw = "CREATE TABLE"
        if if_not_exists:
            create_kw += " IF NOT EXISTS"
        return f"{create_kw} {self.table_name} AS\n{self.main_query.strip()}"

    def gen_view_query(self, or_replace: bool = True) -> str:
        rep = "OR REPLACE " if or_replace else ""
        return f"CREATE {rep}VIEW {self.table_name} AS\n{self.main_query.strip()}"

    def gen_overwrite_partition_query(
        self,
        partition_col: str,
        partition_value: str,
        create_table: bool = True,
        engine: QueryEngine = QueryEngine.TRINO,
    ) -> str:
        """Prepare statements to (optionally) create and then append/overwrite dynamic partitions.

        When engine is Spark/CUSTOM_SPARK the INSERT uses OVERWRITE semantics (assumes
        ``spark.sql.sources.partitionOverwriteMode = dynamic``).
        """
        # Auto-quote the partition value
        sanitized = partition_value.replace("'", "''")
        partition_value_rendered = f"'{sanitized}'"

        body = self.main_query.strip()

        create_stmt = ""
        if create_table:
            create_kw = "CREATE TABLE IF NOT EXISTS"
            create_stmt = f"""{create_kw} {self.table_name}
                    WITH (
                    partitioned_by = ARRAY['{partition_col}']
                    )
                    AS
                        WITH _src AS (
                        {body}
                    )
                    SELECT _src.*, {partition_value_rendered} AS {partition_col}
                    FROM _src
                    LIMIT 0;"""

        insert_kw = "INSERT INTO"
        if engine in (QueryEngine.SPARK, QueryEngine.CUSTOM_SPARK):
            insert_kw = "INSERT OVERWRITE"

        insert_stmt = f"""{insert_kw}  {self.table_name}
                    WITH _src AS (
                        {body}
                    )
                    SELECT _src.*, {partition_value_rendered} AS {partition_col}
                    FROM _src"""

        return create_stmt + insert_stmt


def gen_insert_partition_from_source(
    source_table: str,
    target_table: str,
    materialization,  # expected: TableMaterialization with partition_col / partition_value
    create_table: bool = True,
) -> str:
    """Generate SQL to (optionally) create a partitioned target table inserting rows from ``source_table``.

    Partition metadata (column & value) are taken from the provided ``materialization`` object
    to avoid proliferating separate arguments.

    Args:
        source_table: Source (view/temp) table.
        target_table: Destination physical (partitioned) table.
        materialization: TableMaterialization (must be INCREMENTAL) providing partition_col & partition_value.
        create_table: Whether to emit guarded CREATE TABLE IF NOT EXISTS.
    """
    partition_col = getattr(materialization, "partition_col", None)
    partition_value = getattr(materialization, "partition_value", None)
    mtype = getattr(materialization, "materialization_type", None)
    if not partition_col or not partition_value:
        raise ValueError("materialization must define partition_col and partition_value")
    # Soft check for incremental type (only if enum present)
    if mtype != MaterializationType.INCREMENTAL:  # type: ignore
        raise ValueError("gen_insert_partition_from_source expects an INCREMENTAL materialization")

    engine = getattr(materialization, "engine", QueryEngine.TRINO)
    tq = TableQuery(table_name=target_table, main_query=f"SELECT * FROM {source_table}")
    return tq.gen_overwrite_partition_query(
        partition_col=partition_col,
        partition_value=partition_value,
        create_table=create_table,
        engine=engine,
    )

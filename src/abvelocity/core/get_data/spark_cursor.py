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
# Original author: 


import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pandas as pd

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import lit
except ModuleNotFoundError:
    lit = None  # type: ignore[assignment]

from abvelocity.core.get_data.cursor import DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY_SECONDS, Cursor, SqlQueryResult
from abvelocity.core.utils.sql_conversion import SqlConversion


@dataclass
class ConnArgs:
    """This is a dataclass to store Spark connection arguments."""

    spark_session: Optional[Any] = None
    """An existing SparkSession instance to use (e.g., from CloudNotebook). If None, a new session will be created."""
    app_name: Optional[str] = "abvelocity_spark"
    """The name of the Spark application (only used if creating new session)."""
    database: Optional[str] = "default"
    """The default database/schema to use."""
    catalog: Optional[str] = "hive"
    """The catalog to use (e.g., 'hive')."""


class SparkCursor(Cursor):
    """A Spark-specific cursor that inherits from the generic Cursor class."""

    def __init__(
        self,
        conn_args: ConnArgs,
        convert_from_presto: bool = False,
        query_engine: str = None,
        **kwargs,
    ):
        """Initializes the Spark cursor with connection parameters.

        Args:
            conn_args: Connection arguments for Spark session
            convert_from_presto: If True, automatically converts Presto/Trino SQL to Spark SQL before execution. Default is False.
            query_engine: The query engine type (e.g., 'custom_spark'). When 'custom_spark',
                additional pre/post processing is applied for Blah-specific Spark compatibility.
            **kwargs: Additional arguments passed to parent Cursor class
        """

        # Pass conn_args to Cursor.__init__
        super().__init__(
            conn_args,
            max_retries=getattr(conn_args, "max_retries", DEFAULT_MAX_RETRIES),
            retry_delay_seconds=getattr(conn_args, "retry_delay_seconds", DEFAULT_RETRY_DELAY_SECONDS),
            **kwargs,
        )

        # Initialize state variables for Spark DataFrame management
        # _spark_df is a pointer to distributed Spark DataFrame (lazy, no memory impact)
        # Only materializes when .collect() or .toPandas() is called
        self._spark_df = None
        self._collected_spark_df = None

        # Store query engine for SQL conversion
        self.query_engine = query_engine

        # Initialize SQL converter if needed (Presto -> Spark)
        if convert_from_presto:
            self.sql_converter = SqlConversion(
                from_dialect="presto",
                to_dialect="spark",
                query_engine=query_engine,
            )
            engine_info = f" (query_engine={query_engine})" if query_engine else ""
            print(f"\n***setting SQL conversion from Presto to Spark{engine_info}")
        else:
            self.sql_converter = None

        # Set the default database if specified
        if conn_args.database and conn_args.database != "default":
            print("\n***setting database")
            database_query = "USE {}".format(conn_args.database)
            print("\n*** database_query: {} was run on Spark.".format(database_query))

            # Use the inherited execute method for retries
            try:
                self.execute(database_query)
            except Exception as e:
                print("\n*** Warning: Could not set database {}: {}".format(conn_args.database, e))

    @property
    def spark_session(self):
        """Returns the SparkSession (reuses existing or creates new)."""
        if self.conn_args.spark_session is not None:
            return self.conn_args.spark_session
        return SparkSession.builder.appName(self.conn_args.app_name).enableHiveSupport().getOrCreate()

    # --- IMPLEMENTATION OF ABSTRACT METHOD ---
    def _execute_core(self, query: str):
        """
        Implements the core execution logic for Spark SQL (required by Cursor base class).
        This method will be wrapped by the parent's retry logic.
        Automatically converts Presto/Trino SQL to Spark SQL if enabled.
        """
        # Clear previous query state
        self._spark_df = None
        self._collected_spark_df = None

        try:
            # Convert SQL if needed (Presto -> Spark)
            if self.sql_converter is not None:
                query = self.sql_converter.convert(query)

            # Execute the query using Spark SQL
            df_result = self.spark_session.sql(query)

            # Store the Spark DataFrame for later fetching (lazy pointer, no memory impact)
            self._spark_df = df_result

            # Update description based on DataFrame schema
            if df_result is not None:
                schema = df_result.schema
                self.description = [(field.name, str(field.dataType), None, None, None, None, None) for field in schema.fields]
                print("self.description: {}".format(self.description))

        except Exception as e:
            # Contextual logging before the parent's retry loop catches the exception
            print("*** Spark specific error: {}".format(e))

            # Raise the exception for the parent's _execute_with_retries to handle the retry/failure
            raise e

    # --- IMPLEMENTATION OF ABSTRACT FETCH METHODS ---
    def fetchall(self) -> List[Tuple[Any, ...]]:
        """Fetches all rows of a query result as a list of tuples."""
        if self._spark_df is None:
            raise ValueError("Spark DataFrame is not initialized.")

        # Use cached results if available to avoid expensive collect()
        if self._collected_spark_df is None:
            self._collected_spark_df = self._spark_df.collect()

        return [tuple(row) for row in self._collected_spark_df]

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        """Fetches the next row of a query result set."""
        if self._spark_df is None:
            raise ValueError("Spark DataFrame is not initialized.")

        # If we haven't collected the results yet, do it now
        if self._collected_spark_df is None:
            self._collected_spark_df = self._spark_df.collect()

        # Return first uncollected row
        if self._collected_spark_df and len(self._collected_spark_df) > 0:
            return tuple(self._collected_spark_df.pop(0))

        return None

    # --- OVERRIDE BASE CLASS METHOD ---

    def get_df(self, query: str) -> SqlQueryResult:
        """
        Override base class get_df to execute query and return SqlQueryResult.
        Consistent with Cursor class - returns SqlQueryResult containing pandas DataFrame.

        Args:
            query: SQL query to execute

        Returns:
            SqlQueryResult with pandas DataFrame (you can skip time and size fields)
        """
        start_time = time.time()
        self.execute(query)

        if self._spark_df is not None:
            # toPandas() materializes data to driver memory - use only for small data
            df = self._spark_df.toPandas()
            end_time = time.time()
            query_time_taken = end_time - start_time
            size_in_megabytes = sys.getsizeof(df) / 10**6 if df is not None else 0

            return SqlQueryResult(df=df, query_time_taken=query_time_taken, size_in_megabytes=size_in_megabytes)

        return SqlQueryResult(df=None, query_time_taken=0, size_in_megabytes=0)

    def get_spark_df(self, query: str):
        """
        Execute query and return Spark DataFrame.
        Use this when you need the native Spark DataFrame for distributed operations.

        Args:
            query: SQL query to execute

        Returns:
            Spark DataFrame with query results
        """
        self.execute(query)
        if self._spark_df is not None:
            return self._spark_df
        raise ValueError("Query execution did not produce a DataFrame.")

    # --- UTILITY METHODS (NOT IN BASE CLASS) ---

    def show_df(self, n=20, truncate=True):
        """
        Shows the current query result (Spark DataFrame.show()).
        This is a convenience method for testing in CloudNotebook.

        Args:
            n: Number of rows to show (default 20)
            truncate: Whether to truncate long strings (default True)
        """
        if not hasattr(self, "_spark_df") or self._spark_df is None:
            raise ValueError("No query has been executed yet.")

        self._spark_df.show(n=n, truncate=truncate)

    def write_spark_df(
        self,
        df: Any,
        table_name: str,
        mode: str = "append",
        data_format: Optional[str] = None,
        partition_col: Optional[str] = None,
        partition_value: Optional[str] = None,
    ) -> None:
        """Write a Spark DataFrame into a Spark/Hive table.

        More efficient than ``write_pandas_df`` when you already have a native
        Spark DataFrame (e.g. from ``get_spark_df``), as it skips the
        pandas→Spark conversion.

        Args:
            df: The Spark DataFrame to write.
            table_name: Fully qualified target table name
                (e.g. ``"hive.schema.forecast_store"``).
            mode: ``"append"`` or ``"overwrite"``.
            data_format: Storage format, e.g. ``"orc"`` or ``"parquet"``.
            partition_col: Column name to partition by.
            partition_value: Required when ``partition_col`` is set.
        """
        if partition_col and not partition_value:
            raise ValueError("partition_value is required when partition_col is set.")
        if partition_col and partition_value and partition_col not in df.columns:
            df = df.withColumn(partition_col, lit(partition_value))
        writer = df.write.mode(mode)
        if data_format:
            writer = writer.format(data_format)
        if partition_col:
            writer = writer.partitionBy(partition_col)
        writer.saveAsTable(table_name)
        print(
            f"\n***: wrote Spark DataFrame into '{table_name}' " f"(mode={mode!r}, format={data_format!r}, " f"partition={partition_col}={partition_value!r})."
        )

    def write_pandas_df(
        self,
        df: "pd.DataFrame",
        table_name: str,
        mode: str = "append",
        data_format: Optional[str] = None,
        partition_col: Optional[str] = None,
        partition_value: Optional[str] = None,
    ) -> None:
        """Write a pandas DataFrame into a Spark/Hive table.

        Converts the pandas DataFrame to a Spark DataFrame, then delegates to
        ``write_spark_df``.

        Args:
            df: The pandas DataFrame to write.
            table_name: Fully qualified target table name
                (e.g. ``"hive.schema.forecast_store"``).
            mode: ``"append"`` or ``"overwrite"``.
            data_format: Storage format, e.g. ``"orc"`` or ``"parquet"``.
            partition_col: Column name to partition by.
            partition_value: Required when ``partition_col`` is set.
        """
        spark_df = self.spark_session.createDataFrame(df)
        self.write_spark_df(
            df=spark_df,
            table_name=table_name,
            mode=mode,
            data_format=data_format,
            partition_col=partition_col,
            partition_value=partition_value,
        )

    def close(self):
        """Closes the Spark session and clears internal state."""
        # Clear cached state
        self._spark_df = None
        self._collected_spark_df = None

        # Only close if we created the session (not provided by user)
        if self.conn_args.spark_session is None:
            self.spark_session.stop()

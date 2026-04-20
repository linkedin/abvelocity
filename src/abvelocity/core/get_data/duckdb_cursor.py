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


from dataclasses import dataclass
from typing import Any, Optional

import duckdb
import pandas as pd
from abvelocity.core.get_data.cursor import Cursor


@dataclass
class DuckDBConnArgs:
    """This is a dataclass to store DuckDB connection arguments, primarily the database path."""

    database_path: str = ":memory:"
    """The path to the DuckDB file, or ':memory:' for an in-memory database."""
    read_only: Optional[bool] = False
    """Whether to open the connection in read-only mode."""


class DuckDBCursor(Cursor):
    """A DuckDB-specific cursor that inherits from the generic Cursor class."""

    def __init__(self, conn_args: DuckDBConnArgs = None, **kwargs):
        """Initializes the DuckDB cursor with connection parameters.
        If conn_args is None, it defaults to in-memory mode."""

        # 1. Handle Default Argument
        if conn_args is None:
            conn_args = DuckDBConnArgs()

        # 2. Call the base class __init__ (passes through retry params via kwargs)
        super().__init__(conn_args, **kwargs)

        # 3. Create connection and assign the specific cursor to self._cursor
        self._db_connection = self._create_duckdb_connection()
        self._cursor = self._db_connection.cursor()
        print(f"\n***: DuckDB cursor created, connected to '{conn_args.database_path}'.")

    def _create_duckdb_connection(self) -> Any:
        """Creates and returns a DuckDB connection object."""
        conn = duckdb.connect(database=self.conn_args.database_path, read_only=self.conn_args.read_only)
        return conn

    # --------------------------------------------------------------------------
    # IMPLEMENTATION OF ABSTRACT METHOD (Core Execution)
    # --------------------------------------------------------------------------
    def _execute_core(self, query: str):
        """
        Implements the core execution logic for DuckDB (required by Cursor base class).
        This method is wrapped by the parent's retry logic.
        """
        print(f"\n*** executing query:\n{query}")

        # Execute the query
        self._cursor.execute(query)

        # Update description for the base class
        self.description = self._cursor.description
        print(f"self.description: {self.description}")

        # Note: No need for fetchall/self.results manipulation here,
        # as the base class handles the fetching via execute_and_fetchall.

    # --------------------------------------------------------------------------
    # IMPLEMENTATION OF ABSTRACT FETCH METHODS
    # --------------------------------------------------------------------------

    # NOTE: The public execute method is now inherited from Cursor and provides retries.
    # The redundant, original `execute` method is removed.

    def fetchall(self):
        """Fetches all rows of a query result as a list of tuples or lists."""
        return self._cursor.fetchall()

    def fetchone(self):
        """Fetches the next row of a query result set."""
        return self._cursor.fetchone()

    def write_pandas_df(
        self,
        df: "pd.DataFrame",
        table_name: str,
        mode: str = "append",
    ) -> None:
        """Write a pandas DataFrame into a DuckDB table.

        DuckDB can reference a pandas DataFrame directly by name after
        registering it, so no intermediate files are needed.

        Args:
            df: The pandas DataFrame to write.
            table_name: Target table name (may include schema, e.g.
                ``"main.forecast_store"``).
            mode: ``"append"`` inserts rows into an existing table (creating
                it first if it does not exist);  ``"overwrite"`` replaces the
                table completely.
        """
        tmp = "_write_pandas_df_tmp"
        self._db_connection.register(tmp, df)
        try:
            if mode == "overwrite":
                self._db_connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {tmp}")
            else:
                # Create table from the DataFrame schema if it doesn't exist yet,
                # then INSERT all rows.
                self._db_connection.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS " f"SELECT * FROM {tmp} WHERE 1=0")
                self._db_connection.execute(f"INSERT INTO {table_name} SELECT * FROM {tmp}")
        finally:
            self._db_connection.unregister(tmp)
        print(f"\n***: wrote {len(df)} rows into DuckDB table '{table_name}' (mode={mode!r}).")

    def close(self):
        """Closes the DuckDB connection."""
        if self._db_connection:
            self._db_connection.close()
            print("\n***: DuckDB connection closed.")

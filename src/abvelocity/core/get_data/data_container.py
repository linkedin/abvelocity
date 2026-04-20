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
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini


from dataclasses import dataclass
from typing import Optional

import pandas as pd
from abvelocity.core.get_data.cursor import Cursor
from abvelocity.core.get_data.db_interact import CreateDBTableSpecs
from abvelocity.core.param.io_param import IOParam


@dataclass
class DataContainer:
    """A dataclass to manage a data flow, materializing to either pandas DataFrames or SQL tables."""

    table_name: Optional[str] = None  # SQL table name; if set, implies table exists unless set by join
    pandas_df: Optional[pd.DataFrame] = None  # Pandas DataFrame, used when materialized to df
    query: Optional[str] = None  # SQL query to define data; ignored if table_name exists (for source)
    is_pandas_df: bool = False  # True if data is materialized in df
    is_sql_table: bool = False  # True if data is materialized in a SQL table

    def join(
        self,
        other: "DataContainer",
        join_type: str = "INNER",
        on: str = None,
        table_name: Optional[str] = None,
        cursor: Optional["Cursor"] = None,
        materialize_to_sql: bool = False,
    ) -> "DataContainer":
        """
        Join this DataContainer with another, preparing the result without materializing unless both are DataFrames
        or materialization to SQL is requested for SQL-related joins.

        Args:
            other: The other DataContainer to join with.
            join_type: Type of SQL join ("INNER", "LEFT", "RIGHT", "FULL").
            on: SQL join condition (e.g., "a.id = b.id AND a.date = b.date"). Required for all joins.
                For DataFrames, only equality conditions are supported (parsed into left_on/right_on).
            table_name: Optional target table name for the join result (required if materialize_to_sql=True).
            cursor: Optional Cursor instance to execute_and_fetchall the join query if materialize_to_sql=True (SQL cases only).
            materialize_to_sql: If True and cursor is provided, materializes SQL-related joins to a SQL table.

        Returns:
            A new DataContainer with the join prepared (as a DataFrame if both are DataFrames,
            as a SQL table if materialized, else a query).

        Raises:
            ValueError: If data is missing, types mismatch, 'on' is invalid, or materialization args are inconsistent.
            TypeError: If cursor is provided but not a Cursor instance.
        """
        # Check if both have data to join
        has_left_data = (
            (self.is_pandas_df and self.pandas_df is not None)
            or (self.is_sql_table and self.table_name is not None)
            or (not self.is_pandas_df and not self.is_sql_table and self.query is not None)
        )
        has_right_data = (
            (other.is_pandas_df and other.pandas_df is not None)
            or (other.is_sql_table and other.table_name is not None)
            or (not other.is_pandas_df and not other.is_sql_table and other.query is not None)
        )

        if not has_left_data:
            raise ValueError("Left DataContainer has no data (no df, table_name, or query)")
        if not has_right_data:
            raise ValueError("Right DataContainer has no data (no df, table_name, or query)")

        # Abort if one is DataFrame and the other is not
        if (self.is_pandas_df and not other.is_pandas_df) or (not self.is_pandas_df and other.is_pandas_df):
            raise ValueError("Cannot join: one is a DataFrame and the other is not")

        # Ensure we have a join condition
        if on is None:
            raise ValueError("Join condition ('on') is required for all joins")

        # Validate materialization parameters (only applies to SQL case)
        if materialize_to_sql:
            if cursor is None:
                raise ValueError("Cursor must be provided when materialize_to_sql is True")
            if table_name is None:
                raise ValueError("table_name must be provided when materialize_to_sql is True")
            if not isinstance(cursor, Cursor):
                raise TypeError("cursor must be an instance of Cursor")

        # Case 1: Both are DataFrames - join immediately in pandas, no SQL materialization
        if self.is_pandas_df and other.is_pandas_df:
            if self.pandas_df is None or other.pandas_df is None:
                raise ValueError("Both DataFrames must be non-None for join")
            # Parse SQL-like 'on' condition for equality joins (e.g., "a.id = b.id AND a.date = b.date")
            conditions = on.split(" AND ")
            left_keys = []
            right_keys = []
            for cond in conditions:
                try:
                    left, right = cond.split(" = ")
                    left_keys.append(left.split(".")[-1])  # Extract column name after alias
                    right_keys.append(right.split(".")[-1])
                except ValueError:
                    raise ValueError("For DataFrame joins, 'on' must contain equality conditions (e.g., 'a.id = b.id')")
            result_df = self.pandas_df.merge(other.pandas_df, how=join_type.lower(), left_on=left_keys, right_on=right_keys)
            return DataContainer(pandas_df=result_df, table_name=table_name, is_pandas_df=True)

        # Case 2: Both are SQL-related - create a join query and optionally materialize to SQL
        left_ref = self.table_name if self.table_name is not None else self.query
        right_ref = other.table_name if other.table_name is not None else other.query

        if not (left_ref and right_ref):
            raise ValueError("Insufficient data to create join query")

        # Select columns from first table (a) and non-duplicate columns from second table (b)
        join_query = "SELECT a.*, "
        cursor.execute_and_fetchall(f"SELECT * FROM {right_ref}")  # get the column names of the second table
        right_cols = [desc[0] for desc in cursor.description]
        cursor.execute_and_fetchall(f"SELECT * FROM {left_ref}")  # get the column names of the first table
        left_cols = [desc[0] for desc in cursor.description]

        right_cols_to_select = [col for col in right_cols if col not in left_cols]

        join_query += ", ".join(f"b.{col}" for col in right_cols_to_select)
        join_query += f" FROM {left_ref} AS a {join_type} JOIN {right_ref} AS b ON {on}"

        if materialize_to_sql:
            # Drop the table if it exists, then create it
            drop_query = f"DROP TABLE IF EXISTS {table_name}"
            cursor.execute_and_fetchall(drop_query)
            create_query = f"CREATE TABLE {table_name} AS {join_query}"
            cursor.execute_and_fetchall(create_query)
            return DataContainer(table_name=table_name, is_sql_table=True)

        # Default: return as query
        return DataContainer(query=join_query, table_name=table_name, is_pandas_df=False, is_sql_table=False)

    def materialize(self, cursor: "Cursor", to_pandas_df: bool = True, to_sql_table: bool = False) -> "DataContainer":
        """
        Materializes the DataContainer's query using the provided Cursor.

        Args:
            cursor: A Cursor instance to execute_and_fetchall the query.
            to_pandas_df: If True, materializes the result into a DataFrame (sets is_pandas_df=True).
            to_sql_table: If True and table_name is set, materializes into a SQL table (sets is_sql_table=True).

        Returns:
            Self with updated state (df or table_name materialized).

        Raises:
            ValueError: If no query exists, or if to_sql_table is True but table_name is None.
            TypeError: If cursor is not a Cursor instance.
        """
        if self.query is None:
            raise ValueError("No query to materialize")

        if to_sql_table and self.table_name is None:
            raise ValueError("Cannot materialize to SQL table: table_name is None")

        if to_pandas_df:
            result = cursor.get_df(self.query)
            self.pandas_df = result.df
            self.is_pandas_df = True
            self.is_sql_table = False
            print(f"Materialized to DataFrame: {result.size_in_megabytes} MB, took {result.query_time_taken} seconds")

            # Remove duplicate columns based on name, after the query is run.
            cols = pd.Series(self.pandas_df.columns)
            seen = set()
            unique_cols = []
            for col in cols:
                if col not in seen:
                    unique_cols.append(col)
                    seen.add(col)
            self.pandas_df = self.pandas_df[unique_cols]

        if to_sql_table:
            create_query = f"CREATE TABLE {self.table_name} AS {self.query}"
            cursor.execute_and_fetchall(create_query)
            self.pandas_df = None
            self.is_pandas_df = False
            self.is_sql_table = True
            print(f"Materialized to SQL table: {self.table_name}")

        return self

    def materialize_to_pandas_df(self, cursor: "Cursor") -> "DataContainer":
        """If there is a table_name or query, it takes that and materialize to pandas df.
        It does check to see if no pandas_df is already existing, and will issue a warning if it does.
        Note this could be used to update the df too in some cases.
        """
        if self.table_name is not None:
            print(f"\n*** In `DataContainer`, we rely on dc.table_name to extract data:\n{self.table_name}")
            materialized_df_result = cursor.get_df(query=f"SELECT * FROM ({self.table_name})")
            self.pandas_df = materialized_df_result.df
        elif self.query is not None:
            print(f"\n*** In `DataContainer` we rely on dc.query to extract data:\n{self.query}")
            materialized_df_result = cursor.get_df(query=self.query)
            self.pandas_df = materialized_df_result.df
            print("\n*** In `DataContainer` data was materialized as pandas_df.")
        else:
            raise ValueError(
                "In `DataContainer`, Since dc.pandas_df is to be extracted, DataContainer must have either table_name or query to materialize data."
            )
        self.is_pandas_df = True

    def materialize_to_db_table(
        self,
        io_param: IOParam,
        create_db_table_specs: CreateDBTableSpecs,
    ) -> "DataContainer":
        """Simply takes the self.query and use that to materialize to DB sql table.
        See `CreateDBTableSpecs` for arguments.
        Note that

            - dc.query is expected to exist and relied upon to materialize the data.
            - an exsiting table name could be over-written
            - if table_name and table_name_core are not passed to this function, self.table_name will be used.


        """
        if self.query is None:
            raise ValueError(
                "You have requested to materizale a DataContainer to a DB table. "
                "This implies that the dc {dc} contains a .query attribute / field. "
                "However no such attribute was found."
            )

        # If no table_name of any sorts can be found in the create_db_table_specs or DC, then we return an error
        # If DC includes a table_name and create_db_table_specs does not include either of table_name and table_name_core, then
        # we add the one from DC to the specs
        if not create_db_table_specs.table_name and not create_db_table_specs.table_name_core:
            if self.table_name is None:
                raise ValueError(
                    "You have called `DataContainer.materialize_to_db_table` w/o passing a table_name or table_name_core. "
                    "An attempt was made to use self.table_name but that field was also None."
                )
            create_db_table_specs.table_name = self.table_name

        res = create_db_table_specs.run(io_param=io_param, query=self.query)

        self.table_name = res["table_name"]

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


import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pandas as pd

DEFAULT_MAX_RETRIES = 50
"""Default maximum number of retries for query execution."""
DEFAULT_RETRY_DELAY_SECONDS = 1
"""Default delay between retries in seconds."""


@dataclass
class SqlQueryResult:
    """This is a dataclass to store a SQL query dataframe, time taken and the size."""

    df: Optional[pd.DataFrame] = None
    """A `pandas` dataframe which includes the data returned by the SQL query or None."""
    query_time_taken: Optional[float] = 0
    """The time taken to run the query in seconds (float)."""
    size_in_megabytes: Optional[float] = 0
    """The memory size of resulting dataframe."""


class Cursor:
    """
    A generic cursor class that provides a consistent interface for executing queries
    and fetching results, adhering to the PEP 249 standard where possible.
    Includes built-in retry logic for execution.

    Subclasses must implement:
    1. _execute_core(self, query: str)
    2. fetchall(self)
    3. fetchone(self)
    """

    def __init__(
        self,
        conn_args: Any,
        max_retries: Optional[int] = 20,
        retry_delay_seconds: Optional[int] = DEFAULT_RETRY_DELAY_SECONDS,
    ):
        """
        Initializes the cursor with a data source and optional retry configuration.
        """
        self.conn_args = conn_args
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.description = None
        self.results = None
        self._cursor = None
        """This is the internal / specific cursor which depends on implementation.
        We expect this internal cursor to have a description field.
        This is used in `execute_and_fetchall` method.
        """

    # --------------------------------------------------------------------------
    # CORE EXECUTION METHODS
    # --------------------------------------------------------------------------

    def _execute_with_retries(self, query: str):
        """
        Implements the retry loop, delegating the core execution logic to the abstract
        _execute_core method.
        """
        max_retries = self.max_retries
        retry_delay = self.retry_delay_seconds

        attempt = 0
        last_exception = None
        success = False

        while attempt < max_retries and not success:
            attempt += 1

            if attempt > 1:
                print(f"*** Operation failed on attempt {attempt-1}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

            try:
                # Delegate to the child class for actual database execution
                self._execute_core(query)
                success = True

            except Exception as e:
                last_exception = e
                # The implementation of _execute_core is responsible for specific logging

        # If the loop finishes without success, raise the last exception
        if not success:
            print(f"\n***: Operation failed after {max_retries} attempts.")
            raise last_exception

    def _execute_and_fetchall_with_retries(self, query: str):
        """
        Implements the retry loop, delegating the core execution logic to the abstract
        _execute_core method.
        """
        max_retries = self.max_retries
        retry_delay = self.retry_delay_seconds

        attempt = 0
        last_exception = None
        success = False

        while attempt < max_retries and not success:
            attempt += 1

            if attempt > 1:
                print(f"*** Operation failed on attempt {attempt-1}. Retrying in {retry_delay} second(s) ...")
                time.sleep(retry_delay)

            try:
                # Delegate to the child class for actual database execution
                self._execute_core(query)
                self.results = self.fetchall()
                success = True

            except Exception as e:
                last_exception = e
                # The implementation of _execute_core is responsible for specific logging

        # If the loop finishes without success, raise the last exception
        if not success:
            print(f"\n***: Operation failed after {max_retries} attempts.")
            raise last_exception

    def _execute_core(self, query: str):
        """
        ABSTRACT METHOD: Subclasses must implement this method to handle the
        database-specific cursor.execute() call and update self.description.
        Must raise an exception on failure.
        """
        raise NotImplementedError("Subclasses must implement _execute_core to handle database execution.")

    def execute(self, query: str):
        """
        Executes a query on the data source, wrapping the core execution with retry logic.
        """
        self._execute_with_retries(query)

    # --------------------------------------------------------------------------
    # FETCH METHODS (ABSTRACT)
    # --------------------------------------------------------------------------

    def fetchall(self) -> List[Tuple[Any, ...]]:
        """
        ABSTRACT METHOD: Fetches all rows of a query result as a list of tuples or lists.
        Subclasses must implement.
        """
        raise NotImplementedError("Subclasses must implement fetchall.")

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        """
        ABSTRACT METHOD: Fetches the next row of a query result set.
        Subclasses must implement.
        """
        raise NotImplementedError("Subclasses must implement fetchone.")

    # --------------------------------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------------------------------

    def execute_and_fetchall(self, query: str) -> None:
        """
        Executes a query on the data source and fetches all results immediately.
        """
        self._execute_and_fetchall_with_retries(query)
        # If the specific cursor updated its description, use it.
        if self._cursor is not None and hasattr(self._cursor, "description"):
            try:
                self.description = self._cursor.description
            except Exception as e:
                print(f"\n*** No description was found in the cursor, bypassed error {e}")

    def get_df(self, query: str) -> SqlQueryResult:
        """
        Executes a query, fetches all results, and returns them as a SqlQueryResult
        dataclass instance containing a pandas DataFrame.
        """
        start_time = time.time()

        # Execute the query, and fetch all results immediately to build the DataFrame
        self.execute_and_fetchall(query)

        if self.description:
            # Assumes description format is (name, type_code, ...)
            columns = [col_info[0] for col_info in self.description]
            print(f"\n*** col_names: {columns}")

            print("\n*** create dataframe")
            # Uses self.results populated by execute_and_fetchall
            df = pd.DataFrame(self.results, columns=columns)
            end_time = time.time()
            query_time_taken = end_time - start_time
            size_in_megabytes = sys.getsizeof(df) / 10**6 if df is not None else 0

            return SqlQueryResult(df=df, query_time_taken=query_time_taken, size_in_megabytes=size_in_megabytes)

        return SqlQueryResult(df=None, query_time_taken=0, size_in_megabytes=0)

    def execute_multi(self, query: str) -> None:
        """
        Executes multiple queries separated by ';'.
        """
        sub_queries = query.strip().split(";")

        for sub_query in sub_queries:
            sub_query = sub_query.strip()
            if sub_query:
                print(f"\n***\n sub_query: {sub_query}")
                self.execute(sub_query)

    def execute_and_fetchall_multi(self, query: str) -> None:
        """
        Executes multiple queries separated by ';', fetching results for each.
        """
        sub_queries = query.strip().split(";")

        for sub_query in sub_queries:
            sub_query = sub_query.strip()
            if sub_query:
                print(f"\n***\n sub_query: {sub_query}")
                self.execute_and_fetchall(sub_query)

    def write_pandas_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str = "append",
    ) -> None:
        """Write a pandas DataFrame into a database table.

        ABSTRACT METHOD: subclasses must implement this for their engine.

        Args:
            df: The pandas DataFrame to write.
            table_name: Fully qualified target table name
                (e.g. ``"schema.forecast_store"``).
            mode: ``"append"`` adds rows to an existing table (creating it if
                needed); ``"overwrite"`` replaces the table contents.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement write_pandas_df.")

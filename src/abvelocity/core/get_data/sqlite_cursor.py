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

import sqlite3
from typing import Any, List, Optional, Tuple

from abvelocity.core.get_data.cursor import Cursor


class SqliteCursor(Cursor):
    """
    A SQLite-specific cursor implementation for testing and compatibility with DataContainer.
    """

    def __init__(self, conn_args: sqlite3.Connection, **kwargs):
        """
        Initializes the SqliteCursor with a SQLite connection.

        Args:
            conn_args: A sqlite3.Connection object.
        """
        # Pass conn_args and kwargs (for retries) to the base class
        super().__init__(conn_args, **kwargs)

        # Assign the underlying SQLite cursor to the inherited self._cursor attribute
        self._cursor = conn_args.cursor()

    # --------------------------------------------------------------------------
    # IMPLEMENTATION OF ABSTRACT METHOD (Core Execution)
    # --------------------------------------------------------------------------
    def _execute_core(self, query: str) -> None:
        """
        Implements the core execution logic for SQLite, wrapped by the parent's retry logic.
        """
        try:
            # 1. Execute the query using the internal cursor
            self._cursor.execute(query)

            # 2. Update description for the base class
            self.description = self._cursor.description

            # 3. Handle DDL/DML commit (SQLite specific behavior)
            if not query.strip().upper().startswith("SELECT"):
                self.conn_args.commit()
                self.description = None  # Clear description for non-SELECT

        except sqlite3.Error as e:
            # Must re-raise the error so the base class retry logic can catch it
            raise e

    # NOTE: The custom `execute` method is removed, replaced by the inherited one.

    # --------------------------------------------------------------------------
    # IMPLEMENTATION OF ABSTRACT FETCH METHODS
    # --------------------------------------------------------------------------
    def fetchall(self) -> List[Tuple[Any, ...]]:
        """
        Fetches all rows of a query result as a list of tuples.
        """
        # Read from the internal cursor
        results = self._cursor.fetchall()
        return results if results is not None else []

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        """
        Fetches the next row of a query result set.
        """
        # Read from the internal cursor
        return self._cursor.fetchone()

    def close(self) -> None:
        """Closes the SQLite connection."""
        # Close the internal cursor first
        if self._cursor:
            self._cursor.close()

        # Close the connection stored in self.conn_args
        if self.conn_args:
            self.conn_args.close()

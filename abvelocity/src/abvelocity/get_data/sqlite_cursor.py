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

from abvelocity.get_data.cursor import Cursor


class SqliteCursor(Cursor):
    """
    A SQLite-specific cursor implementation for testing and compatibility with DataContainer.
    """

    def __init__(self, data_source: sqlite3.Connection):
        """
        Initializes the SqliteCursor with a SQLite connection.

        Args:
            data_source: A sqlite3.Connection object.
        """
        super().__init__(data_source)

    def execute(self, query: str) -> None:
        """
        Executes a SQLite query.

        Args:
            query: The SQL query string.
        """
        conn = self.data_source
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            if query.strip().upper().startswith("SELECT"):
                self.description = cursor.description
                self.results = cursor.fetchall()
            else:
                conn.commit()  # Persist DDL/DML changes
                self.description = None
                self.results = None
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            raise
        finally:
            cursor.close()

    def fetchall(self) -> Optional[List[Tuple[Any, ...]]]:
        """
        Fetches all rows of a query result as a list of tuples.

        Returns:
            A list of tuples representing the query results, or None if no results.
        """
        return self.results

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        """
        Fetches the next row of a query result set.

        Returns:
            A tuple representing the next row, or None if no more rows.
        """
        if self.results:
            return self.results.pop(0) if len(self.results) > 0 else None
        return None

    def close(self) -> None:
        """Closes the SQLite connection."""
        if self.data_source:
            self.data_source.close()

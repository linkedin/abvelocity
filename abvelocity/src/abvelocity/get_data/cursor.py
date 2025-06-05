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
from typing import Any, Optional

import pandas as pd


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
    """

    def __init__(self, data_source: Any):
        """
        Initializes the cursor with a data source.
        """
        self.data_source = data_source
        self.description = None
        self.results = None

    def execute(self, query: str):
        """
        Executes a query on the data source.
        Needs to update
            self.description
            self.results
        """
        pass

    def fetchall(self):
        """
        Fetches all rows of a query result as a list of tuples or lists.
        """
        return []

    def fetchone(self):
        """
        Fetches the next row of a query result set.
        """
        return None

    def get_df(self, query: str) -> SqlQueryResult:
        """
        Executes a query and returns the results as a SqlQueryResult dataclass instance.
        """
        start_time = time.time()
        self.execute(query)
        if self.description:
            columns = [col_info[0] for col_info in self.description]
            print(f"\n*** col_names: {columns}")

            print("\n*** create dataframe")
            df = pd.DataFrame(self.results, columns=columns)
            end_time = time.time()
            query_time_taken = end_time - start_time
            size_in_megabytes = sys.getsizeof(df) / 10**6 if df is not None else 0

            return SqlQueryResult(
                df=df, query_time_taken=query_time_taken, size_in_megabytes=size_in_megabytes
            )

        return SqlQueryResult(df=None, query_time_taken=0, size_in_megabytes=0)

    def execute_multi(self, query: str) -> None:
        """
        Executes multiple queries separated by ';'.
        """
        sub_queries = query.strip().split(";")

        for sub_query in sub_queries:
            sub_query = sub_query.strip()
            if sub_query:  # Ensure it's not empty
                print(f"\n***\n sub_query: {sub_query}")
                self.execute(sub_query)

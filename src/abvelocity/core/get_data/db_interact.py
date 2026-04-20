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

# Function to create a table in the database based on a query

from dataclasses import dataclass
from typing import Any, Dict, Optional

from abvelocity.core.param.io_param import IOParam


def create_db_table(
    io_param: IOParam,
    query: str,
    table_name: Optional[str] = None,
    table_name_core: Optional[str] = None,
    add_date_suffix: Optional[bool] = False,
    add_date_suffix_format: Optional[str] = "%Y%m%d",
    drop_if_exists: Optional[bool] = True,
    dialect: Optional[str] = "Trino",
    retries: Optional[int] = 1,
) -> Dict[str, Any]:
    """Creates a table in the database based on the provided query.

    Args:
            io_param (IOParam): IO parameters including the cursor.
            query (str): The SQL query to create the table from.
            table_name (Optional[str]): The name of the table to be created. If not provided,
                table_name_core must be provided.
            table_name_core (Optional[str]): Core part of the table name to be used if
                table_name is not provided.
            add_date_suffix (Optional[bool]): Whether to add a date suffix to the table name.
            add_date_suffix_format (Optional[str]): The date format for the suffix.
            drop_if_exists (Optional[bool]): Whether to drop the table if it already exists.
            dialect (Optional[str]): The SQL dialect to use.
            retries (Optional[int]): Number of times to retry the operation in case of failure.

    Returns:
            dict: A dictionary with the table name and success status.
    """

    cursor = io_param.cursor
    if table_name is None:
        if table_name_core is None:
            raise ValueError("Either table_name or table_name_core must be provided.")
        table_name = io_param.gen_table_name(
            table_name_core=table_name_core,
            add_date_suffix=add_date_suffix,
            date_format=add_date_suffix_format,
        )

    if dialect != "Trino":
        raise NotImplementedError(f"Dialect {dialect} not supported yet.")

    creation_query = f"CREATE TABLE {table_name} AS {query}"
    if drop_if_exists:
        creation_query = f"DROP TABLE IF EXISTS {table_name};" + creation_query
        # Note: CREATE alone should work too. But for safety REPLACE is included.
        # creation_query = f"CREATE OR REPLACE TABLE {table_name} AS ({query})"
        # creation_query = f"CREATE TABLE {table_name} AS ({query})"

    success = False
    while retries > 0 and not success:
        retries -= 1
        try:
            cursor.execute_and_fetchall_multi(creation_query)
            success = True
            print(f"Table {table_name} created successfully.")
        except Exception as e:
            print(f"Failed to create table {table_name}: {e}")

    return {"table_name": table_name, "success_status": success}


@dataclass
class CreateDBTableSpecs:
    """See args in create_db_table."""

    table_name: Optional[str] = None

    table_name_core: Optional[str] = None

    add_date_suffix: Optional[bool] = False

    add_date_suffix_format: Optional[str] = "%Y%m%d"

    drop_if_exists: Optional[bool] = True

    dialect: Optional[str] = "Trino"

    retries: Optional[int] = 1

    def run(self, io_param: IOParam, query: str) -> Dict[str, Any]:
        return create_db_table(
            io_param=io_param,
            query=query,
            table_name=self.table_name,
            table_name_core=self.table_name_core,
            add_date_suffix=self.add_date_suffix,
            add_date_suffix_format=self.add_date_suffix_format,
            drop_if_exists=self.drop_if_exists,
            dialect=self.dialect,
            retries=self.retries,
        )

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


import os
from typing import Any, Dict, List, Optional

from abvelocity.get_data.cursor import Cursor


def desc_table(
    cursor: Cursor,
    table_name: str,
    count_distinct_cols: Optional[List[str]] = None,
    distinct_values_cols: Optional[List[str]] = None,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Describes a table by retrieving metadata, number of rows, and optionally calculates
    count distinct and distinct values for specified columns.

    Args:
        cursor: A database cursor (already connected) with `execute` method.
        table_name (str): The name of the table to describe.
        count_distinct_cols (Optional[List[str]]): List of column names to calculate COUNT(DISTINCT).
        distinct_values_cols (Optional[List[str]]): List of column names to retrieve distinct values.
        log_file (Optional[str]): File path to store the printed results. Default is None.

    Returns:
        Dict: A dictionary containing table metadata and optional calculations.
    """
    desc_dict = {"table_name": table_name}

    try:
        # Retrieve column metadata and row count
        cursor.execute(f"DESCRIBE {table_name}")
        columns_info = cursor.fetchall()

        desc_dict["columns"] = [{"name": col[0], "type": col[1]} for col in columns_info]

        # Fetch the total number of rows in the table
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        desc_dict["number_of_rows"] = total_rows

        # Fetch the first 10 rows of the table
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
        desc_dict["rows"] = cursor.fetchall()

        # Calculate COUNT(DISTINCT) for specified columns
        if count_distinct_cols:
            count_distinct = {}
            for column in count_distinct_cols:
                cursor.execute(f"SELECT COUNT(DISTINCT {column}) FROM {table_name}")
                distinct_count = cursor.fetchone()[0]
                count_distinct[column] = distinct_count
            desc_dict["count_distinct"] = count_distinct

        # Retrieve distinct values for specified columns
        if distinct_values_cols:
            distinct_values = {}
            for column in distinct_values_cols:
                cursor.execute(f"SELECT DISTINCT {column} FROM {table_name}")
                values = [row[0] for row in cursor.fetchall()]
                distinct_values[column] = values
            desc_dict["distinct_values"] = distinct_values

    except Exception as e:
        desc_dict["error"] = str(e)

    # Convert results to string format
    result_str = "\n".join([f"{key}: {value}" for key, value in desc_dict.items()])

    # Print results
    print(result_str)

    # Write results to log file if provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            f.write(result_str + "\n")

    return desc_dict

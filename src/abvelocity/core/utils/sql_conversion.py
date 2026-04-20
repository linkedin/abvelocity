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
import re
from pathlib import Path
from typing import Optional

from sqlglot import transpile


class SqlConversion:
    """
    A class to handle SQL dialect conversions using sqlglot.

    The class is initialized with a source and a target dialect.
    It can then convert a SQL string from the source to the target.

    Valid dialects can be found in the sqlglot documentation, for example:
    'spark', 'trino', 'hive', 'presto', 'mysql', 'postgres', etc.
    """

    def __init__(self, from_dialect: str, to_dialect: str, query_engine: Optional[str] = None):
        """
        Initializes the SqlConversion instance.

        Args:
            from_dialect (str): The name of the source SQL dialect.
            to_dialect (str): The name of the target SQL dialect.
            query_engine (str, optional): The query engine type (e.g., 'custom_spark').
                When 'custom_spark', additional post-processing is applied for
                Blah-specific Spark compatibility.
        """
        # Store the source and target dialects
        self.from_dialect = from_dialect
        self.to_dialect = to_dialect
        self.query_engine = query_engine

    def convert(self, sql_query: str) -> str:
        """
        Transpiles a given SQL query from the source to the target dialect.

        Args:
            sql_query (str): The SQL query string to be converted.

        Returns:
            str: The converted SQL query string.
        """
        try:
            # Use sqlglot's transpile function to perform the conversion
            converted_query = transpile(
                sql_query,
                read=self.from_dialect,
                write=self.to_dialect,
                pretty=True,  # Optional: formats the output nicely
            )[
                0
            ]  # transpile returns a list, take the first element

            # Post-process for functions that sqlglot doesn't handle correctly
            # Only apply for Blah-specific Spark (custom_spark)
            if self.query_engine == "custom_spark" and self.to_dialect == "spark":
                return self._post_process_for_custom_spark(converted_query)

            return converted_query
        except Exception as e:
            # Simple error handling for cases where the SQL can't be parsed
            return f"Error during conversion: {e}"

    def _post_process_for_custom_spark(self, converted_sql: str) -> str:
        """
        Post-process SQL for Blah-specific Spark compatibility.
        These conversions handle functions that sqlglot doesn't convert correctly
        for the older Spark versions used at Blah.

        Conversions applied:
        - SET SESSION → SET: Spark uses SET without SESSION keyword
        - TRY_ELEMENT_AT → element_at: Spark's element_at returns NULL for out-of-bounds
        - MAP_AGG → MAP_FROM_ENTRIES(COLLECT_LIST(STRUCT(...))): Equivalent aggregation
        - CARDINALITY → SIZE: Both return array/map length

        Args:
            converted_sql (str): The sqlglot-converted SQL query string.

        Returns:
            str: The post-processed SQL query string.
        """
        # SET SESSION in Trino/Presto -> SET in Spark (remove SESSION keyword)
        # sqlglot passes this through unchanged, so we handle it here
        converted_sql = re.sub(
            r"^(\s*)SET\s+SESSION\s+",
            r"\1SET ",
            converted_sql,
            flags=re.IGNORECASE,
        )

        # TRY_ELEMENT_AT doesn't exist in Spark - use element_at instead
        # (Spark's element_at already returns NULL for out-of-bounds, same as TRY version)
        converted_sql = converted_sql.replace("TRY_ELEMENT_AT", "element_at")

        # MAP_AGG doesn't exist in Spark - convert to equivalent
        # MAP_AGG(key, value) -> MAP_FROM_ENTRIES(COLLECT_LIST(STRUCT(key, value)))
        # Both aggregate key-value pairs into a map (last value wins for duplicate keys)
        converted_sql = re.sub(
            r"MAP_AGG\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
            r"MAP_FROM_ENTRIES(COLLECT_LIST(STRUCT(\1, \2)))",
            converted_sql,
        )

        # CARDINALITY doesn't exist in Spark - use SIZE instead
        # cardinality(array) -> size(array) (both return array length)
        converted_sql = re.sub(
            r"\bCARDINALITY\s*\(",
            r"SIZE(",
            converted_sql,
            flags=re.IGNORECASE,
        )

        return converted_sql

    def convert_directory(self, input_path: str, output_path: str = None):
        """
        Recursively converts all SQL files in a directory and its subdirectories.

        Args:
            input_path (str): The path to the directory containing SQL files.
            output_path (str, optional): The path to the output directory
                for converted files. If None,
                files are saved in the same directory
                with a new filename. Defaults to None.
        """
        # Use Path for easy path manipulation and creation
        input_dir = Path(input_path)

        if not input_dir.is_dir():
            print(f"Error: The provided input path '{input_path}' is not a directory.")
            return

        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_dir

        print(f"Starting conversion from '{self.from_dialect}' to '{self.to_dialect}'...")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".sql"):
                    # Construct the full path to the input file
                    input_file_path = Path(root) / file
                    print(f"Converting file: {input_file_path}")

                    # Read the SQL content
                    with open(input_file_path, "r") as f:
                        sql_content = f.read()

                    # Convert the content
                    converted_sql = self.convert(sql_content)

                    # Determine the output file path
                    if output_path:
                        # Recreate the subdirectory structure in the output path
                        relative_path = input_file_path.relative_to(input_dir)
                        output_file_path = output_dir / relative_path
                        output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        # Append the target dialect to the filename
                        filename_stem = input_file_path.stem
                        output_file_path = input_file_path.parent / f"{filename_stem}_{self.to_dialect}.sql"

                    # Write the converted content to the new file
                    with open(output_file_path, "w") as f:
                        f.write(converted_sql)

                    print(f"Saved converted query to: {output_file_path}")
        print("Conversion of all files completed.")

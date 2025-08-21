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

import os
import re
from typing import Dict, List


def _sanitize_filename(name: str) -> str:
    """
    Converts a string into a filename-safe format.
    - Replaces non-alphanumeric characters (except underscore, hyphen, period) with underscores.
    - Replaces multiple underscores with a single underscore.
    - Trims leading/trailing underscores.
    - Converts to lowercase.
    """
    # Replace non-alphanumeric characters with underscores
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    # Replace multiple underscores with a single one
    safe_name = re.sub(r"__+", "_", safe_name)
    # Trim leading/trailing underscores
    safe_name = safe_name.strip("_")
    # Convert to lowercase for consistency
    safe_name = safe_name.lower()
    return safe_name


def write_queries(
    queries_dict: Dict[str, str], output_dir: str, use_prefix: bool = True
) -> List[str]:
    """
    Writes SQL queries from a dictionary to individual .sql files in a specified directory.
    File names are generated with an ordered numerical prefix and sanitized keys.

    Args:
        queries_dict: A dictionary where keys are arbitrary names (e.g., event labels,
                      query descriptions) and values are SQL query strings.
        output_dir: The path to the directory where the SQL files should be written.
                    The directory will be created if it does not exist.
        use_prefix: If True (default), prefixes filenames with a zero-padded number
                    (e.g., "01_query_name.sql"). If False, no prefix is added.

    Returns:
        A list of the full paths to the SQL files that were created.

    Raises:
        IOError: If there's an issue creating the directory, a file with the
                 same name as the output directory already exists, or writing fails.
    """
    if not queries_dict:
        print("The provided queries dictionary is empty. No files will be written.")
        return []

    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            # A file with the same name exists, which is a fatal error.
            raise IOError(
                f"Cannot create directory '{output_dir}' because a file with that name already exists."
            )

        # Check if the directory is not empty.
        if os.listdir(output_dir):
            print(
                f"WARNING: The directory '{output_dir}' already exists and is not empty."
                " New files will be added, and existing files with the same name will be overwritten."
            )

    # Ensure the output directory exists. This is safe even if it already exists.
    os.makedirs(output_dir, exist_ok=True)

    written_files: List[str] = []

    keys = list(queries_dict.keys())
    num_queries = len(keys)
    padding_width = len(str(num_queries))
    # If the number of queries is less than 10, ensure a minimum padding width of 2.
    if num_queries < 10:
        padding_width = 2

    for i, key in enumerate(keys):
        sanitized_key = _sanitize_filename(key)

        if use_prefix:
            # Generate the numerical prefix with zero-padding
            padded_index = str(i + 1).zfill(padding_width)
            # Construct the full filename (e.g., "01_my_query.sql")
            filename = f"{padded_index}_{sanitized_key}.sql"
        else:
            # Construct the filename without a prefix
            filename = f"{sanitized_key}.sql"

        filepath = os.path.join(output_dir, filename)

        # Write the query to the file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(queries_dict[key])
            written_files.append(filepath)
            print(f"Successfully wrote query to: {filepath}")
        except IOError as e:
            print(f"Error writing query to {filepath}: {e}")
            raise  # Re-raise the exception after printing

    return written_files

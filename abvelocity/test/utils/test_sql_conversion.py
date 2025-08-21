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

import shutil
from pathlib import Path

from abvelocity.utils.sql_conversion import SqlConversion

# Use static base path for writing test results and input files.
WRITE_PATH = (
    Path(__file__).parents[5].joinpath("docs/static/test-results/write_queries/conversion_test/")
)


def _normalize_sql_whitespace(sql: str) -> str:
    """
    Removes leading/trailing whitespace and replaces all internal
    whitespace (newlines, tabs, multiple spaces) with a single space.
    This also converts the string to lowercase to handle case sensitivity.
    This makes string comparisons robust to formatting and casing differences.
    """
    return " ".join(sql.strip().lower().split())


def test_convert_from_presto_to_spark():
    """
    Tests the conversion of several SQL queries from Presto/Trino to Spark.
    This function checks for correct dialect conversion on different query structures.
    """
    # Initialize the converter from Presto/Trino to Spark SQL
    presto_to_spark = SqlConversion(from_dialect="presto", to_dialect="spark")

    # Test Case 1: Simple SELECT statement with a date function
    presto_sql_1 = "SELECT CAST(DATE_TRUNC('hour', from_iso8601_timestamp(created)) AS DATE) AS created_date FROM events"
    expected_spark_1 = """
SELECT
    CAST(DATE_TRUNC('hour', from_iso8601_timestamp(created)) AS DATE) AS created_date
FROM events
"""
    converted_sql_1 = presto_to_spark.convert(presto_sql_1)
    assert _normalize_sql_whitespace(converted_sql_1) == _normalize_sql_whitespace(expected_spark_1)

    # Test Case 2: Query with a CTE (Common Table Expression) and multiple joins
    presto_sql_2 = """
    WITH latest_events AS (
        SELECT
            user_id,
            MAX(event_timestamp) AS max_timestamp
        FROM
            events
        GROUP BY
            user_id
    )
    SELECT
        e.user_id,
        e.event_name
    FROM
        events AS e
    JOIN
        latest_events AS le ON e.user_id = le.user_id AND e.event_timestamp = le.max_timestamp
    """
    # The expected Spark SQL output should be identical for this type of query
    expected_spark_2 = """
    WITH latest_events AS (
        SELECT
            user_id,
            MAX(event_timestamp) AS max_timestamp
        FROM
            events
        GROUP BY
            user_id
    )
    SELECT
        e.user_id,
        e.event_name
    FROM
        events AS e
    JOIN
        latest_events AS le
        ON e.user_id = le.user_id AND e.event_timestamp = le.max_timestamp
    """
    converted_sql_2 = presto_to_spark.convert(presto_sql_2)
    assert _normalize_sql_whitespace(converted_sql_2) == _normalize_sql_whitespace(expected_spark_2)

    # Test Case 3: Query with a window function
    presto_sql_3 = """
    SELECT
        user_id,
        event_name,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_timestamp DESC) AS rn
    FROM
        events
    """
    expected_spark_3 = """
    SELECT
        user_id,
        event_name,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_timestamp DESC) AS rn
    FROM
        events
    """
    converted_sql_3 = presto_to_spark.convert(presto_sql_3)
    assert _normalize_sql_whitespace(converted_sql_3) == _normalize_sql_whitespace(expected_spark_3)


def test_convert_directory_presto_to_spark(tmp_path):
    """
    Tests the convert_directory method by creating a temporary
    directory structure and verifying the conversions.
    """
    # Define temp directories with new names
    input_dir = tmp_path / "presto_input"
    output_dir = tmp_path / "sparksql_output"

    # Create input directories and files
    input_dir.mkdir()
    (input_dir / "subdir1").mkdir()
    (input_dir / "subdir2").mkdir()

    query1_presto = """
    CREATE TABLE temp_table
    WITH (
        format = 'ORC',
        partitioned_by = ARRAY['ds']
    ) AS
    WITH ranked_events AS (
        SELECT
            event_id,
            user_id,
            event_type,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_timestamp DESC) AS rn
        FROM
            events
    )
    SELECT
        event_id,
        user_id,
        event_type
    FROM
        ranked_events
    WHERE
        rn = 1
    """
    with open(input_dir / "query1.sql", "w") as f:
        f.write(query1_presto)

    query2_presto = "SELECT date_add('day', 1, CURRENT_DATE) FROM users"
    with open(input_dir / "subdir1" / "query2.sql", "w") as f:
        f.write(query2_presto)

    query3_presto = "DROP TABLE IF EXISTS another_table"
    with open(input_dir / "subdir2" / "query3.sql", "w") as f:
        f.write(query3_presto)

    # Initialize the conversion class
    presto_to_spark = SqlConversion(from_dialect="presto", to_dialect="spark")

    # Run the directory conversion
    presto_to_spark.convert_directory(input_path=str(input_dir), output_path=str(output_dir))

    # Define the expected Spark SQL after conversion, including the extra spaces.
    expected_spark_1 = """
    CREATE TABLE temp_table
    USING ORC
    PARTITIONED BY ( ds ) AS
    WITH ranked_events AS (
        SELECT
            event_id,
            user_id,
            event_type,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_timestamp DESC) AS rn
        FROM
            events
    )
    SELECT
        event_id,
        user_id,
        event_type
    FROM
        ranked_events
    WHERE
        rn = 1
    """
    expected_spark_2 = "SELECT DATE_ADD(CURRENT_DATE, 1) FROM users"
    expected_spark_3 = "DROP TABLE IF EXISTS another_table"

    # Verify that the output directory and files exist
    assert output_dir.exists()
    assert (output_dir / "query1.sql").exists()
    assert (output_dir / "subdir1" / "query2.sql").exists()
    assert (output_dir / "subdir2" / "query3.sql").exists()

    # Verify the content of the converted files
    with open(output_dir / "query1.sql", "r") as f:
        converted_content_1 = f.read()
    assert _normalize_sql_whitespace(converted_content_1) == _normalize_sql_whitespace(
        expected_spark_1
    )

    with open(output_dir / "subdir1" / "query2.sql", "r") as f:
        converted_content_2 = f.read()
    assert _normalize_sql_whitespace(converted_content_2) == _normalize_sql_whitespace(
        expected_spark_2
    )

    with open(output_dir / "subdir2" / "query3.sql", "r") as f:
        converted_content_3 = f.read()
    assert _normalize_sql_whitespace(converted_content_3) == _normalize_sql_whitespace(
        expected_spark_3
    )


def test_convert_directory_presto_to_spark_with_write_path():
    """
    Tests the convert_directory method using the predefined WRITE_PATH.
    This test creates fixed input and output subdirectories under the
    WRITE_PATH and saves the converted files to the output directory.
    The test does not clean up the output.
    """
    input_dir = WRITE_PATH / "presto_input"
    output_dir = WRITE_PATH / "sparksql_output"

    # Ensure the directories are clean before starting
    if WRITE_PATH.exists():
        shutil.rmtree(WRITE_PATH)

    # Create the necessary directory structure
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    # Create input files for the test
    (input_dir / "subdir").mkdir()

    query1_presto = """
    CREATE TABLE temp_table
    WITH (
        format = 'ORC',
        partitioned_by = ARRAY['ds']
    ) AS
    WITH ranked_events AS (
        SELECT
            event_id,
            user_id,
            event_type,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_timestamp DESC) AS rn
        FROM
            events
    )
    SELECT
        event_id,
        user_id,
        event_type
    FROM
        ranked_events
    WHERE
        rn = 1
    """
    with open(input_dir / "query1.sql", "w") as f:
        f.write(query1_presto)

    query2_presto = "SELECT date_add('day', 1, CURRENT_DATE) FROM users"
    with open(input_dir / "subdir" / "query2.sql", "w") as f:
        f.write(query2_presto)

    # Initialize the conversion class
    presto_to_spark = SqlConversion(from_dialect="presto", to_dialect="spark")

    # Run the directory conversion, saving to the static WRITE_PATH
    presto_to_spark.convert_directory(input_path=str(input_dir), output_path=str(output_dir))

    # Define the expected Spark SQL after conversion, including the extra spaces.
    expected_spark_1 = """
    CREATE TABLE temp_table
    USING ORC
    PARTITIONED BY ( ds ) AS
    WITH ranked_events AS (
        SELECT
            event_id,
            user_id,
            event_type,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_timestamp DESC) AS rn
        FROM
            events
    )
    SELECT
        event_id,
        user_id,
        event_type
    FROM
        ranked_events
    WHERE
        rn = 1
    """
    expected_spark_2 = "SELECT DATE_ADD(CURRENT_DATE, 1) FROM users"

    # Verify that the output files exist in the static path
    output_file_1 = output_dir / "query1.sql"
    output_file_2 = output_dir / "subdir" / "query2.sql"
    assert output_file_1.exists()
    assert output_file_2.exists()

    # Verify the content of the converted files
    with open(output_file_1, "r") as f:
        converted_content_1 = f.read()
    assert _normalize_sql_whitespace(converted_content_1) == _normalize_sql_whitespace(
        expected_spark_1
    )

    with open(output_file_2, "r") as f:
        converted_content_2 = f.read()
    assert _normalize_sql_whitespace(converted_content_2) == _normalize_sql_whitespace(
        expected_spark_2
    )

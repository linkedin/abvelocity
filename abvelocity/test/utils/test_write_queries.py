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
from pathlib import Path

import pytest

from abvelocity.utils.write_queries import write_queries

# Use a static base path for writing test results.
WRITE_PATH = Path(__file__).parents[5].joinpath("docs/static/test-results/write_queries/")


@pytest.fixture
def sample_queries_dict():
    """A dictionary of queries with unsorted keys for testing."""
    return {
        "query_z_last": "SELECT * FROM z_table;",
        "query_a_first": "SELECT * FROM a_table;",
        "query_b_second": "SELECT * FROM b_table;",
    }


@pytest.fixture
def complex_key_dict():
    """A dictionary with keys that require sanitization."""
    return {
        "My-First_Query! (with spaces)": "SELECT column_1 FROM my_view WHERE condition = TRUE;",
    }


def test_write_queries_with_prefix(sample_queries_dict, request):
    """Tests the default behavior with numerical prefixes and sorted keys using a unique subdirectory."""
    test_dir = WRITE_PATH / request.node.name
    written_files = write_queries(queries_dict=sample_queries_dict, output_dir=str(test_dir))

    # Assert that the directory and files were created
    assert test_dir.exists()
    assert len(list(test_dir.iterdir())) == 3
    assert len(written_files) == 3

    # Assert filenames are correct and in sorted order of the keys
    expected_filenames = ["01_query_a_first.sql", "02_query_b_second.sql", "03_query_z_last.sql"]
    assert sorted([f.name for f in test_dir.iterdir()]) == expected_filenames

    # Assert contents of the first file are correct
    with open(test_dir / "01_query_a_first.sql", "r") as f:
        assert f.read() == sample_queries_dict["query_a_first"]


def test_write_queries_without_prefix(sample_queries_dict, request):
    """Tests writing files without a numerical prefix using a unique subdirectory."""
    test_dir = WRITE_PATH / request.node.name
    written_files = write_queries(
        queries_dict=sample_queries_dict, output_dir=str(test_dir), use_prefix=False
    )

    # Assert that the directory and files were created
    assert test_dir.exists()
    assert len(list(test_dir.iterdir())) == 3
    assert len(written_files) == 3

    # Assert filenames are correct and sorted by sanitized name
    expected_filenames = ["query_a_first.sql", "query_b_second.sql", "query_z_last.sql"]
    assert sorted([f.name for f in test_dir.iterdir()]) == expected_filenames

    # Assert contents are correct
    with open(test_dir / "query_a_first.sql", "r") as f:
        assert f.read() == sample_queries_dict["query_a_first"]


def test_write_queries_sanitizes_filenames(complex_key_dict, request):
    """Tests that keys with special characters are sanitized correctly using a unique subdirectory."""
    test_dir = WRITE_PATH / request.node.name
    written_files = write_queries(queries_dict=complex_key_dict, output_dir=str(test_dir))

    assert test_dir.exists()
    assert len(written_files) == 1
    # Check if the filename matches the expected sanitized format
    assert written_files[0].endswith("01_my-first_query_with_spaces.sql")


def test_write_queries_warns_on_non_empty_dir(sample_queries_dict, capfd, request):
    """Tests that a warning is issued when the output directory is not empty and still writes files."""
    test_dir = WRITE_PATH / request.node.name
    # Create the unique subdirectory and a dummy file inside it before the function call
    os.makedirs(test_dir, exist_ok=True)
    (test_dir / "existing_file.txt").write_text("This file should not be overwritten.")

    write_queries(queries_dict=sample_queries_dict, output_dir=str(test_dir))

    # Capture the output from stdout and stderr
    captured = capfd.readouterr()

    # Check if the warning message is present in the output
    assert "WARNING: The directory" in captured.out
    assert "already exists and is not empty" in captured.out

    # Check that both the new files and the original file exist
    assert (test_dir / "01_query_a_first.sql").exists()
    assert (test_dir / "existing_file.txt").exists()


def test_write_queries_raises_error_on_file_conflict(sample_queries_dict, request):
    """Tests that an IOError is raised when a file exists with the same name as the output directory."""
    test_dir = WRITE_PATH / request.node.name
    # Create a file with the same name as the intended directory
    test_dir.touch()

    # Assert that calling the function raises an IOError
    with pytest.raises(IOError) as excinfo:
        write_queries(queries_dict=sample_queries_dict, output_dir=str(test_dir))

    # Check if the error message is clear and relevant
    assert "Cannot create directory" in str(excinfo.value)
    assert "file with that name already exists" in str(excinfo.value)


def test_write_queries_handles_empty_dict(capfd, request):
    """Tests that the function handles an empty input dictionary gracefully."""
    test_dir = WRITE_PATH / request.node.name

    written_files = write_queries(queries_dict={}, output_dir=str(test_dir))

    # Assert that no files were written and the directory was not created since it's empty
    assert written_files == []
    assert not test_dir.exists()

    # Check for the message indicating an empty dictionary
    captured = capfd.readouterr()
    assert "The provided queries dictionary is empty" in captured.out

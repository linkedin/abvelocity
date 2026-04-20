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
from abvelocity.core.journey.param.table_query import TableQuery
from abvelocity.core.utils.write_queries import write_queries

# Static path so test results can be inspected after a run.
WRITE_PATH = Path(__file__).parents[4].joinpath("docs/static/test-results/write_queries/")


@pytest.fixture
def sample_queries_dict():
    return {
        "query_z_first": TableQuery(table_name="t_z", main_query="SELECT * FROM z_table;"),
        "query_a_second": TableQuery(table_name="t_a", main_query="SELECT * FROM a_table;"),
        "query_b_last": TableQuery(table_name="t_b", main_query="SELECT * FROM b_table;"),
    }


@pytest.fixture
def complex_key_dict():
    return {
        "My-First_Query! (with spaces)": TableQuery(
            table_name=None,
            main_query="SELECT column_1 FROM my_view WHERE condition = TRUE;",
        ),
    }


def test_write_queries_with_prefix(sample_queries_dict, request):
    test_dir = WRITE_PATH / request.node.name
    written_files = write_queries(queries_dict=sample_queries_dict, output_dir=str(test_dir))

    assert test_dir.exists()
    assert len([f for f in test_dir.iterdir() if f.suffix == ".sql"]) == 3
    assert len(written_files) == 3

    expected_filenames = ["01_query_z_first.sql", "02_query_a_second.sql", "03_query_b_last.sql"]
    assert sorted([f.name for f in test_dir.iterdir() if f.suffix == ".sql"]) == sorted(expected_filenames)

    with open(test_dir / "01_query_z_first.sql", "r") as f:
        assert f.read() == sample_queries_dict["query_z_first"].main_query


def test_write_queries_without_prefix(sample_queries_dict, request):
    test_dir = WRITE_PATH / request.node.name
    written_files = write_queries(queries_dict=sample_queries_dict, output_dir=str(test_dir), use_prefix=False)

    assert test_dir.exists()
    assert len([f for f in test_dir.iterdir() if f.suffix == ".sql"]) == 3
    assert len(written_files) == 3

    expected_filenames = ["query_z_first.sql", "query_a_second.sql", "query_b_last.sql"]
    assert sorted([f.name for f in test_dir.iterdir() if f.suffix == ".sql"]) == sorted(expected_filenames)

    with open(test_dir / "query_z_first.sql", "r") as f:
        assert f.read() == sample_queries_dict["query_z_first"].main_query


def test_write_queries_sanitizes_filenames(complex_key_dict, request):
    test_dir = WRITE_PATH / request.node.name
    written_files = write_queries(queries_dict=complex_key_dict, output_dir=str(test_dir))

    assert test_dir.exists()
    assert len(written_files) == 1
    assert written_files[0].endswith("01_my-first_query_with_spaces.sql")


def test_write_queries_warns_on_non_empty_dir(sample_queries_dict, capfd, request):
    test_dir = WRITE_PATH / request.node.name
    os.makedirs(test_dir, exist_ok=True)
    (test_dir / "existing_file.txt").write_text("This file should not be overwritten.")

    write_queries(queries_dict=sample_queries_dict, output_dir=str(test_dir))

    captured = capfd.readouterr()
    assert "WARNING: The directory" in captured.out
    assert "already exists and is not empty" in captured.out

    assert (test_dir / "01_query_z_first.sql").exists()
    assert (test_dir / "existing_file.txt").exists()


def test_write_queries_raises_error_on_file_conflict(sample_queries_dict, tmp_path):
    # Uses tmp_path (not WRITE_PATH) — this is a pure error-path test, nothing to inspect.
    conflict_path = tmp_path / "output"
    conflict_path.touch()

    with pytest.raises(IOError) as excinfo:
        write_queries(queries_dict=sample_queries_dict, output_dir=str(conflict_path))

    assert "Cannot create directory" in str(excinfo.value)
    assert "file with that name already exists" in str(excinfo.value)


def test_write_queries_handles_empty_dict(capfd, request):
    test_dir = WRITE_PATH / request.node.name

    written_files = write_queries(queries_dict={}, output_dir=str(test_dir))

    assert written_files == []
    assert not test_dir.exists()

    captured = capfd.readouterr()
    assert "The provided queries dictionary is empty" in captured.out

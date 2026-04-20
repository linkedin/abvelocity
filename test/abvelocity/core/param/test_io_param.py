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

import re

from abvelocity.core.param.io_param import IOParam


def test_io_param_initialization():
    io_param = IOParam(
        cursor=None,
        create_table_prefix="tmp_",
        create_table_suffix="_final",
        file_name_prefix="run_",
        file_name_suffix="_v2",
        print_to_html=None,
        save_path="/tmp/output",
    )

    assert io_param.create_table_prefix == "tmp_"
    assert io_param.create_table_suffix == "_final"
    assert io_param.file_name_prefix == "run_"
    assert io_param.file_name_suffix == "_v2"
    assert io_param.save_path == "/tmp/output"
    assert io_param.cursor is None
    assert io_param.print_to_html is None


# --- Table Name Generation Tests ---


def test_gen_table_name_basic():
    io_param = IOParam(create_table_prefix="dev", create_table_suffix="v2")
    assert io_param.gen_table_name("sessions") == "dev_sessions_v2"
    assert io_param.gen_table_name("user_events") == "dev_user_events_v2"


def test_gen_table_name_no_prefix_no_suffix():
    io_param = IOParam()
    assert io_param.gen_table_name("clean_table") == "clean_table"


def test_gen_table_name_only_prefix():
    io_param = IOParam(create_table_prefix="prod")
    assert io_param.gen_table_name("metrics") == "prod_metrics"


def test_gen_table_name_only_suffix():
    io_param = IOParam(create_table_suffix="bak")
    assert io_param.gen_table_name("metrics") == "metrics_bak"


def test_gen_table_name_with_date_default_format_no_patch():
    io_param = IOParam(create_table_prefix="tmp", create_table_suffix="final")
    result = io_param.gen_table_name("mea_results", add_date_suffix=True)

    # Assert that the base name is present and followed by an underscore
    assert result.startswith("tmp_mea_results_final_")

    # Assert that the date part is non-empty and follows the expected format (YYYYMMDD)
    date_part = result.split("_")[-1]
    assert re.fullmatch(r"\d{8}", date_part) is not None


def test_gen_table_name_date_at_very_end_even_with_empty_suffix_no_patch():
    io_param = IOParam(create_table_prefix="x", create_table_suffix="")
    result = io_param.gen_table_name("test", add_date_suffix=True)

    # Assert that the base name is present and followed by an underscore
    assert result.startswith("x_test_")

    # Assert that the date part is non-empty and follows the expected format (YYYYMMDD)
    date_part = result.split("_")[-1]
    assert re.fullmatch(r"\d{8}", date_part) is not None


def test_gen_table_name_custom_date_format_no_patch():
    io_param = IOParam(create_table_prefix="run", create_table_suffix="done")
    date_format = "%Y%m%d_%H%M%S"
    result = io_param.gen_table_name(
        "sessions",
        add_date_suffix=True,
        date_format=date_format,
    )

    # Assert that the base name is present and followed by an underscore
    assert result.startswith("run_sessions_done_")

    # Expected date format is YYYYMMDD_HHMMSS (8 digits, underscore, 6 digits).
    # Since the parts are joined by underscores, we split and check the last two segments.
    parts = result.split("_")
    date_part_1 = parts[-2]
    date_part_2 = parts[-1]

    assert re.fullmatch(r"\d{8}", date_part_1) is not None
    assert re.fullmatch(r"\d{6}", date_part_2) is not None


def test_gen_table_name_safety_weird_characters_no_patch():
    # Note: For table names, the implementation does NOT sanitize prefix/suffix
    # (to preserve DB/schema names). Only the core name and date are sanitized.
    io_param = IOParam(create_table_prefix="dev", create_table_suffix="bak")

    # The date format is "%Y-%m-%d %H:%M", which sanitizes to YYYY_MM_DD_HH_MM
    result = io_param.gen_table_name(
        "my/table:name!",
        add_date_suffix=True,
        date_format="%Y-%m-%d %H:%M",
    )

    # Assert that the sanitized base name is correct
    # Core name "my/table:name!" is NOT sanitized in table names per implementation
    assert "dev" in result
    assert "my/table:name!" in result
    assert "bak" in result

    # The date portion should be sanitized
    parts = result.split("_")
    # Last 5 parts should be the sanitized date: YYYY, MM, DD, HH, MM
    assert len(parts) >= 5


def test_gen_table_name_empty_core():
    # With empty core, only prefix and suffix are joined
    io_param = IOParam(create_table_prefix="a", create_table_suffix="b")
    assert io_param.gen_table_name("") == "a_b"


# --- File Name Generation Tests ---


def test_gen_file_name_basic():
    io_param = IOParam(file_name_prefix="temp", file_name_suffix="report")
    assert io_param.gen_file_name("results") == "temp_results_report"
    assert io_param.gen_file_name("metrics_summary") == "temp_metrics_summary_report"


def test_gen_file_name_no_prefix_no_suffix():
    io_param = IOParam()
    assert io_param.gen_file_name("dashboard") == "dashboard"


def test_gen_file_name_only_prefix():
    io_param = IOParam(file_name_prefix="export")
    assert io_param.gen_file_name("user_data") == "export_user_data"


def test_gen_file_name_only_suffix():
    io_param = IOParam(file_name_suffix="html")
    assert io_param.gen_file_name("landing_page") == "landing_page_html"


def test_gen_file_name_with_date_default_format_no_patch():
    io_param = IOParam(file_name_prefix="daily", file_name_suffix="backup")
    result = io_param.gen_file_name("data", add_date_suffix=True)

    # Assert that the base name is present and followed by an underscore
    assert result.startswith("daily_data_backup_")

    # Assert that the date part is non-empty and follows the expected format (YYYYMMDD)
    date_part = result.split("_")[-1]
    assert re.fullmatch(r"\d{8}", date_part) is not None


def test_gen_file_name_safety_weird_characters():
    # File names sanitize all components including prefix and suffix
    # weird: -> weird, report-v1 -> report_v1, /stuff -> stuff
    io_param = IOParam(file_name_prefix="weird:", file_name_suffix="/stuff")
    assert io_param.gen_file_name("report-v1") == "weird_report_v1_stuff"

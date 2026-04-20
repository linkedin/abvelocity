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

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Callable, Optional

# Assuming this import path is correct for your environment
from abvelocity.core.get_data.cursor import Cursor


@dataclass
class IOParam:
    """Dataclass to hold input/output parameters."""

    cursor: Optional[Cursor] = None
    """A cursor for interacting with databases and executing queries."""

    create_table_prefix: str = ""
    """Prefix added to all table names that are created during the run."""

    create_table_suffix: str = ""
    """Suffix added to all table names that are created during the run."""

    file_name_prefix: str = ""
    """Prefix added to all generated output filenames (e.g. reports, CSVs)."""

    file_name_suffix: str = ""
    """Suffix appended to all generated output filenames, before the extension."""

    print_to_html: Optional[Callable] = None
    """Optional callable used to write/publish HTML reports."""

    save_path: str = "./"
    """Directory where all output files (reports, CSVs, etc.) will be saved."""

    def _sanitize_date_string(self, raw_string: str) -> str:
        """
        Cleans a raw string (date, table name core, prefix, or suffix) to be
        safe for filenames/table names.

        The resulting string only contains alphanumeric characters and underscores.

        1. Replace *all* illegal characters (like '-', ':', '/') with '_'.
        2. Collapse multiple adjacent underscores and strip leading/trailing ones.
        """

        # 1. Replace non-alphanumeric and non-underscore with underscore
        safe_string = re.sub(r"[^a-zA-Z0-9_]", "_", raw_string)

        # 2. Remove leading/trailing underscores and collapse multiples
        safe_string = re.sub(r"_+", "_", safe_string.strip("_"))
        return safe_string

    def gen_table_name(
        self,
        table_name_core: str,
        add_date_suffix: bool = False,
        date_format: str = "%Y%m%d",
    ) -> str:
        """
        Generate a full table name with optional date suffix at the very end.
        The name is constructed as: (prefix)_(core)_(suffix)_(date).
        All parts are sanitized and empty parts are skipped to avoid double underscores.
        """
        parts = []

        # 1. Add Prefix
        # Note that we do not sanitize the prefix here, as it is expected to be
        # already in a safe format for table names.
        # Cleaning can be problematic if prefix includes DB/schema names eg include a period.
        if self.create_table_prefix:
            parts.append(self.create_table_prefix)

        # 2. Add Core Name (Fix: Only append if not empty after sanitization)
        if table_name_core:
            parts.append(table_name_core)

        # 3. Add Suffix
        if self.create_table_suffix:
            parts.append(self.create_table_suffix)

        # Join the non-empty parts with a single underscore.
        final_name = "_".join(parts)

        # 4. Add date at the very end, if requested
        if add_date_suffix:
            raw_date = datetime.datetime.now().strftime(date_format)
            safe_date = self._sanitize_date_string(raw_date)
            if safe_date:
                # Append date. Sanitization in _sanitize_date_string prevents
                # issues like a final underscore on final_name followed by one on safe_date.
                if final_name:
                    final_name = f"{final_name}_{safe_date}"
                else:
                    # Handle case where prefix/core/suffix were all empty
                    final_name = safe_date

        return final_name

    def gen_file_name(
        self,
        file_name_core: str,
        add_date_suffix: bool = False,
        date_format: str = "%Y%m%d",
    ) -> str:
        """
        Generate a full file name (without extension) with optional date suffix.
        The name is constructed as: (prefix)_(core)_(suffix)_(date).
        All parts are sanitized and empty parts are skipped to avoid double underscores.
        """
        parts = []

        # 1. Add Prefix
        cleaned_prefix = self._sanitize_date_string(self.file_name_prefix)
        if cleaned_prefix:
            parts.append(cleaned_prefix)

        # 2. Add Core Name (Fix: Only append if not empty after sanitization)
        cleaned_core = self._sanitize_date_string(file_name_core)
        if cleaned_core:
            parts.append(cleaned_core)

        # 3. Add Suffix
        cleaned_suffix = self._sanitize_date_string(self.file_name_suffix)
        if cleaned_suffix:
            parts.append(cleaned_suffix)

        # Join the non-empty parts with a single underscore.
        final_name = "_".join(parts)

        # 4. Add date at the very end, if requested
        if add_date_suffix:
            raw_date = datetime.datetime.now().strftime(date_format)
            safe_date = self._sanitize_date_string(raw_date)
            if safe_date:
                # Append date. Sanitization in _sanitize_date_string prevents
                # issues like a final underscore on final_name followed by one on safe_date.
                if final_name:
                    final_name = f"{final_name}_{safe_date}"
                else:
                    # Handle case where prefix/core/suffix were all empty
                    final_name = safe_date

        return final_name

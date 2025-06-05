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
"""Module introduces a function to create a where clause for subsampling data."""

from typing import Optional


def subsample_condition(
    unit_col: str, mods: Optional[list[int]] = None, percent: Optional[int] = None
) -> Optional[str]:
    """Generates a SQL WHERE Clause based on `unit_col` which is assumed to
    be an int / bigint column.
    This is helpful to be able to subsample predictably across datasets.

    Args:
        unit_col: The column of data we are going to subsample based off.
        mods: A list of integers between 0 to 99 which to subsample based off.
            For example if

                - `mods = [1, 3]`
                - then only units in `unit_col` with remainders of 1 and 3
                when dividing by 100 will be accepted.

            If None, then `percent` will be used to construct mods.
        percent: An integer from 0 to 100 and determines the number of mods.
            For example if `percent = 3` then mods will be `[0, 1, 2]`.

    Returns:
        where_clause: The WHERE Clause for SQL.
            Assuming `unit_col = "id"`, a string of the form:

                - MOD(id, 100) IN (1, 2, 13)

            If inputs are None, then it returns None.

    Raises:
        None.
    """
    # If both mods and percent are None,
    # then None is returned.
    if mods is None and percent is None:
        return None

    if percent is not None:
        # Since mods start from zero, `range` works (as it also starts form zero).
        mods = range(int(percent))

    mods_string = """(""" + """,""".join([str(mod) for mod in mods]) + """)"""
    return f"""MOD({unit_col}, 100) IN {mods_string}"""

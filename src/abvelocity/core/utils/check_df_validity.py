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

"""A function to check validity of a dataframe according to some specs."""

from typing import Optional

import pandas as pd


def check_df_validity(
    df: pd.DataFrame,
    needed_cols: list[str],
    err_trigger_source: str = None,
    err_message: Optional[str] = None,
) -> None:
    """Checks validity of data according to certain criteria.
    # TODO: Reza to add more needed validations.

    Args:
    df: A pandas dataframe to be validated.
    needed_cols: The list of columns needed for the task e.g. columns needed for a join.
    err_trigger_source: A function / class / method / module name for the error source.

    Returns:
        None.

    Raises:
        ValueError if any of these happen:

            - needed columns (`needed_cols`) are not found in df.

    """
    cols = df.columns

    if not set(needed_cols).issubset(set(cols)):
        err_message_complete = ""
        if err_trigger_source is not None:
            err_message_complete += f"\n error triggered in `{err_trigger_source}`\n"
        if err_message is not None:
            err_message_complete += f"\n {err_message} \n"

        err_message_complete += f"\n {needed_cols} is not subset of these columns: {cols}.\n"

        raise ValueError(err_message_complete)

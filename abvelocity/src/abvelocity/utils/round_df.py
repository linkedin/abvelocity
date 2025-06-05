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

from typing import Optional

import numpy as np
import pandas as pd


def round_float_cols(df: pd.DataFrame, rounding_digits: int = 4, cols: Optional[list[str]] = None):
    """Round the float columns of the dataframe based on `rounding_digits`.

    Args:
        df: Dataframe to round the columns of.
        rounding_digits: Number of digits to round to.
            Default is 4.
        cols: Columns to round.
            Default is None, which means all columns.

    Returns:
        None

    Alters:
        df: The dataframe passed in.

    """
    if cols is None:
        cols = list(df.columns)

    for col in cols:
        if col in df.columns and df[col].dtype == "float64":
            df[col] = df[col].apply(lambda x: round(x, rounding_digits))

    return None


def round_tuple_cols(df: pd.DataFrame, rounding_digits: int = 4, cols: Optional[list[str]] = None):
    """Round the columns that are tuples in the dataframe.
        In this case, we do two checks to pick the revelant columns:

            - If the column is `CI_COL` or `CI_PERCENT_COL`
            - the column consist of numpy array as elements.

    Args:
        df: Dataframe to round the columns of.

    Returns:
        None

    Alters:
        df: The dataframe passed in.

    """
    if cols is None:
        cols = list(df.columns)

    for col in cols:
        if col in df.columns:
            # Check if the column is a tuple of floats.
            # This could be either a numpy array or tuple of floats.
            # We include both possibilities.
            if (
                isinstance(df[col].values[0], np.ndarray)
                or isinstance(df[col].values[0], tuple)
                and isinstance(df[col].values[0][0], float)
            ):
                df[col] = df[col].apply(lambda x: tuple([round(y, rounding_digits) for y in x]))

    return None


def round_df(
    df,
    rounding_digits: int = 4,
    float_cols: Optional[list[str]] = None,
    tuple_cols: Optional[list[str]] = None,
):
    """Round the float columns and tuple columns of the dataframe.

    Args:
        df: Dataframe to round the columns of.
        rounding_digits: Number of digits to round to.
            Default is 4.
        float_cols: Columns to round that are floats.
            Default is None, which means all columns.
        tuple_cols: Columns to round that are tuples.
            Default is None, which means all columns.

    Returns:
        None

    Alters:
        df: The dataframe passed in.

    """
    round_float_cols(df=df, rounding_digits=rounding_digits, cols=float_cols)
    round_tuple_cols(df=df, rounding_digits=rounding_digits, cols=tuple_cols)

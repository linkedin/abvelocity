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

"""Helper function to add random dates to simulated dataframes."""

import numpy as np
import pandas as pd


def add_random_dates_to_df(
    df,
    start_date: str,
    end_date: str,
    date_unit: str = "D",
    date_col: str = "date",
    seed: int = None,
):
    """
    Adds a column with random dates to a dataframe.

    This function augments simulated experiment data with random dates,
    enabling timeseries analysis and testing of dimension-aware queries.

    Args:
        df: Input dataframe to augment with dates
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        date_unit: Pandas frequency string for date granularity.
                   Common values: 'D' (daily), 'H' (hourly), 'W' (weekly),
                   'W-MON' (weekly starting Monday), 'M' (monthly)
                   See pandas date_range freq parameter for full list
        date_col: Name for the date column (default: 'date')
        seed: Random seed for reproducibility

    Returns:
        DataFrame with added date column containing random dates from the range

    Example:
        >>> from abvelocity.core.sim.examples import simulate_data_uni1
        >>> from abvelocity.core.sim.add_random_dates_to_df import add_random_dates_to_df
        >>> sim = simulate_data_uni1()
        >>> df_with_dates = add_random_dates_to_df(
        ...     sim.expt_metric_df,
        ...     start_date='2024-01-01',
        ...     end_date='2024-01-31',
        ...     date_unit='D'
        ... )
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Generate all possible dates in the range
    date_range = pd.date_range(start=start_date, end=end_date, freq=date_unit)

    # Randomly assign dates to each row
    random_dates = np.random.choice(date_range, size=len(df))

    # Create a copy to avoid modifying original
    df_with_dates = df.copy()

    # Add date column (convert to string format)
    df_with_dates[date_col] = pd.to_datetime(random_dates).strftime("%Y-%m-%d")

    return df_with_dates

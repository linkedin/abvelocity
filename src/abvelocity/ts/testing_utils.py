# BSD 2-CLAUSE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
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
"""Shared synthetic time-series data generators for abvelocity tests.

Adapted from ``blah.greykite.common.testing_utils`` (BSD 2-Clause,
same author). All functions are standalone numpy/pandas — no greykite
dependency required.
"""

import numpy as np
import pandas as pd
from abvelocity.ts.constants import TIME_COL


def make_daily_series(
    n_days: int,
    col: str,
    base: float = 1000.0,
    trend: float = 0.5,
    weekly_amp: float = 25.0,
    annual_amp: float = 80.0,
    noise: float = 25.0,
    start: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic daily series: linear trend + weekly + annual seasonality + Gaussian noise.

    Signal formula::

        base + trend * t
            + annual_amp * sin(2π t / 365 - π/2)
            + weekly_amp * sin(2π t / 7)
            + Normal(0, noise)

    Values are clipped to ≥ 0.

    Adapted from ``blah.greykite.common.testing_utils.generate_df_for_tests``
    (BSD 2-Clause, original author: Reza Hosseini).

    Args:
        n_days: Number of daily rows to generate.
        col: Name of the value column in the returned DataFrame.
        base: Constant baseline level.
        trend: Linear slope added per day.
        weekly_amp: Amplitude of the weekly sine component (period = 7 days).
        annual_amp: Amplitude of the annual sine component (period = 365 days).
        noise: Standard deviation of Gaussian noise.
        start: ISO date string for the first timestamp.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``TIME_COL`` (``"ts"``) and ``col``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dates = pd.date_range(start, periods=n_days, freq="D")
    signal = base + trend * t + annual_amp * np.sin(2 * np.pi * t / 365 - np.pi / 2) + weekly_amp * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, n_days)
    return pd.DataFrame({TIME_COL: dates, col: signal.clip(0)})

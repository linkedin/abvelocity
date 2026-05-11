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
"""Shared anomaly injection utilities for abvelocity tests.

Adapted from ``blah.greykite.common.testing_utils_anomalies``
(BSD 2-Clause, same authors). All functions are standalone numpy/pandas —
no greykite dependency required.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_anomaly_blocks(
    timeseries_length: int,
    block_number: int,
    mean_block_size: float = 5,
    seed: int = 42,
) -> Dict:
    """Return randomly placed contiguous index blocks for anomaly injection.

    Adapted from
    ``blah.greykite.common.testing_utils_anomalies.generate_anomaly_blocks``.

    Args:
        timeseries_length: Length of the target time series.
        block_number: Initial number of anomaly blocks requested. Adjacent
            blocks that overlap or touch are merged, so the final count may
            be lower.
        mean_block_size: Mean block size (drawn from a Poisson distribution).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            ``"anomaly_block_list"`` — list of lists of row indices, one per block.
            ``"block_number"`` — actual number of blocks after merging.
            ``"block_sizes"`` — list of block lengths.
    """
    rng = np.random.default_rng(seed)
    anomaly_starts = np.sort(rng.choice(timeseries_length - 1, block_number, replace=False))
    block_lengths = rng.poisson(lam=mean_block_size, size=block_number)

    raw_blocks = [list(range(start, min(timeseries_length, start + length + 1))) for start, length in zip(anomaly_starts, block_lengths)]

    # Merge overlapping / adjacent blocks.
    merged: List[List[int]] = [raw_blocks[0]]
    for block in raw_blocks[1:]:
        if block[0] <= merged[-1][-1] + 1:
            merged[-1] = sorted(set(merged[-1] + block))
        else:
            merged.append(block)

    return {
        "anomaly_block_list": merged,
        "block_number": len(merged),
        "block_sizes": [len(b) for b in merged],
    }


def contaminate_df_with_anomalies(
    df: pd.DataFrame,
    anomaly_block_list: List[List[int]],
    delta_range_lower: float,
    delta_range_upper: float,
    value_col: str = "y",
    min_admissible_value: Optional[float] = None,
    max_admissible_value: Optional[float] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject multiplicative anomalies into a DataFrame column.

    For each anomaly block the value is multiplied by ``(1 ± delta)`` where
    ``delta`` is drawn uniformly from ``[delta_range_lower, delta_range_upper]``
    and the sign is chosen randomly per block.

    Adapted from
    ``blah.greykite.common.testing_utils_anomalies.contaminate_df_with_anomalies``.

    Args:
        df: DataFrame containing the target column.
        anomaly_block_list: List of row-index blocks returned by
            :func:`generate_anomaly_blocks`.
        delta_range_lower: Lower bound of the multiplicative delta interval.
        delta_range_upper: Upper bound of the multiplicative delta interval.
        value_col: Name of the column to contaminate.
        min_admissible_value: Floor applied after contamination (``None`` = no floor).
        max_admissible_value: Ceiling applied after contamination (``None`` = no ceiling).
        seed: Random seed for reproducibility.

    Returns:
        Copy of ``df`` with two extra columns:
            ``"contaminated_<value_col>"`` — values with injected anomalies.
            ``"is_anomaly"`` — 1 for anomalous rows, 0 otherwise.
    """
    rng = np.random.default_rng(seed)
    values = np.array(df[value_col], dtype=float)
    is_anomaly = np.zeros(len(df), dtype=float)

    for block in anomaly_block_list:
        sign = 1 if rng.integers(0, 2) == 1 else -1
        for idx in block:
            delta = rng.uniform(delta_range_lower, delta_range_upper)
            values[idx] = values[idx] * (1 + sign * delta)
            if min_admissible_value is not None:
                values[idx] = max(min_admissible_value, values[idx])
            if max_admissible_value is not None:
                values[idx] = min(max_admissible_value, values[idx])
            is_anomaly[idx] = 1.0

    result = df.copy()
    result[f"contaminated_{value_col}"] = values
    result["is_anomaly"] = is_anomaly
    return result


def make_anomaly_df(
    dates: pd.DatetimeIndex,
    anomaly_block_list: List[List[int]],
    start_ts_col: str = "start_ts",
    end_ts_col: str = "end_ts",
) -> pd.DataFrame:
    """Convert row-index blocks into a period DataFrame of (start_ts, end_ts) pairs.

    This is the format expected by ``TSRunner`` / ``BackfillRunner`` for
    anomaly masking.

    Args:
        dates: Full timestamp index of the series.
        anomaly_block_list: Output of :func:`generate_anomaly_blocks`.
        start_ts_col: Column name for the period start timestamp.
        end_ts_col: Column name for the period end timestamp.

    Returns:
        DataFrame with one row per anomaly block and two timestamp columns.
    """
    rows: List[Dict] = []
    for block in anomaly_block_list:
        rows.append({start_ts_col: dates[block[0]], end_ts_col: dates[block[-1]]})
    return pd.DataFrame(rows)


def inject_spike(
    df: pd.DataFrame,
    value_col: str,
    start_idx: int,
    end_idx: int,
    magnitude: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add a flat additive spike over a row range and return the anomaly_df.

    Convenience wrapper for the common test pattern of injecting a single
    spike and building the corresponding ``anomaly_df``.

    Args:
        df: DataFrame containing ``TIME_COL`` and ``value_col``.
        value_col: Column to spike.
        start_idx: First row index of the spike (inclusive).
        end_idx: Last row index of the spike (exclusive).
        magnitude: Amount added to every row in the spike window.

    Returns:
        ``(spiked_df, anomaly_df)`` — both are new DataFrames; ``df`` is
        not mutated. ``anomaly_df`` has columns ``"start_ts"`` / ``"end_ts"``.
    """
    from abvelocity.ts.constants import TIME_COL  # local to avoid circular

    spiked = df.copy()
    spiked.iloc[start_idx:end_idx, spiked.columns.get_loc(value_col)] += magnitude
    anomaly_df = pd.DataFrame(
        {
            "start_ts": [df[TIME_COL].iloc[start_idx]],
            "end_ts": [df[TIME_COL].iloc[end_idx - 1]],
        }
    )
    return spiked, anomaly_df

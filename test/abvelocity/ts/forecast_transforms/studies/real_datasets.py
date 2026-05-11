# Original author: Reza Hosseini
"""Loaders for the greykite built-in datasets used by the studies.

Wraps :class:`abvelocity.ts.common.data_loader.DataLoader` and
shapes each dataset into the (``ts``, ``y``) frame our
:class:`~abvelocity.ts.forecast_runner.ForecastRunner` expects, plus any extra dim
columns (e.g. ``station`` for parking).

All loaders are deterministic — same call produces the same frame.
"""

from __future__ import annotations

from typing import List

import pandas as pd
from abvelocity.ts.common.data_loader import DataLoader

# A small fixed set of stations from the parking dataset chosen for
# coverage breadth.  Picked among the 30 to keep panels fast and visually
# clean while covering distinct usage patterns.
PARKING_STATIONS: tuple = (
    "BHMBCCMKT01",
    "BHMBCCPST01",
    "BHMBCCSNH01",
    "BHMBCCTHL01",
)


def load_bikesharing_daily(start: str = "2018-01-01", end: str = "2019-09-01") -> pd.DataFrame:
    """Daily bikesharing counts (sum of hourly) for a date window.

    Args:
        start: Inclusive lower-bound date.
        end: Exclusive upper-bound date.

    Returns:
        Frame with columns ``ts`` (daily), ``y`` (count summed across
        the day), ``tmin``, ``tmax`` (degrees F), ``pn`` (precipitation).
    """
    hourly_df = DataLoader().load_bikesharing()
    hourly_df["ts"] = pd.to_datetime(hourly_df["ts"]).dt.normalize()
    daily_df = (
        hourly_df.groupby(by="ts", as_index=False)
        .agg({"count": "sum", "tmin": "mean", "tmax": "mean", "pn": "mean"})
        .rename(columns={"count": "y"})
    )
    return daily_df.loc[(daily_df["ts"] >= start) & (daily_df["ts"] < end)].reset_index(drop=True)


def load_parking_panel(stations: tuple = PARKING_STATIONS) -> pd.DataFrame:
    """Daily parking occupancy panel for a selected set of stations.

    Half-hourly raw measurements are aggregated to daily mean
    occupancy ratio (so ``y`` is in [0, 1]).

    Args:
        stations: ``SystemCodeNumber`` IDs to include.

    Returns:
        Long-format frame with ``ts`` (daily), ``station`` (str), and
        ``y`` (mean daily occupancy ratio).
    """
    rows: List[pd.DataFrame] = []
    for station in stations:
        station_df = DataLoader().load_parking(system_code_number=station)
        station_df["LastUpdated"] = pd.to_datetime(station_df["LastUpdated"]).dt.normalize()
        daily_df = (
            station_df.groupby(by="LastUpdated", as_index=False)["OccupancyRatio"]
            .mean()
            .rename(columns={"LastUpdated": "ts", "OccupancyRatio": "y"})
        )
        daily_df["station"] = station
        rows.append(daily_df)
    return pd.concat(objs=rows, ignore_index=True)


def load_peyton_manning_window(start: str = "2010-01-01", end: str = "2016-01-01") -> pd.DataFrame:
    """Daily Peyton Manning Wikipedia views for a date window.

    Args:
        start: Inclusive lower-bound date.
        end: Exclusive upper-bound date.

    Returns:
        Frame with columns ``ts`` (daily) and ``y`` (log views).
    """
    pey_df = DataLoader().load_peyton_manning()
    pey_df["ts"] = pd.to_datetime(pey_df["ts"])
    return pey_df.loc[(pey_df["ts"] >= start) & (pey_df["ts"] < end)].reset_index(drop=True)

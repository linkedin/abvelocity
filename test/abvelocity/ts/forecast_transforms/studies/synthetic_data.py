# Original author: Reza Hosseini
"""Synthetic data generators for ``forecast_transforms`` studies.

Produces simple daily / panel time series with trend + weekly
seasonality + Gaussian noise.  Intended for end-to-end studies (greykite
fit → transform → plot), not for unit-test fixtures.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def daily_series(
    start: str = "2024-01-01",
    n_days: int = 365 * 2,
    base: float = 100.0,
    trend_per_day: float = 0.05,
    weekly_amp: float = 15.0,
    annual_amp: float = 20.0,
    noise_std: float = 3.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Single-metric daily series: trend + weekly + annual + noise.

    Args:
        start: First date (inclusive).
        n_days: Number of days.
        base: Constant baseline added to every observation.
        trend_per_day: Linear drift per day.
        weekly_amp: Amplitude of the day-of-week sinusoid.
        annual_amp: Amplitude of the day-of-year sinusoid.
        noise_std: Std-dev of the additive Gaussian noise.
        seed: RNG seed for reproducibility.

    Returns:
        Two-column DataFrame: ``ts`` (date) + ``y`` (float).
    """
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    t = np.arange(n_days)
    weekly = weekly_amp * np.sin(2 * np.pi * t / 7.0)
    annual = annual_amp * np.sin(2 * np.pi * t / 365.0)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_days)
    y = base + trend_per_day * t + weekly + annual + noise
    return pd.DataFrame({"ts": dates, "y": y})


def country_device_panel(
    start: str = "2024-01-01",
    n_days: int = 365,
    seed: int = 11,
) -> pd.DataFrame:
    """4-segment panel: (US, GB) × (android, iphone), daily.

    Each segment has its own baseline and weekly amplitude so aggregation
    isn't trivial.  Single-metric.

    Args:
        start: First date (inclusive).
        n_days: Number of days per segment.
        seed: RNG seed for reproducibility.

    Returns:
        Long-format DataFrame: ``ts`` + ``country`` + ``device`` + ``y``.
    """
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    segments = [
        ("US", "android", 200.0, 30.0),
        ("US", "iphone", 300.0, 40.0),
        ("GB", "android", 60.0, 8.0),
        ("GB", "iphone", 80.0, 10.0),
    ]
    rows: List[pd.DataFrame] = []
    for country, device, base, weekly_amp in segments:
        t = np.arange(n_days)
        weekly = weekly_amp * np.sin(2 * np.pi * t / 7.0)
        trend = 0.03 * t
        noise = rng.normal(loc=0.0, scale=base * 0.02, size=n_days)
        y = base + trend + weekly + noise
        rows.append(
            pd.DataFrame(
                {
                    "ts": dates,
                    "country": country,
                    "device": device,
                    "y": y,
                }
            )
        )
    return pd.concat(objs=rows, ignore_index=True)

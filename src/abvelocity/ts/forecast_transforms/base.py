# Original author: Reza Hosseini
"""Abstract base for post-forecast transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ForecastTransform(ABC):
    """Base class for post-forecast transforms.

    Operates on a forecast frame produced by :func:`TSFlow.run`,
    returning a new forecast frame with the same column shape (minus
    ``stage``, which is dropped at the door — see
    :doc:`architecture` for why).  Implementations are frozen
    dataclasses; state is fields only, instances are reusable and
    hashable.
    """

    @abstractmethod
    def apply(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Apply this transform; return a new frame.  Pure function."""
        ...

    @abstractmethod
    def str_name(self) -> str:
        """Snippet that appears in the derived ``metric_id``."""
        ...

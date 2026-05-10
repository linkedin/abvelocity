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
"""Backfill configuration dataclass."""

from dataclasses import dataclass
from typing import List, Optional

from abvelocity.ts.config.forecast_config import ForecastConfig
from mashumaro.mixins.json import DataClassJSONMixin

VALID_WINDOW_TYPES = {"expanding", "rolling"}


@dataclass
class BackfillConfig(DataClassJSONMixin):
    """Configuration for rolling backfill of historical forecasts.

    ``BackfillRunner`` uses this config to slide a training cutoff forward
    through a prepped DataFrame (actuals present for all rows), fitting the
    forecast algo at each cutoff and collecting ``horizon`` steps of forecasts.
    The result is a long-format DataFrame of past forecasts tagged by
    ``cutoff`` and ``horizon_step``, suitable for eval or backfilling a
    forecast store.

    Attributes:
        forecast_config: Forecast algorithm config used at every cutoff.
            ``forecast_config.forecast_horizon`` must be >= ``horizon``.
        initial_train_size: Minimum number of rows used for the first
            training window. The first cutoff index is ``initial_train_size``.
        horizon: Number of steps ahead to collect per cutoff. Must satisfy
            ``horizon <= forecast_config.forecast_horizon``.
        step: Number of rows to advance the cutoff between iterations.
            Equivalent to ``step_size`` in StatsForecast. Default 1.
        window_type: ``"expanding"`` (all history up to cutoff) or
            ``"rolling"`` (fixed window of ``window_size`` rows ending at
            cutoff). Default ``"expanding"``.
        window_size: Training window length in rows, used only when
            ``window_type="rolling"``. Must be provided for rolling windows.
        n_windows: Optional cap on the total number of cutoffs evaluated.
            When set, the ``n_windows`` most recent cutoffs are used.
            Equivalent to greykite's ``cv_max_splits`` /
            StatsForecast's ``n_windows``. ``None`` means use all cutoffs.
        cutoffs: Optional explicit list of cutoff dates (ISO ``"YYYY-MM-DD"``
            strings). When set, ``initial_train_size`` / ``step`` /
            ``n_windows`` / ``window_type`` / ``window_size`` are ignored
            for cutoff *generation* (window_type still controls the
            training-window shape per cutoff: ``"expanding"`` uses
            everything before the cutoff; ``"rolling"`` uses
            ``window_size`` rows before it). Each date must exist in the
            input DataFrame's ``time_col`` at runtime.
    """

    forecast_config: ForecastConfig
    """Forecast algorithm config; ``forecast_horizon`` must be >= ``horizon``."""

    initial_train_size: int
    """Minimum rows for the first training window."""

    horizon: int
    """Steps ahead to collect per cutoff."""

    step: int = 1
    """Rows to advance between cutoffs."""

    window_type: str = "expanding"
    """``"expanding"`` or ``"rolling"``."""

    window_size: Optional[int] = None
    """Training window size in rows (rolling only)."""

    n_windows: Optional[int] = None
    """Cap on number of cutoffs; uses the most recent ones. ``None`` = all."""

    cutoffs: Optional[List[str]] = None
    """Explicit list of cutoff dates (ISO ``"YYYY-MM-DD"``). When set, this
    overrides the algorithmic ``initial_train_size``/``step``/``n_windows``
    spec for cutoff generation. Each date must exist in the input
    DataFrame's ``time_col`` at runtime."""

    def __post_init__(self) -> None:
        if self.window_type not in VALID_WINDOW_TYPES:
            raise ValueError(f"window_type must be one of {VALID_WINDOW_TYPES!r}, got {self.window_type!r}.")
        if self.cutoffs is not None:
            if not isinstance(self.cutoffs, list) or len(self.cutoffs) == 0:
                raise ValueError(f"cutoffs must be a non-empty list of date strings, got {self.cutoffs!r}.")
            # Pandas parses many date formats; we just need fail-fast at config time.
            import pandas as pd
            for c in self.cutoffs:
                if not isinstance(c, str):
                    raise ValueError(f"cutoffs entries must be date strings, got {c!r} ({type(c).__name__}).")
                try:
                    pd.Timestamp(c)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"cutoffs entry {c!r} is not a parseable date "
                        f"(expected e.g. '2025-06-15'): {exc}"
                    ) from exc
        if self.initial_train_size < 1:
            raise ValueError(f"initial_train_size must be >= 1, got {self.initial_train_size}.")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}.")
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}.")
        if self.window_type == "rolling" and self.window_size is None:
            raise ValueError("window_size must be set when window_type='rolling'.")
        if self.window_size is not None and self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}.")
        if self.n_windows is not None and self.n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {self.n_windows}.")
        if self.forecast_config.forecast_horizon < self.horizon:
            raise ValueError(f"forecast_config.forecast_horizon ({self.forecast_config.forecast_horizon}) " f"must be >= horizon ({self.horizon}).")

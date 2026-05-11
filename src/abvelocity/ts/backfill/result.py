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
"""Backfill result dataclass."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from abvelocity.core.utils.serialization import DataFrameConfig
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class BackfillResult(DataClassJSONMixin):
    """Result produced by :class:`~abvelocity.ts.backfill.runner.BackfillRunner`.

    Contains every historical forecast generated across all cutoffs, in the
    same long-format schema as :class:`~abvelocity.ts.result.ts_result.TSResult`
    but with two additional columns: ``cutoff`` (last training timestamp) and
    ``horizon_step`` (integer 1 … h indicating which step ahead each row is).

    ``result_df`` can be passed directly to
    :func:`~abvelocity.ts.eval.compute_eval` for accuracy
    analysis, or stored as-is for backfilling a forecast table.

    Attributes:
        result_df: Long-format DataFrame with columns
            ``ts, metric, actual, forecast, forecast_lower, forecast_upper,
            cutoff, horizon_step, algo_ver, train_end_date``.
            One row per (cutoff × metric × horizon_step).
            ``actual`` is populated from the prepped input DataFrame.
        fit_info: Per-cutoff diagnostics keyed by cutoff timestamp string.
    """

    result_df: Optional[pd.DataFrame] = None
    """Long-format backfill DataFrame with ``cutoff`` and ``horizon_step`` columns."""

    fit_info: Optional[Dict[str, Any]] = None
    """Per-cutoff fit diagnostics keyed by cutoff timestamp string."""

    class Config(DataFrameConfig):
        pass

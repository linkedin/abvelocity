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
"""TSFlowResult: result dataclass for a single TSFlow run."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class TSFlowResult:
    """Result from a single :class:`~abvelocity.ts.flow.flow.TSFlow`
    run for one metric group.

    Attributes:
        result_df: Main algorithm result (forecast, anomaly detection, or
            backfill) in long format.  ``None`` when the algo produces no
            output (e.g. no valid cutoffs in backfill mode).
        eval_df: Optional eval metrics computed on ``result_df``.  Populated
            only when :attr:`~TSFlowConfig.eval_metrics` is non-empty and
            ``result_df`` is not ``None``.
        fit_info: Optional fit/backtest info returned by the algorithm
            (e.g. greykite backtest evaluation metrics, keyed by value column
            or ``(dim_vals, value_col)``).
    """

    result_df: Optional[pd.DataFrame] = None
    """Long-format algorithm result."""

    eval_df: Optional[pd.DataFrame] = None
    """Eval metrics DataFrame; ``None`` if eval was not requested or unavailable."""

    fit_info: Optional[Dict[str, Any]] = None
    """Algorithm fit info (e.g. backtest evaluation metrics)."""

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
"""Evaluation package — forecast + anomaly-detection metrics.

Primary API (class-based, parameterizable, pass to ``TSFlowConfig.eval``):

* :class:`ForecastEval` — forecast regression metrics (mae/rmse/mape/…)
* :class:`ADEval` — AD classification metrics (precision/recall/f1,
  plus soft_* with tolerance window)

Functional API (for direct use):

* :func:`compute_eval` / :func:`compute_ad_eval`
"""

from abvelocity.ts.eval.ad_eval import ADEval, compute_ad_eval
from abvelocity.ts.eval.forecast_eval import (
    ForecastEval,
    compute_coverage,
    compute_eval,
    compute_mape,
    compute_medape,
    compute_r2,
    compute_smape,
)
from abvelocity.ts.eval.forecast_eval_report import (
    DEFAULT_FIGURE_MAKERS,
    FigureMaker,
    FigureMakerContext,
    FigureSection,
    ForecastEvalReport,
    evaluate_forecasts_vs_actuals,
    make_per_horizon_error_curve,
    make_per_horizon_forecast_overlays,
    make_per_training_cutoff_forecast_overlays,
)
from abvelocity.ts.eval.live_eval import (
    fetch_forecasts_at_training_cutoffs,
    list_training_cutoffs,
    run_live_forecast_eval,
)

__all__ = [
    "ADEval",
    "DEFAULT_FIGURE_MAKERS",
    "FigureMaker",
    "FigureMakerContext",
    "FigureSection",
    "ForecastEval",
    "ForecastEvalReport",
    "compute_ad_eval",
    "compute_coverage",
    "compute_eval",
    "compute_mape",
    "compute_medape",
    "compute_r2",
    "compute_smape",
    "evaluate_forecasts_vs_actuals",
    "fetch_forecasts_at_training_cutoffs",
    "list_training_cutoffs",
    "make_per_horizon_error_curve",
    "make_per_horizon_forecast_overlays",
    "make_per_training_cutoff_forecast_overlays",
    "run_live_forecast_eval",
]

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
"""End-to-end model-selection tests using GreykiteForecastAlgo.

Mirrors :mod:`test_model_selection_end_to_end` but swaps the algo to
``"greykite"`` to validate that the same :class:`GridModelSelection` /
:class:`GroupedModelSelection` code paths drive a different algo with
its own (nested) ``algo_params`` shape — confirming the framework is
genuinely algo-agnostic.

Skipped automatically when ``blah.greykite`` is not importable.
Greykite fits are slow, so the test data is one year of daily values
and the candidate count is intentionally small (2-3 per stage).

Inspect after running:

    open docs/static/test-results/model-selection-greykite/grid/results.html
    open docs/static/test-results/model-selection-greykite/grouped/results.html
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Self-registers GreykiteForecastAlgo into ALGO_REGISTRY when greykite is importable.
import abvelocity.ts.algo.greykite_forecast_algo as greykite_forecast_algo  # noqa: F401
from abvelocity.ts.algo.base import ALGO_REGISTRY
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import TIME_COL
from abvelocity.ts.model_selection.eval_criteria import EvalCriteria
from abvelocity.ts.model_selection.grid import GridModelSelection
from abvelocity.ts.model_selection.grouped import GroupedModelSelection
from abvelocity.ts.model_selection.report import write_report
from abvelocity.ts.model_selection.space import ParamGroup, SearchSpace

GREYKITE_AVAILABLE = "greykite" in ALGO_REGISTRY

ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "docs" / "static" / "test-results" / "model-selection-greykite"
)


@pytest.fixture
def simulated_daily_df():
    """4 months of daily data with weekly cycle + noise.

    Kept small on purpose so silverkite fits stay light — these tests
    run inside a parallel pytest-xdist worker pool on Blah CI, and
    memory-heavy fits crash the worker (``[gw60] node down: Not
    properly terminated``). 120 rows is enough to exercise the
    framework's plumbing without stressing the fit.
    """
    rng = np.random.default_rng(2026)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    weekly = 8.0 * np.sin(2.0 * np.pi * np.arange(n) / 7.0)
    noise = rng.normal(0.0, 1.5, n)
    return pd.DataFrame({TIME_COL: dates, "y": 100.0 + weekly + noise})


@pytest.fixture
def template_backfill_config():
    """SILVERKITE_EMPTY template — no auto-seasonality / auto-changepoints /
    auto-holidays / auto-growth — so the fit is as lightweight as possible
    while still going through silverkite's real code path.

    ``SILVERKITE`` (the full template) auto-enables yearly + weekly
    Fourier, US holidays, and trend changepoints; on a parallel CI
    worker that's enough to OOM-crash the worker. ``SILVERKITE_EMPTY``
    skips all of that — the framework still exercises the
    GreykiteParamConverter, the nested algo_params merge, the predict
    + persist + eval pipeline.
    """
    fc = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("y",),
        freq="D",
        coverage=0.80,
        forecast_horizon=7,
        algo_name="greykite",
        algo_params={
            "model_template": "SILVERKITE_EMPTY",
            "model_components": {
                "growth": {"growth_term": "linear"},
                "seasonality": {
                    "yearly_seasonality": False,
                    "weekly_seasonality": 3,
                    "daily_seasonality": False,
                    "quarterly_seasonality": 0,
                },
                "events": {"holidays_to_model_separately": []},
                "changepoints": {"changepoints_dict": None},
                "autoregression": {"autoreg_dict": None},
                "custom": {"fit_algorithm_dict": {"fit_algorithm": "ridge"}},
                "uncertainty": {"uncertainty_dict": None},
            },
        },
    )
    return BackfillConfig(
        forecast_config=fc,
        initial_train_size=80,
        horizon=7,
        step=20,
        n_windows=2,
    )


def wipe(out_dir: Path) -> None:
    """Remove every file in ``out_dir`` so the test always reflects current code."""
    if out_dir.exists():
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    os.makedirs(out_dir, exist_ok=True)


@pytest.mark.skipif(not GREYKITE_AVAILABLE, reason="blah.greykite not installed")
def test_e2e_greykite_grid_writes_inspectable_artifacts(simulated_daily_df, template_backfill_config):
    """Grid sweep over greykite ``model_components`` configs."""
    out_dir = ARTIFACT_ROOT / "grid"
    wipe(out_dir)

    # Flat search-space form — the GreykiteParamConverter translates each
    # candidate into the nested ``model_components.custom.fit_algorithm_dict``
    # override that GreykiteForecastAlgo expects.
    space = SearchSpace.flat({"fit_algorithm": ["ridge", "linear"]})
    criteria = EvalCriteria(
        eval_metrics=("mape", "smape", "medae", "mae", "rmse"),
        primary_eval_metric="mape",
        primary_eval_reduction="mean",
    )
    selection = GridModelSelection(
        search_space=space,
        backfill_config=template_backfill_config,
        eval_criteria=criteria,
        param_converter=greykite_forecast_algo.GreykiteParamConverter(),
        output_dir=out_dir,
        verbose=False,
        n_jobs=1,  # explicit — never spin joblib inside an xdist test worker
    )
    result = selection.run(df=simulated_daily_df)
    write_report(result, criteria, title="ModelSelection — Greykite Grid (SILVERKITE_EMPTY, ridge vs linear)")

    candidates_log = pd.read_csv(out_dir / "model_candidates.csv")
    assert len(candidates_log) == 2
    assert (candidates_log["status"] == "ok").all()
    for path in candidates_log["predict_path"]:
        assert (out_dir / path).exists()

    results_csv = pd.read_csv(out_dir / "results.csv")
    assert len(results_csv) == 2
    assert results_csv["score"].is_monotonic_increasing
    assert "mape_mean" in results_csv.columns
    assert (out_dir / "results.html").exists()
    html = (out_dir / "results.html").read_text()
    assert "Greykite" in html

    # Per-candidate eval CSVs landed.
    eval_dir = out_dir / "evals"
    assert eval_dir.exists() and len(list(eval_dir.glob("*.csv"))) == 2

    # Audit trail uses the flat search-space form, not the nested algo_params.
    assert "fit_algorithm" in result.best_params
    assert result.best_params["fit_algorithm"] in {"ridge", "linear"}
    # results.csv breaks each search-space param out as its own column.
    assert "fit_algorithm" in results_csv.columns
    assert set(results_csv["fit_algorithm"]) == {"ridge", "linear"}
    # The candidates_log "params" JSON encodes the flat search-space form
    # (not the nested algo_params override).
    import json
    params_json = candidates_log["params"].iloc[0]
    decoded = json.loads(params_json)
    assert "fit_algorithm" in decoded
    assert "model_components" not in decoded


@pytest.mark.skipif(not GREYKITE_AVAILABLE, reason="blah.greykite not installed")
def test_e2e_greykite_grouped_writes_inspectable_artifacts(simulated_daily_df, template_backfill_config):
    """Grouped sweep — two stages with frozen-forward winners."""
    out_dir = ARTIFACT_ROOT / "grouped"
    wipe(out_dir)

    # Stage 1: pick the regression algorithm (flat key `fit_algorithm`).
    # Stage 2: with stage-1 winner frozen, tune the changepoint regularization
    # penalty — a real numeric tuning knob (regularization strength controls
    # how aggressively greykite trims spurious changepoints).
    # Both stages use flat keys; the converter handles the nesting.
    space = SearchSpace(
        groups=[
            ParamGroup(name="regression", params={"fit_algorithm": ["ridge", "linear"]}),
            ParamGroup(name="changepoint", params={"changepoint_reg": [0.01, 1.0]}),
        ]
    )
    criteria = EvalCriteria(
        eval_metrics=("mape", "smape", "medae", "mae", "rmse"),
        primary_eval_metric="mape",
        primary_eval_reduction="mean",
    )
    selection = GroupedModelSelection(
        search_space=space,
        backfill_config=template_backfill_config,
        eval_criteria=criteria,
        param_converter=greykite_forecast_algo.GreykiteParamConverter(),
        output_dir=out_dir,
        verbose=False,
        n_jobs=1,  # explicit — never spin joblib inside an xdist test worker
    )
    result = selection.run(df=simulated_daily_df)
    write_report(result, criteria, title="ModelSelection — Greykite Grouped (paper-cited layered stepwise)")

    candidates_log = pd.read_csv(out_dir / "model_candidates.csv")
    # Stage 1: 2 candidates (fit_algorithm). Stage 2: 2 candidates (changepoint_reg
    # with stage-1 winner frozen). Total: 4.
    assert len(candidates_log) == 4
    assert (candidates_log["status"] == "ok").all()
    assert int(candidates_log["is_winner"].sum()) == 2  # one per stage

    assert (out_dir / "stage_winners.json").exists()
    results_csv = pd.read_csv(out_dir / "results.csv")
    assert len(results_csv) == 4
    assert (out_dir / "results.html").exists()
    html = (out_dir / "results.html").read_text()
    assert "Greykite" in html
    assert "Stage Winners" in html

    # Final winner committed to both stage choices, in flat search-space form.
    assert "fit_algorithm" in result.best_params
    assert "changepoint_reg" in result.best_params
    assert result.best_params["fit_algorithm"] in {"ridge", "linear"}
    assert result.best_params["changepoint_reg"] in {0.01, 1.0}

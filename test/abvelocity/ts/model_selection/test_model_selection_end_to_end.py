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
"""End-to-end model-selection tests that write inspectable artifacts.

These tests mirror what a real user would do:

1. Generate / load some daily time-series data.
2. Build a :class:`BackfillConfig` template + :class:`SearchSpace` +
   :class:`EvalCriteria`.
3. Run :class:`GridModelSelection` and :class:`GroupedModelSelection` end-to-end.
4. Write the candidates log, per-candidate prediction CSVs, ranked
   ``results.csv``, and heat-mapped ``results.html`` to a stable location
   under ``docs/static/test-results/model-selection/`` so the artifacts
   can be inspected by hand after the test run.

Inspect after running:

    open docs/static/test-results/model-selection/grid/results.html
    open docs/static/test-results/model-selection/grouped/results.html
"""

import os
from pathlib import Path

import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401  registers "simple"
import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import TIME_COL
from abvelocity.ts.model_selection.eval_criteria import EvalCriteria
from abvelocity.ts.model_selection.grid import GridModelSelection
from abvelocity.ts.model_selection.grouped import GroupedModelSelection
from abvelocity.ts.model_selection.report import write_report
from abvelocity.ts.model_selection.space import ParamGroup, SearchSpace

# Resolve the inspect-able artifact root relative to this test file.
ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[3] / "docs" / "static" / "test-results" / "model-selection"
)


@pytest.fixture
def simulated_daily_df():
    """90 days of daily data with weekly + annual seasonality and noise.

    The series is realistic enough that hyperparameter choice (k, period,
    agg) materially affects the rolling-origin MAPE — making the
    selection report visually meaningful.
    """
    rng = np.random.default_rng(2026)
    n = 90
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    weekly = 6.0 * np.sin(2.0 * np.pi * np.arange(n) / 7.0)
    monthly = 3.0 * np.sin(2.0 * np.pi * np.arange(n) / 30.0)
    noise = rng.normal(0.0, 0.7, n)
    return pd.DataFrame({TIME_COL: dates, "y": 100.0 + weekly + monthly + noise})


@pytest.fixture
def template_backfill_config():
    """SimpleForecastAlgo with a 7-day period. SearchSpace patches k / agg."""
    fc = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("y",),
        freq="D",
        forecast_horizon=7,
        algo_name="simple",
        algo_params={"period": 7, "k": 3, "agg": "mean"},
    )
    return BackfillConfig(
        forecast_config=fc,
        initial_train_size=60,
        horizon=7,
        step=3,
        n_windows=5,
    )


def test_e2e_grid_writes_inspectable_artifacts(simulated_daily_df, template_backfill_config):
    out_dir = ARTIFACT_ROOT / "grid"
    if out_dir.exists():
        # Wipe so the test always reflects the current code path.
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    os.makedirs(out_dir, exist_ok=True)

    space = SearchSpace.flat({"k": [2, 3, 4], "agg": ["mean", "median"]})
    criteria = EvalCriteria(
        eval_metrics=("mape", "smape", "medae", "mae", "rmse"),
        primary_eval_metric="mape",
        primary_eval_reduction="mean",
    )
    selection = GridModelSelection(
        search_space=space,
        backfill_config=template_backfill_config,
        eval_criteria=criteria,
        output_dir=out_dir,
        verbose=False,
    )
    result = selection.run(df=simulated_daily_df)
    write_report(result, criteria, title="ModelSelection — Grid (simulated weekly+annual series)")

    # Hard assertions on what landed.
    candidates_log = pd.read_csv(out_dir / "model_candidates.csv")
    assert len(candidates_log) == 6                                # 3 k × 2 agg
    assert (candidates_log["status"] == "ok").all()
    for path in candidates_log["predict_path"]:
        assert (out_dir / path).exists()

    results_csv = pd.read_csv(out_dir / "results.csv")
    assert len(results_csv) == 6
    assert results_csv["score"].is_monotonic_increasing
    assert "mape_mean" in results_csv.columns
    assert "smape_mean" in results_csv.columns
    assert (out_dir / "results.html").exists()
    assert "Grid" in (out_dir / "results.html").read_text()


def test_e2e_grouped_writes_inspectable_artifacts(simulated_daily_df, template_backfill_config):
    out_dir = ARTIFACT_ROOT / "grouped"
    if out_dir.exists():
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    os.makedirs(out_dir, exist_ok=True)

    space = SearchSpace(
        groups=[
            ParamGroup(name="lookback", params={"k": [2, 3, 4]}),
            ParamGroup(name="aggregation", params={"agg": ["mean", "median"]}),
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
        output_dir=out_dir,
        verbose=False,
    )
    result = selection.run(df=simulated_daily_df)
    write_report(result, criteria, title="ModelSelection — Grouped (paper-cited layered stepwise)")

    candidates_log = pd.read_csv(out_dir / "model_candidates.csv")
    # 3 candidates in stage 1 (k) + 2 in stage 2 (agg, with stage-1 k frozen) = 5.
    assert len(candidates_log) == 5
    assert (candidates_log["status"] == "ok").all()
    assert int(candidates_log["is_winner"].sum()) == 2  # one winner per stage

    assert (out_dir / "stage_winners.json").exists()
    results_csv = pd.read_csv(out_dir / "results.csv")
    assert len(results_csv) == 5
    assert (out_dir / "results.html").exists()
    html = (out_dir / "results.html").read_text()
    assert "Grouped" in html
    assert "Stage Winners" in html

    # Final winner committed to both lookback + aggregation choices.
    assert "k" in result.best_params
    assert "agg" in result.best_params

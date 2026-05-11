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
"""End-to-end tests for GridModelSelection using SimpleForecastAlgo."""

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
from abvelocity.ts.model_selection.space import SearchSpace


@pytest.fixture
def daily_df():
    """30 days of seasonal data with a strong weekly cycle and some noise."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    weekly = 5.0 * np.sin(2.0 * np.pi * np.arange(30) / 7.0)
    noise = rng.normal(0.0, 0.5, 30)
    return pd.DataFrame({TIME_COL: dates, "y": 100.0 + weekly + noise})


@pytest.fixture
def base_backfill_config():
    fc = ForecastConfig(
        time_col=TIME_COL,
        value_cols=("y",),
        freq="D",
        forecast_horizon=3,
        algo_name="simple",
        algo_params={"period": 7, "k": 2},  # patched per candidate
    )
    return BackfillConfig(
        forecast_config=fc,
        initial_train_size=21,
        horizon=3,
        step=1,
        n_windows=3,  # 3 cutoffs is enough to make the eval table non-trivial
    )


def test_grid_writes_candidates_log_and_predictions(tmp_path: Path, daily_df, base_backfill_config):
    space = SearchSpace.flat({"k": [2, 3]})
    selection = GridModelSelection(
        search_space=space,
        backfill_config=base_backfill_config,
        eval_criteria=EvalCriteria(eval_metrics=("mape", "smape"), primary_eval_metric="mape"),
        output_dir=tmp_path,
        verbose=False,
    )
    result = selection.run(df=daily_df)

    # Two candidates, both should land predict files + candidates_log rows.
    candidates_csv = pd.read_csv(tmp_path / "model_candidates.csv")
    assert len(candidates_csv) == 2
    assert (candidates_csv["status"] == "ok").all()

    # Each candidate's parquet exists.
    for path in candidates_csv["predict_path"]:
        assert (tmp_path / path).exists()

    # Results CSV exists and is sorted ascending by score.
    results_csv = pd.read_csv(tmp_path / "results.csv")
    assert len(results_csv) == 2
    assert results_csv["score"].is_monotonic_increasing

    # Best params point to one of the swept k values.
    assert result.best_params["k"] in {2, 3}
    assert np.isfinite(result.best_score)


def test_grid_predict_only_mode(tmp_path: Path, daily_df, base_backfill_config):
    space = SearchSpace.flat({"k": [2, 3]})
    selection = GridModelSelection(
        search_space=space,
        backfill_config=base_backfill_config,
        eval_criteria=None,  # predict-only mode
        output_dir=tmp_path,
        verbose=False,
    )
    result = selection.run(df=daily_df)

    # ModelCandidatesLog still written.
    assert (tmp_path / "model_candidates.csv").exists()
    # Results.csv written but score column is NaN.
    results_csv = pd.read_csv(tmp_path / "results.csv")
    assert len(results_csv) == 2
    assert results_csv["score"].isna().all()
    # Best params is empty in predict-only mode.
    assert result.best_params == {}


def test_grid_resume_skips_cached_candidates(tmp_path: Path, daily_df, base_backfill_config):
    space = SearchSpace.flat({"k": [2]})
    selection = GridModelSelection(
        search_space=space,
        backfill_config=base_backfill_config,
        eval_criteria=EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape"),
        output_dir=tmp_path,
        verbose=False,
    )
    selection.run(df=daily_df)
    candidates_after_first = pd.read_csv(tmp_path / "model_candidates.csv")
    assert len(candidates_after_first) == 1
    pred_path = tmp_path / candidates_after_first.iloc[0]["predict_path"]
    first_mtime = pred_path.stat().st_mtime

    # Re-run; same candidate should be skipped.
    selection.run(df=daily_df)
    candidates_after_second = pd.read_csv(tmp_path / "model_candidates.csv")
    assert len(candidates_after_second) == 1  # no duplicate row
    assert pred_path.stat().st_mtime == first_mtime  # parquet not rewritten

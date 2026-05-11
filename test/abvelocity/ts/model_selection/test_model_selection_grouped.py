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
"""End-to-end tests for GroupedModelSelection."""

import json
from pathlib import Path

import abvelocity.ts.algo.simple_forecast_algo  # noqa: F401  registers "simple"
import numpy as np
import pandas as pd
import pytest
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import TIME_COL
from abvelocity.ts.model_selection.eval_criteria import EvalCriteria
from abvelocity.ts.model_selection.grouped import GroupedModelSelection
from abvelocity.ts.model_selection.space import ParamGroup, SearchSpace


@pytest.fixture
def daily_df():
    rng = np.random.default_rng(7)
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
        algo_params={"period": 7, "k": 2, "agg": "mean"},
    )
    return BackfillConfig(
        forecast_config=fc,
        initial_train_size=21,
        horizon=3,
        step=1,
        n_windows=3,
    )


def test_grouped_runs_each_stage_and_records_winners(tmp_path: Path, daily_df, base_backfill_config):
    space = SearchSpace(
        groups=[
            ParamGroup(name="lookback", params={"k": [2, 3]}),
            ParamGroup(name="aggregation", params={"agg": ["mean", "median"]}),
        ]
    )
    selection = GroupedModelSelection(
        search_space=space,
        backfill_config=base_backfill_config,
        eval_criteria=EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape"),
        output_dir=tmp_path,
        verbose=False,
    )
    result = selection.run(df=daily_df)

    # 2 stages, mini-grids of 2 each → 4 candidates total.
    candidates_log = pd.read_csv(tmp_path / "model_candidates.csv")
    assert len(candidates_log) == 4
    assert (candidates_log["status"] == "ok").all()

    # Two stage winners (one per stage) marked is_winner=True.
    winners = candidates_log[candidates_log["is_winner"]]
    assert len(winners) == 2
    assert sorted(winners["stage_idx"].tolist()) == [0, 1]

    # Stage winner JSON exists and has 2 entries (one per stage).
    stage_winners = json.loads((tmp_path / "stage_winners.json").read_text())
    assert len(stage_winners) == 2
    # Stage 1 should have committed a `k`. Stage 2 should have both `k` (frozen) and `agg` (chosen).
    assert "k" in stage_winners[0]
    assert "k" in stage_winners[1] and "agg" in stage_winners[1]

    # Final best_params committed both stage choices.
    assert "k" in result.best_params
    assert "agg" in result.best_params

    # results.csv covers every candidate.
    results_csv = pd.read_csv(tmp_path / "results.csv")
    assert len(results_csv) == 4


def test_grouped_multi_param_group_takes_cartesian_within_stage(tmp_path: Path, daily_df, base_backfill_config):
    """A single ParamGroup with multiple multi-valued params sweeps the cartesian
    product within that stage. This is the "joint tuning" case — params that
    interact strongly enough that they should be picked together rather than
    sequentially. Stage 1 has 2x2=4 candidates; stage 2 freezes both stage-1
    winners and adds 2 more.
    """
    space = SearchSpace(
        groups=[
            # Stage 1 jointly tunes (k, agg) — 2 × 2 = 4 candidates.
            ParamGroup(name="lookback_and_agg", params={"k": [2, 3], "agg": ["mean", "median"]}),
            # Stage 2 picks the seasonal period with both stage-1 params frozen.
            ParamGroup(name="period", params={"period": [1, 7]}),
        ]
    )
    selection = GroupedModelSelection(
        search_space=space,
        backfill_config=base_backfill_config,
        eval_criteria=EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape"),
        output_dir=tmp_path,
        verbose=False,
    )
    result = selection.run(df=daily_df)

    candidates_log = pd.read_csv(tmp_path / "model_candidates.csv")
    # Stage 1: 4 candidates (2x2 cartesian within the joint group). Stage 2: 2.
    stage_1 = candidates_log[candidates_log["stage_idx"] == 0]
    stage_2 = candidates_log[candidates_log["stage_idx"] == 1]
    assert len(stage_1) == 4
    assert len(stage_2) == 2
    assert (candidates_log["status"] == "ok").all()

    # Stage 1 winner commits BOTH joint params.
    assert "k" in result.stage_winners[0]
    assert "agg" in result.stage_winners[0]
    # Stage 2 freezes them and adds period.
    assert result.stage_winners[1]["k"] == result.stage_winners[0]["k"]
    assert result.stage_winners[1]["agg"] == result.stage_winners[0]["agg"]
    assert "period" in result.stage_winners[1]


def test_grouped_requires_eval_criteria(tmp_path: Path, base_backfill_config):
    space = SearchSpace.flat({"k": [2, 3]})
    with pytest.raises(ValueError, match="requires eval_criteria"):
        GroupedModelSelection(
            search_space=space,
            backfill_config=base_backfill_config,
            eval_criteria=None,  # not allowed for grouped
            output_dir=tmp_path,
            verbose=False,
        )


def test_grouped_freezes_winner_into_next_stage(tmp_path: Path, daily_df, base_backfill_config):
    """Stage 2 candidates carry stage 1's winning `k` in their full params dict."""
    space = SearchSpace(
        groups=[
            ParamGroup(name="lookback", params={"k": [2, 3]}),
            ParamGroup(name="aggregation", params={"agg": ["mean"]}),  # single value
        ]
    )
    selection = GroupedModelSelection(
        search_space=space,
        backfill_config=base_backfill_config,
        eval_criteria=EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape"),
        output_dir=tmp_path,
        verbose=False,
    )
    selection.run(df=daily_df)

    candidates_log = pd.read_csv(tmp_path / "model_candidates.csv")
    stage_2 = candidates_log[candidates_log["stage_idx"] == 1]
    # Stage 2 has 1 candidate (one agg × frozen k).
    assert len(stage_2) == 1
    params = json.loads(stage_2.iloc[0]["params"])
    assert "k" in params and "agg" in params
    assert params["agg"] == "mean"


def test_grouped_reopen_keeps_cached_candidate_eligible(tmp_path: Path, daily_df, base_backfill_config):
    """When stage 2 ``reopen=['k']`` re-evaluates a (k, agg) combo already
    computed in stage 1, the cached row's stage_idx stays at 0. The
    per-stage filter must therefore key on candidate_id, not stage_idx —
    otherwise that cached row is silently dropped from stage 2's winner
    pool and the wrong candidate wins.

    This regression-tests the bug noted on PR #517: a stage-1 (k=2, agg=mean)
    candidate cached as stage_idx=0 must still compete in stage 2 when stage
    2 reopens k and re-emits (k=2, agg=mean) as one of its combinations.
    """
    space = SearchSpace(
        groups=[
            # Stage 1 pins agg=mean, sweeps k. Each candidate's params dict
            # is {"k": ..., "agg": "mean"} → distinct candidate_ids.
            ParamGroup(name="lookback", params={"k": [2, 3], "agg": ["mean"]}),
            # Stage 2 reopens k. stage_candidates emits (k, agg) for every
            # combination; (k=stage1_winner, agg="mean") collides with a
            # stage-1 candidate's id and hits the cache in predict_and_persist.
            ParamGroup(
                name="aggregation",
                params={"agg": ["mean", "median"]},
                reopen=["k"],
            ),
        ]
    )
    selection = GroupedModelSelection(
        search_space=space,
        backfill_config=base_backfill_config,
        eval_criteria=EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape"),
        output_dir=tmp_path,
        verbose=False,
    )
    result = selection.run(df=daily_df)

    # Stage 2's winner must be one of (k ∈ {2,3} × agg ∈ {mean,median}). When
    # the bug was present, (k=stage1_winner, agg=mean) was silently filtered
    # out as "stage_idx=0" — now it's eligible because the filter keys on
    # candidate_id. Both possible final winners must be reachable.
    assert result.best_params["agg"] in {"mean", "median"}
    assert result.best_params["k"] in {2, 3}
    # Two stages, each commits a winner.
    assert len(result.stage_winners) == 2
    # The stage-2 results frame must contain the cached (k=stage1_winner,
    # agg=mean) candidate as one of the rows considered for stage-2 winner.
    candidates_log = pd.read_csv(tmp_path / "model_candidates.csv")
    stage_1_winner_k = result.stage_winners[0]["k"]
    cached_combo_mask = (
        (candidates_log["status"] == "ok")
        & (candidates_log["params"].apply(
            lambda j: json.loads(j) == {"k": stage_1_winner_k, "agg": "mean"}
        ))
    )
    assert cached_combo_mask.any(), "the (stage1_winner_k, mean) candidate must exist in the log"
    cached_row = candidates_log[cached_combo_mask].iloc[0]
    # Cached row keeps its original stage_idx (0); this is intentional —
    # the audit trail records when each candidate was first computed.
    assert cached_row["stage_idx"] == 0


def test_grouped_writes_per_stage_top_k(tmp_path: Path, daily_df, base_backfill_config):
    """The Grouped run produces ``stage_results.csv`` with the top-K of each
    stage side-by-side, so a reader can compare scores across stages and
    decide whether each layer's added complexity actually helped."""
    space = SearchSpace(
        groups=[
            ParamGroup(name="lookback", params={"k": [2, 3, 4]}),
            ParamGroup(name="aggregation", params={"agg": ["mean", "median"]}),
        ]
    )
    selection = GroupedModelSelection(
        search_space=space,
        backfill_config=base_backfill_config,
        eval_criteria=EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape"),
        output_dir=tmp_path,
        verbose=False,
        stage_top_k=2,
    )
    selection.run(df=daily_df)

    stage_results = pd.read_csv(tmp_path / "stage_results.csv")
    # Stage 1: 3 candidates (k=2,3,4), top_k=2 → 2 rows. Stage 2: 2 candidates
    # (frozen k × {mean, median}), top_k=2 but only 2 candidates exist → 2 rows.
    assert len(stage_results) == 2 + 2
    assert set(stage_results["stage_idx"].unique()) == {0, 1}
    assert set(stage_results["stage_name"].unique()) == {"lookback", "aggregation"}
    # Each stage's ranks start at 1.
    for sidx in (0, 1):
        sub = stage_results[stage_results["stage_idx"] == sidx]
        assert list(sub["rank"]) == list(range(1, len(sub) + 1))
        # Sorted ascending by score (lower is better for MAPE).
        assert sub["score"].is_monotonic_increasing
    # Eval-metric mean column is present.
    assert "mape_mean" in stage_results.columns


def test_grouped_returns_best_eval_df(tmp_path: Path, daily_df, base_backfill_config):
    """SelectionResult.best_eval_df carries the cumulative winner's full
    per-(metric_id, horizon_step) eval frame — so the caller can inspect
    horizon-1 vs horizon-N degradation without re-loading the on-disk CSV."""
    space = SearchSpace(
        groups=[
            ParamGroup(name="lookback", params={"k": [2, 3]}),
            ParamGroup(name="aggregation", params={"agg": ["mean"]}),
        ]
    )
    selection = GroupedModelSelection(
        search_space=space,
        backfill_config=base_backfill_config,
        eval_criteria=EvalCriteria(eval_metrics=("mape", "mae"), primary_eval_metric="mape"),
        output_dir=tmp_path,
        verbose=False,
    )
    result = selection.run(df=daily_df)

    assert result.best_eval_df is not None
    # Per-(metric_id, horizon_step) frame: at least one column per eval metric.
    assert "mape" in result.best_eval_df.columns
    assert "mae" in result.best_eval_df.columns
    # 1 metric_id × 3 horizon steps (horizon=3 in base_backfill_config).
    assert len(result.best_eval_df) == 3

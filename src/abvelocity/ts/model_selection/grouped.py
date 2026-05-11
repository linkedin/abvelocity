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
"""Grouped model selection — layered grouped-stepwise sweep.

Each stage is an inner :class:`GridModelSelection` over that stage's
mini-grid (with prior-stage winners pinned and any ``reopen`` keys
expanded). The compositional structure means stages get every Grid
feature for free — joblib parallelism via ``n_jobs``, content-hashed
candidate caching across stages, the same per-candidate eval-CSV
pipeline. Stages still run sequentially because each stage's winner
determines what's pinned in the next stage's grid.

Implements the algorithm described in:

    Hosseini, R., Newlands, N. K., Dean, C. B., & Takemura, A. (2015).
    "Statistical Modeling of Soil Moisture, Integrating Satellite
    Remote-Sensing (SAR) and Ground-Based Data."
    Remote Sensing 7(3), 2752–2780.
    https://doi.org/10.3390/rs70302752

Quoting §3.4 ("Model Structure") and the discussion section:

    "...we utilized the first of these approaches, devising a grouped,
    stepwise method that conducts an iterative search of the predictor
    space corresponding to a group of selected leading predictors. This
    extends regular stepwise methods to the multivariate case."

    "At the highest layer, we selected the best model (in terms of
    prediction error as explained below) for each of the five model
    families; in the second layer we choose the best model for each of
    the families in conjunction with ground data; in the third layer, we
    choose the best of all the models over the families of models; and
    finally in the last layer we add spatial correlation."

The implementation here is the simpler linear form of that procedure:
groups are evaluated stepwise in the order declared on the
:class:`SearchSpace`, each stage's mini-grid is scored with prior stages'
winners frozen, and the lowest-primary-score candidate in each stage
becomes the running winner whose params are merged forward.

The richer DAG-of-stages form (e.g. parallel branches that merge before a
final stage) is not in v1 but the persistence and candidates-log layout
already support it: every candidate carries a ``stage_idx`` /
``stage_name``, and the running cumulative template is recorded in
``stage_winners.json``.

Inspecting "did added complexity help?"
---------------------------------------
After the run, ``stage_results.csv`` contains the per-stage top-K
candidates side by side: stage 1's K best vs. stage 2's K best, etc.
Comparing the best score across stages tells you whether each stage's
added complexity actually improved the chosen eval metric or just added
noise.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
from abvelocity.ts.model_selection.base import ModelSelection, SelectionResult
from abvelocity.ts.model_selection.grid import GridModelSelection
from abvelocity.ts.model_selection.model_candidates import (
    ModelCandidatesLog,
    STAGE_WINNERS_FILENAME,
    compute_candidate_id,
)
from abvelocity.ts.model_selection.space import ParamGroup, SearchSpace

STAGE_RESULTS_FILENAME = "stage_results.csv"
"""Per-stage top-K candidate report — one row per (stage, rank) entry."""


@dataclass
class GroupedModelSelection(ModelSelection):
    """Layered grouped-stepwise selection.

    For each :class:`ParamGroup` in :attr:`search_space.groups`:

    1. Enumerate the stage's mini-grid candidates with the prior stages'
       running winners frozen
       (:meth:`SearchSpace.stage_candidates`).
    2. Predict each via :meth:`ModelSelection.predict_and_persist`.
    3. Evaluate the stage's rows via
       :meth:`ModelSelection.evaluate_candidates` filtered to this stage.
    4. The lowest-primary-score candidate is the stage winner; its
       *stage-only* params are merged into the running winners dict and
       used as part of the frozen template for the next stage.

    After the final stage, a global pass evaluates every candidate in the
    candidates_log so that ``results.csv`` covers the full audit trail (not
    just stage winners).

    :attr:`eval_criteria` is required (no predict-only mode for grouped).

    Attributes:
        method: Always ``"grouped"``.
    """

    method: str = "grouped"
    """Always ``"grouped"``."""

    stage_top_k: int = 5
    """How many of each stage's top candidates to record in
    ``stage_results.csv`` for the post-run "did added complexity help?"
    report. Default 5; clamped to the stage's actual candidate count
    when smaller."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.eval_criteria is None:
            raise ValueError(
                "GroupedModelSelection requires eval_criteria — needed inline at every stage "
                "to pick the running winner. Use GridModelSelection if you want predict-only mode."
            )
        if self.stage_top_k < 1:
            raise ValueError(f"stage_top_k must be >= 1, got {self.stage_top_k}.")

    def run(self, df: pd.DataFrame) -> SelectionResult:
        """Sweep groups in order, delegating each stage to a Grid run.

        Each stage:

        1. Constructs the stage's flat search grid: the group's own
           ``params``, plus any ``reopen`` keys expanded to their original
           value lists, plus every prior-stage winner pinned to a
           single-value list. The cartesian product of this grid is the
           stage's candidate set.
        2. Runs that grid via :class:`GridModelSelection` over the
           shared ``output_dir``. The inner Grid handles cache hits
           (cross-stage), joblib parallelism, eval-CSV caching, and
           per-candidate audit-log writes.
        3. Filters the inner Grid's results to this stage's candidate
           ids (a cached candidate from an earlier stage can compete
           here if its params match a current-stage combination, and
           we want to consider it). Picks the top-scoring candidate
           as the stage winner.
        4. Records the top-K of this stage's results into
           ``stage_results.csv`` for post-hoc inspection.

        Args:
            df: Prepped time-series DataFrame for the predictor.

        Returns:
            :class:`SelectionResult` with the global ``results_df``
            (every candidate from every stage),
            :attr:`~SelectionResult.best_params` = the cumulative
            winner after the last stage, :attr:`~SelectionResult.best_score`
            = its score, :attr:`~SelectionResult.best_eval_df` = the
            cumulative winner's full per-(``metric_id``,
            ``horizon_step``) eval frame, and
            :attr:`~SelectionResult.stage_winners` listing the running
            params dict at the close of each stage.
        """
        self.write_cutoffs_log(df)
        running_winners: Dict[str, Any] = {}
        stage_winner_history: List[Dict[str, Any]] = []
        stage_top_k_records: List[Dict[str, Any]] = []

        for stage_idx, group in enumerate(self.search_space.groups):
            if self.verbose:
                print(
                    f"\n*** GroupedModelSelection stage {stage_idx + 1}/{len(self.search_space.groups)}: "
                    f"{group.name!r}"
                )
                print(f"    frozen so far: {running_winners}")

            # 1. Build the stage's flat search grid (own params + reopen + frozen pinned).
            stage_grid_dict = self._build_stage_grid_dict(group, running_winners)
            stage_search_space = SearchSpace.flat(stage_grid_dict, name=group.name)
            stage_candidate_ids = {
                compute_candidate_id(p) for p in stage_search_space.cartesian_candidates()
            }

            # 2. Delegate to GridModelSelection — same output_dir, same eval_criteria,
            #    same n_jobs, same param_converter. ``write_results_csv=False`` so the
            #    inner Grid doesn't clobber the outer ``results.csv``.
            stage_grid = GridModelSelection(
                search_space=stage_search_space,
                backfill_config=self.backfill_config,
                output_dir=self.output_dir,
                eval_criteria=self.eval_criteria,
                param_converter=self.param_converter,
                anomaly_df=self.anomaly_df,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                stage_idx=stage_idx,
                stage_name=group.name,
                write_results_csv=False,
            )
            stage_grid_result = stage_grid.run(df)

            # 3. Filter inner Grid's results to this stage's candidates and pick the winner.
            stage_results_df = stage_grid_result.results_df[
                stage_grid_result.results_df["candidate_id"].isin(stage_candidate_ids)
            ]
            ok_rows = stage_results_df[stage_results_df["status"] == "ok"]
            if ok_rows.empty:
                raise RuntimeError(
                    f"Stage {stage_idx} ({group.name!r}) produced no successful candidates; "
                    f"check the candidates_log at "
                    f"{(self.output_dir / 'model_candidates.csv')!r} for errors."
                )

            top = ok_rows.iloc[0]
            stage_winner_full = {
                k: top[k]
                for k in self.search_space.all_param_names()
                if k in top and pd.notna(top[k])
            }
            stage_only_winner = self.extract_stage_choice(group, top)
            running_winners.update(stage_only_winner)
            stage_winner_history.append(dict(running_winners))

            # Mark winner in the candidates log.
            candidates_log = ModelCandidatesLog.load(self.output_dir)
            self.mark_stage_winner(candidates_log, candidate_id=str(top["candidate_id"]))

            # 4. Collect this stage's top-K for the per-stage report.
            top_k_slice = ok_rows.head(self.stage_top_k)
            for rank, (_, candidate_row) in enumerate(top_k_slice.iterrows(), start=1):
                record: Dict[str, Any] = {
                    "stage_idx": stage_idx,
                    "stage_name": group.name,
                    "rank": rank,
                    "candidate_id": candidate_row["candidate_id"],
                    "label": candidate_row["label"],
                    "score": float(candidate_row["score"]),
                }
                # Emit each eval metric's mean column if present.
                for eval_metric_name in self.eval_criteria.eval_metrics:
                    col = f"{eval_metric_name}_mean"
                    if col in candidate_row:
                        record[col] = candidate_row[col]
                # Emit every parameter that was decided at this stage AND every
                # frozen-from-prior-stage param so the row is self-contained.
                for param_name in self.search_space.all_param_names():
                    if param_name in candidate_row and pd.notna(candidate_row[param_name]):
                        record[param_name] = candidate_row[param_name]
                stage_top_k_records.append(record)

            if self.verbose:
                print(
                    f"    winner: {stage_only_winner}  "
                    f"(score={top['score']:.4f}, full={stage_winner_full})"
                )

        # Final global eval pass — re-rank every candidate in the candidates_log
        # so the outer results.csv contains the full audit trail (not just per-
        # stage winners).
        candidates_log = ModelCandidatesLog.load(self.output_dir)
        global_results = self.evaluate_candidates(candidates_log, criteria=self.eval_criteria)
        global_results.to_csv(self.output_dir / "results.csv", index=False)

        # Per-stage top-K report.
        if stage_top_k_records:
            stage_top_df = pd.DataFrame(stage_top_k_records)
            stage_top_df.to_csv(self.output_dir / STAGE_RESULTS_FILENAME, index=False)

        with open(self.output_dir / STAGE_WINNERS_FILENAME, "w") as fh:
            json.dump(stage_winner_history, fh, default=str, indent=2)

        # The grouped-stepwise winner is the cumulative result of the greedy walk —
        # NOT the globally-best candidate in the candidates_log. A stage-1-only
        # candidate may appear "best" globally because earlier-stage params are
        # unset (NaN) for later-stage rows or vice versa, but the algorithm's
        # actual answer is the final running_winners dict.
        best_params: Dict[str, Any] = dict(running_winners)
        best_score = self.lookup_score(global_results, running_winners)

        # best_eval_df: load the cumulative winner's per-group eval frame.
        winner_id = self._find_winner_candidate_id(global_results, running_winners)
        best_eval_df = (
            self._load_best_eval_df(candidates_log, winner_id) if winner_id else None
        )

        return SelectionResult(
            results_df=global_results,
            best_params=best_params,
            best_score=best_score,
            output_dir=self.output_dir,
            method=self.method,
            stage_winners=stage_winner_history,
            best_eval_df=best_eval_df,
            search_space=self.search_space,
            backfill_config=self.backfill_config,
            param_converter=self.param_converter,
        )

    def _build_stage_grid_dict(
        self,
        group: ParamGroup,
        frozen_winners: Dict[str, Any],
    ) -> Dict[str, List[Any]]:
        """Build the per-stage flat search grid for a stage's inner Grid run.

        The grid is the cartesian-product input the stage's
        :class:`GridModelSelection` enumerates. It contains:

        * every key in ``group.params`` with the group's own value list;
        * every key in ``group.reopen`` (if any) with the value list from
          the *originating* prior group (so the reopened parameter is
          re-swept across all its candidate values);
        * every other ``frozen_winners`` key not already covered by the
          two above, pinned to a single-value list (so the cartesian
          product holds it constant at the prior stage's choice).

        The cartesian product of this dict produces exactly the same
        full-params dicts that
        :meth:`SearchSpace.stage_candidates` yields for this stage —
        modulo iteration order, which doesn't matter because Grid sorts
        by score.

        Args:
            group: The current stage's :class:`ParamGroup`.
            frozen_winners: Cumulative winners from prior stages.

        Returns:
            ``{param_name: [value, ...]}`` ready to pass to
            :meth:`SearchSpace.flat`.

        Raises:
            ValueError: If ``group.reopen`` references a parameter not
                declared by any earlier group.
        """
        stage_grid: Dict[str, List[Any]] = {k: list(v) for k, v in group.params.items()}

        if group.reopen:
            earlier_lookup: Dict[str, List[Any]] = {}
            this_group_idx = self.search_space.groups.index(group)
            for prior in self.search_space.groups[:this_group_idx]:
                for key, values in prior.params.items():
                    earlier_lookup[key] = list(values)
            for reopen_key in group.reopen:
                if reopen_key not in earlier_lookup:
                    raise ValueError(
                        f"ParamGroup({group.name!r}).reopen lists {reopen_key!r}, "
                        f"but no earlier group declared that parameter."
                    )
                stage_grid[reopen_key] = earlier_lookup[reopen_key]

        for key, value in frozen_winners.items():
            if key not in stage_grid:
                stage_grid[key] = [value]

        return stage_grid

    def _find_winner_candidate_id(
        self,
        results_df: pd.DataFrame,
        running_winners: Dict[str, Any],
    ) -> str:
        """Return the candidate_id whose params exactly match ``running_winners``.

        Used to look up :attr:`SelectionResult.best_eval_df`. Empty
        string when no exact match is found.
        """
        if results_df.empty:
            return ""
        mask = pd.Series(True, index=results_df.index)
        for key, value in running_winners.items():
            if key not in results_df.columns:
                return ""
            mask &= results_df[key] == value
        matches = results_df[mask]
        if matches.empty:
            return ""
        return str(matches.iloc[0]["candidate_id"])

    def extract_stage_choice(self, group: ParamGroup, candidate_row: pd.Series) -> Dict[str, Any]:
        """Return the stage-only param values from a results-frame row.

        At each stage, the running winner only commits to the keys
        introduced (or reopened) at that stage; earlier-stage keys remain
        their already-frozen values.

        Numpy scalars are coerced to their Python natives (``np.int64(3)``
        → ``int(3)``, ``np.float64(0.5)`` → ``float(0.5)``). Without this
        coercion the next stage's params dict carries a numpy type while
        prior stages had Python natives — the resulting mixed-dtype
        column in ``results_df`` makes downstream ``==`` lookups (e.g.
        :meth:`lookup_score`, :meth:`_find_winner_candidate_id`) silently
        return no match.

        Args:
            group: The :class:`ParamGroup` for the current stage.
            candidate_row: One row from the per-stage results DataFrame.

        Returns:
            Dict with one entry per key in ``group.params`` (and any
            ``group.reopen`` keys), pulled from ``candidate_row`` if
            present, with numpy scalars normalised.
        """
        keys: List[str] = list(group.params.keys())
        if group.reopen:
            keys.extend(group.reopen)
        out: Dict[str, Any] = {}
        for k in keys:
            if k in candidate_row and pd.notna(candidate_row[k]):
                value = candidate_row[k]
                if hasattr(value, "item"):
                    value = value.item()
                out[k] = value
        return out

    def lookup_score(self, results_df: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Return the score of the candidate whose params match ``params`` exactly.

        Args:
            results_df: Output of :meth:`ModelSelection.evaluate_candidates`.
            params: Cumulative running winners after the last stage.

        Returns:
            The matching candidate's ``score``. ``EvalCriteria.worst_score()`` if
            no exact match is found (should not happen in practice — the
            cumulative winner was just predicted, persisted, and evaluated).
        """
        if results_df.empty:
            return self.eval_criteria.worst_score()
        mask = pd.Series(True, index=results_df.index)
        for key, value in params.items():
            if key not in results_df.columns:
                return self.eval_criteria.worst_score()
            mask &= results_df[key] == value
        matches = results_df[mask]
        if matches.empty:
            return self.eval_criteria.worst_score()
        return float(matches.iloc[0]["score"])

    def mark_stage_winner(self, candidates_log: ModelCandidatesLog, candidate_id: str) -> None:
        """Set ``is_winner=True`` on the matching candidates_log candidate and flush.

        Args:
            candidates_log: Mutable candidates_log.
            candidate_id: Id of the stage's winning candidate.
        """
        for candidate_row in candidates_log.rows:
            if candidate_row.candidate_id == candidate_id:
                candidate_row.is_winner = True
        candidates_log.flush()

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
"""Grid model selection — exhaustive cartesian sweep."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from joblib import Parallel, delayed
from abvelocity.ts.model_selection.base import ModelSelection, SelectionResult
from abvelocity.ts.model_selection.model_candidates import ModelCandidatesLog, compute_candidate_id, format_label


@dataclass
class GridModelSelection(ModelSelection):
    """Exhaustive cartesian sweep over the search space.

    Behaviour
    ---------
    1. Flatten the :class:`SearchSpace` into a single cartesian product
       via :meth:`SearchSpace.cartesian_candidates`.
    2. For each candidate dict, predict via
       :meth:`ModelSelection.predict_and_persist` (writes prediction CSV
       + candidates-log candidate, skips if cached).
    3. After all predictions land, if :attr:`eval_criteria` is set, run
       :meth:`ModelSelection.evaluate_candidates` to produce the ranked
       ``results.csv``. If it is ``None``, the run is *predict-only*; the
       caller can later score with
       :func:`~abvelocity.ts.model_selection.base.evaluate_existing`.

    Attributes:
        method: Always ``"grid"``.
        stage_idx: Stage index recorded on every candidate produced by
            this run. ``-1`` (default) for a standalone Grid;
            :class:`GroupedModelSelection` sets this to the current
            stage index when invoking Grid as a sub-step of a layered
            sweep.
        stage_name: Stage name recorded on every candidate produced by
            this run. Empty (default) for a standalone Grid; set by
            :class:`GroupedModelSelection` to the current
            :class:`ParamGroup` name.
        write_results_csv: When ``True`` (default), the run writes
            ``output_dir/results.csv`` at the end. Set to ``False`` when
            running as a sub-step of :class:`GroupedModelSelection` so
            the per-stage Grid run doesn't clobber the outer
            ``results.csv`` that Grouped writes after all stages have
            completed.
    """

    method: str = "grid"
    """Always ``"grid"``."""

    stage_idx: int = -1
    """Stage index recorded on candidates persisted by this run; ``-1``
    for a standalone Grid (not invoked as a Grouped sub-step)."""

    stage_name: str = ""
    """Stage name recorded on candidates persisted by this run; empty
    for a standalone Grid."""

    write_results_csv: bool = True
    """If True (default) write ``output_dir/results.csv`` at the end of
    the run. Set to False when running as a Grouped sub-step."""

    def run(self, df: pd.DataFrame) -> SelectionResult:
        """Predict every cartesian candidate, then optionally evaluate.

        Args:
            df: Prepped time-series DataFrame for the predictor; passed
                unchanged to every per-candidate
                :class:`~abvelocity.ts.backfill.runner.BackfillRunner`
                run.

        Returns:
            :class:`SelectionResult` with the ranked ``results_df``,
            ``best_params``, ``best_score``, and ``output_dir``. If
            :attr:`eval_criteria` is ``None``, ``results_df`` lists every
            candidate but ``score`` is ``NaN`` and ``best_*`` reflect the
            empty case.
        """
        candidates = self.search_space.cartesian_candidates()
        if self.verbose:
            print(f"\n*** GridModelSelection: {len(candidates)} candidates  (n_jobs={self.n_jobs})")

        self.write_cutoffs_log(df)
        candidates_log = ModelCandidatesLog.load(self.output_dir)

        if self.n_jobs == 1:
            for params in candidates:
                self.predict_and_persist(
                    params=params,
                    df=df,
                    candidates_log=candidates_log,
                    stage_idx=self.stage_idx,
                    stage_name=self.stage_name,
                    skip_if_cached=True,
                )
        else:
            # Parallel path: skip cache check + candidates-log writes inside the
            # workers (each worker is a separate process — concurrent writes to
            # model_candidates.csv would race). We split cached from fresh up
            # front, then dispatch only the fresh candidates to joblib, then
            # append cached + fresh rows back to the candidates_log serially in
            # the main process.
            cached_ids = candidates_log.existing_ids()
            fresh = [p for p in candidates if compute_candidate_id(p) not in cached_ids]
            if self.verbose:
                print(f"    {len(cached_ids)} cached, {len(fresh)} fresh; dispatching to {self.n_jobs} workers")
            for cached_row in cached_ids.values():
                if self.verbose:
                    print(f"\n*** ModelSelection cached {cached_row.label}  (id={cached_row.candidate_id})")
            fresh_rows = Parallel(n_jobs=self.n_jobs)(
                delayed(self.predict_one)(
                    params=params, df=df, stage_idx=self.stage_idx, stage_name=self.stage_name,
                )
                for params in fresh
            )
            for candidate_row in fresh_rows:
                candidates_log.append(candidate_row)
                if self.verbose and candidate_row.status == "ok":
                    print(f"\n*** ModelSelection ran {format_label(candidate_row.params)}  (id={candidate_row.candidate_id})")
            candidates_log.flush()

        # Restrict the result set to THIS sweep's candidate ids. The candidates
        # log on disk may carry rows from prior sweeps that reused this
        # output_dir with a different SearchSpace; without this filter those
        # stale rows would be evaluated, ranked, and could even surface as
        # ``best_params`` / appear in the visible results.csv.
        current_candidate_ids = {compute_candidate_id(p) for p in candidates}

        if self.eval_criteria is None:
            # Predict-only mode: no notion of "best" because there's no
            # scoring criterion. NaN signals "not applicable" — clearer
            # than a sentinel infinity that would imply a direction.
            results_df = self.bare_results_df(candidates_log)
            results_df = results_df[results_df["candidate_id"].isin(current_candidate_ids)].reset_index(drop=True)
            if self.write_results_csv:
                results_df.to_csv(self.output_dir / "results.csv", index=False)
            best_params: Dict[str, Any] = {}
            best_score = float("nan")
            best_eval_df: Optional[pd.DataFrame] = None
        else:
            results_df = self.evaluate_candidates(candidates_log, criteria=self.eval_criteria)
            results_df = results_df[results_df["candidate_id"].isin(current_candidate_ids)].reset_index(drop=True)
            if self.write_results_csv:
                results_df.to_csv(self.output_dir / "results.csv", index=False)
            ok_rows = results_df[results_df["status"] == "ok"]
            if not ok_rows.empty:
                top = ok_rows.iloc[0]
                best_params = {k: top[k] for k in self.search_space.all_param_names() if k in top}
                best_score = float(top["score"])
                best_eval_df = self._load_best_eval_df(candidates_log, str(top["candidate_id"]))
            else:
                # Eval criteria was set but every candidate raised — return
                # ``worst_score()`` (direction-aware: +inf when lower-is-better,
                # -inf otherwise) so callers comparing against this value still
                # get sensible ranking.
                best_params = {}
                best_score = self.eval_criteria.worst_score()
                best_eval_df = None

        return SelectionResult(
            results_df=results_df,
            best_params=best_params,
            best_score=best_score,
            output_dir=self.output_dir,
            method=self.method,
            best_eval_df=best_eval_df,
            search_space=self.search_space,
            backfill_config=self.backfill_config,
            param_converter=self.param_converter,
        )

    def bare_results_df(self, candidates_log: ModelCandidatesLog) -> pd.DataFrame:
        """Build a results frame for predict-only runs (no eval).

        Args:
            candidates_log: ModelCandidatesLog with all rows persisted.

        Returns:
            DataFrame with the param columns, ``status``, ``candidate_id``,
            ``label``, and ``score=NaN``. Rows are in candidates_log order.
        """
        records = []
        for candidate_row in candidates_log.rows:
            record: Dict[str, Any] = {
                **candidate_row.params,
                "candidate_id": candidate_row.candidate_id,
                "label": candidate_row.label,
                "status": candidate_row.status,
                "score": float("nan"),
            }
            records.append(record)
        return pd.DataFrame(records)

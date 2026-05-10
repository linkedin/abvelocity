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
"""Abstract base for model-selection methods.

Future-direction note
---------------------
This module is intentionally forecast-coupled in v1: :class:`ModelSelection`
holds a :class:`BackfillConfig` directly and uses :class:`BackfillRunner`
as the prediction engine. The natural v2 evolution is to extract a
generic base — a hypothetical ``BaseModelSelection`` that owns the
SearchSpace, ModelCandidatesLog, and EvalCriteria plumbing while delegating
prediction to an abstract ``predict_one`` (or a composed ``Predictor``
object). The current :class:`ModelSelection` would then become the
forecast specialisation. Naming the public surface generically
(``ModelSelection``, ``GridModelSelection``, ``GroupedModelSelection``)
keeps callers stable across that future split — only the
forecast-specific fields (``backfill_config``) would migrate.

A :class:`ModelSelection` instance bundles everything needed to sweep a
:class:`~abvelocity.ts.model_selection.space.SearchSpace` of
parameter overrides on top of a :class:`~abvelocity.ts.backfill.config.BackfillConfig`
template:

* the **template** itself — a :class:`BackfillConfig` that fixes the
  prediction spec (``forecast_horizon``, ``step``, ``window_type``, etc.)
  and supplies the default :class:`ForecastConfig` whose ``algo_params``
  the search space patches;
* the **search space** — declares which keys to override per candidate;
* the **eval criteria** — required for grouped, optional for grid;
* the **output directory** — destination for per-candidate prediction
  CSVs, the candidates log, stage winners, and the final ranked results.

Concrete subclasses
(:class:`~abvelocity.ts.model_selection.grid.GridModelSelection`,
:class:`~abvelocity.ts.model_selection.grouped.GroupedModelSelection`)
override :meth:`run` to control which candidates get enumerated and in
which order. The shared per-candidate predict-and-persist loop lives on
this base class as :meth:`predict_and_persist` so both methods share
identical I/O semantics.
"""

import json
import logging
from abc import abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.backfill.runner import BackfillRunner
from abvelocity.ts.model_selection.eval_criteria import EvalCriteria
from abvelocity.ts.model_selection.model_candidates import (
    ModelCandidatesLog,
    ModelCandidate,
    compute_candidate_id,
    eval_path_for,
    fit_info_path_for,
    format_label,
    predictions_path_for,
)
from abvelocity.ts.model_selection.param_converter import ParamConverter
from abvelocity.ts.model_selection.space import SearchSpace

logger = logging.getLogger(__name__)

ERROR_MSG_TRUNCATION = 500
"""Maximum number of characters of a traceback recorded in the candidates_log."""

CUTOFFS_FILENAME = "cutoffs.json"
"""Filename for the per-run cutoff-dates log written next to the candidates CSV."""


@dataclass
class SelectionResult:
    """Output of a :class:`ModelSelection.run` call.

    Attributes:
        results_df: One row per candidate, sorted ascending by primary
            score (lower is better). Columns include every parameter name
            from the search space, ``score``, one ``<eval-metric>_mean``
            column per eval metric in the criteria, ``stage_idx``,
            ``stage_name``, ``is_winner``, ``status``, ``candidate_id``,
            ``label``. This is the *summary* — one collapsed-to-means
            row per candidate.
        best_params: Parameter dict of the top-ranked candidate (lowest
            score). Empty if every candidate failed.
        best_score: Primary score of the top-ranked candidate.
            ``float('inf')`` if every candidate failed.
        best_eval_df: Full per-group eval frame for the top-ranked
            candidate — one row per ``EvalCriteria.group_by`` combination
            (typically ``(metric_id, horizon_step)``) and one column per
            metric in :attr:`EvalCriteria.eval_metrics` (mape, smape,
            medae, mae, rmse, …) plus the ``n`` row-count column. Use
            this to inspect per-horizon error curves without re-loading
            the on-disk ``evals/<id>.csv``. ``None`` if every candidate
            failed or the run was predict-only (no eval criteria).
        output_dir: Root directory where the run wrote its artifacts.
        method: Name of the method that produced this result
            (``"grid"`` or ``"grouped"``).
        stage_winners: For grouped runs, the cumulative parameter dict
            after each stage (stage 1 winner, stage 1+2 winners, ...).
            Empty for grid.
        search_space: The SearchSpace that was swept (None when the
            result is constructed by :func:`evaluate_existing`).
        backfill_config: The template BackfillConfig used by every
            candidate (None when the result is constructed by
            :func:`evaluate_existing`).
    """

    results_df: pd.DataFrame
    """One row per candidate; sorted ascending by ``score``."""

    best_params: Dict[str, Any]
    """Parameter dict of the top-ranked candidate."""

    best_score: float
    """Primary score of the top-ranked candidate (lower = better)."""

    output_dir: Path
    """Root directory containing candidates_log, predictions, results."""

    method: str
    """Method name: ``"grid"`` or ``"grouped"``."""

    stage_winners: List[Dict[str, Any]] = field(default_factory=list)
    """Cumulative winning params after each grouped stage; empty for grid."""

    best_eval_df: Optional[pd.DataFrame] = None
    """Full per-group eval frame for the top-ranked candidate (every
    metric in :attr:`EvalCriteria.eval_metrics`, one row per
    ``group_by`` combination). ``None`` for predict-only runs and when
    every candidate failed."""

    search_space: Optional["SearchSpace"] = None
    """The :class:`SearchSpace` the run swept. Surfaced for the report so
    it can show what grid was searched (per-stage for Grouped, flat for
    Grid). ``None`` for results constructed by
    :func:`evaluate_existing`, which doesn't re-run any sweep."""

    backfill_config: Optional[BackfillConfig] = None
    """The template :class:`BackfillConfig` used by every candidate.
    Surfaced for the report so the cutoff schedule (``horizon``,
    ``n_windows``, ``step``, ``initial_train_size`` / explicit
    ``cutoffs``, ``window_type``) is visible alongside the search
    space. ``None`` for results constructed by :func:`evaluate_existing`."""

    param_converter: Optional["ParamConverter"] = None
    """The :class:`ParamConverter` used to translate flat search-space
    params into algo-specific config. Surfaced for the report so it
    can dump ``param_converter.convert({})`` — the default-resolved
    algo_params template — alongside the search space, making the
    report self-contained even for converter-driven scripts where
    ``backfill_config.forecast_config.algo_params`` is intentionally
    empty (the converter supplies everything per-candidate).
    ``None`` for results constructed by :func:`evaluate_existing`."""


@dataclass
class ModelSelection:
    """Abstract base for model-selection methods.

    Subclasses implement :meth:`run` to choose which candidates to
    evaluate and in what order. They reuse :meth:`predict_and_persist`
    for the common prediction-plus-write loop and
    :meth:`evaluate_candidates` to compute metrics over the persisted
    output.

    Uses ``@dataclass`` + ``@abstractmethod`` (no ``ABC`` base), matching
    :class:`~abvelocity.ts.algo.base.TSAlgo` and the
    :class:`~abvelocity.stats.estimator.Estimator` pattern.

    Attributes:
        search_space: Defines which parameter overrides to evaluate.
        backfill_config: Template prediction spec. Its inner
            :attr:`forecast_config.algo_params` is the dict that
            candidate overrides patch into.
        output_dir: Root directory for all per-run artifacts.
        eval_criteria: Optional for grid (eval can be deferred); required
            for grouped (needed to pick winners between stages).
        anomaly_df: Optional known-anomaly intervals forwarded to every
            :class:`~abvelocity.ts.backfill.runner.BackfillRunner`
            invocation.
        verbose: If ``True``, print one progress line per candidate. Errors
            and warnings always log through the standard ``logging`` framework
            regardless of this flag.
        n_jobs: Number of joblib workers for the per-candidate predict
            loop. ``1`` (default) runs serially in-process; ``>= 2``
            spawns that many workers; ``-1`` uses all cores; ``-2`` uses
            all but one. See the field-level docstring on :attr:`n_jobs`
            for the oversubscription guidance.
    """

    search_space: SearchSpace
    """SearchSpace declaring which params to override per candidate."""

    backfill_config: BackfillConfig
    """Template :class:`BackfillConfig` patched per candidate."""

    output_dir: Path
    """Root output directory; candidates_log + predictions + results live here."""

    eval_criteria: Optional[EvalCriteria] = None
    """Optional for grid, required for grouped."""

    param_converter: Optional[ParamConverter] = None
    """Optional adapter that translates flat search-space params into the
    algo's ``algo_params`` shape (e.g. nested dicts for greykite). When
    ``None``, candidate params are merged into the template's
    ``algo_params`` verbatim (correct for algos with flat config). The
    audit trail (``model_candidates.csv``, label, candidate_id) always
    uses the flat search-space form regardless of whether a converter
    is set."""

    anomaly_df: Optional[pd.DataFrame] = None
    """Optional anomaly intervals forwarded to every BackfillRunner call."""

    verbose: bool = True
    """If True, print one progress line per candidate."""

    n_jobs: int = 1
    """Number of parallel workers (joblib semantics).

    * ``1`` (default) — serial; no joblib involvement.
    * ``N >= 2`` — exactly ``N`` worker processes.
    * ``-1`` — use all available CPU cores.
    * ``-2`` — use all but one core (good default for laptops).

    For algos that already do internal parallelism (e.g. greykite's
    Silverkite uses sklearn's ``n_jobs`` for its 3-fold CV), pin the
    algo's internal parallelism to 1 to avoid oversubscription. For
    greykite, set ``algo_params["computation"] = {"n_jobs": 1}`` on
    the template ``BackfillConfig.forecast_config``.
    """

    method: str = "base"
    """Subclass-set method name; surfaced in the SelectionResult."""

    def __post_init__(self) -> None:
        if not isinstance(self.search_space, SearchSpace):
            raise ValueError(f"search_space must be a SearchSpace; got {type(self.search_space).__name__}.")
        if not isinstance(self.backfill_config, BackfillConfig):
            raise ValueError(f"backfill_config must be a BackfillConfig; got {type(self.backfill_config).__name__}.")
        if self.eval_criteria is not None and not isinstance(self.eval_criteria, EvalCriteria):
            raise ValueError(f"eval_criteria must be EvalCriteria or None; got {type(self.eval_criteria).__name__}.")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero (>=1 for an explicit count, -1 for all cores, -2 for all-but-one — joblib convention).")
        self.output_dir = Path(self.output_dir)

    @abstractmethod
    def run(self, df: pd.DataFrame) -> SelectionResult:
        """Execute the selection method end-to-end.

        Args:
            df: Long- or wide-format DataFrame with the prepped time-series
                data. Forwarded to :class:`BackfillRunner.run` per
                candidate. Must contain :attr:`backfill_config.forecast_config.time_col`
                and every column in
                :attr:`backfill_config.forecast_config.value_cols`.

        Returns:
            :class:`SelectionResult` with all candidates ranked.
        """
        ...

    def predict_and_persist(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        candidates_log: ModelCandidatesLog,
        stage_idx: int = -1,
        stage_name: str = "",
        skip_if_cached: bool = True,
    ) -> ModelCandidate:
        """Predict one candidate, write its prediction CSV, append a candidates-log candidate.

        Caching behaviour
        -----------------
        ``params`` is hashed to a stable :func:`compute_candidate_id` (12
        hex chars of SHA-256 over canonical-JSON of the dict). When
        ``skip_if_cached=True`` (default) and a candidate with the same id is
        already in :attr:`candidates_log` with ``status='ok'``, this
        method returns the cached :class:`ModelCandidate` immediately —
        no prediction is re-run, no candidate is appended. This makes a re-run
        of the same sweep effectively free (it just reads the existing
        log) and lets a crashed sweep resume from where it left off.

        ``status='error'`` and ``status='skipped'`` rows are NOT cached —
        they trigger a re-run (the previous failure may have been due to
        a transient infra issue). To force re-prediction for an ``ok``
        candidate, pass ``skip_if_cached=False`` or wipe the
        ``predictions/<id>.csv`` from disk first.

        Args:
            params: Parameter override dict for this candidate. Patches
                :attr:`backfill_config.forecast_config.algo_params`.
                Expected to be flat (``{key: scalar}``) — see
                :func:`~abvelocity.ts.model_selection.model_candidates.compute_candidate_id`
                for the rationale; nested algo configs come from the
                :class:`ParamConverter` at predict time.
            df: Prepped input DataFrame for the predictor.
            candidates_log: Mutable :class:`ModelCandidatesLog`; the new candidate is appended
                and the candidates_log is flushed to disk.
            stage_idx: 0-based stage index (grouped); ``-1`` for grid.
            stage_name: Stage name (grouped); empty for grid.
            skip_if_cached: If ``True`` and the candidate's id is already
                in the candidates_log with ``status='ok'``, skip prediction and
                reuse the cached candidate. Default ``True``.

        Returns:
            The :class:`ModelCandidate` recorded for this candidate.
        """
        candidate_id = compute_candidate_id(params)
        label = format_label(params)

        if skip_if_cached:
            cached = candidates_log.existing_ids().get(candidate_id)
            if cached is not None:
                if self.verbose:
                    print(f"\n*** ModelSelection cached {label}  (id={candidate_id})")
                return cached

        candidate_row = self.predict_one(params=params, df=df, stage_idx=stage_idx, stage_name=stage_name)
        candidates_log.append(candidate_row)
        candidates_log.flush()
        if self.verbose and candidate_row.status == "ok":
            print(f"\n*** ModelSelection ran {label}  (id={candidate_id})")
        return candidate_row

    def predict_one(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        stage_idx: int = -1,
        stage_name: str = "",
    ) -> ModelCandidate:
        """Run one candidate end-to-end, write its prediction CSV (and fit_info JSON).

        This is the parallelism-safe primitive: it does NOT touch the
        candidates_log (no shared file writes), so it can be invoked from
        worker processes. Caching and candidates-log appends remain in
        :meth:`predict_and_persist`, which calls this method.

        Args:
            params: Parameter override dict for this candidate.
            df: Prepped input DataFrame for the predictor.
            stage_idx: 0-based stage index (grouped); ``-1`` for grid.
            stage_name: Stage name (grouped); empty for grid.

        Returns:
            A :class:`ModelCandidate` recording the result. ``status="ok"``
            when the prediction CSV was written; ``status="error"`` with
            a truncated traceback in ``error`` otherwise.
        """
        candidate_id = compute_candidate_id(params)
        label = format_label(params)
        merged_config = self.merge_params(params)

        try:
            result = BackfillRunner(merged_config).run(df=df, anomaly_df=self.anomaly_df)
            if result.result_df is None or result.result_df.empty:
                candidate_row = ModelCandidate(
                    candidate_id=candidate_id,
                    label=label,
                    params=params,
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                    status="error",
                    error="BackfillRunner produced empty result_df",
                )
            else:
                predict_path = predictions_path_for(candidate_id, output_dir=self.output_dir)
                (self.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
                # CSV (not parquet/pickle) so the per-candidate output is pure-Python,
                # cat-friendly, and free of binary deps. The frame is small (one candidate per
                # cutoff × metric_id × horizon_step) so the size penalty is negligible.
                result.result_df.to_csv(predict_path, index=False)

                fit_info_rel = ""
                if result.fit_info:
                    fit_path = fit_info_path_for(candidate_id, output_dir=self.output_dir)
                    (self.output_dir / "fits").mkdir(parents=True, exist_ok=True)
                    with open(fit_path, "w") as fh:
                        json.dump(result.fit_info, fh, default=str, indent=2)
                    fit_info_rel = str(fit_info_path_for(candidate_id, output_dir=None))

                candidate_row = ModelCandidate(
                    candidate_id=candidate_id,
                    label=label,
                    params=params,
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                    status="ok",
                    predict_path=str(predictions_path_for(candidate_id, output_dir=None)),
                    fit_info_path=fit_info_rel,
                )
        except Exception as exc:
            # ``Exception`` (not ``BaseException``) — KeyboardInterrupt /
            # SystemExit / GeneratorExit must still propagate so a user can
            # cancel a long sweep. Programming errors like AttributeError
            # are intentionally caught and recorded per-candidate: when a
            # model_template + algo_params combination fails to fit (which
            # third-party algos express as a wide variety of exception
            # types — greykite alone raises numpy/scipy/patsy/sklearn
            # internals depending on the failure mode), the sweep
            # continues and the candidate's ``error`` column lets the user see
            # which combination broke and why.
            #
            # NOTE: avoid traceback.format_exc() — Python's TracebackException
            # calls list(frame.f_locals) for did-you-mean suggestions, and
            # patsy's eval namespace raises KeyError(0) on iteration. That
            # crashes the error path before we can record anything. Capture
            # type+message only; full traceback is gone but it's enough to
            # diagnose with the params dict that's also persisted.
            err_msg = f"{type(exc).__name__}: {exc}"[:ERROR_MSG_TRUNCATION]
            candidate_row = ModelCandidate(
                candidate_id=candidate_id,
                label=label,
                params=params,
                stage_idx=stage_idx,
                stage_name=stage_name,
                status="error",
                error=err_msg,
            )
            # Always log the error (not gated on ``verbose``) so
            # programming errors don't disappear into the candidates log.
            logger.error("ModelSelection ERROR %s: %s", label, err_msg)

        return candidate_row

    def write_cutoffs_log(self, df: pd.DataFrame) -> Path:
        """Persist the cutoff dates this run will use to ``output_dir/cutoffs.json``.

        Resolves the cutoffs via :meth:`BackfillRunner.resolve_cutoff_dates`,
        which honors :attr:`BackfillConfig.cutoffs` (mode B) or the
        algorithmic ``initial_train_size``/``step``/``n_windows`` spec
        (mode A). Either way, the resulting list of dates is the same
        format on disk, so downstream tooling doesn't need to know which
        mode produced them.

        Args:
            df: Prepped input DataFrame (used to resolve dates from
                indices when in algorithmic mode).

        Returns:
            Path to the written JSON file.
        """
        dates = BackfillRunner(self.backfill_config).resolve_cutoff_dates(df)
        out_path = self.output_dir / CUTOFFS_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "mode": "explicit" if self.backfill_config.cutoffs else "algorithmic",
            "cutoffs": [pd.Timestamp(d).date().isoformat() for d in dates],
            "horizon": self.backfill_config.horizon,
            "window_type": self.backfill_config.window_type,
        }
        if not self.backfill_config.cutoffs:
            body["initial_train_size"] = self.backfill_config.initial_train_size
            body["step"] = self.backfill_config.step
            body["n_windows"] = self.backfill_config.n_windows
        out_path.write_text(json.dumps(body, indent=2))
        return out_path

    def merge_params(self, params: Dict[str, Any]) -> BackfillConfig:
        """Return a clone of :attr:`backfill_config` with ``params`` merged.

        Patches :attr:`backfill_config.forecast_config.algo_params` only —
        per the locked v1 design, the search space cannot override
        top-level forecast / backfill fields. To sweep those, construct
        multiple :class:`ModelSelection` instances.

        Args:
            params: Parameter override dict for one candidate.

        Returns:
            New :class:`BackfillConfig` with patched ``algo_params``.
        """
        fc = self.backfill_config.forecast_config
        merged_algo_params: Dict[str, Any] = dict(fc.algo_params or {})
        translated = self.param_converter(params) if self.param_converter is not None else params
        merged_algo_params.update(translated)
        new_fc = replace(fc, algo_params=merged_algo_params)
        return replace(self.backfill_config, forecast_config=new_fc)

    def evaluate_candidates(
        self,
        candidates_log: ModelCandidatesLog,
        criteria: Optional[EvalCriteria] = None,
    ) -> pd.DataFrame:
        """Compute per-candidate scores for every ``status='ok'`` candidate.

        For each candidate: read the cached per-candidate eval CSV if it exists
        and contains every metric in :attr:`criteria.eval_metrics`;
        otherwise read the prediction CSV, run
        :meth:`EvalCriteria.evaluate_metrics`, and persist the new eval
        CSV. Reduce to a per-candidate record (params + score + per-metric mean
        + bookkeeping).

        The cache makes repeat calls (e.g. grouped's per-stage + final
        global passes) cheap: only newly-predicted candidates incur the
        compute cost; prior-stage candidates' metrics are read straight
        from disk.

        Args:
            candidates_log: ModelCandidatesLog with rows already persisted.
            criteria: Override criteria for this evaluation. Defaults to
                :attr:`eval_criteria`. At least one must be set.

        Returns:
            DataFrame ranked ascending by ``score``. One candidate per
            candidates-log entry; failed/skipped candidates still appear
            with ``score=NaN`` so the audit trail is preserved.

        Raises:
            ValueError: If neither ``criteria`` nor :attr:`eval_criteria`
                is set.
        """
        eval_criteria = criteria if criteria is not None else self.eval_criteria
        if eval_criteria is None:
            raise ValueError("Either pass `criteria=` or construct ModelSelection with eval_criteria set.")

        records: List[Dict[str, Any]] = []
        any_row_freshly_evaluated = False
        for candidate_row in candidates_log.rows:
            base_record: Dict[str, Any] = {
                **candidate_row.params,
                "candidate_id": candidate_row.candidate_id,
                "label": candidate_row.label,
                "stage_idx": candidate_row.stage_idx,
                "stage_name": candidate_row.stage_name,
                "is_winner": candidate_row.is_winner,
                "status": candidate_row.status,
                "error": candidate_row.error,
            }
            if candidate_row.status != "ok" or not candidate_row.predict_path:
                base_record["score"] = eval_criteria.worst_score()
                for metric_name in eval_criteria.eval_metrics:
                    base_record[f"{metric_name}_mean"] = float("nan")
                records.append(base_record)
                continue

            metrics_df = self._load_cached_metrics(candidates_log, candidate_row, eval_criteria)
            if metrics_df is None:
                predict_path = candidates_log.output_dir / candidate_row.predict_path
                prediction_df = pd.read_csv(predict_path)
                metrics_df = eval_criteria.evaluate_metrics(prediction_df)
                # Persist per-candidate eval CSV so each model tried has both its
                # backfill predictions and its eval frame stored on disk.
                candidates_log.evals_dir.mkdir(parents=True, exist_ok=True)
                eval_csv_path = eval_path_for(candidate_row.candidate_id, output_dir=candidates_log.output_dir)
                metrics_df.to_csv(eval_csv_path, index=False)
                candidate_row.eval_path = str(eval_path_for(candidate_row.candidate_id, output_dir=None))
                any_row_freshly_evaluated = True

            base_record["score"] = eval_criteria.primary_score(metrics_df)
            for metric_name in eval_criteria.eval_metrics:
                if metric_name in metrics_df.columns:
                    base_record[f"{metric_name}_mean"] = float(metrics_df[metric_name].mean())
                else:
                    base_record[f"{metric_name}_mean"] = float("nan")

            records.append(base_record)

        # Only re-flush the candidates log if at least one candidate's eval_path was newly populated.
        if any_row_freshly_evaluated:
            candidates_log.flush()

        results_df = pd.DataFrame(records)
        if "score" in results_df.columns and not results_df.empty:
            results_df = results_df.sort_values(
                "score",
                kind="stable",
                ascending=eval_criteria.lower_is_better,
                na_position="last",
            ).reset_index(drop=True)
        return results_df

    @staticmethod
    def _load_best_eval_df(
        candidates_log: ModelCandidatesLog,
        best_candidate_id: str,
    ) -> Optional[pd.DataFrame]:
        """Load the per-group eval frame for the best candidate.

        Reads ``evals/<best_candidate_id>.csv`` so the SelectionResult can
        expose the full per-(``metric_id``, ``horizon_step``) × eval-metric
        matrix without forcing the caller to touch disk. Returns ``None``
        if the file is missing or unreadable (recompute paths handle the
        latter via the cache-rejection logic in
        :meth:`_load_cached_metrics`).

        Args:
            candidates_log: For its ``output_dir``.
            best_candidate_id: 12-char id of the winning candidate.

        Returns:
            DataFrame from ``evals/<id>.csv``, or ``None``.
        """
        if not best_candidate_id:
            return None
        for candidate_row in candidates_log.rows:
            if candidate_row.candidate_id != best_candidate_id:
                continue
            if not candidate_row.eval_path:
                return None
            full_path = candidates_log.output_dir / candidate_row.eval_path
            if not full_path.exists():
                return None
            try:
                return pd.read_csv(full_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError):
                return None
        return None

    @staticmethod
    def _load_cached_metrics(
        candidates_log: ModelCandidatesLog,
        candidate_row: ModelCandidate,
        eval_criteria: EvalCriteria,
    ) -> Optional[pd.DataFrame]:
        """Return cached per-candidate metrics if the eval CSV is fresh enough.

        "Fresh enough" means: ``candidate_row.eval_path`` is set, the file exists,
        and the cached frame has every column in
        ``eval_criteria.eval_metrics``. Otherwise return ``None`` to
        signal a recompute is needed.

        Args:
            candidates_log: For its ``output_dir`` (eval paths are
                stored relative to it).
            candidate_row: The candidate row.
            eval_criteria: For checking column-completeness of the cache.

        Returns:
            DataFrame from the cached eval CSV, or ``None`` to recompute.
        """
        if not candidate_row.eval_path:
            return None
        eval_csv_path = candidates_log.output_dir / candidate_row.eval_path
        if not eval_csv_path.exists():
            return None
        try:
            cached = pd.read_csv(eval_csv_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
            # Corrupt / truncated eval CSVs are not fatal — just recompute.
            # But surface the rejection so a corrupt cache doesn't silently
            # masquerade as "fresh recompute" forever.
            logger.warning(
                "ModelSelection: rejecting corrupt cached eval CSV %s: %s: %s",
                eval_csv_path,
                type(exc).__name__,
                exc,
            )
            return None
        missing = [m for m in eval_criteria.eval_metrics if m not in cached.columns]
        if missing:
            return None
        return cached


def evaluate_existing(
    output_dir: Path,
    eval_criteria: EvalCriteria,
) -> pd.DataFrame:
    """Re-rank a previously-completed selection run with new criteria.

    No predictions are re-run — the function reads
    ``output_dir/model_candidates.csv`` and the per-candidate prediction
    CSVs, then applies ``eval_criteria`` to each. Useful when you want to
    swap the primary eval metric, change the group_by, or inspect a
    different reduction without paying the prediction cost again.

    Args:
        output_dir: Root output directory of a prior run.
        eval_criteria: Criteria to apply.

    Returns:
        Same shape as :meth:`ModelSelection.evaluate_candidates`. Sorted
        ascending by ``score``.

    Raises:
        FileNotFoundError: If ``output_dir/model_candidates.csv`` does not exist.
    """
    candidates_log = ModelCandidatesLog.load(Path(output_dir))
    if not candidates_log.rows:
        raise FileNotFoundError(
            f"No model_candidates.csv found at {output_dir / 'model_candidates.csv'}; nothing to re-evaluate."
        )
    helper = ReEvalSelection(
        search_space=SearchSpace.flat({"placeholder": [None]}),
        backfill_config=None,
        output_dir=Path(output_dir),
        eval_criteria=eval_criteria,
    )
    return helper.evaluate_candidates(candidates_log, criteria=eval_criteria)


@dataclass
class ReEvalSelection(ModelSelection):
    """Lightweight subclass used internally by :func:`evaluate_existing`.

    No predictions are run — only
    :meth:`ModelSelection.evaluate_candidates` is invoked over an
    already-persisted candidates log. The parent's ``isinstance`` check
    on :attr:`backfill_config` is therefore skipped (re-eval has no
    template to validate).
    """

    method: str = "evaluate_existing"

    def __post_init__(self) -> None:
        # Skip the parent's BackfillConfig validation — re-eval has no template.
        self.output_dir = Path(self.output_dir)

    def run(self, df: pd.DataFrame) -> SelectionResult:
        """Not supported; :class:`ReEvalSelection` is for re-eval only.

        Args:
            df: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("ReEvalSelection.run is not supported; use evaluate_existing.")

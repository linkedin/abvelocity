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
"""Per-candidate registry (``model_candidates.csv``) and stable id hashing.

Every candidate written to the model-selection ``output_dir`` produces:

* one prediction CSV at ``output_dir/predictions/<candidate_id>.csv``
  containing the long-format backtest result. CSV (not parquet/pickle)
  so it's pure-Python, human-readable, ``cat``-friendly, and avoids any
  pyarrow dependency;
* one row in ``output_dir/model_candidates.csv`` encoding the candidate's identity
  (params dict as JSON), its stage info for grouped runs, and its persistence
  status.

``model_candidates.csv`` is the source of truth for "what was tried." Two re-runs
that hit the same ``params`` produce the same ``candidate_id``, so the
second run can skip the prediction step and only re-evaluate. The internal
:class:`ModelCandidatesLog` class wraps that table — the name is a code-level
abstraction; the artifact filename is ``model_candidates.csv``.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PREDICTIONS_SUBDIR = "predictions"
FITS_SUBDIR = "fits"
EVALS_SUBDIR = "evals"
MODEL_CANDIDATES_FILENAME = "model_candidates.csv"
"""Filename of the per-candidate registry CSV (one row per model tried)."""

STAGE_WINNERS_FILENAME = "stage_winners.json"

CANDIDATE_ID_LEN = 12
"""Number of leading hex chars from SHA-256 used as the candidate id."""

MODEL_CANDIDATES_COLUMNS: List[str] = [
    "candidate_id",
    "label",
    "params",
    "stage_idx",
    "stage_name",
    "is_winner",
    "status",
    "predict_path",
    "fit_info_path",
    "eval_path",
    "error",
]
"""Columns present in every model_candidates CSV. ``params`` is JSON-encoded."""


@dataclass
class ModelCandidate:
    """In-memory representation of one candidates_log row.

    Attributes:
        candidate_id: Stable content hash of :attr:`params` (12 hex chars).
        label: Human-readable params signature (sorted ``"k=v ..."``).
        params: Parameter override dict for this candidate.
        stage_idx: 0-based stage index for grouped runs; ``-1`` for grid.
        stage_name: Stage name for grouped runs; empty string for grid.
        is_winner: ``True`` for the winning candidate of its stage in a
            grouped run; always ``False`` in a grid run (winners are
            determined post-hoc by sorting ``results.csv``).
        status: ``"ok"`` (predict succeeded), ``"error"`` (predictor
            raised), or ``"skipped"`` (cached from a prior run).
        predict_path: Relative path inside ``output_dir`` to the
            prediction CSV, e.g. ``"predictions/<id>.csv"``.
            Empty when ``status="error"``.
        fit_info_path: Relative path inside ``output_dir`` to a JSON
            sidecar holding ``BackfillResult.fit_info`` for this
            candidate, e.g. ``"fits/<id>.json"``. Empty when no fit
            info was emitted by the predictor or when ``status='error'``.
            v1 ``EvalCriteria`` ignores this; v2 will use it for
            likelihood-based metrics (AIC / BIC / DIC).
        error: Empty when ``status="ok"``; truncated traceback message
            otherwise.
    """

    candidate_id: str
    """Stable content hash of params (first 12 hex chars of SHA-256)."""

    label: str
    """``"key=value key=value ..."`` ordered by sorted key."""

    params: Dict[str, Any]
    """Parameter override dict for this candidate."""

    stage_idx: int = -1
    """0-based stage index (grouped); ``-1`` for grid."""

    stage_name: str = ""
    """Stage name (grouped); empty for grid."""

    is_winner: bool = False
    """True if this candidate won its grouped stage; always False for grid."""

    status: str = "ok"
    """``"ok"`` / ``"error"`` / ``"skipped"``."""

    predict_path: str = ""
    """Relative path to prediction CSV; empty when status='error'."""

    fit_info_path: str = ""
    """Relative path to fit_info JSON sidecar; empty when no fit_info or status='error'."""

    eval_path: str = ""
    """Relative path to per-candidate eval CSV; populated by
    :meth:`~abvelocity.ts.model_selection.base.ModelSelection.evaluate_candidates`
    after a successful prediction. Empty until eval has run."""

    error: str = ""
    """Truncated traceback when status='error'; empty otherwise."""

    def to_record(self) -> Dict[str, Any]:
        """Serialise this row for CSV writing.

        Returns:
            Dict with one entry per :data:`MODEL_CANDIDATES_COLUMNS`.
            ``params`` is encoded as a sorted-keys JSON string.
        """
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "params": json.dumps(self.params, sort_keys=True, default=str),
            "stage_idx": self.stage_idx,
            "stage_name": self.stage_name,
            "is_winner": self.is_winner,
            "status": self.status,
            "predict_path": self.predict_path,
            "fit_info_path": self.fit_info_path,
            "eval_path": self.eval_path,
            "error": self.error,
        }


@dataclass
class ModelCandidatesLog:
    """In-memory candidates_log plus IO helpers.

    A candidates_log is a list of :class:`ModelCandidate` s. The class is mutable —
    selectors append to :attr:`rows` and call :meth:`flush` after each
    candidate so that a crash mid-sweep still leaves a usable candidates_log on
    disk.

    Attributes:
        output_dir: Root directory; ``candidates_log.csv`` lives at the top.
        rows: All rows accumulated so far.
    """

    output_dir: Path
    """Root output directory; ``candidates_log.csv`` lives at the top."""

    rows: List[ModelCandidate] = field(default_factory=list)
    """All rows accumulated so far."""

    @property
    def model_candidates_path(self) -> Path:
        """Path to the model_candidates CSV (``output_dir/model_candidates.csv``)."""
        return self.output_dir / MODEL_CANDIDATES_FILENAME

    @property
    def predictions_dir(self) -> Path:
        """Path to the predictions sub-directory."""
        return self.output_dir / PREDICTIONS_SUBDIR

    @property
    def fits_dir(self) -> Path:
        """Path to the fits sub-directory (fit_info JSON sidecars)."""
        return self.output_dir / FITS_SUBDIR

    @property
    def evals_dir(self) -> Path:
        """Path to the evals sub-directory (per-candidate eval CSVs)."""
        return self.output_dir / EVALS_SUBDIR

    def append(self, row: ModelCandidate) -> None:
        """Add a row to the in-memory candidates_log. Does not write to disk."""
        self.rows.append(row)

    def existing_ids(self) -> Dict[str, ModelCandidate]:
        """Return ``{candidate_id: row}`` for rows already in this candidates_log.

        Used by selectors to skip re-prediction on resume. Only
        ``status='ok'`` rows are considered cacheable; ``error`` /
        ``skipped`` rows trigger a re-run.

        Returns:
            Mapping from candidate id to the latest matching row.
        """
        out: Dict[str, ModelCandidate] = {}
        for row in self.rows:
            if row.status == "ok":
                out[row.candidate_id] = row
        return out

    def flush(self) -> None:
        """Write the model_candidates CSV. Overwrites any existing file.

        Creates :attr:`predictions_dir` and :attr:`output_dir` if needed.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([r.to_record() for r in self.rows], columns=MODEL_CANDIDATES_COLUMNS)
        df.to_csv(self.model_candidates_path, index=False)

    @classmethod
    def load(cls, output_dir: Path) -> "ModelCandidatesLog":
        """Read a model_candidates CSV back into a :class:`ModelCandidatesLog` instance.

        Args:
            output_dir: Directory containing ``model_candidates.csv``.

        Returns:
            :class:`ModelCandidatesLog` with rows reconstructed from disk. Empty
            (no rows) if the file does not exist yet.
        """
        candidates_log = cls(output_dir=Path(output_dir))
        if not candidates_log.model_candidates_path.exists():
            return candidates_log
        df = pd.read_csv(candidates_log.model_candidates_path, dtype={"candidate_id": str})
        # Drop duplicate candidate_id rows on read, keeping the most recent.
        # Errors aren't cached (only ``status="ok"`` is), so an error candidate
        # re-run on resume gets a second row appended. ``keep="last"`` carries
        # the freshest result forward (typically the same error, but the
        # newest stack trace) and prevents the report from duplicating rows.
        df = df.drop_duplicates(subset=["candidate_id"], keep="last").reset_index(drop=True)
        for _, record in df.iterrows():
            candidates_log.rows.append(
                ModelCandidate(
                    candidate_id=str(record["candidate_id"]),
                    label=str(record["label"]),
                    params=json.loads(record["params"]) if pd.notna(record["params"]) else {},
                    stage_idx=int(record["stage_idx"]) if pd.notna(record["stage_idx"]) else -1,
                    stage_name=str(record["stage_name"]) if pd.notna(record["stage_name"]) else "",
                    is_winner=bool(record["is_winner"]) if pd.notna(record["is_winner"]) else False,
                    status=str(record["status"]) if pd.notna(record["status"]) else "ok",
                    predict_path=str(record["predict_path"]) if pd.notna(record["predict_path"]) else "",
                    fit_info_path=(
                        str(record["fit_info_path"])
                        if "fit_info_path" in record and pd.notna(record["fit_info_path"])
                        else ""
                    ),
                    eval_path=(
                        str(record["eval_path"])
                        if "eval_path" in record and pd.notna(record["eval_path"])
                        else ""
                    ),
                    error=str(record["error"]) if pd.notna(record["error"]) else "",
                )
            )
        return candidates_log


def compute_candidate_id(params: Dict[str, Any]) -> str:
    """Return a stable 12-char content hash for a params dict.

    The hash is the first 12 hex chars of SHA-256 over the canonical-JSON
    encoding of ``params`` (sorted keys, ``str`` fallback for non-JSON
    natives like NumPy scalars). Stable across processes and Python
    versions. SHA-256 is used (not SHA-1) per "no insecure hashes in
    source" guidance, even though we're not using
    the digest as a cryptographic signature here — collision-resistance
    on the candidate id is still desirable so two distinct param dicts
    don't accidentally map to the same on-disk slot.

    Expected params shape
    ---------------------
    Flat ``{key: scalar}`` only at this layer. The model-selection
    framework treats every param key as something a user wants to see
    individually in ``model_candidates.csv`` and ``results.csv`` — i.e.
    one column per key. Nested algo-specific configs (e.g. greykite's
    ``model_components.changepoints.changepoints_dict.regularization_strength``)
    must be flattened by the user-visible search space (e.g. as
    ``changepoint_reg``) and translated to the nested shape at predict
    time by a
    :class:`~abvelocity.ts.model_selection.param_converter.ParamConverter`.
    The hash + JSON serialisation work for nested values too, but the
    audit-trail columns and label become unreadable.

    Args:
        params: Flat parameter override dict.

    Returns:
        12-character lowercase hex string.
    """
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:CANDIDATE_ID_LEN]


def format_label(params: Dict[str, Any]) -> str:
    """Return a human-readable label for a params dict.

    The label is deterministic across runs (sorted by key) so that two
    runs with the same params produce the same label. Used in printed
    progress output and as the leading column in every report table.

    As with :func:`compute_candidate_id`, this assumes ``params`` is a
    flat dict; nested values render as their ``repr`` and produce ugly
    labels.

    Args:
        params: Flat parameter override dict.

    Returns:
        ``"k1=v1 k2=v2 ..."`` ordered by sorted key.
    """
    return " ".join(f"{k}={params[k]!r}" for k in sorted(params))


def predictions_path_for(candidate_id: str, output_dir: Optional[Path] = None) -> Path:
    """Return the prediction CSV path for a candidate id.

    Args:
        candidate_id: 12-char hex id.
        output_dir: Optional output root; if ``None``, returns a relative
            path (``predictions/<id>.csv``) suitable for the candidates_log's
            ``predict_path`` column.

    Returns:
        :class:`pathlib.Path`.
    """
    rel = Path(PREDICTIONS_SUBDIR) / f"{candidate_id}.csv"
    if output_dir is None:
        return rel
    return output_dir / rel


def fit_info_path_for(candidate_id: str, output_dir: Optional[Path] = None) -> Path:
    """Return the fit_info JSON sidecar path for a candidate id.

    Args:
        candidate_id: 12-char hex id.
        output_dir: Optional output root; if ``None``, returns a relative
            path (``fits/<id>.json``) suitable for the candidates_log's
            ``fit_info_path`` column.

    Returns:
        :class:`pathlib.Path`.
    """
    rel = Path(FITS_SUBDIR) / f"{candidate_id}.json"
    if output_dir is None:
        return rel
    return output_dir / rel


def eval_path_for(candidate_id: str, output_dir: Optional[Path] = None) -> Path:
    """Return the per-candidate eval CSV path for a candidate id.

    The CSV holds the per-(``metric_id``, ``horizon_step``) metrics frame
    produced by :meth:`EvalCriteria.evaluate_metrics` for this candidate.
    One row per group, one column per eval metric.

    Args:
        candidate_id: 12-char hex id.
        output_dir: Optional output root; if ``None``, returns a relative
            path (``evals/<id>.csv``) suitable for the candidates_log's
            ``eval_path`` column.

    Returns:
        :class:`pathlib.Path`.
    """
    rel = Path(EVALS_SUBDIR) / f"{candidate_id}.csv"
    if output_dir is None:
        return rel
    return output_dir / rel

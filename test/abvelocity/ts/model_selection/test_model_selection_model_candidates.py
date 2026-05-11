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
"""Tests for the model-selection candidates_log module."""

from pathlib import Path

import pandas as pd
from abvelocity.ts.model_selection.model_candidates import (
    CANDIDATE_ID_LEN,
    ModelCandidatesLog,
    ModelCandidate,
    compute_candidate_id,
    fit_info_path_for,
    format_label,
    predictions_path_for,
)


def test_compute_candidate_id_stable_across_dict_orderings():
    a = compute_candidate_id({"yearly": 6, "reg": 0.4})
    b = compute_candidate_id({"reg": 0.4, "yearly": 6})
    assert a == b
    assert len(a) == CANDIDATE_ID_LEN
    assert all(c in "0123456789abcdef" for c in a)


def test_compute_candidate_id_different_for_different_params():
    a = compute_candidate_id({"reg": 0.4})
    b = compute_candidate_id({"reg": 0.6})
    assert a != b


def test_format_label_sorted_keys():
    label = format_label({"yearly": 6, "reg": 0.4, "ridge": 5})
    # Keys appear in alphabetical order.
    assert label == "reg=0.4 ridge=5 yearly=6"


def test_predictions_path_for_relative_and_absolute(tmp_path: Path):
    rel = predictions_path_for("abc123def456")
    assert str(rel) == "predictions/abc123def456.csv"
    abs_path = predictions_path_for("abc123def456", output_dir=tmp_path)
    assert abs_path == tmp_path / "predictions" / "abc123def456.csv"


def test_fit_info_path_for_relative_and_absolute(tmp_path: Path):
    rel = fit_info_path_for("abc123def456")
    assert str(rel) == "fits/abc123def456.json"
    abs_path = fit_info_path_for("abc123def456", output_dir=tmp_path)
    assert abs_path == tmp_path / "fits" / "abc123def456.json"


def test_candidates_log_round_trip(tmp_path: Path):
    candidates_log = ModelCandidatesLog(output_dir=tmp_path)
    candidates_log.append(
        ModelCandidate(
            candidate_id="aaa111aaa111",
            label="reg=0.4",
            params={"reg": 0.4},
            stage_idx=0,
            stage_name="trend",
            status="ok",
            predict_path="predictions/aaa111aaa111.csv",
        )
    )
    candidates_log.append(
        ModelCandidate(
            candidate_id="bbb222bbb222",
            label="reg=0.6",
            params={"reg": 0.6},
            status="error",
            error="boom",
        )
    )
    candidates_log.flush()

    # File written.
    assert candidates_log.model_candidates_path.exists()
    df = pd.read_csv(candidates_log.model_candidates_path)
    assert len(df) == 2
    assert set(df.columns) >= {"candidate_id", "label", "params", "status", "predict_path", "fit_info_path"}

    # Round-trip via load.
    reloaded = ModelCandidatesLog.load(tmp_path)
    assert len(reloaded.rows) == 2
    by_id = {r.candidate_id: r for r in reloaded.rows}
    assert by_id["aaa111aaa111"].params == {"reg": 0.4}
    assert by_id["aaa111aaa111"].status == "ok"
    assert by_id["bbb222bbb222"].status == "error"
    assert by_id["bbb222bbb222"].error == "boom"


def test_existing_ids_only_returns_ok_rows(tmp_path: Path):
    candidates_log = ModelCandidatesLog(output_dir=tmp_path)
    candidates_log.append(ModelCandidate(candidate_id="ok_id", label="x=1", params={"x": 1}, status="ok"))
    candidates_log.append(ModelCandidate(candidate_id="err_id", label="x=2", params={"x": 2}, status="error"))
    cached = candidates_log.existing_ids()
    assert "ok_id" in cached
    assert "err_id" not in cached


def test_load_returns_empty_when_no_file(tmp_path: Path):
    candidates_log = ModelCandidatesLog.load(tmp_path / "nope")
    assert candidates_log.rows == []

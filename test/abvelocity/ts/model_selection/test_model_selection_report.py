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
"""Tests for the model-selection HTML report writer."""

from pathlib import Path

import pandas as pd
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.constants import TIME_COL
from abvelocity.ts.model_selection.base import SelectionResult
from abvelocity.ts.model_selection.eval_criteria import EvalCriteria
from abvelocity.ts.model_selection.report import (
    REPORT_FILENAME,
    build_html,
    render_heat_cell,
    render_run_config,
    render_score_cell,
    write_report,
)
from abvelocity.ts.model_selection.space import ParamGroup, SearchSpace


def make_results_df():
    return pd.DataFrame(
        [
            {"k": 3, "score": 1.5, "mape_mean": 1.5, "smape_mean": 1.4, "status": "ok",
             "is_winner": True, "candidate_id": "aaa", "label": "k=3"},
            {"k": 2, "score": 2.7, "mape_mean": 2.7, "smape_mean": 2.5, "status": "ok",
             "is_winner": False, "candidate_id": "bbb", "label": "k=2"},
            {"k": 1, "score": float("inf"), "mape_mean": None, "smape_mean": None,
             "status": "error", "is_winner": False, "candidate_id": "ccc", "label": "k=1"},
        ]
    )


def make_selection_result(tmp_path: Path):
    return SelectionResult(
        results_df=make_results_df(),
        best_params={"k": 3},
        best_score=1.5,
        output_dir=tmp_path,
        method="grid",
    )


def test_write_report_creates_html_file(tmp_path: Path):
    selection_result = make_selection_result(tmp_path)
    criteria = EvalCriteria(eval_metrics=("mape", "smape"), primary_eval_metric="mape")
    out_path = write_report(selection_result, criteria, title="UnitTest")

    assert out_path == tmp_path / REPORT_FILENAME
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "<h1>UnitTest</h1>" in text
    # Best params surfaced in the summary block.
    assert "k=3" in text
    # Decision metric mentioned.
    assert "mape" in text
    # Winner row present.
    assert "winner" in text
    # Error row gets the error CSS class.
    assert "error" in text


def test_build_html_includes_all_metric_columns():
    df = make_results_df()
    selection_result = SelectionResult(
        results_df=df,
        best_params={"k": 3},
        best_score=1.5,
        output_dir=Path("/tmp/dummy"),
        method="grid",
    )
    criteria = EvalCriteria(eval_metrics=("mape", "smape"), primary_eval_metric="mape")
    html = build_html(selection_result, criteria, title="t")
    assert "mape_mean" in html
    assert "smape_mean" in html


def test_render_heat_cell_lower_is_better_picks_green_for_min():
    cell = render_heat_cell(val=1.0, min_max=(1.0, 5.0), lower_is_better=True)
    # Min value: green-dominant (high green channel, low red channel).
    assert "rgb(30,210,40)" in cell
    cell = render_heat_cell(val=5.0, min_max=(1.0, 5.0), lower_is_better=True)
    # Max value: red-dominant.
    assert "rgb(230,30,40)" in cell


def test_render_heat_cell_handles_nan():
    cell = render_heat_cell(val=float("nan"), min_max=(1.0, 5.0), lower_is_better=True)
    assert cell == "<td>—</td>"


def test_render_score_cell_handles_inf():
    cell = render_score_cell(float("inf"))
    assert "—" in cell
    finite = render_score_cell(2.5)
    assert "2.500000" in finite


def _backfill_config():
    return BackfillConfig(
        forecast_config=ForecastConfig(
            time_col=TIME_COL,
            value_cols=("y",),
            freq="D",
            forecast_horizon=21,
            algo_name="greykite",
            algo_params={},
        ),
        initial_train_size=1095,
        horizon=21,
        step=29,
        n_windows=12,
    )


def test_render_run_config_lists_search_space_and_backfill():
    space = SearchSpace.flat({"changepoint_reg": [0.7, 0.8], "regression_weight_col": ["ct1", "ct2"]})
    sr = SelectionResult(
        results_df=make_results_df(),
        best_params={"changepoint_reg": 0.7, "regression_weight_col": "ct1"},
        best_score=1.5,
        output_dir=Path("/tmp/dummy"),
        method="grid",
        search_space=space,
        backfill_config=_backfill_config(),
    )
    criteria = EvalCriteria(eval_metrics=("mape", "smape"), primary_eval_metric="mape", trim=0.01)
    html = render_run_config(sr, criteria)
    # Search-space rendering surfaces param names + values.
    assert "Search space" in html
    assert "changepoint_reg" in html
    assert "[0.7, 0.8]" in html
    # Backfill schedule rendered.
    assert "Backfill schedule" in html
    assert "horizon: 21" in html
    assert "n_windows: 12" in html
    assert "algo_name: greykite" in html
    # Eval block present.
    assert "Evaluation" in html
    assert "primary: mape (mean, lower wins)" in html
    assert "trim: 0.01" in html


def test_render_run_config_returns_empty_when_no_metadata():
    sr = SelectionResult(
        results_df=make_results_df(),
        best_params={},
        best_score=float("nan"),
        output_dir=Path("/tmp/dummy"),
        method="grid",
    )
    criteria = EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape")
    assert render_run_config(sr, criteria) == ""


def test_render_run_config_dumps_converter_default_algo_params():
    """When a param_converter is set, the report dumps its ``convert({})``
    output as the forecast template — exactly the right view for
    converter-driven scripts where ``forecast_config.algo_params`` is
    intentionally empty (the converter supplies everything per-candidate).
    """
    from abvelocity.ts.model_selection.param_converter import ParamConverter

    class _DefaultsConverter(ParamConverter):
        def convert(self, params):
            return {
                "model_template": "SILVERKITE",
                "model_components": {
                    "growth": {"growth_term": "linear"},
                    "custom": {"fit_algorithm_dict": {"fit_algorithm": "ridge"}},
                },
                "computation": {"n_jobs": 1},
            }

    space = SearchSpace.flat({"changepoint_reg": [0.7, 0.8]})
    sr = SelectionResult(
        results_df=make_results_df(),
        best_params={"changepoint_reg": 0.7},
        best_score=1.5,
        output_dir=Path("/tmp/dummy"),
        method="grid",
        search_space=space,
        backfill_config=_backfill_config(),
        param_converter=_DefaultsConverter(),
    )
    criteria = EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape")
    html = render_run_config(sr, criteria)
    assert "Forecast template" in html
    assert "SILVERKITE" in html
    assert "fit_algorithm" in html
    assert "ridge" in html


def test_render_run_config_renders_grouped_stages():
    space = SearchSpace(
        groups=[
            ParamGroup(name="regression", params={"fit_algorithm": ["ridge", "linear"]}),
            ParamGroup(name="changepoint", params={"changepoint_reg": [0.01, 1.0]}),
        ]
    )
    sr = SelectionResult(
        results_df=make_results_df(),
        best_params={"fit_algorithm": "ridge", "changepoint_reg": 1.0},
        best_score=1.5,
        output_dir=Path("/tmp/dummy"),
        method="grouped",
        search_space=space,
        backfill_config=_backfill_config(),
    )
    criteria = EvalCriteria(eval_metrics=("mape",), primary_eval_metric="mape")
    html = render_run_config(sr, criteria)
    assert "[stage 1]" in html
    assert "[stage 2]" in html
    assert "fit_algorithm" in html
    assert "changepoint_reg" in html

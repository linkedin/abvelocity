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
"""HTML + CSV summary report for a model-selection run.

:func:`write_report` consumes a :class:`SelectionResult` plus an
:class:`EvalCriteria` and produces a static HTML page next to
``results.csv``. The HTML highlights the winning candidate, lists every
candidate ranked by score, and heat-maps the per-eval-metric columns so
visual scanning works at a glance.

The report intentionally has no JS / no plotly dependency — it's a static
``<table>`` that opens cleanly from the HDFS browser, GitHub
preview, or local file system without external assets.
"""

import html as html_module
import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from abvelocity.ts.model_selection.base import CUTOFFS_FILENAME, SelectionResult
from abvelocity.ts.model_selection.eval_criteria import EvalCriteria

REPORT_FILENAME = "results.html"
"""HTML report filename; written next to ``results.csv`` in the output dir."""


def write_report(
    selection_result: SelectionResult,
    eval_criteria: EvalCriteria,
    title: str = "ModelSelection — Results",
) -> Path:
    """Write the heat-mapped HTML report for a completed run.

    Args:
        selection_result: Result returned by :meth:`ModelSelection.run`.
        eval_criteria: Criteria used for the run (drives which columns
            are heat-mapped and how).
        title: Page title and ``<h1>`` text.

    Returns:
        Path to the written HTML file
        (``selection_result.output_dir / "results.html"``).
    """
    out_path = selection_result.output_dir / REPORT_FILENAME
    html = build_html(selection_result, eval_criteria, title=title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def build_html(
    selection_result: SelectionResult,
    eval_criteria: EvalCriteria,
    title: str,
) -> str:
    """Assemble the full HTML string for a selection result.

    Args:
        selection_result: Result returned by :meth:`ModelSelection.run`.
        eval_criteria: Criteria used for the run.
        title: Page title and ``<h1>`` text.

    Returns:
        Complete HTML document as a string.
    """
    df = selection_result.results_df.copy()
    eval_metric_cols = [f"{m}_mean" for m in eval_criteria.eval_metrics if f"{m}_mean" in df.columns]

    head = render_head(title)
    summary = render_summary(selection_result, eval_criteria)
    run_config = render_run_config(selection_result, eval_criteria)
    stage_summary = render_stage_summary(selection_result)
    table = render_table(df, eval_metric_cols, eval_criteria.lower_is_better)

    return f"""<!DOCTYPE html>
<html>
<head>{head}</head>
<body>
  <h1>{html_module.escape(title)}</h1>
  {summary}
  {run_config}
  {stage_summary}
  <h2>All Candidates</h2>
  {table}
</body>
</html>"""


def render_head(title: str) -> str:
    """Return the ``<head>`` block (charset, title, embedded CSS).

    Args:
        title: Page title.

    Returns:
        HTML fragment for the ``<head>`` element.
    """
    css = """
      body { background:#111; color:#ddd; font-family:'Inter','Segoe UI',sans-serif; margin:30px; }
      h1 { color:#fff; }
      h2 { color:#4f8ef7; margin-top:32px; }
      .summary { background:#1a2b1a; border:1px solid #3dd68c; border-radius:8px; padding:16px;
                 font-family:monospace; font-size:13px; margin:16px 0; }
      .stages { background:#1a1a2b; border:1px solid #3d6ad6; border-radius:8px; padding:16px;
                font-family:monospace; font-size:12px; margin:16px 0; }
      .runcfg { background:#1a1f2b; border:1px solid #4f8ef7; border-radius:8px; padding:16px;
                font-family:monospace; font-size:12px; margin:16px 0; white-space:pre-wrap; }
      .runcfg .label { color:#aaa; }
      table { border-collapse:collapse; font-size:12px; font-family:monospace; white-space:nowrap; }
      th, td { border:1px solid #222; padding:4px 8px; }
      th { background:#1e2130; color:#aaa; }
      tr.winner { background:#1a2b1a; }
      tr.error  { background:#2a1010; color:#aaa; }
    """
    return f"<meta charset='utf-8'><title>{html_module.escape(title)}</title><style>{css}</style>"


def render_summary(selection_result: SelectionResult, eval_criteria: EvalCriteria) -> str:
    """Render the green summary block (best candidate + run config).

    Args:
        selection_result: Result returned by :meth:`ModelSelection.run`.
        eval_criteria: Criteria used for the run.

    Returns:
        HTML fragment for the summary block.
    """
    df = selection_result.results_df
    n_total = len(df)
    n_ok = int((df["status"] == "ok").sum()) if "status" in df.columns else n_total
    n_err = int((df["status"] == "error").sum()) if "status" in df.columns else 0

    direction = "lowest" if eval_criteria.lower_is_better else "highest"

    best_block = (
        " ".join(f"{k}={v!r}" for k, v in selection_result.best_params.items())
        or "(none — every candidate failed)"
    )
    best_score = (
        f"{selection_result.best_score:.6f}"
        if np.isfinite(selection_result.best_score)
        else "—"
    )

    return f"""<div class='summary'>
      <strong>Method:</strong> {html_module.escape(selection_result.method)}<br>
      <strong>Output:</strong> {html_module.escape(str(selection_result.output_dir))}<br>
      <strong>Candidates:</strong> {n_total} total ({n_ok} ok, {n_err} error)<br>
      <strong>Decision eval metric:</strong> {html_module.escape(eval_criteria.primary_eval_metric)}
      ({html_module.escape(eval_criteria.primary_eval_reduction)}, {direction} wins)<br>
      <strong>Best params:</strong> {html_module.escape(best_block)}<br>
      <strong>Best score:</strong> {best_score}
    </div>"""


def _resolve_template_algo_params(selection_result: SelectionResult, backfill: Any) -> Any:
    """Return the algo_params template every candidate inherits.

    Tries the param_converter first (its ``convert({})`` output represents
    the no-override defaults — the right view for converter-driven
    scripts where the BackfillConfig's algo_params is intentionally
    empty). Falls back to the literal template dict from the BackfillConfig
    when no converter is set or the converter raises.

    Returns:
        The resolved algo_params dict (possibly empty) or ``None`` when
        nothing useful is available.
    """
    if selection_result.param_converter is not None:
        try:
            return selection_result.param_converter.convert({})
        except Exception:
            # Some converters may require specific keys; falling back to
            # the literal template is more useful than crashing the report.
            pass
    if backfill is not None:
        return dict(backfill.forecast_config.algo_params or {})
    return None


def render_run_config(selection_result: SelectionResult, eval_criteria: EvalCriteria) -> str:
    """Render the run-config block: search space + cutoff schedule + eval set.

    Surfaces what was actually swept and how it was evaluated, so the
    report is self-documenting (no need to cross-reference the launch
    script). Falls back to an empty string when
    :attr:`SelectionResult.search_space` is ``None`` (e.g. for
    :func:`evaluate_existing` results, which never re-ran a sweep).

    Args:
        selection_result: Result returned by :meth:`ModelSelection.run`.
        eval_criteria: Criteria used for the run.

    Returns:
        HTML fragment for the run-config block, or empty string when
        ``search_space`` and ``backfill_config`` are both unset.
    """
    space = selection_result.search_space
    backfill = selection_result.backfill_config
    if space is None and backfill is None:
        return ""

    lines: List[str] = []

    if space is not None:
        lines.append("<span class='label'>Search space</span>")
        lines.append(html_module.escape(str(space)))
        lines.append("")

    if backfill is not None:
        lines.append("<span class='label'>Backfill schedule</span>")
        if backfill.cutoffs:
            lines.append(f"  mode: explicit cutoffs ({len(backfill.cutoffs)} dates)")
        else:
            lines.append("  mode: algorithmic")
            lines.append(f"  initial_train_size: {backfill.initial_train_size}")
            lines.append(f"  step: {backfill.step}")
            n_windows = backfill.n_windows if backfill.n_windows is not None else "all"
            lines.append(f"  n_windows: {n_windows}")
        lines.append(f"  horizon: {backfill.horizon}")
        lines.append(f"  window_type: {backfill.window_type}")
        fc = backfill.forecast_config
        lines.append(f"  algo_name: {fc.algo_name}")
        lines.append(f"  freq: {fc.freq}")
        lines.append(f"  forecast_horizon: {fc.forecast_horizon}")
        if getattr(fc, "regressor_cols", None):
            lines.append(f"  regressor_cols: {list(fc.regressor_cols)}")
        if getattr(fc, "coverage", None) is not None:
            lines.append(f"  coverage: {fc.coverage}")
        # Resolved cutoff dates from ``cutoffs.json`` (written by every run).
        # The on-disk list is the source of truth for *which* cutoffs were
        # actually evaluated — independent of whether the BackfillConfig
        # specified them explicitly or generated them algorithmically.
        cutoffs_path = selection_result.output_dir / CUTOFFS_FILENAME
        if cutoffs_path.exists():
            try:
                cutoffs_doc = json.loads(cutoffs_path.read_text(encoding="utf-8"))
                resolved = cutoffs_doc.get("cutoffs", [])
            except (json.JSONDecodeError, OSError):
                resolved = []
            if resolved:
                lines.append(f"  cutoffs ({len(resolved)}): {', '.join(resolved)}")
        lines.append("")

        # Forecast template — the algo_params every candidate inherits before
        # sweep overrides. Two sources, in priority order:
        #   1. ``param_converter.convert({})`` — the converter's default-
        #      resolved nested dict. Use when a converter is set (typical
        #      for greykite); the BackfillConfig.forecast_config.algo_params
        #      is then usually ``{}`` because the converter supplies
        #      everything per-candidate.
        #   2. ``backfill_config.forecast_config.algo_params`` — the literal
        #      template dict. Use when no converter is set (simple algos).
        template = _resolve_template_algo_params(selection_result, backfill)
        if template:
            lines.append("<span class='label'>Forecast template</span>")
            lines.append(html_module.escape(json.dumps(template, indent=2, default=str, sort_keys=True)))
            lines.append("")

    lines.append("<span class='label'>Evaluation</span>")
    lines.append(f"  metrics: {list(eval_criteria.eval_metrics)}")
    direction = "lower wins" if eval_criteria.lower_is_better else "higher wins"
    lines.append(
        f"  primary: {eval_criteria.primary_eval_metric} "
        f"({eval_criteria.primary_eval_reduction}, {direction})"
    )
    if getattr(eval_criteria, "trim", None):
        lines.append(f"  trim: {eval_criteria.trim} (top-|error| dropped per group before mean)")

    body = "\n".join(lines)
    return f"<h2>Run Configuration</h2><div class='runcfg'>{body}</div>"


def render_stage_summary(selection_result: SelectionResult) -> str:
    """Render the per-stage cumulative-winner block (grouped runs only).

    Args:
        selection_result: Result returned by :meth:`ModelSelection.run`.

    Returns:
        HTML fragment, or empty string for grid runs.
    """
    if not selection_result.stage_winners:
        return ""
    lines: List[str] = []
    for idx, frozen in enumerate(selection_result.stage_winners):
        params_text = " ".join(f"{k}={v!r}" for k, v in frozen.items())
        lines.append(f"<strong>Stage {idx + 1}:</strong> {html_module.escape(params_text)}")
    body = "<br>".join(lines)
    return f"<h2>Stage Winners</h2><div class='stages'>{body}</div>"


def render_table(df: pd.DataFrame, eval_metric_cols: List[str], lower_is_better: bool) -> str:
    """Render the full candidate table with heat-mapped eval-metric columns.

    Args:
        df: ``selection_result.results_df``.
        eval_metric_cols: ``<eval-metric>_mean`` columns to heat-map.
        lower_is_better: Drives the colour gradient direction.

    Returns:
        HTML fragment for the ``<table>``.
    """
    if df.empty:
        return "<p>(empty)</p>"

    cols = list(df.columns)
    col_min_max = {c: (df[c].dropna().min(), df[c].dropna().max()) for c in eval_metric_cols if c in df.columns}

    headers = "".join(f"<th>{html_module.escape(str(c))}</th>" for c in cols)
    rows: List[str] = []
    for _, row in df.iterrows():
        css_class = ""
        if row.get("is_winner"):
            css_class = "winner"
        elif row.get("status") == "error":
            css_class = "error"
        cells = []
        for col in cols:
            val = row[col]
            if col in col_min_max:
                cells.append(render_heat_cell(val, col_min_max[col], lower_is_better))
            elif col == "score":
                cells.append(render_score_cell(val))
            else:
                cells.append(f"<td>{format_value(val)}</td>")
        rows.append(f"<tr class='{css_class}'>{''.join(cells)}</tr>")

    return f"<table><tr>{headers}</tr>{''.join(rows)}</table>"


def render_heat_cell(val: Any, min_max: Tuple[float, float], lower_is_better: bool) -> str:
    """Return one ``<td>`` with a green-to-red background based on ``val``.

    Args:
        val: Cell value (numeric or NaN).
        min_max: ``(col_min, col_max)`` from non-NaN values in the column.
        lower_is_better: If True, low values are green; else inverted.

    Returns:
        HTML ``<td>`` fragment.
    """
    if pd.isna(val):
        return "<td>—</td>"
    col_min, col_max = min_max
    if col_max == col_min or pd.isna(col_min) or pd.isna(col_max):
        return f"<td>{val:.4f}</td>"
    ratio = (float(val) - col_min) / (col_max - col_min)
    if not lower_is_better:
        ratio = 1.0 - ratio
    red = int(ratio * 200 + 30)
    green = int((1.0 - ratio) * 180 + 30)
    style = f"background:rgb({red},{green},40);color:#fff"
    return f"<td style='{style}'>{val:.4f}</td>"


def render_score_cell(val: Any) -> str:
    """Return one ``<td>`` for the unified ``score`` column with bold styling.

    Args:
        val: Score value (numeric or NaN/inf).

    Returns:
        HTML ``<td>`` fragment.
    """
    if pd.isna(val) or not np.isfinite(val):
        return "<td><strong>—</strong></td>"
    return f"<td><strong>{val:.6f}</strong></td>"


def format_value(val: Any) -> str:
    """Return an HTML-safe stringified cell value for non-heatmap, non-score columns.

    Args:
        val: Any cell value.

    Returns:
        Escaped string form. NaN renders as ``"—"`` for visual lightness.
    """
    if val is None:
        return "—"
    if isinstance(val, float):
        if pd.isna(val):
            return "—"
        return f"{val:g}"
    return html_module.escape(str(val))

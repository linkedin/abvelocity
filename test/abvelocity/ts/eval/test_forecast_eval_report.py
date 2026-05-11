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
"""Tests for :mod:`abvelocity.ts.eval.forecast_eval_report` — pure-pandas core."""

import warnings

import pandas as pd
import plotly.graph_objects as go
import pytest
from abvelocity.ts.eval.forecast_eval_report import (
    ForecastEvalReport,
    evaluate_forecasts_vs_actuals,
    make_per_horizon_error_curve,
    make_per_horizon_forecast_overlays,
    make_per_training_cutoff_forecast_overlays,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_forecast_df() -> pd.DataFrame:
    """Two metrics × 3 cutoffs × 7-day horizon, with deterministic forecast = actual + 1.

    Constructed so:
      * MAE = 1.0 in every (metric, horizon) cell.
      * sMAPE > 0; r² well-defined (non-constant actual).
      * Coverage = 1.0 (CI band ±5 always covers the 1-unit error).

    Returns:
        Long DataFrame with 42 rows and columns ``metric_id``, ``ts``,
        ``last_training_date``, ``horizon_step``, ``forecast``, ``actual``,
        ``forecast_lower``, ``forecast_upper``.
    """
    rows = []
    cutoffs = [pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-13"), pd.Timestamp("2024-01-20")]
    for metric in ("metric_a", "metric_b"):
        base = 100.0 if metric == "metric_a" else 200.0
        for cutoff in cutoffs:
            for h in range(1, 8):
                ts = cutoff + pd.Timedelta(days=h)
                actual = base + 0.5 * h
                rows.append(
                    {
                        "metric_id": metric,
                        "ts": ts,
                        "last_training_date": cutoff,
                        "horizon_step": h,
                        "forecast": actual + 1.0,
                        "actual": actual,
                        "forecast_lower": actual + 1.0 - 5.0,
                        "forecast_upper": actual + 1.0 + 5.0,
                    }
                )
    return pd.DataFrame(rows)


def _make_actuals_df() -> pd.DataFrame:
    """Continuous (metric, ts, actual) for ~30 days — wider than the forecast_df windows.

    Returns:
        Long DataFrame with 60 rows (2 metrics × 30 days) and columns
        ``metric_id``, ``ts``, ``actual``.
    """
    rows = []
    start = pd.Timestamp("2024-01-01")
    for metric in ("metric_a", "metric_b"):
        base = 100.0 if metric == "metric_a" else 200.0
        for offset in range(30):
            ts = start + pd.Timedelta(days=offset)
            rows.append({"metric_id": metric, "ts": ts, "actual": base + 0.5 * offset})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_forecasts_vs_actuals — shape + figure plumbing
# ─────────────────────────────────────────────────────────────────────────────


def test_evaluate_returns_report_with_expected_panel_size():
    """Panel survives the horizon filter and is reset; no rows dropped at default settings."""
    forecast_df = _make_forecast_df()
    actuals_df = _make_actuals_df()
    report = evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=actuals_df)
    # 2 metrics × 3 cutoffs × 7 horizons = 42 rows.
    assert len(report.forecast_df) == 42
    assert isinstance(report, ForecastEvalReport)


def test_evaluate_default_figure_sections_emitted_in_order():
    """Default figure_makers produce 3 sections (cutoff overlays, error curve, horizon overlays)."""
    forecast_df = _make_forecast_df()
    actuals_df = _make_actuals_df()
    report = evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=actuals_df)
    section_names = [s.name for s in report.figure_sections]
    assert section_names == [
        "per_training_cutoff_forecast_overlays",
        "per_horizon_error_curve",
        "per_horizon_forecast_overlays",
    ]


def test_evaluate_each_section_has_one_figure_per_metric():
    """Each default section emits one Plotly figure per metric_id (2 metrics here)."""
    forecast_df = _make_forecast_df()
    actuals_df = _make_actuals_df()
    report = evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=actuals_df)
    for section in report.figure_sections:
        assert set(section.figures.keys()) == {"metric_a", "metric_b"}
        for fig in section.figures.values():
            assert isinstance(fig, go.Figure)


def test_evaluate_empty_figure_makers_produces_no_sections():
    """Passing an empty figure_makers list yields no figure_sections (errors still computed)."""
    forecast_df = _make_forecast_df()
    actuals_df = _make_actuals_df()
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=actuals_df, figure_makers=[],
    )
    assert report.figure_sections == []
    # Errors are still computed independent of figure makers.
    assert len(report.errors_per_metric_horizon) > 0


def test_evaluate_custom_single_figure_maker():
    """Passing only the per-cutoff maker yields exactly one section by that name."""
    forecast_df = _make_forecast_df()
    actuals_df = _make_actuals_df()
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=actuals_df,
        figure_makers=[make_per_training_cutoff_forecast_overlays()],
    )
    assert len(report.figure_sections) == 1
    assert report.figure_sections[0].name == "per_training_cutoff_forecast_overlays"


def test_evaluate_per_cutoff_overlays_caps_to_max_cutoffs_to_plot():
    """``max_cutoffs_to_plot=2`` keeps only the last 2 cutoffs in the plot (errors unaffected)."""
    forecast_df = _make_forecast_df()  # 3 cutoffs
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=None,
        figure_makers=[make_per_training_cutoff_forecast_overlays(max_cutoffs_to_plot=2)],
    )
    section = report.figure_sections[0]
    fig = section.figures["metric_a"]
    # The visible legend names are 2 cutoffs + 1 "Actual" trace = 3 visible entries.
    visible = [trace.name for trace in fig.data if trace.showlegend is not False]
    cutoff_traces = [name for name in visible if name != "Actual"]
    assert len(cutoff_traces) == 2
    # But errors still reflect all 3 cutoffs (3 cutoffs × 7 horizons = 21 rows / 2 metrics).
    assert len(report.forecast_df) == 42


def test_evaluate_per_horizon_error_curve_drops_unknown_metrics():
    """Asking for a metric that wasn't computed silently drops it from the curve."""
    forecast_df = _make_forecast_df()
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=None,
        metrics=("mae",),  # so MAPE / sMAPE absent from errors_per_metric_horizon
        figure_makers=[make_per_horizon_error_curve(error_metrics=("mape", "smape"))],
    )
    # Both requested error metrics absent → empty figures dict (no traces possible).
    section = report.figure_sections[0]
    assert section.figures == {}


def test_evaluate_per_horizon_forecast_overlays_filters_to_present_horizons():
    """Asking for horizons not in the panel only renders the present ones."""
    forecast_df = _make_forecast_df()  # horizons 1..7
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=None,
        figure_makers=[make_per_horizon_forecast_overlays(horizons=(1, 3, 99))],
    )
    section = report.figure_sections[0]
    # 99 is dropped (not present); 1 and 3 survive → figures emitted for both metrics.
    assert set(section.figures.keys()) == {"metric_a", "metric_b"}


def test_evaluate_html_renders_all_default_section_titles():
    """All three default section titles appear in the HTML."""
    forecast_df = _make_forecast_df()
    actuals_df = _make_actuals_df()
    report = evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=actuals_df)
    assert "Forecast vs actual at training cutoffs" in report.html_str
    assert "Per-horizon error curve" in report.html_str
    assert "Per-horizon forecast overlays" in report.html_str


def test_evaluate_per_metric_horizon_errors_have_expected_shape():
    """One row per (metric, horizon_step), columns include configured metrics."""
    forecast_df = _make_forecast_df()
    actuals_df = _make_actuals_df()
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=actuals_df, metrics=("mae", "mape", "coverage"),
    )
    # 2 metrics × 7 horizons = 14 rows.
    assert len(report.errors_per_metric_horizon) == 14
    assert {"metric_id", "horizon_step", "n", "mae", "mape", "coverage"} <= set(
        report.errors_per_metric_horizon.columns
    )


def test_evaluate_mae_equals_constructed_error():
    """Forecast = actual + 1 → MAE = 1.0 in every per-(metric, horizon) cell."""
    forecast_df = _make_forecast_df()
    report = evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=None, metrics=("mae",))
    # 3 cutoffs each contribute one row at every horizon → 3 errors of 1.0 each → MAE = 1.0.
    assert report.errors_per_metric_horizon["mae"].unique().tolist() == [1.0]


def test_evaluate_per_metric_aggregate_mae_one():
    """Aggregated across horizons MAE is still 1.0 (constant 1-unit error)."""
    forecast_df = _make_forecast_df()
    report = evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=None, metrics=("mae",))
    assert len(report.errors_per_metric) == 2
    assert report.errors_per_metric["mae"].unique().tolist() == [1.0]


def test_evaluate_coverage_one_when_ci_covers_error():
    """CI ±5 contains the 1-unit error everywhere → coverage = 1.0."""
    forecast_df = _make_forecast_df()
    report = evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=None, metrics=("coverage",))
    assert report.errors_per_metric["coverage"].unique().tolist() == [1.0]


def test_evaluate_drops_coverage_with_warning_when_ci_columns_missing():
    """If CI cols absent, coverage is dropped + a warning fires (not an error)."""
    forecast_df = _make_forecast_df().drop(columns=["forecast_lower", "forecast_upper"])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        report = evaluate_forecasts_vs_actuals(
            forecast_df=forecast_df, actuals_df=None, metrics=("mae", "coverage"),
        )
    assert "coverage" not in report.errors_per_metric.columns
    assert any("coverage" in str(w.message) for w in captured)


def test_evaluate_horizon_filter_applied():
    """``max_days_ahead=3`` drops horizons 4..7."""
    forecast_df = _make_forecast_df()
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=None, max_days_ahead=3,
    )
    assert report.errors_per_metric_horizon["horizon_step"].max() == 3


def test_evaluate_explicit_training_cutoffs_filter_forecast_df():
    """Passing only one cutoff drops the rest."""
    forecast_df = _make_forecast_df()
    only = [pd.Timestamp("2024-01-06")]
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=None, training_cutoffs=only,
    )
    assert report.forecast_df["last_training_date"].unique().tolist() == [pd.Timestamp("2024-01-06")]


def test_evaluate_unknown_training_cutoff_raises():
    """Cutoffs that don't exist in forecast_df → ValueError."""
    forecast_df = _make_forecast_df()
    with pytest.raises(ValueError, match="not present in forecast_df"):
        evaluate_forecasts_vs_actuals(
            forecast_df=forecast_df, actuals_df=None,
            training_cutoffs=[pd.Timestamp("2099-12-31")],
        )


def test_evaluate_missing_required_column_raises():
    """Missing ``forecast`` column → ValueError listing it."""
    forecast_df = _make_forecast_df().drop(columns=["forecast"])
    with pytest.raises(ValueError, match="forecast"):
        evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=None)


def test_evaluate_no_rows_after_horizon_filter_raises():
    """``max_days_ahead`` too small → empty forecast_df → ValueError."""
    forecast_df = _make_forecast_df()
    with pytest.raises(ValueError, match="No rows in forecast_df after filtering"):
        evaluate_forecasts_vs_actuals(
            forecast_df=forecast_df, actuals_df=None, max_days_ahead=0,
        )


def test_evaluate_html_str_contains_summary_block():
    """``report_summary`` is rendered into the HTML inside a ``note`` block."""
    forecast_df = _make_forecast_df()
    actuals_df = _make_actuals_df()
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=actuals_df,
        report_title="Title for test",
        report_summary="My custom summary.",
    )
    assert "Title for test" in report.html_str
    assert "My custom summary." in report.html_str
    assert "<table" in report.html_str  # error tables embedded


def test_evaluate_horizon_step_inferred_from_cutoff_diff():
    """If forecast_df lacks ``horizon_step``, it's computed from ts − cutoff in days."""
    forecast_df = _make_forecast_df().drop(columns=["horizon_step"])
    report = evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=None)
    assert report.errors_per_metric_horizon["horizon_step"].max() == 7


def test_evaluate_panel_actual_with_nan_excluded_from_errors():
    """Rows with actual=NaN are dropped from error compute (no mean-of-NaN)."""
    forecast_df = _make_forecast_df()
    forecast_df.loc[0, "actual"] = float("nan")
    report = evaluate_forecasts_vs_actuals(
        forecast_df=forecast_df, actuals_df=None, metrics=("mae",),
    )
    # The MAE for the affected (metric, horizon) bucket should now be over
    # 2 rows instead of 3, but the constant 1-unit error keeps MAE = 1.0.
    assert report.errors_per_metric_horizon["mae"].unique().tolist() == [1.0]


# ─────────────────────────────────────────────────────────────────────────────
# write_html
# ─────────────────────────────────────────────────────────────────────────────


def test_write_html_creates_file(tmp_path):
    """``write_html`` writes the html_str to disk and returns the path."""
    forecast_df = _make_forecast_df()
    actuals_df = _make_actuals_df()
    report = evaluate_forecasts_vs_actuals(forecast_df=forecast_df, actuals_df=actuals_df)
    out = tmp_path / "subdir" / "report.html"
    written = report.write_html(out)
    assert written == out
    assert out.read_text().startswith("<!DOCTYPE html>")

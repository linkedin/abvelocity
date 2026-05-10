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
"""Tests for :mod:`abvelocity.ts.viz`."""

import pandas as pd
import plotly.graph_objects as go

from abvelocity.ts.viz import (
    DEFAULT_EVENT_MARKER_PALETTE,
    add_event_traces,
    hex_to_rgba,
    plot_forecast,
    plot_forecast_at_training_cutoffs,
    plot_forecast_groups_vs_actual,
    plot_forecast_vs_actual,
)


def make_long_format_forecast_df() -> pd.DataFrame:
    """Tiny long-format forecast frame with one metric for plot tests."""
    n = 14
    return pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
            "metric_id": ["signups"] * n,
            "actual": [10.0] * 7 + [None] * 7,
            "forecast": [9.0] * n,
            "forecast_lower": [7.0] * n,
            "forecast_upper": [11.0] * n,
        }
    )


def test_plot_forecast_vs_actual_default_hover_shows_day_of_week():
    """Default ``xaxis_hoverformat`` is ``"%a %Y-%m-%d"`` so hover reads
    ``"Mon 2024-01-01"`` — useful for spotting weekly seasonality."""
    df = make_long_format_forecast_df()
    fig = plot_forecast_vs_actual(df=df)
    assert fig.layout.xaxis.hoverformat == "%a %Y-%m-%d"


def test_plot_forecast_vs_actual_hoverformat_is_overridable():
    """Caller can pass any strftime format string."""
    df = make_long_format_forecast_df()
    fig = plot_forecast_vs_actual(df=df, xaxis_hoverformat="%Y-%m-%d (%A)")
    assert fig.layout.xaxis.hoverformat == "%Y-%m-%d (%A)"


def test_plot_forecast_vs_actual_hoverformat_can_be_disabled_with_none():
    """``xaxis_hoverformat=None`` lets plotly pick its default."""
    df = make_long_format_forecast_df()
    fig = plot_forecast_vs_actual(df=df, xaxis_hoverformat=None)
    # plotly leaves hoverformat unset (empty string in the layout).
    assert not fig.layout.xaxis.hoverformat


def test_plot_forecast_passes_hoverformat_through_kwargs():
    """``plot_forecast`` forwards kwargs to ``plot_forecast_vs_actual`` —
    so passing ``xaxis_hoverformat`` at the wrapper level reaches the
    inner function."""
    df = make_long_format_forecast_df()
    fig = plot_forecast(result_df=df, xaxis_hoverformat="%a")
    # subplots=False (single metric) → wrapper returns the inner fig directly.
    assert fig.layout.xaxis.hoverformat == "%a"


# ─────────────────────────────────────────────────────────────────────────────
# plot_forecast_groups_vs_actual
# ─────────────────────────────────────────────────────────────────────────────


def make_long_format_groups_df(group_col: str = "horizon_step", values=(1, 7)) -> pd.DataFrame:
    """Long-format frame: 14 dates × 2 group values, with CI columns.

    The same calendar date appears once per group value, so dedupe of
    actuals in the function-under-test is exercised. Generic across
    grouping dimensions — pass ``group_col="country", values=("us", "in")``
    to test the country-overlay use case.
    """
    n = 14
    rows = []
    for value in values:
        for offset in range(n):
            forecast_offset = 0.1 * (hash(value) % 10)
            rows.append(
                {
                    "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(days=offset),
                    group_col: value,
                    "actual": 10.0 if offset < 7 else None,
                    "forecast": 9.0 + forecast_offset,
                    "forecast_lower": 7.0,
                    "forecast_upper": 11.0,
                }
            )
    return pd.DataFrame(rows)


def test_plot_forecast_groups_returns_figure():
    """Smoke: returns a plotly Figure."""
    df = make_long_format_groups_df()
    fig = plot_forecast_groups_vs_actual(df=df, group_col="horizon_step", group_values=(1, 7))
    assert isinstance(fig, go.Figure)


def test_plot_forecast_groups_legend_has_one_entry_per_value_plus_actual():
    """Visible legend entries = one per group value + ``Actual``; CI traces hidden from legend."""
    df = make_long_format_groups_df()
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col="horizon_step",
        group_values=(1, 7),
        group_label_template="horizon={value}d",
    )
    visible_names = sorted(trace.name for trace in fig.data if trace.showlegend is not False)
    assert visible_names == ["Actual", "horizon=1d", "horizon=7d"]


def test_plot_forecast_groups_default_label_template_is_raw_value():
    """Without a custom ``group_label_template``, the legend shows the raw value."""
    df = make_long_format_groups_df(group_col="country", values=("us", "in"))
    fig = plot_forecast_groups_vs_actual(df=df, group_col="country", group_values=("us", "in"))
    visible_names = sorted(trace.name for trace in fig.data if trace.showlegend is not False)
    assert visible_names == ["Actual", "in", "us"]


def test_plot_forecast_groups_default_hover_shows_day_of_week():
    """Default ``xaxis_hoverformat`` matches the rest of the module — ``"%a %Y-%m-%d"``."""
    df = make_long_format_groups_df()
    fig = plot_forecast_groups_vs_actual(df=df, group_col="horizon_step", group_values=(1,))
    assert fig.layout.xaxis.hoverformat == "%a %Y-%m-%d"


def test_plot_forecast_groups_hoverformat_overridable():
    """Caller-provided format wins over the day-of-week default."""
    df = make_long_format_groups_df()
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col="horizon_step",
        group_values=(1,),
        xaxis_hoverformat="%Y-%m-%d",
    )
    assert fig.layout.xaxis.hoverformat == "%Y-%m-%d"


def test_plot_forecast_groups_no_ci_when_cols_omitted():
    """``forecast_lower_col=None`` (or upper) drops both CI traces per group."""
    df = make_long_format_groups_df()
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col="horizon_step",
        group_values=(1, 7),
        forecast_lower_col=None,
        forecast_upper_col=None,
    )
    # 2 forecast lines + 1 actual = 3 traces (no CI bands).
    assert len(fig.data) == 3


def test_plot_forecast_groups_with_ci_has_band_traces_per_value():
    """Default (CI cols present) produces 2 invisible band traces + 1 line per group value, plus actual."""
    df = make_long_format_groups_df()
    fig = plot_forecast_groups_vs_actual(df=df, group_col="horizon_step", group_values=(1, 7))
    # 2 forecast lines + 4 CI traces (2 per value) + 1 actual = 7 traces.
    assert len(fig.data) == 7


def test_plot_forecast_groups_unknown_value_raises_with_useful_message():
    """Unknown group value → ``ValueError`` listing the bad value and the available ones."""
    df = make_long_format_groups_df()
    try:
        plot_forecast_groups_vs_actual(df=df, group_col="horizon_step", group_values=(1, 99))
    except ValueError as error:
        assert "99" in str(error)
        assert "horizon_step" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_plot_forecast_groups_empty_values_raises():
    """``group_values=()`` is a programming error — no curves to plot."""
    df = make_long_format_groups_df()
    try:
        plot_forecast_groups_vs_actual(df=df, group_col="horizon_step", group_values=())
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_plot_forecast_groups_missing_required_column_raises():
    """Missing ``forecast`` column → clear error before any trace gets built."""
    df = make_long_format_groups_df().drop(columns=["forecast"])
    try:
        plot_forecast_groups_vs_actual(df=df, group_col="horizon_step", group_values=(1,))
    except ValueError as error:
        assert "forecast" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_plot_forecast_groups_actuals_df_relaxes_actual_col_requirement_on_df():
    """When ``actuals_df`` is provided, ``df`` doesn't need an ``actual`` column at all.

    Real OH-table caller pattern: forecasts and actuals come from separate Trino
    queries; the forecast panel has no ``actual`` column.
    """
    df = make_long_format_groups_df().drop(columns=["actual"])
    actuals_df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
            "actual": [10.0 + 0.1 * i for i in range(14)],
        }
    )
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col="horizon_step",
        group_values=(1, 7),
        actuals_df=actuals_df,
    )
    actual_trace = next(trace for trace in fig.data if trace.name == "Actual")
    assert len(actual_trace.x) == 14


def test_plot_forecast_groups_line_and_band_share_base_rgb():
    """When ``group_colors`` is unspecified, the line and CI-band fill share the
    same RGB triple — only opacity differs — so the band visually belongs to its line."""
    df = make_long_format_groups_df()
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col="horizon_step",
        group_values=(1,),
        group_label_template="horizon={value}d",
    )
    line_trace = next(trace for trace in fig.data if trace.name == "horizon=1d")
    upper_trace = next(
        trace for trace in fig.data if trace.legendgroup == "horizon_step=1" and trace.fillcolor
    )
    line_rgb = line_trace.line.color.split("(")[1].rsplit(",", 1)[0]
    band_rgb = upper_trace.fillcolor.split("(")[1].rsplit(",", 1)[0]
    assert line_rgb == band_rgb


def test_plot_forecast_groups_actual_deduped_across_slices():
    """Same date appears once per group value in ``df``; dedupe by time → one row per date.

    The fixture has 14 dates total; ``actual`` is filled for the first 7 and NaN for
    the last 7. After dedupe-by-time, we get 14 points (7 valued + 7 NaN). NaN values
    are preserved so plotly breaks the actuals line at the gap.
    """
    df = make_long_format_groups_df()
    fig = plot_forecast_groups_vs_actual(df=df, group_col="horizon_step", group_values=(1, 7))
    actual_trace = next(trace for trace in fig.data if trace.name == "Actual")
    assert len(actual_trace.x) == 14
    y_vals = list(actual_trace.y)
    valid_count = sum(1 for value in y_vals if not pd.isna(value))
    assert valid_count == 7


def test_plot_forecast_groups_explicit_colors_used_verbatim():
    """When ``group_colors`` is provided, those colors land on both line and band."""
    df = make_long_format_groups_df()
    explicit = ["rgba(255, 0, 0, 0.7)", "rgba(0, 0, 255, 0.7)"]
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col="horizon_step",
        group_values=(1, 7),
        group_label_template="horizon={value}d",
        group_colors=explicit,
    )
    line_h1 = next(trace for trace in fig.data if trace.name == "horizon=1d")
    assert line_h1.line.color == "rgba(255, 0, 0, 0.7)"


def test_plot_forecast_groups_explicit_colors_length_mismatch_raises():
    """Passing fewer / more colors than group values is a programming error."""
    df = make_long_format_groups_df()
    try:
        plot_forecast_groups_vs_actual(
            df=df,
            group_col="horizon_step",
            group_values=(1, 7),
            group_colors=["rgba(255, 0, 0, 0.7)"],
        )
    except ValueError as error:
        assert "group_colors" in str(error)
    else:
        raise AssertionError("expected ValueError")


# ─────────────────────────────────────────────────────────────────────────────
# plot_forecast_groups_vs_actual — auto-discover and multi-dim group_col
# ─────────────────────────────────────────────────────────────────────────────


def make_long_format_two_dim_df() -> pd.DataFrame:
    """14 dates × 3 (country, surface) combinations, all sharing the same actual.

    Combinations: (us, voyager), (us, web), (in, voyager). Mid-tuple
    pair (in, web) is intentionally absent so auto-discover doesn't
    invent it.
    """
    n = 14
    rows = []
    combos = [("us", "voyager"), ("us", "web"), ("in", "voyager")]
    for country, surface in combos:
        for offset in range(n):
            forecast_offset = 0.1 * (hash((country, surface)) % 10)
            rows.append(
                {
                    "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(days=offset),
                    "country": country,
                    "surface": surface,
                    "actual": 10.0 if offset < 7 else None,
                    "forecast": 9.0 + forecast_offset,
                    "forecast_lower": 7.0,
                    "forecast_upper": 11.0,
                }
            )
    return pd.DataFrame(rows)


def test_plot_forecast_groups_auto_discover_single_dim():
    """``group_values=None`` discovers every distinct value in ``df[group_col]``."""
    df = make_long_format_groups_df()
    fig = plot_forecast_groups_vs_actual(df=df, group_col="horizon_step", group_values=None)
    visible_names = sorted(trace.name for trace in fig.data if trace.showlegend is not False)
    # Fixture has horizons 1 and 7; auto-discover should yield both + Actual.
    assert visible_names == ["1", "7", "Actual"]


def test_plot_forecast_groups_multi_dim_explicit_combinations():
    """``group_col=tuple`` + ``group_values=list of tuples`` → one curve per combo."""
    df = make_long_format_two_dim_df()
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col=("country", "surface"),
        group_values=[("us", "voyager"), ("us", "web")],
    )
    visible_names = sorted(trace.name for trace in fig.data if trace.showlegend is not False)
    # Default label template is "{value}" → comma-joined: "us, voyager", "us, web".
    assert visible_names == ["Actual", "us, voyager", "us, web"]


def test_plot_forecast_groups_multi_dim_auto_discover():
    """``group_col=tuple`` + ``group_values=None`` discovers all combos in df."""
    df = make_long_format_two_dim_df()
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col=("country", "surface"),
        group_values=None,
    )
    visible_names = sorted(trace.name for trace in fig.data if trace.showlegend is not False)
    # Three distinct combos in the fixture, sorted alphabetically by (country, surface).
    assert visible_names == ["Actual", "in, voyager", "us, voyager", "us, web"]


def test_plot_forecast_groups_multi_dim_named_kwargs_in_template():
    """Per-column kwargs are available in ``group_label_template`` alongside ``{value}``."""
    df = make_long_format_two_dim_df()
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col=("country", "surface"),
        group_values=[("us", "voyager")],
        group_label_template="{country}/{surface}",
    )
    visible_names = sorted(trace.name for trace in fig.data if trace.showlegend is not False)
    assert visible_names == ["Actual", "us/voyager"]


def test_plot_forecast_groups_multi_dim_wrong_tuple_length_raises():
    """A 1-tuple where a 2-tuple is expected is a programming error."""
    df = make_long_format_two_dim_df()
    try:
        plot_forecast_groups_vs_actual(
            df=df,
            group_col=("country", "surface"),
            group_values=[("us",)],
        )
    except ValueError as error:
        assert "tuple of length 2" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_plot_forecast_groups_multi_dim_missing_combination_raises():
    """Asking for ``(in, web)`` when only three other combos exist → ValueError."""
    df = make_long_format_two_dim_df()
    try:
        plot_forecast_groups_vs_actual(
            df=df,
            group_col=("country", "surface"),
            group_values=[("in", "web")],
        )
    except ValueError as error:
        assert "in" in str(error) and "web" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_plot_forecast_groups_actuals_df_takes_priority_over_df():
    """When ``actuals_df`` is passed, the actuals trace's y values come from it (not from ``df``)."""
    df = make_long_format_groups_df()
    # Distinct sentinel values so the assertion is unambiguous about which source was used.
    actuals_df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
            "actual": [99.0 + 0.5 * i for i in range(14)],
        }
    )
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col="horizon_step",
        group_values=(1, 7),
        actuals_df=actuals_df,
    )
    actual_trace = next(trace for trace in fig.data if trace.name == "Actual")
    y_vals = list(actual_trace.y)
    # First and last values come from actuals_df, not from df's "10.0"s.
    assert y_vals[0] == 99.0
    assert y_vals[-1] == 99.0 + 0.5 * 13


def test_plot_forecast_groups_actuals_nan_breaks_line_not_dropped():
    """NaN actuals are preserved in the trace so plotly draws a gap (not a connecting line)."""
    df = make_long_format_groups_df()
    actuals_df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=14, freq="D"),
            "actual": [10.0, 10.5, 11.0, float("nan"), float("nan"), 13.0, 13.5,
                       14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0],
        }
    )
    fig = plot_forecast_groups_vs_actual(
        df=df,
        group_col="horizon_step",
        group_values=(1,),
        actuals_df=actuals_df,
    )
    actual_trace = next(trace for trace in fig.data if trace.name == "Actual")
    # 14 timestamps preserved (NaN rows NOT dropped — plotly breaks the line at NaN).
    assert len(actual_trace.x) == 14
    # NaN positions are preserved at indices 3 and 4.
    import numpy as np
    y_vals = list(actual_trace.y)
    assert np.isnan(y_vals[3]) and np.isnan(y_vals[4])
    # Surrounding non-NaN values are intact.
    assert y_vals[2] == 11.0 and y_vals[5] == 13.0


def test_plot_forecast_groups_actuals_df_missing_columns_raises():
    """``actuals_df`` without ``time_col`` or ``actual_col`` is a programming error."""
    df = make_long_format_groups_df()
    bad_actuals = pd.DataFrame({"wrong": [1, 2, 3]})
    try:
        plot_forecast_groups_vs_actual(
            df=df,
            group_col="horizon_step",
            group_values=(1,),
            actuals_df=bad_actuals,
        )
    except ValueError as error:
        assert "actuals source" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_plot_forecast_groups_auto_discover_empty_df_raises():
    """Auto-discover on a df where ``group_col`` is all-NaN → ValueError."""
    df = make_long_format_groups_df()
    df["horizon_step"] = pd.NA
    try:
        plot_forecast_groups_vs_actual(df=df, group_col="horizon_step", group_values=None)
    except ValueError as error:
        assert "Auto-discovery" in str(error)
    else:
        raise AssertionError("expected ValueError")


# ─────────────────────────────────────────────────────────────────────────────
# plot_forecast_at_training_cutoffs
# ─────────────────────────────────────────────────────────────────────────────


def make_training_cutoffs_panel_df() -> pd.DataFrame:
    """Long-format panel: 3 cutoffs × 21 days-ahead, with ts/last_training_date/forecast/actual."""
    rows = []
    cutoffs = [pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-13"), pd.Timestamp("2024-01-20")]
    for cutoff in cutoffs:
        for h in range(1, 22):
            ts = cutoff + pd.Timedelta(days=h)
            rows.append(
                {
                    "ts": ts,
                    "last_training_date": cutoff,
                    "forecast": 100.0 + 0.3 * h,
                    "actual": 100.0 + 0.25 * h,
                    "forecast_lower": 95.0,
                    "forecast_upper": 105.0,
                }
            )
    return pd.DataFrame(rows)


def test_plot_forecast_at_training_cutoffs_returns_figure():
    """Smoke: returns a plotly Figure when given a valid panel."""
    df = make_training_cutoffs_panel_df()
    fig = plot_forecast_at_training_cutoffs(df=df, max_days_ahead=7)
    assert isinstance(fig, go.Figure)


def test_plot_forecast_at_training_cutoffs_one_line_per_cutoff_plus_actual():
    """N cutoffs → N legend entries + ``Actual`` (CI traces hidden from legend)."""
    df = make_training_cutoffs_panel_df()
    fig = plot_forecast_at_training_cutoffs(df=df, max_days_ahead=7)
    visible = sorted(trace.name for trace in fig.data if trace.showlegend is not False)
    # 3 cutoffs in fixture → 3 cutoff lines + 1 actual.
    assert len(visible) == 4
    assert "Actual" in visible


def test_plot_forecast_at_training_cutoffs_days_ahead_filter_excludes_long_horizons():
    """A row at days_ahead=14 with max_days_ahead=7 must NOT appear in the cutoff trace."""
    df = make_training_cutoffs_panel_df()
    fig = plot_forecast_at_training_cutoffs(df=df, max_days_ahead=7)
    cutoff_label = next(
        trace for trace in fig.data
        if trace.showlegend is not False and trace.name != "Actual"
    )
    # Each cutoff trajectory should have at most 7 points (days +1..+7), not 21.
    assert len(cutoff_label.x) == 7


def test_plot_forecast_at_training_cutoffs_explicit_cutoffs_subset():
    """Passing only a subset of cutoffs renders only those (others ignored)."""
    df = make_training_cutoffs_panel_df()
    only = [pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-13")]
    fig = plot_forecast_at_training_cutoffs(df=df, training_cutoffs=only, max_days_ahead=7)
    # 2 cutoff traces + actual = 3 visible legend entries.
    visible = [trace.name for trace in fig.data if trace.showlegend is not False]
    assert len(visible) == 3


def test_plot_forecast_at_training_cutoffs_no_rows_after_filter_raises():
    """If max_days_ahead is too small to capture any rows, raise with a useful hint."""
    df = make_training_cutoffs_panel_df()
    df = df[df["ts"] - df["last_training_date"] > pd.Timedelta(days=10)]  # only h>=11
    try:
        plot_forecast_at_training_cutoffs(df=df, max_days_ahead=5)
    except ValueError as error:
        assert "days-ahead filter" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_plot_forecast_at_training_cutoffs_missing_cutoff_col_raises():
    """Missing ``training_cutoff_col`` (e.g. wrong column name) → clear ValueError."""
    df = make_training_cutoffs_panel_df().drop(columns=["last_training_date"])
    try:
        plot_forecast_at_training_cutoffs(df=df, max_days_ahead=7)
    except ValueError as error:
        assert "last_training_date" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_plot_forecast_at_training_cutoffs_with_ci_renders_band_traces_per_cutoff():
    """Default (CI cols present) → 2 invisible band traces + 1 line per cutoff + 1 actual."""
    df = make_training_cutoffs_panel_df()
    fig = plot_forecast_at_training_cutoffs(df=df, max_days_ahead=7)
    # 3 cutoffs × (2 CI + 1 line) + 1 actual = 10 traces.
    assert len(fig.data) == 10


def test_plot_forecast_at_training_cutoffs_no_ci_when_columns_omitted():
    """Setting CI cols to None drops the band traces."""
    df = make_training_cutoffs_panel_df()
    fig = plot_forecast_at_training_cutoffs(
        df=df, max_days_ahead=7,
        forecast_lower_col=None, forecast_upper_col=None,
    )
    # 3 cutoff lines + 1 actual = 4 traces.
    assert len(fig.data) == 4


def test_plot_forecast_at_training_cutoffs_default_label_uses_cutoff_value():
    """Default ``training_cutoff_label_template`` formats as ``"cutoff <value>"``."""
    df = make_training_cutoffs_panel_df()
    fig = plot_forecast_at_training_cutoffs(df=df, max_days_ahead=7)
    visible = sorted(
        trace.name for trace in fig.data
        if trace.showlegend is not False and trace.name != "Actual"
    )
    assert all(name.startswith("cutoff ") for name in visible)


# ─────────────────────────────────────────────────────────────────────────────
# hex_to_rgba
# ─────────────────────────────────────────────────────────────────────────────


def test_hex_to_rgba_round_trips_known_color():
    """``#FF8000`` + alpha 0.5 → ``rgba(255, 128, 0, 0.5)``."""
    assert hex_to_rgba("#FF8000", 0.5) == "rgba(255, 128, 0, 0.5)"


def test_hex_to_rgba_accepts_no_leading_hash():
    """Leading ``#`` is optional — both forms parse identically."""
    assert hex_to_rgba("00FF00", 1.0) == hex_to_rgba("#00FF00", 1.0)


def test_hex_to_rgba_handles_zero_alpha():
    """Alpha 0 means fully transparent — value still reaches the output."""
    assert hex_to_rgba("#000000", 0.0) == "rgba(0, 0, 0, 0.0)"


# ─────────────────────────────────────────────────────────────────────────────
# add_event_traces
# ─────────────────────────────────────────────────────────────────────────────


def make_events_df() -> pd.DataFrame:
    """Realistic event annotations frame — dates, names, country, y-values."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-07-04", "2024-08-15"]),
            "name": ["New Years", "Independence", "Independence"],
            "country": ["UnitedStates", "UnitedStates", "India"],
            "y": [10.0, 20.0, 5.0],
        }
    )


def test_add_event_traces_adds_one_trace_per_group():
    """Each distinct ``group_col`` value becomes its own toggleable trace."""
    fig = go.Figure()
    add_event_traces(fig=fig, events_df=make_events_df())
    # Two countries → two traces.
    assert len(fig.data) == 2
    names = sorted(trace.name for trace in fig.data)
    assert names == ["event (India)", "event (UnitedStates)"]


def test_add_event_traces_mutates_figure_in_place_returns_none():
    """Helper has no return — it appends traces to the passed figure."""
    fig = go.Figure()
    result = add_event_traces(fig=fig, events_df=make_events_df())
    assert result is None
    assert len(fig.data) > 0


def test_add_event_traces_drops_rows_with_nan_y():
    """Rows where ``y_col`` is NaN have no plotting position — they're dropped."""
    df = make_events_df()
    df.loc[0, "y"] = float("nan")  # New Years, US — drop
    fig = go.Figure()
    add_event_traces(fig=fig, events_df=df)
    # US trace still present (Independence remains), but only 1 point on it.
    us_trace = next(t for t in fig.data if "UnitedStates" in t.name)
    assert len(us_trace.x) == 1


def test_add_event_traces_empty_df_is_noop():
    """Empty events_df → no traces added, no error."""
    fig = go.Figure()
    add_event_traces(fig=fig, events_df=pd.DataFrame(columns=["date", "name", "country", "y"]))
    assert len(fig.data) == 0


def test_add_event_traces_all_nan_y_is_noop():
    """All rows have NaN y → nothing to plot, no error."""
    df = make_events_df()
    df["y"] = float("nan")
    fig = go.Figure()
    add_event_traces(fig=fig, events_df=df)
    assert len(fig.data) == 0


def test_add_event_traces_legend_prefix_is_customizable():
    """``legend_prefix`` controls the trace name template."""
    fig = go.Figure()
    add_event_traces(fig=fig, events_df=make_events_df(), legend_prefix="holiday")
    assert all(trace.name.startswith("holiday (") for trace in fig.data)


def test_add_event_traces_recycles_palette_when_more_groups_than_colors():
    """Palette of length 1 → both traces get the same color."""
    fig = go.Figure()
    add_event_traces(fig=fig, events_df=make_events_df(), palette=("#FF0000",))
    colors = [trace.marker.color for trace in fig.data]
    assert colors == ["#FF0000", "#FF0000"]


def test_add_event_traces_passes_custom_marker_symbol_and_size():
    """Marker ``symbol`` / ``size`` kwargs reach the trace."""
    fig = go.Figure()
    add_event_traces(fig=fig, events_df=make_events_df(), marker_symbol="diamond", marker_size=20)
    assert all(trace.marker.symbol == "diamond" for trace in fig.data)
    assert all(trace.marker.size == 20 for trace in fig.data)


def test_default_event_marker_palette_has_distinct_hex_strings():
    """Sanity check on the default palette — non-empty, well-formed hex."""
    assert len(DEFAULT_EVENT_MARKER_PALETTE) > 0
    for color in DEFAULT_EVENT_MARKER_PALETTE:
        assert color.startswith("#")
        assert len(color) == 7  # "#RRGGBB"

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
"""Tests for :mod:`abvelocity.ts.eval.live_eval` — DataContainer fetchers + orchestrator."""

from datetime import date

import pandas as pd
import pytest
from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.ts.eval.forecast_eval_report import (
    ForecastEvalReport,
    make_per_training_cutoff_forecast_overlays,
)
from abvelocity.ts.eval.live_eval import (
    fetch_forecasts_at_training_cutoffs,
    list_training_cutoffs,
    run_live_forecast_eval,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — raw partition snapshots (pandas-mode DataContainer source)
# ─────────────────────────────────────────────────────────────────────────────


def _make_partitions_df() -> pd.DataFrame:
    """Long partition rows with stage column — what a raw OH table snapshot looks like.

    Forecast / actual values are deterministic functions of the *date* (day-of-month),
    so when the same date appears in multiple partitions (e.g. forecast row in an
    earlier cutoff and fitted row in a later cutoff), the partitions agree — no
    cross-partition CV warning.

    Returns:
        Long DataFrame with 22 rows (2 cutoffs × (4 fitted + 7 forecast))
        and columns ``metric_id``, ``last_training_date``,
        ``forecasted_date``, ``stage``, ``actual``, ``forecast``,
        ``forecast_lower``, ``forecast_upper``. ``actual`` is ``NaN`` on
        forecast-stage rows.
    """
    def value_at(ts: pd.Timestamp) -> float:
        """Date-keyed deterministic value — partition rows at the same date agree.

        Args:
            ts: Forecasted date.

        Returns:
            ``100.0 + ts.day`` (day-of-month).
        """
        return 100.0 + float(ts.day)

    rows = []
    cutoffs = [pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-13")]
    for cutoff in cutoffs:
        # Fitted history rows.
        for h in range(-3, 1):
            ts = cutoff + pd.Timedelta(days=h)
            v = value_at(ts)
            rows.append(
                {
                    "metric_id": "metric_a",
                    "last_training_date": cutoff,
                    "forecasted_date": ts,
                    "stage": "fitted",
                    "actual": v,
                    "forecast": v,
                    "forecast_lower": v - 5.0,
                    "forecast_upper": v + 5.0,
                }
            )
        # Forecast rows.
        for h in range(1, 8):
            ts = cutoff + pd.Timedelta(days=h)
            v = value_at(ts)
            rows.append(
                {
                    "metric_id": "metric_a",
                    "last_training_date": cutoff,
                    "forecasted_date": ts,
                    "stage": "forecast",
                    "actual": float("nan"),
                    "forecast": v,
                    "forecast_lower": v - 5.0,
                    "forecast_upper": v + 5.0,
                }
            )
    return pd.DataFrame(rows)


def _make_partitions_with_observed_actuals() -> pd.DataFrame:
    """Partitions where the actual at every date is uniformly ``forecast − 1``.

    Adds synthetic ``stage='fitted'`` rows at the forecast dates so the dedup
    can populate an actual for every forecasted date. All fitted rows agree
    on ``actual = forecast − 1`` so MAE is exactly 1.0 with no cross-partition
    CV warning.

    Returns:
        Long DataFrame extending :func:`_make_partitions_df` with 14
        additional ``stage='fitted'`` rows that mirror the original
        forecast rows but carry ``actual = forecast - 1``. Same column
        shape as the source.
    """
    df = _make_partitions_df()
    df.loc[df["stage"] == "fitted", "actual"] = (
        df.loc[df["stage"] == "fitted", "forecast"] - 1.0
    )
    extra = df[df["stage"] == "forecast"].copy()
    extra["stage"] = "fitted"
    extra["actual"] = extra["forecast"] - 1.0
    return pd.concat([df, extra], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# fetch_forecasts_at_training_cutoffs — pandas-mode DataContainer
# ─────────────────────────────────────────────────────────────────────────────


def test_fetch_forecasts_pandas_mode_returns_only_forecast_stage_and_horizons():
    """Pandas-mode filter keeps only stage=forecast within [1, max_days_ahead]."""
    df = _make_partitions_df()
    dc = DataContainer(is_pandas_df=True, pandas_df=df)
    out = fetch_forecasts_at_training_cutoffs(
        dc=dc,
        training_cutoffs=[date(2024, 1, 6), date(2024, 1, 13)],
        max_days_ahead=5,
    )
    # 2 cutoffs × 5 horizons = 10 rows.
    assert len(out) == 10
    assert (out["days_ahead"] >= 1).all()
    assert (out["days_ahead"] <= 5).all()


def test_fetch_forecasts_pandas_mode_unknown_cutoff_returns_empty():
    """Cutoff not in pandas_df → empty result."""
    df = _make_partitions_df()
    dc = DataContainer(is_pandas_df=True, pandas_df=df)
    out = fetch_forecasts_at_training_cutoffs(
        dc=dc, training_cutoffs=[date(2099, 12, 31)], max_days_ahead=5,
    )
    assert out.empty


def test_fetch_forecasts_empty_cutoffs_returns_empty_with_columns():
    """Empty cutoffs → empty result with the expected column shape."""
    df = _make_partitions_df()
    dc = DataContainer(is_pandas_df=True, pandas_df=df)
    out = fetch_forecasts_at_training_cutoffs(
        dc=dc, training_cutoffs=[], max_days_ahead=5,
    )
    assert out.empty
    assert "last_training_date" in out.columns


# ─────────────────────────────────────────────────────────────────────────────
# list_training_cutoffs — pandas-mode
# ─────────────────────────────────────────────────────────────────────────────


def test_list_training_cutoffs_pandas_mode_returns_distinct_sorted():
    """Cutoffs come back unique + ascending."""
    df = _make_partitions_df()
    dc = DataContainer(is_pandas_df=True, pandas_df=df)
    cutoffs = list_training_cutoffs(dc=dc)
    assert cutoffs == [date(2024, 1, 6), date(2024, 1, 13)]


def test_list_training_cutoffs_lookback_days_filters_to_recent():
    """Lookback-days drops cutoffs older than the floor."""
    df = _make_partitions_df()
    dc = DataContainer(is_pandas_df=True, pandas_df=df)
    cutoffs = list_training_cutoffs(dc=dc, lookback_days=3)
    # 2024-01-13 is the latest; floor = 2024-01-10. Only 2024-01-13 survives.
    assert cutoffs == [date(2024, 1, 13)]


# ─────────────────────────────────────────────────────────────────────────────
# run_live_forecast_eval — pandas-mode end-to-end
# ─────────────────────────────────────────────────────────────────────────────


def test_run_live_forecast_eval_pandas_mode_smoke():
    """Pandas-mode end-to-end orchestrator returns a populated report."""
    dc = DataContainer(is_pandas_df=True, pandas_df=_make_partitions_with_observed_actuals())
    report = run_live_forecast_eval(
        dc=dc,
        training_cutoffs=[date(2024, 1, 6), date(2024, 1, 13)],
        max_days_ahead=5,
        # Include MAPE so the per-horizon error curve has data to plot — exercises
        # all three default figure makers end-to-end.
        metrics=("mae", "mape"),
    )
    assert isinstance(report, ForecastEvalReport)
    # End-to-end produces the default 3 figure sections, each with metric_a.
    assert len(report.figure_sections) == 3
    assert all("metric_a" in section.figures for section in report.figure_sections)
    # Forecast - actual = +1 (constant) → MAE = 1.0.
    assert report.errors_per_metric["mae"].unique().tolist() == [1.0]


def test_run_live_forecast_eval_transform_callback_invoked():
    """Optional ``transform`` is applied between fetch and evaluate."""
    dc = DataContainer(is_pandas_df=True, pandas_df=_make_partitions_with_observed_actuals())

    invoked: dict = {}

    def transform(forecast_df, actuals_df):
        """Identity transform that records arg sizes for the assertion."""
        invoked["called"] = True
        invoked["panel_rows"] = len(forecast_df)
        invoked["overlay_rows"] = len(actuals_df)
        return forecast_df, actuals_df

    run_live_forecast_eval(
        dc=dc,
        training_cutoffs=[date(2024, 1, 6), date(2024, 1, 13)],
        max_days_ahead=5,
        transform=transform,
        metrics=("mae",),
    )
    assert invoked == {
        "called": True,
        "panel_rows": invoked["panel_rows"],
        "overlay_rows": invoked["overlay_rows"],
    }
    assert invoked["panel_rows"] > 0
    assert invoked["overlay_rows"] > 0


def test_run_live_forecast_eval_forwards_figure_makers():
    """Custom ``figure_makers`` are forwarded to the pure-pandas core.

    Mirrors how ``live_weight_quality_report.py`` restricts the report to
    just the two overlay sections (skipping the per-horizon error curve).
    """
    dc = DataContainer(is_pandas_df=True, pandas_df=_make_partitions_with_observed_actuals())
    report = run_live_forecast_eval(
        dc=dc,
        training_cutoffs=[date(2024, 1, 6), date(2024, 1, 13)],
        max_days_ahead=5,
        metrics=("mae",),
        figure_makers=[make_per_training_cutoff_forecast_overlays()],
    )
    assert len(report.figure_sections) == 1
    assert report.figure_sections[0].name == "per_training_cutoff_forecast_overlays"


def test_run_live_forecast_eval_no_cutoffs_raises():
    """No cutoffs in lookback window → ValueError."""
    empty_df = pd.DataFrame(
        columns=[
            "metric_id", "last_training_date", "forecasted_date", "stage",
            "actual", "forecast", "forecast_lower", "forecast_upper",
        ]
    )
    dc = DataContainer(is_pandas_df=True, pandas_df=empty_df)
    with pytest.raises(ValueError, match="No training cutoffs"):
        run_live_forecast_eval(dc=dc, lookback_days=30)

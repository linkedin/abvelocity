# `forecast_transforms` — Architecture

**Author**: Reza Hosseini

This package derives one forecast from another by aggregating along
time or dim axes — the *post-forecast* layer that runs AFTER
`TSFlow.run` produces a forecast frame.

Two motivations:

1. **Consistency.** A weekly derived forecast = sum(daily forecasts)
   is guaranteed to add up. Independently-trained daily and weekly
   models can disagree.
2. **No extra fits.** A single daily fit yields weekly / monthly /
   annual / DoW-share / per-segment-share forecasts deterministically.

## The 4 ops

|              | Time axis              | Dim axis             |
|--------------|------------------------|----------------------|
| **Aggregate**| `SumOverPeriod`            | `SumOverDims`        |
| **Reweight** | `WeightOverPeriod`     | `WeightOverDims`     |

All four share `ForecastTransform` (`base.py`) — frozen dataclasses,
pure `apply(forecast_df) -> forecast_df`, plus a `str_name()` snippet
that contributes to the derived metric_id.

## Module layout

```
forecast_transforms/
├── __init__.py              public API: ForecastTransform + 4 classes
├── architecture.md          this file
│
├── column_classes.py        which columns play which role in the math
├── period.py                input-freq inference + expected-row-count
├── sigma.py                 σ propagation for sums and shares
├── bounds.py                z-score inference + bound recomputation
├── aggregation.py           sum_strict + the four "side" helpers
├── base.py                  ForecastTransform ABC
│
├── sum_over_period.py           SumOverPeriod
├── weight_over_period.py    WeightOverPeriod
├── sum_over_dims.py         SumOverDims
└── weight_over_dims.py      WeightOverDims
```

The five utility modules (`column_classes`, `period`, `sigma`,
`bounds`, `aggregation`) are independently testable.  The four
transforms each fit in ~100 lines because they orchestrate utilities
instead of inlining math.

## Column classes

Numeric columns on the forecast frame split into three roles
(`column_classes.py`):

| Class | Columns | Math under SUM | Math under SHARE |
|---|---|---|---|
| **POINT_SHAREABLE** | `actual`, `forecast` | `sum_strict` | `value / own_total` |
| **BREAKDOWN** | `longterm_growth`, `*_seasonality`, `holiday_impact`, `residual` | `sum_strict` | `value / total_forecast` |
| **SIGMA** | `std` | `sqrt(Σ σ²)` | `σ / |total_forecast|` |
| **BOUND** | `forecast_lower`, `forecast_upper` | recomputed from new (forecast, σ, z) | recomputed |

POINT_SUMMABLE = POINT_SHAREABLE ∪ BREAKDOWN — all columns that
combine additively.

Why breakdown uses `total_forecast` (universal denom) under share:
to preserve `Σ component_share = forecast_share` at the row level.
With own-column denominators, a near-zero component sum would blow up
into an unstable share.

Why σ uses an independent-Gaussian sum: variance of a sum of
independent random variables is the sum of variances.  This UNDERSTATES
sigma when day-to-day forecasts are correlated — a known approximation
called out in the docstrings.

Why σ uses a constant-denominator approximation under share: we treat
the observed group total as a known fact rather than an estimate, so
the delta method's σ_Y term drops.  Documented as `σ_X / |Y|` in
`sigma.propagate_sigma_share`.

## The three-pass + join structure

Each `Sum*` transform's `apply()` follows the same shape:

```
forecast frame in
       │
       ▼
[derive group key]               ── timestamp anchor + identity + dims
       │ (stage and other metadata cols excluded — see METADATA_COLS)
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
[forecast side]  [actual side]  [stage side]
 forecast/σ/      just `actual`   "forecast" if any
 bounds/                          row in group is forecast,
 breakdown                        else "fitted"
       │              │              │
       ▼              ▼              ▼
[sum w/ σ propag] [sum_strict]  [any-forecast-wins]
       │              │              │
       └─────── join on group key ───┘
       ▼
[completeness mask]              ── time axis only
       ▼
forecast frame out
```

The split keeps the actual side **skinny** — no variance plumbing, no
component handling, no bound recomputation.  The forecast side carries
all the math complexity.  The stage side is one line of pandas.

For `Weight*` transforms, rows aren't reduced (each row is reweighted
in place), so the "join" collapses to "two in-place passes on the same
frame" — forecast-side share + actual-side share.  Stage flows through
untouched (it's a string column, transforms only touch numeric
columns).  The helpers in `aggregation.py` handle both flavors.

## Time-period conventions

- **Period anchor** = period START (lower bound).  W-SAT week →
  Sunday; MS month → 1st; YS year → Jan 1.  Set by
  `dt.to_period(freq).dt.start_time.dt.normalize()`.
- **Completeness** is checked via
  `period.expected_count_in_period(start, end, input_freq)` which
  uses `pd.date_range` so monthly correctly expects 28 / 29 / 30 / 31
  by calendar month, and yearly 365 or 366 by leap year.  No hardcoded
  table.
- **Incomplete periods** are dropped (sum) or NaN'd (share).  A
  Mon-Wed slice in a Sun-Sat week never masquerades as a full week.

## Stage handling

The input frame's `stage` column (`"fitted"` vs `"forecast"`) is
**preserved** through every transform — downstream plotters use it to
draw the train-end cutoff line, and the canonical schema is the same
shape going in and coming out.

For `Sum*` transforms, stage is aggregated by **any-forecast-wins**
(`aggregate_stage_side`): a period / dim group is `"forecast"` if any
underlying row was forecast-stage, otherwise `"fitted"`.  This handles
the cutoff-straddling case (a week with 3 fitted + 4 forecast days
ends up `"forecast"`) without dropping the row.

For `Weight*` transforms, rows aren't reduced — stage flows through
unchanged on each input row.

Effect on the math:

- A week entirely in the training window: `actual` and `forecast` both
  sum normally; output stage = `"fitted"`.
- A week entirely in the forecast horizon: `actual` is all-NaN →
  `sum_strict` returns NaN; `forecast` sums normally; stage =
  `"forecast"`.
- A straddling week (cutoff mid-week): `actual` has some NaNs (the
  forecast-horizon days) → `sum_strict` returns NaN for the whole
  week's actual.  `forecast` is fully populated → sums normally.  No
  partial sums anywhere.  Stage = `"forecast"` (any-forecast-wins).

## Metadata columns

`METADATA_COLS` (`column_classes.py`) lists the TSRunner-stamped
non-grouping columns: `ts` (the timestamp duplicate of
`forecasted_date`), `last_training_date`, `algo_name`, `algo_version`,
`extras`, `run_id`, `run_date`.  `grouping_columns()` excludes these
by default so they don't split each row into its own group.  They pass
through aggregation unchanged when present.

## Identity columns

`metric_id`, `metric_name`, etc. flow through unchanged — they're part
of the group key when present.  Renaming a derived forecast (e.g.,
`randomProduct_new_signups:daily` → `randomProduct_new_signups:weekly`) is the
JobConfig layer's job.  This module only does the math.

## Caveats

- **σ propagation under sum is independent-Gaussian.**  Real
  day-to-day forecast errors have positive autocorrelation; the
  reported σ underestimates true uncertainty.
- **σ propagation under share is configurable** via the transform's
  `sigma_method` arg: ``"constant"`` (default — denominator treated as
  a known constant, drops the `w² · σ_Y²/Y²` term of the delta method)
  or ``"delta"`` (full delta-method propagation including denominator
  uncertainty).  ``"constant"`` understates uncertainty; ``"delta"``
  is more honest at the cost of slightly higher reported σ.
- **Actual-side share is all-or-nothing.**  If any row in a group has
  NaN actual (typical at the train/forecast boundary where the
  forecast horizon's actuals haven't landed yet), the whole group's
  actual share comes back NaN — partially-populated denominators give
  bogus shares, so we mask uniformly rather than divide by them.
- **Bound recomputation assumes Gaussian symmetry** (`upper - forecast
  = forecast - lower = z·σ`).  Greykite's default uncertainty model
  satisfies this; non-Gaussian models would need a different
  `recompute_bounds`.
- **No completeness check for dim aggregation.**  No canonical "all
  dim levels expected" concept; caller validates upstream.

## Where breakdown columns come from

The breakdown columns (`longterm_growth`, `weekly_seasonality`,
`annual_seasonality`, `holiday_impact`) are stamped on the forecast
frame by ``GreykiteForecastAlgo.predict()`` (in
``abvelocity.ts.algo``) — the algo calls greykite's
``forecast_breakdown`` and re-anchors the components per the
``algo_params["breakdown_origin"]`` setting (``"first_value"`` by
default; see ``BREAKDOWN_ORIGINS``).  This module just operates on the
columns when they're present; if a non-additive algo produced the
forecast frame, the columns are absent and the transforms still work
(the breakdown branch is a no-op).

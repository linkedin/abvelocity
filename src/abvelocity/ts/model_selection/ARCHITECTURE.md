# `model_selection` — Architecture

## Purpose

Pick the best forecasting model from a set of candidate parameter
configurations using **rolling-origin backtest** (not the algo's internal
CV) and a **persistent audit trail** of every candidate that was tried.

The package is intentionally external to the algo: any algo registered
in `ALGO_REGISTRY` (greykite, simple, …) plugs in via the standard
`TSAlgo` / `BackfillRunner` machinery. The selection layer doesn't know
or care which algo it's driving — it just patches `algo_params`,
runs the backtest, persists artifacts, and ranks.

## Module map

```
model_selection/
├── space.py            SearchSpace + ParamGroup — what to try
├── param_converter.py  ParamConverter ABC + IdentityParamConverter —
│                       translates flat search-space params to algo-specific
│                       config shape (e.g. greykite's nested dicts)
├── eval_criteria.py    EvalCriteria — how to score (which eval metrics,
│                       primary metric, reduction, lower-is-better)
├── model_candidates.py ModelCandidate (one row's record) +
│                       ModelCandidatesLog (the table) +
│                       compute_candidate_id (stable SHA-256 hash) +
│                       path helpers (predictions/, evals/, fits/)
├── base.py             ModelSelection ABC (template + persist + eval)
│                       SelectionResult dataclass
│                       evaluate_existing() helper for re-ranking with new criteria
├── grid.py             GridModelSelection — full cartesian sweep
├── grouped.py          GroupedModelSelection — paper-cited layered stepwise
│                       (Hosseini, Newlands, Dean, Takemura 2015 §3.4)
├── report.py           write_report — heat-mapped HTML + best-config callout
└── ARCHITECTURE.md     this file
```

## Key concepts

### `SearchSpace`
Ordered list of `ParamGroup`s. Each `ParamGroup` carries a mini-grid
`{param_name: [values]}` and metadata for grouped-stepwise (`augment`,
`reopen`).

* `GridModelSelection` flattens the whole space into one cartesian
  product.
* `GroupedModelSelection` walks groups in order, freezing each stage's
  winner before the next stage's mini-grid is evaluated.

### `ParamConverter`
Adapter between **search-space form** (flat, user-friendly,
`{"changepoint_reg": 0.6}`) and **algo_params form** (whatever the
algo expects — e.g. greykite's nested
`{"model_components": {"changepoints": {"changepoints_dict": {...}}}}`).

* `IdentityParamConverter` — no-op default for algos with flat config
  (e.g. `SimpleForecastAlgo`).
* `GreykiteParamConverter` — lives next to `GreykiteForecastAlgo`,
  ships with the algo. Recognises `changepoint_reg`, `fit_algorithm`,
  `regression_weight_col`, the seasonality keys, etc.

The audit trail (`model_candidates.csv`, `results.csv`, candidate ids)
always uses the **flat search-space form** regardless of conversion —
keeps rows human-readable.

### `EvalCriteria`
* `eval_metrics` — which accuracy metrics to compute on every persisted
  prediction frame (`mape`, `smape`, `medae`, `mae`, `rmse`).
* `group_by` — how to bucket the eval (default per-`(metric_id, horizon_step)`).
* `primary_eval_metric` + `primary_eval_reduction` + `lower_is_better`
  — collapse the per-group metric column to one scalar for ranking.

### `BackfillConfig`
The prediction template. Defines how training cutoffs are generated.
Two modes:
* **Algorithmic** (default): `initial_train_size`, `step`, `n_windows` —
  produces evenly-spaced cutoffs.
* **Explicit**: `cutoffs=["2025-06-15", ...]` — exact dates. Overrides
  the algorithmic spec when set.

Either way, the resolved cutoff dates are written to `cutoffs.json` in
the run's output directory at the start of every `run()`.

### `ModelCandidatesLog`
The on-disk registry of every candidate tried. Each row records params,
status, and pointers to the candidate's prediction CSV, eval CSV, and
optional fit_info JSON. Resumable via content-hashed `candidate_id`.

## Data flow

```
                          ┌─────────────────────────────────────┐
                          │        ModelSelection               │
                          │  ─────────────────────────────────  │
                          │  search_space    : SearchSpace      │
                          │  backfill_config : BackfillConfig   │
                          │  eval_criteria   : EvalCriteria     │
                          │  param_converter : ParamConverter?  │
                          │  output_dir      : Path             │
                          └────────────────┬────────────────────┘
                                           │  .run(df)
                                           ▼
              ┌──────────────────────────────────────────┐
              │  resolve cutoffs → cutoffs.json          │
              │  enumerate candidates                    │
              │  (Grid: cartesian. Grouped: stage-by-    │
              │   stage with prior winners frozen.)      │
              └─────────────────┬────────────────────────┘
                                │   for each candidate:
                                ▼
       ┌───────────────────────────────────────────────────┐
       │  PREDICT  (per candidate)                         │
       │  ───────                                          │
       │  param_converter(params) → algo_params override   │
       │  merge into BackfillConfig.forecast_config        │
       │  BackfillRunner(cfg).run(df)                      │
       │    → result_df (long-format: ts, metric_id,       │
       │                 actual, forecast, cutoff,         │
       │                 horizon_step, …)                  │
       └────────────────────────┬──────────────────────────┘
                                │
                                ▼
       ┌───────────────────────────────────────────────────┐
       │  PERSIST  (per candidate)                         │
       │  ─────────                                        │
       │  output_dir/                                      │
       │   ├── predictions/<id>.csv     ← long backtest    │
       │   ├── evals/<id>.csv           ← per-group metrics│
       │   ├── fits/<id>.json (opt.)    ← fit_info if any  │
       │   └── model_candidates.csv     ← registry row     │
       └────────────────────────┬──────────────────────────┘
                                │  after all candidates persisted
                                ▼
       ┌───────────────────────────────────────────────────┐
       │  EVAL + REPORT                                    │
       │  ──────────                                       │
       │  for row in model_candidates.csv:                 │
       │    metrics_df = compute_eval(predictions)         │
       │    score = criteria.primary_score(metrics_df)     │
       │  → ranked results.csv + heat-mapped results.html  │
       └────────────────────────┬──────────────────────────┘
                                ▼
                        SelectionResult
                        (best_params, best_score,
                         results_df, output_dir,
                         stage_winners — grouped only)
```

## Output directory contract

```
output_dir/
├── cutoffs.json              the cutoff dates this run used
├── model_candidates.csv      one row per candidate tried (audit trail)
├── predictions/
│   └── <candidate_id>.csv    long-format backtest result per candidate
├── evals/
│   └── <candidate_id>.csv    per-(metric_id, horizon_step) metrics frame
├── fits/                     optional, only when BackfillResult.fit_info exists
│   └── <candidate_id>.json
├── stage_winners.json        grouped runs only — running cumulative winner
│                              after each stage
├── results.csv               ranked candidates with one column per param +
│                              score + per-metric mean columns
└── results.html              heat-mapped report; open in any browser
```

`candidate_id` is the first 12 hex chars of SHA-256 over the canonical-JSON
of the candidate's params dict. Two re-runs with the same params produce
the same id, so a re-run skips already-persisted candidates (resumable).

## Public API sketch

```python
from abvelocity.ts.backfill.config import BackfillConfig
from abvelocity.ts.config.forecast_config import ForecastConfig
from abvelocity.ts.model_selection.eval_criteria import EvalCriteria
from abvelocity.ts.model_selection.grid import GridModelSelection
from abvelocity.ts.model_selection.grouped import GroupedModelSelection
from abvelocity.ts.model_selection.space import ParamGroup, SearchSpace

# 1. Define the prediction template (everything that's NOT being swept).
backfill_config = BackfillConfig(
    forecast_config=ForecastConfig(
        time_col="ts", value_cols=("y",), freq="D",
        forecast_horizon=21,
        algo_name="simple",
        algo_params={"period": 7, "k": 3, "agg": "mean"},
    ),
    initial_train_size=1095, horizon=21, step=29, n_windows=12,
)

# 2. Define what to sweep.
space = SearchSpace.flat({"k": [2, 3, 4], "agg": ["mean", "median"]})

# 3. Define how to score.
criteria = EvalCriteria(
    eval_metrics=("mape", "smape", "medae"),
    primary_eval_metric="mape",
    primary_eval_reduction="mean",
)

# 4. Run.
selection = GridModelSelection(
    search_space=space,
    backfill_config=backfill_config,
    eval_criteria=criteria,
    output_dir=Path("results/my_run"),
)
result = selection.run(df=daily_df)
print(result.best_params, result.best_score)
```

For grouped-stepwise selection (paper §3.4):

```python
space = SearchSpace(groups=[
    ParamGroup(name="lookback", params={"k": [2, 3, 4]}),
    ParamGroup(name="aggregation", params={"agg": ["mean", "median"]}),
])
selection = GroupedModelSelection(... eval_criteria=criteria ...)  # required for grouped
```

## Algo-agnostic via `ParamConverter`

For algos whose `algo_params` shape is non-flat (e.g. greykite), supply
a `ParamConverter`:

```python
import abvelocity.ts.algo.greykite_forecast_algo as gk

selection = GridModelSelection(
    search_space=SearchSpace.flat({
        "changepoint_reg":       [0.4, 0.6],
        "regression_weight_col": [None, "ct1"],
    }),
    backfill_config=...,
    eval_criteria=...,
    param_converter=gk.GreykiteParamConverter(),
    output_dir=...,
)
```

The user types `changepoint_reg=0.6`; the converter translates that to
`{"model_components": {"changepoints": {"changepoints_dict":
{"method": "auto", "regularization_strength": 0.6}}}}`. The audit trail
keeps the flat form.

## Forward direction

* **Generic `BaseModelSelection`** — split out the SearchSpace +
  ModelCandidatesLog + EvalCriteria plumbing from the BackfillRunner-
  specific bits, so non-TS algos can plug in via an abstract
  `predict_one(params, df)`. Current `ModelSelection` becomes a
  forecast specialisation. Public class names (`GridModelSelection`,
  `GroupedModelSelection`) stay stable.
* **Richer DAG-of-stages** for `GroupedModelSelection` — the precipitation-
  process paper allows graph-structured stage dependencies. v1 ships
  the linear form; the persistence layout (stage_idx / stage_name)
  already supports DAGs.
* **JS-sortable HTML report** — current `results.html` is a static
  table. A small JS sort layer would let users re-sort by any metric
  column without re-running.
* **Per-candidate component plots** — for greykite, the breakdown
  components (trend, weekly, yearly, holiday) are in the prediction
  CSVs; a per-candidate sub-report could render them. v2.
* **Likelihood-based eval metrics** (AIC / BIC / DIC) — the
  `fit_info_path` column already plumbs `BackfillResult.fit_info` to
  disk; a v2 `EvalCriteria.fit_metrics` field would consume it for
  ranking by penalised likelihood.

## Citations

The grouped-stepwise method is described in:

> Hosseini, R., Newlands, N. K., Dean, C. B., & Takemura, A. (2015).
> "Statistical Modeling of Soil Moisture, Integrating Satellite
> Remote-Sensing (SAR) and Ground-Based Data."
> *Remote Sensing* 7(3), 2752–2780.
> https://doi.org/10.3390/rs70302752

Specifically §3.4 ("Model Structure") and the discussion section's
description of the layered stepwise algorithm:

> "...we utilized the first of these approaches, devising a grouped,
> stepwise method that conducts an iterative search of the predictor
> space corresponding to a group of selected leading predictors. This
> extends regular stepwise methods to the multivariate case."

# Timeseries Flow — Architecture

## Overview

The `flow/` sub-package is the **end-to-end pipeline layer** for the timeseries module.
It wraps data fetching, algorithm execution, and optional evaluation into a single
class — `TSFlow` — that operates on one metric group at a time.

---

## Package layout

```
timeseries/flow/
    ARCHITECTURE.md        ← this file
    FLOW_DIAGRAM.md        ← end-to-end flow diagram (DAG → Spark → TSFlow → OH)
    flow_config.py         ← TSFlowConfig  (what to run and how)
    flow_result.py         ← TSFlowResult  (what came out)
    flow.py                ← TSFlow        (the pipeline class)
```

---

## Core classes

### `TSFlowConfig`

Bundles everything `TSFlow` needs:

| Field               | Type                      | Purpose                                                        |
|---------------------|---------------------------|----------------------------------------------------------------|
| `ts_metrics_config` | `TSMetricsConfig`         | Time-bucketing + SQL dialect for data fetching via `TSMetricsQuery` |
| `ts_model_config`   | `TSModelConfig` (subclass)     | Algo config — `ForecastConfig` or `DetectConfig`               |
| `mode`              | `str`                     | `"forecast"` / `"detect"` / `"backfill"`                       |
| `backfill_config`   | `BackfillConfig` (opt)    | Required when `mode="backfill"`; owns the `ForecastConfig`     |
| `prediction_window` | `(str, str)` (opt)        | ISO date range to restrict the output                          |
| `eval_metrics`      | `list[str]` (opt)         | Eval metric names to compute (see `timeseries/eval.py`)        |
| `eval_group_by`     | `tuple[str]` (opt)        | Group-by columns for eval; defaults to `(METRIC_ID_COL,)`      |

Validation in `__post_init__`:
- `mode` must be one of the three valid values.
- `backfill_config` required when `mode="backfill"`.
- `ts_model_config` required when `mode` is `"forecast"` or `"detect"`.

### `TSFlowResult`

Plain dataclass holding the output of one `TSFlow.run()` call:

| Field       | Type                     | Content                                         |
|-------------|--------------------------|-------------------------------------------------|
| `result_df` | `DataFrame` (opt)        | Long-format algo result (forecast / detect / backfill) |
| `eval_df`   | `DataFrame` (opt)        | Eval metrics; `None` if not requested           |
| `fit_info`  | `dict` (opt)             | Algo fit/backtest info keyed by value column    |

### `TSFlow`

The pipeline class. Stateless after construction — each `run()` call is independent.

```
TSFlow(flow_config, io_param=None)
    │
    ├── fetch_data(metric_info)          → wide DataFrame  [step 1]
    ├── run_algo(df, anomaly_df=None)    → ForecastResult | DetectResult | BackfillResult  [step 2]
    ├── compute_eval(result_df)          → eval DataFrame or None  [step 3]
    │
    └── run(metric_info, df=None, anomaly_df=None) → TSFlowResult  [orchestrates 1-2-3]
```

Passing `df` directly to `run()` skips `fetch_data` — useful for testing or when
data is already in memory.

---

## Data flow

```
MetricInfo + TSMetricsConfig
        │
        ▼ fetch_data()
  wide DataFrame
  (one row per time_bucket × dims)
        │
        ▼ run_algo()
  ┌─────────────────────────────────────┐
  │ mode="forecast"  → ForecastRunner   │
  │ mode="detect"    → AnomalyDetect-   │
  │                    Runner           │
  │ mode="backfill"  → BackfillRunner   │
  └─────────────────────────────────────┘
        │
        ▼
  long-format result_df
  (ts, metric_id, actual, forecast, ...
   + scheduled-pipeline columns stamped by TSRunner)
        │
        ▼ compute_eval()  [optional]
  eval_df  (mae, rmse, bias, sigma, ...)
        │
        ▼
  TSFlowResult(result_df, eval_df, fit_info)
```

---

## Mode reference

| mode        | Config required     | Runner              | Result type      |
|-------------|---------------------|---------------------|------------------|
| `forecast`  | `ForecastConfig`    | `ForecastRunner`    | `ForecastResult` |
| `detect`    | `DetectConfig`      | `AnomalyDetectRunner` | `DetectResult` |
| `backfill`  | `BackfillConfig`    | `BackfillRunner`    | `BackfillResult` |

---

## Relationship to the rest of the timeseries module

```
timeseries/
    config/         ← TSModelConfig, ForecastConfig, DetectConfig
    algo/           ← ALGO_REGISTRY, SimpleForecastAlgo, GreykiteForecastAlgo, ...
    result/         ← TSResult, ForecastResult, DetectResult
    backfill/       ← BackfillConfig, BackfillRunner, BackfillResult
    get_data/       ← TSMetricsConfig, TSMetricsQuery
    eval.py         ← compute_eval()
    forecast_runner.py / detect_runner.py  ← typed runner facades
    flow/           ← TSFlowConfig, TSFlowResult, TSFlow   ← YOU ARE HERE
```

`TSFlow` sits at the top of the call stack; it delegates downward to the typed
runners, which in turn delegate to `TSRunner` → `ALGO_REGISTRY` → the concrete
algo class.

---

## Spark scaling note

`TSFlow` is intentionally stateless after construction — `run()` reads from
`self.flow_config` and `self.io_param` but writes nothing back. To scale across
metric groups in Spark, replace sequential calls to `run()` with a distributed
`map` over the metric group list; each invocation is safe to parallelize.

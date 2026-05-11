# End-to-End Flow Diagram

How a single scheduled-pipeline run travels from DAG trigger to persisted
forecast rows. `TSFlow` sits in the middle — everything above it is caller
orchestration, everything below is the algo stack.

---

## Bird's-eye view

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SCHEDULED DAG (Airflow)                            │
│                                                                         │
│   trigger → build MetricInfo + TSFlowConfig + IOParam → launch Spark    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SPARK JOB                                       │
│                                                                         │
│   for each metric_group in metric_groups:                               │
│     result = TSFlow(flow_config, io_param).run(metric_info)             │
│     stamp run_id / run_date on result.result_df                         │
│     write result.result_df → Warehouse (MERGE INTO)                     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              TSFlow                                     │
│                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│   │  fetch_data │ ──► │  run_algo   │ ──► │ compute_eval│               │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘               │
│          │                   │                   │                      │
│          ▼                   ▼                   ▼                      │
│     wide DataFrame     long result_df         eval_df                   │
│                        (schema stamped)                                 │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ returns
                                ▼
                         TSFlowResult(
                           result_df,
                           eval_df,
                           fit_info,
                         )
```

---

## Step 1 — fetch_data

```
MetricInfo                  TSMetricsConfig
(table, metrics, dims,      (time bucketing +
 start_date, end_date)       SQL dialect)
       │                           │
       └─────────────┬─────────────┘
                     ▼
              TSMetricsQuery
                     │
                     │  .get_df(cursor)
                     ▼
              ┌──────────────┐
              │  io_param    │
              │  .cursor     │  ← LiCursor / Trino / Spark
              └──────┬───────┘
                     ▼
          wide-format DataFrame
          (one row per time_bucket × dims,
           one column per metric)
```

Skipped when a pre-fetched `df` is passed directly to `TSFlow.run()`.

---

## Step 2 — run_algo

```
                  flow_config.mode
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   "forecast"      "detect"        "backfill"
        │               │               │
        ▼               ▼               ▼
 ForecastRunner  AnomalyDetectRunner  BackfillRunner
        │               │               │
        └───────┬───────┘               │
                │                       │
                ▼                       ▼
           TSRunner.run()       BackfillRunner.run()
                │                       │
                │  get_algo()           │  slides cutoff,
                │  ALGO_REGISTRY        │  calls ForecastRunner
                ▼                       │  per cutoff
           TSAlgo.fit()                 │
                │                       │
                ▼                       │
           TSAlgo.predict()             │
                │                       │
                ▼                       ▼
         result_df            result_df (+ cutoff, horizon_step)
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────────┐
               │  _stamp_pipeline_columns() │
               │                            │
               │  • metric_id  (template)   │
               │  • metric_name (template)  │
               │  • stage (fitted/forecast) │
               │  • forecasted_date         │
               │  • last_training_date      │
               │  • algo_name / algo_version│
               │  • components + std + extras (NaN default) │
               │  • run_id / run_date (placeholders)        │
               └────────────┬───────────────┘
                            │
                            ▼
             long-format result_df with
             the full scheduled-pipeline
             column set (see TSResultRow)
```

---

## Step 3 — compute_eval (optional)

```
           flow_config.eval_metrics set?
                        │
             ┌──────────┴──────────┐
            no                    yes
             │                     │
             ▼                     ▼
        eval_df = None       compute_eval(result_df,
                                metrics=…,
                                group_by=eval_group_by
                                          or (METRIC_ID_COL,))
                                    │
                                    ▼
                                 eval_df
                                 (mae, rmse, mape, bias, sigma, …)
```

---

## Result → Persistence

```
TSFlowResult
      │
      ▼
 caller stamps:
    result_df[RUN_ID_COL]   = <dag_run_id>
    result_df[RUN_DATE_COL] = <execution_date>
      │
      ▼
 project onto FORECAST_TABLE_COLUMNS
 (canonical write order from constants.py)
      │
      ▼
 LiCursor.gen_insert_partition_from_source(...)
    partitioned by last_training_date
      │
      ▼
 Warehouse (Table) forecast table
```

---

## Scaling note

`TSFlow` is stateless after construction — each `run(metric_info, …)` call
is independent and safe to parallelize. The Spark layer maps metric groups
across executors; each executor builds its own `TSFlow` from the shared
`flow_config` and calls `run()` per group.

---

## Class touchpoints, in call order

| Layer              | Class                              | Module                                         |
|--------------------|------------------------------------|------------------------------------------------|
| Orchestrator (DAG) | *(caller — abvelocity-airflow)*     | —                                              |
| Orchestrator (job) | *(caller — abvelocity-spark)*       | —                                              |
| Pipeline           | `TSFlow`                           | `flow/flow.py`                                 |
| Config             | `TSFlowConfig`                     | `flow/flow_config.py`                          |
| Data fetch         | `TSMetricsQuery`                   | `get_data/ts_metrics_query.py`                 |
| Typed runner       | `ForecastRunner` / `AnomalyDetectRunner` / `BackfillRunner` | `forecast_runner.py` / `detect_runner.py` / `backfill/runner.py` |
| Generic runner     | `TSRunner`                         | `runner.py`                                    |
| Registry           | `ALGO_REGISTRY`                    | `algo/base.py`                                 |
| Algo               | `TSAlgo` subclasses (Simple, Greykite, …) | `algo/*.py`                             |
| Row schema         | `TSResultRow`                      | `result/ts_result_row.py`                      |
| Result             | `TSResult` / `ForecastResult` / `DetectResult` | `result/*.py`                      |
| Eval               | `compute_eval`                     | `eval.py`                                      |
| Output             | `TSFlowResult`                     | `flow/flow_result.py`                          |

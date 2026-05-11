# `timeseries/` — Architecture

## Purpose

A pluggable, algo-swappable scaffold for multivariate **time-series forecasting
and anomaly detection**. Design goals:

- **Fixed output schema** — result DataFrames always have the same columns
  regardless of how many metrics are requested.
- **Algo-agnostic core** — swap algorithms by name; no hard dependency on any
  specific library in the shared code.
- **Multivariate by default** — `value_cols` is a tuple; each algorithm loops
  over metrics internally and stacks results into a single long-format
  DataFrame.
- **JSON-serializable everywhere** — configs and results round-trip through
  JSON via mashumaro `DataClassJSONMixin`.

---

## Package layout

```
timeseries/
├── ARCHITECTURE.md          ← this file
├── constants.py             ← fixed output column names
├── config/
│   ├── ts_config.py         ← TSModelConfig (base config)
│   ├── forecast_config.py   ← ForecastConfig (adds forecast_horizon)
│   └── detect_config.py     ← DetectConfig (adds nested forecast_config)
├── result/
│   ├── ts_result.py         ← TSResult (base result: result_df + rows + fit_info)
│   ├── ts_result_row.py     ← TSResultRow (row-level dataclass schema; covers forecast + detection)
│   ├── forecast_result.py   ← ForecastResult (subclass, no extra fields yet)
│   └── detect_result.py     ← DetectResult (adds anomalies_df)
├── algo/
│   ├── base.py              ← TSAlgo abstract base + ALGO_REGISTRY
│   ├── simple_forecast_algo.py     ← "simple" seasonal mean algo
│   ├── greykite_forecast_algo.py   ← greykite forecast impl (conditional import)
│   └── greykite_detect_algo.py     ← greykite detect stub (conditional import)
├── runner.py                ← TSRunner generic core (fit → predict)
├── forecast_runner.py       ← ForecastRunner typed facade (ForecastConfig → ForecastResult)
├── detect_runner.py         ← AnomalyDetectRunner typed facade (DetectConfig → DetectResult)
├── backfill/
│   ├── config.py            ← BackfillConfig (sliding-window parameters)
│   ├── result.py            ← BackfillResult
│   └── runner.py            ← BackfillRunner (multi-cutoff historical forecasts)
├── eval.py                  ← compute_eval (mae/rmse/mape/smape/r2/medae/coverage/bias/sigma/quartiles)
└── viz.py                   ← plot_forecast_vs_actual / add_multi_vrects / plot_forecast
```

---

## Data flow

```
caller
  │
  │  ForecastConfig / DetectConfig
  │  (time_col, value_cols, freq, coverage, algo_name, algo_params, ...)
  ▼
ForecastRunner.run()  or  AnomalyDetectRunner.run()   ← typed public API
  │  delegates to
  ▼
TSRunner.run(df, prediction_window, anomaly_df)        ← generic core
  │
  │  looks up ALGO_REGISTRY[config.algo_name]
  ▼
TSAlgo.fit(df, config, anomaly_df)   ← trains the model
  │
TSAlgo.predict(prediction_window)    ← returns ForecastResult / DetectResult
  │
  ▼
result_df  (long format, fixed schema — see below)
```

For historical evaluation or backfilling a forecast store, wrap `ForecastRunner`
with `BackfillRunner` — it slides the training cutoff forward and collects one
`ForecastResult` per cutoff, tagged with `cutoff` and `horizon_step`.

---

## Config hierarchy

```
TSModelConfig                  ← base: time_col, value_cols, dim_cols, regressor_cols,
  │                              freq, train_end_date, coverage,
  │                              algo_name, algo_version, algo_params,
  │                              metric_id_template, metric_name_template
  ├── ForecastConfig      ← adds: forecast_horizon
  └── DetectConfig        ← adds: forecast_config (Optional[ForecastConfig])
```

**`DetectConfig.algo_name / algo_params`** control the *detection* algorithm.
**`DetectConfig.forecast_config.algo_name / algo_params`** control the
*forecast* algorithm used inside the detection step. This mirrors the
greykite-detection pattern of two independent sibling configs passed to
`GreykiteDetector`.

**`prediction_window`** (`tuple[str, str]`) is passed to the runner's `run()`,
not stored in the config. This follows the oi-schemas split between
`AlgoConfig` (what the model is) and `PredictionRequest` (what window to
score).

All config classes are `@dataclass` + `DataClassJSONMixin` and
JSON-round-trip cleanly.

---

## Result hierarchy

```
TSResult                  ← result_df (pandas) + rows (typed list) + fit_info
  ├── ForecastResult      ← no extra fields; subclassing enables future split
  └── DetectResult        ← adds anomalies_df
```

`TSResult` exposes two views of the same data:

- **`result_df`** — pandas DataFrame (hot-path runtime container).
- **`rows`** — optional `list[TSResultRow]` (for JSON / gRPC / test fixtures).

Conversion is explicit: `to_df()`, `to_rows()`, `from_df()`, `from_rows()`.
Not auto-synced — callers populate whichever representation they produced.

### Fixed output schema

The canonical row schema is defined by `TSResultRow` (in
`result/ts_result_row.py`) — the single source of truth for column names,
Python types, and pandas dtypes. The same dataclass covers both forecast and
anomaly-detection outputs: forecast-only and detection-only fields are
`Optional` and default to `NaN`/`None` in rows from the other mode.

`result_df` is long format — one row per **(timestamp × metric_id)**:

| Column               | Type    | Notes                                      |
|----------------------|---------|--------------------------------------------|
| `ts`                 | datetime | Timestamp (name follows `config.time_col`) |
| `metric_id`          | str      | Machine key (rendered from `metric_id_template` when set) |
| `metric_name`        | str      | Human label (rendered from `metric_name_template`) |
| `stage`              | str      | `"fitted"` when `actual` is observed, `"forecast"` otherwise |
| `forecasted_date`    | date     | DATE projection of `ts`                    |
| `last_training_date` | date     | End of training window                     |
| `actual`             | float    | Observed value; `NaN` for future periods   |
| `forecast`           | float    | Point forecast                             |
| `forecast_lower`     | float    | Lower prediction-interval bound            |
| `forecast_upper`     | float    | Upper prediction-interval bound            |
| `std`                | float    | Forecast std-dev; `NaN` for quantile-based algos |
| `longterm_growth`, `shortterm_growth`, `daily_seasonality`, `weekly_seasonality`, `annual_seasonality`, `holiday_impact`, `residual` | float | Additive decomposition components; `NaN` when algo doesn't produce them |
| `is_anomaly`, `is_anomaly_predicted`, `anomaly_score`, `anomaly_severity` | mixed | Detection-only; `NaN`/`None` for forecast rows |
| `extras`             | dict    | Open-ended `dict[str, float]` for algo-specific components |
| `algo_name`          | str      | Algorithm name (from `config.algo_name`)  |
| `algo_version`       | str      | Algorithm version (from `config.algo_version`) |
| `run_id`             | str      | DAG run id; stamped by the caller         |
| `run_date`           | date     | DAG run date; stamped by the caller       |

Scheduled-pipeline columns (`metric_id`, `metric_name`, `stage`,
`forecasted_date`, `last_training_date`, `algo_name`, `algo_version`,
components, `extras`, `run_id`, `run_date`) are stamped by
`TSRunner._stamp_pipeline_columns()` after `predict()` returns. Columns the
algorithm didn't populate are filled with `NaN`/`None` so the column set is
uniform regardless of algo.

`metric_id_template` / `metric_name_template` (on `TSModelConfig`) use
format-string placeholders: `{value_col}` resolves to the algo-stamped value
column name, any `{dim_name}` resolves to the corresponding dim column.
`__post_init__` validates placeholders against `value_cols` + `dim_cols`.

Adding a new `value_col` produces more **rows**, never new columns.

`anomalies_df` (on `DetectResult`) is also long format:

| Column        | Type    | Notes                    |
|---------------|---------|--------------------------|
| `metric_id`   | str     | Value-column name        |
| `start_ts`    | str/ts  | Anomaly interval start   |
| `end_ts`      | str/ts  | Anomaly interval end     |

All column names are defined as constants in `constants.py`. The canonical
write-order tuple is `FORECAST_TABLE_COLUMNS` — downstream persistence
modules rely on the ordering when constructing write SQL.

---

## Algorithm registry

```python
# algo/base.py
ALGO_REGISTRY: dict[str, type[TSAlgo]] = {}
```

Algo modules self-register at import time:

```python
# greykite_forecast_algo.py  (simplified)
if GREYKITE_AVAILABLE:
    ALGO_REGISTRY["greykite"] = GreykiteForecastAlgo
```

`TSRunner.get_algo()` does `ALGO_REGISTRY[config.algo_name]` and raises a
descriptive `ValueError` (listing available names) if the key is missing.

**Important:** algo modules self-register at *import time*. If the module has
not been imported before `TSRunner.run()` is called, the registry lookup will
fail. Callers must import the algo module explicitly before running:

```python
import abvelocity.ts.algo.greykite_forecast_algo  # registers "greykite"
```

To plug in a new algorithm:

1. Create `algo/my_algo.py` — subclass `TSAlgo`, implement `fit` and `predict`.
2. At the bottom: `ALGO_REGISTRY["my_algo"] = MyAlgo`.
3. Import the module before `TSRunner.run()` is called.

---

## `TSAlgo` base class

```python
@dataclass
class TSAlgo:
    algo_params: Optional[Dict[str, Any]] = None   # normalised to {} in __post_init__

    @abstractmethod
    def fit(self, df, config, anomaly_df=None) -> "TSAlgo": ...

    @abstractmethod
    def predict(self, df=None, prediction_window=None) -> TSResult: ...
```

Uses `@dataclass` + `@abstractmethod` (no `ABC` base), consistent with
`stats/estimator.py`.

**`anomaly_df`** is passed at *train time* to mask known-bad periods before
fitting — it is not a prediction-time artifact. Columns: `start_ts`, `end_ts`.

---

## Greykite implementation (`greykite_forecast_algo.py`)

`blah.greykite` is an optional dependency; the module guards with
`try/except ImportError` and only defines/registers the class when available.

> Both algo files import from `blah.greykite.*` (Blah's internal
> fork). The `greykite-framework` and `greykite-detection` satellites are
> declared in `abvelocity/build.gradle`.

### Multivariate loop

greykite is univariate. `GreykiteForecastAlgo.fit` loops over each
`value_col`, calls `GKForecaster().run_forecast_config(df_col, gk_config)`,
and stores the greykite `ForecastResult` per metric in `self.gk_results`.

### Config translation (`build_gk_config`)

| `TSModelConfig` / `ForecastConfig` field | greykite equivalent                          |
|-------------------------------------|----------------------------------------------|
| `time_col`                          | `MetadataParam.time_col`                     |
| `value_col` (one at a time)         | `MetadataParam.value_col`                    |
| `freq`                              | `MetadataParam.freq`                         |
| `train_end_date`                    | `MetadataParam.train_end_date`               |
| `coverage`                          | `GKForecastConfig.coverage`                  |
| `forecast_horizon`                  | `GKForecastConfig.forecast_horizon`          |
| `regressor_cols`                    | `ModelComponentsParam.regressors`            |
| `anomaly_df` (`start_ts`/`end_ts`)  | `MetadataParam.anomaly_info` — column names passed via `start_time_col`/`end_time_col`; no rename needed |
| `algo_params["model_template"]`     | `GKForecastConfig.model_template`            |
| `algo_params["model_components"]`   | `ModelComponentsParam(**...)`                |

### Output assembly

greykite already uses `actual`, `forecast`, `forecast_lower`,
`forecast_upper` as column names — these exactly match our fixed schema.
`predict()` adds a `metric_id` column to each per-col DataFrame and
`pd.concat`s them (long format). A `prediction_window` filter is applied
after the concat.

`fit_info` is populated from `gk_result.backtest.test_evaluation` per metric
(backtest metrics are more reliable than CV metrics for production monitoring).

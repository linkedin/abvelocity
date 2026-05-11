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
"""Fixed-schema column name constants for time-series result DataFrames.

All result DataFrames (``result_df`` on :class:`TSResult` and subclasses)
share the same fixed column set regardless of how many value columns were
requested:

+-----------------------+-----------------------------------------------+
| Column                | Meaning                                       |
+=======================+===============================================+
| ``ts``                | Timestamp                                     |
+-----------------------+-----------------------------------------------+
| ``metric_id``         | Value-column name / metric key                |
|                       | (one row per ts × metric_id)                  |
+-----------------------+-----------------------------------------------+
| ``actual``            | Observed value (``NaN`` for future periods)   |
+-----------------------+-----------------------------------------------+
| ``forecast``          | Point forecast                                |
+-----------------------+-----------------------------------------------+
| ``forecast_lower``    | Lower prediction-interval bound               |
+-----------------------+-----------------------------------------------+
| ``forecast_upper``    | Upper prediction-interval bound               |
+-----------------------+-----------------------------------------------+
| ``algo_name``         | Algorithm name that produced the row          |
+-----------------------+-----------------------------------------------+
| ``last_training_date``| End of training window (date)                 |
+-----------------------+-----------------------------------------------+

Anomaly detection adds ``anomaly`` and ``anomaly_score`` to ``result_df``,
and ``anomalies_df`` uses ``metric_id``, ``start_ts``, ``end_ts``.

Scheduled-pipeline additions
----------------------------

For scheduled pipelines that persist forecasts to an external table,
the following additional columns are stamped on ``result_df`` by
:class:`~abvelocity.ts.runner.TSRunner` when the relevant
config fields are set (otherwise the columns are added with NULL /
``NaN``, so downstream consumers always see the same column set):

+-----------------------+--------+--------------------------------------+
| Column                | Type   | Meaning                              |
+=======================+========+======================================+
| ``metric_id``         | str    | Machine key; dims encoded inline as  |
|                       |        | ``<base>:<k1>=<v1>|<k2>=<v2>``       |
+-----------------------+--------+--------------------------------------+
| ``metric_name``       | str    | Human label distinct from metric_id  |
+-----------------------+--------+--------------------------------------+
| ``stage``             | str    | ``"fitted"`` (in-sample) or          |
|                       |        | ``"forecast"`` (out-of-sample).      |
|                       |        | Derived from whether ``actual`` is   |
|                       |        | observed.                            |
+-----------------------+--------+--------------------------------------+
| ``std``               | float? | Forecast std-dev; NaN for            |
|                       |        | quantile-based algos.                |
+-----------------------+--------+--------------------------------------+
| ``algo_name``         | str    | Algorithm name (from                 |
|                       |        | ``config.algo_name``).               |
+-----------------------+--------+--------------------------------------+
| ``algo_version``      | str    | Algorithm version (from              |
|                       |        | ``config.algo_version``).            |
+-----------------------+--------+--------------------------------------+
| ``last_training_date``| date   | Alias of ``train_end_date`` as       |
|                       |        | a proper DATE for OH partitioning.   |
+-----------------------+--------+--------------------------------------+
| ``forecasted_date``   | date   | DATE projection of ``ts`` for OH     |
|                       |        | convenience.                         |
+-----------------------+--------+--------------------------------------+
| ``longterm_growth``,  | float? | Additive decomposition components    |
| ``shortterm_growth``, |        | (NaN if algo doesn't produce them).  |
| ``daily_seasonality``,|        |                                      |
| ``weekly_seasonality``|        |                                      |
| ``annual_seasonality``|        |                                      |
| ``holiday_impact``    |        |                                      |
| ``residual``          |        |                                      |
+-----------------------+--------+--------------------------------------+
| ``extras``            | dict?  | Open-ended ``dict[str, float]``      |
|                       |        | for algo-specific components.        |
|                       |        | Keys prefixed ``regressor:``,        |
|                       |        | ``holiday:``, ``event:``, or bare.   |
+-----------------------+--------+--------------------------------------+
| ``run_id``            | str    | DAG run identifier (stamped by       |
|                       |        | caller; NaN if absent).              |
+-----------------------+--------+--------------------------------------+
| ``run_date``          | date   | DAG run date (stamped by caller;     |
|                       |        | NaN if absent).                      |
+-----------------------+--------+--------------------------------------+
"""

TIME_COL = "ts"
METRIC_ID_COL = "metric_id"
METRIC_NAME_COL = "metric_name"
STAGE_COL = "stage"
ACTUAL_COL = "actual"
FORECAST_COL = "forecast"
STD_COL = "std"
FORECAST_LOWER_COL = "forecast_lower"
FORECAST_UPPER_COL = "forecast_upper"
ANOMALY_COL = "anomaly"
ANOMALY_SCORE_COL = "anomaly_score"
START_TS_COL = "start_ts"
END_TS_COL = "end_ts"
ALGO_NAME_COL = "algo_name"
ALGO_VERSION_COL = "algo_version"
LAST_TRAINING_DATE_COL = "last_training_date"
FORECASTED_DATE_COL = "forecasted_date"
LONGTERM_GROWTH_COL = "longterm_growth"
SHORTTERM_GROWTH_COL = "shortterm_growth"
DAILY_SEASONALITY_COL = "daily_seasonality"
WEEKLY_SEASONALITY_COL = "weekly_seasonality"
ANNUAL_SEASONALITY_COL = "annual_seasonality"
HOLIDAY_IMPACT_COL = "holiday_impact"
RESIDUAL_COL = "residual"
EXTRAS_COL = "extras"
RUN_ID_COL = "run_id"
RUN_DATE_COL = "run_date"
CUTOFF_COL = "cutoff"
HORIZON_STEP_COL = "horizon_step"

# Stage values.
STAGE_FITTED = "fitted"
STAGE_FORECAST = "forecast"

# Exhaustive scheduled-pipeline column list in write order. Downstream
# persistence modules flatten ``result_df`` to this column set and rely
# on the ordering when constructing their write SQL.
FORECAST_TABLE_COLUMNS: tuple[str, ...] = (
    LAST_TRAINING_DATE_COL,
    METRIC_ID_COL,
    METRIC_NAME_COL,
    STAGE_COL,
    FORECASTED_DATE_COL,
    FORECAST_COL,
    STD_COL,
    FORECAST_LOWER_COL,
    FORECAST_UPPER_COL,
    ACTUAL_COL,
    LONGTERM_GROWTH_COL,
    SHORTTERM_GROWTH_COL,
    DAILY_SEASONALITY_COL,
    WEEKLY_SEASONALITY_COL,
    ANNUAL_SEASONALITY_COL,
    HOLIDAY_IMPACT_COL,
    RESIDUAL_COL,
    EXTRAS_COL,
    ALGO_NAME_COL,
    ALGO_VERSION_COL,
    RUN_DATE_COL,
    RUN_ID_COL,
)

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
"""Greykite-based time-series forecasting algorithm.

Conditionally imports ``blah.greykite``. If the package is not installed
the module still loads cleanly and ``GreykiteForecastAlgo`` is simply not
registered, mirroring the ``cloudNotebook_cursor.py`` pattern.

Design notes
------------
- **Univariate loop**: greykite is univariate, so ``fit`` loops over each
  ``value_col`` and runs an independent ``Forecaster.run_forecast_config``
  call. Results are stored in ``gk_results`` and merged in ``predict``.
- **anomaly_df** columns ``start_ts`` / ``end_ts`` (our convention) are
  renamed to ``start_time`` / ``end_time`` for greykite's ``anomaly_info``.
  The anomaly delta is left as ``NaN`` (mask to NaN strategy).
- **regressors**: if ``config.regressor_cols`` is non-empty, regressor
  columns are passed in the DataFrame *and* declared via
  ``ModelComponentsParam.regressors``. The caller is responsible for
  providing future regressor values alongside historical data.
- **forecast_horizon**: read from ``ForecastConfig.forecast_horizon`` if
  ``config`` is a ``ForecastConfig``, otherwise from
  ``algo_params["forecast_horizon"]``, otherwise defaults to 1.
- **model_template**: controlled via ``algo_params["model_template"]``;
  defaults to ``"SILVERKITE"``.
"""

try:
    from abvelocity.ts.common.constants import PREDICTED_COL, PREDICTED_LOWER_COL, PREDICTED_UPPER_COL
    from abvelocity.ts.common.constants import TIME_COL as GK_TIME_COL
    from abvelocity.ts.gk.framework.templates.autogen.forecast_config import ComputationParam, EvaluationPeriodParam
    from abvelocity.ts.gk.framework.templates.autogen.forecast_config import ForecastConfig as GKForecastConfig
    from abvelocity.ts.gk.framework.templates.autogen.forecast_config import MetadataParam, ModelComponentsParam
    from abvelocity.ts.gk.framework.templates.forecaster import Forecaster as GKForecaster

    GREYKITE_AVAILABLE = True
except ImportError:
    GREYKITE_AVAILABLE = False

if GREYKITE_AVAILABLE:
    from dataclasses import dataclass
    from typing import Any, Dict, Optional

    import pandas as pd
    from abvelocity.ts.algo.base import ALGO_REGISTRY, TSAlgo
    from abvelocity.ts.common.constants import DETAILED_SEASONALITY_COMPONENTS_REGEX_DICT
    from abvelocity.ts.config.forecast_config import ForecastConfig
    from abvelocity.ts.config.ts_model_config import TSModelConfig
    from abvelocity.ts.constants import (
        ACTUAL_COL,
        ANNUAL_SEASONALITY_COL,
        FORECAST_COL,
        FORECAST_LOWER_COL,
        FORECAST_UPPER_COL,
        HOLIDAY_IMPACT_COL,
        LONGTERM_GROWTH_COL,
        METRIC_ID_COL,
        STD_COL,
        WEEKLY_SEASONALITY_COL,
    )
    from abvelocity.ts.result.forecast_result import ForecastResult
    from scipy.stats import norm

    # ------------------------------------------------------------------
    # Breakdown extraction — turns greykite's design-matrix grouping
    # into our canonical breakdown columns.  Used inside
    # GreykiteForecastAlgo.predict().
    # ------------------------------------------------------------------

    BREAKDOWN_ORIGINS = ("first_value", "centered", "raw")
    """Allowed values for ``algo_params["breakdown_origin"]``.

    - ``"first_value"`` (default): each component's value at the first
      timestamp is subtracted; the first value is folded into Intercept.
      Components *start* at 0 and drift from there.  Easiest to read on
      a plot — the trend rise / seasonality swing is the value itself.
    - ``"centered"``: each non-intercept component has zero mean over
      the breakdown window; the mean is folded into Intercept.
      Components oscillate symmetrically around 0.  Matches greykite's
      ``plot_components`` convention.
    - ``"raw"``: greykite's raw breakdown — components carry whatever
      absolute level the design-matrix columns fitted to.
    """

    BREAKDOWN_GROUP_TO_COL = {
        "Trend": LONGTERM_GROWTH_COL,
        "Weekly": WEEKLY_SEASONALITY_COL,
        "Yearly": ANNUAL_SEASONALITY_COL,
        "Event": HOLIDAY_IMPACT_COL,
    }
    """Map greykite breakdown group name → canonical abvelocity column.

    ``Trend`` is special-cased: combined with ``Intercept`` to form the
    full level (longterm_growth) so the level component carries the
    response baseline rather than dangling around 0.
    """

    def extract_breakdown_components(gk_result: Any, origin: str = "first_value") -> Optional["pd.DataFrame"]:
        """Pull silverkite's breakdown breakdown into a date-indexed frame.

        Args:
            gk_result: Greykite ``ForecastResult`` from
                ``Forecaster.run_forecast_config()``.
            origin: Re-anchor mode — see :data:`BREAKDOWN_ORIGINS`.

        Returns:
            Date-indexed DataFrame with one column per breakdown group
            (``Trend``, ``Weekly``, ``Yearly``, ``Event``, ...).  Returns
            ``None`` when the model isn't an additive regression
            (breakdown only works for those).

        Raises:
            ValueError: When ``origin`` isn't in :data:`BREAKDOWN_ORIGINS`.
        """
        if origin not in BREAKDOWN_ORIGINS:
            raise ValueError(f"breakdown_origin must be one of {BREAKDOWN_ORIGINS}; got {origin!r}.")

        estimator = gk_result.forecast.estimator
        if estimator.model_dict is None:
            return None
        try:
            breakdown_dict = estimator.forecast_breakdown(
                grouping_regex_patterns_dict=DETAILED_SEASONALITY_COMPONENTS_REGEX_DICT,
                center_components=False,
            )
        except (AttributeError, ValueError, KeyError):
            return None

        breakdown_df = breakdown_dict["breakdown_df_with_index_col"].copy()
        time_col = estimator.model_dict["time_col"]
        breakdown_df[time_col] = pd.to_datetime(breakdown_df[time_col]).dt.normalize()
        breakdown_df = breakdown_df.set_index(time_col).sort_index()

        if origin == "raw":
            return breakdown_df

        # Re-anchor every non-intercept component; the offset goes into
        # Intercept so each row's sum (= forecast) stays invariant.
        component_names = [col for col in breakdown_df.columns if col != "Intercept"]
        for col in component_names:
            offset = breakdown_df[col].iloc[0] if origin == "first_value" else breakdown_df[col].mean()
            breakdown_df[col] = breakdown_df[col] - offset
            if "Intercept" in breakdown_df.columns:
                breakdown_df["Intercept"] = breakdown_df["Intercept"] + offset
        return breakdown_df

    def attach_breakdown_columns(fc_df: "pd.DataFrame", breakdown_df: "pd.DataFrame", time_col: str) -> "pd.DataFrame":
        """Stamp canonical breakdown columns onto ``fc_df``.

        ``Trend + Intercept`` → ``longterm_growth`` so the level column
        reflects the actual baseline (without intercept the trend
        dangles around 0).  Remaining mappings come from
        :data:`BREAKDOWN_GROUP_TO_COL`.

        Args:
            fc_df: Forecast frame whose ``time_col`` will be used as the
                lookup key into ``breakdown_df``.
            breakdown_df: Date-indexed breakdown from
                :func:`extract_breakdown_components`.
            time_col: Name of the date column in ``fc_df``.

        Returns:
            ``fc_df`` with new breakdown columns merged in.
        """
        if breakdown_df is None or breakdown_df.empty:
            return fc_df
        if "Trend" in breakdown_df.columns:
            level_series = breakdown_df["Trend"].copy()
            if "Intercept" in breakdown_df.columns:
                level_series = level_series + breakdown_df["Intercept"]
            fc_df[LONGTERM_GROWTH_COL] = pd.to_datetime(fc_df[time_col]).dt.normalize().map(arg=level_series)
        for group_name, target_col in BREAKDOWN_GROUP_TO_COL.items():
            if group_name == "Trend":
                continue
            if group_name in breakdown_df.columns:
                fc_df[target_col] = pd.to_datetime(fc_df[time_col]).dt.normalize().map(arg=breakdown_df[group_name])
        return fc_df

    @dataclass
    class GreykiteForecastAlgo(TSAlgo):
        """Greykite-based multivariate time-series forecasting algorithm.

        Registered as ``"greykite"`` in :data:`ALGO_REGISTRY` when
        ``blah.greykite`` is importable.

        ``algo_params`` keys (all optional):

        - ``model_template`` (str): greykite model template name, e.g.
          ``"SILVERKITE"`` (default) or ``"AUTO"``.
        - ``forecast_horizon`` (int): periods ahead to forecast; overridden
          by ``ForecastConfig.forecast_horizon`` when available.
        - ``model_components`` (dict): kwargs for ``ModelComponentsParam``
          (growth, seasonality, changepoints, regressors, custom, …).
        - ``evaluation_period`` (dict): kwargs for ``EvaluationPeriodParam``
          (cv_max_splits, cv_horizon, test_horizon, …).
        - ``computation`` (dict): kwargs for ``ComputationParam``
          (hyperparameter_budget, n_jobs, verbose).
        - ``breakdown_origin`` (str): re-anchor mode for breakdown
          components — one of :data:`BREAKDOWN_ORIGINS`.  Default
          ``"first_value"`` (every component starts at 0; the offset
          folds into Intercept and so into ``longterm_growth``).

        Output (``predict()``) — in addition to the standard fixed-schema
        columns (``ts``/``time_col``, ``metric_id``, ``actual``,
        ``forecast``, ``forecast_lower``, ``forecast_upper``), the
        forecast frame carries:

        - ``std`` — Gaussian-derived from the upper bound and ``coverage``.
        - ``longterm_growth`` (= Trend + Intercept), ``weekly_seasonality``,
          ``annual_seasonality``, ``holiday_impact`` — pulled from
          greykite's ``forecast_breakdown`` (additive Silverkite only;
          silently skipped for non-additive templates).
        """

        def __post_init__(self) -> None:
            super().__post_init__()
            # Populated during fit(); keyed by (dim_vals_tuple, value_col).
            # dim_vals_tuple is () when no dim_cols are configured.
            self.gk_results: Dict[tuple, Any] = {}
            # Stored so predict() can rename the greykite time column back.
            self.time_col: str = "ts"
            # Dim column names stored at fit() time for use in predict().
            self.dim_cols: tuple = ()
            # Coverage + breakdown_origin captured at fit() for reuse in predict().
            self.coverage: float = 0.80
            self.breakdown_origin: str = "first_value"

        # ------------------------------------------------------------------
        # Public interface
        # ------------------------------------------------------------------

        def fit(
            self,
            df: pd.DataFrame,
            config: TSModelConfig,
            anomaly_df: Optional[pd.DataFrame] = None,
        ) -> "GreykiteForecastAlgo":
            """Fit a greykite model for each value column.

            Args:
                df: Training DataFrame containing ``config.time_col``,
                    all ``config.value_cols``, and any
                    ``config.regressor_cols``. Future rows with regressor
                    values but ``NaN`` targets should be appended here when
                    regressors are used.
                config: Time-series configuration.
                anomaly_df: Optional anomaly intervals with columns
                    ``start_ts`` and ``end_ts``. Anomalous periods are
                    masked to ``NaN`` before fitting (greykite
                    ``anomaly_info`` strategy).

            Returns:
                Self.
            """
            self.gk_results = {}
            self.time_col = config.time_col
            self.dim_cols = config.dim_cols
            self.coverage = float(config.coverage) if config.coverage is not None else 0.80
            shared_params = config.get_algo_params(config.value_cols[0]) if config.value_cols else {}
            origin = shared_params.get("breakdown_origin", "first_value")
            if origin not in BREAKDOWN_ORIGINS:
                raise ValueError(f"breakdown_origin must be one of {BREAKDOWN_ORIGINS}; got {origin!r}.")
            self.breakdown_origin = origin

            # Build (dim_group, df_slice) pairs.
            # When dim_cols is empty there is one group covering the full df.
            if config.dim_cols:
                grouped = df.groupby(list(config.dim_cols))
                group_iter = [(dim_vals if isinstance(dim_vals, tuple) else (dim_vals,), df_slice) for dim_vals, df_slice in grouped]
            else:
                group_iter = [((), df)]

            for dim_vals, df_slice in group_iter:
                for col in config.value_cols:
                    gk_config = self.build_gk_config(config, col, anomaly_df)
                    cols_to_use = [config.time_col, col] + list(config.regressor_cols)
                    df_col = df_slice[cols_to_use].copy()
                    self.gk_results[(dim_vals, col)] = GKForecaster().run_forecast_config(df=df_col, config=gk_config)

            return self

        def predict(
            self,
            df: Optional[pd.DataFrame] = None,
            prediction_window: Optional[tuple[str, str]] = None,
        ) -> ForecastResult:
            """Return forecast results for all fitted value columns.

            Args:
                df: Unused; present for API compatibility with
                    :class:`~abvelocity.ts.algo.base.TSAlgo`.
                prediction_window: Optional ``(start_date, end_date)`` ISO
                    strings to restrict the returned forecast range.

            Returns:
                :class:`~abvelocity.ts.result.forecast_result.ForecastResult`
                with ``result_df`` in long format (one row per timestamp × metric)
                and fixed columns ``ts``, ``metric``, ``actual``, ``forecast``,
                ``forecast_lower``, ``forecast_upper``. ``fit_info`` holds
                backtest evaluation metrics keyed by value column name.
            """
            if not self.gk_results:
                raise ValueError("Must call fit() before predict().")

            per_col_dfs = []
            fit_info: Dict[str, Any] = {}

            # Captured at fit() time — see __post_init__.
            breakdown_origin = self.breakdown_origin
            coverage = self.coverage

            for (dim_vals, col), gk_result in self.gk_results.items():
                fc_df = gk_result.forecast.df.copy()

                # Greykite already uses our fixed column names:
                #   ACTUAL_COL="actual", PREDICTED_COL="forecast",
                #   PREDICTED_LOWER_COL="forecast_lower",
                #   PREDICTED_UPPER_COL="forecast_upper".
                # Just select the columns we need and add the metric label.
                keep_cols = [GK_TIME_COL, ACTUAL_COL, PREDICTED_COL]
                if PREDICTED_LOWER_COL in fc_df.columns:
                    keep_cols.append(PREDICTED_LOWER_COL)
                if PREDICTED_UPPER_COL in fc_df.columns:
                    keep_cols.append(PREDICTED_UPPER_COL)

                fc_df = fc_df[keep_cols].copy()
                fc_df.insert(1, METRIC_ID_COL, col)

                # Inject dim columns after metric (position 2, 3, ...).
                for i, (dim_col, dim_val) in enumerate(zip(self.dim_cols, dim_vals)):
                    fc_df.insert(2 + i, dim_col, dim_val)

                # Stamp std from the (forecast, upper) bound + coverage
                # before any column rename — std is the standard
                # deviation under a Gaussian assumption: σ = (upper − μ) / z.
                if PREDICTED_UPPER_COL in fc_df.columns:
                    z_score = float(norm.ppf(q=(1.0 + coverage) / 2.0))
                    fc_df[STD_COL] = (fc_df[PREDICTED_UPPER_COL] - fc_df[PREDICTED_COL]) / z_score

                # Stamp breakdown columns from greykite's breakdown.
                # Silently skipped for non-additive models.
                breakdown_df = extract_breakdown_components(gk_result=gk_result, origin=breakdown_origin)
                if breakdown_df is not None:
                    fc_df = attach_breakdown_columns(fc_df=fc_df, breakdown_df=breakdown_df, time_col=GK_TIME_COL)

                # Rename time column if caller used a non-default name.
                if self.time_col != GK_TIME_COL:
                    fc_df = fc_df.rename(columns={GK_TIME_COL: self.time_col})

                # Rename greykite predicted columns to our fixed schema names.
                fc_df = fc_df.rename(
                    columns={
                        PREDICTED_COL: FORECAST_COL,
                        PREDICTED_LOWER_COL: FORECAST_LOWER_COL,
                        PREDICTED_UPPER_COL: FORECAST_UPPER_COL,
                    }
                )

                per_col_dfs.append(fc_df)

                # Collect backtest evaluation metrics keyed by (dim_vals, col).
                evaluation = getattr(gk_result.backtest, "test_evaluation", None)
                if evaluation:
                    fit_info_key = (dim_vals, col) if dim_vals else col
                    fit_info[fit_info_key] = dict(evaluation)

            # Stack all per-metric DataFrames (long format).
            result_df = pd.concat(per_col_dfs, ignore_index=True)

            # Filter to prediction_window if provided.
            if prediction_window is not None:
                start, end = prediction_window
                ts = result_df[self.time_col]
                mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
                result_df = result_df.loc[mask].reset_index(drop=True)

            return ForecastResult(
                result_df=result_df,
                fit_info=fit_info if fit_info else None,
            )

        def build_gk_config(
            self,
            config: TSModelConfig,
            value_col: str,
            anomaly_df: Optional[pd.DataFrame] = None,
        ) -> "GKForecastConfig":
            """Build a greykite ForecastConfig for a single value column."""
            params = config.get_algo_params(value_col)

            # anomaly_info: rename our start_ts/end_ts to greykite convention.
            anomaly_info = None
            if anomaly_df is not None:
                anomaly_info = {
                    "value_col": value_col,
                    "anomaly_df": anomaly_df,
                    "start_time_col": "start_ts",
                    "end_time_col": "end_ts",
                    # adjustment_delta_col omitted → greykite masks period to NaN
                }

            metadata = MetadataParam(
                time_col=config.time_col,
                value_col=value_col,
                freq=config.freq,
                train_end_date=config.train_end_date,
                anomaly_info=anomaly_info,
            )

            # Regressors via ModelComponentsParam.
            model_components: Optional[ModelComponentsParam] = None
            if config.regressor_cols:
                mc_kwargs = dict(params.get("model_components", {}))
                mc_kwargs.setdefault("regressors", {"regressor_cols": list(config.regressor_cols)})
                model_components = ModelComponentsParam(**mc_kwargs)
            elif "model_components" in params:
                model_components = ModelComponentsParam(**params["model_components"])

            # forecast_horizon: ForecastConfig > algo_params > default 1.
            forecast_horizon = 1
            if isinstance(config, ForecastConfig):
                forecast_horizon = config.forecast_horizon
            elif "forecast_horizon" in params:
                forecast_horizon = int(params["forecast_horizon"])

            evaluation_period = None
            if "evaluation_period" in params:
                evaluation_period = EvaluationPeriodParam(**params["evaluation_period"])

            computation = None
            if "computation" in params:
                computation = ComputationParam(**params["computation"])

            return GKForecastConfig(
                metadata_param=metadata,
                model_template=params.get("model_template", "SILVERKITE"),
                forecast_horizon=forecast_horizon,
                coverage=config.coverage,
                model_components_param=model_components,
                evaluation_period_param=evaluation_period,
                computation_param=computation,
            )

    ALGO_REGISTRY["greykite"] = GreykiteForecastAlgo

    from abvelocity.ts.model_selection.param_converter import ParamConverter

    @dataclass
    class GreykiteParamConverter(ParamConverter):
        """Translate flat search-space keys into greykite's nested ``algo_params`` shape.

        :class:`GreykiteForecastAlgo` accepts a deeply nested ``algo_params``
        dict (``model_template`` / ``model_components.<group>.<key>`` /
        ``breakdown_origin`` / etc.). Putting those nested dicts directly in
        a :class:`SearchSpace` works mechanically but produces unreadable
        labels and CSV cells, and the shallow merge against the template
        replaces whole sub-dicts.

        This converter accepts a small flat vocabulary that's pleasant to
        sweep over and emits the corresponding nested override. Supported
        keys (all optional — only the ones present in ``params`` are
        emitted):

        - ``model_template`` (str) — passes through verbatim.
        - ``fit_algorithm`` (str, e.g. ``"ridge"`` / ``"linear"``) →
          ``model_components.custom.fit_algorithm_dict.fit_algorithm``.
        - ``yearly_seasonality``, ``weekly_seasonality``, ``daily_seasonality``,
          ``quarterly_seasonality``, ``monthly_seasonality`` (int/str) →
          ``model_components.seasonality.<key>``.
        - ``changepoint_reg`` (float) →
          ``model_components.changepoints.changepoints_dict.regularization_strength``.
          Renamed from greykite's verbose internal name to make the
          search-space term unambiguous (it's *changepoint* regularization,
          not generic regularization).
        - ``regression_weight_col`` (str | None) →
          ``model_components.custom.regression_weight_col``. The
          column name (e.g. ``"ct1"``, ``"ct2"``) used to weight
          regression observations; the column must be present on the
          input DataFrame.
        - ``breakdown_origin`` (str) — passes through verbatim.

        Unknown keys are forwarded to the top-level override unchanged so
        callers can mix in raw nested overrides when the flat vocabulary
        doesn't cover their use case.
        """

        SEASONALITY_KEYS = (
            "yearly_seasonality",
            "weekly_seasonality",
            "daily_seasonality",
            "quarterly_seasonality",
            "monthly_seasonality",
        )
        RECOGNISED_KEYS = frozenset({
            "model_template",
            "breakdown_origin",
            "fit_algorithm",
            "regression_weight_col",
            "changepoint_reg",
            *SEASONALITY_KEYS,
        })

        def convert(self, params: Dict[str, Any]) -> Dict[str, Any]:
            """Return the nested ``algo_params`` override dict.

            Args:
                params: Flat search-space params.

            Returns:
                Override dict ready to merge into
                :attr:`ForecastConfig.algo_params`.
            """
            out: Dict[str, Any] = {}
            model_components: Dict[str, Any] = {}

            if "model_template" in params:
                out["model_template"] = params["model_template"]

            if "breakdown_origin" in params:
                out["breakdown_origin"] = params["breakdown_origin"]

            custom: Dict[str, Any] = {}
            if "fit_algorithm" in params:
                custom["fit_algorithm_dict"] = {"fit_algorithm": params["fit_algorithm"]}
            if "regression_weight_col" in params:
                custom["regression_weight_col"] = params["regression_weight_col"]
            if custom:
                model_components["custom"] = custom

            seasonality = {k: params[k] for k in self.SEASONALITY_KEYS if k in params}
            if seasonality:
                model_components["seasonality"] = seasonality

            if "changepoint_reg" in params:
                # ``method="auto"`` is required alongside ``regularization_strength`` —
                # the strength is meaningless without telling greykite which CP detection
                # method to apply. ``"auto"`` lets the framework pick (LASSO-based for
                # SILVERKITE).
                model_components["changepoints"] = {
                    "changepoints_dict": {
                        "method": "auto",
                        "regularization_strength": params["changepoint_reg"],
                    }
                }

            if model_components:
                out["model_components"] = model_components

            for key, value in params.items():
                if key not in self.RECOGNISED_KEYS:
                    out[key] = value

            return out

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
"""Base time-series configuration dataclass."""

import string
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from abvelocity.ts.constants import TIME_COL
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class TSModelConfig(DataClassJSONMixin):
    """Base configuration shared by forecasting and anomaly detection.

    Attributes:
        time_col: Name of the timestamp column in the input DataFrame.
        value_cols: Target metric column names (multivariate by default).
            Must be non-empty. Using a tuple enforces immutability and
            hashability.
        regressor_cols: Covariate/feature column names used as regressors.
            Mirrors ``extraMetricIds`` in oi-schemas ``AlgoConfig``.
        freq: Pandas offset alias for the time-series frequency, e.g.
            ``"D"`` for daily or ``"H"`` for hourly.
        train_end_date: ISO-format date string marking the end of the
            training window.
        coverage: Prediction interval coverage probability; must be in
            the open interval (0, 1).
        algo_name: Key into
            :data:`~abvelocity.ts.algo.base.ALGO_REGISTRY`
            selecting which algorithm to use.
        algo_params: Algorithm-specific pass-through parameters; must be
            JSON-serializable. Mirrors ``customAlgoConfig`` in oi-schemas.
    """

    time_col: str = TIME_COL
    """Name of the timestamp column."""

    value_cols: Optional[Tuple[str, ...]] = None
    """Target metric column names.

    When ``None`` (default), the value is derived from ``metric_info.metrics``
    at run time by :class:`~abvelocity.ts.flow.flow.TSFlow`.
    Must be explicitly set (non-empty) when using :class:`TSRunner` directly
    outside of a flow.
    """

    dim_cols: tuple[str, ...] = ()
    """Dimension column names to split on before fitting.

    When non-empty the algorithm trains one model per
    ``(dim_combination × value_col)`` and the result DataFrame contains
    one column per dim alongside the standard fixed-schema columns.
    """

    regressor_cols: tuple[str, ...] = ()
    """Covariate/feature column names (mirrors extraMetricIds in oi-schemas)."""

    freq: Optional[str] = None
    """Pandas offset alias, e.g. ``"D"`` for daily, ``"H"`` for hourly."""

    train_start_date: Optional[str] = None
    """ISO-format start date for the training window.  ``None`` means use all
    available data from the beginning of the input DataFrame."""

    train_end_date: Optional[str] = None
    """ISO-format end date for the training window."""

    coverage: float = 0.95
    """Prediction interval coverage probability; must be in (0, 1)."""

    algo_name: str = ""
    """Algorithm name; looked up in ALGO_REGISTRY at run time."""

    algo_version: Optional[str] = None
    """Version string for the specific algorithm/config combination.

    Stamped onto ``result_df[ALGO_VERSION_COL]``. Bumping this lets two
    configs running the same ``algo_name`` coexist for the same
    ``train_end_date`` in a downstream output table. ``None`` leaves
    the column populated with ``NaN``."""

    algo_params: Optional[Dict[str, Any]] = None
    """Algorithm-specific pass-through parameters (mirrors customAlgoConfig)."""

    algo_params_by_metric: Optional[Dict[str, Dict[str, Any]]] = None
    """Per-metric algo_params overrides, keyed by value_col name.
    When set, :meth:`get_algo_params` returns the override for that metric
    and falls back to ``algo_params`` for metrics not listed here."""

    metric_id_template: Optional[str] = None
    """Machine-key template for rows of ``result_df[METRIC_ID_COL]``.

    Accepts either a plain string (stamped as-is on every row) or a
    format-string template with ``{value_col}`` + ``{<dim_name>}``
    placeholders. Per-row rendering happens in
    :class:`~abvelocity.ts.runner.TSRunner`.

    ``None`` leaves the ``metric_id`` column at whatever the algorithm
    stamped (by default, the value-column name).

    Examples::

        metric_id_template = "randomProduct_signups_daily"                       # scalar
        metric_id_template = "randomProduct_daily:m={value_col}"                 # per value_col
        metric_id_template = "randomProduct:country={country}"                   # per dim
        metric_id_template = "randomProduct:m={value_col}|country={country}"     # both
    """

    metric_name_template: Optional[str] = None
    """Human-label template for ``result_df[METRIC_NAME_COL]``.

    Same placeholder rules as ``metric_id_template``. ``None`` leaves
    the column as ``NaN``."""

    def __post_init__(self) -> None:
        if not (0 < self.coverage < 1):
            raise ValueError(f"coverage must be in the open interval (0, 1), got {self.coverage!r}.")
        if self.value_cols is not None and not self.value_cols:
            raise ValueError("value_cols must be non-empty. Pass None to derive from metric_info in TSFlow.")
        # Validate that any format-string placeholders in the metric
        # templates refer to {value_col} or a declared dim.
        allowed = {"value_col", *self.dim_cols}
        _validate_template_placeholders(self.metric_id_template, "metric_id_template", allowed)
        _validate_template_placeholders(self.metric_name_template, "metric_name_template", allowed)

    def get_algo_params(self, metric: str) -> Dict[str, Any]:
        """Return resolved algo params for a given metric.

        Starts from ``algo_params`` (the common base) and merges any
        per-metric override from ``algo_params_by_metric`` on top.
        Only the keys present in the override are updated — unspecified
        keys keep their value from the common base.

        Args:
            metric: Value column name to resolve params for.

        Returns:
            Dict of algo params (never ``None``).
        """
        params = dict(self.algo_params or {})
        if self.algo_params_by_metric:
            override = self.algo_params_by_metric.get(metric)
            if override:
                params.update(override)
        return params


def _validate_template_placeholders(template: Optional[str], field_name: str, allowed: set) -> None:
    """Raise ``ValueError`` if ``template`` references placeholder names
    not in ``allowed``. Plain strings (no ``{...}`` fields) always pass.
    """
    if template is None:
        return
    used = set()
    for _literal, field_ref, _spec, _conv in string.Formatter().parse(template):
        if field_ref is None:
            continue
        # Strip nested attribute/index access like {country[0]} or {x.y} —
        # we only need the top-level name for validation.
        name = field_ref.split(".", 1)[0].split("[", 1)[0]
        if name:
            used.add(name)
    unknown = used - allowed
    if unknown:
        raise ValueError(f"{field_name} references unknown placeholders: {sorted(unknown)}. " f"Allowed: {sorted(allowed)}.")

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
"""Adapter from flat search-space params to algo-specific ``algo_params`` overrides.

Some algos accept a flat ``algo_params`` dict (e.g.
:class:`~abvelocity.ts.algo.simple_forecast_algo.SimpleForecastAlgo`
takes ``period`` / ``k`` / ``agg`` directly). Others — notably
:class:`~abvelocity.ts.algo.greykite_forecast_algo.GreykiteForecastAlgo`
— take a deeply nested structure (``model_components.seasonality.yearly_seasonality``).

Stuffing nested dicts into a :class:`SearchSpace` works mechanically but
makes the audit trail ugly: ``model_candidates.csv`` ends up with cells
like ``{'custom': {'fit_algorithm_dict': {'fit_algorithm': 'ridge'}}}``
and labels become unreadable. Worse, candidate-level overrides do a
*shallow* update against the template's ``algo_params`` — so providing
``model_components`` in a candidate REPLACES the entire template
``model_components``, which is rarely what the user wants.

A :class:`ParamConverter` solves both problems: the user defines a
search space in the flat form they want to reason about
(``{"fit_algorithm": ["ridge", "linear"]}``); the converter — supplied
by the algo or the use-case — turns that into the nested override the
algo expects, with deep-merge semantics if needed. The candidates log,
report, and candidate id all stay in the flat form.

Two implementations live here:

* :class:`ParamConverter` — abstract base; subclasses implement
  :meth:`convert`.
* :class:`IdentityParamConverter` — no-op; the default when an algo's
  ``algo_params`` keys already match the search-space keys.

Algo-specific converters (e.g. ``GreykiteParamConverter``) live
alongside their algo module so the converter ships with the algo.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ParamConverter:
    """Abstract base — translate a flat search-space params dict into algo_params overrides.

    Subclasses must implement :meth:`convert`. The class is also callable
    (``converter(params)``) so it can be used wherever a
    ``Callable[[dict], dict]`` is expected.

    Uses ``@dataclass`` + ``@abstractmethod`` (no ``ABC`` base), matching
    :class:`~abvelocity.ts.algo.base.TSAlgo` and the
    :class:`~abvelocity.stats.estimator.Estimator` pattern in this
    repo.
    """

    @abstractmethod
    def convert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``algo_params`` overrides for this candidate.

        The returned dict is shallow-merged into the template's
        ``algo_params`` by
        :meth:`~abvelocity.ts.model_selection.base.ModelSelection.merge_params`.
        If the algo's config has nested structure, the converter is
        responsible for producing the full nested shape (and, if
        desired, deep-merging with whatever it knows about the
        template).

        Args:
            params: One candidate's search-space params (flat form).

        Returns:
            Override dict in the algo's expected ``algo_params`` shape.
        """
        ...

    def __call__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for :meth:`convert`; makes the converter directly callable."""
        return self.convert(params)


@dataclass
class IdentityParamConverter(ParamConverter):
    """Pass-through converter; returns ``params`` unchanged.

    Use when the algo's ``algo_params`` keys already match the
    search-space keys exactly (the typical case for
    :class:`~abvelocity.ts.algo.simple_forecast_algo.SimpleForecastAlgo`).
    Equivalent to passing no converter at all to
    :class:`ModelSelection`; included so callers can be explicit.
    """

    def convert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` verbatim.

        Args:
            params: Search-space params dict.

        Returns:
            The same dict (a shallow copy, to avoid accidental mutation
            of the candidate's stored params downstream).
        """
        return dict(params)

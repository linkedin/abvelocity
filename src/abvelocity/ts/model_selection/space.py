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
"""Search-space primitives for model selection.

A :class:`SearchSpace` is an ordered list of :class:`ParamGroup` s. Each group
carries a mini-grid of one or more parameters (each with a list of candidate
values). A candidate parameter dict is one combination drawn from the
cartesian product of a group's mini-grid.

The values placed in a group are *override* values — they patch the
``forecast_config.algo_params`` of the
:class:`~abvelocity.ts.backfill.config.BackfillConfig` template held
by the surrounding :class:`~abvelocity.ts.model_selection.base.ModelSelection`
instance. Every other field of the template (``forecast_horizon``, ``step``,
``time_col``, etc.) stays fixed for the entire selection run.

Method-specific semantics
-------------------------
* :class:`~abvelocity.ts.model_selection.grid.GridModelSelection`
  flattens the entire space and evaluates every cartesian combination.
* :class:`~abvelocity.ts.model_selection.grouped.GroupedModelSelection`
  walks the groups in order, evaluating each stage's mini-grid with prior
  stages' winners frozen — the layered grouped-stepwise method described in
  Hosseini, Newlands, Dean, Takemura (2015) §3.4. Two cross-stage helpers:

  * ``augment=True``: the group introduces parameters absent from earlier
    groups (e.g. paper's "+ ground covariates" / "+ spatial correlation"
    layers).
  * ``reopen=[...]``: list of earlier-group parameter names to re-sweep
    alongside this group's own params (catches interactions when the
    augmenting layer changes the optimum of a prior choice).
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class ParamGroup:
    """One stage of a grouped-stepwise sweep.

    Attributes:
        name: Human-readable group name; used in logs, model_candidates.csv, and report.
            Must be unique within a :class:`SearchSpace`.
        params: ``{param_name: [value, ...]}`` mini-grid for this group. The
            cartesian product across these params is evaluated within the
            stage. Each ``param_name`` must correspond to a key in the
            target template's ``forecast_config.algo_params``.
        augment: If ``True``, declares that this group contributes
            parameters not present in any earlier group. Mirrors the
            layered augmentation pattern from Hosseini, Newlands, Dean,
            Takemura (2015) §3.4 ("+ ground covariates" / "+ spatial
            correlation" layers).
        reopen: Optional list of parameter names from earlier groups to
            re-sweep alongside this group's mini-grid. The candidate values
            for a reopened parameter come from the originating group's
            ``params``. ``None`` (default) keeps earlier winners frozen.
    """

    name: str
    """Human-readable group name; unique within the SearchSpace."""

    params: Dict[str, List[Any]]
    """``{param_name: [value, ...]}``; mini-grid for this stage."""

    augment: bool = False
    """If True, this group introduces parameters absent from earlier groups."""

    reopen: Optional[List[str]] = None
    """Earlier-group param names to re-sweep alongside this group's params."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"ParamGroup.name must be a non-empty string, got {self.name!r}.")
        if not isinstance(self.params, dict) or not self.params:
            raise ValueError(f"ParamGroup({self.name!r}).params must be a non-empty dict.")
        for key, values in self.params.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(
                    f"ParamGroup({self.name!r}).params[{key!r}] must be a non-empty list, got {values!r}."
                )

    def candidates(self) -> List[Dict[str, Any]]:
        """Return all parameter dicts in this group's mini-grid.

        Returns:
            List of dicts; one per cartesian-product combination of the
            group's parameters. Order is determined by the iteration order
            of :attr:`params` (insertion order).
        """
        keys = list(self.params.keys())
        value_lists = [self.params[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of this group.

        Includes ``name``, ``params``, ``augment``, ``reopen``, and the
        ``size`` of this group's mini-grid (cartesian product of its
        params). Useful for the report and for persisting run config.
        """
        return {
            "name": self.name,
            "params": {k: list(v) for k, v in self.params.items()},
            "augment": self.augment,
            "reopen": list(self.reopen) if self.reopen else None,
            "size": int(math.prod(len(v) for v in self.params.values())),
        }

    def __str__(self) -> str:
        """Multi-line human-readable rendering for logs / REPL.

        Format::

            ParamGroup(name='regression', size=2)
              fit_algorithm: ['ridge', 'linear']

        ``augment`` and ``reopen`` are appended only when set.
        """
        head = f"ParamGroup(name={self.name!r}, size={int(math.prod(len(v) for v in self.params.values()))})"
        body = "\n".join(f"  {k}: {v!r}" for k, v in self.params.items())
        flags: List[str] = []
        if self.augment:
            flags.append("  augment=True")
        if self.reopen:
            flags.append(f"  reopen={list(self.reopen)!r}")
        if flags:
            body = body + "\n" + "\n".join(flags)
        return f"{head}\n{body}"


@dataclass
class SearchSpace:
    """Ordered collection of :class:`ParamGroup` s.

    The order of :attr:`groups` only matters for grouped-stepwise selection;
    grid selection flattens the space.

    Attributes:
        groups: Stages in evaluation order. Group names must be unique.
    """

    groups: List[ParamGroup] = field(default_factory=list)
    """Stages in evaluation order. Group names must be unique."""

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("SearchSpace requires at least one ParamGroup.")
        names = [g.name for g in self.groups]
        if len(names) != len(set(names)):
            raise ValueError(f"ParamGroup names must be unique within a SearchSpace; got {names}.")

    @classmethod
    def flat(cls, params: Dict[str, List[Any]], name: str = "all") -> "SearchSpace":
        """Build a single-group space — equivalent to a flat cartesian grid.

        Convenience constructor for grid selection when grouping isn't
        needed.

        Args:
            params: ``{param_name: [value, ...]}`` mini-grid.
            name: Group name. Default ``"all"``.

        Returns:
            :class:`SearchSpace` with exactly one :class:`ParamGroup`.
        """
        return cls(groups=[ParamGroup(name=name, params=params)])

    def all_param_names(self) -> List[str]:
        """Return every parameter name across all groups in declaration order.

        Returns:
            Ordered list of unique parameter names. Group iteration order
            then within-group insertion order.
        """
        names: List[str] = []
        seen = set()
        for group in self.groups:
            for key in group.params:
                if key not in seen:
                    seen.add(key)
                    names.append(key)
        return names

    def cartesian_candidates(self) -> List[Dict[str, Any]]:
        """Return the flat cartesian product of all groups' parameters.

        Used by :class:`~abvelocity.ts.model_selection.grid.GridModelSelection`.
        If a parameter appears in multiple groups, all groups must declare
        the same list of values (otherwise the cartesian product is
        ambiguous); a :class:`ValueError` is raised when they conflict.

        Returns:
            List of parameter dicts. Each dict contains every parameter
            from :meth:`all_param_names`. Total length equals
            ``∏ |params[k]|`` over the unique parameter names.

        Raises:
            ValueError: If a parameter appears in multiple groups with
                different value lists.
        """
        merged: Dict[str, List[Any]] = {}
        for group in self.groups:
            for key, values in group.params.items():
                if key in merged and merged[key] != values:
                    raise ValueError(
                        f"Parameter {key!r} appears in multiple groups with conflicting "
                        f"values: {merged[key]} vs {values}. Either dedupe the values or "
                        f"use GroupedModelSelection where multi-group params have explicit semantics."
                    )
                merged[key] = values
        keys = list(merged.keys())
        value_lists = [merged[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    def stage_candidates(
        self,
        stage_index: int,
        frozen: Dict[str, Any],
    ) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Yield ``(stage_only_params, full_params)`` pairs for one grouped stage.

        Each pair represents one candidate at this stage:

        * ``stage_only_params`` — the values being chosen *at* this stage
          (this group's mini-grid plus any ``reopen`` params from earlier
          groups). This is the dict whose values get logged as "what's
          being decided at this stage."
        * ``full_params`` — ``frozen`` updated with ``stage_only_params``.
          This is the dict that goes to the predictor.

        Args:
            stage_index: 0-based index into :attr:`groups`.
            frozen: Winners from prior stages, ``{param_name: value}``.

        Yields:
            ``(stage_only_params, full_params)`` for every cartesian
            combination at this stage.

        Raises:
            ValueError: If the group's ``reopen`` list references a
                parameter not declared by any earlier group.
        """
        group = self.groups[stage_index]
        stage_grid: Dict[str, List[Any]] = dict(group.params)

        if group.reopen:
            earlier_lookup: Dict[str, List[Any]] = {}
            for prior in self.groups[:stage_index]:
                for key, values in prior.params.items():
                    earlier_lookup[key] = values
            for reopen_key in group.reopen:
                if reopen_key not in earlier_lookup:
                    raise ValueError(
                        f"ParamGroup({group.name!r}).reopen lists {reopen_key!r}, "
                        f"but no earlier group declared that parameter."
                    )
                stage_grid[reopen_key] = earlier_lookup[reopen_key]

        keys = list(stage_grid.keys())
        value_lists = [stage_grid[k] for k in keys]
        for combo in itertools.product(*value_lists):
            stage_only = dict(zip(keys, combo))
            full = {**frozen, **stage_only}
            yield stage_only, full

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable description of the search space.

        Includes the ordered list of group dicts (see
        :meth:`ParamGroup.to_dict`) plus the total cartesian-product
        size (the same number :meth:`cartesian_candidates` would emit).
        """
        return {
            "groups": [g.to_dict() for g in self.groups],
            "cartesian_size": len(self.cartesian_candidates()) if len(self.groups) >= 1 else 0,
        }

    def __str__(self) -> str:
        """Multi-line human-readable rendering for logs / REPL.

        Format::

            SearchSpace(groups=2, cartesian=4)
              [stage 1] ParamGroup(name='regression', size=2)
                fit_algorithm: ['ridge', 'linear']
              [stage 2] ParamGroup(name='changepoint', size=2)
                changepoint_reg: [0.01, 1.0]

        For grouped sweeps the ``[stage N]`` prefix calls out the
        evaluation order; for a flat grid (single group) it still shows
        the bracketed marker for consistency.
        """
        try:
            cartesian = len(self.cartesian_candidates())
        except ValueError:
            # Conflicting param values across groups — cartesian product is
            # ambiguous (Grouped is fine; only Grid would raise). Render the
            # space without a total so the rest of the description still
            # surfaces.
            cartesian = -1
        head = (
            f"SearchSpace(groups={len(self.groups)}, "
            f"cartesian={cartesian if cartesian >= 0 else 'ambiguous'})"
        )
        body_lines: List[str] = []
        for idx, group in enumerate(self.groups, 1):
            stage_str = str(group)
            # Indent each line of the group's __str__ + prefix the head.
            lines = stage_str.splitlines()
            body_lines.append(f"  [stage {idx}] {lines[0]}")
            body_lines.extend(f"  {line}" for line in lines[1:])
        return head + "\n" + "\n".join(body_lines)

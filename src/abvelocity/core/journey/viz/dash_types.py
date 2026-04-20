"""Shared type definitions for configurable Dash components.

This module holds pure data-class and type-alias definitions that are
imported by both ``dash_components`` and ``create_configurable_dash``,
breaking the circular import that would otherwise exist between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional


@dataclass
class ParameterSpec:
    """Declarative description of a single user input control.

    Fields map directly to Dash component construction. Only static option
    lists are supported (derive dynamic values before creating the spec).

    Attributes:
        name: Parameter name; becomes the keyword passed into ``plot_func``.
        label: Human readable label shown above the control.
        component: Control type (``'dropdown'``, ``'text'``, ``'number'``, ``'date'``).
        options: Static list of dropdown options (ignored for non-dropdown components).
        default: Initial value.
        required: If False and value empty/None, parameter is omitted from the call.
        multi: Whether dropdown allows multi-select.
        help: Helper text rendered below the control.
        transform: Optional callable applied to the raw Dash value before dispatch.
    """

    name: str
    label: str
    component: str  # 'dropdown' | 'text' | 'number'
    options: Optional[List[Any]] = None  # now only static lists
    default: Any = None
    required: bool = False
    multi: bool = False
    help: str = ""
    transform: Optional[Callable[[Any], Any]] = None  # applied before passing to plot func

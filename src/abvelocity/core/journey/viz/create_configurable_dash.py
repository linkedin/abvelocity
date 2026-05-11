"""Generic configurable Dash components (library).

This module intentionally excludes any product / sequence specific logic.
Use it by importing the generic classes & helper functions and supplying
your own plot functions and ParameterSpec lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from abvelocity.core.journey.viz.dash_components import (
    COMPONENT_FACTORY,
    build_app_header,
    build_button,
    build_cell_header,
    build_controls_layout,
    build_global_filter_control,
    build_labeled_control,
)
from abvelocity.core.journey.viz.dash_styles import STYLES
from abvelocity.core.journey.viz.dash_types import ParameterSpec

# Re-export so existing ``from ...create_configurable_dash import ParameterSpec`` keeps working.
__all__ = ["ParameterSpec"]

# Base URL for the dashboard app_link. Factored out so it can be overridden
# (monkey-patched or swapped by the OSS transform) without touching the
# URL-building site in __init__.
DASHBOARD_BASE_URL = ""

# -----------------------------
# Parameter + Cell Definitions (Static / no dynamic context)
# -----------------------------

ComponentValue = Any
# Plot / HTML result types supported by plot_func (restricted / simplified).
# Collections (lists / dicts) are no longer supported – the plot_func must
# return exactly one of these primitives and declare which via DashCellConfig.return_type.
PlotResult = Union[go.Figure, html.Div, str]


class PlotReturnType(Enum):
    """Enumeration of supported plot function return types.

    FIGURE: plot_func returns a single ``plotly.graph_objects.Figure``.
    HTML_DIV: plot_func returns a Dash ``html.Div`` component.
    RAW_HTML: plot_func returns a raw HTML / markdown string (rendered with Markdown, HTML allowed).
    """

    FIGURE = "figure"
    HTML_DIV = "html_div"
    RAW_HTML = "raw_html"


@dataclass
class DashCellConfig:
    """Configuration bundle for a dashboard cell (form + output region).

    ``plot_func`` is invoked with keyword arguments whose names correspond to
    each ``ParameterSpec.name``.

    Attributes:
        id: Unique identifier used to derive element IDs.
        title: Cell title displayed in the UI.
        parameter_specs: Ordered list of parameter specifications.
        plot_func: Callable returning a Plotly Figure or collection of figures.
        trigger_label: Text on the run button.
        preserve_history: Placeholder flag for future multi-run history.
        show_loading: Wrap output in a loading spinner if True.
        description: Optional descriptive text below the title.
        button_position: Controls button placement relative to parameter controls.
            - "auto": Left-aligned if no controls, below controls if controls exist (default).
            - "left": Always left-aligned on same row as controls.
            - "right": Always right-aligned on same row as controls.
            - "below": Always on separate row below controls, left-aligned.
    """

    # id must be unique within the app for everything to work correctly
    id: str
    title: str
    parameter_specs: List[ParameterSpec]
    plot_func: Callable[..., PlotResult]
    trigger_label: str = "Generate"
    preserve_history: bool = False
    show_loading: bool = True
    description: str = ""
    # Expected return type of plot_func (used for normalization / validation).
    return_type: PlotReturnType = PlotReturnType.FIGURE
    # Button position: "auto" | "left" | "right" | "below"
    button_position: str = "auto"


@dataclass
class GlobalFiltersConfig:
    """Configuration for global filters that apply to all dashboard cells.

    Attributes:
        primary_specs: Parameters shown by default (e.g., date, activity window, deduping).
        advanced_specs: Parameters hidden in collapsible "Advanced Filters" section.
    """

    primary_specs: List[ParameterSpec]
    advanced_specs: Optional[List[ParameterSpec]] = None

    @property
    def all_specs(self) -> List[ParameterSpec]:
        """Return all parameter specs (primary + advanced)."""
        return self.primary_specs + (self.advanced_specs or [])


class DashCell:
    """Runtime wrapper around :class:`DashCellConfig`.

    Responsibilities:
        * Build the UI layout (controls + output container).
        * Register the Dash callback that wires the run button to ``plot_func``.
    """

    def __init__(self, config: DashCellConfig):
        """Initialize the runtime cell.

        Args:
            config: Immutable configuration describing parameters and plot function.
        """
        self.config = config

    # ---------- ID helpers ----------
    def _param_id(self, param_name: str) -> str:
        """Create a stable component id for a parameter control."""
        return f"{self.config.id}__param__{param_name}"

    @property
    def run_id(self) -> str:  # pragma: no cover - simple accessor
        """ID of the run (trigger) button."""
        return f"{self.config.id}__run"

    @property
    def output_id(self) -> str:  # pragma: no cover - simple accessor
        """ID of the output container (div)."""
        return f"{self.config.id}__output"

    def history_id(self) -> str:  # pragma: no cover - simple accessor
        """ID for the (future) history container."""
        return f"{self.config.id}__history"

    # ---------- Layout builders ----------
    def build_controls(self) -> html.Div:
        """Render the strip of input controls and action button.

        Returns:
            html.Div: Horizontal container of parameter controls and run button.
        """
        controls: List[html.Div] = []
        for spec in self.config.parameter_specs:
            comp = COMPONENT_FACTORY.create_component(spec, self._param_id(spec.name))
            controls.append(build_labeled_control(spec, comp))

        btn = build_button(self.run_id)

        # Determine effective button position based on config
        pos = self.config.button_position
        if pos == "auto":
            effective_pos = "left" if not controls else "below"
        else:
            effective_pos = pos

        return build_controls_layout(controls, btn, effective_pos)

    def build_layout(self) -> html.Div:
        """Compose the full cell layout.

        Returns:
            html.Div: Container with title, optional description, controls, and output.
        """
        # Header with accent bar, icon, and optional description
        cell_children = build_cell_header(self.config.title, self.config.description)

        # Controls section
        cell_children.append(html.Div(self.build_controls(), style={"marginBottom": "20px"}))

        # Separator line before output
        cell_children.append(html.Div(style=STYLES.separator))

        # Output area
        if self.config.show_loading:
            output_div = dcc.Loading(
                id=f"{self.output_id}__loading",
                children=[
                    html.Div(
                        id=self.output_id,
                        style=STYLES.output_container,
                    )
                ],
                type="circle",
                color=STYLES.colors.primary,
            )
        else:
            output_div = html.Div(
                id=self.output_id,
                style=STYLES.output_container,
            )
        cell_children.append(output_div)

        if self.config.preserve_history:
            cell_children.append(html.Div(id=self.history_id(), style={"marginTop": "20px"}))

        return html.Div(cell_children, style=STYLES.cell_card)

    def register_callbacks(self, app: Dash):
        """Attach the cell's callback to the Dash app.

        Args:
            app: Dash application instance.
        """
        state_specs = list(self.config.parameter_specs)
        input_component = Input(self.run_id, "n_clicks")
        states = [State(self._param_id(spec.name), "value") for spec in state_specs]

        @app.callback(
            Output(self.output_id, "children"),
            [input_component],
            states,
            prevent_initial_call=True,
        )
        def _run_cell(n_clicks, *values, _state_specs=state_specs, _cell_config=self.config):  # type: ignore
            try:
                params = _process_params(_state_specs, values)
                result = _cell_config.plot_func(**params)
                return wrap_plot_result(
                    result=result,
                    return_type=_cell_config.return_type,
                )
            except Exception as e:  # pragma: no cover - defensive
                print(e)
                return error_block(str(e))


# -----------------------------
# Utility functions
# -----------------------------


def _process_params(
    specs: List["ParameterSpec"],
    values: tuple,
) -> Dict[str, Any]:
    """Process parameter specs against raw Dash values.

    Applies transforms, skips empty non-required values, and returns a
    keyword-argument dict ready for ``plot_func``.

    Args:
        specs: Ordered parameter specifications.
        values: Corresponding raw values from Dash State components.

    Returns:
        Dict mapping parameter names to processed values.
    """
    params: Dict[str, Any] = {}
    for spec, raw_val in zip(specs, values):
        val = raw_val
        if spec.transform:
            try:
                val = spec.transform(raw_val)
            except Exception:
                pass
        if (val is None or val == "") and not spec.required:
            continue
        params[spec.name] = val
    return params


def dash_placeholder(msg: str) -> html.Div:
    """Return a muted placeholder message.

    Args:
        msg: Message to display.

    Returns:
        html.Div: Styled placeholder.
    """
    return html.Div(
        [
            html.Span("💡", style={"marginRight": "8px"}),
            html.Span(msg),
        ],
        style=STYLES.placeholder,
    )


def error_block(msg: str) -> html.Div:
    """Render a styled error panel.

    Args:
        msg: Error text.

    Returns:
        html.Div: Error block suitable for direct layout insertion.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Span("⚠️", style={"marginRight": "8px", "fontSize": "16px"}),
                    html.Span("Error", style={"fontWeight": "700"}),
                ],
                style=STYLES.error_header,
            ),
            html.Pre(msg, style=STYLES.error_pre),
        ],
        style=STYLES.error_container,
    )


def wrap_plot_result(
    result: PlotResult,
    title: Optional[str] = None,
    return_type: PlotReturnType = PlotReturnType.FIGURE,
) -> html.Div:
    """Normalize a single supported return primitive based on declared type.

    Args:
        result: The value returned by ``plot_func``.
        title: Optional section title. If ``None`` or blank, no header (title + timestamp) is rendered.
        return_type: Declared expected type (validation + rendering choice).

    Returns:
        html.Div: Wrapped output ready for insertion into layout.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def header(sub: Optional[str] = None) -> List[Any]:
        if not title:
            return []
        return [
            html.Div(
                [
                    html.H4(
                        f"{title}{' - ' + sub if sub else ''}",
                        style=STYLES.result_title,
                    ),
                    html.Span(
                        f"Generated: {ts}",
                        style=STYLES.result_timestamp,
                    ),
                ],
                style={"marginBottom": "12px"},
            )
        ]

    def block(children: List[Any]) -> html.Div:
        return html.Div(children, style=STYLES.result_block)

    # Validate & render according to declared type.
    if return_type == PlotReturnType.FIGURE:
        if not isinstance(result, go.Figure):
            return error_block(f"Return type mismatch: expected Plotly Figure, got {type(result).__name__}")
        return block(
            header()
            + [
                dcc.Graph(
                    figure=result,
                    style={"marginTop": "4px"},
                    config={"displayModeBar": True, "displaylogo": False},
                )
            ]
        )

    if return_type == PlotReturnType.HTML_DIV:
        if not isinstance(result, html.Div):
            return error_block(f"Return type mismatch: expected html.Div, got {type(result).__name__}")
        return block(header() + [result])

    if return_type == PlotReturnType.RAW_HTML:
        if not isinstance(result, str):
            return error_block(f"Return type mismatch: expected raw HTML string, got {type(result).__name__}")
        return block(
            header()
            + [
                dcc.Markdown(
                    result,
                    dangerously_allow_html=True,
                    style={"marginTop": "8px"},
                )
            ]
        )

    return error_block("Unrecognized PlotReturnType configuration")


# -----------------------------
# Default Plot Adapters
# -----------------------------


def sankey_plot_adapter_placeholder(*_args, **_kwargs):  # pragma: no cover
    """Placeholder illustrating expected adapter signature.

    Real adapters should live in domain-specific modules.

    Args:
        *_args: Positional arguments (ignored).
        **_kwargs: Keyword arguments (ignored).

    Returns:
        html.Div: Error block indicating the adapter is not implemented.
    """
    return error_block("sankey_plot_adapter not implemented in generic library")


# (Future) def sunburst_plot_adapter(...): pass

# -----------------------------
# High-Level App Wrapper
# -----------------------------


class ConfigurableDashApp:
    """High-level wrapper that assembles and serves multiple dashboard cells.

    Layout modes (``cells`` argument):
        * Flat list -> each cell is its own row.
        * List of lists -> inner list elements share a row and are laid out side-by-side.

    Args:
        cells: Cell configs or grouped row configs.
        app_name: Title of the Dash application.
        port: Port used when running ``start()``.
        global_filters: Optional GlobalFiltersConfig for shared filters across all cells.
    """

    def __init__(
        self,
        cells: List[Union[DashCellConfig, List[DashCellConfig]]],
        app_name: str = "Configurable Funnel Dashboard",
        port: int = 5030,
        dash_class: Optional[Type] = None,
        global_filters: Optional[GlobalFiltersConfig] = None,
        logo_src: Optional[str] = None,
        dashboard_base_url: str = DASHBOARD_BASE_URL,
    ):
        """Create the configurable dashboard application.

        Args:
            cells: Cell configs (flat or grouped by row).
            app_name: Title passed to the Dash constructor.
            port: Server port for ``start()``.
            dash_class: Optional custom Dash-compatible class to instantiate instead
                of the default ``dash.Dash``. Supply e.g. ``CloudNotebookDashWrapper`` to
                integrate Blah's wrapper; it must accept ``app_name`` as first
                init param (``dash.Dash(__name__)`` style variations are handled by
                passing the string only). If ``None`` the standard ``Dash`` import
                is used.
            global_filters: Optional GlobalFiltersConfig for shared filters.
            logo_src: Optional base64-encoded image data URI for the app header logo.
            dashboard_base_url: Base URL used to build ``self.app_link``. Defaults
                to the module-level ``DASHBOARD_BASE_URL`` constant.
        """
        self.port = port
        self.app_name = app_name
        self.global_filters = global_filters
        self.logo_src = logo_src
        # Instantiate either provided custom Dash class (with port) or default Dash
        if dash_class is not None:
            self.app = dash_class(app_name, port=port)
        else:
            self.app = Dash(app_name)
        self.app_link = f"{dashboard_base_url}/user/{{}}/proxy/{port}/"

        # Normalize to list-of-lists for rows; keep flattened list for callback registration.
        self._row_cfgs: List[List[DashCellConfig]] = []
        if not cells:
            raise ValueError("At least one cell config required")
        for item in cells:
            if isinstance(item, list):
                if not item:
                    continue
                self._row_cfgs.append(item)
            else:
                self._row_cfgs.append([item])

        self._rows: List[List[DashCell]] = [[DashCell(c) for c in row] for row in self._row_cfgs]
        # Flatten for callback registration convenience
        self._cells_flat: List[DashCell] = [cell for row in self._rows for cell in row]

        self._build_layout()
        self._register_callbacks()
        self.is_running = False

    def _global_filter_id(self, param_name: str) -> str:
        """Create a stable component id for a global filter control."""
        return f"global__filter__{param_name}"

    def _build_global_filters(self) -> html.Div:
        """Build the global filters panel with primary and collapsible advanced sections."""
        if not self.global_filters:
            return html.Div()

        # Build primary filters row
        primary_controls = [build_global_filter_control(spec, self._global_filter_id(spec.name)) for spec in self.global_filters.primary_specs]
        primary_section = html.Div(
            primary_controls,
            style={**STYLES.controls_row, "gap": "32px"},
        )

        # Build advanced filters (collapsible)
        advanced_section = None
        if self.global_filters.advanced_specs:
            advanced_controls = [build_global_filter_control(spec, self._global_filter_id(spec.name)) for spec in self.global_filters.advanced_specs]
            advanced_section = html.Details(
                [
                    html.Summary(
                        [
                            html.Span("⚙️", style={"marginRight": "8px"}),
                            html.Span("Advanced Filters"),
                        ],
                        style={
                            **STYLES.collapsible_header,
                            "listStyle": "none",
                        },
                    ),
                    html.Div(
                        advanced_controls,
                        style={
                            **STYLES.controls_row,
                            "gap": "32px",
                            "paddingTop": "12px",
                        },
                    ),
                ],
                open=False,
                style={"marginTop": "12px"},
            )

        return html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "🎛️",
                            style={"marginRight": "10px", "fontSize": "20px"},
                        ),
                        html.Span(
                            "Global Filters",
                            style={
                                "fontSize": "18px",
                                "fontWeight": "700",
                                "color": STYLES.colors.text,
                            },
                        ),
                    ],
                    style={"marginBottom": "16px"},
                ),
                primary_section,
                advanced_section if advanced_section else html.Div(),
            ],
            style=STYLES.global_filters_card,
        )

    def _build_layout(self):
        """Construct the application layout with grouped rows."""
        row_divs: List[html.Div] = []
        for row in self._rows:
            cell_divs = [
                html.Div(
                    c.build_layout(),
                    style={
                        "flex": "1 1 0",
                        "minWidth": "300px",
                        "width": "100%",
                        "boxSizing": "border-box",
                    },
                )
                for c in row
            ]
            row_divs.append(
                html.Div(
                    cell_divs,
                    style={
                        "display": "flex",
                        "gap": "24px",
                        "flexWrap": "wrap",
                        "marginBottom": "24px",
                        "width": "100%",
                        "justifyContent": "stretch",
                    },
                )
            )

        header = build_app_header(self.app_name, logo_src=self.logo_src)

        # Build content with optional global filters
        content_children = []
        if self.global_filters:
            content_children.append(self._build_global_filters())
        content_children.append(html.Div(row_divs))

        self.app.layout = html.Div(
            [
                header,
                html.Div(
                    content_children,
                    style=STYLES.content_container,
                ),
            ],
            style=STYLES.app_container,
        )

    def _register_callbacks(self):
        """Register callbacks for all cells, including global filter values."""
        if self.global_filters:
            # Register callbacks with global filters
            for cell in self._cells_flat:
                self._register_cell_callback_with_global_filters(cell)
        else:
            # No global filters - use standard registration
            for cell in self._cells_flat:
                cell.register_callbacks(self.app)

    def _register_cell_callback_with_global_filters(self, cell: DashCell):
        """Register a cell callback that includes global filter values.

        Args:
            cell: The DashCell to register.
        """
        local_specs = list(cell.config.parameter_specs)
        global_specs = self.global_filters.all_specs if self.global_filters else []

        input_component = Input(cell.run_id, "n_clicks")

        # Local states from cell parameters
        local_states = [State(cell._param_id(spec.name), "value") for spec in local_specs]

        # Global states from global filters
        global_states = [State(self._global_filter_id(spec.name), "value") for spec in global_specs]

        all_states = local_states + global_states

        @self.app.callback(
            Output(cell.output_id, "children"),
            [input_component],
            all_states,
            prevent_initial_call=True,
        )
        def _run_cell_with_globals(
            n_clicks,
            *values,
            _local_specs=local_specs,
            _global_specs=global_specs,
            _cell_config=cell.config,
        ):
            try:
                num_local = len(_local_specs)
                params = _process_params(_local_specs, values[:num_local])
                params.update(_process_params(_global_specs, values[num_local:]))

                result = _cell_config.plot_func(**params)
                return wrap_plot_result(
                    result=result,
                    return_type=_cell_config.return_type,
                )
            except Exception as e:
                print(e)
                return error_block(str(e))

    def start(self):
        """Run the Dash development server if not already running."""
        if not self.is_running:
            self.is_running = True
            self.app.run()

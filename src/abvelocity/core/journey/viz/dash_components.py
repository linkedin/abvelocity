"""Component factory and layout builders for configurable Dash components.

This module provides factory functions for creating Dash components and
helper functions for building common layouts, reducing code duplication.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from dash import dcc, html
from abvelocity.core.journey.viz.dash_styles import STYLES, get_chart_icon
from abvelocity.core.journey.viz.dash_types import ParameterSpec

# -----------------------------
# Component Factory
# -----------------------------


class ComponentFactory:
    """Factory for creating Dash input components from ParameterSpec."""

    def __init__(self, styles: Any = None):
        """Initialize the factory with styles.

        Args:
            styles: DashStyles instance (defaults to STYLES).
        """
        self.styles = styles or STYLES

    def create_component(
        self,
        spec: "ParameterSpec",
        component_id: str,
    ) -> Any:
        """Create a Dash component based on the ParameterSpec.

        Args:
            spec: The parameter specification.
            component_id: The unique ID for the component.

        Returns:
            A Dash component (dcc.Dropdown, dcc.Input, etc.).
        """
        creator = self.get_creator(spec.component)
        return creator(spec, component_id)

    def get_creator(self, component_type: str) -> Callable[["ParameterSpec", str], Any]:
        """Get the creator function for a component type.

        Args:
            component_type: The type of component to create.

        Returns:
            A creator function.
        """
        creators = {
            "dropdown": self.create_dropdown,
            "text": self.create_text_input,
            "number": self.create_number_input,
            "date": self.create_date_input,
        }
        return creators.get(component_type, self.create_unsupported)

    def create_dropdown(self, spec: "ParameterSpec", component_id: str) -> dcc.Dropdown:
        """Create a dropdown component."""
        options = [{"label": str(o).title().replace("_", " "), "value": o} for o in (spec.options or [])]
        return dcc.Dropdown(
            id=component_id,
            options=options,
            value=spec.default,
            multi=spec.multi,
            clearable=not spec.required,
            style=self.styles.dropdown,
        )

    def create_text_input(self, spec: "ParameterSpec", component_id: str) -> dcc.Input:
        """Create a text input component."""
        return dcc.Input(
            id=component_id,
            type="text",
            value=spec.default or "",
            placeholder=spec.help or "Enter value...",
            style=self.styles.input,
        )

    def create_number_input(self, spec: "ParameterSpec", component_id: str) -> dcc.Input:
        """Create a number input component."""
        return dcc.Input(
            id=component_id,
            type="number",
            value=spec.default,
            style=self.styles.input,
        )

    def create_date_input(self, spec: "ParameterSpec", component_id: str) -> dcc.Input:
        """Create a date input component."""
        return dcc.Input(
            id=component_id,
            type="date",
            value=spec.default or "",
            placeholder="YYYY-MM-DD",
            style=self.styles.input,
        )

    def create_unsupported(self, spec: "ParameterSpec", component_id: str) -> html.Div:
        """Create a placeholder for unsupported component types."""
        return html.Div(f"Unsupported component: {spec.component}")


# Default factory instance
COMPONENT_FACTORY = ComponentFactory()


# -----------------------------
# Layout Builders
# -----------------------------


def build_labeled_control(
    spec: "ParameterSpec",
    component: Any,
    styles: Any = None,
) -> html.Div:
    """Build a control with label and optional help text.

    Args:
        spec: The parameter specification.
        component: The Dash component to wrap.
        styles: DashStyles instance (defaults to STYLES).

    Returns:
        html.Div containing the labeled control.
    """
    styles = styles or STYLES

    # Build label with required indicator
    label_children = [html.Span(spec.label, style=styles.label)]
    if spec.required:
        label_children.append(html.Span(" *", style=styles.required_indicator))

    children = [
        html.Label(label_children, style=styles.label_container),
        component,
    ]

    # Add help text (except for text inputs which use placeholder)
    if spec.help and spec.component != "text":
        children.append(html.Div(spec.help, style=styles.help_text))

    return html.Div(children, style=styles.control_container)


def build_controls_row(
    controls: List[html.Div],
    styles: Any = None,
) -> html.Div:
    """Build a horizontal row of controls.

    Args:
        controls: List of control divs.
        styles: DashStyles instance (defaults to STYLES).

    Returns:
        html.Div containing the controls in a flex row.
    """
    styles = styles or STYLES
    return html.Div(controls, style=styles.controls_row)


def build_button(
    button_id: str,
    label: str = "Generate",
    icon: str = "↻",
    styles: Any = None,
) -> html.Button:
    """Build a styled action button.

    Args:
        button_id: The unique ID for the button.
        label: The button text.
        icon: The icon to display before the label.
        styles: DashStyles instance (defaults to STYLES).

    Returns:
        html.Button with the specified styling.
    """
    styles = styles or STYLES
    return html.Button(
        [
            html.Span(icon, style={"marginRight": "5px", "fontSize": "12px"}),
            html.Span(label),
        ],
        id=button_id,
        n_clicks=0,
        style=styles.button,
    )


def build_controls_layout(
    controls: List[html.Div],
    button: html.Button,
    position: str,
    styles: Any = None,
) -> html.Div:
    """Build the controls section layout based on button position.

    Args:
        controls: List of control divs.
        button: The action button.
        position: Button position ("left", "right", "below").
        styles: DashStyles instance (defaults to STYLES).

    Returns:
        html.Div containing the complete controls layout.
    """
    styles = styles or STYLES
    button_row = html.Div(button, style=styles.button_row)

    if position == "below" and controls:
        # Stacked layout: controls on top, button below
        return html.Div(
            [
                html.Div(
                    controls,
                    style={**styles.controls_row, "marginBottom": "12px"},
                ),
                button_row,
            ],
            style={**styles.controls_section, "display": "flex", "flexDirection": "column"},
        )
    elif position == "right":
        # Side-by-side: controls left, button right
        controls_div = (
            html.Div(
                controls,
                style={**styles.controls_row, "flex": "1"},
            )
            if controls
            else html.Div()
        )
        return html.Div(
            [
                controls_div,
                html.Div(button, style={**styles.button_row, "marginLeft": "auto"}),
            ],
            style={
                **styles.controls_section,
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "12px",
                "alignItems": "center",
                "justifyContent": "space-between",
            },
        )
    else:
        # Left-aligned: controls and button in row
        children = []
        if controls:
            children.append(html.Div(controls, style=styles.controls_row))
        children.append(button_row)
        return html.Div(
            children,
            style={
                **styles.controls_section,
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "12px",
                "alignItems": "center",
            },
        )


def build_cell_header(
    title: str,
    description: Optional[str] = None,
    styles: Any = None,
) -> List[Any]:
    """Build cell header components with accent bar and icon.

    Args:
        title: The cell title.
        description: Optional description text.
        styles: DashStyles instance (defaults to STYLES).

    Returns:
        List of header components.
    """
    styles = styles or STYLES
    icon = get_chart_icon(title)

    header_row = html.Div(
        [
            html.Div(style=styles.header_accent),
            html.Div(
                [
                    html.Span(icon, style=styles.icon),
                    html.Span(title, style=styles.title),
                ],
                style={"marginBottom": "4px"},
            ),
        ]
    )

    components = [header_row]

    if description:
        components.append(html.Div(description, style=styles.description))

    return components


def build_app_header(
    app_name: str,
    subtitle: str = "Analyze user journeys with interactive visualizations",
    logo_src: Optional[str] = None,
    styles: Any = None,
) -> html.Div:
    """Build the main app header with optional logo.

    Args:
        app_name: The application name.
        subtitle: The subtitle/tagline.
        logo_src: Optional base64-encoded image data URI for a logo.
        styles: DashStyles instance (defaults to STYLES).

    Returns:
        html.Div containing the app header.
    """
    styles = styles or STYLES

    title_children: List[Any] = []
    if logo_src:
        title_children.append(
            html.Div(
                html.Img(
                    src=logo_src,
                    style={"height": "40px", "width": "40px"},
                ),
                style={"marginRight": "16px", "display": "flex", "alignItems": "center"},
            )
        )
    title_children.append(
        html.Div(
            [
                html.H1(app_name, style=styles.app_title),
                html.P(subtitle, style=styles.app_subtitle),
            ]
        )
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        title_children,
                        style={"display": "flex", "alignItems": "center"},
                    ),
                ],
                style={"maxWidth": "1400px", "margin": "0 auto"},
            )
        ],
        style=styles.app_header,
    )


def build_global_filter_control(
    spec: "ParameterSpec",
    component_id: str,
    styles: Any = None,
) -> html.Div:
    """Build a single global filter control with lightweight styling.

    Uses ComponentFactory for component creation and lighter container
    styling appropriate for the global filters panel.

    Args:
        spec: The parameter specification.
        component_id: The unique ID for the component.
        styles: DashStyles instance (defaults to STYLES).

    Returns:
        html.Div containing the labeled control.
    """
    styles = styles or STYLES
    comp = COMPONENT_FACTORY.create_component(spec, component_id)

    label_children = [html.Span(spec.label, style=styles.label)]
    if spec.required:
        label_children.append(html.Span(" *", style=styles.required_indicator))

    children = [
        html.Label(label_children, style=styles.label_container),
        comp,
    ]

    if spec.help and spec.component != "text":
        children.append(html.Div(spec.help, style=styles.global_filter_help_text))

    return html.Div(children, style=styles.global_filter_control_container)

"""Centralized styles for configurable Dash components.

This module contains all styling definitions used by the dashboard components,
making it easy to maintain consistent styling and modify the design system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ColorPalette:
    """Design system color palette"""

    primary: str = "#0077B5"  # Blue
    primary_hover: str = "#005885"
    primary_light: str = "#E8F4F9"
    background: str = "#ffffff"
    surface: str = "#f8fafc"
    border: str = "#e2e8f0"
    border_light: str = "#f1f5f9"
    text: str = "#1e293b"
    text_secondary: str = "#64748b"
    text_muted: str = "#94a3b8"
    success: str = "#10b981"
    error: str = "#ef4444"


# Default color palette instance
COLORS = ColorPalette()


@dataclass
class DashStyles:
    """Centralized style definitions for Dash components.

    All styles are computed based on the color palette, ensuring consistency.
    """

    colors: ColorPalette = field(default_factory=lambda: COLORS)

    @property
    def input(self) -> Dict[str, Any]:
        """Style for text/number/date input fields."""
        return {
            "width": "100%",
            "padding": "10px 12px",
            "border": f"1px solid {self.colors.border}",
            "borderRadius": "8px",
            "fontSize": "14px",
            "transition": "border-color 0.2s, box-shadow 0.2s",
            "outline": "none",
            "backgroundColor": self.colors.background,
        }

    @property
    def dropdown(self) -> Dict[str, Any]:
        """Style for dropdown components."""
        return {"width": "100%", "fontSize": "14px"}

    @property
    def button(self) -> Dict[str, Any]:
        """Style for action buttons (outlined style)."""
        return {
            "padding": "8px 16px",
            "fontSize": "13px",
            "fontWeight": "600",
            "color": self.colors.primary,
            "backgroundColor": "#ffffff",
            "border": f"1.5px solid {self.colors.primary}",
            "borderRadius": "6px",
            "cursor": "pointer",
            "transition": "all 0.15s ease",
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
            "whiteSpace": "nowrap",
        }

    @property
    def label(self) -> Dict[str, Any]:
        """Style for form labels."""
        return {
            "fontWeight": "600",
            "color": self.colors.text,
            "fontSize": "13px",
            "letterSpacing": "0.01em",
        }

    @property
    def label_container(self) -> Dict[str, Any]:
        """Style for label container."""
        return {"display": "block", "marginBottom": "6px"}

    @property
    def required_indicator(self) -> Dict[str, Any]:
        """Style for required field indicator (*)."""
        return {"color": self.colors.error, "fontWeight": "600"}

    @property
    def help_text(self) -> Dict[str, Any]:
        """Style for help/hint text below controls."""
        return {
            "fontSize": "12px",
            "color": self.colors.text_muted,
            "marginTop": "6px",
            "lineHeight": "1.4",
        }

    @property
    def global_filter_control_container(self) -> Dict[str, Any]:
        """Style for global filter control wrapper (lightweight, no border)."""
        return {
            "flex": "1 1 200px",
            "minWidth": "160px",
            "maxWidth": "300px",
        }

    @property
    def global_filter_help_text(self) -> Dict[str, Any]:
        """Style for help text in global filter controls (compact)."""
        return {
            "fontSize": "11px",
            "color": self.colors.text_muted,
            "marginTop": "4px",
            "lineHeight": "1.3",
        }

    @property
    def control_container(self) -> Dict[str, Any]:
        """Style for individual control wrapper."""
        return {
            "flex": "1 1 200px",
            "minWidth": "180px",
            "maxWidth": "280px",
            "padding": "12px 14px",
            "boxSizing": "border-box",
            "backgroundColor": self.colors.background,
            "border": f"1px solid {self.colors.border}",
            "borderRadius": "10px",
            "transition": "box-shadow 0.2s, border-color 0.2s",
        }

    @property
    def controls_row(self) -> Dict[str, Any]:
        """Style for horizontal row of controls."""
        return {
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "12px",
            "alignItems": "flex-start",
        }

    @property
    def controls_section(self) -> Dict[str, Any]:
        """Style for controls section container."""
        return {
            "margin": "0",
            "padding": "16px",
            "backgroundColor": self.colors.surface,
            "borderRadius": "12px",
        }

    @property
    def button_row(self) -> Dict[str, Any]:
        """Style for button container row."""
        return {"display": "flex", "alignItems": "center"}

    @property
    def cell_card(self) -> Dict[str, Any]:
        """Style for cell card container."""
        return {
            "backgroundColor": self.colors.background,
            "border": f"1px solid {self.colors.border}",
            "borderRadius": "12px",
            "padding": "20px",
            "boxShadow": ("0 4px 6px -1px rgba(0, 0, 0, 0.07), " "0 2px 4px -1px rgba(0, 0, 0, 0.04)"),
            "transition": "box-shadow 0.2s ease",
            "width": "100%",
            "boxSizing": "border-box",
        }

    @property
    def header_accent(self) -> Dict[str, Any]:
        """Style for header accent bar."""
        return {
            "height": "4px",
            "background": f"linear-gradient(90deg, {self.colors.primary} 0%, #00a0dc 100%)",
            "borderRadius": "12px 12px 0 0",
            "margin": "-20px -20px 16px -20px",
        }

    @property
    def title(self) -> Dict[str, Any]:
        """Style for section titles."""
        return {
            "fontSize": "20px",
            "fontWeight": "700",
            "color": self.colors.text,
            "verticalAlign": "middle",
        }

    @property
    def description(self) -> Dict[str, Any]:
        """Style for section descriptions."""
        return {
            "color": self.colors.text_secondary,
            "marginBottom": "16px",
            "fontSize": "14px",
            "lineHeight": "1.5",
        }

    @property
    def icon(self) -> Dict[str, Any]:
        """Style for section icons."""
        return {
            "fontSize": "24px",
            "marginRight": "12px",
            "verticalAlign": "middle",
        }

    @property
    def separator(self) -> Dict[str, Any]:
        """Style for horizontal separator lines."""
        return {
            "height": "1px",
            "backgroundColor": self.colors.border_light,
            "margin": "8px 0 16px 0",
        }

    @property
    def output_container(self) -> Dict[str, Any]:
        """Style for output/plot container."""
        return {"minHeight": "100px", "padding": "16px 0"}

    @property
    def app_header(self) -> Dict[str, Any]:
        """Style for main app header."""
        return {
            "background": "linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%)",
            "padding": "32px 24px",
            "marginBottom": "24px",
            "borderBottom": "1px solid #e2e8f0",
        }

    @property
    def app_title(self) -> Dict[str, Any]:
        """Style for main app title."""
        return {
            "margin": "0",
            "fontSize": "28px",
            "fontWeight": "700",
            "color": "#1e293b",
            "letterSpacing": "-0.02em",
        }

    @property
    def app_subtitle(self) -> Dict[str, Any]:
        """Style for app subtitle/tagline."""
        return {
            "margin": "4px 0 0 0",
            "fontSize": "15px",
            "color": "#64748b",
        }

    @property
    def app_container(self) -> Dict[str, Any]:
        """Style for main app container."""
        return {
            "fontFamily": ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, " "'Helvetica Neue', Arial, sans-serif"),
            "backgroundColor": "#f8fafc",
            "minHeight": "100vh",
        }

    @property
    def content_container(self) -> Dict[str, Any]:
        """Style for main content container."""
        return {
            "maxWidth": "2700px",
            "margin": "0 auto",
            "padding": "0 24px 40px 24px",
        }

    @property
    def global_filters_card(self) -> Dict[str, Any]:
        """Style for global filters card."""
        return {
            "backgroundColor": self.colors.background,
            "border": f"1px solid {self.colors.border}",
            "borderRadius": "12px",
            "padding": "20px",
            "marginBottom": "24px",
            "boxShadow": ("0 4px 6px -1px rgba(0, 0, 0, 0.07), " "0 2px 4px -1px rgba(0, 0, 0, 0.04)"),
        }

    @property
    def collapsible_header(self) -> Dict[str, Any]:
        """Style for collapsible section header."""
        return {
            "display": "flex",
            "alignItems": "center",
            "cursor": "pointer",
            "padding": "8px 0",
            "color": self.colors.text_secondary,
            "fontSize": "14px",
            "fontWeight": "500",
            "userSelect": "none",
        }

    # --- Placeholder & error styles ---

    @property
    def placeholder(self) -> Dict[str, Any]:
        """Style for placeholder/hint messages."""
        return {
            "color": self.colors.text_secondary,
            "fontStyle": "italic",
            "fontSize": "14px",
            "padding": "24px",
            "textAlign": "center",
            "backgroundColor": self.colors.surface,
            "borderRadius": "8px",
            "border": f"1px dashed {self.colors.border}",
        }

    @property
    def error_container(self) -> Dict[str, Any]:
        """Style for error block outer container."""
        return {
            "backgroundColor": self.colors.background,
            "padding": "16px",
            "border": "1px solid #fecaca",
            "borderRadius": "10px",
            "boxShadow": "0 1px 3px rgba(220, 38, 38, 0.1)",
        }

    @property
    def error_header(self) -> Dict[str, Any]:
        """Style for error block header row."""
        return {
            "color": "#dc2626",
            "marginBottom": "8px",
            "fontSize": "14px",
        }

    @property
    def error_pre(self) -> Dict[str, Any]:
        """Style for error block pre-formatted text."""
        return {
            "whiteSpace": "pre-wrap",
            "color": "#991b1b",
            "fontSize": "13px",
            "margin": "0",
            "padding": "12px",
            "backgroundColor": "#fef2f2",
            "borderRadius": "6px",
            "border": "1px solid #fecaca",
        }

    @property
    def result_title(self) -> Dict[str, Any]:
        """Style for result section title (H4)."""
        return {
            "margin": "0 0 4px 0",
            "fontSize": "16px",
            "fontWeight": "600",
            "color": self.colors.text,
        }

    @property
    def result_timestamp(self) -> Dict[str, Any]:
        """Style for result generation timestamp."""
        return {
            "fontSize": "12px",
            "color": self.colors.text_muted,
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "4px",
        }

    @property
    def result_block(self) -> Dict[str, Any]:
        """Style for result/output wrapper block."""
        return {
            "backgroundColor": self.colors.background,
            "borderRadius": "8px",
            "padding": "4px",
        }


# Default styles instance
STYLES = DashStyles()


# Icon mapping for chart types
CHART_ICONS = {
    "sankey": "🔀",
    "sunburst": "☀️",
    "bar": "📊",
    "default": "📈",
}


def get_chart_icon(title: str) -> str:
    """Get the appropriate icon for a chart based on its title.

    Args:
        title: The chart title to match against.

    Returns:
        The icon string for the chart type.
    """
    title_lower = title.lower()
    for key, icon in CHART_ICONS.items():
        if key in title_lower:
            return icon
    return CHART_ICONS["default"]

# BSD 2-CLAUSE LICENSE
# noqa: E501
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unit tests for dash_styles module.

Tests cover:
- ColorPalette dataclass defaults and immutability
- DashStyles property-based style definitions
- Global filter specific styles
- get_chart_icon helper function
"""

import pytest
from abvelocity.core.journey.viz.dash_styles import CHART_ICONS, COLORS, STYLES, ColorPalette, DashStyles, get_chart_icon

# --- ColorPalette tests ---


class TestColorPalette:
    """Tests for ColorPalette dataclass."""

    def test_default_primary_is_blah_blue(self):
        """Primary color should be Blah blue."""
        assert COLORS.primary == "#0077B5"

    def test_all_defaults_are_strings(self):
        """All color values should be strings."""
        palette = ColorPalette()
        for attr in [
            "primary",
            "primary_hover",
            "primary_light",
            "background",
            "surface",
            "border",
            "border_light",
            "text",
            "text_secondary",
            "text_muted",
            "success",
            "error",
        ]:
            assert isinstance(getattr(palette, attr), str)

    def test_frozen_prevents_mutation(self):
        """ColorPalette should be immutable (frozen)."""
        with pytest.raises(AttributeError):
            COLORS.primary = "#ff0000"

    def test_custom_palette(self):
        """ColorPalette can be created with custom values."""
        custom = ColorPalette(primary="#123456", error="#abcdef")
        assert custom.primary == "#123456"
        assert custom.error == "#abcdef"
        # Other fields should retain defaults
        assert custom.background == "#ffffff"


# --- DashStyles tests ---


class TestDashStyles:
    """Tests for DashStyles dataclass."""

    def test_default_instance_uses_default_colors(self):
        """Default STYLES instance should use default COLORS."""
        assert STYLES.colors == COLORS

    def test_custom_colors_propagate(self):
        """Styles should reflect custom color palette."""
        custom_palette = ColorPalette(primary="#111111", border="#222222")
        custom_styles = DashStyles(colors=custom_palette)
        assert custom_styles.colors.primary == "#111111"
        assert "#222222" in custom_styles.input["border"]

    def test_input_style_has_required_keys(self):
        """Input style should have standard CSS properties."""
        style = STYLES.input
        assert style["width"] == "100%"
        assert "border" in style
        assert "borderRadius" in style
        assert "fontSize" in style

    def test_dropdown_style(self):
        """Dropdown style should set width and font size."""
        style = STYLES.dropdown
        assert style["width"] == "100%"
        assert style["fontSize"] == "14px"

    def test_button_style_has_required_keys(self):
        """Button style should have interactive CSS properties."""
        style = STYLES.button
        assert style["cursor"] == "pointer"
        assert "border" in style
        assert style["display"] == "inline-flex"

    def test_label_style(self):
        """Label style should have font properties."""
        style = STYLES.label
        assert style["fontWeight"] == "600"
        assert style["fontSize"] == "13px"

    def test_control_container_has_flex_properties(self):
        """Control container should have flex layout properties."""
        style = STYLES.control_container
        assert "flex" in style
        assert "minWidth" in style
        assert "maxWidth" in style
        assert "border" in style

    def test_controls_row_is_flex(self):
        """Controls row should use flex layout."""
        style = STYLES.controls_row
        assert style["display"] == "flex"
        assert style["flexWrap"] == "wrap"

    def test_cell_card_has_shadow(self):
        """Cell card should have box shadow."""
        style = STYLES.cell_card
        assert "boxShadow" in style
        assert style["borderRadius"] == "12px"

    def test_global_filter_control_container_is_lightweight(self):
        """Global filter container should NOT have border or background."""
        style = STYLES.global_filter_control_container
        assert "flex" in style
        assert "minWidth" in style
        assert "maxWidth" in style
        # Should be lighter than regular control_container
        assert "border" not in style
        assert "backgroundColor" not in style

    def test_global_filter_help_text_is_compact(self):
        """Global filter help text should be smaller than regular help text."""
        gf_style = STYLES.global_filter_help_text
        regular_style = STYLES.help_text
        assert gf_style["fontSize"] == "11px"
        assert regular_style["fontSize"] == "12px"

    def test_app_header_has_gradient(self):
        """App header should have gradient background."""
        style = STYLES.app_header
        assert "linear-gradient" in style["background"]

    def test_separator_style(self):
        """Separator should be a thin horizontal line."""
        style = STYLES.separator
        assert style["height"] == "1px"

    def test_global_filters_card_has_shadow(self):
        """Global filters card should have box shadow."""
        style = STYLES.global_filters_card
        assert "boxShadow" in style
        assert style["marginBottom"] == "24px"

    def test_collapsible_header_is_clickable(self):
        """Collapsible header should have pointer cursor."""
        style = STYLES.collapsible_header
        assert style["cursor"] == "pointer"
        assert style["userSelect"] == "none"

    def test_placeholder_style(self):
        """Placeholder style should have dashed border and italic font."""
        style = STYLES.placeholder
        assert style["fontStyle"] == "italic"
        assert "dashed" in style["border"]

    def test_error_container_style(self):
        """Error container should have border and shadow."""
        style = STYLES.error_container
        assert "border" in style
        assert "boxShadow" in style

    def test_error_header_style(self):
        """Error header should have red color."""
        style = STYLES.error_header
        assert "#dc2626" in style["color"]

    def test_error_pre_style(self):
        """Error pre should have pre-wrap white-space."""
        style = STYLES.error_pre
        assert style["whiteSpace"] == "pre-wrap"

    def test_result_block_style(self):
        """Result block should have background and border radius."""
        style = STYLES.result_block
        assert "backgroundColor" in style
        assert "borderRadius" in style


# --- get_chart_icon tests ---


class TestGetChartIcon:
    """Tests for get_chart_icon helper function."""

    def test_sankey_icon(self):
        """Sankey chart titles should get the sankey icon."""
        assert get_chart_icon("Sankey Diagram") == CHART_ICONS["sankey"]

    def test_sunburst_icon(self):
        """Sunburst chart titles should get the sunburst icon."""
        assert get_chart_icon("Sunburst Chart") == CHART_ICONS["sunburst"]

    def test_bar_icon(self):
        """Bar chart titles should get the bar icon."""
        assert get_chart_icon("Barchart View") == CHART_ICONS["bar"]

    def test_unknown_title_gets_default(self):
        """Unknown chart types should get the default icon."""
        assert get_chart_icon("My Custom Visualization") == CHART_ICONS["default"]

    def test_case_insensitive(self):
        """Icon lookup should be case-insensitive."""
        assert get_chart_icon("SANKEY") == CHART_ICONS["sankey"]
        assert get_chart_icon("sunBURST") == CHART_ICONS["sunburst"]

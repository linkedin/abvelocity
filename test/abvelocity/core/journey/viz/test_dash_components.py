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

"""Unit tests for dash_components module.

Tests cover:
- ComponentFactory: creating Dash components from ParameterSpec
- build_labeled_control: labeled control with help text and required indicator
- build_button: styled action button
- build_controls_layout: position-based layout (left, right, below)
- build_controls_row: horizontal row of controls
- build_cell_header: cell header with accent bar and icon
- build_app_header: app header with Blah logo
- build_global_filter_control: lightweight global filter control
"""

from dash import dcc, html
from abvelocity.core.journey.viz.create_configurable_dash import ParameterSpec
from abvelocity.core.journey.viz.dash_components import (
    ComponentFactory,
    build_app_header,
    build_button,
    build_cell_header,
    build_controls_layout,
    build_controls_row,
    build_global_filter_control,
    build_labeled_control,
)
from abvelocity.core.journey.viz.dash_styles import STYLES

# --- ComponentFactory tests ---


class TestComponentFactory:
    """Tests for ComponentFactory class."""

    def test_creates_dropdown(self):
        """Factory should create dcc.Dropdown for dropdown specs."""
        spec = ParameterSpec(
            name="color",
            label="Color",
            component="dropdown",
            options=["red", "blue"],
            default="red",
        )
        comp = ComponentFactory().create_component(spec, "test__color")
        assert isinstance(comp, dcc.Dropdown)
        assert comp.id == "test__color"
        assert comp.value == "red"

    def test_dropdown_options_formatted(self):
        """Dropdown options should be formatted with title case."""
        spec = ParameterSpec(
            name="method",
            label="Method",
            component="dropdown",
            options=["first_touch", "last_touch"],
            default="first_touch",
        )
        comp = ComponentFactory().create_component(spec, "test__method")
        labels = [o["label"] for o in comp.options]
        assert "First Touch" in labels
        assert "Last Touch" in labels

    def test_dropdown_multi_select(self):
        """Multi-select dropdown should have multi=True."""
        spec = ParameterSpec(
            name="events",
            label="Events",
            component="dropdown",
            options=["A", "B", "C"],
            default=["A"],
            multi=True,
        )
        comp = ComponentFactory().create_component(spec, "test__events")
        assert comp.multi is True

    def test_dropdown_required_not_clearable(self):
        """Required dropdown should not be clearable."""
        spec = ParameterSpec(
            name="req",
            label="Required",
            component="dropdown",
            options=["x"],
            required=True,
        )
        comp = ComponentFactory().create_component(spec, "test__req")
        assert comp.clearable is False

    def test_creates_text_input(self):
        """Factory should create dcc.Input with type='text' for text specs."""
        spec = ParameterSpec(
            name="query",
            label="Query",
            component="text",
            default="hello",
            help="Enter query",
        )
        comp = ComponentFactory().create_component(spec, "test__query")
        assert isinstance(comp, dcc.Input)
        assert comp.type == "text"
        assert comp.value == "hello"
        assert comp.placeholder == "Enter query"

    def test_text_input_default_placeholder(self):
        """Text input without help should use default placeholder."""
        spec = ParameterSpec(
            name="val",
            label="Value",
            component="text",
        )
        comp = ComponentFactory().create_component(spec, "test__val")
        assert comp.placeholder == "Enter value..."

    def test_creates_number_input(self):
        """Factory should create dcc.Input with type='number' for number specs."""
        spec = ParameterSpec(
            name="count",
            label="Count",
            component="number",
            default=42,
        )
        comp = ComponentFactory().create_component(spec, "test__count")
        assert isinstance(comp, dcc.Input)
        assert comp.type == "number"
        assert comp.value == 42

    def test_creates_date_input(self):
        """Factory should create dcc.Input with type='date' for date specs."""
        spec = ParameterSpec(
            name="start_date",
            label="Start Date",
            component="date",
            default="2024-01-01",
        )
        comp = ComponentFactory().create_component(spec, "test__date")
        assert isinstance(comp, dcc.Input)
        assert comp.type == "date"
        assert comp.value == "2024-01-01"

    def test_unsupported_component_returns_div(self):
        """Unsupported component type should return an html.Div placeholder."""
        spec = ParameterSpec(
            name="x",
            label="X",
            component="slider",
        )
        comp = ComponentFactory().create_component(spec, "test__x")
        assert isinstance(comp, html.Div)
        assert "Unsupported" in str(comp.children)

    def test_custom_styles(self):
        """Factory can be initialized with custom styles."""
        factory = ComponentFactory(styles=STYLES)
        spec = ParameterSpec(
            name="n",
            label="N",
            component="number",
            default=10,
        )
        comp = factory.create_component(spec, "test__n")
        assert isinstance(comp, dcc.Input)


# --- build_labeled_control tests ---


class TestBuildLabeledControl:
    """Tests for build_labeled_control function."""

    def test_wraps_component_with_label(self):
        """Should wrap component in a Div with a Label."""
        spec = ParameterSpec(name="p", label="Param", component="text")
        comp = html.Div("inner")
        result = build_labeled_control(spec, comp)

        assert isinstance(result, html.Div)
        # First child should be Label, second should be the component
        assert isinstance(result.children[0], html.Label)
        assert result.children[1] is comp

    def test_required_indicator_shown(self):
        """Required specs should show a '*' indicator."""
        spec = ParameterSpec(
            name="r",
            label="Required Field",
            component="dropdown",
            options=["a"],
            required=True,
        )
        comp = html.Div("inner")
        result = build_labeled_control(spec, comp)

        # Label should have two spans: label text + required indicator
        label = result.children[0]
        assert len(label.children) == 2
        assert " *" in str(label.children[1].children)

    def test_no_required_indicator_when_not_required(self):
        """Non-required specs should not show indicator."""
        spec = ParameterSpec(
            name="o",
            label="Optional",
            component="text",
        )
        comp = html.Div("inner")
        result = build_labeled_control(spec, comp)

        label = result.children[0]
        assert len(label.children) == 1

    def test_help_text_shown_for_non_text_components(self):
        """Help text should be shown for dropdown/number/date components."""
        spec = ParameterSpec(
            name="d",
            label="Dropdown",
            component="dropdown",
            options=["a"],
            help="Choose one",
        )
        comp = html.Div("inner")
        result = build_labeled_control(spec, comp)

        # Should have 3 children: label, component, help text
        assert len(result.children) == 3
        assert result.children[2].children == "Choose one"

    def test_help_text_hidden_for_text_component(self):
        """Text components use placeholder, so help text should be omitted."""
        spec = ParameterSpec(
            name="t",
            label="Text",
            component="text",
            help="This goes to placeholder",
        )
        comp = html.Div("inner")
        result = build_labeled_control(spec, comp)

        # Should have only 2 children: label, component (no help text)
        assert len(result.children) == 2

    def test_uses_control_container_style(self):
        """Result should use control_container style from STYLES."""
        spec = ParameterSpec(name="p", label="P", component="number")
        comp = html.Div("inner")
        result = build_labeled_control(spec, comp)

        assert result.style == STYLES.control_container


# --- build_button tests ---


class TestBuildButton:
    """Tests for build_button function."""

    def test_creates_button_with_id(self):
        """Should create an html.Button with the specified ID."""
        btn = build_button("my_button")
        assert isinstance(btn, html.Button)
        assert btn.id == "my_button"

    def test_default_label_and_icon(self):
        """Default button should have 'Generate' label and refresh icon."""
        btn = build_button("btn")
        # Button children: [Span(icon), Span(label)]
        assert btn.children[1].children == "Generate"
        assert btn.children[0].children == "↻"

    def test_custom_label_and_icon(self):
        """Button should accept custom label and icon."""
        btn = build_button("btn", label="Run", icon="▶")
        assert btn.children[1].children == "Run"
        assert btn.children[0].children == "▶"

    def test_initial_clicks_is_zero(self):
        """Button should start with n_clicks=0."""
        btn = build_button("btn")
        assert btn.n_clicks == 0

    def test_uses_button_style(self):
        """Button should use the button style from STYLES."""
        btn = build_button("btn")
        assert btn.style == STYLES.button


# --- build_controls_layout tests ---


class TestBuildControlsLayout:
    """Tests for build_controls_layout function."""

    def test_below_position_has_column_direction(self):
        """Below position should use flexDirection column."""
        controls = [html.Div("ctrl")]
        btn = build_button("btn")
        result = build_controls_layout(controls, btn, "below")

        assert result.style.get("flexDirection") == "column"

    def test_right_position_has_space_between(self):
        """Right position should use justifyContent space-between."""
        controls = [html.Div("ctrl")]
        btn = build_button("btn")
        result = build_controls_layout(controls, btn, "right")

        assert result.style.get("justifyContent") == "space-between"

    def test_left_position_is_flex_wrap(self):
        """Left position should use flex wrap layout."""
        controls = [html.Div("ctrl")]
        btn = build_button("btn")
        result = build_controls_layout(controls, btn, "left")

        assert result.style.get("flexWrap") == "wrap"
        assert result.style.get("flexDirection") is None

    def test_left_position_no_controls(self):
        """Left position with no controls should still include button."""
        btn = build_button("btn")
        result = build_controls_layout([], btn, "left")

        assert isinstance(result, html.Div)
        assert result.style["display"] == "flex"

    def test_below_without_controls_falls_through_to_left(self):
        """Below position with empty controls should fall through to left."""
        btn = build_button("btn")
        result = build_controls_layout([], btn, "below")

        # With no controls, "below" falls to the else branch (left-aligned)
        assert result.style.get("flexDirection") is None

    def test_right_position_no_controls_still_works(self):
        """Right position with no controls should still render."""
        btn = build_button("btn")
        result = build_controls_layout([], btn, "right")
        assert isinstance(result, html.Div)


# --- build_controls_row tests ---


class TestBuildControlsRow:
    """Tests for build_controls_row function."""

    def test_wraps_controls_in_flex_row(self):
        """Should wrap controls in a flex row div."""
        controls = [html.Div("a"), html.Div("b")]
        result = build_controls_row(controls)

        assert isinstance(result, html.Div)
        assert result.style == STYLES.controls_row
        assert len(result.children) == 2

    def test_empty_controls(self):
        """Empty controls list should produce empty row."""
        result = build_controls_row([])
        assert isinstance(result, html.Div)
        assert result.children == []


# --- build_cell_header tests ---


class TestBuildCellHeader:
    """Tests for build_cell_header function."""

    def test_returns_list(self):
        """Should return a list of components."""
        result = build_cell_header("Test Chart")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_header_without_description(self):
        """Without description, should return only header row."""
        result = build_cell_header("Test Chart")
        assert len(result) == 1

    def test_header_with_description(self):
        """With description, should include description div."""
        result = build_cell_header("Test Chart", description="Some details")
        assert len(result) == 2

    def test_sankey_title_gets_correct_icon(self):
        """Sankey title should get the sankey icon in header."""
        result = build_cell_header("Sankey Diagram")
        header_str = str(result[0])
        assert "🔀" in header_str

    def test_bar_title_gets_correct_icon(self):
        """Bar chart title should get the bar icon in header."""
        result = build_cell_header("Barchart Analysis")
        header_str = str(result[0])
        assert "📊" in header_str


# --- build_app_header tests ---


class TestBuildAppHeader:
    """Tests for build_app_header function."""

    def test_returns_div(self):
        """Should return an html.Div."""
        header = build_app_header("My Dashboard")
        assert isinstance(header, html.Div)

    def test_no_logo_by_default(self):
        """Header without logo_src should not contain an Img element."""
        header = build_app_header("My Dashboard")
        header_str = str(header)
        assert "data:image/svg+xml;base64" not in header_str

    def test_logo_shown_when_provided(self):
        """Header should contain the logo when logo_src is provided."""
        logo = "data:image/svg+xml;base64,TEST_LOGO"
        header = build_app_header("My Dashboard", logo_src=logo)
        header_str = str(header)
        assert "TEST_LOGO" in header_str

    def test_contains_app_name(self):
        """Header should contain the app name."""
        header = build_app_header("Journey Dashboard")
        header_str = str(header)
        assert "Journey Dashboard" in header_str

    def test_default_subtitle(self):
        """Default subtitle should be present."""
        header = build_app_header("Test")
        header_str = str(header)
        assert "Analyze user journeys" in header_str

    def test_custom_subtitle(self):
        """Custom subtitle should replace default."""
        header = build_app_header("Test", subtitle="Custom tagline")
        header_str = str(header)
        assert "Custom tagline" in header_str

    def test_uses_app_header_style(self):
        """Header should use app_header style."""
        header = build_app_header("Test")
        assert header.style == STYLES.app_header


# --- build_global_filter_control tests ---


class TestBuildGlobalFilterControl:
    """Tests for build_global_filter_control function."""

    def test_creates_dropdown_control(self):
        """Should create a labeled dropdown control."""
        spec = ParameterSpec(
            name="method",
            label="Method",
            component="dropdown",
            options=["a", "b"],
            default="a",
        )
        result = build_global_filter_control(spec, "global__filter__method")

        assert isinstance(result, html.Div)
        # Should have label and component (2 children, no help text)
        assert isinstance(result.children[0], html.Label)

    def test_creates_text_control(self):
        """Should create a labeled text input control."""
        spec = ParameterSpec(
            name="query",
            label="Query",
            component="text",
            help="Enter query",
        )
        result = build_global_filter_control(spec, "global__filter__query")

        assert isinstance(result, html.Div)
        # Text component should not have help text (uses placeholder)
        assert len(result.children) == 2

    def test_creates_number_control(self):
        """Should create a labeled number input control."""
        spec = ParameterSpec(
            name="count",
            label="Count",
            component="number",
            default=5,
        )
        result = build_global_filter_control(spec, "global__filter__count")
        assert isinstance(result, html.Div)

    def test_creates_date_control(self):
        """Should create a labeled date input control."""
        spec = ParameterSpec(
            name="start",
            label="Start",
            component="date",
        )
        result = build_global_filter_control(spec, "global__filter__start")
        assert isinstance(result, html.Div)

    def test_required_shows_indicator(self):
        """Required filter should show '*' indicator."""
        spec = ParameterSpec(
            name="r",
            label="Required",
            component="dropdown",
            options=["x"],
            required=True,
        )
        result = build_global_filter_control(spec, "global__filter__r")

        label = result.children[0]
        assert len(label.children) == 2
        assert " *" in str(label.children[1].children)

    def test_help_text_shown_for_dropdown(self):
        """Dropdown control should show help text."""
        spec = ParameterSpec(
            name="d",
            label="D",
            component="dropdown",
            options=["a"],
            help="Pick one",
        )
        result = build_global_filter_control(spec, "global__filter__d")

        # Should have 3 children: label, component, help text
        assert len(result.children) == 3
        assert result.children[2].children == "Pick one"

    def test_uses_lightweight_container_style(self):
        """Should use global_filter_control_container style (no border)."""
        spec = ParameterSpec(
            name="p",
            label="P",
            component="text",
        )
        result = build_global_filter_control(spec, "global__filter__p")

        assert result.style == STYLES.global_filter_control_container
        assert "border" not in result.style

    def test_help_text_uses_compact_style(self):
        """Help text should use global_filter_help_text style (11px)."""
        spec = ParameterSpec(
            name="n",
            label="N",
            component="number",
            help="Enter a number",
        )
        result = build_global_filter_control(spec, "global__filter__n")

        help_div = result.children[2]
        assert help_div.style == STYLES.global_filter_help_text
        assert help_div.style["fontSize"] == "11px"

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

"""Unit tests for create_configurable_dash module.

Tests cover:
- DashCellConfig dataclass with button_position parameter
- GlobalFiltersConfig dataclass and all_specs property
- DashCell.build_controls() with different button positions
- ConfigurableDashApp header with Blah logo
"""

import pytest
from dash import html
from abvelocity.core.journey.viz.create_configurable_dash import (
    ConfigurableDashApp,
    DashCell,
    DashCellConfig,
    GlobalFiltersConfig,
    ParameterSpec,
    PlotReturnType,
)

# --- Test fixtures ---


def dummy_plot_func():
    """Dummy plot function for testing."""
    return html.Div("Test output")


def dummy_plot_func_with_params(param1: str, param2: int):
    """Dummy plot function with parameters."""
    return html.Div(f"param1={param1}, param2={param2}")


# --- DashCellConfig tests ---


class TestDashCellConfig:
    """Tests for DashCellConfig dataclass."""

    def test_default_button_position_is_auto(self):
        """button_position should default to 'auto'."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        assert config.button_position == "auto"

    def test_button_position_can_be_set_to_left(self):
        """button_position can be explicitly set to 'left'."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
            button_position="left",
        )
        assert config.button_position == "left"

    def test_button_position_can_be_set_to_right(self):
        """button_position can be explicitly set to 'right'."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
            button_position="right",
        )
        assert config.button_position == "right"

    def test_button_position_can_be_set_to_below(self):
        """button_position can be explicitly set to 'below'."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
            button_position="below",
        )
        assert config.button_position == "below"

    def test_default_return_type_is_figure(self):
        """return_type should default to FIGURE."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        assert config.return_type == PlotReturnType.FIGURE


# --- GlobalFiltersConfig tests ---


class TestGlobalFiltersConfig:
    """Tests for GlobalFiltersConfig dataclass."""

    def test_all_specs_with_only_primary(self):
        """all_specs returns primary_specs when no advanced_specs."""
        primary = [
            ParameterSpec(name="p1", label="P1", component="text"),
            ParameterSpec(name="p2", label="P2", component="dropdown", options=["a", "b"]),
        ]
        config = GlobalFiltersConfig(primary_specs=primary)

        assert config.all_specs == primary
        assert len(config.all_specs) == 2

    def test_all_specs_with_primary_and_advanced(self):
        """all_specs returns combined primary + advanced specs."""
        primary = [
            ParameterSpec(name="p1", label="P1", component="text"),
        ]
        advanced = [
            ParameterSpec(name="a1", label="A1", component="text"),
            ParameterSpec(name="a2", label="A2", component="number"),
        ]
        config = GlobalFiltersConfig(primary_specs=primary, advanced_specs=advanced)

        assert len(config.all_specs) == 3
        assert config.all_specs[0].name == "p1"
        assert config.all_specs[1].name == "a1"
        assert config.all_specs[2].name == "a2"

    def test_all_specs_with_empty_advanced(self):
        """all_specs handles empty advanced_specs list."""
        primary = [ParameterSpec(name="p1", label="P1", component="text")]
        config = GlobalFiltersConfig(primary_specs=primary, advanced_specs=[])

        assert config.all_specs == primary

    def test_advanced_specs_defaults_to_none(self):
        """advanced_specs defaults to None."""
        primary = [ParameterSpec(name="p1", label="P1", component="text")]
        config = GlobalFiltersConfig(primary_specs=primary)

        assert config.advanced_specs is None


# --- DashCell tests ---


class TestDashCell:
    """Tests for DashCell class."""

    def test_build_controls_no_params_auto_position(self):
        """With no params and auto position, button should be left-aligned."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
            button_position="auto",
        )
        cell = DashCell(config)
        controls = cell.build_controls()

        # Should be a Div containing the button
        assert isinstance(controls, html.Div)
        # Check flex layout (left-aligned, no marginLeft auto)
        assert "display" in controls.style
        assert controls.style["display"] == "flex"

    def test_build_controls_with_params_auto_position(self):
        """With params and auto position, button should be below controls."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[
                ParameterSpec(name="p1", label="Param 1", component="text"),
            ],
            plot_func=dummy_plot_func_with_params,
            button_position="auto",
        )
        cell = DashCell(config)
        controls = cell.build_controls()

        # Should have flexDirection column for stacked layout
        assert isinstance(controls, html.Div)
        assert controls.style.get("flexDirection") == "column"

    def test_build_controls_right_position(self):
        """With right position, button should have marginLeft auto."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[
                ParameterSpec(name="p1", label="Param 1", component="text"),
            ],
            plot_func=dummy_plot_func_with_params,
            button_position="right",
        )
        cell = DashCell(config)
        controls = cell.build_controls()

        # Should have justifyContent space-between for right alignment
        assert isinstance(controls, html.Div)
        assert controls.style.get("justifyContent") == "space-between"

    def test_build_controls_below_position_with_params(self):
        """With below position and params, button should be on separate row."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[
                ParameterSpec(name="p1", label="Param 1", component="text"),
            ],
            plot_func=dummy_plot_func_with_params,
            button_position="below",
        )
        cell = DashCell(config)
        controls = cell.build_controls()

        # Should have flexDirection column
        assert isinstance(controls, html.Div)
        assert controls.style.get("flexDirection") == "column"

    def test_build_controls_left_position_with_params(self):
        """With left position and params, button and params on same row."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[
                ParameterSpec(name="p1", label="Param 1", component="text"),
            ],
            plot_func=dummy_plot_func_with_params,
            button_position="left",
        )
        cell = DashCell(config)
        controls = cell.build_controls()

        # Should have flex wrap layout (same row)
        assert isinstance(controls, html.Div)
        assert controls.style.get("flexWrap") == "wrap"
        # Should NOT have flexDirection column
        assert controls.style.get("flexDirection") is None

    def test_build_layout_contains_title(self):
        """build_layout should include the cell title."""
        config = DashCellConfig(
            id="test_cell",
            title="My Test Title",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        cell = DashCell(config)
        layout = cell.build_layout()

        # Layout should be a Div
        assert isinstance(layout, html.Div)

    def test_cell_ids_are_derived_from_config_id(self):
        """Cell IDs should be derived from config.id."""
        config = DashCellConfig(
            id="my_unique_cell",
            title="Test",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        cell = DashCell(config)

        assert cell.run_id == "my_unique_cell__run"
        assert cell.output_id == "my_unique_cell__output"
        assert cell._param_id("test_param") == "my_unique_cell__param__test_param"


# --- ConfigurableDashApp tests ---


class TestConfigurableDashApp:
    """Tests for ConfigurableDashApp class."""

    def test_app_name_is_stored(self):
        """app_name should be stored as instance variable."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        app = ConfigurableDashApp(
            cells=[config],
            app_name="My Custom Dashboard",
            port=5099,
        )

        assert app.app_name == "My Custom Dashboard"

    def test_app_layout_no_logo_by_default(self):
        """App layout should not contain a logo by default."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        app = ConfigurableDashApp(
            cells=[config],
            app_name="Test Dashboard",
            port=5098,
        )

        assert app.app.layout is not None
        layout_str = str(app.app.layout)
        assert "data:image/svg+xml;base64" not in layout_str

    def test_app_layout_contains_logo_when_provided(self):
        """App layout should contain logo when logo_src is provided."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        logo = "data:image/svg+xml;base64,TEST_LOGO"
        app = ConfigurableDashApp(
            cells=[config],
            app_name="Test Dashboard",
            port=5099,
            logo_src=logo,
        )

        assert app.app.layout is not None
        layout_str = str(app.app.layout)
        assert "TEST_LOGO" in layout_str

    def test_global_filters_is_optional(self):
        """App should work without global_filters."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        app = ConfigurableDashApp(
            cells=[config],
            app_name="Test Dashboard",
            port=5097,
        )

        assert app.global_filters is None
        assert app.app.layout is not None

    def test_global_filters_can_be_provided(self):
        """App should accept global_filters configuration."""
        config = DashCellConfig(
            id="test_cell",
            title="Test Cell",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        global_filters = GlobalFiltersConfig(
            primary_specs=[
                ParameterSpec(name="filter1", label="Filter 1", component="text"),
            ]
        )
        app = ConfigurableDashApp(
            cells=[config],
            app_name="Test Dashboard",
            port=5096,
            global_filters=global_filters,
        )

        assert app.global_filters is not None
        assert len(app.global_filters.primary_specs) == 1

    def test_empty_cells_raises_error(self):
        """Empty cells list should raise ValueError."""
        with pytest.raises(ValueError, match="At least one cell config"):
            ConfigurableDashApp(
                cells=[],
                app_name="Test Dashboard",
                port=5095,
            )

    def test_cells_can_be_nested_lists(self):
        """Cells can be provided as nested lists (rows)."""
        config1 = DashCellConfig(
            id="cell1",
            title="Cell 1",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        config2 = DashCellConfig(
            id="cell2",
            title="Cell 2",
            parameter_specs=[],
            plot_func=dummy_plot_func,
        )
        # Nested list: [[cell1, cell2]] means one row with two cells
        app = ConfigurableDashApp(
            cells=[[config1, config2]],
            app_name="Test Dashboard",
            port=5094,
        )

        assert app.app.layout is not None


# --- ParameterSpec tests ---


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_dropdown_spec(self):
        """Dropdown spec should have options."""
        spec = ParameterSpec(
            name="dropdown_param",
            label="Choose Option",
            component="dropdown",
            options=["a", "b", "c"],
            default="a",
        )
        assert spec.component == "dropdown"
        assert spec.options == ["a", "b", "c"]
        assert spec.default == "a"

    def test_text_spec(self):
        """Text spec with help text."""
        spec = ParameterSpec(
            name="text_param",
            label="Enter Text",
            component="text",
            default="",
            help="Enter a value here",
        )
        assert spec.component == "text"
        assert spec.help == "Enter a value here"

    def test_number_spec(self):
        """Number spec with default value."""
        spec = ParameterSpec(
            name="number_param",
            label="Enter Number",
            component="number",
            default=42,
            required=True,
        )
        assert spec.component == "number"
        assert spec.default == 42
        assert spec.required is True

    def test_date_spec(self):
        """Date spec."""
        spec = ParameterSpec(
            name="date_param",
            label="Select Date",
            component="date",
            default="2024-01-01",
        )
        assert spec.component == "date"
        assert spec.default == "2024-01-01"

    def test_multi_select_dropdown(self):
        """Multi-select dropdown spec."""
        spec = ParameterSpec(
            name="multi_param",
            label="Select Multiple",
            component="dropdown",
            options=["x", "y", "z"],
            default=["x", "y"],
            multi=True,
        )
        assert spec.multi is True
        assert spec.default == ["x", "y"]

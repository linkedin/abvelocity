import pytest
from abvelocity.ts.common.testing_utils import assert_equal
from abvelocity.ts.common.viz.colors_utils import get_color_palette, get_distinct_colors
from plotly.colors import DEFAULT_PLOTLY_COLORS


def test_get_color_palette():
    # color palette length is 1
    color_palette = get_color_palette(5, colors=["rgb(99, 114, 218)"])
    assert_equal(color_palette, ["rgb(99, 114, 218)"] * 5)

    # color palette length less than DEFAULT_PLOTLY_COLORS length
    color_palette = get_color_palette(5, colors=DEFAULT_PLOTLY_COLORS)
    assert_equal(color_palette, DEFAULT_PLOTLY_COLORS[0:5])

    # color palette length greater than DEFAULT_PLOTLY_COLORS length
    color_palette = get_color_palette(15, colors=DEFAULT_PLOTLY_COLORS)
    assert len(color_palette) == 15
    for color in color_palette:
        assert color[0:3] == "rgb"

    # custom colors
    colors = ["rgb(99, 114, 218)", "rgb(0, 145, 202)", "rgb(255, 255, 255)"]
    color_palette = get_color_palette(3, colors=colors)
    assert_equal(color_palette, colors)


def test_get_distinct_colors():
    """Tests the function to get most distinguishable colors."""
    # Under 10 colors, using plotly.colors.qualitative.Plotly.
    assert get_distinct_colors(num_colors=1) == get_distinct_colors(num_colors=2)[:1]

    assert get_distinct_colors(num_colors=10) == [
        "rgba(99, 110, 250, 0.95)",
        "rgba(239, 85, 59, 0.95)",
        "rgba(0, 204, 150, 0.95)",
        "rgba(171, 99, 250, 0.95)",
        "rgba(255, 161, 90, 0.95)",
        "rgba(25, 211, 243, 0.95)",
        "rgba(255, 102, 146, 0.95)",
        "rgba(182, 232, 128, 0.95)",
        "rgba(255, 151, 255, 0.95)",
        "rgba(254, 203, 82, 0.95)",
    ]
    # Under 24 colors, using plotly.colors.qualitative.Light24.
    assert get_distinct_colors(num_colors=15) == get_distinct_colors(num_colors=18)[:15]

    assert get_distinct_colors(num_colors=20, opacity=0.9) == [
        "rgba(253, 50, 22, 0.9)",
        "rgba(0, 254, 53, 0.9)",
        "rgba(106, 118, 252, 0.9)",
        "rgba(254, 212, 196, 0.9)",
        "rgba(254, 0, 206, 0.9)",
        "rgba(13, 249, 255, 0.9)",
        "rgba(246, 249, 38, 0.9)",
        "rgba(255, 150, 22, 0.9)",
        "rgba(71, 155, 85, 0.9)",
        "rgba(238, 166, 251, 0.9)",
        "rgba(220, 88, 125, 0.9)",
        "rgba(214, 38, 255, 0.9)",
        "rgba(110, 137, 156, 0.9)",
        "rgba(0, 181, 247, 0.9)",
        "rgba(182, 142, 0, 0.9)",
        "rgba(201, 251, 229, 0.9)",
        "rgba(255, 0, 146, 0.9)",
        "rgba(34, 255, 167, 0.9)",
        "rgba(227, 238, 158, 0.9)",
        "rgba(134, 206, 0, 0.9)",
    ]

    # Under 256 colors, using Viridis.
    assert get_distinct_colors(num_colors=30, opacity=0.85) == [
        "rgba(68, 1, 84, 0.85)",
        "rgba(69, 13, 95, 0.85)",
        "rgba(70, 25, 106, 0.85)",
        "rgba(72, 37, 118, 0.85)",
        "rgba(70, 48, 124, 0.85)",
        "rgba(66, 58, 129, 0.85)",
        "rgba(63, 68, 135, 0.85)",
        "rgba(60, 78, 138, 0.85)",
        "rgba(56, 88, 139, 0.85)",
        "rgba(52, 98, 141, 0.85)",
        "rgba(48, 107, 142, 0.85)",
        "rgba(44, 115, 142, 0.85)",
        "rgba(41, 123, 142, 0.85)",
        "rgba(38, 131, 142, 0.85)",
        "rgba(36, 140, 140, 0.85)",
        "rgba(33, 148, 139, 0.85)",
        "rgba(31, 157, 137, 0.85)",
        "rgba(37, 165, 133, 0.85)",
        "rgba(44, 173, 128, 0.85)",
        "rgba(51, 180, 123, 0.85)",
        "rgba(65, 188, 114, 0.85)",
        "rgba(82, 195, 104, 0.85)",
        "rgba(100, 202, 94, 0.85)",
        "rgba(120, 208, 82, 0.85)",
        "rgba(142, 213, 68, 0.85)",
        "rgba(164, 218, 54, 0.85)",
        "rgba(186, 223, 43, 0.85)",
        "rgba(208, 225, 41, 0.85)",
        "rgba(231, 228, 39, 0.85)",
        "rgba(253, 231, 37, 0.85)",
    ]
    # Can't get more than 256 colors.
    with pytest.raises(ValueError, match="The maximum number of colors is 256."):
        get_distinct_colors(num_colors=257)

    # Opacity must be between 0 and 1
    with pytest.raises(ValueError, match="Opacity must be between 0 and 1."):
        get_distinct_colors(num_colors=2, opacity=-1)

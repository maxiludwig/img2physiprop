"""Test Visualization Routine."""

from unittest.mock import MagicMock

import pytest
import pyvista as pv
from i2pp.core.image_reader_classes.image_reader import PixelValueType
from i2pp.core.visualization_classes.discretization_visualization import (
    DiscretizationVisualizer,
)


@pytest.fixture
def visualizer():
    """Creates a Visualizer instance with a mocked plotter."""
    vis = DiscretizationVisualizer(
        pixel_type=PixelValueType.RGB,
        pixel_range=[0, 255],
        title="Test Visualization",
    )
    vis.plotter = MagicMock()
    return vis


def test_update_display(visualizer):
    """Test that the display is updated correctly."""
    grid = pv.UnstructuredGrid()

    visualizer._update_display(
        grid, scalars="RGB_values", rgb=True, opacity=0.5
    )

    assert visualizer.current_actor is not None
    visualizer.plotter.add_mesh.assert_called_once_with(
        grid, scalars="RGB_values", rgb=True, opacity=0.5
    )


def test_toggle_slicer_3d(visualizer):
    """Test the slicer toggle in 3D mode."""
    visualizer.show_3d = True
    visualizer._toggle_slicer()

    visualizer.plotter.clear_slider_widgets.assert_called_once()
    visualizer.plotter.add_slider_widget.assert_called_once()


def test_toggle_slicer_2d(visualizer):
    """Test the slicer toggle in 2D mode."""
    visualizer.show_3d = False
    visualizer.grid_visible = MagicMock()
    visualizer.grid_visible.bounds = [0, 0, 0, 0, 0, 10]

    visualizer._toggle_slicer()

    visualizer.plotter.clear_slider_widgets.assert_called_once()
    visualizer.plotter.add_slider_widget.assert_called_once()


def test_update_grid(visualizer):
    """Test that the grid updates based on NaN element visibility."""
    grid = pv.UnstructuredGrid()
    grid.points = [[0, 0, 0]]

    extracted_grid = pv.UnstructuredGrid()
    extracted_grid.points = [[1, 1, 1]]

    visualizer.grid = grid
    visualizer.extracted_grid = extracted_grid

    visualizer._update_grid(True)
    assert visualizer.grid_visible is grid

    visualizer._update_grid(False)
    assert visualizer.grid_visible is extracted_grid


def test_show_3d_model(visualizer):
    """Test displaying the 3D model."""
    grid = pv.UnstructuredGrid()
    visualizer.grid_visible = grid
    visualizer._show_3d_model()

    visualizer.plotter.render.assert_called_once()


def test_slice_at_height(visualizer):
    """Test slicing at a given height."""
    visualizer.grid_visible = MagicMock()
    visualizer.grid_visible.slice.return_value = pv.UnstructuredGrid()

    visualizer.slice_height = 5
    visualizer._slice_at_height()

    visualizer.grid_visible.slice.assert_called_once_with(
        normal=[0, 0, 1], origin=(0, 0, 5)
    )
    visualizer.plotter.render.assert_called_once()


def test_toggle_nan_view(visualizer):
    """Test toggling NaN element visibility."""
    visualizer._toggle_nan_view(True)
    assert visualizer.show_3d


def test_toggle_view_to_3d(visualizer):
    """Test switching to 3D mode."""

    visualizer._show_3d_model = MagicMock()
    visualizer._slice_at_height = MagicMock()
    visualizer._toggle_slicer = MagicMock()

    visualizer._toggle_view(is_checked=True)

    visualizer._show_3d_model.assert_called_once()
    visualizer._slice_at_height.assert_not_called()
    visualizer._toggle_slicer.assert_called_once()


def test_toggle_view_to_2d(visualizer):
    """Test switching to 2D slice mode."""

    visualizer._show_3d_model = MagicMock()
    visualizer._slice_at_height = MagicMock()
    visualizer._toggle_slicer = MagicMock()

    visualizer._toggle_view(is_checked=False)

    visualizer._slice_at_height.assert_called_once()
    visualizer._show_3d_model.assert_not_called()
    visualizer._toggle_slicer.assert_called_once()


def test_update_opacity(visualizer):
    """Test updating opacity."""
    visualizer._update_opacity(0.7)
    assert visualizer.opacity == 0.7


def test_update_slice_height(visualizer):
    """Test updating the slice height."""
    visualizer._update_slice_height(3.5)
    assert visualizer.slice_height == 3.5


def test_plot_grid_without_nan(visualizer):
    """Test `plot_grid` without NaN elements."""

    visualizer.plotter.add_text = MagicMock()
    visualizer.plotter.add_checkbox_button_widget = MagicMock()
    visualizer.plotter.show = MagicMock()
    visualizer._toggle_slicer = MagicMock()
    visualizer._show_3d_model = MagicMock()

    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    cells = [8, 0, 1, 3, 2, 4, 5, 6, 7]
    cell_types = [12]

    visualizer.grid = pv.UnstructuredGrid(cells, cell_types, points)

    visualizer.nan_exist = False

    visualizer.plot_grid()

    assert visualizer.slice_height == 0.5
    assert visualizer.plotter.window_size == [1024, 768]
    assert visualizer.plotter.title == "Test Visualization"

    visualizer._toggle_slicer.assert_called_once()
    visualizer._show_3d_model.assert_called_once()
    assert visualizer.plotter.add_text.call_count == 2
    visualizer.plotter.add_checkbox_button_widget.assert_any_call(
        visualizer._toggle_view,
        value=True,
        position=(1, 650),
        size=30,
    )

    visualizer.plotter.add_checkbox_button_widget.assert_any_call(
        visualizer._toggle_edges, size=30, value=False, position=(1, 600)
    )

    visualizer.plotter.show.assert_called_once()


def test_plot_grid_with_nan(visualizer):
    """Test `plot_grid` with NaN elements."""

    visualizer.plotter.add_text = MagicMock()
    visualizer.plotter.add_checkbox_button_widget = MagicMock()
    visualizer.plotter.show = MagicMock()
    visualizer._toggle_slicer = MagicMock()
    visualizer._show_3d_model = MagicMock()

    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    cells = [8, 0, 1, 3, 2, 4, 5, 6, 7]
    cell_types = [12]

    visualizer.grid = pv.UnstructuredGrid(cells, cell_types, points)

    visualizer.nan_exist = True

    visualizer.plot_grid()

    visualizer.plotter.add_text.assert_any_call(
        "Toggle Nan-Elements:",
        position="upper_left",
        font_size=8,
        color="black",
    )

    visualizer.plotter.add_text.assert_any_call(
        "Toggle Edges", position=(35, 605), font_size=8, color="black"
    )
    visualizer.plotter.add_text.assert_any_call(
        "Toggle 3D/Slice View", position=(35, 655), font_size=8, color="black"
    )

    visualizer.plotter.add_checkbox_button_widget.assert_any_call(
        visualizer._toggle_nan_view, size=30, value=True, position=(210, 740)
    )
    visualizer.plotter.add_checkbox_button_widget.assert_any_call(
        visualizer._toggle_view,
        value=True,
        position=(1, 650),
        size=30,
    )
    visualizer.plotter.show.assert_called_once()

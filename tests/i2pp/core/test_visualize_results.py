"""Test visualize_results."""

from unittest.mock import MagicMock, patch

import pytest
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Element,
)
from i2pp.core.image_reader_classes.image_reader import (
    ImageData,
    PixelValueType,
)
from i2pp.core.visualization_classes.discretization_visualization import (
    DiscretizationVisualizer,
)
from i2pp.core.visualization_classes.image_visualization import ImageVisualizer
from i2pp.core.visualize_results import visualize_results, visualize_smoothing


@pytest.fixture
def mock_image_data():
    """Creates a mock ImageData object."""
    return ImageData(
        grid_coords=None,
        pixel_type=PixelValueType.RGB,
        pixel_data=None,
        orientation=None,
        position=None,
    )


@pytest.fixture
def mock_elements():
    """Creates a list of mock Elements."""
    return [MagicMock(spec=Element), MagicMock(spec=Element)]


@patch("i2pp.core.visualize_results.Process")
@patch.object(ImageVisualizer, "plot_grid", MagicMock())
@patch.object(DiscretizationVisualizer, "plot_grid", MagicMock())
@patch.object(DiscretizationVisualizer, "compute_grid", MagicMock())
@patch.object(ImageVisualizer, "compute_grid", MagicMock())
def test_visualize_results(mock_process, mock_image_data, mock_elements):
    """Test that visualize_results starts two processes without actually
    running them."""

    visualize_results({}, mock_elements, mock_image_data)

    assert mock_process.call_count == 2

    process_instance = mock_process.return_value
    process_instance.start.assert_called()
    process_instance.join.assert_called()


@patch("i2pp.core.visualize_results.Process")
@patch.object(ImageVisualizer, "plot_grid", MagicMock())
@patch.object(ImageVisualizer, "compute_grid", MagicMock())
def test_visualize_smoothing(mock_process, mock_image_data):
    """Test that visualize_smoothing starts two processes without actually
    running them."""

    visualize_smoothing(mock_image_data, mock_image_data)

    assert mock_process.call_count == 2

    process_instance = mock_process.return_value
    process_instance.start.assert_called()
    process_instance.join.assert_called()

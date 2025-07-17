"""Test Visualization Routine."""

from unittest.mock import MagicMock, patch

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Element,
)
from i2pp.core.image_reader_classes.image_reader import PixelValueType
from i2pp.core.visualization_classes.discretization_visualization import (
    DiscretizationVisualizer,
)


def test_create_vtk_from_unfiltered_discretization():
    """Test create_vtk_from_unfiltered_discretization."""
    element1 = Element([0, 1, 2, 3], 0, data=10)
    element2 = Element([0, 1, 2, 3], 1, data=20)
    element3 = Element([0, 1, 2, 3], 2, data=np.nan)

    test_config = {"processing options": {}}

    mock_unstructured_grid = MagicMock()
    mock_unstructured_grid.extract_cells = MagicMock()

    visualizer = DiscretizationVisualizer(PixelValueType.CT, [])

    with patch(
        "i2pp.core.visualization_classes.discretization_visualization."
        "initialize_unstructured_grid",
        return_value=(mock_unstructured_grid, np.array([True, True, False])),
    ) as mock_initialize_unstructured_grid:
        dis_mock = MagicMock()
        visualizer.compute_grid(
            test_config, [element1, element2, element3], dis=dis_mock
        )

        mock_initialize_unstructured_grid.assert_called_once_with(
            [element1, element2, element3], PixelValueType.CT, dis_mock
        )
        assert visualizer.nan_exist is True
        assert np.all(np.equal(visualizer.ele_has_value, [True, True, False]))
        assert visualizer.grid is mock_unstructured_grid
        assert (
            visualizer.extracted_grid
            is mock_unstructured_grid.extract_cells.return_value
        )
        assert visualizer.grid_visible is mock_unstructured_grid

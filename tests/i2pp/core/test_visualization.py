"""Test Visualization Routine."""

from unittest.mock import patch

import numpy as np
import vtk
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
    Nodes,
)
from i2pp.core.image_reader_classes.image_reader import PixelValueType
from i2pp.core.visualization import create_vtk_from_unfiltered_discretizazion


def test_create_vtk_from_unfiltered_discretizazion():
    """Test create_vtk_from_unfiltered_discretizazion."""

    ids = [0, 1, 2, 3]
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    nodes = Nodes(coords=coords, ids=ids)

    element1 = Element([0, 1, 2, 3], 0, data=10)
    element2 = Element([0, 1, 2, 3], 1, data=20)

    test_config = {"processing options": {}}

    pixel_range = np.array([0, 20])

    mock_elements = [element1, element2]
    mock_dis = Discretization(nodes=nodes, elements=mock_elements)

    with patch(
        "i2pp.core.visualization.verify_and_load_discretization",
        return_value=mock_dis,
    ) as mock_verify_and_load_discretization:
        grid = create_vtk_from_unfiltered_discretizazion(
            test_config, mock_elements, PixelValueType.CT, pixel_range
        )

        assert isinstance(grid, vtk.vtkUnstructuredGrid)
        assert grid.GetNumberOfCells() == 2
        assert grid.GetNumberOfPoints() == 4
        assert grid.GetCellData().GetScalars().GetNumberOfTuples() == 2
        mock_verify_and_load_discretization.assert_called_once_with(
            test_config
        )

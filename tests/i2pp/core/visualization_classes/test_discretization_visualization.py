"""Test Visualization Routine."""

from unittest.mock import patch

import numpy as np
import pyvista as pv
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
    Nodes,
)
from i2pp.core.image_reader_classes.image_reader import PixelValueType
from i2pp.core.visualization_classes.discretization_visualization import (
    DiscretizationVisualizer,
)


def test_create_vtk_from_unfiltered_discretization():
    """Test create_vtk_from_unfiltered_discretization."""

    ids = [0, 1, 2, 3]
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    nodes = Nodes(coords=coords, ids=ids)

    element1 = Element([0, 1, 2, 3], 0, data=10)
    element2 = Element([0, 1, 2, 3], 1, data=20)
    element3 = Element([0, 1, 2, 3], 2, data=20)

    test_config = {"processing options": {}}

    mock_elements = [element1, element2, element3]
    mock_dis = Discretization(nodes=nodes, elements=mock_elements)
    visualizer = DiscretizationVisualizer(PixelValueType.CT, [])

    str1 = "i2pp.core.visualization_classes"
    str2 = ".discretization_visualization.verify_and_load_discretization"

    path = str1 + str2
    with patch(
        path,
        return_value=mock_dis,
    ) as mock_verify_and_load_discretization:
        visualizer.compute_grid(test_config, [element1, element2], [0, 2000])

        assert isinstance(visualizer.grid, pv.UnstructuredGrid)
        assert visualizer.grid.GetNumberOfCells() == 3
        assert visualizer.grid.GetNumberOfPoints() == 4
        assert (
            visualizer.grid.GetCellData().GetScalars().GetNumberOfTuples() == 3
        )
        mock_verify_and_load_discretization.assert_called_once_with(
            test_config
        )

        assert np.all(np.equal(visualizer.ele_has_value, [True, True, False]))
        assert np.all(
            np.equal(
                visualizer.grid.GetCellData().GetScalars(), [10, 20, 2000]
            )
        )

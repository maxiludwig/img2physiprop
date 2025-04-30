"""Test Interpolator Routine."""

from unittest.mock import patch

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
    Nodes,
)
from i2pp.core.image_reader_classes.image_reader import (
    GridCoords,
    ImageData,
    PixelValueType,
)
from i2pp.core.interpolator_classes.interpolator_nodes import InterpolatorNodes


def test_compute_element_data():
    """Test compute_element_data for RGB."""

    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    element1 = Element(node_ids=[0, 1], id=0)
    element2 = Element(node_ids=[1, 0], id=1)
    elements = [element1, element2]

    nodes = Nodes(coords=node_coords, ids=[0, 1, 2, 3])
    dis = Discretization(nodes=nodes, elements=elements)

    pixel_data = np.random.randint(0, 256, size=(4, 4, 4, 3))

    slice_coords = np.arange(4)
    row_coords = np.arange(4)
    col_coords = np.arange(4)
    grid_coords = GridCoords(slice_coords, row_coords, col_coords)

    image_data = ImageData(
        pixel_data,
        grid_coords,
        orientation=np.eye(3),
        position=np.array([0, 0, 0]),
        pixel_type=PixelValueType.RGB,
    )

    interpolator = InterpolatorNodes()

    return_grid_coords = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    path = "i2pp.core.interpolator_classes.interpolator_nodes."

    function_name = "get_node_position_of_element"

    path_string = path + function_name

    with patch.object(
        InterpolatorNodes,
        "world_to_grid_coords",
        return_value=return_grid_coords,
    ) as mock_world_to_grid_coords:
        with patch.object(
            InterpolatorNodes,
            "interpolate_image_values_to_points",
            return_value=np.array([[100, 150, 200], [120, 180, 240]]),
        ) as mock_interpol_image_values_to_points:
            with patch(
                path_string,
                return_value=np.array([0, 1]),
            ) as mock_get_node_position_of_element:
                result = interpolator.compute_element_data(dis, image_data)
                expected_value = np.mean(
                    [[100, 150, 200], [120, 180, 240]], axis=0
                )

                assert np.array_equal(result[0].data, expected_value)
                assert np.array_equal(result[1].data, expected_value)

                mock_world_to_grid_coords.assert_called_once_with(
                    node_coords, image_data.orientation, image_data.position
                )
                mock_interpol_image_values_to_points.assert_called_once_with(
                    return_grid_coords, image_data
                )
                mock_get_node_position_of_element.call_count == 2

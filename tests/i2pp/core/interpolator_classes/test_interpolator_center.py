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
from i2pp.core.interpolator_classes.interpolator_center import (
    InterpolatorCenter,
)


def test_get_center():
    """Test get_center if center of elements are calculated correctly."""
    ele1_ids = np.array([0, 2, 4], dtype=int)
    ele2_ids = np.array([1, 3, 5], dtype=int)

    elements = []
    elements.append(Element(ele1_ids, 1, [], []))
    elements.append(Element(ele2_ids, 2, [], []))
    nodes_coords = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 3, 6], [6, 2, 0], [9, 3, 6]]
    )
    node_ids = np.array([0, 1, 2, 3, 4, 5], dtype=int)

    nodes = Nodes(nodes_coords, node_ids)

    test_dis = Discretization(nodes=nodes, elements=elements)

    interpol = InterpolatorCenter()

    interpol = interpol.compute_element_centers(test_dis)

    expected_output1 = np.array([2, 1, 0])
    expected_output2 = np.array([4, 2, 4])

    assert np.array_equal(test_dis.elements[0].center_coords, expected_output1)
    assert np.array_equal(test_dis.elements[1].center_coords, expected_output2)


def test_compute_element_data():
    """Test compute_element_data for RGB."""

    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    element1 = Element(node_ids=[0, 1], id=0, center_coords=[0, 0, 0])
    element2 = Element(node_ids=[1, 0], id=1, center_coords=[1, 1, 1])
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

    interpolator = InterpolatorCenter()

    return_grid_coords = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])

    with patch.object(
        InterpolatorCenter, "compute_element_centers", return_value=dis
    ) as mock_compute_element_centers:
        with patch.object(
            InterpolatorCenter,
            "world_to_grid_coords",
            return_value=return_grid_coords,
        ) as mock_world_to_grid_coords:
            with patch.object(
                InterpolatorCenter,
                "interpolate_image_values_to_points",
                return_value=np.array([[100, 150, 200], [120, 180, 240]]),
            ) as mock_interpol_image_values_to_points:

                result = interpolator.compute_element_data(dis, image_data)

                assert np.array_equal(
                    result[0].data, np.array([100, 150, 200])
                )

                mock_compute_element_centers.assert_called_once_with(dis)
                mock_world_to_grid_coords.assert_called_once()
                mock_interpol_image_values_to_points.assert_called_once_with(
                    return_grid_coords, image_data
                )

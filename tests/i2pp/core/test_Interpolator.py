"""Test Interpolator Routine."""

from unittest.mock import patch

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
    Nodes,
)
from i2pp.core.image_data_converter import ProcessedImageData
from i2pp.core.interpolator import Interpolator


def test_interpolate_imagevalues_to_points():
    """Test interpolate_imagevalues_to_points if interpolation is done
    correctly."""

    target_points = np.array([[0, 0], [1, 0], [1, 2]])

    points_image = np.array([[0, 0], [1, 2], [2, 0]])
    values_image = np.array([5, 3, 0])

    image_data = ProcessedImageData(points_image, values_image)

    test_inperpolation = Interpolator()
    interpol_value = test_inperpolation.interpolate_image_values_to_points(
        target_points, image_data
    )
    expected_output = np.array([5, 2.5, 3])
    assert np.array_equal(interpol_value, expected_output)


def test_get_elementvalues_nodes():
    """Test get_value_of_elements if the calculation of element values is
    correct."""

    element1 = Element(np.array([0, 2]), 0, [], [])
    element2 = Element(np.array([3, 1]), 0, [], [])

    node_coords = np.array([[0, 0, 0], [1, 0, 10], [1, 0, 0], [1, 1, 10]])
    node_ids = np.array([0, 1, 2, 3])
    nodes = Nodes(node_coords, node_ids)
    dis = Discretization(nodes, [element1, element2])

    coords_image = np.array(
        [[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [1, 0, 10], [1, 1, 10]]
    )

    values_image = np.array([10, 10, 30, 50, 150])
    image_data = ProcessedImageData(coords_image, values_image)

    interpolator = Interpolator()
    elements = interpolator.get_elementvalues_nodes(dis, image_data)

    assert elements[0].value == 15
    assert elements[1].value == 100


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

    interpol = Interpolator()

    interpol = interpol.compute_element_centers(test_dis)

    expected_output1 = np.array([2, 1, 0])
    expected_output2 = np.array([4, 2, 4])

    assert np.array_equal(test_dis.elements[0].center_coords, expected_output1)
    assert np.array_equal(test_dis.elements[1].center_coords, expected_output2)


def test_get_elementvalues_center():
    """Test get_value_of_elements_center."""

    element1 = Element([], 0, [3, 1, 1], [])
    element2 = Element([], 1, [1, 1, 1], [])
    element3 = Element([], 2, [2, 0, 0], [])
    element4 = Element([], 3, [4, 3, 1], [])

    dis = Discretization([], [element1, element2, element3, element4], [])

    coord1 = np.array([1, 1, 1])
    coord2 = np.array([2, 0, 0])
    coord3 = np.array([4, 3, 1])
    coord4 = np.array([3, 1, 1])

    pxl_value = np.array([1, 2, 3, 4])

    coord_array = np.array([coord1, coord2, coord3, coord4])
    image_data = ProcessedImageData(coord_array, pxl_value)

    interpolator = Interpolator()
    with patch.object(
        Interpolator, "compute_element_centers", return_value=dis
    ):
        elements = interpolator.get_elementvalues_center(dis, image_data)

        assert elements[0].value, 4
        assert elements[1].value, 1
        assert elements[2].value, 2
        assert elements[3].value, 3


def test_get_elementvalues_all_voxels():
    """Test get_value_of_elements_all_Voxels."""
    node0 = [0, 0, 0]
    node1 = [1, 0, 0]
    node2 = [0, 1, 0]
    node3 = [1, 1, 0]

    node4 = [0, 0, 1]
    node5 = [1, 0, 1]
    node6 = [0, 1, 1]
    node7 = [1, 1, 1]

    nodes_coords = np.array(
        [node0, node1, node2, node3, node4, node5, node6, node7]
    )

    nodes_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    nodes = Nodes(nodes_coords, nodes_ids)
    element1 = Element(np.array([0, 1, 2, 3, 4, 6]), 0, [], [])
    element2 = Element(np.array([4, 5, 6, 7, 1, 3]), 1, [], [])

    dis = Discretization(nodes, [element1, element2], [])

    coord1 = np.array([-1, 2, 1])
    coord2 = np.array([0.5, 0.5, 0.5])
    coord3 = np.array([0.25, 0.5, 0.25])
    coord4 = np.array([1, 0, 1])
    coord5 = np.array([1.5, 0.5, 0.5])

    pxl_value = np.array([1, 2, 3, 4, 5])

    coord_array = np.array([coord1, coord2, coord3, coord4, coord5])
    image_data = ProcessedImageData(coord_array, pxl_value)

    interpolator = Interpolator()

    elements = interpolator.get_elementvalues_all_voxels(dis, image_data)

    assert elements[0].value, 2.5
    assert elements[1].value, 3

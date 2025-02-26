"""Test Interpolator Routine."""

import numpy as np
from i2pp.core.image_data_converter import ProcessedImageData
from i2pp.core.interpolator import InterpolatorClass
from i2pp.core.model_reader_classes.model_reader import (
    Element,
    ModelData,
    Nodes,
)


def test_ImageValues_2_MeshCoords():
    """Test ImageValues_2_MeshCoords if interpolation is done correctly."""

    target_points = np.array([[0, 0], [1, 0], [1, 2]])

    points_image = np.array([[0, 0], [1, 2], [2, 0]])
    values_image = np.array([5, 3, 0])

    image_data = ProcessedImageData(points_image, values_image)

    mesh_data = ModelData(nodes=[], elements=[], limits=[])

    test_inperpolation = InterpolatorClass(image_data, mesh_data)
    interpol_value = test_inperpolation.interpolate_imagevalues_to_points(
        target_points
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
    model_data = ModelData(nodes, [element1, element2], [])

    coords_image = np.array(
        [[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [1, 0, 10], [1, 1, 10]]
    )

    values_image = np.array([10, 10, 30, 50, 150])
    image_data = ProcessedImageData(coords_image, values_image)

    interpolator = InterpolatorClass(image_data, model_data)
    elements = interpolator.get_elementvalues_nodes()

    assert elements[0].value == 15
    assert elements[1].value == 100


def test_get_elementvalues_center():
    """Test get_value_of_elements_center."""

    element1 = Element([], 0, [3, 1, 1], [])
    element2 = Element([], 1, [1, 1, 1], [])
    element3 = Element([], 2, [2, 0, 0], [])
    element4 = Element([], 3, [4, 3, 1], [])

    mesh_data = ModelData([], [element1, element2, element3, element4], [])

    coord1 = np.array([1, 1, 1])
    coord2 = np.array([2, 0, 0])
    coord3 = np.array([4, 3, 1])
    coord4 = np.array([3, 1, 1])

    pxl_value = np.array([1, 2, 3, 4])

    coord_array = np.array([coord1, coord2, coord3, coord4])
    image_data = ProcessedImageData(coord_array, pxl_value)

    interpolator = InterpolatorClass(image_data, mesh_data)
    elements = interpolator.get_elementvalues_center()

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

    mesh_data = ModelData(nodes, [element1, element2], [])

    coord1 = np.array([-1, 2, 1])
    coord2 = np.array([0.5, 0.5, 0.5])
    coord3 = np.array([0.25, 0.5, 0.25])
    coord4 = np.array([1, 0, 1])
    coord5 = np.array([1.5, 0.5, 0.5])

    pxl_value = np.array([1, 2, 3, 4, 5])

    coord_array = np.array([coord1, coord2, coord3, coord4, coord5])
    image_data = ProcessedImageData(coord_array, pxl_value)

    interpolator = InterpolatorClass(image_data, mesh_data)

    elements = interpolator.get_elementvalues_all_voxels()

    assert elements[0].value, 2.5
    assert elements[1].value, 3

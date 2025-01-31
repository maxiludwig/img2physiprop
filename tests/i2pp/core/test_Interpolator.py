"""Test Interpolator Routine."""

from unittest.mock import MagicMock

import numpy as np
from i2pp.core.image_data_converter import ProcessedImageData
from i2pp.core.interpolator import InterpolatorClass
from i2pp.core.model_reader_classes.model_reader import ModelData


def test_ImageValues_2_MeshCoords():
    """Test ImageValues_2_MeshCoords if interpolation is done correctly."""

    points_mesh = np.array([[0, 0], [1, 0], [1, 2]])

    points_image = np.array([[0, 0], [1, 2], [2, 0]])
    values_image = np.array([5, 3, 0])

    image_data = ProcessedImageData(points_image, values_image)

    mesh_data = ModelData(
        nodes=points_mesh, element_ids=[], element_center=[], limits=[]
    )

    test_inperpolation = InterpolatorClass(image_data, mesh_data)
    inperpol_value = test_inperpolation.interpolate_imagevalues_to_points(
        mesh_data.nodes
    )

    expected_output = np.array([5, 2.5, 3])
    assert np.array_equal(inperpol_value, expected_output)


def test_get_value_of_elements_nodes():
    """Test get_value_of_elements if the calculation of element values is
    correct."""

    mesh_data_mock = MagicMock()

    mock_element_ids = [[0, 2], [3, 1]]

    mesh_data_mock.element_ids = mock_element_ids

    values = np.array([10, 20, 200, 80])

    interpolator = InterpolatorClass(None, mesh_data_mock)

    expected_output = np.array([105, 50])

    assert np.array_equal(
        interpolator.get_value_of_elements_nodes(values), expected_output
    )


def test_get_value_of_elements_all_Voxels():
    """Test get_value_of_elements_all_Voxels"""
    node0 = [0, 0, 0]
    node1 = [1, 0, 0]
    node2 = [0, 1, 0]
    node3 = [1, 1, 0]

    node4 = [0, 0, 1]
    node5 = [1, 0, 1]
    node6 = [0, 1, 1]
    node7 = [1, 1, 1]

    nodes = np.array([node0, node1, node2, node3, node4, node5, node6, node7])
    element_ids = np.array([[0, 1, 2, 3, 4, 6], [4, 5, 6, 7, 1, 3]])
    mesh_data = ModelData(nodes, element_ids, [], [])

    coord1 = [-1, 2, 1]
    coord2 = [0.5, 0.5, 0.5]
    coord3 = [0.25, 0.5, 0.25]
    coord4 = [1, 0, 1]
    coord5 = [1.5, 0.5, 0.5]

    pxl_value = np.array([1, 2, 3, 4, 5])

    coord_array = np.array([coord1, coord2, coord3, coord4, coord5])
    image_data = ProcessedImageData(coord_array, pxl_value)

    interpolator = InterpolatorClass(image_data, mesh_data)

    expected_output = np.array([2.5, 3])

    assert np.array_equal(
        interpolator.get_value_of_elements_all_Voxels(), expected_output
    )

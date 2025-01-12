"""Test Interpolator Routine."""

from unittest.mock import MagicMock

import numpy as np
from i2pp.core.Image_Data_Converter import ImageData
from i2pp.core.Interpolator import InterpolatorClass, InterpolData
from i2pp.core.Mesh_Reader import MeshData


def test_ImageValues_2_MeshCoords():
    """Test ImageValues_2_MeshCoords if interpolation is done correctly."""

    points_mesh = np.array([[0, 0], [1, 0], [1, 2]])

    points_image = np.array([[0, 0], [1, 2], [2, 0]])
    values_image = np.array([5, 3, 0])

    image_data = ImageData(points_image, values_image, "")

    mesh_data = MeshData(nodes=points_mesh, element_ids=[], limits=[])
    interpol_data = InterpolData(
        points=[], points_value=[], element_ids=[], element_value=[]
    )

    test_inperpolation = InterpolatorClass(
        image_data, mesh_data, interpol_data
    )
    inperpol_value = test_inperpolation.imagevalues_2_meshcoords()

    expected_output = np.array([5, 2.5, 3])
    assert np.array_equal(inperpol_value, expected_output)


def test_get_value_of_elements():
    """Test get_value_of_elements if the calculation of element values is
    correct."""

    mesh_data_mock = MagicMock()

    mock_element_ids = [[0, 2], [3, 1]]

    mesh_data_mock.element_ids = mock_element_ids

    values = [10, 20, 200, 80]

    interpol_data = InterpolData(
        points=[], points_value=values, element_ids=[], element_value=[]
    )

    interpolator = InterpolatorClass(None, mesh_data_mock, interpol_data)

    expected_output = np.array([105, 50])

    assert np.array_equal(
        interpolator.get_value_of_elements(), expected_output
    )

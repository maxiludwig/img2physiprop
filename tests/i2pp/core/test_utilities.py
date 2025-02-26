"""Test Utilities Routine."""

import numpy as np
from i2pp.core.model_reader_classes.model_reader import Limits
from i2pp.core.utilities import (
    find_mins_maxs,
    get_node_position_of_element,
    normalize_values,
)


def test_find_mins_maxs():
    """Test find_mins_maxs."""

    points = np.array([[0, 4, 1], [4, 9, 100], [-200, 39, 1], [20, 50, -93]])

    expected_output = Limits(
        max=np.array([22, 52, 102]), min=np.array([-202, 2, -95])
    )
    limit = find_mins_maxs(points)

    assert np.array_equal(limit[0], expected_output.min)
    assert np.array_equal(limit[1], expected_output.max)


def test_norm_values_RGB():
    """Test normalize_values for RGB."""
    data = np.array([[0, 255, 255], [255, 255, 0], [0, 255, 0]])
    pxl_range = np.array([0, 255])

    assert np.array_equal(
        normalize_values(data, pxl_range),
        np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),
    )


def test_norm_values_Gray():
    """Test normalize_values for Float-values."""
    pxl_range = np.array([-100, 100])
    data = np.array([100, 50, 0, -50, -100])

    assert np.array_equal(
        normalize_values(data, pxl_range),
        np.array([1.0, 0.75, 0.5, 0.25, 0.0]),
    )


def test_get_element_coordinates():
    """Test get_element_coordinates."""

    element_node_ids = np.array([5, 2, 4])
    node_ids = np.array([4, 8, 2, 9, 5])

    assert np.array_equal(
        get_node_position_of_element(element_node_ids, node_ids),
        np.array([4, 2, 0]),
    )

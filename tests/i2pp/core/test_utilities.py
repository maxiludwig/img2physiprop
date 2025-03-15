"""Test Utilities Routine."""

import numpy as np
from i2pp.core.discretization_reader_classes.discretization_reader import (
    BoundingBox,
)
from i2pp.core.utilities import (
    find_mins_maxs,
    get_node_position_of_element,
    normalize_values,
    smooth_data,
)


def test_find_mins_maxs_enlargement_0():
    """Test find_mins_maxs."""

    points = np.array([[0, 4, 1], [4, 9, 100], [-200, 39, 1], [20, 50, -93]])

    expected_output = BoundingBox(
        max=np.array([20, 50, 100]), min=np.array([-200, 4, -93])
    )
    limit = find_mins_maxs(points, 0)

    assert np.array_equal(limit[0], expected_output.min)
    assert np.array_equal(limit[1], expected_output.max)


def test_find_mins_maxs_enlargement_2():
    """Test find_mins_maxs."""

    points = np.array([[0, 4, 1], [4, 9, 100], [-200, 39, 1], [20, 50, -93]])

    expected_output = BoundingBox(
        max=np.array([22, 52, 102]), min=np.array([-202, 2, -95])
    )
    limit = find_mins_maxs(points, 2)

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

    element_node_ids = np.array([5, 2, 4, 10])
    node_ids = np.array([0, 2, 4, 5, 8, 10])

    assert np.array_equal(
        get_node_position_of_element(element_node_ids, node_ids),
        np.array([3, 1, 2, 5]),
    )


def test_smoothing_dicom():
    """Test smooth_data if pxl_values are floats."""
    array_slice1 = np.array(
        [[5, 4, 3], [0, 5, 7], [9, 2, 1]], dtype=np.float32
    )

    array_slice2 = np.array(
        [[6, 6, 6], [10, 4, 4], [7, 7, 4]], dtype=np.float32
    )

    array_slice3 = np.array(
        [[10, 5, 0], [2, 8, 5], [3, 3, 9]], dtype=np.float32
    )
    pixel_data = np.array([array_slice1, array_slice2, array_slice3])

    pixel_data_smoothed = smooth_data(pixel_data, 3)

    assert pixel_data_smoothed[1][1][1] == 5


def test_smoothing_rgb():
    """Test smooth_data if pxl_values are arrays (e.g. RGB)"""

    array_slice1 = np.array(
        [
            [[0, 2, 1], [0, 6, 3], [3, 1, 2]],
            [[6, 2, 1], [9, 0, 0], [3, 3, 1]],
            [[3, 2, 1], [0, 0, 0], [3, 2, 0]],
        ],
        dtype=np.float32,
    )

    array_slice2 = np.array(
        [
            [[7, 5, 12], [5, 15, 4], [6, 7, 5]],
            [[8, 5, 0], [5, 0, 4], [6, 8, 5]],
            [[0, 5, 0], [5, 0, 4], [3, 0, 2]],
        ],
        dtype=np.float32,
    )

    array_slice3 = np.array(
        [
            [[1, 2, 0], [0, 5, 1], [3, 6, 0]],
            [[1, 2, 0], [2, 1, 1], [0, 0, 2]],
            [[1, 2, 3], [1, 0, 1], [0, 0, 1]],
        ],
        dtype=np.float32,
    )

    pixel_data = np.array([array_slice1, array_slice2, array_slice3])

    pixel_data_smoothed = smooth_data(pixel_data, 3)

    assert np.array_equal(pixel_data_smoothed[1][1][1], np.array([3, 3, 2]))

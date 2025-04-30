"""Test Interpolator Routine."""

import numpy as np
from i2pp.core.image_reader_classes.image_reader import GridCoords
from i2pp.core.interpolator_classes.interpolator_all_voxel import (
    InterpolatorAllVoxel,
)
from scipy.spatial import ConvexHull


def test__search_bounding_box_all_inside():
    """Test _search_bounding_box if all points are inside the grid."""
    slice_coords = np.arange(20) * 0.5
    row_coords = np.arange(20) * 1
    col_coords = np.arange(20) * 2

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)

    ele_grid_coords = np.array(
        [[3, 12, 22], [0.9, 4, 3], [5, 19, 16], [7, 9, 11]]
    )

    interpolator = InterpolatorAllVoxel()
    i_slice, i_row, i_col = interpolator._search_bounding_box(
        grid_coords, ele_grid_coords
    )

    print(slice_coords[i_slice])
    assert [min(i_slice), max(i_slice)] == [2, 14]
    assert [min(i_row), max(i_row)] == [4, 19]
    assert [min(i_col), max(i_col)] == [2, 11]


def test__search_bounding_box_element_outside():
    """Test _search_bounding_box if part of the element is outside of the
    grid."""
    slice_coords = np.arange(10) * 1
    row_coords = np.arange(10) * 1
    col_coords = np.arange(10) * 1

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)

    ele_grid_coords = np.array(
        [[-1, -1, 0], [1, 4, 3], [5, -5, 16], [7, 8, -7]]
    )

    interpolator = InterpolatorAllVoxel()
    i_slice, i_row, i_col = interpolator._search_bounding_box(
        grid_coords, ele_grid_coords
    )

    assert [min(i_slice), max(i_slice)] == [0, 7]
    assert [min(i_row), max(i_row)] == [0, 8]
    assert [min(i_col), max(i_col)] == [0, 9]


def test__is_inside_element_in_element():
    """Test _is_inside_element_in_element for a tetraeder if the point is
    inside."""

    element = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])

    point = np.array([0.5, 0.5, 0.5])

    hull = ConvexHull(element)
    interpolator = InterpolatorAllVoxel()
    assert interpolator._is_inside_element(point, hull)


def test__is_inside_element_on_element():
    """Test _is_inside_element_in_element for a tetraeder if the point is on
    the surface."""

    element = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])

    point = np.array([0.5, 0.5, 0])

    hull = ConvexHull(element)
    interpolator = InterpolatorAllVoxel()
    assert interpolator._is_inside_element(point, hull)


def test__is_inside_element_outside():
    """Test _is_inside_element_in_element for a tetraeder if the point is not
    in element."""

    element = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])

    point = np.array([0.5, 0.5, -0.01])

    hull = ConvexHull(element)
    interpolator = InterpolatorAllVoxel()
    assert interpolator._is_inside_element(point, hull) is not True


def test_get_data_of_element_element_in_grid_scalar():
    """get_data_of_element if the element is in grid for scalar data."""
    element = np.array(
        [
            [0, 2, 1],
            [0, 3, 1],
            [1, 3, 1],
            [1, 2, 1],
            [0, 2, 2],
            [0, 3, 2],
            [1, 3, 2],
            [1, 2, 2],
        ]
    )

    N_slice, N_row, N_col = 5, 5, 5
    pixel_data = (
        np.arange(N_slice * N_row * N_col)
        .reshape((N_slice, N_row, N_col))
        .astype(float)
    )

    slice_coords = np.arange(5)
    row_coords = np.arange(5)
    col_coords = np.arange(5)

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)

    interpolator = InterpolatorAllVoxel()

    data_inside = [11, 12, 16, 17, 36, 37, 41, 42]
    assert interpolator._get_data_of_element(
        element, pixel_data, grid_coords
    ) == np.mean(data_inside)


def test_get_data_of_element_element_in_grid_RGB():
    """get_data_of_element if the element is in grid for RGB data."""

    pixel_data = np.random.randint(0, 256, size=(4, 4, 4, 3))
    grid_coords = GridCoords(
        slice=np.array([0, 1, 2, 3]),
        row=np.array([0, 1, 2, 3]),
        col=np.array([0, 1, 2, 3]),
    )

    element_node_grid_coords = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    interpolator = InterpolatorAllVoxel()

    data = interpolator._get_data_of_element(
        element_node_grid_coords, pixel_data, grid_coords
    )

    assert data is not None
    assert data.shape == (3,)
    assert np.all(data >= 0) and np.all(data <= 255)


def test_get_data_of_element_element_not_in_grid():
    """get_data_of_element if the element is not in grid."""

    element = np.array(
        [
            [0, 2, -1],
            [0, 3, -1],
            [1, 3, -1],
            [1, 2, -1],
            [0, 2, -2],
            [0, 3, -2],
            [1, 3, -2],
            [1, 2, -2],
        ]
    )

    N_slice, N_row, N_col = 5, 5, 5
    pixel_data = (
        np.arange(N_slice * N_row * N_col)
        .reshape((N_slice, N_row, N_col))
        .astype(float)
    )

    slice_coords = np.arange(5)
    row_coords = np.arange(5)
    col_coords = np.arange(5)

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)

    interpolator = InterpolatorAllVoxel()

    assert (
        interpolator._get_data_of_element(element, pixel_data, grid_coords)
        is None
    )

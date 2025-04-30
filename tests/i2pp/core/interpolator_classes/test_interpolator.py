"""Test Interpolator Routine."""

import numpy as np
from i2pp.core.image_reader_classes.image_reader import (
    GridCoords,
    ImageData,
    PixelValueType,
)
from i2pp.core.interpolator_classes.interpolator import Interpolator


def test_world_to_grid_coords_no_roation():
    """Test world_to_grid_coords if the conversion is done correctly."""

    orientation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    coords_grid = np.array([10, 20, 30])
    world_coords = np.array([[11, 21, 31], [9, 19, 29], [10, 22, 32]])

    expected_grid_coords = np.array([[1, 1, 1], [-1, -1, -1], [2, 0, 2]])

    interpolator = Interpolator()

    result = interpolator.world_to_grid_coords(
        world_coords, orientation, coords_grid
    )

    assert np.array_equal(result, expected_grid_coords)


def test_world_to_grid_coords_90_deg_rotation():
    """Test world_to_grid_coords if the conversion is done correctly."""

    orientation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    coords_grid = np.array([5, 10, 20])
    world_coords = np.array([[2, 8, 18], [7, 13, 25], [4, 11, 21]])

    expected_grid_coords = np.array([[-2, 3, -2], [3, -2, 5], [1, 1, 1]])

    expected_grid_coords = np.array([[-2, -2, 3], [5, 3, -2], [1, 1, 1]])

    interpolator = Interpolator()

    result = interpolator.world_to_grid_coords(
        world_coords, orientation, coords_grid
    )

    assert np.array_equal(result, expected_grid_coords)


def test_world_to_grid_coords_45_deg_rotation():
    """Test world_to_grid_coords if the conversion is done correctly."""

    theta = np.pi / 4
    orientation = np.array(
        [
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
            [1, 0, 0],
        ]
    )

    coords_grid = np.array([5, 5, 10])

    world_coords = np.array(
        [
            [6, 5, 10],
            [5, 6, 10],
            [5, 5, 11],
            [4, 5, 10],
        ]
    )

    expected_grid_coords = np.array(
        [
            [0, np.sqrt(2) / 2, -np.sqrt(2) / 2],
            [0, np.sqrt(2) / 2, np.sqrt(2) / 2],
            [1, 0, 0],
            [0, -np.sqrt(2) / 2, np.sqrt(2) / 2],
        ]
    )

    interpolator = Interpolator()

    result = interpolator.world_to_grid_coords(
        world_coords, orientation, coords_grid
    )

    assert np.allclose(result, expected_grid_coords)


def test_interpolate_image_values_to_points():
    """Tests the interpolation of image pixel values onto given target
    points."""

    N_slice, N_row, N_col = 5, 5, 5
    pixel_data = (
        np.arange(N_slice * N_row * N_col)
        .reshape((N_slice, N_row, N_col))
        .astype(float)
    )

    position = np.array([0, 0, 0])

    slice_coords = np.arange(N_slice)
    row_coords = np.arange(N_row)
    col_coords = np.arange(N_col)

    grid_coords = GridCoords(slice_coords, row_coords, col_coords)

    orientation = np.eye(3)

    image_data = ImageData(
        pixel_data=pixel_data,
        grid_coords=grid_coords,
        orientation=orientation,
        position=position,
        pixel_type=PixelValueType.CT,
    )

    target_points = np.array(
        [
            [1.5, 1.5, 1.5],
            [3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0],
            [3, 2, 1],
            [6.0, 6.0, 6.0],
        ]
    )

    expected_results = np.array(
        [
            (31 + 32 + 36 + 37 + 56 + 57 + 61 + 62) / 8,
            93,
            0,
            3 * 25 + 2 * 5 + 1,
            np.nan,
        ]
    )

    interpolator = Interpolator()

    result = interpolator.interpolate_image_values_to_points(
        target_points, image_data
    )

    assert np.all(np.isclose(result, expected_results, equal_nan=True))

"""Test Visualization Routine."""

import numpy as np
from i2pp.core.image_reader_classes.image_reader import (
    GridCoords,
    ImageData,
    PixelValueType,
)
from i2pp.core.visualization_classes.image_visualization import ImageVisualizer


def test_create_structured_grid_from_image_data_scalar_no_transformation():
    """Test create_structured_grid_from_image_data for scalar if no
    transformation is needed."""
    visualizer = ImageVisualizer(PixelValueType.CT, [])

    grid_coords = GridCoords(
        slice=np.array([0, 1], dtype=np.float32),
        row=np.array([0, 1], dtype=np.float32),
        col=np.array([0, 1], dtype=np.float32),
    )

    image_data = ImageData(
        grid_coords=grid_coords,
        pixel_type=PixelValueType.CT,
        pixel_data=np.random.randint(0, 10, size=(2, 2, 2), dtype=np.uint8),
        orientation=np.eye(3),
        position=np.array([0, 0, 0]),
    )

    visualizer.compute_grid(image_data)

    assert "ScalarValues" in visualizer.grid.point_data
    assert visualizer.grid.point_data["ScalarValues"].shape == (8,)


def test_create_structured_grid_from_image_data_RGB_no_transformation():
    """Test create_structured_grid_from_image_data for RGB if no transformation
    is needed."""
    visualizer = ImageVisualizer(PixelValueType.RGB, [])

    grid_coords = GridCoords(
        slice=np.array([0, 1], dtype=np.float32),
        row=np.array([0, 1], dtype=np.float32),
        col=np.array([0, 1], dtype=np.float32),
    )

    image_data = ImageData(
        grid_coords=grid_coords,
        pixel_type=PixelValueType.RGB,
        pixel_data=np.random.randint(
            0, 255, size=(2, 2, 2, 3), dtype=np.uint8
        ),
        orientation=np.eye(3),
        position=np.array([0, 0, 0]),
    )

    visualizer.compute_grid(image_data)

    assert "rgb_values" in visualizer.grid.point_data
    assert visualizer.grid.point_data["rgb_values"].shape == (8, 3)


def test_create_structured_grid_from_image_data_scalar_with_transformation():
    """Test create_structured_grid_from_image_data for scalar with
    transformation."""
    visualizer = ImageVisualizer(PixelValueType.CT, [])

    grid_coords = GridCoords(
        slice=np.array([0, 1], dtype=np.float32),
        row=np.array([0, 1], dtype=np.float32),
        col=np.array([0, 1], dtype=np.float32),
    )

    pixel_data = np.random.randint(0, 255, size=(2, 2, 2, 3), dtype=np.uint8)

    orientation = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32
    )

    position = np.array([1, 2, 3], dtype=np.float32)

    image_data = ImageData(
        grid_coords=grid_coords,
        pixel_type=PixelValueType.RGB,
        pixel_data=pixel_data,
        orientation=orientation,
        position=position,
    )

    visualizer.compute_grid(image_data)

    transformed_points = visualizer.grid.points
    expected_point = np.array([1, 2, 3])

    assert np.allclose(transformed_points[0], expected_point)
    assert np.allclose(transformed_points[1], position + np.array([0, 1, 0]))

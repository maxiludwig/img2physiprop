"""Test Image Reader Routine."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from i2pp.core.discretization_reader_classes.discretization_reader import (
    BoundingBox,
)
from i2pp.core.image_reader_classes.image_reader import PixelValueType
from i2pp.core.image_reader_classes.png_reader import PngReader
from PIL import Image


def test_verify_image_metadata_input_missing():
    """_verify_image_metadata if information is missing."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1, 1],
            "image_position": None,
            "row_direction": [0, -1, 0],
            "column_direction": [1, 0, 0],
            "slice_direction": [1, 0, 0],
        }
    }
    test_class = PngReader(test_config, [])
    with pytest.raises(
        RuntimeError,
        match="Missing parameter 'image_position' in image_metadata.",
    ):
        test_class._verify_image_metadata(test_config["image_metadata"])


def test_verify_image_metadata_wrong_type():
    """_verify_image_metadata when information has wrong type."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1, 1],
            "image_position": 1,
            "row_direction": [0, -1, 0],
            "column_direction": [1, 0, 0],
            "slice_direction": [1, 0, 0],
        }
    }
    test_class = PngReader(test_config, [])
    with pytest.raises(
        RuntimeError, match="Parameter 'image_position' has the wrong type."
    ):
        test_class._verify_image_metadata(test_config["image_metadata"])


def test_verify_image_metadata_wrong_shape():
    """_verify_image_metadata when Spacing is wrong."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1],
            "image_position": [0, 0, 0],
            "row_direction": [0, -1, 0],
            "column_direction": [1, 0, 0],
            "slice_direction": [1, 0, 0],
        }
    }
    test_class = PngReader(test_config, [])
    with pytest.raises(
        RuntimeError, match="Parameter 'pixel_spacing' has the wrong shape."
    ):
        test_class._verify_image_metadata(test_config["image_metadata"])


def test_verify_image_metadata_default_orientation():
    """Test _verify_image_metadata if all config_parameters are None."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1, 1],
            "image_position": [0, 0, 0],
            "row_direction": None,
            "column_direction": None,
            "slice_direction": None,
        }
    }
    test_class = PngReader(
        test_config, BoundingBox(max=[0, 0, 1000], min=[1, 1, -1000])
    )

    assert (
        test_class._verify_image_metadata(test_config["image_metadata"])
        is None
    )


def test_load_image_png(tmp_path: Path):
    """Test load_image for PNG."""

    width, height = 2, 2
    image_1 = Image.new("RGB", (width, height), color=(255, 255, 255))
    image_2 = Image.new("RGB", (width, height), color=(55, 55, 55))

    png_file_path_1 = tmp_path / "testpng_1.png"
    png_file_path_2 = tmp_path / "testpng_2.png"
    image_1.save(png_file_path_1)
    image_2.save(png_file_path_2)

    test_config = {"image_metadata": "Test"}

    test_input = PngReader(test_config, BoundingBox([], []))
    input_path = tmp_path

    with patch("PIL.Image.open", wraps=Image.open) as mock_image_open:
        with patch.object(
            PngReader, "_verify_image_metadata", return_value=None
        ):
            test_input.load_image(input_path)

            assert mock_image_open.call_count == 2


def test_convert_to_image_data_PNG():
    """Test image_to_slices for PNG."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [1, 1, 1],
            "image_position": [0, 0, 0],
            "row_direction": [1, 0, 0],
            "column_direction": [0, 1, 0],
            "slice_direction": [0, 0, 1],
        }
    }

    png = []
    for _ in range(2):
        png.append(np.random.rand(3, 3, 3))

    test_input = PngReader(
        test_config, BoundingBox(max=[0, 0, 1000], min=[1, 1, -1000])
    )
    image_data = test_input.convert_to_image_data(png)

    assert image_data.pixel_type == PixelValueType.RGB

    expected_orientation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    assert np.array_equal(image_data.orientation, expected_orientation)


def test_convert_to_image_data_png_dafault_orientation():
    """Test image_to_slices for PNG if Image_Orientation is None."""

    test_config = {
        "image_metadata": {
            "pixel_spacing": [0.5, 0.5, 0.5],
            "slice_thickness": 1,
            "image_position": [0, 0, 0],
            "row_direction": None,
            "column_direction": None,
            "slice_direction": None,
        }
    }
    png = []
    for _ in range(3):
        png.append(np.random.rand(3, 3, 3))

    test_input = PngReader(
        test_config, BoundingBox(max=[1, 1, 2], min=[0, 0, 0.5])
    )
    image_data = test_input.convert_to_image_data(png)

    assert np.array_equal(image_data.grid_coords.slice, np.array([0, 0.5]))
    assert np.array_equal(image_data.position, np.array([0, 0, 0.5]))
    assert image_data.pixel_data.shape == (2, 3, 3, 3)

    expected_orientation = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])

    assert np.array_equal(image_data.orientation, expected_orientation)

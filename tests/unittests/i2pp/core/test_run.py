"""Test run routine."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from i2pp.core.run import run_i2pp


@pytest.fixture
def minimal_valid_config(tmp_path):
    """Fixture to provide a minimal valid configuration for testing i2pp
    run."""

    # Create dummy discretization file
    disc_path = tmp_path / "discretization.mesh"
    disc_path.write_bytes(b"dummy mesh content")

    # Create dummy image file
    image_path = tmp_path / "image.dcm"
    image_path.write_bytes(b"dummy image content")

    # Create dummy user script
    script_path = tmp_path / "user_script.py"
    script_path.write_text(
        "def process_image_data(id, data):\n    return data*7\n"
    )

    return {
        "import": {
            "discretization": {
                "path": str(disc_path),
                "type": "mesh",
            },
            "image": {
                "path": str(image_path),
                "type": "dicom",
            },
        },
        "processing": {
            "interpolation_method": "nodes",
            "transformation": {
                "user_script": str(script_path),
                "user_function": "process_image_data",
                "normalize_values": False,
                "visualize": False,
            },
        },
        "export": {
            "folder_path": str(tmp_path),
            "file_name": "output",
            "type": "pattern",
            "output_parameter_name": "parameter",
        },
    }


@patch("i2pp.core.run.verify_and_load_discretization")
@patch("i2pp.core.run.verify_and_load_imagedata")
@patch("i2pp.core.run.smooth_data")
@patch("i2pp.core.run.interpolate_image_to_discretization")
@patch("i2pp.core.run.transform_data")
@patch("i2pp.core.run.export_data")
@patch("i2pp.core.run.visualize_smoothing")
@patch("i2pp.core.run.visualize_results")
@patch("i2pp.core.run.create_mesh_mask")
def test_run_i2pp_with_minimal_valid_config(
    mock_create_mesh_mask,
    mock_visualize_results,
    mock_visualize_smoothing,
    mock_export_data,
    mock_transform_data,
    mock_interpolate,
    mock_smooth_data,
    mock_verify_image,
    mock_verify_discretization,
    minimal_valid_config,
):
    """Test the run_i2pp function with a minimal valid configuration."""
    mock_discretization = MagicMock()
    mock_discretization.bounding_box = ((0, 0, 0), (1, 1, 1))
    mock_verify_discretization.return_value = mock_discretization

    mock_image_data = MagicMock()
    mock_image_data.pixel_data = [[0]]
    mock_image_data.pixel_range = (0, 255)
    mock_image_data.pixel_type = "uint8"
    mock_verify_image.return_value = mock_image_data

    mock_smooth_data.side_effect = lambda pixel_data, area: pixel_data

    mock_interpolate.return_value = "interpolated_elements"
    mock_transform_data.return_value = "transformed_data"

    # Call the function
    run_i2pp(minimal_valid_config)

    # Assertions to ensure key steps were called
    mock_verify_discretization.assert_called_once()
    mock_verify_image.assert_called_once()
    mock_create_mesh_mask.assert_called_once()
    mock_smooth_data.assert_not_called()
    mock_visualize_smoothing.assert_not_called()
    mock_interpolate.assert_called_once()
    mock_transform_data.assert_called_once()
    mock_visualize_results.assert_not_called()
    mock_export_data.assert_called_once()


@pytest.fixture
def maximal_valid_config(tmp_path):
    """Fixture to provide a maximal valid configuration for testing i2pp
    run."""

    # Create dummy discretization file
    disc_path = tmp_path / "discretization.yaml"
    disc_path.write_bytes(b"dummy mesh content")

    # Create dummy image file
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"dummy image content")

    # Create dummy user script
    script_path = tmp_path / "user_script.py"
    script_path.write_text(
        "def process_image_data(id, data):\n    return data*7\n"
    )

    return {
        "import": {
            "discretization": {
                "path": str(disc_path),
                "type": "yaml",
            },
            "image": {
                "path": str(image_path),
                "type": "png",
                "options": {
                    "metadata": {
                        "pixel_spacing": [0.5, 0.5, 1.0],
                        "row_direction": [0, -1, 0],
                        "column_direction": [1, 0, 0],
                        "slice_direction": [0, 0, 1],
                        "image_position": [0, 0, 0],
                    },
                },
            },
        },
        "processing": {
            "interpolation_method": "nodes",
            "transformation": {
                "user_script": str(script_path),
                "user_function": "process_image_data",
                "normalize_values": True,
                "visualize": True,
            },
            "smoothing": {
                "area": 1.0,
                "visualize": True,
            },
        },
        "export": {
            "folder_path": str(tmp_path),
            "file_name": "output",
            "type": "pattern",
            "output_parameter_name": "parameter",
        },
    }


@patch("i2pp.core.run.verify_and_load_discretization")
@patch("i2pp.core.run.verify_and_load_imagedata")
@patch("i2pp.core.run.smooth_data")
@patch("i2pp.core.run.interpolate_image_to_discretization")
@patch("i2pp.core.run.transform_data")
@patch("i2pp.core.run.export_data")
@patch("i2pp.core.run.visualize_smoothing")
@patch("i2pp.core.run.visualize_results")
@patch("i2pp.core.run.create_mesh_mask")
def test_run_i2pp_with_maximal_valid_config(
    mock_create_mesh_mask,
    mock_visualize_results,
    mock_visualize_smoothing,
    mock_export_data,
    mock_transform_data,
    mock_interpolate,
    mock_smooth_data,
    mock_verify_image,
    mock_verify_discretization,
    maximal_valid_config,
):
    """Test the run_i2pp function with a maximal valid configuration."""
    mock_discretization = MagicMock()
    mock_discretization.bounding_box = ((0, 0, 0), (1, 1, 1))
    mock_verify_discretization.return_value = mock_discretization

    mock_image_data = MagicMock()
    mock_image_data.pixel_data = np.array([[0, 1], [2, 3]])
    mock_image_data.pixel_range = (0, 255)
    mock_image_data.pixel_type = "uint8"
    mock_verify_image.return_value = mock_image_data

    mock_smooth_data.side_effect = lambda *args, **kwargs: args[0]

    mock_interpolate.return_value = "interpolated_elements"
    mock_transform_data.return_value = "transformed_data"

    # Call the function
    run_i2pp(maximal_valid_config)

    # Assertions to ensure key steps were called
    mock_verify_discretization.assert_called_once()
    mock_verify_image.assert_called_once()
    mock_create_mesh_mask.assert_called_once()
    mock_smooth_data.assert_called_once()
    mock_visualize_smoothing.assert_called_once()
    mock_interpolate.assert_called_once()
    mock_transform_data.assert_called_once()
    mock_visualize_results.assert_called_once()
    mock_export_data.assert_called_once()

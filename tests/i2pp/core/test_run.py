"""Test run routine."""

from pathlib import Path
from unittest import mock

import pytest
from i2pp.core.run import run_i2pp


@pytest.fixture
def minimal_valid_config(tmp_path):
    """Fixture to provide a minimal valid configuration for testing i2pp
    run."""

    return {
        "discretization": {
            "path": "tests/testdata/discretization.mesh",
            "type": "mesh",
        },
        "image": {"path": "tests/testdata/image.dcm", "type": "dicom"},
        "processing options": {
            "smoothing": True,
            "smoothing_area": 3,
            "interpolation_method": "nodes",
            "user_script": "tests/testdata/user_script.py",
            "user_function": "process_image_data",
            "normalize_values": False,
        },
        "visualization_options": {
            "plot_smoothing": False,
            "plot_results": False,
        },
        "output options": {
            "output_path": tmp_path,
            "output_name": "output",
            "export_format": "pattern",
        },
    }


@pytest.fixture
def large_valid_config(tmp_path):
    """Fixture to provide a larger, more complex configuration for testing i2pp
    run."""
    return {
        "discretization": {
            "path": "tests/testdata/discretization_large.mesh",
            "type": "mesh",
        },
        "image": {
            "path": "tests/testdata/image_large.dcm",
            "type": "dicom",
            "metadata": {"spacing": [0.5, 0.5, 1.0]},
        },
        "processing options": {
            "smoothing": True,
            "smoothing_area": 5,
            "interpolation_method": "elements",
            "material_ids": [1, 2, 3],
            "user_script": "tests/testdata/user_script.py",
            "user_function": "process_image_data",
            "normalize_values": True,
        },
        "visualization_options": {
            "plot_smoothing": True,
            "plot_results": True,
        },
        "output options": {
            "output_path": tmp_path,
            "output_name": "output_large",
            "export_format": "json",
        },
    }


@mock.patch("i2pp.core.run.verify_and_load_discretization")
@mock.patch("i2pp.core.run.verify_and_load_imagedata")
@mock.patch("i2pp.core.run.interpolate_image_to_discretization")
@mock.patch("i2pp.core.run.export_data")
@mock.patch("i2pp.core.run.smooth_data")
@mock.patch("i2pp.core.run.visualize_results")
@mock.patch("i2pp.core.run.visualize_smoothing")
def test_run_i2pp_runs_successfully(
    mock_vis_smoothing,
    mock_vis_results,
    mock_smooth_data,
    mock_export,
    mock_interpolate,
    mock_load_image,
    mock_load_dis,
    minimal_valid_config,
):
    """Test that run_i2pp executes successfully with a minimal valid
    configuration."""

    mock_dis = mock.Mock()
    mock_dis.bounding_box = ((0, 0, 0), (1, 1, 1))
    mock_load_dis.return_value = mock_dis

    mock_image = mock.Mock()
    mock_image.pixel_data = [[0]]
    mock_image.pixel_range = (0, 255)
    mock_image.pixel_type = "dummy"
    mock_load_image.return_value = mock_image

    mock_smooth_data.return_value = mock_image

    mock_elements = [{"id": 1, "value": 123}]
    mock_interpolate.return_value = mock_elements

    run_i2pp(minimal_valid_config)

    mock_load_dis.assert_called_once()
    mock_load_image.assert_called_once()
    mock_interpolate.assert_called_once()
    mock_export.assert_called_once_with(
        elements=mock_elements,
        dis=mock_dis,
        user_script_path=Path("tests/testdata/user_script.py"),
        user_function_name="process_image_data",
        export_format="pattern",
        property_output_file=Path(
            minimal_valid_config["output options"]["output_path"]
        )
        / "output.pattern",
        name_of_output_property=None,
        normalize=False,
        vtk_output_file=Path(
            minimal_valid_config["output options"]["output_path"]
        )
        / "output.vtu",
        pixel_range=(0, 255),
        pixel_type="dummy",
    )
    mock_smooth_data.assert_called_once_with([[0]], 3)
    mock_vis_results.assert_not_called()
    mock_vis_smoothing.assert_not_called()


@mock.patch("i2pp.core.run.verify_and_load_discretization")
@mock.patch("i2pp.core.run.verify_and_load_imagedata")
@mock.patch("i2pp.core.run.interpolate_image_to_discretization")
@mock.patch("i2pp.core.run.export_data")
@mock.patch("i2pp.core.run.smooth_data")
@mock.patch("i2pp.core.run.visualize_results")
@mock.patch("i2pp.core.run.visualize_smoothing")
def test_run_i2pp_with_large_config(
    mock_vis_smoothing,
    mock_vis_results,
    mock_smooth_data,
    mock_export,
    mock_interpolate,
    mock_load_image,
    mock_load_dis,
    large_valid_config,
):
    """Test that run_i2pp executes successfully with a larger, more complex
    configuration."""

    mock_dis = mock.Mock()
    mock_dis.bounding_box = ((0, 0, 0), (10, 10, 10))
    mock_load_dis.return_value = mock_dis

    mock_image = mock.Mock()
    mock_image.pixel_data = [[0] * 100] * 100
    mock_image.pixel_range = (0, 255)
    mock_image.pixel_type = "dummy"
    mock_load_image.return_value = mock_image

    mock_smooth_data.return_value = mock_image

    mock_elements = [{"id": 1, "value": 123}]
    mock_interpolate.return_value = mock_elements

    run_i2pp(large_valid_config)

    assert len(mock_load_dis.call_args_list) == 1
    assert len(mock_load_image.call_args_list) == 1
    assert len(mock_interpolate.call_args_list) == 1
    assert len(mock_export.call_args_list) == 1
    mock_export.assert_called_once_with(
        elements=mock_elements,
        dis=mock_dis,
        user_script_path=Path("tests/testdata/user_script.py"),
        user_function_name="process_image_data",
        export_format="json",
        property_output_file=Path(
            large_valid_config["output options"]["output_path"]
        )
        / "output_large.json",
        name_of_output_property=None,
        normalize=True,
        vtk_output_file=Path(
            large_valid_config["output options"]["output_path"]
        )
        / "output_large.vtu",
        pixel_range=(0, 255),
        pixel_type="dummy",
    )
    mock_smooth_data.assert_called_once_with([[0] * 100] * 100, 5)
    mock_vis_results.assert_called_once()
    mock_vis_smoothing.assert_called_once()

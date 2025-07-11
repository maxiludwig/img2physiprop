"""Test run routine."""

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
        },
        "visualization_options": {
            "plot_smoothing": False,
            "plot_results": False,
        },
        "export": {"path": tmp_path / "output.pattern", "format": "pattern"},
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
    mock_load_image.return_value = mock_image

    mock_smooth_data.return_value = mock_image

    mock_elements = [{"id": 1, "value": 123}]
    mock_interpolate.return_value = mock_elements

    run_i2pp(minimal_valid_config)

    mock_load_dis.assert_called_once()
    mock_load_image.assert_called_once()
    mock_interpolate.assert_called_once()
    mock_export.assert_called_once_with(
        mock_elements, minimal_valid_config, (0, 255)
    )
    mock_smooth_data.assert_called_once()
    mock_vis_results.assert_not_called()
    mock_vis_smoothing.assert_not_called()

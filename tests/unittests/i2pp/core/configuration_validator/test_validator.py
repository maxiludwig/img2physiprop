"""Test the configuration validator."""

import pytest
from i2pp.core.configuration_validator.validator import I2PPConfig


@pytest.fixture
def minimal_valid_config(tmp_path):
    """Fixture to provide a minimal valid configuration for testing i2pp
    run."""

    # Create dummy files
    image_path = tmp_path / "image.dcm"
    image_path.write_text("dummy image data")

    disc_path = tmp_path / "discretization.mesh"
    disc_path.write_text("dummy mesh data")

    script_path = tmp_path / "user_script.py"
    script_path.write_text("def process_image_data(data): return data")

    return {
        "import": {
            "discretization": {
                "path": str(disc_path),
                "type": "mesh",
            },
            "image": {"path": str(image_path), "type": "dicom"},
        },
        "processing": {
            "interpolation": {
                "method": "nodes",
            },
            "transformation": {
                "user_script": str(script_path),
                "user_function": "process_image_data",
                "normalize_values": False,
            },
        },
        "export": {
            "folder_path": str(tmp_path),
            "file_name": "output",
            "type": "pattern",
        },
    }


@pytest.fixture
def large_valid_config(tmp_path):
    """Fixture to provide a larger, more complex configuration for testing i2pp
    run."""

    # Create dummy discretization file
    disc_path = tmp_path / "discretization_large.mesh"
    disc_path.write_text("dummy mesh content")

    # Create dummy image file
    image_path = tmp_path / "image_large.dcm"
    image_path.write_text("dummy image content")

    # Create dummy user script
    script_path = tmp_path / "user_script.py"
    script_path.write_text("def process_image_data(data): return data")

    # Create output directory (optional — could be same as tmp_path)
    output_path = tmp_path / "output"
    output_path.mkdir()

    return {
        "import": {
            "discretization": {
                "path": str(disc_path),
                "type": "mesh",
                "options": {"material_ids": [1, 2, 3]},
            },
            "image": {
                "path": str(image_path),
                "type": "dicom",
                "options": {
                    "metadata": {
                        "pixel_spacing": [0.5, 0.5, 1.0],
                        "row_direction": [1, 0, 0],
                        "column_direction": [0, 1, 0],
                        "slice_direction": [0, 0, 1],
                        "image_position": [0, 0, 0],
                    },
                },
            },
        },
        "processing": {
            "smoothing": {"area": 5, "visualize": True},
            "interpolation": {
                "method": "elementcenter",
            },
            "transformation": {
                "user_script": str(script_path),
                "user_function": "process_image_data",
                "normalize_values": True,
                "visualize": True,
            },
        },
        "export": {
            "folder_path": str(output_path),
            "file_name": "output_large",
            "type": "json",
            "output_parameter_name": "STIFFNESS",
        },
    }


def test_minimal_valid_config(minimal_valid_config):
    """Test that the minimal valid configuration can be loaded without
    errors."""
    config = I2PPConfig.from_dict(minimal_valid_config)
    assert config is not None

    # check that smoothing is None
    assert config.processing.smoothing is None

    # assert options is an empty dictionary
    assert config.import_.image.options == {}


def test_large_valid_config(large_valid_config):
    """Test that the larger valid configuration can be loaded without
    errors."""
    config = I2PPConfig.from_dict(large_valid_config)
    assert config is not None


def test_config_validation_missing_key(tmp_path):
    """Test that run_i2pp raises an error when a required key is missing in the
    configuration."""

    invalid_config = {
        "processing": {},
    }

    with pytest.raises(KeyError, match="import"):
        _ = I2PPConfig.from_dict(invalid_config)


def test_config_validation_non_existing_path(tmp_path):
    """Test that run_i2pp raises an error when a specified path does not
    exist."""

    invalid_config = {
        "import": {
            "discretization": {
                "path": str(tmp_path / "non_existing.mesh"),
                "type": "mesh",
            },
            "image": {"path": str(tmp_path / "image.dcm"), "type": "dicom"},
        }
    }

    with pytest.raises(
        FileNotFoundError,
        match=f"Path does not exist: {str(tmp_path / 'non_existing.mesh')}",
    ):
        _ = I2PPConfig.from_dict(invalid_config)


def test_config_validation_both_surface_values_set(minimal_valid_config):
    """Test that an error is raised if both surface node and element values are
    set."""
    invalid_config = minimal_valid_config.copy()
    invalid_int_config = invalid_config["processing"]["interpolation"]
    invalid_int_config["set_surface_node_value"] = 1.0
    invalid_int_config["set_surface_element_value"] = 2.0

    with pytest.raises(
        ValueError,
        match="Both 'set_surface_node_value' and 'set_surface_element_value' "
        "cannot be set at the same time.",
    ):
        _ = I2PPConfig.from_dict(invalid_config)

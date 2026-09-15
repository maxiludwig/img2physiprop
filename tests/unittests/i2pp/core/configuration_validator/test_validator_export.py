"""Test the validation of the export configuration."""

from pathlib import Path

import pytest
from i2pp.core.configuration_validator.validator import Export, I2PPConfig


def _config(tmp_path: Path, discretization: str, export_type: str) -> dict:
    """Create a configuration with the given discretization file name."""
    for name in (discretization, "ct.nii.gz", "user_function.py"):
        (tmp_path / name).write_bytes(b"")
    return {
        "import": {
            "discretization": {
                "path": str(tmp_path / discretization),
                "type": "mesh",
            },
            "image": {"path": str(tmp_path / "ct.nii.gz"), "type": "nifti"},
        },
        "processing": {
            "interpolation": {"method": "elementcenter"},
            "transformation": {
                "user_script": str(tmp_path / "user_function.py"),
                "user_function": "f",
            },
        },
        "export": {
            "folder_path": str(tmp_path),
            "file_name": "output",
            "type": export_type,
        },
    }


def test_discretization_export_for_vtu(tmp_path: Path):
    """Test that the discretization export is accepted for .vtu input."""
    config = I2PPConfig.from_dict(
        _config(tmp_path, "tree.vtu", "discretization")
    )

    assert config.export.type == "discretization"


@pytest.mark.parametrize("discretization", ["mesh.4C.yaml", "mesh.mesh"])
def test_discretization_export_rejects_other_formats(
    tmp_path: Path, discretization
):
    """Test that the discretization export is rejected for other input."""
    with pytest.raises(ValueError, match="only supported for .vtu"):
        I2PPConfig.from_dict(
            _config(tmp_path, discretization, "discretization")
        )


def _export(export_type: str, output_parameter_name=None) -> Export:
    """Create an Export configuration."""
    return Export.from_dict(
        {
            "folder_path": "output",
            "file_name": "output",
            "type": export_type,
            "output_parameter_name": output_parameter_name,
        }
    )


def test_output_parameter_name_for_json():
    """Test that the output parameter name is accepted for json export."""
    assert _export("json", "STIFFNESS").output_parameter_name == "STIFFNESS"


@pytest.mark.parametrize("export_type", ["txt", "discretization"])
def test_output_parameter_name_only_for_json(export_type):
    """Test that the output parameter name is rejected for other exports."""
    assert _export(export_type).output_parameter_name is None

    with pytest.raises(ValueError, match="only used for the export type"):
        _export(export_type, "STIFFNESS")

"""Test the validation of the labelmap interpolation configuration."""

from pathlib import Path

import pytest
from i2pp.core.configuration_validator.validator import Interpolation


def test_labelmap_valid_config(tmp_path: Path):
    """Test a labelmap configuration with the default label field."""
    label_map = tmp_path / "labels.nii.gz"
    label_map.write_bytes(b"")

    interpolation = Interpolation.from_dict(
        {"method": "labelmap", "labelmap": {"path": str(label_map)}}
    )

    assert interpolation.labelmap is not None
    assert interpolation.labelmap.path == label_map.resolve()
    assert interpolation.labelmap.label_field == "tu_label"


def test_labelmap_ignored_for_other_methods():
    """Test that other methods do not require a label map."""
    assert Interpolation.from_dict({"method": "nodes"}).labelmap is None


@pytest.mark.parametrize(
    "labelmap, error, match",
    [
        (None, ValueError, "requires the label map"),
        ({"label_field": "tu_label"}, ValueError, "requires the label map"),
        ({"path": "missing.nii.gz"}, FileNotFoundError, "does not exist"),
    ],
)
def test_labelmap_invalid_config(labelmap, error, match):
    """Test invalid labelmap configurations."""
    with pytest.raises(error, match=match):
        Interpolation.from_dict({"method": "labelmap", "labelmap": labelmap})


def test_labelmap_invalid_label_field(tmp_path: Path):
    """Test that the label field must be a name."""
    label_map = tmp_path / "labels.nii.gz"
    label_map.write_bytes(b"")

    with pytest.raises(ValueError, match="label_field"):
        Interpolation.from_dict(
            {
                "method": "labelmap",
                "labelmap": {"path": str(label_map), "label_field": 3},
            }
        )

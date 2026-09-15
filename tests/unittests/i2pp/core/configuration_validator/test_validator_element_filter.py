"""Test the validation of the element filter of the discretization."""

import logging
from pathlib import Path

import pytest
from i2pp.core.configuration_validator.validator import Discretization


def _discretization(tmp_path: Path, options) -> Discretization:
    """Create a Discretization configuration with the given options."""
    mesh = tmp_path / "tree.vtu"
    mesh.write_bytes(b"")
    return Discretization.from_dict(
        {"path": str(mesh), "type": "vtu", "options": options}
    )


def test_element_filter_is_kept(tmp_path: Path):
    """Test that a valid element filter is passed on unchanged."""
    element_filter = {"field": "block_id", "values": [2]}

    dis = _discretization(tmp_path, {"element_filter": element_filter})

    assert dis.options == {"element_filter": element_filter}


def test_without_options(tmp_path: Path):
    """Test that missing options result in an empty dictionary."""
    assert _discretization(tmp_path, None).options == {}


def test_material_ids_are_translated_with_warning(tmp_path: Path, caplog):
    """Test that the deprecated material_ids become an element filter on the
    material field MAT."""
    with caplog.at_level(logging.WARNING):
        dis = _discretization(tmp_path, {"material_ids": [1, 3]})

    assert dis.options == {
        "element_filter": {"field": "MAT", "values": [1, 3]}
    }
    assert "'material_ids' is deprecated" in caplog.text


def test_material_ids_none_is_ignored(tmp_path: Path, caplog):
    """Test that material_ids set to None neither filters nor warns."""
    with caplog.at_level(logging.WARNING):
        dis = _discretization(tmp_path, {"material_ids": None})

    assert dis.options == {}
    assert caplog.text == ""


def test_empty_material_ids_are_ignored(tmp_path: Path, caplog):
    """Test that an empty list of material_ids means no filtering."""
    with caplog.at_level(logging.WARNING):
        dis = _discretization(tmp_path, {"material_ids": []})

    assert dis.options == {}
    assert caplog.text == ""


def test_material_ids_and_element_filter_raise(tmp_path: Path):
    """Test that both filter options cannot be set at the same time."""
    with pytest.raises(ValueError, match="cannot be set at the same time"):
        _discretization(
            tmp_path,
            {
                "material_ids": [1],
                "element_filter": {"field": "block_id", "values": [2]},
            },
        )


@pytest.mark.parametrize(
    "element_filter",
    [
        {"field": "block_id"},
        {"values": [2]},
        {"field": "block_id", "values": []},
        {"field": "block_id", "values": 2},
        {"field": 1, "values": [2]},
        "block_id",
    ],
)
def test_malformed_element_filter_raises(tmp_path: Path, element_filter):
    """Test that malformed element filters are rejected."""
    with pytest.raises(ValueError, match="requires a 'field'"):
        _discretization(tmp_path, {"element_filter": element_filter})

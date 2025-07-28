"""Test cases for discretization format detection class in
i2pp.core.discretization_readers."""

import pytest
from i2pp.core.discretization_readers.discretization_format import (
    DiscretizationFormat,
)
from i2pp.core.discretization_readers.fourc_yaml_reader import FourCYamlReader
from i2pp.core.discretization_readers.mesh_reader import MeshReader


def test_get_reader_for_yaml():
    """Test that the YAML reader is returned for the YAML format."""
    assert DiscretizationFormat.YAML.get_reader() == FourCYamlReader


def test_get_reader_for_mesh():
    """Test that the Mesh reader is returned for the Mesh format."""
    assert DiscretizationFormat.MESH.get_reader() == MeshReader


def test_enum_values_are_correct():
    """Test that the enum values are correctly defined."""
    assert DiscretizationFormat.YAML.value == ".yaml"
    assert DiscretizationFormat.MESH.value == ".mesh"


def test_invalid_format_access():
    """Test that accessing an invalid format raises a KeyError."""
    with pytest.raises(KeyError):
        _ = {
            DiscretizationFormat.MESH: MeshReader,
            DiscretizationFormat.YAML: FourCYamlReader,
        }[".invalid"]

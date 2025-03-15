"""Test Mesh Reader Routine."""

from pathlib import Path
from unittest.mock import patch

import pytest
from i2pp.core.discretization_reader_classes.dat_reader import DatReader
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Nodes,
)
from i2pp.core.import_discretization import (
    BoundingBox,
    DiscretizationFormat,
    determine_discretization_format,
    verify_and_load_discretization,
)


def test_determine_discretization_format_path_not_exist():
    """Test determine_discretization_format if path not exist."""
    path = "not_existing_path"

    with pytest.raises(
        RuntimeError,
        match="Path not_existing_path to the Discretization cannot be found!",
    ):
        determine_discretization_format(Path(path))


def test_determine_discretization_format_wrong_format(tmp_path: Path) -> None:
    """Test determine_discretization_format if Mesh has wrong format."""

    test_path = tmp_path / "test_mesh.stl"

    with patch("pathlib.Path.is_file", returnValue=True):
        with pytest.raises(RuntimeError, match=".stl not readable!"):
            determine_discretization_format(Path(test_path))


def test_determine_discretization_format_dat_format(tmp_path: Path) -> None:
    """Test determine_discretization_format if file is DAT."""

    test_path = tmp_path / "test_mesh.dat"

    with patch("pathlib.Path.is_file", returnValue=True):
        assert (
            determine_discretization_format(Path(test_path))
            == DiscretizationFormat.DAT
        )


def test_determine_discretization_format_mesh_format(tmp_path: Path) -> None:
    """Test determine_discretization_format if File is mesh."""

    test_path = tmp_path / "test_mesh.mesh"

    with patch("pathlib.Path.is_file", returnValue=True):
        assert (
            determine_discretization_format(Path(test_path))
            == DiscretizationFormat.MESH
        )


def test_verify_and_load_discretization():
    """Test verify_and_load_discretization."""

    test_config = {
        "input informations": {
            "discretization_file_path": "test_path.dat",
        },
        "processing options": "options",
    }

    absolut_path = Path.cwd() / "test_path.dat"
    mock_bounding = tuple([[0, 0, 0], [1, 1, 1]])
    nodes = Nodes([0, 0, 0], 0)
    mock_dis = Discretization(nodes, [])

    with patch(
        "i2pp.core.import_discretization.determine_discretization_format",
        return_value=DiscretizationFormat.DAT,
    ) as mock_determine_discretization_format:
        with patch.object(
            DatReader, "load_discretization", return_value=mock_dis
        ) as mock_load_discretization:
            with patch(
                "i2pp.core.import_discretization.find_mins_maxs",
                return_value=mock_bounding,
            ) as mock_find_mins_maxs:
                dis = verify_and_load_discretization(test_config)

                mock_determine_discretization_format.assert_called_once_with(
                    absolut_path
                )
                mock_load_discretization.assert_called_once_with(
                    absolut_path, "options"
                )
                mock_find_mins_maxs.assert_called_once_with(
                    points=[0, 0, 0], enlargement=2
                )
                assert dis.bounding_box == BoundingBox(
                    min=[0, 0, 0], max=[1, 1, 1]
                )

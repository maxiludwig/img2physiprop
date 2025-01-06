"""Test Mesh Reader Routine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from i2pp.core.Mesh_Reader import MeshReader


def test_verify_input_path_not_exist():
    """Not verify_input if path not exist."""
    test_input = MeshReader([])
    directory = "not_existing_path"

    with pytest.raises(RuntimeError, match="Mesh data not found!"):
        test_input.verify_input(directory)


def test_verify_input_wrong_format(tmp_path: Path) -> None:
    """Test verify_input if Mesh has wrong format."""

    test_path = tmp_path / "test_mesh.stl"
    test_mesh = MeshReader([])

    with patch("os.path.isfile", returnValue=True):
        with pytest.raises(RuntimeError, match="Mesh data not readable!"):
            test_mesh.verify_input(test_path)


def test_load_mesh_mesh(tmp_path: Path) -> None:
    """Test load_mesh if input is .mesh."""

    test_path = tmp_path / "test_mesh.mesh"
    test_mesh = MeshReader([])
    with patch("trimesh.load", returnValue=None) as mock_trimesh:
        with patch(
            "i2pp.core.Mesh_Reader.MeshData", returnValue=None
        ) as MockClass:
            test_mesh.load_mesh(test_path)
            assert mock_trimesh.call_count == 1
            assert MockClass.call_count == 1


def test_load_mesh_dat(tmp_path: Path) -> None:
    """Test load_mesh if input is .dat."""

    test_path = tmp_path / "test_mesh.dat"
    test_mesh = MeshReader([])
    with patch("lnmmeshio.read") as mock_lnmreade:

        node1 = MagicMock(coords=[0.0, 0.0, 0.0])
        node2 = MagicMock(coords=[1.0, 0.0, 0.0])
        node3 = MagicMock(coords=[1.0, 1.0, 0.0])
        nodes = MagicMock(nodes=[node1, node2, node3])

        mock_lnmreade.return_value = nodes

        mesh_loaded = test_mesh.load_mesh(test_path)

        assert np.array_equal(
            mesh_loaded.nodes, np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        )

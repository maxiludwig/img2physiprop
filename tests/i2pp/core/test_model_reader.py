"""Test Mesh Reader Routine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from i2pp.core.import_model import verify_input
from i2pp.core.model_reader_classes.dat_reader import DatReader
from i2pp.core.model_reader_classes.mesh_reader import MeshReader
from i2pp.core.model_reader_classes.model_reader import ModelData


def test_verify_input_path_not_exist():
    """Test verify_input if path not exist."""
    directory = "not_existing_path"

    with pytest.raises(RuntimeError, match="Mesh data not found!"):
        verify_input(directory)


def test_verify_input_wrong_format(tmp_path: Path) -> None:
    """Test verify_input if Mesh has wrong format."""

    test_path = tmp_path / "test_mesh.stl"

    with patch("os.path.isfile", returnValue=True):
        with pytest.raises(RuntimeError, match="Mesh data not readable!"):
            verify_input(test_path)


def test_load_mesh_mesh(tmp_path: Path) -> None:
    """Test load_mesh if input is .mesh."""

    test_path = tmp_path / "test_model.mesh"
    test_model = MeshReader()
    with patch("trimesh.load", returnValue=None) as mock_trimesh:
        with patch(
            "i2pp.core.model_reader_classes.mesh_reader.ModelData",
            returnValue=None,
        ) as MockClass:
            test_model.load_model(str(test_path))
            assert mock_trimesh.call_count == 1
            assert MockClass.call_count == 1


def test_load_mesh_dat(tmp_path: Path) -> None:
    """Test load_mesh if input is .dat."""

    test_path = tmp_path / "test_mesh.dat"
    test_model = DatReader()
    with patch("lnmmeshio.read") as mock_lnmread:

        mock_model = MagicMock()
        mock_lnmread.return_value = mock_model

        mock_model.compute_ids.return_value = None
        node1 = [0.0, 0.0, 0.0]
        node2 = [1, 0.0, 0.0]
        node3 = [1, 1, 0.0]
        mock_model.get_node_coords.return_value = np.array(
            [node1, node2, node3]
        )

        # Mock für `ele.nodes`
        node1 = MagicMock(id=1)
        node2 = MagicMock(id=2)
        node3 = MagicMock(id=3)
        node4 = MagicMock(id=4)

        # Mock für `ele` in `structure`
        ele1 = MagicMock(nodes=[node3, node1])
        ele2 = MagicMock(nodes=[node2, node4])

        # `structure` als Liste von `ele`-Objekten
        mock_model.elements.structure = [ele1, ele2]

        model_loaded = test_model.load_model(str(test_path))

        # print(mesh_loaded.element_ids)

        assert np.array_equal(
            model_loaded.nodes, np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        )
        print(model_loaded.element_ids)
        assert np.array_equal(
            model_loaded.element_ids, np.array([[3, 1], [2, 4]])
        )


def test_get_center():
    """Test get_center if center of elements are calculated correctly."""
    ele1_ids = np.array([0, 2, 4], dtype=int)
    ele2_ids = np.array([1, 3, 5], dtype=int)
    test_element_ids = [ele1_ids, ele2_ids]
    test_nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 3, 6], [6, 2, 0], [9, 3, 6]]
    )
    test_model = ModelData(test_nodes, test_element_ids, [], [])

    data_reder = MeshReader()
    model = data_reder.get_center(test_model)

    expected_output = [[2, 1, 0], [4, 2, 4]]

    assert np.array_equal(model.element_center, expected_output)

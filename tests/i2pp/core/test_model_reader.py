"""Test Mesh Reader Routine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from i2pp.core.import_model import ModelFormat, verify_input
from i2pp.core.model_reader_classes.dat_reader import DatReader
from i2pp.core.model_reader_classes.mesh_reader import MeshReader
from i2pp.core.model_reader_classes.model_reader import (
    Element,
    ModelData,
    Nodes,
)


def test_verify_input_path_not_exist():
    """Test verify_input if path not exist."""
    directory = "not_existing_path"

    with pytest.raises(RuntimeError, match="Mesh data not found!"):
        verify_input(Path(directory))


def test_verify_input_wrong_format(tmp_path: Path) -> None:
    """Test verify_input if Mesh has wrong format."""

    test_path = tmp_path / "test_mesh.stl"

    with patch("pathlib.Path.is_file", returnValue=True):
        with pytest.raises(RuntimeError, match=".stl not readable!"):
            verify_input(Path(test_path))


def test_verify_input_dat_format(tmp_path: Path) -> None:
    """Test verify_input if Mesh has wrong format."""

    test_path = tmp_path / "test_mesh.dat"

    with patch("pathlib.Path.is_file", returnValue=True):
        assert verify_input(Path(test_path)) == ModelFormat.DAT


def test_verify_input_mesh_format(tmp_path: Path) -> None:
    """Test verify_input if Mesh has wrong format."""

    test_path = tmp_path / "test_mesh.mesh"

    with patch("pathlib.Path.is_file", returnValue=True):
        assert verify_input(Path(test_path)) == ModelFormat.MESH


def test_load_model_mesh(tmp_path: Path) -> None:
    """Test load_mesh if input is .mesh."""

    test_path = tmp_path / "test_model.mesh"
    test_model = MeshReader()

    test_config = {"Processing Informations": None}

    with patch("trimesh.load", returnValue=None) as mock_trimesh:
        with patch(
            "i2pp.core.model_reader_classes.mesh_reader.ModelData",
            returnValue=None,
        ) as MockClass:
            test_model.load_model(Path(test_path), test_config)
            assert mock_trimesh.call_count == 1
            assert MockClass.call_count == 1


def test__filter_model_one_filter():
    """Test _filter_model with 1 Filter."""
    mock_dis = MagicMock()
    node1 = MagicMock(id=1, coords=[0.0, 0.0, 0.0])
    node2 = MagicMock(id=2, coords=[1, 0.0, 0.0])
    node3 = MagicMock(id=3, coords=[0.0, 1, 0.0])
    node4 = MagicMock(id=4, coords=[0.0, 0.0, 1])

    ele1 = MagicMock(nodes=[node3, node1], options={"MAT": 1})
    ele2 = MagicMock(nodes=[node2, node4], options={"MAT": 2})
    ele3 = MagicMock(nodes=[node3, node4], options={"MAT": 3})
    ele4 = MagicMock(nodes=[node3, node4], options={"MAT": 2})

    mock_dis.elements.structure = [ele1, ele2, ele3, ele4]
    mock_dis.nodes = [node1, node2, node3, node4]
    test_model = DatReader()
    dis_filtered = test_model._filter_model(mock_dis, [2])
    coords_list = [node.coords for node in dis_filtered.nodes]

    expected_filtered_nodes = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

    assert all(x in coords_list for x in expected_filtered_nodes)

    assert len(coords_list) == 3


def test__filter_model_multiple_filter():
    """Test _filter_model with 1 Filter."""
    mock_dis = MagicMock()
    node1 = MagicMock(id=1, coords=[0.0, 0.0, 0.0])
    node2 = MagicMock(id=2, coords=[1, 0.0, 0.0])
    node3 = MagicMock(id=3, coords=[0.0, 1, 0.0])
    node4 = MagicMock(id=4, coords=[0.0, 0.0, 1])

    ele1 = MagicMock(nodes=[node3, node1], options={"MAT": 1})
    ele2 = MagicMock(nodes=[node2, node4], options={"MAT": 2})
    ele3 = MagicMock(nodes=[node3, node4], options={"MAT": 3})
    ele4 = MagicMock(nodes=[node3, node4], options={"MAT": 2})

    mock_dis.elements.structure = [ele1, ele2, ele3, ele4]
    mock_dis.nodes = [node1, node2, node3, node4]
    test_model = DatReader()
    dis_filtered = test_model._filter_model(mock_dis, [1, 3])
    coords_list = [node.coords for node in dis_filtered.nodes]

    expected_filtered_nodes = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]

    assert all(x in coords_list for x in expected_filtered_nodes)

    assert len(coords_list) == 3


def test_load_model_dat_without_filter(tmp_path: Path) -> None:
    """Test load_mesh if input is .dat."""

    test_path = tmp_path / "test_mesh.dat"
    test_model = DatReader()
    with patch("lnmmeshio.read") as mock_lnmread:

        mock_model = MagicMock()
        mock_lnmread.return_value = mock_model

        mock_model.compute_ids.return_value = None

        node1 = MagicMock(id=1, coords=[0.0, 0.0, 0.0])
        node2 = MagicMock(id=2, coords=[1, 0.0, 0.0])
        node3 = MagicMock(id=3, coords=[0.0, 1, 0.0])
        node4 = MagicMock(id=4, coords=[0.0, 0.0, 1])

        ele1 = MagicMock(nodes=[node3, node1])
        ele2 = MagicMock(nodes=[node2, node4])

        mock_model.elements.structure = [ele1, ele2]
        mock_model.nodes = [node1, node2, node3, node4]

        test_config = {"Matarial_IDs": None}

        model_loaded = test_model.load_model(Path(test_path), test_config)

        assert np.array_equal(
            model_loaded.nodes.coords,
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )

        assert np.array_equal(
            model_loaded.elements[0].node_ids, np.array([3, 1])
        )

        assert np.array_equal(
            model_loaded.elements[1].node_ids, np.array([2, 4])
        )


def test_get_center():
    """Test get_center if center of elements are calculated correctly."""
    ele1_ids = np.array([0, 2, 4], dtype=int)
    ele2_ids = np.array([1, 3, 5], dtype=int)

    elements = []
    elements.append(Element(ele1_ids, 1, [], []))
    elements.append(Element(ele2_ids, 2, [], []))
    nodes_coords = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 3, 6], [6, 2, 0], [9, 3, 6]]
    )
    node_ids = np.array([0, 1, 2, 3, 4, 5], dtype=int)

    nodes = Nodes(nodes_coords, node_ids)

    test_model = ModelData(nodes, elements, [])

    data_reder = MeshReader()
    model = data_reder.get_center(test_model)

    expected_output1 = np.array([2, 1, 0])
    expected_output2 = np.array([4, 2, 4])

    assert np.array_equal(model.elements[0].center_coord, expected_output1)
    assert np.array_equal(model.elements[1].center_coord, expected_output2)

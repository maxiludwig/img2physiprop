"""Test Mesh Reader Routine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from i2pp.core.discretization_reader_classes.dat_reader import DatReader
from i2pp.core.discretization_reader_classes.mesh_reader import MeshReader
from i2pp.core.import_discretization import (
    DiscretizationFormat,
    determine_discretization_format,
)


def test_determine_discretization_format_path_not_exist():
    """Test determine_discretization_format if path not exist."""
    directory = "not_existing_path"

    with pytest.raises(
        RuntimeError,
        match="Path not_existing_path to the Discretization cannot be found!",
    ):
        determine_discretization_format(Path(directory))


def test_determine_discretization_format_wrong_format(tmp_path: Path) -> None:
    """Test determine_discretization_format if Mesh has wrong format."""

    test_path = tmp_path / "test_mesh.stl"

    with patch("pathlib.Path.is_file", returnValue=True):
        with pytest.raises(RuntimeError, match=".stl not readable!"):
            determine_discretization_format(Path(test_path))


def test_determine_discretization_format_dat_format(tmp_path: Path) -> None:
    """Test determine_discretization_format if Mesh has wrong format."""

    test_path = tmp_path / "test_mesh.dat"

    with patch("pathlib.Path.is_file", returnValue=True):
        assert (
            determine_discretization_format(Path(test_path))
            == DiscretizationFormat.DAT
        )


def test_determine_discretization_format_mesh_format(tmp_path: Path) -> None:
    """Test determine_discretization_format if Mesh has wrong format."""

    test_path = tmp_path / "test_mesh.mesh"

    with patch("pathlib.Path.is_file", returnValue=True):
        assert (
            determine_discretization_format(Path(test_path))
            == DiscretizationFormat.MESH
        )


def test_load_discretization_mesh(tmp_path: Path) -> None:
    """Test load_discretization if input is .mesh."""

    test_path = tmp_path / "test_model.mesh"
    test_dis = MeshReader()

    test_config = {"processing informations": None}

    with patch(
        "i2pp.core.discretization_reader_classes.mesh_reader.Discretization",
        returnValue=None,
    ) as MockClass:
        with patch("trimesh.load", returnValue=None) as mock_trimesh:

            test_dis.load_discretization(Path(test_path), test_config)
            assert mock_trimesh.call_count == 1
            assert MockClass.call_count == 1


def test___filter_discretization_one_filter():
    """Test _filter_discretization with 1 Filter."""
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
    test_dis = DatReader()
    dis_filtered = test_dis._filter_discretization(mock_dis, [2])
    coords_list = [node.coords for node in dis_filtered.nodes]

    expected_filtered_nodes = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

    assert all(x in coords_list for x in expected_filtered_nodes)

    assert len(coords_list) == 3


def test__filter_discretization_multiple_filters():
    """Test _filter_discretization with multiple Filter."""
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
    test_dis = DatReader()
    dis_filtered = test_dis._filter_discretization(mock_dis, [1, 3])
    coords_list = [node.coords for node in dis_filtered.nodes]

    expected_filtered_nodes = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]

    assert all(x in coords_list for x in expected_filtered_nodes)

    assert len(coords_list) == 3


def test_load_discretization_dat_without_filter(tmp_path: Path) -> None:
    """Test load_discretization if input is .dat."""

    test_path = tmp_path / "test_mesh.dat"
    test_dis = DatReader()
    with patch("lnmmeshio.read") as mock_lnmread:

        mock_dis = MagicMock()
        mock_lnmread.return_value = mock_dis

        mock_dis.compute_ids.return_value = None

        node1 = MagicMock(id=1, coords=[0.0, 0.0, 0.0])
        node2 = MagicMock(id=2, coords=[1, 0.0, 0.0])
        node3 = MagicMock(id=3, coords=[0.0, 1, 0.0])
        node4 = MagicMock(id=4, coords=[0.0, 0.0, 1])

        ele1 = MagicMock(nodes=[node3, node1])
        ele2 = MagicMock(nodes=[node2, node4])

        mock_dis.elements.structure = [ele1, ele2]
        mock_dis.nodes = [node1, node2, node3, node4]

        test_config = {"material_ids": None}

        dis_loaded = test_dis.load_discretization(Path(test_path), test_config)

        assert np.array_equal(
            dis_loaded.nodes.coords,
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )

        assert np.array_equal(
            dis_loaded.elements[0].node_ids, np.array([3, 1])
        )

        assert np.array_equal(
            dis_loaded.elements[1].node_ids, np.array([2, 4])
        )

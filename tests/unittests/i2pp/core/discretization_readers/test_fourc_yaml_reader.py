"""Test Mesh Reader Routine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pyvista as pv
from i2pp.core.discretization_readers.fourc_yaml_reader import FourCYamlReader


def test___filter_discretization_one_filter():
    """Test _filter_discretization with 1 Filter."""
    mock_dis = MagicMock()
    node1 = MagicMock(id=1, coords=[0.0, 0.0, 0.0])
    node2 = MagicMock(id=2, coords=[1, 0.0, 0.0])
    node3 = MagicMock(id=3, coords=[0.0, 1, 0.0])
    node4 = MagicMock(id=4, coords=[0.0, 0.0, 1])

    ele1 = MagicMock(nodes=[node3, node1], options={"MAT": 1})
    ele2 = MagicMock(nodes=[node4, node2], options={"MAT": 2})
    ele3 = MagicMock(nodes=[node3, node4], options={"MAT": 3})
    ele4 = MagicMock(nodes=[node3, node4], options={"MAT": 2})

    mock_dis.elements.structure = [ele1, ele2, ele3, ele4]
    mock_dis.nodes = [node1, node2, node3, node4]
    test_dis = FourCYamlReader()
    dis_filtered = test_dis._filter_discretization(
        mock_dis, {"field": "MAT", "values": [2]}
    )
    coords_list = [node.coords for node in dis_filtered.nodes]

    expected_filtered_nodes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    assert coords_list == expected_filtered_nodes


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
    test_dis = FourCYamlReader()
    dis_filtered = test_dis._filter_discretization(
        mock_dis, {"field": "MAT", "values": [1, 3]}
    )
    coords_list = [node.coords for node in dis_filtered.nodes]

    expected_filtered_nodes = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]

    assert all(x in coords_list for x in expected_filtered_nodes)

    assert len(coords_list) == 3


def test_load_discretization_fourc_yaml_without_filter(tmp_path: Path) -> None:
    """Test load_discretization if input is 4C.yaml."""

    test_path = tmp_path / "test_mesh.4C.yaml"
    test_dis = FourCYamlReader()
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

        surface1 = MagicMock(nodes=[node1, node2])
        surface2 = MagicMock(nodes=[node3, node4])

        mock_dis.surfacenodesets = [surface1, surface2]
        mock_dis.elements.structure = [ele1, ele2]
        mock_dis.nodes = [node1, node2, node3, node4]

        # test_config
        test_options = {"material_ids": None}
        mock_processing = MagicMock()
        mock_scaling_factors = (
            mock_processing.interpolation.node_scaling_factors
        )
        mock_scaling_factors.interior_node_scaling = 1.0
        mock_scaling_factors.surface_node_scaling = 0.0

        dis_loaded = test_dis.load_discretization(
            Path(test_path), test_options, mock_processing
        )

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

        assert np.array_equal(
            dis_loaded.surfaces[0].node_ids, np.array([1, 2])
        )

        assert np.array_equal(
            dis_loaded.surfaces[1].node_ids, np.array([3, 4])
        )


def _mock_line_dis() -> MagicMock:
    """Create a mocked lnmmeshio discretization with element data."""
    mock_dis = MagicMock()
    nodes = [MagicMock(id=i, coords=[float(i), 0.0, 0.0]) for i in range(1, 5)]
    mock_dis.elements.structure = [
        MagicMock(
            nodes=[nodes[0], nodes[1]],
            options={"MAT": 1},
            data={"block_id": 1},
        ),
        MagicMock(
            nodes=[nodes[1], nodes[2]],
            options={"MAT": 1},
            data={"block_id": 2},
        ),
        MagicMock(
            nodes=[nodes[2], nodes[3]],
            options={"MAT": 2},
            data={"block_id": 2},
        ),
    ]
    mock_dis.nodes = nodes
    return mock_dis


def test__filter_discretization_element_filter():
    """Test _filter_discretization with an element data filter keeps the
    element order and the nodes of the selected elements."""
    mock_dis = _mock_line_dis()
    selected = mock_dis.elements.structure[1:]

    dis_filtered = FourCYamlReader()._filter_discretization(
        mock_dis, element_filter={"field": "block_id", "values": [2]}
    )

    assert dis_filtered.elements.structure == selected
    assert [node.id for node in dis_filtered.nodes] == [2, 3, 4]


def test__filter_discretization_material_from_options():
    """Test that fields missing in the element data (e.g. the material) are
    looked up in the element options, also if stored as string."""
    mock_dis = _mock_line_dis()
    mock_dis.elements.structure[0].options = {"MAT": "2"}
    selected = [mock_dis.elements.structure[0], mock_dis.elements.structure[2]]

    dis_filtered = FourCYamlReader()._filter_discretization(
        mock_dis, element_filter={"field": "MAT", "values": [2]}
    )

    assert dis_filtered.elements.structure == selected


def test__filter_discretization_element_filter_missing_field():
    """Test _filter_discretization raises if the filter field is unknown."""
    with pytest.raises(ValueError, match="'generation' not found"):
        FourCYamlReader()._filter_discretization(
            _mock_line_dis(),
            element_filter={"field": "generation", "values": [2]},
        )


def _mock_processing() -> MagicMock:
    """Create a mocked processing configuration."""
    mock_processing = MagicMock()
    scaling_factors = mock_processing.interpolation.node_scaling_factors
    scaling_factors.interior_node_scaling = 1.0
    scaling_factors.surface_node_scaling = 1.0
    return mock_processing


def test_load_discretization_vtu_line2_with_element_filter(
    tmp_path: Path,
) -> None:
    """Test loading a line2 VTU with an element filter on cell data."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]
    )
    grid = pv.UnstructuredGrid(
        np.array([2, 0, 1, 2, 1, 2, 2, 2, 3]),
        np.full(3, pv.CellType.LINE, dtype=np.uint8),
        points,
    )
    grid.cell_data["block_id"] = np.array([1, 2, 2], dtype=np.int32)
    vtu_path = tmp_path / "tree.vtu"
    grid.save(vtu_path)

    dis = FourCYamlReader().load_discretization(
        vtu_path,
        {"element_filter": {"field": "block_id", "values": [2]}},
        _mock_processing(),
    )

    assert [ele.source_id for ele in dis.elements] == [1, 2]
    assert [ele.id for ele in dis.elements] == [0, 1]
    assert [ele.fields["block_id"] for ele in dis.elements] == [2, 2]
    assert all(len(ele.node_ids) == 2 for ele in dis.elements)
    np.testing.assert_array_equal(dis.nodes.coords, points[1:])


def test_load_discretization_invalid_element_filter(tmp_path: Path) -> None:
    """Test load_discretization raises for an incomplete element filter."""
    with patch("lnmmeshio.read", return_value=_mock_line_dis()):
        with pytest.raises(ValueError, match="requires the keys"):
            FourCYamlReader().load_discretization(
                tmp_path / "tree.vtu",
                {"element_filter": {"field": "block_id"}},
                _mock_processing(),
            )


def test_load_discretization_filter_removes_all_elements(
    tmp_path: Path,
) -> None:
    """Test load_discretization raises if no element passes the filter."""
    with patch("lnmmeshio.read", return_value=_mock_line_dis()):
        with pytest.raises(RuntimeError, match="No elements left"):
            FourCYamlReader().load_discretization(
                tmp_path / "tree.vtu",
                {"element_filter": {"field": "block_id", "values": [7]}},
                _mock_processing(),
            )

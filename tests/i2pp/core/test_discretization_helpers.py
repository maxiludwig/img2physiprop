"""Test Mesh Reader Routine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pyvista as pv
from i2pp.core.discretization_helpers import (
    BoundingBox,
    DiscretizationFormat,
    determine_discretization_format,
    get_elementwise_image_values,
    initialize_unstructured_grid,
    verify_and_load_discretization,
)
from i2pp.core.discretization_reader_classes.dat_reader import DatReader
from i2pp.core.discretization_reader_classes.discretization_reader import (
    Discretization,
    Element,
    Nodes,
)
from i2pp.core.image_reader_classes.image_reader import PixelValueType


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
        "i2pp.core.discretization_helpers.determine_discretization_format",
        return_value=DiscretizationFormat.DAT,
    ) as mock_determine_discretization_format:
        with patch.object(
            DatReader, "load_discretization", return_value=mock_dis
        ) as mock_load_discretization:
            with patch(
                "i2pp.core.discretization_helpers.find_mins_maxs",
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


@pytest.mark.parametrize(
    "node_ids, expected_cell_type",
    [
        ([1, 2, 3, 4], pv.CellType.TETRA),
        ([1, 2, 3, 4, 5, 6, 7, 8], pv.CellType.HEXAHEDRON),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], pv.CellType.QUADRATIC_TETRA),
        ([1], "expect_value_error"),  # Empty list should raise an error
    ],
)
def test_initialize_unstructured_grid(node_ids, expected_cell_type):
    """Test initialize_unstructured_grid function with different cell types."""
    mock_discretization = Discretization(
        nodes=Nodes(coords=np.ones((len(node_ids), 3)), ids=node_ids),
        elements=[
            Element(id=1, node_ids=node_ids),
        ],
    )
    mock_elements_with_values = [
        Element(node_ids=node_ids, id=1, data=np.array([255, 0, 0])),
    ]
    mock_pixel_type = PixelValueType.RGB
    mock_unstructured_grid = MagicMock()
    mock_unstructured_grid.cell_data = {}
    with patch(
        "i2pp.core.discretization_helpers.get_elementwise_image_values",
        return_value=(
            np.array([[255, 0, 0]]),
            np.array([True]),
        ),
    ) as mock_get_elementwise_image_values:
        with patch(
            "i2pp.core.discretization_helpers.pv.UnstructuredGrid",
            return_value=mock_unstructured_grid,
        ) as mock_unstructured_grid_constructor:
            if expected_cell_type == "expect_value_error":
                with pytest.raises(
                    ValueError, match="Unsupported element with 1 nodes."
                ):
                    initialize_unstructured_grid(
                        mock_elements_with_values,
                        mock_pixel_type,
                        mock_discretization,
                    )
                return
            else:
                unstructured_grid, ele_has_value = (
                    initialize_unstructured_grid(
                        mock_elements_with_values,
                        mock_pixel_type,
                        mock_discretization,
                    )
                )
            args, _ = mock_unstructured_grid_constructor.call_args
            np.testing.assert_array_equal(
                args[0],
                np.array([len(node_ids)] + [id - 1 for id in node_ids]),
            )
            np.testing.assert_array_equal(
                args[1], np.array([expected_cell_type], dtype=np.uint8)
            )
            np.testing.assert_array_equal(args[2], np.ones((len(node_ids), 3)))
            assert np.array_equal(
                unstructured_grid.cell_data["RGB_values"],
                np.array([[255, 0, 0]]),
            )
            assert np.array_equal(ele_has_value, [True])
            mock_get_elementwise_image_values.assert_called_once_with(
                mock_elements_with_values,
                mock_discretization,
                mock_pixel_type,
            )


def test_get_elementwise_image_values_rgb():
    """Test get_elementwise_image_values for RGB pixel type."""
    elements_with_values = [
        Element(node_ids=[1, 2], id=1, data=np.array([255, 0, 0])),
        Element(
            node_ids=[1, 2], id=2, data=np.array([np.nan, np.nan, np.nan])
        ),
        Element(node_ids=[1, 2], id=3, data=np.array([0, 255, 0])),
    ]
    dis = Discretization(
        nodes=Nodes(coords=np.array([[0, 0, 0], [1, 1, 1]]), ids=[1, 2]),
        elements=[
            Element(id=1, node_ids=[1, 2]),
            Element(id=2, node_ids=[1, 2]),
            Element(id=3, node_ids=[1, 2]),
        ],
    )
    pixel_type = PixelValueType.RGB
    values, ele_has_value = get_elementwise_image_values(
        elements_with_values, dis, pixel_type
    )
    assert np.array_equal(values[0], [255, 0, 0])
    assert np.array_equal(
        values[1], [0, 0, 0]
    )  # NaN values are replaced with 0
    assert np.array_equal(values[2], [0, 255, 0])
    assert np.array_equal(ele_has_value, [True, False, True])


def test_get_elementwise_image_values_mrt():
    """Test get_elementwise_image_values for MRT pixel type."""
    elements_with_values = [
        Element(node_ids=[1, 2], id=1, data=1.0),
        Element(node_ids=[1, 2], id=2, data=np.nan),
        Element(node_ids=[1, 2], id=3, data=3.0),
    ]
    dis = Discretization(
        nodes=Nodes(coords=np.array([[0, 0, 0], [1, 1, 1]]), ids=[1, 2]),
        elements=[
            Element(id=1, node_ids=[1, 2]),
            Element(id=2, node_ids=[1, 2]),
            Element(id=3, node_ids=[1, 2]),
        ],
    )
    pixel_type = PixelValueType.MRT
    values, ele_has_value = get_elementwise_image_values(
        elements_with_values, dis, pixel_type
    )

    assert np.array_equal(values[0], 1.0)
    assert np.array_equal(values[1], 0.0)  # NaN values are replaced with 0
    assert np.array_equal(values[2], 3.0)
    assert np.array_equal(ele_has_value, [True, False, True])

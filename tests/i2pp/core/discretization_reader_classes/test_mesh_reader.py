"""Test Mesh Reader Routine."""

from pathlib import Path
from unittest.mock import patch

from i2pp.core.discretization_reader_classes.mesh_reader import MeshReader


def test_load_discretization_mesh(tmp_path: Path) -> None:
    """Test load_discretization if input is .mesh."""

    test_path = tmp_path / "test_model.mesh"
    test_dis = MeshReader()

    test_config = {"material_ids": None}

    with patch(
        "i2pp.core.discretization_reader_classes.mesh_reader.Discretization",
        returnValue=None,
    ) as MockClass:
        with patch("trimesh.load", returnValue=None) as mock_trimesh:

            test_dis.load_discretization(Path(test_path), test_config)
            assert mock_trimesh.call_count == 1
            assert MockClass.call_count == 1

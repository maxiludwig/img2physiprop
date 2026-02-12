"""Test Mesh Reader Routine."""

from pathlib import Path
from unittest.mock import patch

from i2pp.core.configuration_validator.validator import (
    Interpolation,
    NodeWeight,
    Processing,
    Transformation,
)
from i2pp.core.discretization_readers.mesh_reader import MeshReader


def test_load_discretization_mesh(tmp_path: Path) -> None:
    """Test load_discretization if input is .mesh."""

    test_path = tmp_path / "test_model.mesh"
    test_dis = MeshReader()

    test_config = {"material_ids": None}
    # create dummy processing config
    test_processing = Processing(
        smoothing=None,
        interpolation=Interpolation(
            method="nodes",
            filter_outliers=False,
            node_weight=NodeWeight(interior=0.5, surface=0.5),
        ),
        transformation=Transformation(
            user_script=Path(""),
            user_function="",
            normalize_values=False,
            visualize=False,
        ),
    )

    with patch(
        "i2pp.core.discretization_readers.mesh_reader.Discretization",
        returnValue=None,
    ) as MockClass:
        with patch("trimesh.load", returnValue=None) as mock_trimesh:

            test_dis.load_discretization(
                Path(test_path), test_config, test_processing
            )
            assert mock_trimesh.call_count == 1
            assert MockClass.call_count == 1

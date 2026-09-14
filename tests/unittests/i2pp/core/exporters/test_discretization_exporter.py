"""Test the export into the input discretization."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
    Element,
    Nodes,
)
from i2pp.core.exporters.discretization_exporter import (
    DiscretizationExporter,
)


def _write_line_mesh(path: Path) -> Path:
    """Write a VTU with three line cells and a 'block_id' cell array."""
    grid = pv.UnstructuredGrid(
        np.array([2, 0, 1, 2, 1, 2, 2, 2, 3]),
        np.full(3, pv.CellType.LINE, dtype=np.uint8),
        np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [2.0, 1.0, 0]]),
    )
    grid.cell_data["block_id"] = np.array([2, 1, 2], dtype=np.int32)
    grid.save(path)
    return path


def _filtered_line_dis(source_ids: list[int]) -> Discretization:
    """Create a filtered discretization referencing cells of the line mesh."""
    elements = [
        Element(node_ids=np.array([0, 1]), id=i, source_id=s)
        for i, s in enumerate(source_ids)
    ]
    return Discretization(
        nodes=Nodes(coords=np.zeros((2, 3)), ids=np.array([0, 1])),
        elements=elements,
        surfaces=[],
    )


def _data(**fields) -> np.ndarray:
    """Structured array as returned by a user function, with the given value
    fields."""
    n = len(next(iter(fields.values())))
    dtype = [("index", "i4")] + [
        (name, "f8", np.shape(values)[1:]) for name, values in fields.items()
    ]
    data = np.zeros(n, dtype=dtype)
    data["index"] = np.arange(1, n + 1)
    for name, values in fields.items():
        data[name] = values
    return data


def test_write_fields_and_merge_consecutive_runs(tmp_path: Path):
    """Test that the complete input mesh is written with each value field in
    the cell data field of the same name, and that a second run on another
    element selection keeps the values of the first run."""
    source = _write_line_mesh(tmp_path / "tree.vtu")

    # first run: cells 0 and 2 (e.g. terminal units)
    first = tmp_path / "first.vtu"
    DiscretizationExporter(source, _filtered_line_dis([0, 2])).write_data(
        _data(E=[10.0, 20.0], nu=[0.3, 0.4]), first
    )

    grid = pv.read(first)
    assert grid.n_cells == 3
    np.testing.assert_array_equal(grid.cell_data["block_id"], [2, 1, 2])
    np.testing.assert_array_equal(grid.cell_data["E"], [10.0, np.nan, 20.0])
    np.testing.assert_array_equal(grid.cell_data["nu"], [0.3, np.nan, 0.4])
    assert "CT_values" not in grid.cell_data

    # second run on the output of the first run: cell 1 (e.g. airways)
    second = tmp_path / "second.vtu"
    DiscretizationExporter(first, _filtered_line_dis([1])).write_data(
        _data(E=[30.0]), second
    )

    grid = pv.read(second)
    np.testing.assert_array_equal(grid.cell_data["E"], [10.0, 30.0, 20.0])
    np.testing.assert_array_equal(grid.cell_data["nu"], [0.3, np.nan, 0.4])


def test_write_vector_field(tmp_path: Path):
    """Test that a vector value field becomes a multi-component field."""
    source = _write_line_mesh(tmp_path / "tree.vtu")
    output = tmp_path / "out.vtu"

    DiscretizationExporter(source, _filtered_line_dis([0])).write_data(
        _data(fractions=[[0.2, 0.8]]), output
    )

    np.testing.assert_array_equal(
        pv.read(output).cell_data["fractions"],
        [[0.2, 0.8], [np.nan, np.nan], [np.nan, np.nan]],
    )


@pytest.mark.parametrize(
    "data, match",
    [
        (np.array([1.0]), "structured numpy array"),
        (np.array([(1,)], dtype=[("index", "i4")]), "at least one value"),
        (
            np.array([(1, "a")], dtype=[("index", "i4"), ("name", "U1")]),
            "must be numeric, but these are not: name",
        ),
        (
            np.array([(2, 1.0)], dtype=[("index", "i4"), ("E", "f8")]),
            "must match the element ids",
        ),
    ],
)
def test_write_data_rejects_invalid_data(tmp_path: Path, data, match):
    """Test that invalid user function results are rejected."""
    source = _write_line_mesh(tmp_path / "tree.vtu")
    exporter = DiscretizationExporter(source, _filtered_line_dis([0]))

    with pytest.raises(RuntimeError, match=match):
        exporter.write_data(data, tmp_path / "out.vtu")


def test_write_data_does_not_overwrite_input(tmp_path: Path):
    """Test that the input discretization is not overwritten."""
    source = _write_line_mesh(tmp_path / "tree.vtu")
    exporter = DiscretizationExporter(source, _filtered_line_dis([0]))

    with pytest.raises(RuntimeError, match="would overwrite the input"):
        exporter.write_data(_data(E=[1.0]), source)


def test_write_data_without_source_ids(tmp_path: Path):
    """Test that unknown element positions in the input mesh raise."""
    source = _write_line_mesh(tmp_path / "tree.vtu")
    dis = _filtered_line_dis([0])
    dis.elements[0].source_id = None

    with pytest.raises(RuntimeError, match="position of the elements"):
        DiscretizationExporter(source, dis).write_data(
            _data(E=[1.0]), tmp_path / "out.vtu"
        )


def test_write_data_with_mixed_cell_dimensions(tmp_path: Path):
    """Test that input meshes with cells of different dimensions, whose cells
    do not match the element positions, are rejected."""
    source = tmp_path / "mixed.vtu"
    pv.UnstructuredGrid(
        np.array([1, 0, 2, 0, 1, 2, 1, 2]),
        np.array(
            [pv.CellType.VERTEX, pv.CellType.LINE, pv.CellType.LINE],
            dtype=np.uint8,
        ),
        np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]),
    ).save(source)

    with pytest.raises(RuntimeError, match="cells of different dimensions"):
        DiscretizationExporter(source, _filtered_line_dis([0])).write_data(
            _data(E=[1.0]), tmp_path / "out.vtu"
        )

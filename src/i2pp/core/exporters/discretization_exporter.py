"""Discretization Exporter for writing results into the input
discretization."""

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from i2pp.core.discretization_readers.discretization_reader import (
    Discretization,
)
from i2pp.core.exporters.exporter import Exporter
from vtkmodules.vtkCommonDataModel import vtkCellTypes


class DiscretizationExporter(Exporter):
    """Exporter writing the results as cell data into a copy of the input
    discretization.

    Currently only .vtu discretizations are supported. The complete
    input mesh is written, including all of its original point and cell
    data. Each value field returned by the user function is written to
    the cell data field of the same name. Elements that were not
    selected by the element filter receive NaN, unless the input mesh
    already contains values for them (e.g. from a previous run on
    another element selection).
    """

    export_format = "vtu"

    def __init__(
        self, discretization_path: Path, discretization: Discretization
    ):
        """Init DiscretizationExporter.

        Arguments:
            discretization_path (Path): Path to the input discretization.
            discretization (Discretization): The (possibly filtered)
                discretization the results belong to.
        """
        self.discretization_path = Path(discretization_path)
        self.discretization = discretization

    def write_data(
        self, data: Any, output_file: Path, name_of_output_property: str = ""
    ) -> dict:
        """Writes the results into a copy of the input discretization.

        Arguments:
            data (Any): The data to be written. It needs to be a structured
                numpy array with the integer field 'index' (element id + 1)
                and at least one numeric value field. Each value field is
                written to the cell data field of the same name.
            output_file (Path): Path to the output file.
            name_of_output_property (str): Not used for the discretization
                export.

        Returns:
            dict: An empty dictionary.

        Raises:
            RuntimeError: If the data or the output file are invalid, or if
                the input mesh does not match the elements.
        """
        output_file = self._validate_outfile(output_file)
        if output_file.resolve() == self.discretization_path.resolve():
            raise RuntimeError(
                f"The export file {output_file} would overwrite the input "
                "discretization. Choose a different export file name."
            )

        fields = self._get_fields(data)
        source_ids = self._get_source_ids()
        grid = self._read_input_mesh()

        for name, values in fields.items():
            self._set_cell_data(grid, name, values, source_ids)
        grid.save(output_file)

        return {}

    def _get_fields(self, data: Any) -> dict[str, np.ndarray]:
        """Checks the data returned by the user function and extracts the
        values of its value fields."""
        if (
            not isinstance(data, np.ndarray)
            or data.dtype.names is None
            or data.dtype.names[0] != "index"
            or not np.issubdtype(data.dtype[0], np.integer)
        ):
            raise RuntimeError(
                "You specified the discretization export format. In this "
                "case, the user function must return a structured numpy "
                "array whose first field is the integer field 'index'."
            )

        value_fields = data.dtype.names[1:]
        if not value_fields:
            raise RuntimeError(
                "For the discretization export format, the user function "
                "must return at least one value field besides 'index'."
            )
        non_numeric = [
            name
            for name in value_fields
            if not np.issubdtype(data.dtype[name].base, np.number)
        ]
        if non_numeric:
            raise RuntimeError(
                "For the discretization export format, all value fields must "
                f"be numeric, but these are not: {', '.join(non_numeric)}."
            )

        element_ids = np.array(
            [ele.id for ele in self.discretization.elements]
        )
        if not np.array_equal(element_ids, data["index"] - 1):
            raise RuntimeError(
                "The 'index' field of the exported data must match the "
                "element ids."
            )

        return {
            name: np.asarray(data[name], dtype=np.float64)
            for name in value_fields
        }

    def _get_source_ids(self) -> np.ndarray:
        """Returns the cell index of each element in the input mesh."""
        if any(ele.source_id is None for ele in self.discretization.elements):
            raise RuntimeError(
                "Cannot export into the input discretization, since the "
                "position of the elements in the input file is unknown."
            )
        return np.array(
            [ele.source_id for ele in self.discretization.elements]
        )

    def _read_input_mesh(self) -> pv.UnstructuredGrid:
        """Reads the input mesh and checks that its cells match the
        elements."""
        grid = pv.read(self.discretization_path)

        # lnmmeshio only reads the cells of the highest dimension as elements,
        # so the element positions only match the cells of single dimension
        # meshes
        cell_dimensions = {
            vtkCellTypes.GetDimension(int(cell_type))
            for cell_type in np.unique(grid.celltypes)
        }
        if len(cell_dimensions) > 1:
            raise RuntimeError(
                "Cannot export into the input discretization "
                f"{self.discretization_path}, since it contains cells of "
                "different dimensions."
            )
        return grid

    @staticmethod
    def _set_cell_data(
        grid: pv.UnstructuredGrid,
        name: str,
        values: np.ndarray,
        source_ids: np.ndarray,
    ) -> None:
        """Writes the values of the elements into a cell data field of the
        input mesh.

        If the input mesh already contains a cell data field with the same
        name and a matching shape, only the cells of the elements are
        overwritten. Otherwise, a new field is created in which all other
        cells are NaN.

        Arguments:
            grid (pv.UnstructuredGrid): The input mesh.
            name (str): Name of the cell data field.
            values (np.ndarray): Values of the elements.
            source_ids (np.ndarray): Cell indices of the elements in the
                input mesh.
        """
        shape = (grid.n_cells,) + values.shape[1:]

        if name in grid.cell_data and grid.cell_data[name].shape == shape:
            full = np.array(grid.cell_data[name], dtype=np.float64)
        else:
            full = np.full(shape, np.nan)

        full[source_ids] = values
        grid.cell_data[name] = full

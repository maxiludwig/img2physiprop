"""Import Discretization data."""

from enum import Enum
from pathlib import Path
from typing import Type, cast

import numpy as np
import pyvista as pv
from i2pp.core.discretization_reader_classes.dat_reader import DatReader
from i2pp.core.discretization_reader_classes.discretization_reader import (
    BoundingBox,
    Discretization,
    DiscretizationReader,
    Element,
)
from i2pp.core.discretization_reader_classes.mesh_reader import MeshReader
from i2pp.core.image_reader_classes.image_reader import PixelValueType
from i2pp.core.utilities import find_mins_maxs


class DiscretizationFormat(Enum):
    """DiscretizationFormat (Enum): Defines the supported file formats for
    Discretization data.

    Attributes:
        MESH: Represents the Discretization data in the '.mesh' format,
            commonly used for 3D Models.
        DAT: Represents the Discretization data in the '.dat' format,
            typically used for storing node and element data.

    This enum helps identify the format of the Discretization data being
    processed and guides the appropriate handling of the data based on its
    format (e.g., .mesh vs. .dat).
    """

    MESH = ".mesh"
    DAT = ".dat"
    YAML = ".yaml"

    def get_reader(self) -> Type[DiscretizationReader]:
        """Returns the appropriate Discretization reader class based on the
        Discretization format.

        Returns:
            Type[DiscretizationReader]: A class that is a subclass of
                `DiscretizationReader`, either `MeshReader` or `DatReader`.
        """
        return {
            DiscretizationFormat.MESH: MeshReader,
            DiscretizationFormat.DAT: DatReader,
            DiscretizationFormat.YAML: DatReader,
        }[self]


def determine_discretization_format(file_path: Path) -> DiscretizationFormat:
    """Verifies if the Discretization file exists, is readable, and has a
    supported format.

    This function checks whether the provided Discretization file exists and
    if its format is valid (either `.mesh` or `.dat`). If the file is missing
    or has an unsupported format, it raises an error.

    Arguments:
        file_path (Path): Path to the Discretization data file.

    Returns:
        DiscretizationFormat: The format of the Discretization file (`.mesh`
            or `.dat`).

    Raises:
        RuntimeError: If the specified file does not exist.
        RuntimeError: If the file format is not `.mesh` or `.dat`.
    """
    if not file_path.is_file():
        raise RuntimeError(
            f"Path {file_path} to the Discretization cannot be found!"
        )

    try:
        return DiscretizationFormat(file_path.suffix)

    except ValueError:
        raise RuntimeError(
            f"{file_path.suffix} not readable! Supported formats are: "
            f"{', '.join(fmt.value for fmt in DiscretizationFormat)}"
        )


def verify_and_load_discretization(config: dict) -> Discretization:
    """Loads and processes mesh data based on the user configuration.

    This function verifies the input file, selects the appropriate reader
    (MeshReader or DatReader), and loads the Discretization data. Finally, it
    determines the Discretization's bounding box.

    Arguments:
        config (dict): User configuration containing paths and processing
            options.

    Returns:
        DiscretizationData: The loaded and processed mesh data.

    Raises:
        RuntimeError: If the mesh file is not valid or in the wrong format.
    """
    relative_path = Path(
        config["input informations"]["discretization_file_path"]
    )

    file_path = Path.cwd() / relative_path

    dis_format = determine_discretization_format(file_path)

    dis_reader = cast(DiscretizationReader, dis_format.get_reader()())

    dis = dis_reader.load_discretization(
        file_path, config["processing options"]
    )

    bounding_box = find_mins_maxs(points=dis.nodes.coords, enlargement=2)

    dis.bounding_box = BoundingBox(min=bounding_box[0], max=bounding_box[1])

    return dis


def initialize_unstructured_grid(
    elements_with_values: list[Element],
    pixel_type: PixelValueType,
    dis: Discretization,
) -> tuple[pv.UnstructuredGrid, np.ndarray]:
    """Initializes a PyVista UnstructuredGrid from discretization data. This
    function creates a PyVista `UnstructuredGrid` from the provided
    discretization data, including nodes and elements. It adds the interpolated
    image values to the grid's cell data.

    Arguments:
        elements_with_values (list[Element]): List of elements with assigned
            values.
        pixel_type (PixelValueType): The type of pixel values (e.g., RGB,
            MRT, CT).
        dis (Discretization): The discretization object containing nodes and
            elements.
    Returns:
        tuple[pv.UnstructuredGrid, np.ndarray]: A tuple containing the
            PyVista `UnstructuredGrid` and a boolean array indicating which
            elements have values assigned.
    """
    cells = []
    cell_types = []

    node_id_to_index = {node_id: i for i, node_id in enumerate(dis.nodes.ids)}

    for ele in dis.elements:
        node_indices = [node_id_to_index[nid] for nid in ele.node_ids]
        if len(node_indices) == 4:
            cell_types.append(pv.CellType.TETRA)
        elif len(node_indices) == 8:
            cell_types.append(pv.CellType.HEXAHEDRON)
        elif len(node_indices) == 10:
            cell_types.append(pv.CellType.QUADRATIC_TETRA)
        else:
            raise ValueError(
                f"Unsupported element with {len(node_indices)} nodes."
            )

        cells.append([len(node_indices)] + node_indices)

    cells = np.array(
        [item for sublist in cells for item in sublist], dtype=np.int64
    )

    cell_types = np.array(cell_types, dtype=np.uint8)
    unstructured_grid = pv.UnstructuredGrid(
        cells, cell_types, dis.nodes.coords
    )

    values, ele_has_value = get_elementwise_image_values(
        elements_with_values, dis, pixel_type
    )

    unstructured_grid.cell_data[f"{pixel_type.value}_values"] = values
    return unstructured_grid, ele_has_value


def get_elementwise_image_values(
    elements_with_values: list[Element],
    dis: Discretization,
    pixel_type: PixelValueType,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts element-wise image values list of elements and returns the
    values and a boolean array indicating which elements have values. The two
    arrays are in the same order as the elements in the discretization.

    Arguments:
        elements_with_values (list[Element]): List of elements with assigned
            values.
        dis (Discretization): The discretization object containing nodes and
            elements.
        pixel_type (PixelValueType): The type of pixel values (e.g., RGB,
            MRT, CT).
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - `values`: A NumPy array of image values corresponding to the
              elements.
            - `ele_has_value`: A boolean array indicating which elements have
              values assigned.
    """
    id_to_value = {ele.id: ele.data for ele in elements_with_values}
    num_cells = len(dis.elements)
    ele_has_value = np.zeros(num_cells, dtype=bool)

    if pixel_type == PixelValueType.RGB:
        values = np.zeros((num_cells, 3), dtype=np.uint8)
    elif pixel_type == PixelValueType.MRT or pixel_type == PixelValueType.CT:
        values = np.zeros(num_cells, dtype=np.float32)

    for i, ele in enumerate(dis.elements):
        if ele.id in id_to_value:

            value = id_to_value[ele.id]
            if not np.all(np.isnan(value)):

                ele_has_value[i] = True

                if pixel_type.num_values > 1:
                    values[i, :] = value
                else:
                    values[i] = value
    return values, ele_has_value
